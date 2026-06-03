# Delaying rebalance in a background thread (design notes)

> **Status: implemented** as `KDTreeSingleIndexIncrementalAdaptorMT` (gated by
> `NANOFLANN_NO_THREADS`). This document motivates the design; §5 summarizes what
> shipped and the measured tail-latency win.


The one place the synchronous incremental index loses to ikd-Tree is the
**update tail**: a near-root partial rebuild of an N-node tree is O(N) and runs
inline, so a frame that triggers it stalls (p95 ≈ 1.3–1.6 s on a ~4.8 M-point
KITTI map, vs a ~100 ms median). This note records (a) whether ikd-Tree solves
this and how, and (b) what it would take to do it safely in nanoflann.

## 1. Does ikd-Tree have this? Yes — it is its headline feature

ikd-Tree's "parallel rebuilding" offloads any rebuild of a subtree larger than
`Multi_Thread_Rebuild_Point_Num` (1500) to a **dedicated background thread**,
while the main thread keeps inserting, deleting and querying. The machinery
(in `ikd_Tree.cpp`) is:

- A persistent **rebuild thread** (`pthread_create` → `multi_thread_rebuild`)
  started in the constructor and joined in the destructor.
- A `Rebuild_Logger` — a fixed **10⁶-entry operation queue** (`MANUAL_Q`,
  `Q_LEN`). While the background thread rebuilds a detached copy of the
  scapegoat subtree, every insert/delete/box op that would have touched that
  subtree is **appended to the log** instead.
- Per-node synchronization: every node carries a `pthread_mutex_t
  push_down_mutex_lock` plus `working_flag` / `need_push_down_to_{left,right}`
  bits; a `search_flag` / `search_mutex` gate lets the rebuild thread wait for
  in-flight readers to leave the subtree before it swaps.
- On completion the thread **replays** the logged ops onto the freshly-balanced
  subtree and splices it back under a lock.

So the rebuild cost is *hidden*, not removed — the amortized work is the same,
but it no longer lands on the query/update critical path. This is exactly what
gives FAST-LIO its real-time guarantee, and it is why ikd-Tree's update **p95**
in our benchmark is a flat ~335 ms while its *mean* is ~260 ms.

nanoflann's plan deliberately **declined** to copy this (plan §9.7): per-node
mutexes, raw `pthread`, a fixed global queue, and `using namespace std`
contradict nanoflann's "header-only, dependency-free, const-queries-are-the-only-
concurrency" contract. The incremental index ships synchronous.

## 2. How to do it safely in nanoflann (if ever wanted)

The goal: run the big rebuild on a worker thread while **concurrent readers keep
querying the live tree lock-free and always see a consistent structure**. Three
viable designs, cleaner than ikd-Tree's per-node mutexes:

### Option A — RCU / atomic pointer swap (recommended)
This is the natural fit and avoids per-node locks entirely.

1. The writer picks a scapegoat node and takes a **read-only snapshot** of its
   live point indices (a cheap DFS into a scratch vector). The live tree is
   *not* modified; queries continue on it (just slightly unbalanced/stale).
2. A worker thread `buildBalanced()`s a brand-new subtree from the snapshot into
   **fresh nodes from a separate arena** (never touching live nodes or the
   shared free-list).
3. Inserts/deletes that arrive meanwhile are applied to the live tree **and**
   recorded in a small op-log (a `std::deque` under a short mutex, bounded by
   the rebuild duration — not a fixed 10⁶ array).
4. When the build finishes, replay the op-log onto the new subtree, then publish
   it with a single `parent->child.store(new, std::memory_order_release)`.
   Readers do `child.load(std::memory_order_acquire)` at each step, so they see
   either the entire old subtree or the entire new one — never a torn state.
5. **Retire the old subtree** only after every reader that may still be inside it
   has left. This is the one genuinely hard part (see §3).

Child pointers would become `std::atomic<INode*>` (only the few on the path to a
publish need be atomic; the rest can stay plain and be published transitively by
the release store). Queries stay **wait-free**; no per-node locks.

### Option B — per-subtree shared/exclusive lock
Readers take a `shared_lock` on the subtree being entered; the publish step takes
the `unique_lock` only for the O(1) pointer swap. The rebuild itself is lock-free
on private memory. Simpler than RCU (no deferred reclamation — the exclusive lock
guarantees no reader is inside at swap time) but readers briefly block at the
swap and you pay lock traffic on the hot query path.

### Option C — double-buffered whole-tree rebuild
Keep two trees; a worker rebuilds the standby while queries hit the active one;
swap an `atomic<root*>` when done. Dead simple and a great fit for the
*clear+rebuild* strategy (build next frame's tree in the background, query the
current one — turning rebuild's 600 ms/frame stall into a pipelined background
cost). Costs 2× peak memory and doesn't help fine-grained incremental updates.

## 3. The hard parts (why it is not in the core)

- **Safe reclamation.** A reader may still be traversing the old subtree at swap
  time. You need epoch-based reclamation, hazard pointers, or a quiescent-state
  barrier (e.g. queries register an epoch; old nodes are freed only once all
  pre-swap readers finish). Our pooled allocator + free-list is **not
  thread-safe**, so the worker needs its own arena and reclamation must be
  deferred — a real chunk of machinery.
- **Op-log replay correctness.** Deletes/box-ops that targeted points which the
  snapshot already captured must be re-applied to the new subtree (by index),
  and the `removeOutsideBox` lazy whole-subtree kills must be representable in
  the log. Bounded by rebuild time, so a `std::deque` suffices (no fixed queue).
- **Contract change.** It introduces a background *writer*, so it breaks the
  current "const queries are safe for concurrent readers; no concurrent writer"
  guarantee. It must therefore be **opt-in**, never the default path.

## 4. Recommendation

Keep the core synchronous; offer async as an **opt-in class**. The synchronous
`KDTreeSingleIndexIncrementalAdaptor` (with `alpha_balance≈0.8`) stays the default
— it has the best query latency and lowest memory. Users who need a bounded
update tail reach for `KDTreeSingleIndexIncrementalAdaptorMT`.

## 5. What shipped

`KDTreeSingleIndexIncrementalAdaptorMT` implements a pragmatic variant of the
above (gated by `NANOFLANN_NO_THREADS`):

- The active tree runs with **inline rebalancing disabled** (`setInlineRebuild(false)`)
  — append + lazy-tombstone only — so no foreground call ever pays an O(N) rebuild.
- When physical size grows past `rebuild_growth × live` (default 1.3), the
  foreground snapshots the live indices and a **`std::async` worker bulk-builds a
  fresh balanced tree** into its own arena. This sidesteps the lock-free-mutation
  and epoch-reclamation machinery of Option A: the worker only *reads* the
  dataset and writes its own private tree, so the only synchronization is the
  `std::future` handoff. The price is the documented requirement that the dataset
  keep **stable element storage** while a rebuild is in flight.
- Foreground ops during the build go to the active tree and a bounded op-log;
  at the next call after the build completes the log is replayed onto the fresh
  tree (cheap — one rebuild's worth of ops) and it replaces the active tree.

Measured (KITTI seq-00, ~4.8 M-point sliding window; see the main report §3.1):

| | upd median | upd **p95** | query µs/q | phys |
|---|---:|---:|---:|---:|
| incremental (sync, α0.85) | 92 ms | **943 ms** | 7.1 | 6.47 M |
| **incremental MT (async)** | 71 ms | **132 ms** | 10.9 | 9.31 M |
| ikd-Tree | 260 ms | 333 ms | 14.0 | 9.26 M |

The async build **cuts the update p95 by ~7×** (943 → 132 ms) and beats ikd-Tree
on update median, p95 *and* query. It costs higher query latency (the active tree
is less balanced between background rebuilds) and ~2× memory (comparable to
ikd-Tree), both tunable via the `rebuild_growth` factor (smaller ⇒ fresher tree,
lower memory, faster queries, more background work).

Verified ThreadSanitizer- and AddressSanitizer-clean, C++11-compatible, and a
brute-force-oracle churn test (`async_mt_vs_bruteforce_under_churn`).
