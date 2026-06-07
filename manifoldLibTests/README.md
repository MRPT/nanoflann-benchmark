# manifoldLibTests — head-to-head vs prior exact-manifold NN libraries

Compares the proposed manifold KD-tree against the cited prior exact-manifold
nearest-neighbor libraries (claim **C4** of the RA-L paper), on `SO(3)` and
`SE(3)`. The library under test is anonymized as `proposed`.

## Competitors

| Library | Citation | How | Spaces here |
|---------|----------|-----|-------------|
| `proposed` | nanoflann 2.0 (manifold branch) | local header | SO(3), SE(3) |
| `nigh` | Ichnowski & Alterovitz (`ichnowski2015se3`, `ichnowski2020concurrent`) | FetchContent (header-only, needs Eigen) | SO(3), SE(3) |
| `mpnn` | Yershova & LaValle (`yershova2007mpnn`) | FetchContent (builds a static lib from its bundled ANN) | SO(3) |
| brute force | — | built-in | SO(3) |

## Why SO(3) for the cross-validated comparison

On pure `SO(3)`, geodesic angle, chordal distance, and the quaternion
dot-product metric are all strictly monotone in the rotation angle, so all
libraries induce the **same** nearest-neighbor ordering. Recall can therefore be
cross-validated against a single geodesic ground truth (all attain `1.0`), and
only build/query speed differ. On `SE(3)` the libraries weight rotation vs
translation differently (proposed: `chordal² + ‖t‖²`; nigh: `angle + ‖t‖`), so
only timings are comparable; each is verified exact against its own metric.

## Important: the `proposed` library needs the manifold branch

The manifold feature (`Manifold_Adaptor`, `SO3`, `SE3`, `Product<>`) lives on the
`feat/manifold-topology` branch of nanoflann, not `master`. Create a worktree and
point the build at it:

```bash
git -C $HOME/code/nanoflann worktree add $HOME/code/nanoflann-manifold feat/manifold-topology
```

## Build & run

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DNANOFLANN_INCLUDE_DIR=$HOME/code/nanoflann-manifold/include
cmake --build build --target bench_manifold -j
./build/bench_manifold > results/manifold_libs.csv
python3 plot_manifold_libs.py   # writes .pgf into the paper figs/ + .png previews
```

CSV columns: `space,N,library,build_ms,query_us,recall`.
