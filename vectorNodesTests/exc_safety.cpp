// An exception thrown by the dataset adaptor in the middle of a concurrent
// build, either on the calling thread or inside a spawned build task. Meant to
// run under ASAN: nothing may touch freed memory while unwinding, and the
// index must be left in a consistent "not built" state afterwards.
//
// Usage: exc_safety [N] [main|worker]
#include <nanoflann.hpp>

#include <atomic>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace
{
using num_t = double;

struct ThrowingCloud
{
    struct Point
    {
        num_t x, y, z;
    };
    std::vector<Point> pts;

    std::thread::id           main_id = std::this_thread::get_id();
    bool                      throwInWorker = false;
    mutable std::atomic<bool> armed{false};
    mutable std::atomic<bool> worker_seen{false};
    mutable std::atomic<long> main_calls{0};
    mutable std::atomic<long> worker_calls{0};
    mutable std::atomic<bool> thrown{false};

    size_t kdtree_get_point_count() const { return pts.size(); }

    num_t kdtree_get_pt(const size_t idx, const size_t dim) const
    {
        const bool inWorker = std::this_thread::get_id() != main_id;
        const long n        = inWorker ? worker_calls.fetch_add(1) : main_calls.fetch_add(1);
        if (inWorker) worker_seen.store(true);
        // Throw only once a worker task is demonstrably running.
        if (armed.load() && worker_seen.load() && inWorker == throwInWorker && n > 200000 &&
            !thrown.exchange(true))
        {
            throw std::runtime_error(inWorker ? "worker threw" : "main threw");
        }
        const auto& p = pts[idx];
        return dim == 0 ? p.x : (dim == 1 ? p.y : p.z);
    }
    template <class BBOX>
    bool kdtree_get_bbox(BBOX&) const
    {
        return false;
    }
};

using tree_t = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<num_t, ThrowingCloud>, ThrowingCloud, 3>;
}  // namespace

int main(int argc, char** argv)
{
    const size_t      N    = (argc > 1) ? std::stoul(argv[1]) : 1000000;
    const std::string mode = (argc > 2) ? argv[2] : "main";

    ThrowingCloud cloud;
    cloud.throwInWorker = (mode == "worker");
    std::mt19937                          rng(1234);
    std::uniform_real_distribution<num_t> d(-50, 50);
    cloud.pts.resize(N);
    for (size_t i = 0; i < N; i++) cloud.pts[i] = {d(rng), d(rng), d(rng)};

    nanoflann::KDTreeSingleIndexAdaptorParams p;
    p.leaf_max_size  = 1;
    p.n_thread_build = 4;

    // Build once successfully, so that the failed rebuild below starts from a
    // built index with allocated nodes.
    tree_t idx(3, cloud, p);
    cloud.armed.store(true);
    try
    {
        idx.buildIndex();
        std::printf("no exception was thrown (test inconclusive)\n");
        return 2;
    }
    catch (const std::exception& e)
    {
        std::printf("caught: %s   (main calls=%ld, worker calls=%ld)\n", e.what(),
                    cloud.main_calls.load(), cloud.worker_calls.load());
    }

    // The index must now report "not built", not crash nor return garbage.
    const num_t q[3] = {0, 0, 0};
    uint32_t    id;
    num_t       dist;
    try
    {
        const auto n = idx.knnSearch(q, 1, &id, &dist);
        std::printf("query after failed build returned %zu result(s)\n", size_t(n));
    }
    catch (const std::exception& e)
    {
        std::printf("query after failed build threw: %s\n", e.what());
    }

    // And a clean rebuild must work again.
    cloud.armed.store(false);
    idx.buildIndex();
    const auto n = idx.knnSearch(q, 1, &id, &dist);
    std::printf("rebuild after failure OK: %zu result(s)\n", size_t(n));
    return 0;
}
