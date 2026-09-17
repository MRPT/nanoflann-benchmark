// Does an exception thrown by user code during a concurrent build leave a
// still-running task allocating from an already-destroyed pool?
#include <nanoflann.hpp>

#include <atomic>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

using num_t = double;

struct ThrowingCloud
{
    struct Point
    {
        num_t x, y, z;
    };
    std::vector<Point> pts;

    std::thread::id            main_id     = std::this_thread::get_id();
    mutable std::atomic<bool>  worker_seen{false};
    mutable std::atomic<long>  main_calls{0};
    mutable std::atomic<long>  worker_calls{0};
    mutable std::atomic<bool>  thrown{false};

    size_t kdtree_get_point_count() const { return pts.size(); }

    num_t kdtree_get_pt(const size_t idx, const size_t dim) const
    {
        if (std::this_thread::get_id() != main_id)
        {
            worker_seen.store(true);
            worker_calls.fetch_add(1);
        }
        else
        {
            const long n = main_calls.fetch_add(1);
            // Throw only once a worker task is demonstrably running, and only
            // after the build proper has started (the initial bbox pass is
            // single-threaded).
            if (worker_seen.load() && n > 4000000 && !thrown.exchange(true))
            {
                throw std::runtime_error("user adaptor threw mid-build");
            }
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

int main(int argc, char** argv)
{
    const size_t N = (argc > 1) ? std::stoul(argv[1]) : 3000000;

    ThrowingCloud                         cloud;
    std::mt19937                          rng(1234);
    std::uniform_real_distribution<num_t> d(-50, 50);
    cloud.pts.resize(N);
    for (size_t i = 0; i < N; i++) cloud.pts[i] = {d(rng), d(rng), d(rng)};

    nanoflann::KDTreeSingleIndexAdaptorParams p;
    p.leaf_max_size  = 1;
    p.n_thread_build = 2;

    try
    {
        tree_t idx(3, cloud, p);
        std::printf("no exception was thrown (test inconclusive)\n");
        return 2;
    }
    catch (const std::exception& e)
    {
        std::printf(
            "caught: %s   (main calls=%ld, worker calls=%ld)\n", e.what(),
            cloud.main_calls.load(), cloud.worker_calls.load());
    }
    std::printf("survived unwinding\n");
    return 0;
}
