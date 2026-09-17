// Checks the concurrency bound advertised by the per-task-pool builder: at most
// n_thread_build_ threads (spawned tasks + the calling thread) should ever run
// at once, i.e. peak tasks_in_flight == n_thread_build_ - 1, and the counter
// must never go negative.
//
// Requires the instrumented header produced by gen_variants.py ("peak" variant).
#include <nanoflann.hpp>

#include <atomic>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

namespace nanoflann
{
std::atomic<int>  g_nf_peak{0};
std::atomic<int>  g_nf_min{0};
std::atomic<long> g_nf_threads{0};
}  // namespace nanoflann

struct Cloud
{
    struct Point
    {
        double x, y, z;
    };
    std::vector<Point> pts;

    size_t kdtree_get_point_count() const { return pts.size(); }
    double kdtree_get_pt(const size_t i, const size_t d) const
    {
        return d == 0 ? pts[i].x : (d == 1 ? pts[i].y : pts[i].z);
    }
    template <class BBOX>
    bool kdtree_get_bbox(BBOX&) const
    {
        return false;
    }
};

using tree_t = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<double, Cloud>, Cloud, 3>;

int main(int argc, char** argv)
{
    const size_t N = (argc > 1) ? std::stoul(argv[1]) : 500000;

    Cloud                                  c;
    std::mt19937                           rng(1);
    std::uniform_real_distribution<double> u(-50, 50);
    c.pts.resize(N);
    for (auto& p : c.pts) p = {u(rng), u(rng), u(rng)};

    std::printf("N=%zu points\n\n", N);
    std::printf(
        "%-8s %-8s %-8s %-8s %-8s %s\n", "threads", "bound", "peak", "min", "spawns", "verdict");

    int failures = 0;
    for (unsigned int T : {2u, 3u, 4u, 8u, 16u, 64u})
    {
        nanoflann::g_nf_peak    = 0;
        nanoflann::g_nf_min     = 0;
        nanoflann::g_nf_threads = 0;

        nanoflann::KDTreeSingleIndexAdaptorParams p;
        p.leaf_max_size  = 10;
        p.n_thread_build = T;
        tree_t tree(3, c, p);
        (void)tree;

        const int  bound = static_cast<int>(T) - 1;
        const int  peak  = nanoflann::g_nf_peak.load();
        const int  lo    = nanoflann::g_nf_min.load();
        const bool ok    = (peak <= bound) && (lo >= 0);
        if (!ok) failures++;
        std::printf(
            "%-8u %-8d %-8d %-8d %-8ld %s\n", T, bound, peak, lo,
            nanoflann::g_nf_threads.load(), ok ? "ok" : "BOUND VIOLATED");
    }

    std::printf(failures ? "\n%d FAILURES\n" : "\nCONCURRENCY BOUND HOLDS\n", failures);
    return failures ? 1 : 0;
}
