// Build-time sweep over n_thread_build on real KITTI clouds (clouds loaded once,
// then rebuilt R times per config so disk I/O never enters the timing).
#include <nanoflann.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <numeric>
#include <vector>

#include "kitti.h"
#include <mrpt/obs/CObservationPointCloud.h>

#ifdef NF_COUNT_SPAWNS
namespace nanoflann
{
std::atomic<long> g_nf_spawns{0};
}
#endif

template <typename T>
struct PointCloud
{
    struct Point
    {
        T x, y, z;
    };
    std::vector<Point> pts;
    size_t             kdtree_get_point_count() const { return pts.size(); }
    T                  kdtree_get_pt(const size_t i, const size_t d) const
    {
        return d == 0 ? pts[i].x : (d == 1 ? pts[i].y : pts[i].z);
    }
    template <class BBOX>
    bool kdtree_get_bbox(BBOX&) const
    {
        return false;
    }
};

using num_t  = double;
using tree_t = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<num_t, PointCloud<num_t>>, PointCloud<num_t>, 3>;

int main(int argc, char** argv)
{
    const int nFrames = (argc > 1) ? std::atoi(argv[1]) : 10;
    const int reps    = (argc > 2) ? std::atoi(argv[2]) : 10;
    const char* tag   = (argc > 3) ? argv[3] : "?";

    auto         kitti = benchmark_load_kitti();
    const size_t ds    = kitti->datasetSize();

    std::vector<PointCloud<num_t>> clouds(nFrames);
    for (int f = 0; f < nFrames; f++)
    {
        const size_t idx = 1 + (ds - 2) * (double(f) / nFrames);
        const auto   obs = std::dynamic_pointer_cast<mrpt::obs::CObservationPointCloud>(
            kitti->getPointCloud(idx));
        const auto& p  = *obs->pointcloud;
        const auto  xs = p.getPointsBufferRef_x();
        const auto  ys = p.getPointsBufferRef_y();
        const auto  zs = p.getPointsBufferRef_z();
        clouds[f].pts.resize(xs.size());
        for (size_t i = 0; i < xs.size(); i++) clouds[f].pts[i] = {xs[i], ys[i], zs[i]};
    }

    for (unsigned int T : {1u, 2u, 4u, 8u, 16u})
    {
        std::vector<double> ts;
        long                spawns = 0;
        for (int r = 0; r < reps; r++)
            for (const auto& c : clouds)
            {
                nanoflann::KDTreeSingleIndexAdaptorParams p;
                p.leaf_max_size  = 10;
                p.n_thread_build = T;
#ifdef NF_COUNT_SPAWNS
                nanoflann::g_nf_spawns.store(0);
#endif
                const auto t0 = std::chrono::steady_clock::now();
                tree_t     tree(3, c, p);
                const auto t1 = std::chrono::steady_clock::now();
                ts.push_back(std::chrono::duration<double>(t1 - t0).count());
#ifdef NF_COUNT_SPAWNS
                spawns += nanoflann::g_nf_spawns.load();
#endif
                (void)tree;
            }
        std::sort(ts.begin(), ts.end());
        const double mean = std::accumulate(ts.begin(), ts.end(), 0.0) / ts.size();
        std::printf(
            "%-10s T=%-3u  mean=%8.3f ms  median=%8.3f ms  best=%8.3f ms  spawns/build=%.0f\n",
            tag, T, mean * 1e3, ts[ts.size() / 2] * 1e3, ts.front() * 1e3,
            double(spawns) / double(ts.size()));
    }
    return 0;
}
