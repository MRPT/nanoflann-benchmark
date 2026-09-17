// Pool-memory footprint of the concurrent build on real KITTI clouds.
#include <nanoflann.hpp>

#include <cstdio>
#include <vector>

#include "kitti.h"
#include <mrpt/obs/CObservationPointCloud.h>

template <typename T>
struct PointCloud
{
    struct Point
    {
        T x, y, z;
    };
    std::vector<Point> pts;

    void loadFromMRPT(const mrpt::maps::CPointsMap& p)
    {
        const auto xs = p.getPointsBufferRef_x();
        const auto ys = p.getPointsBufferRef_y();
        const auto zs = p.getPointsBufferRef_z();
        pts.resize(xs.size());
        for (size_t i = 0; i < xs.size(); i++) pts[i] = {xs[i], ys[i], zs[i]};
    }
    size_t kdtree_get_point_count() const { return pts.size(); }
    T      kdtree_get_pt(const size_t i, const size_t d) const
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
    const int nFrames = (argc > 1) ? std::atoi(argv[1]) : 20;
    auto      kitti   = benchmark_load_kitti();
    const size_t ds   = kitti->datasetSize();

    for (unsigned int T : {1u, 2u, 4u, 8u, 16u})
    {
        double sumPool = 0, sumPoints = 0;
        int    n = 0;
        for (int f = 0; f < nFrames; f++)
        {
            const size_t idx = 1 + (ds - 2) * (double(f) / nFrames);
            const auto   obs = std::dynamic_pointer_cast<mrpt::obs::CObservationPointCloud>(
                kitti->getPointCloud(idx));
            PointCloud<num_t> cloud;
            cloud.loadFromMRPT(*obs->pointcloud);

            nanoflann::KDTreeSingleIndexAdaptorParams p;
            p.leaf_max_size  = 10;
            p.n_thread_build = T;
            tree_t tree(3, cloud, p);
            sumPool += tree.pool_.usedMemory + tree.pool_.wastedMemory;
            sumPoints += cloud.pts.size();
            n++;
        }
        std::printf(
            "T=%-3u  pool_bytes=%.0f  pts=%.0f  (%d frames)\n", T, sumPool / n, sumPoints / n,
            n);
    }
    return 0;
}
