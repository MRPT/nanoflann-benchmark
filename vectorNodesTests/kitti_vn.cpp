// KITTI benchmark of the static and dynamic nanoflann indices, compiled once
// per nanoflann header so that two working copies can be compared.
//
// Static index, per (element type, n_thread_build, frame):
//   - build time of a freshly constructed index,
//   - build time of an in-place buildIndex() on a long-lived index object
//     whose dataset is refilled with the frame (the "rebuild per scan" case),
//   - kNN (k=1, k=10) and radius query time, queries taken from the previous
//     frame, as in benchmarkTool/realTests,
//   - usedMemory() and the heap bytes a fresh build leaves allocated.
// Dynamic index: a map grown scan by scan (decimated), with kNN queries after
// each insertion.
//
// With KITTI_ACCUM=A (default 1), each "frame" is a local map: scans
// i..i+A-1 merged, with the ground-truth poses, into the frame of the latest
// one; queries come from scan i+A transformed the same way.
//
// Usage: kitti_vn <frames> <reps> <threads,comma,separated> [static|dynamic|all]
#include <malloc.h>
#include <nanoflann.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "kitti.h"
#include <mrpt/core/get_env.h>
#include <mrpt/obs/CObservationPointCloud.h>
#include <mrpt/poses/CPose3D.h>

namespace
{
template <typename T>
struct PointCloud
{
    struct Point
    {
        T x, y, z;
    };
    std::vector<Point> pts;

    size_t kdtree_get_point_count() const { return pts.size(); }
    T      kdtree_get_pt(const size_t idx, const size_t dim) const
    {
        return dim == 0 ? pts[idx].x : (dim == 1 ? pts[idx].y : pts[idx].z);
    }
    template <class BBOX>
    bool kdtree_get_bbox(BBOX&) const
    {
        return false;
    }
};

double now()
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// Heap arenas plus mmap-ed chunks (large blocks bypass the arenas).
size_t heapInUse()
{
    const auto mi = mallinfo2();
    return mi.uordblks + mi.hblkhd;
}

struct Frame
{
    PointCloud<double> cloud;  // the scan to index
    PointCloud<double> query;  // query points (subsample of the previous scan)
};

template <typename num_t>
PointCloud<num_t> convert(const PointCloud<double>& in)
{
    PointCloud<num_t> out;
    out.pts.resize(in.pts.size());
    for (size_t i = 0; i < in.pts.size(); i++)
        out.pts[i] = {num_t(in.pts[i].x), num_t(in.pts[i].y), num_t(in.pts[i].z)};
    return out;
}

template <typename num_t>
void runStatic(const char* tname, const std::vector<Frame>& frames, int reps,
               const std::vector<unsigned>& threads)
{
    using cloud_t = PointCloud<num_t>;
    using tree_t  = nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<num_t, cloud_t>, cloud_t, 3>;

    std::vector<cloud_t> clouds, queries;
    for (const auto& f : frames)
    {
        clouds.push_back(convert<num_t>(f.cloud));
        queries.push_back(convert<num_t>(f.query));
    }

    for (unsigned T : threads)
    {
        nanoflann::KDTreeSingleIndexAdaptorParams params(10);
        params.n_thread_build = T;

        // Long-lived index for the in-place rebuilds.
        cloud_t live;
        tree_t  reuse(3, live, params);

        for (int rep = 0; rep < reps; rep++)
        {
            for (size_t fi = 0; fi < frames.size(); fi++)
            {
                const cloud_t& cloud = clouds[fi];
                const cloud_t& qc    = queries[fi];

                const size_t heap0 = heapInUse();
                const double t0    = now();
                tree_t       index(3, cloud, params);
                const double t1    = now();
                const size_t heapBuilt = heapInUse() - heap0;

                // kNN queries
                double knnSec[2] = {0, 0};
                size_t sink      = 0;
                for (int ki = 0; ki < 2; ki++)
                {
                    const size_t k = ki == 0 ? 1 : 10;
                    uint32_t     ids[10];
                    num_t        ds[10];
                    const double q0 = now();
                    for (const auto& p : qc.pts)
                    {
                        nanoflann::KNNResultSet<num_t, uint32_t> rs(k);
                        rs.init(ids, ds);
                        const num_t q[3] = {p.x, p.y, p.z};
                        index.findNeighbors(rs, q);
                        sink += ids[0];
                    }
                    knnSec[ki] = now() - q0;
                }
                if (rep == 0 && fi == 0 && T == threads.front())
                {
                    // Registration sanity check: median NN distance of queries.
                    std::vector<num_t> d1;
                    for (const auto& p : qc.pts)
                    {
                        uint32_t    id;
                        num_t       dd;
                        const num_t q[3] = {p.x, p.y, p.z};
                        nanoflann::KNNResultSet<num_t, uint32_t> rs(1);
                        rs.init(&id, &dd);
                        index.findNeighbors(rs, q);
                        d1.push_back(std::sqrt(dd));
                    }
                    std::nth_element(d1.begin(), d1.begin() + d1.size() / 2, d1.end());
                    std::fprintf(stderr, "[%s] median NN distance of queries: %.3f m\n", tname,
                                 double(d1[d1.size() / 2]));
                }
                // Radius search (the metric works on squared distances).
                const num_t radius2 =
                    num_t(std::pow(mrpt::get_env<double>("KITTI_RADIUS", 1.0), 2));
                std::vector<nanoflann::ResultItem<uint32_t, num_t>> m;
                nanoflann::SearchParameters                         sp;
                sp.sorted       = false;
                const double r0 = now();
                size_t       nRad = 0;
                for (const auto& p : qc.pts)
                {
                    const num_t q[3] = {p.x, p.y, p.z};
                    nRad += index.radiusSearch(q, radius2, m, sp);
                }
                const double radSec = now() - r0;

                // In-place rebuild of a long-lived index (dataset refilled
                // outside of the timed region).
                live.pts          = cloud.pts;
                const double b0   = now();
                reuse.buildIndex();
                const double b1   = now();

                const double nq = double(qc.pts.size());
                std::printf(
                    "S,%s,%u,%d,%zu,%zu,%.6f,%.6f,%.4f,%.4f,%.4f,%zu,%zu,%zu,%zu\n", tname, T, rep,
                    fi, cloud.pts.size(), (t1 - t0) * 1e3, (b1 - b0) * 1e3,
                    knnSec[0] / nq * 1e6, knnSec[1] / nq * 1e6, radSec / nq * 1e6,
                    size_t(index.usedMemory(index)), heapBuilt, nRad, sink % 7);
                std::fflush(stdout);
            }
        }
    }
}

template <typename num_t>
void runDynamic(const char* tname, const std::vector<Frame>& frames, int reps,
                const std::vector<unsigned>& threads)
{
    using cloud_t = PointCloud<num_t>;
    using dyn_t   = nanoflann::KDTreeSingleIndexDynamicAdaptor<
        nanoflann::L2_Simple_Adaptor<num_t, cloud_t>, cloud_t, 3>;

    // The map: every 4th point of every scan, appended scan by scan.
    constexpr size_t DECIM = 4;
    for (unsigned T : threads)
    {
        for (int rep = 0; rep < reps; rep++)
        {
            cloud_t map;
            size_t  total = 0;
            for (const auto& f : frames) total += f.cloud.pts.size() / DECIM + 1;
            map.pts.reserve(total);  // the adaptor keeps a reference to it

            nanoflann::KDTreeSingleIndexAdaptorParams params(10);
            params.n_thread_build = T;
            const size_t heap0 = heapInUse();
            dyn_t        index(3, map, params, total);
            double addSec = 0, qSec = 0;
            size_t nq = 0, sink = 0;
            for (const auto& f : frames)
            {
                const size_t first = map.pts.size();
                for (size_t i = 0; i < f.cloud.pts.size(); i += DECIM)
                {
                    const auto& p = f.cloud.pts[i];
                    map.pts.push_back({num_t(p.x), num_t(p.y), num_t(p.z)});
                }
                const double a0 = now();
                index.addPoints(first, map.pts.size() - 1);
                addSec += now() - a0;

                uint32_t     id;
                num_t        d;
                const double q0 = now();
                for (const auto& p : f.query.pts)
                {
                    nanoflann::KNNResultSet<num_t, uint32_t> rs(1);
                    rs.init(&id, &d);
                    const num_t q[3] = {num_t(p.x), num_t(p.y), num_t(p.z)};
                    index.findNeighbors(rs, q, nanoflann::SearchParameters());
                    sink += id;
                }
                qSec += now() - q0;
                nq += f.query.pts.size();
            }
            std::printf("D,%s,%u,%d,%zu,%.4f,%.4f,%zu,%zu\n", tname, T, rep, map.pts.size(),
                        addSec * 1e3, qSec / double(nq) * 1e6, heapInUse() - heap0, sink % 7);
            std::fflush(stdout);
        }
    }
}
}  // namespace

int main(int argc, char** argv)
{
    if (argc < 4)
    {
        std::fprintf(stderr, "Usage: %s <frames> <reps> <threads,csv> [static|dynamic|all]\n",
                     argv[0]);
        return 1;
    }
    const size_t numFrames = std::stoul(argv[1]);
    const int    reps      = std::stoi(argv[2]);
    std::vector<unsigned> threads;
    {
        std::stringstream ss(argv[3]);
        std::string       tok;
        while (std::getline(ss, tok, ',')) threads.push_back(unsigned(std::stoul(tok)));
    }
    const std::string what = argc > 4 ? argv[4] : "all";

    // Pre-load all frames, so that disk I/O never enters the timing.
    const size_t accum = mrpt::get_env<size_t>("KITTI_ACCUM", 1);
    auto         kitti = benchmark_load_kitti();
    const size_t nSteps = kitti->datasetSize();

    // Ground-truth poses of the vehicle (= LiDAR) frame, one per scan.
    std::vector<mrpt::poses::CPose3D> gt;
    if (accum > 1)
    {
        for (const auto& kv : kitti->getGroundTruthTrajectory())
            gt.emplace_back(mrpt::poses::CPose3D(kv.second));
        if (gt.size() != nSteps)
        {
            std::fprintf(stderr, "Ground truth has %zu poses, expected %zu\n", gt.size(), nSteps);
            return 1;
        }
    }

    auto get = [&](size_t i)
    {
        auto o = std::dynamic_pointer_cast<mrpt::obs::CObservationPointCloud>(
            kitti->getPointCloud(i));
        PointCloud<double> c;
        const auto&        xs = o->pointcloud->getPointsBufferRef_x();
        const auto&        ys = o->pointcloud->getPointsBufferRef_y();
        const auto&        zs = o->pointcloud->getPointsBufferRef_z();
        c.pts.resize(xs.size());
        for (size_t k = 0; k < xs.size(); k++) c.pts[k] = {xs[k], ys[k], zs[k]};
        return c;
    };
    // Scan i, expressed in the frame of scan `ref`.
    auto getIn = [&](size_t i, size_t ref, PointCloud<double>& out)
    {
        const PointCloud<double> c = get(i);
        if (i == ref)
        {
            out.pts.insert(out.pts.end(), c.pts.begin(), c.pts.end());
            return;
        }
        const mrpt::poses::CPose3D rel = gt[i] - gt[ref];  // pose of i wrt ref
        for (const auto& p : c.pts)
        {
            typename PointCloud<double>::Point q;
            rel.composePoint(p.x, p.y, p.z, q.x, q.y, q.z);
            out.pts.push_back(q);
        }
    };

    std::vector<Frame> frames(numFrames);
    std::mt19937       rng(1);
    for (size_t f = 0; f < numFrames; f++)
    {
        PointCloud<double> qsrc;
        if (accum <= 1)
        {
            const size_t i1 = 1 + (nSteps - 2) * f / numFrames;
            frames[f].cloud = get(i1);
            qsrc            = get(i1 - 1);
        }
        else
        {
            const size_t first = (nSteps - accum - 1) * f / numFrames;
            const size_t ref   = first + accum - 1;
            for (size_t i = first; i <= ref; i++) getIn(i, ref, frames[f].cloud);
            getIn(ref + 1, ref, qsrc);
        }
        // max(1000, N/100) random query points
        const size_t nQ = std::max<size_t>(1000, frames[f].cloud.pts.size() / 100);
        std::uniform_int_distribution<size_t> pick(0, qsrc.pts.size() - 1);
        frames[f].query.pts.resize(nQ);
        for (auto& q : frames[f].query.pts) q = qsrc.pts[pick(rng)];
    }
    std::fprintf(stderr, "Loaded %zu frames (%zu scans each, %zu points in the first)\n",
                 frames.size(), accum, frames.empty() ? size_t(0) : frames[0].cloud.pts.size());

    if (what == "static" || what == "all")
    {
        runStatic<float>("f32", frames, reps, threads);
        runStatic<double>("f64", frames, reps, threads);
    }
    if (what == "dynamic" || what == "all")
    {
        runDynamic<float>("f32", frames, reps, threads);
    }
    return 0;
}
