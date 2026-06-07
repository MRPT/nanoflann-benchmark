// Unified single-tree build/query benchmark across many C++ NN libraries.
//
// Part of nanoflann-benchmark. Measures, for a shared dataset and each library
// that was found at configure time:
//   * index build time (ms),
//   * mean k-NN query time (us, single thread),
//   * recall vs the exhaustive (brute-force) ground truth (=1.0 for exact
//     libraries; <1.0 exposes approximate methods).
//
// Two datasets:
//   * KITTI Velodyne LiDAR scans (3D), growing N (read directly from .bin,
//     no external dataset dependency).
//   * Synthetic high-dimensional clustered features (SIFT/SURF-like), where
//     exact KD-trees enter the curse-of-dimensionality regime.
//
// Anonymity (double-blind RA-L): the library under test is labeled "proposed"
// in all output; no package name is emitted.
//
// Competitors (compiled in only if found, via HAVE_* macros from CMake):
//   proposed (nanoflann), FLANN exact, FLANN randomized (approx), PCL
//   KdTreeFLANN, libnabo, pico_tree, fastann, libkdtree++, hnswlib (approx).
//   Brute force is always present.

#include "datasets.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

// ---- the proposed library (anonymized include) -----------------------------
#include <nanoflann.hpp>

#ifdef HAVE_FLANN
#include <flann/flann.hpp>
#endif
#ifdef HAVE_PCL
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#endif
#ifdef HAVE_LIBNABO
#include <nabo/nabo.h>
#endif
#ifdef HAVE_PICOTREE
#include <pico_tree/array_traits.hpp>
#include <pico_tree/kd_tree.hpp>
#include <pico_tree/vector_traits.hpp>
#endif
#ifdef HAVE_FASTANN
#include <fastann.hpp>
#endif
#ifdef HAVE_LIBKDTREE
#include <kdtree++/kdtree.hpp>
#endif
#ifdef HAVE_HNSWLIB
#include <hnswlib/hnswlib.h>
#endif

using clk = std::chrono::steady_clock;

namespace
{
constexpr size_t kMaxLeaf = 10;  // identical leaf size for every KD-tree

double ms_since(clk::time_point t0)
{
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

struct GroundTruth
{
    std::vector<size_t> idx;
    std::vector<float>  dist;  // squared L2
};

float sqdist(const float* a, const float* b, int dim)
{
    float s = 0;
    for (int d = 0; d < dim; ++d)
    {
        const float e = a[d] - b[d];
        s += e * e;
    }
    return s;
}

GroundTruth ground_truth(const bench::PointSet& data, const bench::PointSet& q)
{
    GroundTruth gt;
    gt.idx.resize(q.n);
    gt.dist.resize(q.n);
    for (size_t i = 0; i < q.n; ++i)
    {
        const float* qp   = q.point(i);
        float        best = std::numeric_limits<float>::max();
        size_t       bi   = 0;
        for (size_t j = 0; j < data.n; ++j)
        {
            const float dd = sqdist(qp, data.point(j), data.dim);
            if (dd < best)
            {
                best = dd;
                bi   = j;
            }
        }
        gt.idx[i]  = bi;
        gt.dist[i] = best;
    }
    return gt;
}

struct Result
{
    std::string name;
    double      build_ms = 0;
    double      query_us = 0;
    double      recall   = 0;
    bool        valid    = false;
};

bool is_hit(float returned_sqdist, float true_sqdist)
{
    const float tol = 1e-4f * (1.0f + true_sqdist);
    return std::fabs(returned_sqdist - true_sqdist) <= tol;
}

void emit(const char* dset, int dim, size_t N, const Result& r)
{
    if (!r.valid) { return; }
    std::printf(
        "%s,%d,%zu,%s,%.3f,%.4f,%.4f\n", dset, dim, N, r.name.c_str(),
        r.build_ms, r.query_us, r.recall);
    std::fflush(stdout);
}

// ---------------------------------------------------------------------------
// Proposed library (nanoflann), compile-time dimension Dim.
// ---------------------------------------------------------------------------
template <int Dim>
struct FlatAdaptor
{
    const bench::PointSet& ps;
    explicit FlatAdaptor(const bench::PointSet& p) : ps(p) {}
    inline size_t kdtree_get_point_count() const { return ps.n; }
    inline float  kdtree_get_pt(const size_t i, const size_t d) const
    {
        return ps.data[i * size_t(Dim) + d];
    }
    template <class BBOX>
    bool kdtree_get_bbox(BBOX&) const
    {
        return false;
    }
};

template <int Dim>
Result run_proposed(
    const bench::PointSet& data, const bench::PointSet& q, const GroundTruth& gt,
    int k)
{
    using Adaptor = FlatAdaptor<Dim>;
    using Metric  = nanoflann::L2_Simple_Adaptor<float, Adaptor>;
    using Tree    = nanoflann::KDTreeSingleIndexAdaptor<Metric, Adaptor, Dim>;

    Adaptor    ad(data);
    const auto t0 = clk::now();
    Tree       tree(Dim, ad, nanoflann::KDTreeSingleIndexAdaptorParams(kMaxLeaf));
    Result     r;
    r.name     = "proposed";
    r.build_ms = ms_since(t0);

    std::vector<size_t> oi(k);
    std::vector<float>  od(k);
    size_t              hits = 0;
    const auto          tq0  = clk::now();
    for (size_t i = 0; i < q.n; ++i)
    {
        nanoflann::KNNResultSet<float> rs(k);
        rs.init(oi.data(), od.data());
        tree.findNeighbors(rs, q.point(i));
        if (is_hit(od[0], gt.dist[i])) { ++hits; }
    }
    r.query_us = std::chrono::duration<double, std::micro>(clk::now() - tq0).count()
                 / double(q.n);
    r.recall = double(hits) / double(q.n);
    r.valid  = true;
    return r;
}

// ---------------------------------------------------------------------------
// Brute force (always available) -- timed, queries capped for large N.
// ---------------------------------------------------------------------------
volatile float g_sink = 0.0f;  // defeats dead-code elimination of brute force

Result run_brute(const bench::PointSet& data, const bench::PointSet& q, int /*k*/)
{
    Result r;
    r.name          = "brute_force";
    r.build_ms      = 0;
    const size_t Qb = (data.n > 200000) ? std::min<size_t>(q.n, 200) : q.n;
    const auto   tq0 = clk::now();
    for (size_t i = 0; i < Qb; ++i)
    {
        const float* qp   = q.point(i);
        float        best = std::numeric_limits<float>::max();
        for (size_t j = 0; j < data.n; ++j)
        {
            const float dd = sqdist(qp, data.point(j), data.dim);
            if (dd < best) { best = dd; }
        }
        g_sink = g_sink + best;
    }
    r.query_us = std::chrono::duration<double, std::micro>(clk::now() - tq0).count()
                 / double(Qb);
    r.recall = 1.0;
    r.valid  = true;
    return r;
}

#ifdef HAVE_FLANN
Result run_flann(
    const bench::PointSet& data, const bench::PointSet& q, const GroundTruth& gt,
    int k, bool approx)
{
    flann::Matrix<float> dataset(
        const_cast<float*>(data.data.data()), data.n, data.dim);
    Result     r;
    const auto t0 = clk::now();
    flann::Index<flann::L2<float>> index(
        dataset, approx ? flann::IndexParams(flann::KDTreeIndexParams(4))
                        : flann::IndexParams(flann::KDTreeSingleIndexParams(kMaxLeaf)));
    index.buildIndex();
    r.build_ms = ms_since(t0);
    r.name     = approx ? "flann_4rand" : "flann_kdtree";

    std::vector<int>     iarr(k);
    std::vector<float>   darr(k);
    flann::Matrix<int>   im(iarr.data(), 1, k);
    flann::Matrix<float> dm(darr.data(), 1, k);
    flann::SearchParams  sp(approx ? 128 : flann::FLANN_CHECKS_UNLIMITED);
    sp.cores       = 1;
    size_t     hits = 0;
    const auto tq0  = clk::now();
    for (size_t i = 0; i < q.n; ++i)
    {
        flann::Matrix<float> qm(const_cast<float*>(q.point(i)), 1, q.dim);
        index.knnSearch(qm, im, dm, k, sp);
        if (is_hit(darr[0], gt.dist[i])) { ++hits; }
    }
    r.query_us = std::chrono::duration<double, std::micro>(clk::now() - tq0).count()
                 / double(q.n);
    r.recall = double(hits) / double(q.n);
    r.valid  = true;
    return r;
}
#endif

#ifdef HAVE_PCL
Result run_pcl(
    const bench::PointSet& data, const bench::PointSet& q, const GroundTruth& gt,
    int k)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    cloud->resize(data.n);
    for (size_t i = 0; i < data.n; ++i)
    {
        (*cloud)[i].x = data.point(i)[0];
        (*cloud)[i].y = data.point(i)[1];
        (*cloud)[i].z = data.point(i)[2];
    }
    Result                          r;
    const auto                      t0 = clk::now();
    pcl::KdTreeFLANN<pcl::PointXYZ> kd;
    kd.setInputCloud(cloud);
    r.build_ms = ms_since(t0);
    r.name     = "pcl_kdtree";

    std::vector<int>   idx(k);
    std::vector<float> dist(k);
    size_t             hits = 0;
    const auto         tq0  = clk::now();
    for (size_t i = 0; i < q.n; ++i)
    {
        pcl::PointXYZ p;
        p.x = q.point(i)[0];
        p.y = q.point(i)[1];
        p.z = q.point(i)[2];
        kd.nearestKSearch(p, k, idx, dist);
        if (is_hit(dist[0], gt.dist[i])) { ++hits; }
    }
    r.query_us = std::chrono::duration<double, std::micro>(clk::now() - tq0).count()
                 / double(q.n);
    r.recall = double(hits) / double(q.n);
    r.valid  = true;
    return r;
}
#endif

#ifdef HAVE_LIBNABO
Result run_libnabo(
    const bench::PointSet& data, const bench::PointSet& q, const GroundTruth& gt,
    int k)
{
    Eigen::MatrixXf M(data.dim, data.n);
    for (size_t i = 0; i < data.n; ++i)
    {
        for (int d = 0; d < data.dim; ++d) { M(d, i) = data.point(i)[d]; }
    }
    Result           r;
    const auto       t0  = clk::now();
    Nabo::NNSearchF* nns = Nabo::NNSearchF::createKDTreeLinearHeap(M, data.dim);
    r.build_ms           = ms_since(t0);
    r.name               = "libnabo";

    Eigen::VectorXi idx(k);
    Eigen::VectorXf d2(k);
    Eigen::VectorXf qv(data.dim);
    size_t          hits = 0;
    const auto      tq0  = clk::now();
    for (size_t i = 0; i < q.n; ++i)
    {
        for (int d = 0; d < q.dim; ++d) { qv(d) = q.point(i)[d]; }
        nns->knn(qv, idx, d2, k, 0, Nabo::NNSearchF::SORT_RESULTS);
        if (is_hit(d2(0), gt.dist[i])) { ++hits; }
    }
    r.query_us = std::chrono::duration<double, std::micro>(clk::now() - tq0).count()
                 / double(q.n);
    r.recall = double(hits) / double(q.n);
    r.valid  = true;
    delete nns;
    return r;
}
#endif

#ifdef HAVE_PICOTREE
template <int Dim>
Result run_picotree(
    const bench::PointSet& data, const bench::PointSet& q, const GroundTruth& gt,
    int k)
{
    std::vector<std::array<float, Dim>> pts(data.n);
    for (size_t i = 0; i < data.n; ++i)
    {
        std::memcpy(pts[i].data(), data.point(i), sizeof(float) * Dim);
    }
    Result     r;
    const auto t0 = clk::now();
    pico_tree::KdTree<std::vector<std::array<float, Dim>>> tree(
        std::move(pts), kMaxLeaf);
    r.build_ms = ms_since(t0);
    r.name     = "pico_tree";

    std::vector<pico_tree::Neighbor<int, float>> knn;
    std::array<float, Dim>                       qa;
    size_t                                       hits = 0;
    const auto                                   tq0  = clk::now();
    for (size_t i = 0; i < q.n; ++i)
    {
        std::memcpy(qa.data(), q.point(i), sizeof(float) * Dim);
        tree.SearchKnn(qa, k, knn);
        if (is_hit(knn[0].distance, gt.dist[i])) { ++hits; }
    }
    r.query_us = std::chrono::duration<double, std::micro>(clk::now() - tq0).count()
                 / double(q.n);
    r.recall = double(hits) / double(q.n);
    r.valid  = true;
    return r;
}
#endif

#ifdef HAVE_FASTANN
Result run_fastann(
    const bench::PointSet& data, const bench::PointSet& q, const GroundTruth& gt,
    int /*k*/)
{
    Result     r;
    const auto t0 = clk::now();
    fastann::nn_obj<float>* obj =
        fastann::nn_obj_build_exact<float>(data.data.data(), data.n, data.dim);
    r.build_ms = ms_since(t0);
    r.name     = "fastann";

    unsigned   argmin = 0;
    float      mins   = 0;
    size_t     hits   = 0;
    const auto tq0    = clk::now();
    for (size_t i = 0; i < q.n; ++i)
    {
        obj->search_nn(q.point(i), 1, &argmin, &mins);
        if (is_hit(mins, gt.dist[i])) { ++hits; }
    }
    r.query_us = std::chrono::duration<double, std::micro>(clk::now() - tq0).count()
                 / double(q.n);
    r.recall = double(hits) / double(q.n);
    r.valid  = true;
    delete obj;
    return r;
}
#endif

#ifdef HAVE_LIBKDTREE
// libkdtree++ is a 3D-only competitor here (fixed triplet point type).
struct Triplet3
{
    typedef float value_type;
    float         d[3];
    value_type    operator[](size_t n) const { return d[n]; }
};
inline float triplet_ac(Triplet3 t, size_t k) { return t[k]; }

Result run_libkdtree(
    const bench::PointSet& data, const bench::PointSet& q, const GroundTruth& gt)
{
    using tree_t = KDTree::KDTree<
        3, Triplet3, std::pointer_to_binary_function<Triplet3, size_t, float>>;
    tree_t     tree(std::ptr_fun(triplet_ac));
    Result     r;
    const auto t0 = clk::now();
    for (size_t i = 0; i < data.n; ++i)
    {
        Triplet3 t{{data.point(i)[0], data.point(i)[1], data.point(i)[2]}};
        tree.insert(t);
    }
    tree.optimise();
    r.build_ms = ms_since(t0);
    r.name     = "libkdtree";

    size_t     hits = 0;
    const auto tq0  = clk::now();
    for (size_t i = 0; i < q.n; ++i)
    {
        Triplet3 qp{{q.point(i)[0], q.point(i)[1], q.point(i)[2]}};
        auto     found = tree.find_nearest(qp);
        const float d2 = found.second * found.second;  // it returns Euclidean dist
        if (is_hit(d2, gt.dist[i])) { ++hits; }
    }
    r.query_us = std::chrono::duration<double, std::micro>(clk::now() - tq0).count()
                 / double(q.n);
    r.recall = double(hits) / double(q.n);
    r.valid  = true;
    return r;
}
#endif

#ifdef HAVE_HNSWLIB
Result run_hnsw(
    const bench::PointSet& data, const bench::PointSet& q, const GroundTruth& gt,
    int k)
{
    hnswlib::L2Space                space(data.dim);
    Result                          r;
    const size_t                    M   = 16;
    const size_t                    efC = 200;
    const auto                      t0  = clk::now();
    hnswlib::HierarchicalNSW<float> alg(&space, data.n, M, efC);
    for (size_t i = 0; i < data.n; ++i) { alg.addPoint(data.point(i), i); }
    r.build_ms = ms_since(t0);
    r.name     = "hnsw_approx";
    alg.setEf(128);

    size_t     hits = 0;
    const auto tq0  = clk::now();
    for (size_t i = 0; i < q.n; ++i)
    {
        auto  res  = alg.searchKnn(q.point(i), k);
        float best = std::numeric_limits<float>::max();
        while (!res.empty())
        {
            best = std::min(best, res.top().first);
            res.pop();
        }
        if (is_hit(best, gt.dist[i])) { ++hits; }
    }
    r.query_us = std::chrono::duration<double, std::micro>(clk::now() - tq0).count()
                 / double(q.n);
    r.recall = double(hits) / double(q.n);
    r.valid  = true;
    return r;
}
#endif

// KITTI / low-dim 3D: exact KD-tree competitors.
template <int Dim>
void run_all(
    const char* dset, const bench::PointSet& data, const bench::PointSet& q,
    int k)
{
    const GroundTruth gt = ground_truth(data, q);
    emit(dset, Dim, data.n, run_proposed<Dim>(data, q, gt, k));
#ifdef HAVE_FLANN
    emit(dset, Dim, data.n, run_flann(data, q, gt, k, false));
#endif
#ifdef HAVE_LIBNABO
    emit(dset, Dim, data.n, run_libnabo(data, q, gt, k));
#endif
#ifdef HAVE_PICOTREE
    emit(dset, Dim, data.n, run_picotree<Dim>(data, q, gt, k));
#endif
#ifdef HAVE_FASTANN
    emit(dset, Dim, data.n, run_fastann(data, q, gt, k));
#endif
#ifdef HAVE_PCL
    if (Dim == 3) { emit(dset, Dim, data.n, run_pcl(data, q, gt, k)); }
#endif
#ifdef HAVE_LIBKDTREE
    if (Dim == 3) { emit(dset, Dim, data.n, run_libkdtree(data, q, gt)); }
#endif
    emit(dset, Dim, data.n, run_brute(data, q, k));
}

// High-dim: adds approximate methods (exact KD-trees lose here).
template <int Dim>
void run_all_highdim(
    const char* dset, const bench::PointSet& data, const bench::PointSet& q,
    int k)
{
    const GroundTruth gt = ground_truth(data, q);
    emit(dset, Dim, data.n, run_proposed<Dim>(data, q, gt, k));
#ifdef HAVE_FLANN
    emit(dset, Dim, data.n, run_flann(data, q, gt, k, false));
    emit(dset, Dim, data.n, run_flann(data, q, gt, k, true));
#endif
#ifdef HAVE_LIBNABO
    emit(dset, Dim, data.n, run_libnabo(data, q, gt, k));
#endif
#ifdef HAVE_PICOTREE
    emit(dset, Dim, data.n, run_picotree<Dim>(data, q, gt, k));
#endif
#ifdef HAVE_FASTANN
    emit(dset, Dim, data.n, run_fastann(data, q, gt, k));
#endif
#ifdef HAVE_HNSWLIB
    emit(dset, Dim, data.n, run_hnsw(data, q, gt, k));
#endif
    emit(dset, Dim, data.n, run_brute(data, q, k));
}

}  // namespace

int main(int argc, char** argv)
{
    const std::string kitti_seq =
        (argc > 1) ? argv[1] : "/home/jlblanco/datasets/kitti/sequences/00";
    const int    k = 1;
    const size_t Q = 2000;

    std::printf("dataset,dim,N,library,build_ms,query_us,recall\n");
    std::fflush(stdout);

    // ---- Experiment 1: KITTI LiDAR, 3D, growing N ----
    const std::vector<size_t> kitti_N = {
        50000, 100000, 250000, 500000, 1000000, 2000000};
    for (size_t N : kitti_N)
    {
        const size_t    scans = (N / 120000) + 2;  // ~120k pts/scan
        bench::PointSet data  = bench::load_kitti_sequence(kitti_seq, scans, N);
        if (data.n < N) { break; }
        bench::PointSet q = bench::make_queries(data, Q, 12345);
        run_all<3>("kitti", data, q, k);
    }

    // ---- Experiment 2: high-dimensional features ----
    for (int dim : {32, 64, 128})
    {
        const size_t          N    = 100000;
        const bench::PointSet  data = bench::make_highdim(N, dim, 7);
        const bench::PointSet  q    = bench::make_queries(data, Q, 999);
        switch (dim)
        {
            case 32: run_all_highdim<32>("highdim", data, q, k); break;
            case 64: run_all_highdim<64>("highdim", data, q, k); break;
            case 128: run_all_highdim<128>("highdim", data, q, k); break;
            default: break;
        }
    }
    return 0;
}
