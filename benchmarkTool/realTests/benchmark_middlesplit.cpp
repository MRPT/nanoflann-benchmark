/**
 * Benchmark for §6.2 (middleSplit_ allocation removal) and §6.7
 * (computeInitialDistances else-if).
 *
 * Tests build time and query time over a matrix of (leaf_max_size, k)
 * configurations, on both KITTI and random point clouds. Produces CSV rows:
 *   source,leaf_max_size,k,threads,n_points,build_s,query_us
 */
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <nanoflann.hpp>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "kitti.h"
#include <mrpt/core/Clock.h>
#include <mrpt/core/get_env.h>
#include <mrpt/obs/CObservationPointCloud.h>
#include <mrpt/random/random_shuffle.h>

using namespace nanoflann;

template <typename T>
struct PointCloud3D
{
    struct Point
    {
        T x, y, z;
    };
    std::vector<Point> pts;

    inline size_t kdtree_get_point_count() const { return pts.size(); }
    inline T      kdtree_get_pt(const size_t idx, const size_t dim) const
    {
        if (dim == 0) return pts[idx].x;
        if (dim == 1) return pts[idx].y;
        return pts[idx].z;
    }
    template <class BBOX>
    bool kdtree_get_bbox(BBOX&) const
    {
        return false;
    }
};

using PCf = PointCloud3D<float>;
using KDTree =
    KDTreeSingleIndexAdaptor<L2_Simple_Adaptor<float, PCf>, PCf, 3>;

// Returns wall-clock seconds for building the index
static double build_index(const PCf& cloud, int leaf_max_size, int n_threads, KDTree*& out)
{
    KDTreeSingleIndexAdaptorParams p;
    p.leaf_max_size   = leaf_max_size;
    p.n_thread_build  = n_threads;

    auto t0 = mrpt::Clock::now();
    out     = new KDTree(3, cloud, p);
    auto t1 = mrpt::Clock::now();

    return std::chrono::duration<double>(t1 - t0).count();
}

// Returns mean per-query microseconds for k-NN over query_pts
static double query_index(
    const KDTree& index, const PCf& query_pts, int k, size_t nQueries)
{
    std::vector<size_t>       ret_idx(k);
    std::vector<float>        out_dist(k);
    nanoflann::SearchParameters sp;

    auto t0 = mrpt::Clock::now();
    for (size_t j = 0; j < nQueries; j++)
    {
        const float q[3] = {query_pts.pts[j].x, query_pts.pts[j].y, query_pts.pts[j].z};
        KNNResultSet<float> rs(k);
        rs.init(ret_idx.data(), out_dist.data());
        index.findNeighbors(rs, q, sp);
    }
    auto t1 = mrpt::Clock::now();

    double total_s = std::chrono::duration<double>(t1 - t0).count();
    return total_s / nQueries * 1e6;  // microseconds per query
}

// Subsample cloud to nSelect points (random, fixed seed)
static PCf subsample(const PCf& src, size_t nSelect, std::mt19937& rng)
{
    std::vector<size_t> idx(src.pts.size());
    std::iota(idx.begin(), idx.end(), 0);
    mrpt::random::partial_shuffle(idx.begin(), idx.end(), rng, nSelect);
    PCf out;
    out.pts.resize(nSelect);
    for (size_t i = 0; i < nSelect; i++) out.pts[i] = src.pts[idx[i]];
    return out;
}

static PCf random_cloud(size_t N, float range, std::mt19937& rng)
{
    std::uniform_real_distribution<float> dist(0.0f, range);
    PCf                                   out;
    out.pts.resize(N);
    for (size_t i = 0; i < N; i++)
        out.pts[i] = {dist(rng), dist(rng), dist(rng)};
    return out;
}

int main(int argc, char** argv)
{
    // Usage: benchmark_middlesplit <MAX_FRAMES> [random_only]
    int  maxFrames  = (argc >= 2) ? atoi(argv[1]) : 50;
    bool randomOnly = (argc >= 3 && std::string(argv[2]) == "random_only");

    const int nThreads = mrpt::get_env<int>("NANOFLANN_BENCHMARK_THREADS", 1);

    // Configs to sweep
    const std::vector<int> leaf_sizes = {10, 25};
    const std::vector<int> k_vals     = {1, 5, 20};
    const size_t           nQueryPts  = 2000;

    // CSV header
    std::cout << "source,leaf_max_size,k,threads,n_points,build_s,query_us\n";

    std::mt19937 rng(42);

    // -----------------------------------------------------------------------
    // KITTI real data
    // -----------------------------------------------------------------------
    if (!randomOnly)
    {
        auto kitti = benchmark_load_kitti();
        const size_t nFrames =
            std::min<size_t>(maxFrames, kitti->datasetSize() - 1);

        for (size_t fi = 1; fi <= nFrames; fi++)
        {
            std::cerr << "\rKITTI frame " << fi << "/" << nFrames << "  " << std::flush;

            auto pc1 = std::dynamic_pointer_cast<mrpt::obs::CObservationPointCloud>(
                kitti->getPointCloud(fi));
            auto pc2 = std::dynamic_pointer_cast<mrpt::obs::CObservationPointCloud>(
                kitti->getPointCloud(fi - 1));
            if (!pc1 || !pc2) continue;

            // Build cloud from frame fi
            PCf cloudS;
            {
                const auto& xs = pc1->pointcloud->getPointsBufferRef_x();
                const auto& ys = pc1->pointcloud->getPointsBufferRef_y();
                const auto& zs = pc1->pointcloud->getPointsBufferRef_z();
                cloudS.pts.resize(xs.size());
                for (size_t i = 0; i < xs.size(); i++)
                    cloudS.pts[i] = {xs[i], ys[i], zs[i]};
            }

            // Build query cloud from frame fi-1
            PCf cloudT;
            {
                const auto& xs = pc2->pointcloud->getPointsBufferRef_x();
                const auto& ys = pc2->pointcloud->getPointsBufferRef_y();
                const auto& zs = pc2->pointcloud->getPointsBufferRef_z();
                const size_t nq = std::min(nQueryPts, xs.size());
                cloudT = subsample(
                    [&]() {
                        PCf tmp;
                        tmp.pts.resize(xs.size());
                        for (size_t i = 0; i < xs.size(); i++)
                            tmp.pts[i] = {xs[i], ys[i], zs[i]};
                        return tmp;
                    }(),
                    nq, rng);
            }

            const size_t nQ = cloudT.pts.size();
            const size_t N  = cloudS.pts.size();

            for (int ls : leaf_sizes)
            {
                for (int k : k_vals)
                {
                    KDTree* idx = nullptr;
                    double  bt  = build_index(cloudS, ls, nThreads, idx);
                    double  qt  = query_index(*idx, cloudT, k, nQ);
                    delete idx;

                    std::cout << "kitti," << ls << "," << k << "," << nThreads
                              << "," << N << "," << bt << "," << qt << "\n";
                }
            }
        }
        std::cerr << "\n";
    }

    // -----------------------------------------------------------------------
    // Random clouds  (various sizes)
    // -----------------------------------------------------------------------
    const std::vector<size_t> rand_sizes = {10000, 50000, 120000};
    const int                 rand_reps  = 30;

    for (size_t N : rand_sizes)
    {
        for (int rep = 0; rep < rand_reps; rep++)
        {
            std::cerr << "\rRandom N=" << N << " rep " << rep + 1 << "/" << rand_reps
                      << "   " << std::flush;

            PCf cloudS = random_cloud(N, 100.0f, rng);
            PCf cloudQ = random_cloud(nQueryPts, 100.0f, rng);

            for (int ls : leaf_sizes)
            {
                for (int k : k_vals)
                {
                    KDTree* idx = nullptr;
                    double  bt  = build_index(cloudS, ls, nThreads, idx);
                    double  qt  = query_index(*idx, cloudQ, k, nQueryPts);
                    delete idx;

                    std::cout << "random_" << N << "," << ls << "," << k << ","
                              << nThreads << "," << N << "," << bt << "," << qt << "\n";
                }
            }
        }
    }
    std::cerr << "\n";

    return 0;
}
