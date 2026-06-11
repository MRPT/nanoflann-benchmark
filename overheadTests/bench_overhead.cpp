// Paired "no-overhead drop-in" benchmark for the manifold feature.
//
// The SAME source file is compiled twice: once with the manifold layer
// enabled (default under C++17) and once with -DNANOFLANN_NO_MANIFOLDS.
// Both binaries run the identical R^3 KITTI build+query workload and emit,
// per dataset size N: build time, mean k-NN query time, and an FNV-1a
// checksum over the returned (index, distance-bits) pairs. The driver
// (run.sh) asserts the checksums match between the two builds and reports
// the time ratios, which should be 1.00 up to measurement noise.
//
// Usage: bench_overhead <kitti_seq_dir> [reps]

#include <nanoflann.hpp>

#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "../multiLibTests/datasets.h"

namespace
{
using Clock = std::chrono::steady_clock;

double seconds_since(const Clock::time_point& t0)
{
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

struct FlatAdaptor
{
    const bench::PointSet& ps;
    explicit FlatAdaptor(const bench::PointSet& p) : ps(p) {}
    size_t kdtree_get_point_count() const { return ps.n; }
    float  kdtree_get_pt(size_t i, size_t d) const
    {
        return ps.data[i * 3 + d];
    }
    template <class BBOX>
    bool kdtree_get_bbox(BBOX&) const
    {
        return false;
    }
};

uint64_t fnv1a(uint64_t h, const void* buf, size_t len)
{
    const auto* p = static_cast<const unsigned char*>(buf);
    for (size_t i = 0; i < len; ++i)
    {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}
}  // namespace

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::fprintf(stderr, "Usage: %s <kitti_seq_dir> [reps]\n", argv[0]);
        return 1;
    }
    const std::string seqDir = argv[1];
    const int         reps   = (argc > 2) ? std::atoi(argv[2]) : 5;

#if defined(NANOFLANN_HAS_MANIFOLDS)
    const char* variant = "manifold_on";
#else
    const char* variant = "manifold_off";
#endif

    const size_t sizes[] = {1000,    10000,    100000,
                            1000000, 10000000};
    const size_t Q       = 10000;
    const size_t k       = 1;

    // Busy-spin ~1 s so the CPU reaches steady-state frequency before any
    // timed section; otherwise sub-ms timings at small N depend on run order.
    {
        volatile double sink = 0;
        const auto     t0   = Clock::now();
        while (seconds_since(t0) < 1.0) { sink = sink + 1.0; }
        (void)sink;
    }

    std::printf("variant,N,build_s,query_us,checksum\n");
    for (const size_t N : sizes)
    {
        // More reps at small N: each rep is sub-ms and noisier.
        const int repsN = (N <= 10000) ? reps * 20 : reps;
        const bench::PointSet data =
            bench::load_kitti_sequence(seqDir, /*n_scans=*/200, N);
        if (data.n < N)
        {
            std::fprintf(
                stderr, "Not enough KITTI points for N=%zu (got %zu)\n", N,
                data.n);
            return 1;
        }
        const bench::PointSet q = bench::make_queries(data, Q, 12345);

        const FlatAdaptor ad(data);
        using Metric = nanoflann::L2_Simple_Adaptor<float, FlatAdaptor>;
        using Tree = nanoflann::KDTreeSingleIndexAdaptor<Metric, FlatAdaptor, 3>;

        double   bestBuild = 1e300;
        double   bestQuery = 1e300;
        uint64_t checksum  = 0;
        for (int rep = 0; rep < repsN; ++rep)
        {
            const auto tb0 = Clock::now();
            Tree tree(3, ad, nanoflann::KDTreeSingleIndexAdaptorParams(10));
            const double tb = seconds_since(tb0);
            if (tb < bestBuild) { bestBuild = tb; }

            uint64_t h = 14695981039346656037ULL;
            const auto tq0 = Clock::now();
            for (size_t i = 0; i < q.n; ++i)
            {
                size_t idx  = 0;
                float  dist = 0;
                nanoflann::KNNResultSet<float> rs(k);
                rs.init(&idx, &dist);
                tree.findNeighbors(rs, q.point(i));
                h = fnv1a(h, &idx, sizeof(idx));
                h = fnv1a(h, &dist, sizeof(dist));
            }
            const double tq = seconds_since(tq0) / double(q.n);
            if (tq < bestQuery) { bestQuery = tq; }
            if (rep == 0) { checksum = h; }
            else if (h != checksum)
            {
                std::fprintf(stderr, "Non-deterministic results!\n");
                return 2;
            }
        }
        std::printf(
            "%s,%zu,%.6f,%.4f,%016" PRIx64 "\n", variant, N, bestBuild,
            bestQuery * 1e6, checksum);
        std::fflush(stdout);
    }
    return 0;
}
