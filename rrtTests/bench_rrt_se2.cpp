// RRT*-style growing-tree NN benchmark on SE(2) = R^2 x SO(2).
//
// Sampling-based planners (RRT / RRT*) build a tree of explored configurations
// that *only grows*: each iteration samples a configuration, finds its nearest
// node, steers a new node toward it, and queries a near-set for rewiring. The
// nearest/near queries dominate runtime, and the correct metric on SE(2) is
// geodesic (the heading theta wraps at +/-pi). This is the canonical use case
// for an *incremental* manifold KD-tree: the tree grows by thousands of poses
// with no rebuilds, and the SE(2) metric makes the nearest query exact.
//
// We compare, on the identical RRT* node sequence (same RNG, same steering):
//   * incremental : nanoflann KDTreeSingleIndexIncrementalAdaptor + SE(2)
//                   manifold metric (the proposed capability). addPoint per node.
//   * rebuild     : nanoflann KDTreeSingleIndexAdaptor + SE(2) metric, rebuilt
//                   from scratch each iteration (the naive "dynamic" baseline).
//   * brute       : linear scan with the geodesic SE(2) distance (ground truth).
//   * naive-eucl  : incremental tree with a plain L2 metric on raw (x,y,theta)
//                   -- the common workaround; measured for *recall*, to show it
//                   returns the wrong nearest near the +/-pi seam.
//
// Metrics: cumulative nearest-query time, near-set-query time, structure-update
// time (insert or rebuild), total; and recall of the nearest vs the geodesic
// brute-force ground truth. CSV on stdout: backend,N,nearest_ms,near_ms,
// update_ms,total_ms,recall.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include <nanoflann.hpp>

using namespace nanoflann;
using clk = std::chrono::steady_clock;

namespace
{
constexpr double kPi   = 3.14159265358979323846;
constexpr double kSide = 50.0;   // workspace [0, kSide]^2 (meters)
constexpr double kStep = 1.5;    // RRT steering step (meters / radians cap)
constexpr double kNearR = 3.0;   // RRT* near-set radius (meters-equivalent)

double wrap(double a)
{
    while (a > kPi) a -= 2 * kPi;
    while (a < -kPi) a += 2 * kPi;
    return a;
}

struct SE2State
{
    double x;
    double y;
    double th;
};

// Squared SE(2) geodesic distance (translation^2 + wrapped-heading^2).
double se2_dist2(const SE2State& a, const SE2State& b)
{
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    const double dt = wrap(a.th - b.th);
    return dx * dx + dy * dy + dt * dt;
}

SE2State steer(const SE2State& from, const SE2State& to)
{
    const double dx = to.x - from.x;
    const double dy = to.y - from.y;
    const double dl = std::sqrt(dx * dx + dy * dy);
    SE2State     out = from;
    if (dl > kStep)
    {
        out.x = from.x + kStep * dx / dl;
        out.y = from.y + kStep * dy / dl;
    }
    else
    {
        out.x = to.x;
        out.y = to.y;
    }
    double dth = wrap(to.th - from.th);
    if (dth > kStep) dth = kStep;
    if (dth < -kStep) dth = -kStep;
    out.th = wrap(from.th + dth);
    return out;
}

// Growable dataset adaptor over the SE(2) node list (3 coords: x, y, theta).
struct NodeCloud
{
    std::vector<SE2State>& nodes;
    explicit NodeCloud(std::vector<SE2State>& n) : nodes(n) {}
    inline size_t kdtree_get_point_count() const { return nodes.size(); }
    inline double kdtree_get_pt(const size_t i, const size_t d) const
    {
        return d == 0 ? nodes[i].x : (d == 1 ? nodes[i].y : nodes[i].th);
    }
    template <class BBOX>
    bool kdtree_get_bbox(BBOX&) const
    {
        return false;
    }
};

double ms_since(clk::time_point t0)
{
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

// Generate a deterministic RRT* node sequence: at each step sample a random
// target, find its nearest existing node by brute force (the reference, also
// used to define the *same* steering for every backend), and steer a new node.
// Returns the node list (index 0 = root) and the per-step random samples.
struct Workload
{
    std::vector<SE2State> samples;  // q_rand per iteration
};

Workload makeWorkload(size_t iters, unsigned seed)
{
    std::mt19937                           rng(seed);
    std::uniform_real_distribution<double> ux(0, kSide);
    std::uniform_real_distribution<double> ut(-kPi, kPi);
    Workload                               w;
    w.samples.reserve(iters);
    for (size_t i = 0; i < iters; ++i)
        w.samples.push_back({ux(rng), ux(rng), ut(rng)});
    return w;
}
}  // namespace

// ---------------------------------------------------------------------------

int main(int argc, char** argv)
{
    const size_t iters = argc > 1 ? static_cast<size_t>(atoi(argv[1])) : 20000;
    const Workload w   = makeWorkload(iters, 42);

    printf("backend,N,nearest_ms,near_ms,update_ms,total_ms,recall\n");

    using SE2     = nanoflann::SE2;  // Product<Rn<2>, SO2>
    using ManMetr = Manifold_Adaptor<SE2, double, NodeCloud>;
    using EucMetr = L2_Simple_Adaptor<double, NodeCloud>;

    // The brute-force reference defines the canonical node sequence and the
    // ground-truth nearest index per iteration (so recall is well defined).
    std::vector<SE2State> ref;
    ref.reserve(iters + 1);
    ref.push_back({kSide / 2, kSide / 2, 0.0});
    std::vector<size_t> gtNearest;  // ground-truth nearest index per iteration
    gtNearest.reserve(iters);
    double bruteNearMs = 0, bruteNearSetMs = 0;
    {
        const auto t0 = clk::now();
        for (size_t it = 0; it < iters; ++it)
        {
            const SE2State& q = w.samples[it];
            size_t          best = 0;
            double          bestd = 1e300;
            const auto      tn = clk::now();
            for (size_t j = 0; j < ref.size(); ++j)
            {
                const double d = se2_dist2(q, ref[j]);
                if (d < bestd)
                {
                    bestd = d;
                    best  = j;
                }
            }
            bruteNearMs += ms_since(tn);
            gtNearest.push_back(best);
            const SE2State nn = steer(ref[best], q);
            // near-set scan (timing only)
            const auto tnr = clk::now();
            size_t     cnt = 0;
            for (size_t j = 0; j < ref.size(); ++j)
                if (se2_dist2(nn, ref[j]) < kNearR * kNearR) ++cnt;
            bruteNearSetMs += ms_since(tnr);
            (void)cnt;
            ref.push_back(nn);
        }
        const double total = ms_since(t0);
        printf(
            "brute,%zu,%.1f,%.1f,%.1f,%.1f,1.0000\n", ref.size(), bruteNearMs, bruteNearSetMs, 0.0,
            total);
    }

    // ---- helper to run a KD-tree backend over the SAME node sequence --------
    // We reuse the ref[] node positions and gtNearest[] for recall.
    auto runIncremental = [&](const char* name, bool euclidean)
    {
        std::vector<SE2State> nodes;
        nodes.reserve(iters + 1);
        NodeCloud cloud(nodes);

        // Two index types; only one is used per call.
        using TreeM = KDTreeSingleIndexIncrementalAdaptor<ManMetr, NodeCloud, 3, uint32_t>;
        using TreeE = KDTreeSingleIndexIncrementalAdaptor<EucMetr, NodeCloud, 3, uint32_t>;
        TreeM idxM(3, cloud);
        TreeE idxE(3, cloud);

        nodes.push_back(ref[0]);
        if (euclidean)
            idxE.addPoint(0);
        else
            idxM.addPoint(0);

        double  nearestMs = 0, nearMs = 0, updMs = 0;
        size_t  correct = 0;
        const auto t0 = clk::now();
        for (size_t it = 0; it < iters; ++it)
        {
            const SE2State& q     = w.samples[it];
            const double    qp[3] = {q.x, q.y, q.th};

            uint32_t nnIdx = 0;
            double   nnDist = 0;
            const auto tn = clk::now();
            if (euclidean)
            {
                KNNResultSet<double, uint32_t> rs(1);
                rs.init(&nnIdx, &nnDist);
                idxE.findNeighbors(rs, qp);
            }
            else
            {
                KNNResultSet<double, uint32_t> rs(1);
                rs.init(&nnIdx, &nnDist);
                idxM.findNeighbors(rs, qp);
            }
            nearestMs += ms_since(tn);
            if (nnIdx == gtNearest[it]) ++correct;

            const SE2State nn     = steer(nodes[nnIdx], q);
            const double   nnp[3] = {nn.x, nn.y, nn.th};

            const auto tnr = clk::now();
            std::vector<ResultItem<uint32_t, double>> near;
            if (euclidean)
                (void)idxE.radiusSearch(nnp, kNearR * kNearR, near);
            else
                (void)idxM.radiusSearch(nnp, kNearR * kNearR, near);
            nearMs += ms_since(tnr);

            const auto tu = clk::now();
            nodes.push_back(nn);
            if (euclidean)
                idxE.addPoint(static_cast<uint32_t>(nodes.size() - 1));
            else
                idxM.addPoint(static_cast<uint32_t>(nodes.size() - 1));
            updMs += ms_since(tu);
        }
        const double total = ms_since(t0);
        printf(
            "%s,%zu,%.1f,%.1f,%.1f,%.1f,%.4f\n", name, nodes.size(), nearestMs, nearMs, updMs, total,
            static_cast<double>(correct) / static_cast<double>(iters));
    };

    // ---- rebuild-each-iteration static manifold tree (naive dynamic) --------
    auto runRebuild = [&]()
    {
        std::vector<SE2State> nodes;
        nodes.reserve(iters + 1);
        NodeCloud cloud(nodes);
        using TreeS = KDTreeSingleIndexAdaptor<ManMetr, NodeCloud, 3, uint32_t>;

        nodes.push_back(ref[0]);
        double  nearestMs = 0, nearMs = 0, updMs = 0;
        size_t  correct = 0;
        const auto t0 = clk::now();
        for (size_t it = 0; it < iters; ++it)
        {
            // rebuild over the current node set
            const auto tu = clk::now();
            TreeS      idx(3, cloud, KDTreeSingleIndexAdaptorParams(10));
            updMs += ms_since(tu);

            const SE2State& q     = w.samples[it];
            const double    qp[3] = {q.x, q.y, q.th};
            uint32_t        nnIdx = 0;
            double          nnDist = 0;
            const auto      tn = clk::now();
            KNNResultSet<double, uint32_t> rs(1);
            rs.init(&nnIdx, &nnDist);
            idx.findNeighbors(rs, qp);
            nearestMs += ms_since(tn);
            if (nnIdx == gtNearest[it]) ++correct;

            const SE2State nn     = steer(nodes[nnIdx], q);
            const double   nnp[3] = {nn.x, nn.y, nn.th};
            const auto     tnr = clk::now();
            std::vector<ResultItem<uint32_t, double>> near;
            (void)idx.radiusSearch(nnp, kNearR * kNearR, near);
            nearMs += ms_since(tnr);

            nodes.push_back(nn);
        }
        const double total = ms_since(t0);
        printf(
            "rebuild,%zu,%.1f,%.1f,%.1f,%.1f,%.4f\n", nodes.size(), nearestMs, nearMs, updMs, total,
            static_cast<double>(correct) / static_cast<double>(iters));
    };

    runIncremental("incremental", false);
    runRebuild();
    runIncremental("naive-eucl", true);

    return 0;
}
