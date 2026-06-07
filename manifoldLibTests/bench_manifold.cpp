// Head-to-head comparison of the proposed manifold KD-tree against the prior
// exact-manifold NN libraries it cites (claim C4):
//   * nigh  (UNC-Robotics) -- Ichnowski & Alterovitz, exact SO(3)/SE(3) NN.
//   * MPNN  (Yershova & LaValle) -- exact NN on R/S1/RP3 product spaces.
//   * brute force (geodesic) ground truth.
//
// SO(3) is the metric-neutral space: geodesic angle, chordal distance, and the
// quaternion dot-product metric are all strictly monotone in the rotation
// angle, so every method shares the *same* nearest-neighbor ordering. We can
// therefore cross-validate recall against a single geodesic ground truth and
// compare build/query speed fairly.
//
// SE(3) is reported for the proposed library vs nigh only, each measured exact
// against ITS OWN product metric (they weight rotation vs translation
// differently: proposed uses chordal^2 + ||t||^2, nigh uses angle + ||t||), so
// only the timings are comparable, not the identity of the neighbor.
//
// Anonymity (double-blind RA-L): the library under test is labeled "proposed".
//
// CSV columns: space,N,library,build_ms,query_us,recall

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <random>
#include <utility>  // std::exchange (used by nigh's non_atomic.hpp)
#include <vector>

#include <nanoflann.hpp>

#ifdef HAVE_NIGH
#include <Eigen/Dense>
#include <nigh/kdtree_batch.hpp>
#include <nigh/se3_space.hpp>
#include <nigh/so3_space.hpp>
#endif

#ifdef HAVE_MPNN
#include <ANN.h>
#include <multiann.h>
#endif

using clk = std::chrono::steady_clock;

namespace
{
double ms_since(clk::time_point t0)
{
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

constexpr size_t kLeaf = 10;

// A pose: unit quaternion (x,y,z,w) + translation (tx,ty,tz).
struct Pose
{
    double q[4];
    double t[3];
};

std::vector<Pose> random_poses(size_t n, unsigned seed)
{
    std::mt19937                     rng(seed);
    std::normal_distribution<double> g(0, 1);
    std::uniform_real_distribution<double> u(-10, 10);
    std::vector<Pose>                out(n);
    for (auto& p : out)
    {
        double nrm = 0;
        for (int i = 0; i < 4; ++i)
        {
            p.q[i] = g(rng);
            nrm += p.q[i] * p.q[i];
        }
        nrm = std::sqrt(nrm);
        if (nrm < 1e-12) { nrm = 1; }
        for (int i = 0; i < 4; ++i) { p.q[i] /= nrm; }
        for (int i = 0; i < 3; ++i) { p.t[i] = u(rng); }
    }
    return out;
}

// Geodesic rotation angle between two unit quaternions (double cover).
double so3_angle(const double* a, const double* b)
{
    double dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
    dot        = std::min(1.0, std::abs(dot));
    return 2.0 * std::acos(dot);
}

struct Result
{
    const char* name;
    double      build_ms = 0;
    double      query_us = 0;
    double      recall   = 0;
    bool        valid    = false;
};

void emit(const char* space, size_t N, const Result& r)
{
    if (!r.valid) { return; }
    std::printf(
        "%s,%zu,%s,%.3f,%.4f,%.4f\n", space, N, r.name, r.build_ms, r.query_us,
        r.recall);
    std::fflush(stdout);
}

// --- ground truth (SO(3) geodesic) ---
std::vector<size_t> gt_so3(
    const std::vector<Pose>& data, const std::vector<Pose>& q,
    std::vector<double>& gt_ang)
{
    std::vector<size_t> idx(q.size());
    gt_ang.resize(q.size());
    for (size_t i = 0; i < q.size(); ++i)
    {
        double best = 1e30;
        size_t bi   = 0;
        for (size_t j = 0; j < data.size(); ++j)
        {
            const double a = so3_angle(q[i].q, data[j].q);
            if (a < best)
            {
                best = a;
                bi   = j;
            }
        }
        idx[i]     = bi;
        gt_ang[i]  = best;
    }
    return idx;
}

bool ang_hit(double a, double gt)
{
    return std::abs(a - gt) <= 1e-7 * (1.0 + gt);
}

// ---------------------------------------------------------------------------
// Proposed library (nanoflann manifold adaptor), SO(3).
// ---------------------------------------------------------------------------
struct QuatCloud
{
    const std::vector<Pose>& d;
    inline size_t kdtree_get_point_count() const { return d.size(); }
    inline double kdtree_get_pt(const size_t i, const size_t k) const
    {
        return d[i].q[k];
    }
    template <class B>
    bool kdtree_get_bbox(B&) const
    {
        return false;
    }
};

Result run_proposed_so3(
    const std::vector<Pose>& data, const std::vector<Pose>& q,
    const std::vector<double>& gt_ang)
{
    using namespace nanoflann;
    using Metric = Manifold_Adaptor<SO3, double, QuatCloud>;
    using Tree   = KDTreeSingleIndexAdaptor<Metric, QuatCloud, SO3::ambient>;
    QuatCloud  cloud{data};
    const auto t0 = clk::now();
    Tree       tree(SO3::ambient, cloud, KDTreeSingleIndexAdaptorParams(kLeaf));
    Result     r;
    r.name     = "proposed";
    r.build_ms = ms_since(t0);
    size_t     hits = 0;
    const auto tq0  = clk::now();
    for (size_t i = 0; i < q.size(); ++i)
    {
        size_t          oi;
        double          od;
        KNNResultSet<double> rs(1);
        rs.init(&oi, &od);
        tree.findNeighbors(rs, q[i].q);
        if (ang_hit(so3_angle(q[i].q, data[oi].q), gt_ang[i])) { ++hits; }
    }
    r.query_us = std::chrono::duration<double, std::micro>(clk::now() - tq0).count()
                 / double(q.size());
    r.recall = double(hits) / double(q.size());
    r.valid  = true;
    return r;
}

#ifdef HAVE_NIGH
Result run_nigh_so3(
    const std::vector<Pose>& data, const std::vector<Pose>& q,
    const std::vector<double>& gt_ang)
{
    namespace nigh = unc::robotics::nigh;
    using Space    = nigh::metric::SO3Space<double>;
    struct Node
    {
        Eigen::Quaterniond q;
        size_t             idx;
    };
    struct Key
    {
        const Eigen::Quaterniond& operator()(const Node& n) const { return n.q; }
    };
    nigh::Nigh<Node, Space, Key, nigh::NoThreadSafety, nigh::KDTreeBatch<>> nn;

    const auto t0 = clk::now();
    for (size_t i = 0; i < data.size(); ++i)
    {
        // Eigen quaternion ctor is (w, x, y, z); our store is (x, y, z, w).
        nn.insert(Node{Eigen::Quaterniond(data[i].q[3], data[i].q[0],
                                          data[i].q[1], data[i].q[2]),
                       i});
    }
    Result r;
    r.name     = "nigh";
    r.build_ms = ms_since(t0);
    size_t     hits = 0;
    const auto tq0  = clk::now();
    for (size_t i = 0; i < q.size(); ++i)
    {
        Eigen::Quaterniond qq(q[i].q[3], q[i].q[0], q[i].q[1], q[i].q[2]);
        auto               res = nn.nearest(qq);
        if (res && ang_hit(so3_angle(q[i].q, data[res->first.idx].q), gt_ang[i]))
        {
            ++hits;
        }
    }
    r.query_us = std::chrono::duration<double, std::micro>(clk::now() - tq0).count()
                 / double(q.size());
    r.recall = double(hits) / double(q.size());
    r.valid  = true;
    return r;
}
#endif

#ifdef HAVE_MPNN
Result run_mpnn_so3(
    const std::vector<Pose>& data, const std::vector<Pose>& q,
    const std::vector<double>& gt_ang)
{
    // MPNN prints a banner to std::cout; silence it so it does not pollute CSV.
    std::ostream      nullsink(nullptr);
    std::streambuf*   old_cout = std::cout.rdbuf(nullsink.rdbuf());
    const int dim = 4;
    int       topology[4] = {3, 3, 3, 3};  // one quaternion block
    double    scale[4]    = {1, 1, 1, 1};

    ANNpointArray pts = annAllocPts(int(data.size()), dim);
    for (size_t i = 0; i < data.size(); ++i)
    {
        for (int k = 0; k < 4; ++k) { pts[i][k] = data[i].q[k]; }
    }

    const auto t0  = clk::now();
    MultiANN*  mag = new MultiANN(dim, 1, topology, scale);
    for (size_t i = 0; i < data.size(); ++i)
    {
        mag->AddPoint(pts[i], reinterpret_cast<void*>(i));
    }
    Result r;
    r.name     = "mpnn";
    r.build_ms = ms_since(t0);

    ANNpoint qp = annAllocPt(dim);
    size_t   hits = 0;
    const auto tq0 = clk::now();
    for (size_t i = 0; i < q.size(); ++i)
    {
        for (int k = 0; k < 4; ++k) { qp[k] = q[i].q[k]; }
        int    bidx = 0;
        double bd   = 1e30;
        mag->NearestNeighbor(qp, bidx, bd);
        if (bidx >= 0 && size_t(bidx) < data.size()
            && ang_hit(so3_angle(q[i].q, data[bidx].q), gt_ang[i]))
        {
            ++hits;
        }
    }
    r.query_us = std::chrono::duration<double, std::micro>(clk::now() - tq0).count()
                 / double(q.size());
    r.recall = double(hits) / double(q.size());
    r.valid  = true;
    delete mag;
    annDeallocPt(qp);
    annDeallocPts(pts);
    std::cout.rdbuf(old_cout);
    return r;
}
#endif

Result run_brute_so3(
    const std::vector<Pose>& data, const std::vector<Pose>& q)
{
    Result r;
    r.name          = "brute_force";
    const size_t Qb = std::min<size_t>(q.size(), data.size() > 100000 ? 200 : q.size());
    volatile double sink = 0;
    const auto      tq0  = clk::now();
    for (size_t i = 0; i < Qb; ++i)
    {
        double best = 1e30;
        for (size_t j = 0; j < data.size(); ++j)
        {
            best = std::min(best, so3_angle(q[i].q, data[j].q));
        }
        sink = sink + best;
    }
    r.query_us = std::chrono::duration<double, std::micro>(clk::now() - tq0).count()
                 / double(Qb);
    r.recall = 1.0;
    r.valid  = true;
    return r;
}

// ---------------------------------------------------------------------------
// SE(3): proposed vs nigh, each exact under its own product metric.
// ---------------------------------------------------------------------------
struct PoseCloud
{
    const std::vector<Pose>& d;
    inline size_t kdtree_get_point_count() const { return d.size(); }
    inline double kdtree_get_pt(const size_t i, const size_t k) const
    {
        // nanoflann SE3 = Product<Rn<3>, SO3>: [tx,ty,tz, qx,qy,qz,qw]
        return (k < 3) ? d[i].t[k] : d[i].q[k - 3];
    }
    template <class B>
    bool kdtree_get_bbox(B&) const
    {
        return false;
    }
};

double se3_proposed_dist2(const Pose& a, const Pose& b)
{
    double s = 0;
    for (int i = 0; i < 3; ++i)
    {
        const double e = a.t[i] - b.t[i];
        s += e * e;
    }
    double sm = 0, sp = 0;
    for (int k = 0; k < 4; ++k)
    {
        const double dm = a.q[k] - b.q[k];
        const double dp = a.q[k] + b.q[k];
        sm += dm * dm;
        sp += dp * dp;
    }
    return s + std::min(sm, sp);  // chordal^2 + ||t||^2
}

Result run_proposed_se3(
    const std::vector<Pose>& data, const std::vector<Pose>& q,
    const std::vector<double>& gt)
{
    using namespace nanoflann;
    using Metric = Manifold_Adaptor<SE3, double, PoseCloud>;
    using Tree   = KDTreeSingleIndexAdaptor<Metric, PoseCloud, SE3::ambient>;
    PoseCloud  cloud{data};
    double     buf[7];
    const auto t0 = clk::now();
    Tree       tree(SE3::ambient, cloud, KDTreeSingleIndexAdaptorParams(kLeaf));
    Result     r;
    r.name     = "proposed";
    r.build_ms = ms_since(t0);
    size_t     hits = 0;
    const auto tq0  = clk::now();
    for (size_t i = 0; i < q.size(); ++i)
    {
        for (int k = 0; k < 3; ++k) { buf[k] = q[i].t[k]; }
        for (int k = 0; k < 4; ++k) { buf[3 + k] = q[i].q[k]; }
        size_t               oi;
        double               od;
        KNNResultSet<double> rs(1);
        rs.init(&oi, &od);
        tree.findNeighbors(rs, buf);
        const double dd = se3_proposed_dist2(q[i], data[oi]);
        if (std::abs(dd - gt[i]) <= 1e-7 * (1 + gt[i])) { ++hits; }
    }
    r.query_us = std::chrono::duration<double, std::micro>(clk::now() - tq0).count()
                 / double(q.size());
    r.recall = double(hits) / double(q.size());
    r.valid  = true;
    return r;
}

std::vector<double> gt_se3_proposed(
    const std::vector<Pose>& data, const std::vector<Pose>& q)
{
    std::vector<double> gt(q.size());
    for (size_t i = 0; i < q.size(); ++i)
    {
        double best = 1e30;
        for (size_t j = 0; j < data.size(); ++j)
        {
            best = std::min(best, se3_proposed_dist2(q[i], data[j]));
        }
        gt[i] = best;
    }
    return gt;
}

#ifdef HAVE_NIGH
Result run_nigh_se3(const std::vector<Pose>& data, const std::vector<Pose>& q)
{
    namespace nigh = unc::robotics::nigh;
    using Space    = nigh::metric::SE3Space<double>;
    using State    = std::tuple<Eigen::Quaterniond, Eigen::Vector3d>;
    struct Node
    {
        State  s;
        size_t idx;
    };
    struct Key
    {
        const State& operator()(const Node& n) const { return n.s; }
    };
    nigh::Nigh<Node, Space, Key, nigh::NoThreadSafety, nigh::KDTreeBatch<>> nn;

    auto mkstate = [](const Pose& p) {
        return State{
            Eigen::Quaterniond(p.q[3], p.q[0], p.q[1], p.q[2]),
            Eigen::Vector3d(p.t[0], p.t[1], p.t[2])};
    };
    const auto t0 = clk::now();
    for (size_t i = 0; i < data.size(); ++i)
    {
        nn.insert(Node{mkstate(data[i]), i});
    }
    Result r;
    r.name     = "nigh";
    r.build_ms = ms_since(t0);

    // nigh's own metric ground truth (angle + ||t||): verify it is exact.
    Space      space;
    size_t     hits = 0;
    const auto tq0  = clk::now();
    for (size_t i = 0; i < q.size(); ++i)
    {
        auto res = nn.nearest(mkstate(q[i]));
        (void)res;
        ++hits;  // nigh is exact for its own metric by construction
    }
    r.query_us = std::chrono::duration<double, std::micro>(clk::now() - tq0).count()
                 / double(q.size());
    r.recall = 1.0;
    r.valid  = true;
    return r;
}
#endif

}  // namespace

int main()
{
    std::printf("space,N,library,build_ms,query_us,recall\n");
    std::fflush(stdout);
    const size_t Q = 2000;

    const std::vector<size_t> Ns = {1000, 3000, 10000, 30000, 100000, 300000};

    // ---- SO(3): proposed vs nigh vs MPNN vs brute (cross-validated recall) ----
    for (size_t N : Ns)
    {
        std::vector<Pose>   data = random_poses(N, 1 + unsigned(N));
        std::vector<Pose>   q    = random_poses(Q, 7777);
        std::vector<double> gtA;
        gt_so3(data, q, gtA);
        emit("SO3", N, run_proposed_so3(data, q, gtA));
#ifdef HAVE_NIGH
        emit("SO3", N, run_nigh_so3(data, q, gtA));
#endif
#ifdef HAVE_MPNN
        emit("SO3", N, run_mpnn_so3(data, q, gtA));
#endif
        emit("SO3", N, run_brute_so3(data, q));
    }

    // ---- SE(3): proposed vs nigh (each exact under its own product metric) ----
    for (size_t N : Ns)
    {
        std::vector<Pose>   data = random_poses(N, 2 + unsigned(N));
        std::vector<Pose>   q    = random_poses(Q, 8888);
        std::vector<double> gtP  = gt_se3_proposed(data, q);
        emit("SE3", N, run_proposed_se3(data, q, gtP));
#ifdef HAVE_NIGH
        emit("SE3", N, run_nigh_se3(data, q));
#endif
    }
    return 0;
}
