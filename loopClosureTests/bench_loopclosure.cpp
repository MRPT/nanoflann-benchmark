// C5 application benchmark: loop-closure candidate retrieval over real SE(3)
// keyframe poses (KITTI odometry ground truth).
//
// Task: for every keyframe i of a sequence, retrieve the k nearest keyframes j
// with a temporal exclusion |i - j| > gap (so trivially-adjacent frames do not
// count as loop-closure candidates), under the SE(3) product metric
//
//   d^2 = ||t_i - t_j||^2 + wrot^2 * min(||q_i - q_j||^2, ||q_i + q_j||^2)
//
// i.e. squared translation distance plus the (weighted, double-cover-aware)
// squared chordal rotation distance. The rotation weight wrot [meters per unit
// chordal distance] is applied by scaling the quaternion coordinates, which
// requires no support from the index (a chordal metric is homogeneous).
//
// Compared methods (all with identical temporal filtering):
//   * brute        exhaustive scan with the exact geodesic metric (ground truth)
//   * proposed     manifold KD-tree on Product<Rn<3>, SO3> (claim: exact)
//   * naive        Euclidean KD-tree on the raw 7-D coordinates, quaternions
//                  stored canonicalized (qw >= 0), as published in the dataset.
//                  This is the common practitioner shortcut.
//   * naive_sc     same, but quaternions stored sign-continuous along the
//                  trajectory (what integrating odometry produces). Any fixed
//                  sign convention has a seam; this shows the failure is not an
//                  artifact of one storage convention.
//   * rerank       3-D Euclidean KD-tree on translation only; fetch kc >= k
//                  candidates, re-rank them by the full SE(3) metric. The other
//                  common workaround.
//
// CSV (stdout): seq,N,method,wrot,k,gap,build_ms,query_us,recall,top1
// Per-query dump (for the trajectory miss-map figure):
//   results/perquery_<seq>_w<wrot>_k<k>.csv: i,x,z,recall_naive,recall_rerank
//
// Anonymity (double-blind RA-L): the library under test is labeled "proposed".

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <nanoflann.hpp>

using clk = std::chrono::steady_clock;

namespace
{
constexpr size_t kLeaf = 10;

double ms_since(clk::time_point t0)
{
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

// A keyframe pose: translation (m) + unit quaternion (x, y, z, w).
struct Pose
{
    double t[3];
    double q[4];
};

// Row-major 3x3 rotation matrix -> unit quaternion (x, y, z, w), Shepperd.
void rot_to_quat(const double R[9], double q[4])
{
    const double tr = R[0] + R[4] + R[8];
    if (tr > 0)
    {
        double s = std::sqrt(tr + 1.0) * 2;  // s = 4*qw
        q[3]     = 0.25 * s;
        q[0]     = (R[7] - R[5]) / s;
        q[1]     = (R[2] - R[6]) / s;
        q[2]     = (R[3] - R[1]) / s;
    }
    else if (R[0] > R[4] && R[0] > R[8])
    {
        double s = std::sqrt(1.0 + R[0] - R[4] - R[8]) * 2;  // s = 4*qx
        q[3]     = (R[7] - R[5]) / s;
        q[0]     = 0.25 * s;
        q[1]     = (R[1] + R[3]) / s;
        q[2]     = (R[2] + R[6]) / s;
    }
    else if (R[4] > R[8])
    {
        double s = std::sqrt(1.0 + R[4] - R[0] - R[8]) * 2;  // s = 4*qy
        q[3]     = (R[2] - R[6]) / s;
        q[0]     = (R[1] + R[3]) / s;
        q[1]     = 0.25 * s;
        q[2]     = (R[5] + R[7]) / s;
    }
    else
    {
        double s = std::sqrt(1.0 + R[8] - R[0] - R[4]) * 2;  // s = 4*qz
        q[3]     = (R[3] - R[1]) / s;
        q[0]     = (R[2] + R[6]) / s;
        q[1]     = (R[5] + R[7]) / s;
        q[2]     = 0.25 * s;
    }
    // canonicalize like the KITTI TUM files (qw >= 0)
    if (q[3] < 0)
        for (int i = 0; i < 4; ++i) q[i] = -q[i];
}

// MulRan global_pose.csv: "stamp,r11,r12,r13,tx,r21,r22,r23,ty,r31,r32,r33,tz"
// (100 Hz INS ground truth). Decimated to keyframes every `kfDist` meters of
// travel; translations recentered on the first pose (UTM offsets are huge).
std::vector<Pose> load_mulran(const std::string& path, double kfDist)
{
    std::ifstream f(path);
    if (!f)
    {
        std::cerr << "Cannot open: " << path << "\n";
        std::exit(1);
    }
    std::vector<Pose> out;
    std::string       line;
    double            t0[3]    = {0, 0, 0};
    bool              first    = true;
    double            last[3]  = {0, 0, 0};
    while (std::getline(f, line))
    {
        if (line.empty() || line[0] == '#') continue;
        for (auto& ch : line)
            if (ch == ',') ch = ' ';
        std::istringstream ss(line);
        double             ts, M[12];
        ss >> ts;
        for (int i = 0; i < 12; ++i) ss >> M[i];
        if (!ss) continue;
        const double R[9] = {M[0], M[1], M[2], M[4], M[5], M[6], M[8], M[9], M[10]};
        const double t[3] = {M[3], M[7], M[11]};
        if (first)
        {
            for (int i = 0; i < 3; ++i) t0[i] = t[i];
        }
        Pose p;
        for (int i = 0; i < 3; ++i) p.t[i] = t[i] - t0[i];
        const double dx = p.t[0] - last[0];
        const double dy = p.t[1] - last[1];
        const double dz = p.t[2] - last[2];
        if (!first && dx * dx + dy * dy + dz * dz < kfDist * kfDist) continue;
        rot_to_quat(R, p.q);
        out.push_back(p);
        for (int i = 0; i < 3; ++i) last[i] = p.t[i];
        first = false;
    }
    return out;
}

// TUM trajectory format: timestamp tx ty tz qx qy qz qw
std::vector<Pose> load_tum(const std::string& path)
{
    std::ifstream     f(path);
    std::vector<Pose> out;
    if (!f)
    {
        std::cerr << "Cannot open: " << path << "\n";
        std::exit(1);
    }
    std::string line;
    while (std::getline(f, line))
    {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        double             ts;
        Pose               p;
        ss >> ts >> p.t[0] >> p.t[1] >> p.t[2] >> p.q[0] >> p.q[1] >> p.q[2] >>
            p.q[3];
        if (!ss) continue;
        // normalize, just in case
        double n = 0;
        for (int i = 0; i < 4; ++i) n += p.q[i] * p.q[i];
        n = std::sqrt(n);
        for (int i = 0; i < 4; ++i) p.q[i] /= n;
        out.push_back(p);
    }
    return out;
}

// Flat 7-D coordinate buffer (tx,ty,tz, wrot*qx, wrot*qy, wrot*qz, wrot*qw).
// `signContinuous` re-applies the sign continuity an odometry integrator
// would produce instead of the per-row canonicalization of the dataset files.
std::vector<double> make_coords(
    const std::vector<Pose>& poses, double wrot, bool signContinuous)
{
    std::vector<double> c(poses.size() * 7);
    double              prev[4] = {0, 0, 0, 1};
    for (size_t i = 0; i < poses.size(); ++i)
    {
        double q[4] = {
            poses[i].q[0], poses[i].q[1], poses[i].q[2], poses[i].q[3]};
        if (signContinuous && i > 0)
        {
            const double dot =
                q[0] * prev[0] + q[1] * prev[1] + q[2] * prev[2] + q[3] * prev[3];
            if (dot < 0)
                for (int j = 0; j < 4; ++j) q[j] = -q[j];
        }
        for (int j = 0; j < 4; ++j) prev[j] = q[j];

        for (int j = 0; j < 3; ++j) c[i * 7 + j] = poses[i].t[j];
        for (int j = 0; j < 4; ++j) c[i * 7 + 3 + j] = wrot * q[j];
    }
    return c;
}

// Exact squared SE(3) metric on two 7-D coordinate rows (quat already scaled).
double se3_dist2(const double* a, const double* b)
{
    double dt = 0;
    for (int j = 0; j < 3; ++j)
    {
        const double d = a[j] - b[j];
        dt += d * d;
    }
    double sm = 0, sp = 0;
    for (int j = 3; j < 7; ++j)
    {
        const double dm = a[j] - b[j];
        const double dp = a[j] + b[j];
        sm += dm * dm;
        sp += dp * dp;
    }
    return dt + std::min(sm, sp);
}

// --- nanoflann dataset adaptor over the flat buffer ---
struct Cloud
{
    const std::vector<double>* c   = nullptr;
    size_t                     dim = 7;

    size_t kdtree_get_point_count() const { return c->size() / dim; }
    double kdtree_get_pt(const size_t i, const size_t k) const
    {
        return (*c)[i * dim + k];
    }
    template <class BBOX>
    bool kdtree_get_bbox(BBOX&) const
    {
        return false;
    }
};

// k-NN result set with temporal exclusion |index - queryIdx| <= gap.
// Same interface contract as nanoflann::KNNResultSet; worstDist() returns
// +inf until full, so pruning stays admissible under the exclusion.
class FilteredKNNResultSet
{
   public:
    using DistanceType = double;
    using IndexType    = size_t;

    FilteredKNNResultSet(size_t capacity, size_t queryIdx, size_t gap)
        : cap_(capacity), q_(queryIdx), gap_(gap)
    {
        idx_.resize(cap_);
        d_.resize(cap_);
    }
    void reset() { count_ = 0; }

    size_t size() const { return count_; }
    bool   empty() const { return count_ == 0; }
    bool   full() const { return count_ == cap_; }

    bool addPoint(DistanceType dist, IndexType index)
    {
        const size_t lo = index > q_ ? index - q_ : q_ - index;
        if (lo <= gap_) return true;  // temporally excluded; keep searching
        size_t i;
        for (i = count_; i > 0; --i)
        {
            if (d_[i - 1] <= dist) break;
            if (i < cap_)
            {
                d_[i]   = d_[i - 1];
                idx_[i] = idx_[i - 1];
            }
        }
        if (i < cap_)
        {
            d_[i]   = dist;
            idx_[i] = index;
        }
        if (count_ < cap_) ++count_;
        return true;
    }

    DistanceType worstDist() const
    {
        return count_ < cap_ ? std::numeric_limits<double>::max()
                             : d_[count_ - 1];
    }
    void sort() {}  // kept sorted on insertion

    const std::vector<size_t>& indices() const { return idx_; }

   private:
    size_t              cap_, q_, gap_;
    size_t              count_ = 0;
    std::vector<size_t> idx_;
    std::vector<double> d_;
};

struct MethodResult
{
    double build_ms = 0;
    double query_us = 0;
    // per-query k-NN index sets
    std::vector<std::vector<size_t>> nn;
};

// recall@k of `m` against ground truth `gt` (mean set intersection / k)
void score(
    const std::vector<std::vector<size_t>>& gt,
    const std::vector<std::vector<size_t>>& nn, double& recall, double& top1)
{
    double rsum = 0, t1 = 0;
    for (size_t i = 0; i < gt.size(); ++i)
    {
        const std::set<size_t> g(gt[i].begin(), gt[i].end());
        size_t                 hits = 0;
        for (size_t j : nn[i]) hits += g.count(j);
        rsum += double(hits) / double(gt[i].size());
        if (!nn[i].empty() && !gt[i].empty() && nn[i][0] == gt[i][0]) t1 += 1;
    }
    recall = rsum / double(gt.size());
    top1   = t1 / double(gt.size());
}

// per-query recall (for the trajectory miss map)
std::vector<double> per_query_recall(
    const std::vector<std::vector<size_t>>& gt,
    const std::vector<std::vector<size_t>>& nn)
{
    std::vector<double> r(gt.size());
    for (size_t i = 0; i < gt.size(); ++i)
    {
        const std::set<size_t> g(gt[i].begin(), gt[i].end());
        size_t                 hits = 0;
        for (size_t j : nn[i]) hits += g.count(j);
        r[i] = double(hits) / double(gt[i].size());
    }
    return r;
}

// --- methods -----------------------------------------------------------------

// exhaustive exact scan (ground truth)
MethodResult run_brute(const std::vector<double>& c, size_t k, size_t gap)
{
    const size_t N = c.size() / 7;
    MethodResult r;
    r.nn.resize(N);
    const auto t0 = clk::now();
    for (size_t i = 0; i < N; ++i)
    {
        FilteredKNNResultSet rs(k, i, gap);
        for (size_t j = 0; j < N; ++j)
            rs.addPoint(se3_dist2(&c[i * 7], &c[j * 7]), j);
        r.nn[i].assign(rs.indices().begin(), rs.indices().begin() + rs.size());
    }
    r.query_us = ms_since(t0) * 1e3 / double(N);
    return r;
}

// proposed manifold KD-tree (exact)
MethodResult run_proposed(const std::vector<double>& c, size_t k, size_t gap)
{
    using namespace nanoflann;
    using Metric = Manifold_Adaptor<SE3, double, Cloud>;
    using Tree   = KDTreeSingleIndexAdaptor<Metric, Cloud, SE3::ambient>;

    const size_t N = c.size() / 7;
    Cloud        cloud{&c, 7};
    MethodResult r;
    r.nn.resize(N);

    auto t0 = clk::now();
    Tree tree(SE3::ambient, cloud, KDTreeSingleIndexAdaptorParams(kLeaf));
    r.build_ms = ms_since(t0);

    t0 = clk::now();
    for (size_t i = 0; i < N; ++i)
    {
        FilteredKNNResultSet rs(k, i, gap);
        tree.findNeighbors(rs, &c[i * 7]);
        r.nn[i].assign(rs.indices().begin(), rs.indices().begin() + rs.size());
    }
    r.query_us = ms_since(t0) * 1e3 / double(N);
    return r;
}

// naive Euclidean KD-tree on the raw 7-D coordinates (wrong metric)
MethodResult run_naive(const std::vector<double>& c, size_t k, size_t gap)
{
    using namespace nanoflann;
    using Metric = L2_Simple_Adaptor<double, Cloud>;
    using Tree   = KDTreeSingleIndexAdaptor<Metric, Cloud, 7>;

    const size_t N = c.size() / 7;
    Cloud        cloud{&c, 7};
    MethodResult r;
    r.nn.resize(N);

    auto t0 = clk::now();
    Tree tree(7, cloud, KDTreeSingleIndexAdaptorParams(kLeaf));
    r.build_ms = ms_since(t0);

    t0 = clk::now();
    for (size_t i = 0; i < N; ++i)
    {
        FilteredKNNResultSet rs(k, i, gap);
        tree.findNeighbors(rs, &c[i * 7]);
        r.nn[i].assign(rs.indices().begin(), rs.indices().begin() + rs.size());
    }
    r.query_us = ms_since(t0) * 1e3 / double(N);
    return r;
}

// translation-only 3-D tree + re-rank kc candidates by the full SE(3) metric
MethodResult run_rerank(
    const std::vector<double>& c, size_t k, size_t gap, size_t kc)
{
    using namespace nanoflann;
    using Metric = L2_Simple_Adaptor<double, Cloud>;
    using Tree   = KDTreeSingleIndexAdaptor<Metric, Cloud, 3>;

    const size_t        N = c.size() / 7;
    std::vector<double> t3(N * 3);
    for (size_t i = 0; i < N; ++i)
        for (int j = 0; j < 3; ++j) t3[i * 3 + j] = c[i * 7 + j];
    Cloud        cloud{&t3, 3};
    MethodResult r;
    r.nn.resize(N);

    auto t0 = clk::now();
    Tree tree(3, cloud, KDTreeSingleIndexAdaptorParams(kLeaf));
    r.build_ms = ms_since(t0);

    t0 = clk::now();
    std::vector<std::pair<double, size_t>> cand;
    for (size_t i = 0; i < N; ++i)
    {
        FilteredKNNResultSet rs(kc, i, gap);
        tree.findNeighbors(rs, &t3[i * 3]);
        cand.clear();
        for (size_t m = 0; m < rs.size(); ++m)
        {
            const size_t j = rs.indices()[m];
            cand.emplace_back(se3_dist2(&c[i * 7], &c[j * 7]), j);
        }
        std::sort(cand.begin(), cand.end());
        const size_t kk = std::min(k, cand.size());
        r.nn[i].resize(kk);
        for (size_t m = 0; m < kk; ++m) r.nn[i][m] = cand[m].second;
    }
    r.query_us = ms_since(t0) * 1e3 / double(N);
    return r;
}

void emit(
    const std::string& seq, size_t N, const char* method, double wrot, size_t k,
    size_t gap, const MethodResult& r, double recall, double top1)
{
    std::printf(
        "%s,%zu,%s,%g,%zu,%zu,%.3f,%.4f,%.6f,%.6f\n", seq.c_str(), N, method,
        wrot, k, gap, r.build_ms, r.query_us, recall, top1);
    std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv)
{
    // Modes:
    //   bench_loopclosure [--kitti] <kitti_poses_dir> <seq> [<seq> ...]
    //   bench_loopclosure --mulran <mulran_base_dir> <seq> [<seq> ...]
    //     (reads <base>/<seq>/global_pose.csv, keyframes every 1 m)
    int  argi   = 1;
    bool mulran = false;
    if (argc > 1 && std::string(argv[1]) == "--mulran")
    {
        mulran = true;
        ++argi;
    }
    else if (argc > 1 && std::string(argv[1]) == "--kitti")
        ++argi;
    if (argc - argi < 2)
    {
        std::cerr << "Usage: " << argv[0]
                  << " [--kitti] <kitti_poses_dir> <seq> [<seq> ...]\n"
                  << "       " << argv[0]
                  << " --mulran <mulran_base_dir> <seq> [<seq> ...]\n";
        return 1;
    }
    const std::string posesDir   = argv[argi++];
    const double      kfDistM    = 1.0;  // MulRan keyframe spacing [m]
    // ground-plane axes for the per-query dump: KITTI camera frame = (x, z);
    // MulRan ENU-like frame = (x, y)
    const int gp0 = 0;
    const int gp1 = mulran ? 1 : 2;

    const std::vector<double> wrots = {5.0, 15.0, 50.0};
    const std::vector<size_t> ks    = {1, 8};
    const size_t              gap   = 100;  // frames (~10 s at 10 Hz)
    const size_t              kcand = 32;   // candidates for re-rank

    std::printf("seq,N,method,wrot,k,gap,build_ms,query_us,recall,top1\n");

    for (int a = argi; a < argc; ++a)
    {
        const std::string seq = argv[a];
        const auto        poses =
            mulran
                ? load_mulran(posesDir + "/" + seq + "/global_pose.csv", kfDistM)
                : load_tum(posesDir + "/" + seq + "-tum.txt");
        const size_t N = poses.size();
        std::cerr << "seq " << seq << ": " << N << " keyframes\n";

        for (const double wrot : wrots)
        {
            const auto cCanon = make_coords(poses, wrot, false);
            const auto cSignC = make_coords(poses, wrot, true);

            for (const size_t k : ks)
            {
                // ground truth (sign convention irrelevant for the true metric)
                const auto gt = run_brute(cCanon, k, gap);
                emit(seq, N, "brute", wrot, k, gap, gt, 1.0, 1.0);

                double rec, t1;

                const auto pr = run_proposed(cCanon, k, gap);
                score(gt.nn, pr.nn, rec, t1);
                emit(seq, N, "proposed", wrot, k, gap, pr, rec, t1);
                if (rec < 1.0 - 1e-9)
                    std::cerr << "WARNING: proposed recall < 1: " << rec
                              << " (seq " << seq << ", wrot " << wrot << ", k "
                              << k << ")\n";

                const auto nv = run_naive(cCanon, k, gap);
                score(gt.nn, nv.nn, rec, t1);
                emit(seq, N, "naive", wrot, k, gap, nv, rec, t1);

                const auto nsc = run_naive(cSignC, k, gap);
                // GT on sign-continuous coords is the same metric; compare to gt
                score(gt.nn, nsc.nn, rec, t1);
                emit(seq, N, "naive_sc", wrot, k, gap, nsc, rec, t1);

                const auto rr = run_rerank(cCanon, k, gap, kcand);
                score(gt.nn, rr.nn, rec, t1);
                emit(seq, N, "rerank", wrot, k, gap, rr, rec, t1);

                // per-query dump for the trajectory miss-map figure
                if (k == 8)
                {
                    const auto rn  = per_query_recall(gt.nn, nv.nn);
                    const auto rs  = per_query_recall(gt.nn, nsc.nn);
                    const auto rr2 = per_query_recall(gt.nn, rr.nn);
                    char       path[256];
                    std::snprintf(
                        path, sizeof(path),
                        "results/perquery_%s_w%g_k%zu.csv", seq.c_str(), wrot,
                        k);
                    std::ofstream pf(path);
                    pf << "i,x,z,recall_naive,recall_naive_sc,recall_rerank\n";
                    for (size_t i = 0; i < N; ++i)
                        pf << i << "," << poses[i].t[gp0] << ","
                           << poses[i].t[gp1] << "," << rn[i] << "," << rs[i]
                           << "," << rr2[i] << "\n";
                }
            }
        }
    }
    return 0;
}
