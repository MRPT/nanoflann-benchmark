/* Incremental-index benchmark (BSD, part of nanoflann-benchmark).
 *
 * Simulates a LiDAR sliding-window mapping workload: scans are accumulated one
 * after another assuming constant-velocity motion; after each scan we trim the
 * map to a cube around the sensor and run a batch of KNN queries. We compare:
 *
 *   forest      : nanoflann KDTreeSingleIndexDynamicAdaptor (logarithmic forest)
 *   rebuild     : nanoflann KDTreeSingleIndexAdaptor, cleared+rebuilt each frame
 *   rebuild_mt  : same, multi-threaded build
 *   incremental : nanoflann KDTreeSingleIndexIncrementalAdaptor (new)
 *   ikd-Tree    : HKU-MARS ikd-Tree (GPLv2, external, comparison only)
 *
 * Data source: KITTI odometry (via mola) if available, else synthetic.
 */

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <string>

#include <nanoflann.hpp>

#include "common.h"
#include "kitti_loader.h"

// ikd-Tree (external, GPLv2): comparison only, never linked into nanoflann.
#include <ikd_Tree.h>

using namespace nanoflann;

static constexpr int K = 5;  // neighbors per query

static FrameStream makeSynthetic(int frames, int ptsPerScan, float dx, float keepHalf)
{
    FrameStream fs;
    fs.keepHalf = keepHalf;
    std::mt19937 g(1234);
    std::uniform_real_distribution<float> U(-50.f, 50.f);
    std::uniform_real_distribution<float> Uz(-3.f, 3.f);
    for (int i = 0; i < frames; ++i)
    {
        const float ox = dx * static_cast<float>(i);
        Scan s;
        s.reserve(ptsPerScan);
        for (int j = 0; j < ptsPerScan; ++j)
            s.push_back({ox + U(g), U(g), Uz(g)});
        fs.scans.push_back(std::move(s));
        fs.sensor.push_back({ox, 0.f, 0.f});
    }
    return fs;
}

// Query points for a frame: a deterministic subsample of that scan.
static Scan makeQueries(const Scan& scan, size_t maxQ)
{
    Scan q;
    if (scan.empty()) return q;
    const size_t step = std::max<size_t>(1, scan.size() / maxQ);
    for (size_t i = 0; i < scan.size() && q.size() < maxQ; i += step) q.push_back(scan[i]);
    return q;
}

// ===========================================================================
//  Method runners
// ===========================================================================
using forest_t = KDTreeSingleIndexDynamicAdaptor<
    L2_Simple_Adaptor<float, GrowingCloud>, GrowingCloud, 3, uint32_t>;
using static_t = KDTreeSingleIndexAdaptor<
    L2_Simple_Adaptor<float, GrowingCloud>, GrowingCloud, 3, uint32_t>;
using inc_t = KDTreeSingleIndexIncrementalAdaptor<
    L2_Simple_Adaptor<float, GrowingCloud>, GrowingCloud, 3, uint32_t>;

static void runForest(const FrameStream& fs, size_t maxQ, MethodStats& st)
{
    st.name = "forest";
    GrowingCloud cloud;
    forest_t     index(3, cloud, KDTreeSingleIndexAdaptorParams(10), 100000000);
    std::vector<uint32_t> live;  // indices currently in window
    Timer timer;

    for (size_t f = 0; f < fs.scans.size(); ++f)
    {
        const auto keep = makeKeepBox(fs.sensor[f].data(), fs.keepHalf);
        timer.tic();
        const uint32_t start = static_cast<uint32_t>(cloud.pts.size());
        for (const auto& p : fs.scans[f]) cloud.pts.push_back(p);
        const uint32_t end = static_cast<uint32_t>(cloud.pts.size());
        if (end > start) index.addPoints(start, end - 1);
        for (uint32_t i = start; i < end; ++i) live.push_back(i);
        // Trim: the forest has no box op, so we must scan & removePoint().
        size_t w = 0;
        for (size_t i = 0; i < live.size(); ++i)
        {
            const uint32_t idx = live[i];
            if (inKeep(cloud.pts[idx], keep)) { live[w++] = idx; }
            else index.removePoint(idx);
        }
        live.resize(w);
        st.update_ms.push_back(timer.toc_ms());

        const Scan q = makeQueries(fs.scans[f], maxQ);
        st.num_queries_per_frame = q.size();
        uint32_t ri[K];
        float    rd[K];
        timer.tic();
        for (const auto& qp : q)
        {
            // The forest wrapper exposes only findNeighbors().
            KNNResultSet<float, uint32_t> rs(K);
            rs.init(ri, rd);
            index.findNeighbors(rs, qp.data());
        }
        st.query_ms.push_back(timer.toc_ms());

        size_t phys = 0;
        for (const auto& sub : index.getAllIndices()) phys += sub.vAcc_.size();
        st.live_points.push_back(live.size());
        st.phys_points.push_back(phys);
    }
}

static void runIncremental(
    const FrameStream& fs, size_t maxQ, float aBal, float aDel, const std::string& name,
    MethodStats& st)
{
    st.name = name;
    GrowingCloud cloud;
    inc_t        index(3, cloud, KDTreeIncrementalIndexParams(aBal, aDel));
    Timer        timer;

    for (size_t f = 0; f < fs.scans.size(); ++f)
    {
        const auto keep = makeKeepBox(fs.sensor[f].data(), fs.keepHalf);
        inc_t::BoundingBox keepBox;
        for (int d = 0; d < 3; ++d) { keepBox[d].low = keep.lo[d]; keepBox[d].high = keep.hi[d]; }

        timer.tic();
        const uint32_t start = static_cast<uint32_t>(cloud.pts.size());
        for (const auto& p : fs.scans[f]) cloud.pts.push_back(p);
        const uint32_t end = static_cast<uint32_t>(cloud.pts.size());
        if (end > start) index.addPoints(start, end - 1);
        index.removeOutsideBox(keepBox);
        st.update_ms.push_back(timer.toc_ms());

        const Scan q = makeQueries(fs.scans[f], maxQ);
        st.num_queries_per_frame = q.size();
        uint32_t ri[K];
        float    rd[K];
        timer.tic();
        for (const auto& qp : q) (void)index.knnSearch(qp.data(), K, ri, rd);
        st.query_ms.push_back(timer.toc_ms());

        st.live_points.push_back(index.size());
        st.phys_points.push_back(index.physicalSize());
    }
}

using inc_mt_t = KDTreeSingleIndexIncrementalAdaptorMT<
    L2_Simple_Adaptor<float, GrowingCloud>, GrowingCloud, 3, uint32_t>;

static void runIncrementalMT(
    const FrameStream& fs, size_t maxQ, float aBal, float aDel, const std::string& name,
    MethodStats& st)
{
    st.name = name;
    GrowingCloud cloud;
    // The MT index reads dataset coords from a background thread, so the backing
    // storage must not reallocate during a rebuild: reserve the full size.
    size_t total = 0;
    for (const auto& s : fs.scans) total += s.size();
    cloud.pts.reserve(total);

    inc_mt_t index(3, cloud, KDTreeIncrementalIndexParams(aBal, aDel), 1.3, 20000);
    Timer    timer;

    for (size_t f = 0; f < fs.scans.size(); ++f)
    {
        const auto keep = makeKeepBox(fs.sensor[f].data(), fs.keepHalf);
        inc_mt_t::BoundingBox keepBox;
        for (int d = 0; d < 3; ++d) { keepBox[d].low = keep.lo[d]; keepBox[d].high = keep.hi[d]; }

        timer.tic();
        const uint32_t start = static_cast<uint32_t>(cloud.pts.size());
        for (const auto& p : fs.scans[f]) cloud.pts.push_back(p);
        const uint32_t end = static_cast<uint32_t>(cloud.pts.size());
        if (end > start) index.addPoints(start, end - 1);
        index.removeOutsideBox(keepBox);
        st.update_ms.push_back(timer.toc_ms());

        const Scan q = makeQueries(fs.scans[f], maxQ);
        st.num_queries_per_frame = q.size();
        uint32_t ri[K];
        float    rd[K];
        timer.tic();
        for (const auto& qp : q) (void)index.knnSearch(qp.data(), K, ri, rd);
        st.query_ms.push_back(timer.toc_ms());

        st.live_points.push_back(index.size());
        st.phys_points.push_back(index.physicalSize());
    }
}

static void runRebuild(const FrameStream& fs, size_t maxQ, unsigned threads, MethodStats& st)
{
    st.name = threads > 1 ? "rebuild_mt" : "rebuild";
    GrowingCloud window;  // compacted: only current-window points
    Timer        timer;

    for (size_t f = 0; f < fs.scans.size(); ++f)
    {
        const auto keep = makeKeepBox(fs.sensor[f].data(), fs.keepHalf);
        timer.tic();
        for (const auto& p : fs.scans[f]) window.pts.push_back(p);
        // Trim window in place.
        size_t w = 0;
        for (size_t i = 0; i < window.pts.size(); ++i)
            if (inKeep(window.pts[i], keep)) window.pts[w++] = window.pts[i];
        window.pts.resize(w);
        // Rebuild a fresh static index over the current window.
        static_t index(
            3, window, KDTreeSingleIndexAdaptorParams(10, KDTreeSingleIndexAdaptorFlags::None,
                                                      threads));
        st.update_ms.push_back(timer.toc_ms());

        const Scan q = makeQueries(fs.scans[f], maxQ);
        st.num_queries_per_frame = q.size();
        uint32_t ri[K];
        float    rd[K];
        timer.tic();
        for (const auto& qp : q) (void)index.knnSearch(qp.data(), K, ri, rd);
        st.query_ms.push_back(timer.toc_ms());

        st.live_points.push_back(window.pts.size());
        st.phys_points.push_back(window.pts.size());
    }
}

static void runIkd(const FrameStream& fs, size_t maxQ, MethodStats& st)
{
    st.name = "ikd-Tree";
    // Heap-allocate: KD_TREE embeds a ~tens-of-MB operation queue by value, so a
    // stack instance overflows the stack.
    auto  treePtr = std::make_shared<KD_TREE<ikdTree_PointType>>(0.5f, 0.7f, 0.2f);
    auto& tree    = *treePtr;
    Timer timer;
    bool  built = false;

    for (size_t f = 0; f < fs.scans.size(); ++f)
    {
        const auto keep = makeKeepBox(fs.sensor[f].data(), fs.keepHalf);
        KD_TREE<ikdTree_PointType>::PointVector add;
        add.reserve(fs.scans[f].size());
        for (const auto& p : fs.scans[f]) add.emplace_back(p[0], p[1], p[2]);

        timer.tic();
        if (!built && !add.empty())
        {
            tree.Build(add);
            built = true;
        }
        else if (!add.empty())
        {
            tree.Add_Points(add, false);
        }
        // removeOutsideBox via the 6 outer slabs (complement of the keep cube).
        if (built)
        {
            const float BIG = 1e6f;
            std::vector<BoxPointType> boxes;
            for (int d = 0; d < 3; ++d)
            {
                BoxPointType lo, hi;
                for (int e = 0; e < 3; ++e)
                {
                    lo.vertex_min[e] = -BIG;
                    lo.vertex_max[e] = BIG;
                    hi.vertex_min[e] = -BIG;
                    hi.vertex_max[e] = BIG;
                }
                lo.vertex_max[d] = keep.lo[d];  // slab below keep on axis d
                hi.vertex_min[d] = keep.hi[d];  // slab above keep on axis d
                boxes.push_back(lo);
                boxes.push_back(hi);
            }
            tree.Delete_Point_Boxes(boxes);
        }
        st.update_ms.push_back(timer.toc_ms());

        const Scan q = makeQueries(fs.scans[f], maxQ);
        st.num_queries_per_frame = q.size();
        timer.tic();
        for (const auto& qp : q)
        {
            KD_TREE<ikdTree_PointType>::PointVector nn;
            std::vector<float>                      nd;
            tree.Nearest_Search(ikdTree_PointType(qp[0], qp[1], qp[2]), K, nn, nd);
        }
        st.query_ms.push_back(timer.toc_ms());

        st.live_points.push_back(static_cast<size_t>(tree.validnum()));
        st.phys_points.push_back(static_cast<size_t>(tree.size()));
    }
}

// ===========================================================================
//  Reporting
// ===========================================================================
static double median(std::vector<double> v)
{
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}
static double mean(const std::vector<double>& v)
{
    if (v.empty()) return 0;
    double s = 0;
    for (double x : v) s += x;
    return s / v.size();
}
static double p95(std::vector<double> v)
{
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, static_cast<size_t>(v.size() * 0.95))];
}

static void report(const std::vector<MethodStats>& all, size_t warmup)
{
    // Skip the warmup frames (window still filling) for steady-state stats.
    printf(
        "\n%-14s %12s %12s %12s %12s %12s %10s %10s\n", "method", "upd_med[ms]", "upd_p95[ms]",
        "upd_mean[ms]", "qry_med[ms]", "qry_us/q", "live", "phys");
    printf("%s\n", std::string(110, '-').c_str());
    for (const auto& s : all)
    {
        std::vector<double> upd(s.update_ms.begin() + std::min(warmup, s.update_ms.size()),
                                s.update_ms.end());
        std::vector<double> qry(s.query_ms.begin() + std::min(warmup, s.query_ms.size()),
                                s.query_ms.end());
        const double usPerQ = s.num_queries_per_frame
                                  ? mean(qry) * 1000.0 / s.num_queries_per_frame
                                  : 0.0;
        const size_t live = s.live_points.empty() ? 0 : s.live_points.back();
        const size_t phys = s.phys_points.empty() ? 0 : s.phys_points.back();
        printf(
            "%-14s %12.3f %12.3f %12.3f %12.3f %12.3f %10zu %10zu\n", s.name.c_str(), median(upd),
            p95(upd), mean(upd), median(qry), usPerQ, live, phys);
    }
}

static void dumpCsv(const std::vector<MethodStats>& all, const std::string& path)
{
    FILE* fp = fopen(path.c_str(), "w");
    if (!fp) return;
    fprintf(fp, "method,frame,update_ms,query_ms,live,phys\n");
    for (const auto& s : all)
        for (size_t f = 0; f < s.update_ms.size(); ++f)
            fprintf(fp, "%s,%zu,%.4f,%.4f,%zu,%zu\n", s.name.c_str(), f, s.update_ms[f],
                    s.query_ms[f], s.live_points[f], s.phys_points[f]);
    fclose(fp);
    printf("Wrote per-frame CSV: %s\n", path.c_str());
}

int main(int argc, char** argv)
{
    int   frames   = argc > 1 ? atoi(argv[1]) : 80;
    float keepHalf = argc > 2 ? atof(argv[2]) : 40.f;
    float dx       = argc > 3 ? atof(argv[3]) : 1.5f;
    size_t maxQ    = argc > 4 ? atoi(argv[4]) : 500;
    std::string csv = argc > 5 ? argv[5] : "stats_incremental.csv";

    FrameStream fs;
    const char* src = "synthetic";
#ifdef HAVE_MOLA_KITTI
    if (getenv("KITTI_BASE_DIR"))
    {
        try
        {
            fs  = loadKitti(frames, dx, keepHalf);
            src = "KITTI";
        }
        catch (const std::exception& e)
        {
            fprintf(stderr, "KITTI load failed (%s); using synthetic.\n", e.what());
        }
    }
#endif
    if (fs.scans.empty()) fs = makeSynthetic(frames, 30000, dx, keepHalf);

    size_t totalPts = 0;
    for (auto& s : fs.scans) totalPts += s.size();
    printf("Source: %s | frames=%zu keepHalf=%.1f dx=%.2f avgScan=%zu k=%d queries/frame<=%zu\n",
           src, fs.scans.size(), keepHalf, dx, fs.scans.empty() ? 0 : totalPts / fs.scans.size(), K,
           maxQ);

    auto log = [](const char* m) { fprintf(stderr, "  running %s ...\n", m); fflush(stderr); };

    std::vector<MethodStats> all;
    {
        log("forest");
        MethodStats s;
        runForest(fs, maxQ, s);
        all.push_back(s);
    }
    {
        log("rebuild");
        MethodStats s;
        runRebuild(fs, maxQ, 1, s);
        all.push_back(s);
    }
    {
        log("rebuild_mt");
        MethodStats s;
        runRebuild(fs, maxQ, 0 /*=all cores*/, s);
        all.push_back(s);
    }
    // Incremental: sweep a few (alpha_balance, alpha_deleted) design points.
    struct Cfg { float bal, del; const char* tag; };
    const Cfg cfgs[] = {
        {0.75f, 0.5f, "inc_b75_d50"},
        {0.70f, 0.30f, "inc_b70_d30"},
        {0.65f, 0.30f, "inc_b65_d30"},
        {0.85f, 0.50f, "inc_b85_d50"},
    };
    for (const auto& c : cfgs)
    {
        log(c.tag);
        MethodStats s;
        runIncremental(fs, maxQ, c.bal, c.del, c.tag, s);
        all.push_back(s);
    }
    {
        log("inc_async (MT)");
        MethodStats s;
        runIncrementalMT(fs, maxQ, 0.85f, 0.5f, "inc_async", s);
        all.push_back(s);
    }
    {
        log("ikd-Tree");
        MethodStats s;
        runIkd(fs, maxQ, s);
        all.push_back(s);
    }

    const size_t warmup = std::min<size_t>(fs.scans.size() / 3, 30);
    printf("\n(steady-state stats skip the first %zu warmup frames)\n", warmup);
    report(all, warmup);
    dumpCsv(all, csv);
    return 0;
}
