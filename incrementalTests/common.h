// Common helpers for the incremental-index benchmarks (BSD, part of
// nanoflann-benchmark). Compares the nanoflann logarithmic forest, a single
// static clear+rebuild index, and the new incremental self-balancing index,
// plus (externally) ikd-Tree, on a LiDAR sliding-window workload.
#pragma once

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

// ---------------------------------------------------------------------------
//  Timing
// ---------------------------------------------------------------------------
struct Timer
{
    std::chrono::high_resolution_clock::time_point t0;
    void   tic() { t0 = std::chrono::high_resolution_clock::now(); }
    double toc_ms() const
    {
        const auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
};

// ---------------------------------------------------------------------------
//  A growing point cloud: points are appended and never physically removed
//  (indices stay stable), which is exactly the zero-copy contract the
//  nanoflann dynamic/incremental indices expect.
// ---------------------------------------------------------------------------
struct GrowingCloud
{
    using coord_t = float;
    std::vector<std::array<coord_t, 3>> pts;

    inline size_t kdtree_get_point_count() const { return pts.size(); }
    inline coord_t kdtree_get_pt(const size_t idx, const size_t dim) const
    {
        return pts[idx][dim];
    }
    template <class BBOX>
    bool kdtree_get_bbox(BBOX&) const
    {
        return false;
    }
};

// A single LiDAR scan as a flat list of 3D points (already in world frame).
using Scan = std::vector<std::array<float, 3>>;

// Frame stream: per-frame scan points (world frame) + sensor center, and the
// half-side of the sliding keep cube.
struct FrameStream
{
    std::vector<Scan>                 scans;
    std::vector<std::array<float, 3>> sensor;
    float                             keepHalf = 40.f;
};

// ---------------------------------------------------------------------------
//  Per-frame statistics, accumulated per method.
// ---------------------------------------------------------------------------
struct MethodStats
{
    std::string         name;
    std::vector<double> update_ms;    // insert + trim time per frame
    std::vector<double> query_ms;     // total KNN query time per frame
    std::vector<size_t> live_points;  // logical live points after trim
    std::vector<size_t> phys_points;  // physical stored (incl tombstones)
    size_t              num_queries_per_frame = 0;
};

// Sliding cube keep-region centered at sensor position (cx, cy, cz).
struct KeepBox
{
    float lo[3], hi[3];
};

inline KeepBox makeKeepBox(const float c[3], float half)
{
    KeepBox b;
    for (int d = 0; d < 3; ++d)
    {
        b.lo[d] = c[d] - half;
        b.hi[d] = c[d] + half;
    }
    return b;
}

inline bool inKeep(const std::array<float, 3>& p, const KeepBox& b)
{
    return p[0] >= b.lo[0] && p[0] <= b.hi[0] && p[1] >= b.lo[1] && p[1] <= b.hi[1] &&
           p[2] >= b.lo[2] && p[2] <= b.hi[2];
}
