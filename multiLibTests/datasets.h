// Dataset loaders for the Euclidean single-tree build/query benchmark.
//
// Two sources:
//   * KITTI Velodyne scans (.bin, float32 x,y,z,intensity) -> 3D point clouds.
//   * Synthetic high-dimensional clustered features (SIFT/SURF-like) for the
//     curse-of-dimensionality regime, when no real descriptor set is available.
//
// All data is returned as a flat row-major std::vector<float> of N*D values,
// so every library adaptor can view it without an extra copy.

#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

namespace bench
{
// Flat row-major point set: point i, coordinate d is data[i*dim + d].
struct PointSet
{
    std::vector<float> data;
    size_t             n   = 0;
    int                dim = 0;

    const float* point(size_t i) const { return data.data() + i * size_t(dim); }
};

// Read one KITTI Velodyne scan: little-endian float32 quadruples
// (x, y, z, reflectance). Only x,y,z are kept. Returns false on I/O error.
inline bool load_kitti_bin(const std::string& path, PointSet& out)
{
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) { return false; }
    std::fseek(f, 0, SEEK_END);
    const long bytes = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    const size_t       n_floats = size_t(bytes) / sizeof(float);
    const size_t       n_pts    = n_floats / 4;
    std::vector<float> raw(n_floats);
    const size_t       got = std::fread(raw.data(), sizeof(float), n_floats, f);
    std::fclose(f);
    if (got != n_floats) { return false; }

    out.dim = 3;
    out.n   = n_pts;
    out.data.resize(n_pts * 3);
    for (size_t i = 0; i < n_pts; ++i)
    {
        out.data[i * 3 + 0] = raw[i * 4 + 0];
        out.data[i * 3 + 1] = raw[i * 4 + 1];
        out.data[i * 3 + 2] = raw[i * 4 + 2];
    }
    return true;
}

// Stack consecutive KITTI scans into a single point cloud, optionally capping
// the total number of points at max_pts (0 = no cap). Used to grow N.
inline PointSet load_kitti_sequence(
    const std::string& seq_dir, size_t n_scans, size_t max_pts)
{
    PointSet acc;
    acc.dim = 3;
    for (size_t s = 0; s < n_scans; ++s)
    {
        char name[32];
        std::snprintf(name, sizeof(name), "%06zu.bin", s);
        PointSet scan;
        if (!load_kitti_bin(seq_dir + "/velodyne/" + name, scan)) { break; }
        acc.data.insert(acc.data.end(), scan.data.begin(), scan.data.end());
        acc.n += scan.n;
        if (max_pts != 0 && acc.n >= max_pts)
        {
            acc.n = max_pts;
            acc.data.resize(max_pts * 3);
            break;
        }
    }
    return acc;
}

// Synthetic high-dimensional feature cloud: a Gaussian mixture (n_clusters
// blobs) in [0,1]^dim, normalized to unit L2 norm to mimic SIFT/SURF/learned
// descriptors that live near a sphere. Deterministic given the seed.
inline PointSet make_highdim(size_t n, int dim, unsigned seed, int n_clusters = 64)
{
    std::mt19937                          rng(seed);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    std::normal_distribution<float>       gauss(0.0f, 0.10f);

    std::vector<std::vector<float>> centers(n_clusters, std::vector<float>(dim));
    for (auto& c : centers)
    {
        for (int d = 0; d < dim; ++d) { c[d] = uni(rng); }
    }

    PointSet out;
    out.dim = dim;
    out.n   = n;
    out.data.resize(n * size_t(dim));
    std::uniform_int_distribution<int> pick(0, n_clusters - 1);
    for (size_t i = 0; i < n; ++i)
    {
        const std::vector<float>& c    = centers[pick(rng)];
        float*                    p    = out.data.data() + i * size_t(dim);
        float                     norm = 0.0f;
        for (int d = 0; d < dim; ++d)
        {
            p[d] = c[d] + gauss(rng);
            norm += p[d] * p[d];
        }
        norm = std::sqrt(norm);
        if (norm < 1e-12f) { norm = 1.0f; }
        for (int d = 0; d < dim; ++d) { p[d] /= norm; }
    }
    return out;
}

// Draw Q query points by sampling (with a small jitter) from an existing set,
// so queries follow the data distribution (the realistic NN-query regime).
inline PointSet make_queries(const PointSet& src, size_t q, unsigned seed)
{
    std::mt19937                          rng(seed);
    std::uniform_int_distribution<size_t> pick(0, src.n - 1);
    std::normal_distribution<float>       jit(0.0f, 0.01f);
    PointSet                              out;
    out.dim = src.dim;
    out.n   = q;
    out.data.resize(q * size_t(src.dim));
    for (size_t i = 0; i < q; ++i)
    {
        const size_t j = pick(rng);
        for (int d = 0; d < src.dim; ++d)
        {
            out.data[i * src.dim + d] = src.point(j)[d] + jit(rng);
        }
    }
    return out;
}

}  // namespace bench
