// Verify that the concurrent KD-tree build produces exactly the same tree
// (and the same query answers) as the serial build.
#include <nanoflann.hpp>

#include <cstdint>
#include <cstdio>
#include <iostream>
#include <random>
#include <string>
#include <vector>

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

using num_t   = double;
using cloud_t = PointCloud<num_t>;
using tree_t  = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<num_t, cloud_t>, cloud_t, 3>;

struct Fingerprint
{
    uint64_t hash  = 1469598103934665603ULL;  // FNV-1a offset basis
    size_t   nodes = 0;
    size_t   leaves = 0;
    size_t   maxDepth = 0;

    void feed(const void* data, size_t len)
    {
        const auto* p = static_cast<const unsigned char*>(data);
        for (size_t i = 0; i < len; i++)
        {
            hash ^= p[i];
            hash *= 1099511628211ULL;
        }
    }
    void feed(uint64_t v) { feed(&v, sizeof(v)); }
    void feed(double v) { feed(&v, sizeof(v)); }

    bool operator==(const Fingerprint& o) const
    {
        return hash == o.hash && nodes == o.nodes && leaves == o.leaves &&
               maxDepth == o.maxDepth;
    }
};

void walk(const tree_t::Node* n, Fingerprint& fp, size_t depth)
{
    if (!n) return;
    fp.nodes++;
    fp.maxDepth = std::max(fp.maxDepth, depth);
    const bool isLeaf = (n->child1 == nullptr && n->child2 == nullptr);
    fp.feed(static_cast<uint64_t>(isLeaf ? 1 : 0));
    if (isLeaf)
    {
        fp.leaves++;
        fp.feed(static_cast<uint64_t>(n->node_type.lr.left));
        fp.feed(static_cast<uint64_t>(n->node_type.lr.right));
        return;
    }
    fp.feed(static_cast<uint64_t>(n->node_type.sub.divfeat));
    fp.feed(static_cast<double>(n->node_type.sub.divlow));
    fp.feed(static_cast<double>(n->node_type.sub.divhigh));
    walk(n->child1, fp, depth + 1);
    walk(n->child2, fp, depth + 1);
}

Fingerprint treeFingerprint(const tree_t& idx)
{
    Fingerprint fp;
    // the point-index permutation is part of the tree's identity
    for (const auto v : idx.vAcc_) fp.feed(static_cast<uint64_t>(v));
    for (const auto& iv : idx.root_bbox_)
    {
        fp.feed(static_cast<double>(iv.low));
        fp.feed(static_cast<double>(iv.high));
    }
    walk(idx.root_node_, fp, 0);
    return fp;
}

std::vector<std::pair<uint32_t, num_t>> queryAll(
    const tree_t& idx, const std::vector<std::array<num_t, 3>>& qs, size_t K)
{
    std::vector<std::pair<uint32_t, num_t>> out;
    std::vector<uint32_t>                   ri(K);
    std::vector<num_t>                      rd(K);
    for (const auto& q : qs)
    {
        const size_t n = idx.knnSearch(q.data(), K, ri.data(), rd.data());
        for (size_t i = 0; i < n; i++) out.emplace_back(ri[i], rd[i]);
        out.emplace_back(0xffffffffu, static_cast<num_t>(n));
    }
    return out;
}

int failures = 0;

void runCase(const std::string& name, const cloud_t& cloud, size_t leafSize)
{
    std::mt19937                           rng(7);
    std::uniform_real_distribution<num_t>  d(-60, 60);
    std::vector<std::array<num_t, 3>>      qs;
    for (int i = 0; i < 200; i++) qs.push_back({d(rng), d(rng), d(rng)});

    nanoflann::KDTreeSingleIndexAdaptorParams p;
    p.leaf_max_size   = leafSize;
    p.n_thread_build  = 1;
    tree_t      ref(3, cloud, p);
    const auto  refFp = treeFingerprint(ref);
    const auto  refQ  = queryAll(ref, qs, 10);
    const size_t refMem = ref.pool_.usedMemory + ref.pool_.wastedMemory;

    for (unsigned int T : {2u, 3u, 4u, 8u, 16u, 32u, 0u})
    {
        nanoflann::KDTreeSingleIndexAdaptorParams pp;
        pp.leaf_max_size  = leafSize;
        pp.n_thread_build = T;
        tree_t     mt(3, cloud, pp);
        const auto fp = treeFingerprint(mt);
        const auto q  = queryAll(mt, qs, 10);
        const bool okTree = (fp == refFp);
        const bool okQ    = (q == refQ);
        if (!okTree || !okQ)
        {
            failures++;
            std::printf(
                "FAIL %-28s N=%-7zu leaf=%-3zu T=%-2u  tree=%s query=%s "
                "(nodes %zu vs %zu, hash %016llx vs %016llx)\n",
                name.c_str(), cloud.pts.size(), leafSize, T, okTree ? "ok" : "DIFF",
                okQ ? "ok" : "DIFF", fp.nodes, refFp.nodes,
                (unsigned long long)fp.hash, (unsigned long long)refFp.hash);
        }
        const size_t mtMem = mt.pool_.usedMemory + mt.pool_.wastedMemory;
        std::printf(
            "MEM  %-28s N=%-7zu leaf=%-3zu T=%-2u  pool=%-10zu serial=%-10zu x%.2f\n",
            name.c_str(), cloud.pts.size(), leafSize, T, mtMem, refMem,
            refMem ? double(mtMem) / double(refMem) : 1.0);
    }
    std::printf(
        "ok   %-28s N=%-7zu leaf=%-3zu  nodes=%zu leaves=%zu depth=%zu mem=%zu\n",
        name.c_str(), cloud.pts.size(), leafSize, refFp.nodes, refFp.leaves, refFp.maxDepth,
        refMem);
}

cloud_t uniformCloud(size_t N, uint32_t seed)
{
    cloud_t c;
    std::mt19937                          rng(seed);
    std::uniform_real_distribution<num_t> d(-50, 50);
    c.pts.resize(N);
    for (size_t i = 0; i < N; i++) c.pts[i] = {d(rng), d(rng), d(rng)};
    return c;
}

cloud_t duplicateCloud(size_t N)
{
    cloud_t c;
    c.pts.assign(N, {1.0, 2.0, 3.0});
    return c;
}

cloud_t collinearCloud(size_t N)
{
    cloud_t c;
    c.pts.resize(N);
    for (size_t i = 0; i < N; i++) c.pts[i] = {static_cast<num_t>(i), 0.0, 0.0};
    return c;
}

cloud_t skewedCloud(size_t N, uint32_t seed)
{
    // 99% of the points in a tiny cluster, 1% far away: makes the two children
    // of the root very unbalanced, which is exactly what the PR's
    // "spawn the smaller side" heuristic keys on.
    cloud_t c;
    std::mt19937                          rng(seed);
    std::uniform_real_distribution<num_t> tight(-0.01, 0.01);
    std::uniform_real_distribution<num_t> wide(-1000, 1000);
    c.pts.resize(N);
    for (size_t i = 0; i < N; i++)
    {
        if (i % 100 == 0)
            c.pts[i] = {wide(rng), wide(rng), wide(rng)};
        else
            c.pts[i] = {tight(rng), tight(rng), tight(rng)};
    }
    return c;
}

int main()
{
    for (size_t N : {1u, 2u, 3u, 10u, 11u, 100u, 511u, 512u, 513u, 1024u, 1025u, 5000u, 100000u})
    {
        runCase("uniform", uniformCloud(N, 1234u), 10);
    }
    runCase("uniform-leaf1", uniformCloud(50000, 99u), 1);
    runCase("uniform-leaf64", uniformCloud(200000, 99u), 64);
    runCase("duplicates", duplicateCloud(50000), 10);
    runCase("collinear", collinearCloud(50000), 10);
    runCase("skewed", skewedCloud(200000, 5u), 10);
    runCase("skewed-leaf1", skewedCloud(50000, 6u), 1);

    if (failures == 0)
        std::printf("\nALL STRUCTURAL/QUERY CHECKS PASSED\n");
    else
        std::printf("\n%d FAILURES\n", failures);
    return failures ? 1 : 0;
}
