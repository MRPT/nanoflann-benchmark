// Representation-agnostic fingerprint of nanoflann KD-trees and query results.
//
// Compiled once per nanoflann header (baseline: pointer-linked nodes in a
// PooledAllocator; candidate: one std::vector<Node> in pre-order). Each binary
// prints one line per case with hashes of the tree structure, the vAcc_
// permutation and of kNN / radius / box query answers. Diffing the two outputs
// checks that the new storage gives exactly the same trees and answers, for
// every thread count. kNN distances are also checked against brute force.
#include <nanoflann.hpp>

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace
{
int g_failures = 0;

#define CHECK(cond, ...)                                  \
    do                                                    \
    {                                                     \
        if (!(cond))                                      \
        {                                                 \
            std::printf("FAIL %s:%d: ", __FILE__, __LINE__); \
            std::printf(__VA_ARGS__);                     \
            std::printf("\n");                            \
            g_failures++;                                 \
        }                                                 \
    } while (0)

struct Hash
{
    uint64_t h = 1469598103934665603ULL;
    void     bytes(const void* p, size_t n)
    {
        const auto* c = static_cast<const unsigned char*>(p);
        for (size_t i = 0; i < n; i++)
        {
            h ^= c[i];
            h *= 1099511628211ULL;
        }
    }
    template <typename T>
    void feed(const T& v)
    {
        // Normalize through a wide type so float/double/ints hash by value.
        if constexpr (std::is_floating_point<T>::value)
        {
            const double d = static_cast<double>(v);
            bytes(&d, sizeof(d));
        }
        else
        {
            const uint64_t u = static_cast<uint64_t>(v);
            bytes(&u, sizeof(u));
        }
    }
};

// ---- dataset adaptor, generic in element type and dimension ---------------
template <typename T>
struct Cloud
{
    size_t         dim = 3;
    std::vector<T> data;  // row-major, N x dim

    size_t kdtree_get_point_count() const { return data.size() / dim; }
    T      kdtree_get_pt(const size_t idx, const size_t d) const { return data[idx * dim + d]; }
    template <class BBOX>
    bool kdtree_get_bbox(BBOX&) const
    {
        return false;
    }
    const T* pt(size_t i) const { return &data[i * dim]; }
};

enum class Shape
{
    Uniform,
    Duplicates,
    Collinear,
    Skewed,
    Lattice,
    Clustered
};
const char* shapeName(Shape s)
{
    switch (s)
    {
        case Shape::Uniform:
            return "uniform";
        case Shape::Duplicates:
            return "dups";
        case Shape::Collinear:
            return "collinear";
        case Shape::Skewed:
            return "skewed";
        case Shape::Lattice:
            return "lattice";
        case Shape::Clustered:
            return "clustered";
    }
    return "?";
}

template <typename T>
Cloud<T> makeCloud(Shape s, size_t N, size_t dim, unsigned seed)
{
    std::mt19937                           rng(seed);
    std::uniform_real_distribution<double> u(0, 1);
    Cloud<T>                               c;
    c.dim = dim;
    c.data.resize(N * dim);
    // Integer element types get a range that suits them.
    const double scale = std::is_floating_point<T>::value ? 100.0 : 200.0;
    const size_t side  = std::max<size_t>(2, static_cast<size_t>(std::cbrt(double(N))) + 1);
    for (size_t i = 0; i < N; i++)
    {
        for (size_t d = 0; d < dim; d++)
        {
            double v = 0;
            switch (s)
            {
                case Shape::Uniform:
                    v = u(rng) * scale;
                    break;
                case Shape::Duplicates:
                    v = 7;
                    break;
                case Shape::Collinear:
                    v = (d == 0) ? u(rng) * scale : 3;
                    break;
                case Shape::Skewed:
                    // 99% of the points in a tiny corner.
                    v = (i % 100 == 0) ? u(rng) * scale : u(rng) * scale * 0.001;
                    break;
                case Shape::Lattice:
                {
                    // Integer lattice, cubic bbox: several dims tie for the
                    // widest spread.
                    size_t k = i;
                    for (size_t e = 0; e < d; e++) k /= side;
                    v = double(k % side);
                    break;
                }
                case Shape::Clustered:
                    v = std::floor(u(rng) * 5) * 20 + u(rng) * 0.5;
                    break;
            }
            c.data[i * dim + d] = static_cast<T>(v);
        }
    }
    return c;
}

// ---- walk the tree whatever its storage -----------------------------------
template <class, class = void>
struct has_root_node : std::false_type
{
};
template <class T>
struct has_root_node<T, std::void_t<decltype(std::declval<T&>().root_node_)>> : std::true_type
{
};

template <class Index, class NodeP>
void walk(const Index& idx, NodeP n, Hash& h, size_t& count, size_t& maxDepth, size_t depth,
          size_t& nextPoint, bool& layoutOk)
{
    count++;
    maxDepth = std::max(maxDepth, depth);
    bool  leaf;
    NodeP c1 = nullptr;
    NodeP c2 = nullptr;
    if constexpr (has_root_node<Index>::value)
    {
        leaf = (n->child1 == nullptr);
        c1   = n->child1;
        c2   = n->child2;
    }
    else
    {
        leaf = n->isLeaf();
        c1   = n + 1;
        c2   = n + n->child2;
    }
    h.feed(leaf ? 1 : 0);
    if (leaf)
    {
        const size_t l = n->node_type.lr.left;
        const size_t r = n->node_type.lr.right;
        h.feed(l);
        h.feed(r);
        // Leaves must tile vAcc_ in order, with no gap.
        if (l != nextPoint) layoutOk = false;
        nextPoint = r;
        return;
    }
    h.feed(n->node_type.sub.divfeat);
    h.feed(n->node_type.sub.divlow);
    h.feed(n->node_type.sub.divhigh);
    walk(idx, c1, h, count, maxDepth, depth + 1, nextPoint, layoutOk);
    walk(idx, c2, h, count, maxDepth, depth + 1, nextPoint, layoutOk);
}

template <class Index>
std::string treeDigest(const Index& idx)
{
    Hash   h;
    size_t count = 0, maxDepth = 0, nextPoint = 0;
    bool   layoutOk = true;
    for (const auto& v : idx.vAcc_) h.feed(v);
    // root_bbox_ is never computed for an empty index.
    if (!idx.vAcc_.empty())
        for (const auto& iv : idx.root_bbox_)
        {
            h.feed(iv.low);
            h.feed(iv.high);
        }
    if constexpr (has_root_node<Index>::value)
    {
        if (idx.root_node_) walk(idx, idx.root_node_, h, count, maxDepth, 0, nextPoint, layoutOk);
    }
    else
    {
        if (!idx.nodes_.empty())
        {
            walk(idx, idx.nodes_.data(), h, count, maxDepth, 0, nextPoint, layoutOk);
            // Pre-order array: the walk must visit exactly every array slot.
            CHECK(count == idx.nodes_.size(), "walk visited %zu of %zu nodes", count,
                  idx.nodes_.size());
        }
    }
    CHECK(layoutOk && nextPoint == idx.vAcc_.size(), "leaves do not tile vAcc_");
    char buf[128];
    std::snprintf(buf, sizeof(buf), "tree=%016" PRIx64 " nodes=%zu depth=%zu", h.h, count,
                  maxDepth);
    return buf;
}

// ---- queries ----------------------------------------------------------------
template <typename T, typename DistT, class Index, class IndexType>
std::string queryDigest(const Index& idx, const Cloud<T>& cloud, bool bruteForce, bool doBox)
{
    const size_t N   = cloud.kdtree_get_point_count();
    const size_t dim = cloud.dim;
    Hash         hk, hr, hb;
    if (N == 0) return "q=empty";

    std::mt19937                           rng(99);
    std::uniform_int_distribution<size_t>  pick(0, N - 1);
    std::uniform_real_distribution<double> jitter(-1.5, 1.5);
    const size_t                           nQ = std::min<size_t>(300, 20 + N / 50);
    std::vector<T>                         q(dim);

    const DistT radius = static_cast<DistT>(std::is_floating_point<T>::value ? 25.0 : 30.0);
    for (size_t qi = 0; qi < nQ; qi++)
    {
        const T* p = cloud.pt(pick(rng));
        for (size_t d = 0; d < dim; d++) q[d] = static_cast<T>(p[d] + jitter(rng));

        // kNN, k = 1 and k = 10
        for (size_t k : {size_t(1), size_t(10)})
        {
            std::vector<IndexType> ids(k);
            std::vector<DistT>     ds(k);
            nanoflann::KNNResultSet<DistT, IndexType> rs(k);
            rs.init(ids.data(), ds.data());
            idx.findNeighbors(rs, q.data());
            const size_t got = rs.size();
            for (size_t i = 0; i < got; i++)
            {
                hk.feed(ids[i]);
                hk.feed(ds[i]);
            }
            if (bruteForce && std::is_same<DistT, T>::value && std::is_floating_point<T>::value)
            {
                std::vector<double> all(N);
                for (size_t j = 0; j < N; j++)
                {
                    double s = 0;
                    for (size_t d = 0; d < dim; d++)
                    {
                        const double e = double(cloud.pt(j)[d]) - double(q[d]);
                        s += e * e;
                    }
                    all[j] = s;
                }
                std::nth_element(all.begin(), all.begin() + std::min(k, N) - 1, all.end());
                std::sort(all.begin(), all.begin() + std::min(k, N));
                CHECK(got == std::min(k, N), "knn size %zu != %zu", got, std::min(k, N));
                for (size_t i = 0; i < got; i++)
                {
                    const double tol = 1e-4 * std::max(1.0, all[i]);
                    CHECK(std::abs(double(ds[i]) - all[i]) <= tol, "knn[%zu] %g != brute %g", i,
                          double(ds[i]), all[i]);
                }
            }
        }
        // Radius search
        {
            std::vector<nanoflann::ResultItem<IndexType, DistT>> m;
            nanoflann::SearchParameters                          sp;
            sp.sorted = true;
            (void)idx.radiusSearch(q.data(), radius, m, sp);
            hr.feed(m.size());
            for (const auto& r : m)
            {
                hr.feed(r.first);
                hr.feed(r.second);
            }
        }
        // Axis-aligned box search
        if (doBox)
        {
            typename Index::BoundingBox bb;
            nanoflann::resize(bb, dim);
            for (size_t d = 0; d < dim; d++)
            {
                bb[d].low  = static_cast<T>(q[d] - 3);
                bb[d].high = static_cast<T>(q[d] + 3);
            }
            struct BoxRS
            {
                std::vector<IndexType>& out;
                size_t                  size() const { return out.size(); }
                bool                    addPoint(DistT, IndexType i)
                {
                    out.push_back(i);
                    return true;
                }
            };
            std::vector<IndexType> inBox;
            BoxRS                  brs{inBox};
            const size_t           n = idx.findWithinBox(brs, bb);
            std::sort(inBox.begin(), inBox.end());
            hb.feed(n);
            for (auto i : inBox) hb.feed(i);
        }
    }
    char buf[160];
    std::snprintf(buf, sizeof(buf), "knn=%016" PRIx64 " rad=%016" PRIx64 " box=%016" PRIx64, hk.h,
                  hr.h, hb.h);
    return buf;
}

template <typename T, typename DistT, class IndexType, class Index>
std::string queryDigest_(const Index& idx, const Cloud<T>& cloud, bool bruteForce, bool doBox)
{
    return queryDigest<T, DistT, Index, IndexType>(idx, cloud, bruteForce, doBox);
}

// ---- static index cases -----------------------------------------------------
template <typename T, typename DistT, class Metric, int DIM, typename IndexType>
void runStatic(const char* typeName, size_t dim)
{
    using cloud_t = Cloud<T>;
    using tree_t  = nanoflann::KDTreeSingleIndexAdaptor<Metric, cloud_t, DIM, IndexType>;

    const Shape shapes[] = {Shape::Uniform,  Shape::Duplicates, Shape::Collinear,
                            Shape::Skewed,   Shape::Lattice,    Shape::Clustered};
    const size_t Ns[]    = {0, 1, 2, 10, 11, 100, 1000, 20000, 150000};
    const size_t leaves[] = {1, 10, 64};
#ifdef NANOFLANN_NO_THREADS
    const unsigned threads[] = {1};
#else
    const unsigned threads[] = {1, 2, 3, 4, 8, 16, 0};
#endif

    const char*  envMaxN = std::getenv("FP_MAXN");
    const size_t maxN    = envMaxN ? std::stoul(envMaxN) : ~size_t(0);
    for (Shape s : shapes)
        for (size_t N : Ns)
        {
            if (N > maxN) continue;
            // Keep the high-dim and integer cases affordable.
            if (dim > 3 && N > 20000) continue;
            if (s == Shape::Duplicates && N > 20000) continue;
            const cloud_t cloud = makeCloud<T>(s, N, dim, unsigned(N * 7 + int(s)));
            for (size_t leaf : leaves)
            {
                if (leaf == 1 && N > 20000) continue;
                std::string serialTree;
                for (unsigned T_ : threads)
                {
                    nanoflann::KDTreeSingleIndexAdaptorParams p(leaf);
                    p.n_thread_build = T_;
                    tree_t      idx(dim, cloud, p);
                    std::string td = treeDigest(idx);
                    // Serial vs concurrent, within this header.
                    if (T_ == 1)
                        serialTree = td;
                    else
                        CHECK(td == serialTree, "%s %s N=%zu leaf=%zu T=%u: MT tree differs",
                              typeName, shapeName(s), N, leaf, T_);

                    const bool bf = (N <= 20000) && (T_ == 1 || T_ == 4);
                    const bool box = true;
                    std::string qd = (T_ == 1 || T_ == 4 || T_ == 0)
                                         ? queryDigest_<T, DistT, IndexType>(idx, cloud, bf, box)
                                         : std::string("-");
                    std::printf("static %s dim=%zu %s N=%zu leaf=%zu T=%u %s %s\n", typeName, dim,
                                shapeName(s), N, leaf, T_, td.c_str(), qd.c_str());

                    if (T_ == 1 && N > 0 && !std::getenv("FP_NOSAVE"))
                    {
                        // save / load roundtrip
                        std::stringstream ss(std::ios::in | std::ios::out | std::ios::binary);
                        idx.saveIndex(ss);
                        tree_t loaded(
                            dim, cloud,
                            nanoflann::KDTreeSingleIndexAdaptorParams(
                                leaf, nanoflann::KDTreeSingleIndexAdaptorFlags::SkipInitialBuildIndex));
                        loaded.loadIndex(ss);
                        CHECK(treeDigest(loaded) == td, "save/load changed the tree");
                        CHECK((queryDigest_<T, DistT, IndexType>(loaded, cloud, false, box) == qd),
                              "save/load changed query answers");
                    }
                }
            }
        }

    // In-place rebuilds over a changing dataset (grow, shrink, grow): the
    // result must match a fresh index over the same points.
    for (unsigned T_ : threads)
    {
        cloud_t                                   cloud;
        nanoflann::KDTreeSingleIndexAdaptorParams p(10);
        p.n_thread_build = T_;
        cloud            = makeCloud<T>(Shape::Uniform, 5000, dim, 1);
        tree_t idx(dim, cloud, p);
        for (size_t N : {size_t(60000), size_t(300), size_t(0), size_t(90000), size_t(1)})
        {
            if (dim > 3 && N > 20000) N /= 10;
            cloud = makeCloud<T>(Shape::Skewed, N, dim, unsigned(N + 3));
            idx.buildIndex();
            tree_t fresh(dim, cloud, p);
            CHECK(treeDigest(idx) == treeDigest(fresh), "%s rebuild N=%zu T=%u differs from fresh",
                  typeName, N, T_);
            if (N > 0)
                CHECK((queryDigest_<T, DistT, IndexType>(idx, cloud, N <= 20000, true) ==
                           queryDigest_<T, DistT, IndexType>(fresh, cloud, false, true)),
                      "rebuild query mismatch");
        }
    }
}

// ---- dynamic index ----------------------------------------------------------
template <typename T, typename DistT, class Metric, int DIM>
void runDynamic(const char* typeName, size_t dim)
{
    using cloud_t = Cloud<T>;
    using dyn_t   = nanoflann::KDTreeSingleIndexDynamicAdaptor<Metric, cloud_t, DIM>;
#ifdef NANOFLANN_NO_THREADS
    const unsigned threads[] = {1};
#else
    const unsigned threads[] = {1, 4};
#endif
    for (Shape s : {Shape::Uniform, Shape::Skewed, Shape::Duplicates})
        for (unsigned T_ : threads)
        {
            const size_t  N     = 30000;
            const cloud_t all = makeCloud<T>(s, N, dim, 5);
            // The index starts empty (its constructor indexes whatever the
            // dataset already holds); points are appended before each add.
            cloud_t cloud;
            cloud.dim = dim;
            cloud.data.reserve(all.data.size());
            nanoflann::KDTreeSingleIndexAdaptorParams p(10);
            p.n_thread_build = T_;
            dyn_t idx(dim, cloud, p, N);
            // Add in uneven chunks so that sub-trees get merged / rebuilt.
            size_t done = 0, chunk = 1;
            while (done < N)
            {
                const size_t e = std::min(N, done + chunk);
                cloud.data.insert(cloud.data.end(), all.data.begin() + done * dim,
                                  all.data.begin() + e * dim);
                idx.addPoints(done, e - 1);
                done  = e;
                chunk = chunk * 3 + 1;
            }
            // Remove every 7th point.
            for (size_t i = 0; i < N; i += 7) idx.removePoint(i);

            // Queries vs brute force on the surviving points.
            Hash                                   h;
            std::mt19937                           rng(11);
            std::uniform_int_distribution<size_t>  pick(0, N - 1);
            std::vector<T>                         q(dim);
            for (int qi = 0; qi < 200; qi++)
            {
                const T* p0 = cloud.pt(pick(rng));
                for (size_t d = 0; d < dim; d++) q[d] = p0[d];
                const size_t                     k = 5;
                std::vector<uint32_t>            ids(k);
                std::vector<DistT>               ds(k);
                nanoflann::KNNResultSet<DistT, uint32_t> rs(k);
                rs.init(ids.data(), ds.data());
                idx.findNeighbors(rs, q.data(), nanoflann::SearchParameters());
                for (size_t i = 0; i < rs.size(); i++)
                {
                    CHECK(ids[i] % 7 != 0, "dynamic: removed point %u returned", ids[i]);
                    h.feed(ds[i]);
                }
                std::vector<double> all;
                for (size_t j = 0; j < N; j++)
                {
                    if (j % 7 == 0) continue;
                    double ss = 0;
                    for (size_t d = 0; d < dim; d++)
                    {
                        const double e = double(cloud.pt(j)[d]) - double(q[d]);
                        ss += e * e;
                    }
                    all.push_back(ss);
                }
                std::sort(all.begin(), all.end());
                for (size_t i = 0; i < rs.size(); i++)
                    CHECK(std::abs(double(ds[i]) - all[i]) <= 1e-4 * std::max(1.0, all[i]),
                          "dynamic knn[%zu] %g != brute %g", i, double(ds[i]), all[i]);
            }
            std::printf("dynamic %s dim=%zu %s T=%u knn=%016" PRIx64 "\n", typeName, dim,
                        shapeName(s), T_, h.h);
        }
}

// For the uint8 case: a metric with a signed, wide distance type.
template <class T, class DS>
using L2u8 = nanoflann::L2_Simple_Adaptor<T, DS, int32_t, uint32_t>;
}  // namespace

int main()
{
    using namespace nanoflann;
    const bool onlyDynamic = std::getenv("FP_ONLY_DYNAMIC") != nullptr;
    if (!onlyDynamic)
    {
    // float / double, 3D, fixed DIM (the LiDAR case)
    runStatic<float, float, L2_Simple_Adaptor<float, Cloud<float>>, 3, uint32_t>("f32/L2S/D3", 3);
    runStatic<double, double, L2_Simple_Adaptor<double, Cloud<double>>, 3, uint32_t>("f64/L2S/D3",
                                                                                     3);
    // Runtime dimension
    runStatic<float, float, L2_Simple_Adaptor<float, Cloud<float>>, -1, uint32_t>("f32/L2S/Ddyn", 3);
    // 64-bit IndexType: exercises the uint64_t NodeIndex path
    runStatic<double, double, L2_Adaptor<double, Cloud<double>, double, size_t>, 3, size_t>(
        "f64/L2/D3/idx64", 3);
    // Higher dimension, L2 with SIMD-ish unrolled loop
    runStatic<float, float, L2_Adaptor<float, Cloud<float>>, -1, uint32_t>("f32/L2/D12", 12);
    // Integer elements, L1
    runStatic<int32_t, int32_t, L1_Adaptor<int32_t, Cloud<int32_t>>, 2, uint32_t>("i32/L1/D2", 2);
    // Unsigned 8-bit elements with a signed 32-bit distance
    runStatic<uint8_t, int32_t, L2u8<uint8_t, Cloud<uint8_t>>, 3, uint32_t>("u8/L2S/D3", 3);

    }
    runDynamic<float, float, L2_Simple_Adaptor<float, Cloud<float>>, 3>("f32", 3);
    runDynamic<double, double, L2_Simple_Adaptor<double, Cloud<double>>, -1>("f64/Ddyn", 3);

    std::printf("SUMMARY failures=%d\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
