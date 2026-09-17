// Serial vs concurrent build equivalence for the *dynamic* index (the other
// call site the PR touches), including churn (addPoints/removePoint).
#include <nanoflann.hpp>

#include <cstdio>
#include <random>
#include <vector>

using num_t = double;

struct Cloud
{
    struct Point
    {
        num_t x, y, z;
    };
    std::vector<Point> pts;
    size_t             kdtree_get_point_count() const { return pts.size(); }
    num_t              kdtree_get_pt(const size_t i, const size_t d) const
    {
        return d == 0 ? pts[i].x : (d == 1 ? pts[i].y : pts[i].z);
    }
    template <class BBOX>
    bool kdtree_get_bbox(BBOX&) const
    {
        return false;
    }
};

using dyn_t = nanoflann::KDTreeSingleIndexDynamicAdaptor<
    nanoflann::L2_Simple_Adaptor<num_t, Cloud>, Cloud, 3>;

int failures = 0;

std::vector<std::pair<uint32_t, num_t>> run(const Cloud& c, unsigned int T, size_t churn)
{
    nanoflann::KDTreeSingleIndexAdaptorParams p;
    p.leaf_max_size  = 10;
    p.n_thread_build = T;
    dyn_t idx(3, c, p);

    std::mt19937 rng(42);
    for (size_t i = 0; i < churn; i++)
    {
        idx.removePoint(rng() % c.pts.size());
    }

    std::mt19937                          qrng(7);
    std::uniform_real_distribution<num_t> d(-60, 60);
    std::vector<std::pair<uint32_t, num_t>> out;
    std::vector<uint32_t>                 ri(10);
    std::vector<num_t>                    rd(10);
    for (int q = 0; q < 300; q++)
    {
        const num_t                 pt[3] = {d(qrng), d(qrng), d(qrng)};
        nanoflann::KNNResultSet<num_t, uint32_t> rs(10);
        rs.init(ri.data(), rd.data());
        idx.findNeighbors(rs, pt);
        for (size_t i = 0; i < rs.size(); i++) out.emplace_back(ri[i], rd[i]);
        out.emplace_back(0xffffffffu, static_cast<num_t>(rs.size()));
    }
    return out;
}

int main()
{
    for (size_t N : {1000u, 20000u, 200000u})
    {
        Cloud                                 c;
        std::mt19937                          rng(3);
        std::uniform_real_distribution<num_t> d(-50, 50);
        c.pts.resize(N);
        for (size_t i = 0; i < N; i++) c.pts[i] = {d(rng), d(rng), d(rng)};

        for (size_t churn : {size_t(0), N / 10})
        {
            const auto ref = run(c, 1, churn);
            for (unsigned int T : {2u, 4u, 8u, 16u, 0u})
            {
                const auto got = run(c, T, churn);
                if (got != ref)
                {
                    failures++;
                    std::printf("FAIL dynamic N=%zu churn=%zu T=%u\n", N, churn, T);
                }
            }
            std::printf("ok   dynamic N=%-7zu churn=%-6zu (%zu result records)\n", N, churn,
                        ref.size());
        }
    }
    std::printf(failures ? "\n%d FAILURES\n" : "\nDYNAMIC INDEX CHECKS PASSED\n", failures);
    return failures ? 1 : 0;
}
