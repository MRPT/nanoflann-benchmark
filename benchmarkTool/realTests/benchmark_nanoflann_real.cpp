/***********************************************************************
 * Software License Agreement (BSD License)
 *
 * Copyright 2011-2022 Jose Luis Blanco (joseluisblancoc@gmail.com).
 *   All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *************************************************************************/

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <nanoflann.hpp>
#include <string>

#include "kitti.h"
#include <mrpt/core/Clock.h>
#include <mrpt/core/get_env.h>
#include <mrpt/obs/CObservationPointCloud.h>
#include <mrpt/random/random_shuffle.h>

using namespace std;
using namespace nanoflann;

// This is an exampleof a custom data set class
template <typename T> struct PointCloud {
  struct Point {
    T x, y, z;
  };

  std::vector<Point> pts;

  void loadFromMRPT(const mrpt::maps::CPointsMap &p) {
    const auto xs = p.getPointsBufferRef_x();
    const auto ys = p.getPointsBufferRef_y();
    const auto zs = p.getPointsBufferRef_z();
    pts.resize(xs.size());
    for (size_t i = 0; i < xs.size(); i++) {
      pts[i].x = xs[i];
      pts[i].y = ys[i];
      pts[i].z = zs[i];
    }
  }

  // Must return the number of data points
  inline size_t kdtree_get_point_count() const { return pts.size(); }

  // Returns the dim'th component of the idx'th point in the class:
  // Since this is inlined and the "dim" argument is typically an immediate
  // value, the
  //  "if/else's" are actually solved at compile time.
  inline T kdtree_get_pt(const size_t idx, const size_t dim) const {
    if (dim == 0)
      return pts[idx].x;
    else if (dim == 1)
      return pts[idx].y;
    else
      return pts[idx].z;
  }

  // Optional bounding-box computation: return false to default to a standard
  // bbox computation loop.
  //   Return true if the BBOX was already computed by the class and returned in
  //   "bb" so it can be avoided to redo it again. Look at bb.size() to find out
  //   the expected dimensionality (e.g. 2 or 3 for point clouds)
  template <class BBOX> bool kdtree_get_bbox(BBOX & /* bb */) const {
    return false;
  }
};

std::vector<size_t> generatePartialRandomPermutation(size_t numElements,
                                                     size_t numSelected) {
  std::vector<size_t> idxs(numElements);
  std::iota(idxs.begin(), idxs.end(), 0);
  std::random_device rd; // used for random seed
  std::mt19937 g(rd());
  mrpt::random::partial_shuffle(idxs.begin(), idxs.end(), g, numSelected);
  return idxs;
}

template <typename num_t>
void kdtree_demo(int numTimeSteps, unsigned int decimationCount) {

  auto kitti = benchmark_load_kitti();

  const size_t numDatasetTimesteps = kitti->datasetSize();

  const double timestepIncr = 1.0 / (1.0 * numTimeSteps);

  // buildTime : time required to build the kd-tree index
  // queryTime : time required to find nearest neighbor for a single point
  // in the kd-tree
  for (double p = .0; p < 1.0; p += timestepIncr) {
    size_t pcIdx1 = 1 + (numDatasetTimesteps - 2) * p;
    const auto pcIdx2 = pcIdx1 - 1;

    std::cerr << "Loading timestep: " << pcIdx1 << "/" << numDatasetTimesteps
              << std::endl;

    const auto pc1 =
        std::dynamic_pointer_cast<mrpt::obs::CObservationPointCloud>(
            kitti->getPointCloud(pcIdx1));
    ASSERT_(pc1);
    const auto pc2 =
        std::dynamic_pointer_cast<mrpt::obs::CObservationPointCloud>(
            kitti->getPointCloud(pcIdx2));
    ASSERT_(pc2);

    PointCloud<num_t> PcloudS, PcloudT;
    unsigned int N = std::min(pc1->pointcloud->size(), pc2->pointcloud->size());

    PcloudS.loadFromMRPT(*pc1->pointcloud);
    PcloudT.loadFromMRPT(*pc2->pointcloud);

    // Subsample query pointcloud:
    PointCloud<num_t> cloudQuery;
    const auto nQueries = std::max<size_t>(1000, N / 100);
    {
      const auto idxs =
          generatePartialRandomPermutation(PcloudT.pts.size(), nQueries);
      cloudQuery.pts.resize(nQueries);
      for (size_t i = 0; i < nQueries; i++)
        cloudQuery.pts[i] = PcloudT.pts[idxs[i]];
    }

    std::vector<std::optional<PointCloud<num_t>>> all_cloudS(decimationCount);

    for (unsigned int i = 1; i <= decimationCount; i++) {
      // size of dataset currently being used
      unsigned int currSize = ((i * 1.0) / decimationCount) * N;

      std::cout << currSize << " ";

      // Already created?
      auto &cloudS = all_cloudS.at(i - 1);
      if (!cloudS) {
        cloudS.emplace();

        const auto idxs =
            generatePartialRandomPermutation(PcloudS.pts.size(), currSize);
        cloudS->pts.resize(currSize);
        for (unsigned int j = 0; j < currSize; j++) {
          cloudS->pts[j] = PcloudS.pts[idxs[j]];
        }
      }

      const double begin = mrpt::Clock::nowDouble();

      // construct a kd-tree index:
      nanoflann::KDTreeSingleIndexAdaptorParams params;
      params.leaf_max_size = 10;
      params.n_thread_build =
          mrpt::get_env<uint32_t>("NANOFLANN_BENCHMARK_THREADS", 1);

      using my_kd_tree_t =
          KDTreeSingleIndexAdaptor<L2_Simple_Adaptor<num_t, PointCloud<num_t>>,
                                   PointCloud<num_t>, 3 /* dim */
                                   >;

      my_kd_tree_t index(3 /*dim*/, *cloudS, params);

      const double end = mrpt::Clock::nowDouble();
      double elapsed_secs = (end - begin);

      std::cout << elapsed_secs << " ";

      {
        double elapsed_secs = 0;
        for (unsigned int j = 0; j < nQueries; j++) {
          const num_t query_pt[3] = {cloudQuery.pts[j].x, cloudQuery.pts[j].y,
                                     cloudQuery.pts[j].z};

          size_t ret_index;
          num_t out_dist_sqr;
          const size_t num_results = 1;
          KNNResultSet<num_t> resultSet(num_results);
          resultSet.init(&ret_index, &out_dist_sqr);

          double begin = mrpt::Clock::nowDouble();

          // do a knn search
          index.findNeighbors(resultSet, &query_pt[0], {});

          double end = mrpt::Clock::nowDouble();
          elapsed_secs += end - begin;
        }

        std::cout << elapsed_secs / nQueries << "\n";
      }
    }
  }
}

int main(int argc, char **argv) {
  if (argc != 3) {
    cerr << "Usage: " << argv[0] << " <MAX_TIMESTEPS> <DECIMATION_COUNTS>"
         << endl;
    cerr << "Outputs a table with columns:\n"
            " num_cloud_points build_index_time[s] query_index_time[s]"
         << endl;
    return 0;
  }

  kdtree_demo<double>(atoi(argv[1]), atoi(argv[2]));
  return 0;
}
