// KITTI loader implementation (MRPT/mola only — no ikd-Tree/PCL here).
#include "kitti_loader.h"

#ifdef HAVE_MOLA_KITTI
#include <algorithm>

#include <mola_input_kitti_dataset/KittiOdometryDataset.h>
#include <mola_yaml/yaml_helpers.h>
#include <mrpt/obs/CObservationPointCloud.h>

FrameStream loadKitti(int maxFrames, float dx, float keepHalf)
{
    auto       kitti = mola::KittiOdometryDataset::Create();
    mola::Yaml cfg   = mola::Yaml::FromText(R"XX(
    params:
      base_dir: ${KITTI_BASE_DIR}
      sequence: ${KITTI_SEQ}
)XX");
    kitti->setMinLoggingLevel(mrpt::system::LVL_ERROR);
    kitti->initialize(mola::parse_yaml(cfg));

    const size_t N = std::min<size_t>(static_cast<size_t>(maxFrames), kitti->datasetSize());
    FrameStream  fs;
    fs.keepHalf = keepHalf;
    for (size_t i = 0; i < N; ++i)
    {
        auto obs = std::dynamic_pointer_cast<mrpt::obs::CObservationPointCloud>(
            kitti->getPointCloud(i));
        if (!obs) continue;
        const auto& pc = *obs->pointcloud;
        const auto& xs = pc.getPointsBufferRef_x();
        const auto& ys = pc.getPointsBufferRef_y();
        const auto& zs = pc.getPointsBufferRef_z();
        const float ox = dx * static_cast<float>(i);  // constant-velocity along +x
        Scan        s;
        s.reserve(xs.size());
        for (size_t j = 0; j < xs.size(); ++j)
            s.push_back({xs[j] + ox, ys[j], zs[j]});
        fs.scans.push_back(std::move(s));
        fs.sensor.push_back({ox, 0.f, 0.f});
    }
    return fs;
}
#endif
