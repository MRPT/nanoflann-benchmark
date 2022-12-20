#include <mola-input-kitti-dataset/KittiOdometryDataset.h>
#include <mola-yaml/yaml_helpers.h>

#include <iostream>

inline mola::KittiOdometryDataset::Ptr benchmark_load_kitti() {
  auto kitti = mola::KittiOdometryDataset::Create();
  {
    mola::Yaml cfg = mola::Yaml::FromText(R"XX(
    params:
      base_dir: ${KITTI_BASE_DIR}
      sequence: ${KITTI_SEQ}
)XX");
    kitti->setMinLoggingLevel(mrpt::system::LVL_ERROR);
    kitti->initialize(mola::parse_yaml(cfg));
    // std::cout << "KITTI dataset loaded: " << kitti->getTimestepCount() << "
    // entries.\n";
  }

  return kitti;
}
