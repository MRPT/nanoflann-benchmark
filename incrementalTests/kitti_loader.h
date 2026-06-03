// KITTI loader declaration. Implemented in kitti_loader.cpp, which is the only
// TU that pulls in MRPT/mola headers (kept apart from the ikd-Tree PCL include
// to avoid header clashes).
#pragma once
#include "common.h"

#ifdef HAVE_MOLA_KITTI
// Load up to maxFrames KITTI scans (env KITTI_BASE_DIR / KITTI_SEQ), each
// translated by (dx*frame) along +x to emulate constant-velocity motion.
FrameStream loadKitti(int maxFrames, float dx, float keepHalf);
#endif
