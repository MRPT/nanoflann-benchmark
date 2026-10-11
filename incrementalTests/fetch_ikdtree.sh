#!/bin/bash
# Fetch ikd-Tree (HKU-MARS, GPLv2) for the incremental-index benchmark.
# It is an *external comparison baseline* and is intentionally NOT committed
# into this BSD-licensed repository.
set -e
# Commit the published results were measured with.
IKDTREE_COMMIT=c0e36a16b6e4d557d3783b16911207f6398dd478
DEST="$(cd "$(dirname "$0")/.." && pwd)/3rdparty/ikd-Tree"
if [ -d "$DEST/ikd-Tree" ]; then
  echo "ikd-Tree already present at: $DEST"
  exit 0
fi
echo "Cloning ikd-Tree into: $DEST"
git clone https://github.com/hku-mars/ikd-Tree.git "$DEST"
git -C "$DEST" checkout "$IKDTREE_COMMIT"
echo "Done."
