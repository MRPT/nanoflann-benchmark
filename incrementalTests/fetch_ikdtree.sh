#!/bin/bash
# Fetch ikd-Tree (HKU-MARS, GPLv2) for the incremental-index benchmark.
# It is an *external comparison baseline* and is intentionally NOT committed
# into this BSD-licensed repository.
set -e
DEST="$(cd "$(dirname "$0")/.." && pwd)/3rdparty/ikd-Tree"
if [ -d "$DEST/ikd-Tree" ]; then
  echo "ikd-Tree already present at: $DEST"
  exit 0
fi
echo "Cloning ikd-Tree into: $DEST"
git clone --depth 1 https://github.com/hku-mars/ikd-Tree.git "$DEST"
echo "Done."
