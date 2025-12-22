#include <chrono>
#include <iostream>
#include <limits>
#include <vector>

#pragma pack(push, 1)

struct /*alignas(N)*/ Node {
  /** Union used because a node can be either a LEAF node or a non-leaf
   * node, so both data fields are never used simultaneously */
  union {
    struct leaf {
      int left, right; //!< Indices of points in leaf node
    } lr;
    struct nonleaf {
      int divfeat; //!< Dimension used for subdivision.
      /// The values used for subdivision.
      float divlow, divhigh;
    } sub;
  } node_type;

  /** Both child nodes ==0 means leaf node (0: for fastest initialization) */
  size_t child1 = 0; // std::numeric_limits<size_t>::max();
  size_t child2 = 0; // std::numeric_limits<size_t>::max();

  bool is_leaf() const { return !child1 && !child2; }
};

#pragma pack(pop)

struct NodePOD {
  int i;
  int j;
};

int main() {
  int N = 200;
  double dt = 0;
  for (int i = 0; i < N; i++) {
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<Node> v;
    v.resize(1000000);
    auto end = std::chrono::high_resolution_clock::now();
    dt += std::chrono::duration_cast<std::chrono::microseconds>(end - start)
              .count();
  }
  std::cout << "Allocation took: " << 1e-3 * (dt / N) << " ms\n";

  return 0;
}