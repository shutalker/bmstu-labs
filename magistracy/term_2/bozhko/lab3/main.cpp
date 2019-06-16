#include <algorithm>
#include <cmath>
#include <iostream>
#include <functional>
#include <map>
#include <memory>
#include <random>
#include <vector>
#include <unordered_set>

/*
 * graph with clique topology; numbers are indices of vertices
 *
 *         0
 *      1     2
 *   3     4     5
 *      6     7
 *         8
 *
 */

struct FullSquareRootChecker {
  static bool Check1(int *v1) {
    return IsFullSquare(*v1);
  }

  static bool Check2(int *v1, int *v2) {
    return IsFullSquare(10 * (*v1) + (*v2));
  }

  static bool Check3(int *v1, int *v2, int *v3) {
    return IsFullSquare(100 * (*v1) + 10 * (*v2) + (*v3));
  }

  static bool IsFullSquare(int v) {
    double square = sqrt(v);
    int minVal = std::max(0, (int)(floor(square)) - 1);
    int maxVal = (int)(ceil(square)) + 1;

    for (int val = minVal; val <= maxVal; ++val)
      if (val * val == v)
        return true;

    return false;
  }
};

int main() {
  std::vector<int> vertices(9, -1);
  std::vector<int> assignments(9, -1);
  std::vector<int> conflicts(9, -1);
  std::vector<int> domain = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::map<int, int> usedDigits; // digit --> iVertex
  std::map<int, std::function<bool()>> squareRootCheckers;

  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(domain.begin(), domain.end(), g);

  std::cout << "domain: " << std::endl;
  for (const auto &v: domain)
    std::cout << v << " ";
  std::cout << std::endl;

  squareRootCheckers.emplace(0, std::bind(FullSquareRootChecker::Check1,
      &vertices[0]));
  squareRootCheckers.emplace(2, std::bind(FullSquareRootChecker::Check2,
      &vertices[1], &vertices[2]));
  squareRootCheckers.emplace(5, std::bind(FullSquareRootChecker::Check3,
      &vertices[3], &vertices[4], &vertices[5]));
  squareRootCheckers.emplace(7, std::bind(FullSquareRootChecker::Check2,
      &vertices[6], &vertices[7]));
  squareRootCheckers.emplace(8, std::bind(FullSquareRootChecker::Check1,
      &vertices[8]));

  for (int iVertex = 0; iVertex < vertices.size(); ++iVertex) {
    for (int i = 0; i < vertices.size(); ++i)
      std::cout << vertices[i] << " ";

    std::cout << std::endl;

    bool isAssigned = false;

    for (int iVal = assignments[iVertex] + 1; iVal < domain.size(); ++iVal) {
      auto itConflict = usedDigits.find(domain[iVal]);

      if (itConflict != usedDigits.end()) {
        conflicts[iVertex] = std::max(conflicts[iVertex], itConflict->second);
        continue;
      }

      assignments[iVertex] = iVal;
      vertices[iVertex] = domain[iVal];

      if (squareRootCheckers.find(iVertex) != squareRootCheckers.end())
        if (!squareRootCheckers[iVertex]())
          continue;

      isAssigned = true;
      usedDigits.emplace(domain[iVal], iVertex);
      break;
    }

    if (isAssigned)
      continue;

    if (conflicts[iVertex] < 0)
      conflicts[iVertex] = iVertex - 1;

    for (int iReset = conflicts[iVertex] + 1; iReset < iVertex; ++iReset) {
      usedDigits.erase(vertices[iReset]);
      assignments[iReset] = -1;
      vertices[iReset] = -1;
    }

    usedDigits.erase(vertices[conflicts[iVertex]]);
    assignments[iVertex] = -1;
    vertices[iVertex] = -1;
    iVertex = conflicts[iVertex] - 1; // -1 because of increment in the end of for-cycle
  }

  for (int i = 0; i < vertices.size(); ++i)
    std::cout << vertices[i] << " ";

  std::cout << std::endl;

  return 0;
}
