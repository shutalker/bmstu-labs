#ifndef _LAB1_GAME_SOLVER_H_
#define _LAB1_GAME_SOLVER_H_

#include "game_field.h"
#include <ostream>
#include <queue>
#include <map>

struct GameFieldNode {
  GameField field;
  GameFieldNode *parent;
  int hierLevel;
  int weight;

  GameFieldNode(const GameField &f): field(f) {}
  GameFieldNode(GameField &&f): field(f) {}
};

class GameSolver {
 public:
  GameSolver(const GameField &initialField);
  bool FindSolution(const std::string &solutionPattern);
  void DumpSolution(std::ostream &output);

 private:
  struct OpenedInsertComparator {
    bool operator()(GameFieldNode *n1, GameFieldNode *n2) {
      int f1 = n1->hierLevel + n1->weight;
      int f2 = n2->hierLevel + n2->weight;
      return f1 > f2;
    }
  };

  GameFieldNode *terminalFieldNode;
  std::map<std::string, GameFieldNode> fields;
  std::priority_queue<GameFieldNode *, std::vector<GameFieldNode *>,
      OpenedInsertComparator> opened;

  std::string rawSolutionPattern;
  std::multimap<char, char> patternRules;

  bool IsSolution(const GameFieldNode &fieldNode);
  void ParseSolutionPattern(const std::string &pattern);
  int GetFieldWeight(const std::string &pattern);
};

#endif //_LAB1_GAME_SOLVER_H_
