#include "game_solver.h"
#include <iostream>

GameSolver::GameSolver(const GameField &initialField) {
  GameFieldNode fieldNode(initialField);
  fieldNode.parent = nullptr;
  fieldNode.hierLevel = 0;
  fieldNode.weight = 0;

  auto node = fields.emplace(std::make_pair(initialField.GetFieldState(),
      std::move(fieldNode)));

  if (!node.second)
    throw std::runtime_error("failed to initialize GameSolver");

  opened.push(&(node.first->second));
}

bool GameSolver::FindSolution(const std::string &solutionPattern) {
  uint64_t cycleCounter = 0;
  uint64_t MAX_CYCLES = 2000000;
  ParseSolutionPattern(solutionPattern);

  while (!opened.empty()) {

    if (cycleCounter++ > MAX_CYCLES) {
      std::cout << "MAX_CYCLES limit exceeded" << std::endl;
      return false;
    }

    GameFieldNode *currNode = opened.top();
    opened.pop();

    if (IsSolution(*currNode)) {
      terminalFieldNode = currNode;
      std::cout << "Found solution at " << cycleCounter << std::endl;
      std::cout << "Current level: " << currNode->hierLevel << std::endl;
      std::cout << "Solution:" << std::endl;
      std::cout << currNode->field.Dump() << std::endl;
      return true;
    }

    std::vector<GameField> moves = std::move(currNode->field.GetPossibleMoves());

    for (int iMove = 0; iMove < moves.size(); ++iMove) {
      std::string moveId = moves[iMove].GetFieldState();

      if (fields.find(moveId) != fields.end())
        continue;

      GameFieldNode childNode(std::move(moves[iMove]));
      childNode.parent = currNode;
      childNode.hierLevel = currNode->hierLevel + 1;
      childNode.weight = GetFieldWeight(childNode.field.GetMarkedChipsPattern());

      auto node = fields.emplace(std::make_pair(moveId, std::move(childNode)));
      opened.push(&(node.first->second));
    }
  }

  return false;
}

void GameSolver::DumpSolution(std::ostream &output) {
  GameFieldNode *node = terminalFieldNode;

  while (node) {
    output << node->field.Dump() << std::endl;
    node = node->parent;
  }
}

bool GameSolver::IsSolution(const GameFieldNode &fieldNode) {
  if (fieldNode.field.GetMarkedChipsPattern() != rawSolutionPattern)
    return false;

  return fieldNode.field.IsMarkedChipsOnSameLine();
}

void GameSolver::ParseSolutionPattern(const std::string &pattern) {
  rawSolutionPattern = pattern;
  int iLastPatternSym = pattern.size() - 1;

  for (int iSym = 0; iSym < iLastPatternSym; ++iSym)
    patternRules.emplace(std::make_pair(pattern[iSym], pattern[iSym + 1]));

  patternRules.emplace(std::make_pair(pattern[iLastPatternSym], 0));
}


// heuristic part of evaluation function in a-star algorithm
int GameSolver::GetFieldWeight(const std::string &pattern) {
  if (pattern.size() > patternRules.size()) {
    throw std::runtime_error("GameSolver::GetFieldWeight --> invalid pattern: "
        + pattern);
  }

  // the more value of penalty --> the less size of search space (until some limit)
  // the best value of penalty is 11 --> 8609 iterations
  const int WEIGHT_PENALTY = 11;
  int weight = patternRules.size() - pattern.size();
  int iLastPatternSym = pattern.size() - 1;

  for (int iSym = 0; iSym < iLastPatternSym; ++iSym) {
    auto rules = patternRules.equal_range(pattern[iSym]);
    auto itRule = rules.first;

    for (; itRule != rules.second; ++itRule) {
      if (itRule->second == pattern[iSym + 1])
        break;
    }

    if (itRule == rules.second)
      weight += WEIGHT_PENALTY;
  }

  auto rules = patternRules.equal_range(pattern[iLastPatternSym]);

  for (auto itRule = rules.first; itRule != rules.second; ++itRule) {
    if (itRule->second == 0)
      return weight;
  }

  return weight + WEIGHT_PENALTY;
}
