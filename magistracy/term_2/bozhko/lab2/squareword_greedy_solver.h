#ifndef _LAB2_SQUAREWORD_GREEDY_SOLVER_H_
#define _LAB2_SQUAREWORD_GREEDY_SOLVER_H_

#include <fstream>
#include <map>
#include <ostream>
#include <random>
#include <string>
#include <vector>
#include <set>

class SquarewordGreedySolver {
 public:
  struct GameField {
   public:
    // char - field symbol; bool - is field symbol initialized by config file
    std::vector<std::pair<char, bool>> cells;
    int dimension = 0;
    int duplicatedSymbols = 0;

    void Dump(std::ostream &output) const;
    void RecountDuplicatedSymbols();
    static bool DuplicatedSymdolsComparator(const GameField &gf1, const GameField &gf2) {
      return gf1.duplicatedSymbols < gf2.duplicatedSymbols;
    }

   private:
    int RecountDuplicatedSymbolsByRows();
    int RecountDuplicatedSymbolsByCols();
    int RecountDuplicatedSymbolsByDiags();
  };

  bool InitFromFile(const std::string &filename);
  void InitFromGameField(const GameField &gf);
  bool RunGreedySearch(bool randomizeGameField, std::vector<GameField> *terminalFields);
  void DumpCurrentGameField(std::ostream &output) const { currField.Dump(output); }
  void DumpAlphabet(std::ostream &output) const;
  void DumpNonFixedSymbols(std::ostream &output) const;

 private:
  GameField currField;
  std::set<char> alphabet;
  std::vector<std::vector<char>> nonFixedSymbols; // i --> column; vector<char> --> symbols for i-column
  std::vector<int> availableSymbolCount; // i --> column; v[i] available symbols counter for i-column

  static std::random_device randomDevice;
  static std::mt19937 randomGenerator;

  bool InitDimensionFromFile(std::ifstream &fileIn);
  bool InitGameFieldFromFile(std::ifstream &fileIn);
  void FetchNonFixedSymbolsByColumns();
  void RandomizeGameField();
  std::vector<GameField> GenerateGameFieldChildrenByColumnPermutations(int iCol);
};

#endif // _LAB2_SQUAREWORD_GREEDY_SOLVER_H_
