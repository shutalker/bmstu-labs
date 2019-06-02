#include <algorithm>
#include <iostream>
#include <cstring>
#include <unordered_set>
#include "squareword_greedy_solver.h"

std::random_device SquarewordGreedySolver::randomDevice;
std::mt19937 SquarewordGreedySolver::randomGenerator(randomDevice());

std::string& LTrim(std::string& str, const std::string& chars = "\t\n\v\f\r ") {
    str.erase(0, str.find_first_not_of(chars));
    return str;
}

std::string& RTrim(std::string& str, const std::string& chars = "\t\n\v\f\r ") {
    str.erase(str.find_last_not_of(chars) + 1);
    return str;
}

std::string& Trim(std::string& str, const std::string& chars = "\t\n\v\f\r ") {
    return LTrim(RTrim(str, chars), chars);
}

void SquarewordGreedySolver::GameField::Dump(std::ostream &output) const {
  for (int iRow = 0; iRow < dimension; ++iRow) {
    for (int iCol = 0; iCol < dimension; ++iCol)
      output << cells[iCol + iRow * dimension].first << " ";

    output << std::endl;
  }

  output << "f = " << duplicatedSymbols << std::endl;
}

void SquarewordGreedySolver::GameField::RecountDuplicatedSymbols() {
  duplicatedSymbols = 0;
  duplicatedSymbols += RecountDuplicatedSymbolsByRows();
  duplicatedSymbols += RecountDuplicatedSymbolsByCols();
  duplicatedSymbols += RecountDuplicatedSymbolsByDiags();
}

int SquarewordGreedySolver::GameField::RecountDuplicatedSymbolsByRows() {
  int duplicates = 0;

  for (int iRow = 0; iRow < dimension; ++iRow) {
    std::unordered_set<char> syms;

    for (int iCol = 0; iCol < dimension; ++iCol) {
      int iCell = iCol + iRow * dimension;

      if (syms.find(cells[iCell].first) != syms.end()) {
        duplicates += 1;
        continue;
      }

      syms.emplace(cells[iCell].first);
    }
  }

  return duplicates;
}

int SquarewordGreedySolver::GameField::RecountDuplicatedSymbolsByCols() {
  int duplicates = 0;

  for (int iCol = 0; iCol < dimension; ++iCol) {
    std::unordered_set<char> syms;

    for (int iRow = 0; iRow < dimension; ++iRow) {
      int iCell = iCol + iRow * dimension;

      if (syms.find(cells[iCell].first) != syms.end()) {
        duplicates += 1;
        continue;
      }

      syms.emplace(cells[iCell].first);
    }
  }

  return duplicates;
}

int SquarewordGreedySolver::GameField::RecountDuplicatedSymbolsByDiags() {
  int duplicates = 0;
  std::unordered_set<char> symsMainDiag;
  std::unordered_set<char> symsExtraDiag;

  for (int iDiag = 0; iDiag < dimension; ++ iDiag) {
    int iCellMainDiag = iDiag + iDiag * dimension;
    int iCellExtraDiag = (dimension - iDiag - 1) + iDiag * dimension;

    if (symsMainDiag.find(cells[iCellMainDiag].first) != symsMainDiag.end())
      duplicates += 1;
    else
      symsMainDiag.emplace(cells[iCellMainDiag].first);

    if (symsExtraDiag.find(cells[iCellExtraDiag].first) != symsExtraDiag.end())
      duplicates += 1;
    else
      symsExtraDiag.emplace(cells[iCellExtraDiag].first);
  }

  return duplicates;
}

bool SquarewordGreedySolver::InitFromFile(const std::string &filename) {
  std::ifstream fileIn(filename);

  if (!fileIn) {
    std::cerr << "SquarewordGreedySolver::Init --> failed to open file:"
        << filename << "; error: " << strerror(errno) << std::endl;

    return false;
  }

  if (!InitDimensionFromFile(fileIn)) return false;
  if (!InitGameFieldFromFile(fileIn)) return false;
  FetchNonFixedSymbolsByColumns();
  return true;
}

void SquarewordGreedySolver::InitFromGameField(const GameField &gf) {
  currField = gf;
  alphabet.clear();

  for (const auto &c: currField.cells) {
    if (!c.second) continue;
    alphabet.emplace(c.first);
  }

  FetchNonFixedSymbolsByColumns();
}

bool SquarewordGreedySolver::RunGreedySearch(bool randomizeGameField,
    std::vector<GameField> *terminalFields) {
  if (randomizeGameField) RandomizeGameField();
  currField.RecountDuplicatedSymbols();
  std::vector<GameField> children;
  bool isSolutionFound = false;

  for (int iCol = 0; iCol < currField.dimension; ++iCol) {
    std::cout << "iCol = " << iCol << std::endl;
    currField.Dump(std::cout);
    std::cout << std::endl;
    if (currField.duplicatedSymbols == 0) {
      isSolutionFound = true;
      break;
    }

    children = std::move(GenerateGameFieldChildrenByColumnPermutations(iCol));
    if (children.empty()) return false;
    std::sort(children.begin(), children.end(), GameField::DuplicatedSymdolsComparator);
    std::cout << "children.size() = " << children.size() << std::endl;
    currField = children[0];
  }

  isSolutionFound = currField.duplicatedSymbols == 0;
  if (isSolutionFound)
    terminalFields->emplace_back(currField);
  else
    *terminalFields = std::move(children);

  return isSolutionFound;
}

void SquarewordGreedySolver::DumpAlphabet(std::ostream &output) const {
  for (const auto &sym: alphabet) {
    output << sym << " ";
  }

  output << std::endl;
}

void SquarewordGreedySolver::DumpNonFixedSymbols(std::ostream &output) const {
  for (int iCol = 0; iCol < nonFixedSymbols.size(); ++iCol) {
    std::cout << iCol << "-col: ";

    for (const auto &sym: nonFixedSymbols[iCol]) {
      std::cout << sym << "  ";
    }

    std::cout << std::endl;
  }
}

bool SquarewordGreedySolver::InitDimensionFromFile(std::ifstream &fileIn) {
  std::string dim;

  if (!std::getline(fileIn, dim)) {
    std::cerr << "SquarewordGreedySolver::Init --> failed to read dimension value: "
        << strerror(errno) << std::endl;

    return false;
  }

  currField.dimension = atoi(Trim(dim).c_str());
  std::cout << "SquarewordGreedySolver::Init --> dimension = " << currField.dimension << std::endl;
  return true;
}

bool SquarewordGreedySolver::InitGameFieldFromFile(std::ifstream &fileIn) {
  currField.cells.clear();
  currField.cells.reserve(currField.dimension * currField.dimension);
  int lineCount = 0;

  while (!fileIn.eof() && !fileIn.fail()) {
    std::string line;
    int iRow = lineCount;

    if (!std::getline(fileIn, line) && !fileIn.eof()) {
      std::cerr << "SquarewordGreedySolver::Init --> failed to read line: "
          << strerror(errno) << std::endl;

      return false;
    }

    Trim(line);
    if (line.empty()) continue;

    if (++lineCount > currField.dimension) {
      std::cerr << "SquarewordGreedySolver::Init --> line count limit exceeded: "
          << lineCount << "; need to be " << currField.dimension << std::endl;

      return false;
    }

    if (line.size() != currField.dimension) {
      std::cerr << "SquarewordGreedySolver::Init --> invalid line length: "
          << line.size() << "; need to be " << currField.dimension << std::endl;

      return false;
    }

    for (int iCol = 0; iCol < line.size(); ++iCol) {
      const char &c = line[iCol];
      currField.cells.emplace_back(c, c != '.');
      if (c == '.') continue;
      // вставка символов происходит с упорядочиванием кодов по возрастанию
      alphabet.emplace(c);
    }
  }

  return true;
}

void SquarewordGreedySolver::FetchNonFixedSymbolsByColumns() {
  nonFixedSymbols.clear();
  availableSymbolCount.clear();
  nonFixedSymbols.resize(currField.dimension);
  availableSymbolCount.resize(currField.dimension, 0);

  for (int iCol = 0; iCol < currField.dimension; ++iCol) {
    std::set<char> syms = alphabet;

    for (int iRow = 0; iRow < currField.dimension; ++iRow) {
      int iCell = iCol + iRow * currField.dimension;

      if (!currField.cells[iCell].second) {
        availableSymbolCount[iCol] += 1;
        continue;
      }

      syms.erase(currField.cells[iCell].first);
    }

    std::vector<char> &nonFixedSyms = nonFixedSymbols[iCol];
    nonFixedSyms.insert(nonFixedSyms.end(), syms.begin(), syms.end());
  }
}

void SquarewordGreedySolver::RandomizeGameField() {
  std::vector<char> syms(alphabet.begin(), alphabet.end());
  std::uniform_int_distribution<> lettersDistribution(0, syms.size() - 1);

  for (int iCell = 0; iCell < currField.cells.size(); ++iCell) {
    if (currField.cells[iCell].second) continue;
    currField.cells[iCell].first = syms[lettersDistribution(randomGenerator)];
  }
}

std::vector<SquarewordGreedySolver::GameField>
SquarewordGreedySolver::GenerateGameFieldChildrenByColumnPermutations(int iCol) {
  // символы упорядочены по возрастанию кодов
  const std::vector<char> &nonFixedSyms = nonFixedSymbols[iCol];
  int availSyms = availableSymbolCount[iCol];
  int totalSyms = nonFixedSyms.size();

  if (availSyms > totalSyms)
    return std::vector<GameField>();

  std::vector<GameField> currFieldChildren;
  std::vector<char> availSymbols(availSyms);
  std::vector<bool> bitmask(availSyms, true);
  bitmask.resize(totalSyms, false);

  do { // генерация сочетаний (не учитываем порядок букв)
    int iAvailSym = 0;

    // сохранение конкретного сочетания из total по avail
    // символы упорядочены по возрастанию кодов
    for (int iSym = 0; iSym < totalSyms; ++iSym) {
      if (bitmask[iSym])
        availSymbols[iAvailSym++] = nonFixedSyms[iSym];
    }

    do {
      GameField gf = currField;
      int iSym = 0;

      for (int iRow = 0; iRow < gf.dimension; ++iRow) {
        int iCell = iCol + iRow * gf.dimension;
        if (gf.cells[iCell].second) continue;
        gf.cells[iCell].first = availSymbols[iSym++];
      }

      gf.RecountDuplicatedSymbols();
      currFieldChildren.emplace_back(std::move(gf));
    } while (std::next_permutation(availSymbols.begin(), availSymbols.end()));
  } while (std::prev_permutation(bitmask.begin(), bitmask.end()));

  return currFieldChildren;
}
