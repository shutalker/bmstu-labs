#include "game_field.h"
#include <iostream>
#include <iomanip>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <experimental/filesystem>

GameField::GameField(const std::string &jsonConfig) {
  namespace fs = std::experimental::filesystem;

  if (!fs::exists(jsonConfig) || !fs::is_regular_file(jsonConfig))
    throw std::runtime_error("GameField::GameField --> bad file: " + jsonConfig);

  ParseJSON(jsonConfig);
  StringifyFieldState();
}

std::vector<GameField> GameField::GetPossibleMoves() const {
  std::vector<GameField> moves;

  for (auto &chipNode: chips) {
    uint64_t chipId = chipNode.first;
    const Chip &chip = chipNode.second;

    if (IsPossibleMoveLeft(chip)) {
      moves.emplace_back(*this);
      moves.back().MoveLeft(chipId);
    }

    if (IsPossibleMoveRight(chip)) {
      moves.emplace_back(*this);
      moves.back().MoveRight(chipId);
    }

    if (IsPossibleMoveUp(chip)) {
      moves.emplace_back(*this);
      moves.back().MoveUp(chipId);
    }

    if (IsPossibleMoveDown(chip)) {
      moves.emplace_back(*this);
      moves.back().MoveDown(chipId);
    }
  }

  return moves;
}

std::string GameField::Dump() const {
  std::stringstream output;

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int iCell = x + width * y;
      uint64_t chipId = field[iCell];
      char mark = '.';

      if (chipId != ID_UNDEF) {
        char chipName = chips.find(chipId)->second.mark;
        mark = (chipName) ? chipName : '=';
      }

      output << mark << ' ';
    }

    output << '\n';
  }

  return output.str();
}

std::string GameField::GetMarkedChipsPattern() const {
  std::string pattern;

  for (int x = 0; x < width; ++x) {
    for (int y = 0; y < height; ++y) {
      int iCell = x + y * width;
      uint64_t chipId = field[iCell];

      if (chipId == ID_UNDEF)
        continue;

      auto itChip = chips.find(chipId);

      if (itChip == chips.end()) {
        throw std::runtime_error("GameField::GetMarkedChipsPattern --> untracked chip id: "
            + std::to_string(chipId));
      }

      if (itChip->second.mark) {
        pattern += itChip->second.mark;
        break;
      }
    }
  }

  return pattern;
}

bool GameField::IsMarkedChipsOnSameLine() const {
  int yPrev = -1;

  for (auto itChip = chips.begin(); itChip != chips.end(); ++itChip) {
    if (itChip->second.mark != 0) {
      if (yPrev >= 0 && itChip->second.y != yPrev)
        return false;

      yPrev = itChip->second.y;
    }
  }

  return true;
}

void GameField::ParseJSON(const std::string &jsonConfig) {
  using namespace boost::property_tree;

  ptree jsonTree;
  read_json(jsonConfig, jsonTree);

  // assume that all values in json is correct
  // todo: value check (handle negative values)
  width = jsonTree.get<int>("field.width");
  height = jsonTree.get<int>("field.height");
  field.resize(width * height, ID_UNDEF);

  for (ptree::value_type &node: jsonTree.get_child("chips")) {
    idCounter += 1; // 0 means undefined identificator
    Chip chip;
    chip.x = node.second.get<int>("x");
    chip.y = node.second.get<int>("y");
    chip.width  = node.second.get<int>("width");
    chip.height = node.second.get<int>("height");

    std::string name = node.second.get<std::string>("name", "");
    chip.mark = name.empty() ? 0 : name[0];

    for (int xOffset = 0; xOffset < chip.width; ++xOffset) {
      for (int yOffset = 0; yOffset < chip.height; ++yOffset) {
        int iCell = (chip.x + xOffset) + width * (chip.y + yOffset);
        field[iCell] = idCounter;
      }
    }

    chips.try_emplace(idCounter, std::move(chip));
  }
}

void GameField::StringifyFieldState() {
  fieldState.clear();

  for (int iCell = 0; iCell < field.size(); ++iCell) {
    char mark = '.';
    uint64_t chipId = field[iCell];

    if (chipId != ID_UNDEF) {
      auto itChip = chips.find(chipId);

      if (itChip == chips.end()) {
        throw std::runtime_error("GameField::StringifyFieldState --> untracked chip id "
            + std::to_string(chipId));
      }

      mark = (itChip->second.mark) ? itChip->second.mark : '=';
    }

    fieldState += mark;
  }
}

bool GameField::IsPossibleMove(int xStart, int yStart, int xDist, int yDist) const {
  for (int xOffset = 0; xOffset < xDist; ++xOffset) {
    for (int yOffset = 0; yOffset < yDist; ++yOffset) {
      int x = xStart + xOffset;
      int y = yStart + yOffset;

      if (x < 0 || x >= width)
        return false;

      if (y < 0 || y >= height)
        return false;

      int iCell = x + y * width;

      if (field[iCell] != ID_UNDEF)
        return false;
    }
  }

  return true;
}

bool GameField::IsPossibleMoveLeft(const Chip &chip) const {
  int xStart = chip.x - 1;
  int yStart = chip.y;
  int xDist  = 1;
  int yDist  = chip.height;

  return IsPossibleMove(xStart, yStart, xDist, yDist);
}

bool GameField::IsPossibleMoveRight(const Chip &chip) const {
  int xStart = chip.x + chip.width;
  int yStart = chip.y;
  int xDist  = 1;
  int yDist  = chip.height;

  return IsPossibleMove(xStart, yStart, xDist, yDist);
}

bool GameField::IsPossibleMoveUp(const Chip &chip) const {
  int xStart = chip.x;
  int yStart = chip.y - 1;
  int xDist  = chip.width;
  int yDist  = 1;

  return IsPossibleMove(xStart, yStart, xDist, yDist);
}

bool GameField::IsPossibleMoveDown(const Chip &chip) const {
  int xStart = chip.x;
  int yStart = chip.y + chip.height;
  int xDist  = chip.width;
  int yDist  = 1;

  return IsPossibleMove(xStart, yStart, xDist, yDist);
}

void GameField::MoveChipByOffset(uint64_t chipId, int dx, int dy) {
  Chip &chip = chips[chipId];

  // clear old area occupated by chip
  FillFieldArea(ID_UNDEF, chip.x, chip.y, chip.width, chip.height);
  chip.x += dx;
  chip.y += dy;

  // replace chip on field
  FillFieldArea(chipId, chip.x, chip.y, chip.width, chip.height);
  StringifyFieldState();
}

void GameField::MoveLeft(uint64_t chipId) {
  int dx = -1;
  int dy = 0;

  MoveChipByOffset(chipId, dx, dy);
}

void GameField::MoveRight(uint64_t chipId) {
  int dx = 1;
  int dy = 0;

  MoveChipByOffset(chipId, dx, dy);
}

void GameField::MoveUp(uint64_t chipId) {
  int dx = 0;
  int dy = -1;

  MoveChipByOffset(chipId, dx, dy);
}

void GameField::MoveDown(uint64_t chipId) {
  int dx = 0;
  int dy = 1;

  MoveChipByOffset(chipId, dx, dy);
}

void GameField::FillFieldArea(uint64_t chipId, int xStart, int yStart,
    int wSize, int hSize) {
  for (int xOffset = 0; xOffset < wSize; ++xOffset) {
    for (int yOffset = 0; yOffset < hSize; ++yOffset) {
      int iCell = (xStart + xOffset) + (yStart + yOffset) * width;
      field[iCell] = chipId;
    }
  }
}
