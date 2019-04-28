#ifndef _LAB1_GAME_FIELD_H_
#define _LAB1_GAME_FIELD_H_

#include <cstdint>  // uint64_t typedef
#include <map>
#include <vector>
#include <ostream>
#include <string>

struct Chip {
  int x;
  int y;
  int width;
  int height;
  char mark;
};

class GameField {
 public:
  GameField(const std::string &jsonConfig) noexcept(false);
  std::vector<GameField> GetPossibleMoves() const;
  std::string Dump() const;

  // TODO: optimise unique id generating
  std::string GetFieldState() const { return fieldState; }

  // TODO: refactoring, bad data linkage
  std::string GetMarkedChipsPattern() const;
  bool IsMarkedChipsOnSameLine() const;

 private:
  int width;
  int height;
  std::string fieldState;

  std::map<uint64_t, Chip> chips;
  std::vector<uint64_t> field;

  static inline const uint64_t ID_UNDEF = 0;
  static inline uint64_t idCounter = ID_UNDEF;

  void ParseJSON(const std::string &jsonConfig);
  void StringifyFieldState();

  bool IsPossibleMove(int xStart, int yStart, int xEnd, int yEnd) const;
  bool IsPossibleMoveLeft(const Chip &chip) const;
  bool IsPossibleMoveRight(const Chip &chip) const;
  bool IsPossibleMoveUp(const Chip &chip) const;
  bool IsPossibleMoveDown(const Chip &chip) const;

  void MoveChipByOffset(uint64_t chipId, int dx, int dy);
  void MoveLeft(uint64_t chipId);
  void MoveRight(uint64_t chipId);
  void MoveUp(uint64_t chipId);
  void MoveDown(uint64_t chipId);

  void FillFieldArea(uint64_t chipId, int xStart, int yStart, int wSize, int hSize);
};

#endif //_LAB1_GAME_FIELD_H_
