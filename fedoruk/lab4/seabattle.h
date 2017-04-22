#ifndef SEABATTLE_H
#define SEABATTLE_H

#define MAX_SHIP_LENGTH 4
#define FIELD_SIZE_LIMIT 26


typedef struct
{
    int id;
    int health;
    int position[2]; //position[0] = x, position[1] = y, it's position of first ship cell (left-top cell for allocation in battle field)
    int orientation; //0 - horisontal, 1 - vertical
    int length;      
} Ship;

typedef struct
{
    short *battleField;
    int    fieldSize;
    int    shipsAmount;
    char  *cellMarksBuf;
    Ship  *ships; 
} Battlefield;

Battlefield bfield;

void initBattleField();
void destroyBattleField();
int  checkShipCollision(const int);
void allocateShipOnField(const int);
void restrictFreeFieldSpace(const int);
void setShipsOnBattleField();
void writeNum(const int);
void renderBattleField();
int checkCell(char *);
short getDamage(char *);

#endif
