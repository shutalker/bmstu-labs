#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include "seabattle.h"


void initBattleField()
{
    int i, j; 
    int unitsAmount = 0; //units: ships, size of field
    int stagesAmount = 0;
    int shipsOnStageAmount = 0;
    int shipLength = 1;
    int shipHealth = 1;
    int shipIdx = 0;

    stagesAmount = MAX_SHIP_LENGTH;

    for(i = 0; i < stagesAmount; i++)
        unitsAmount += (i + 1);

    if(unitsAmount < 10)
    {
        write(2, "Field size is too small\n", 25);
        exit(1);
    }

    if(unitsAmount > FIELD_SIZE_LIMIT)
    {
        write(2, "Field size is too large\n", 25);
        exit(1);
    }

    bfield.fieldSize = unitsAmount;
    bfield.shipsAmount = unitsAmount;
    bfield.battleField = (short *) malloc(sizeof(short) * unitsAmount * unitsAmount);
    memset(bfield.battleField, 0, sizeof(short) * unitsAmount * unitsAmount);
    bfield.ships = (Ship *) malloc(sizeof(Ship) * unitsAmount);
    bfield.cellMarksBuf = (char *) malloc(sizeof(char) * unitsAmount);

    for(i = 0; i < unitsAmount; i++)
        bfield.cellMarksBuf[i] = (char)(i + 97); //97 is a code of 'a' symbol in ASCII

    srand(time(NULL));

    for(i = 0; i < stagesAmount; i++)
    {
        shipsOnStageAmount = stagesAmount - i;

        for(j = 0; j < shipsOnStageAmount; j++)
        {
            bfield.ships[shipIdx].id = shipIdx + 1;
	    bfield.ships[shipIdx].health = shipHealth;
            bfield.ships[shipIdx].length = shipLength;
            bfield.ships[shipIdx].orientation = rand() % 2;
            bfield.ships[shipIdx].position[0] = -1;
            bfield.ships[shipIdx].position[1] = -1;

            shipIdx++;
        }
        
        shipLength++;
        shipHealth++;
    }
}


void destroyBattleField()
{
    bfield.shipsAmount = 0;    

    if(bfield.battleField)
        free(bfield.battleField);

    if(bfield.ships)
        free(bfield.ships);

    if(bfield.cellMarksBuf)
        free(bfield.cellMarksBuf);
}


int checkShipCollision(const int shipIdx)
{
    int rowIdx, colIdx;
    int i = 0;

    rowIdx = bfield.ships[shipIdx].position[1]; //y-row
    colIdx = bfield.ships[shipIdx].position[0]; //x-col

    while(i < bfield.ships[shipIdx].length)
    {
        if(bfield.battleField[colIdx + bfield.fieldSize * rowIdx] != 0)
            return 1;

        if(bfield.ships[shipIdx].orientation == 0)
            colIdx++;
        else
            rowIdx++;
        
        i++;
    }

    return 0;
}


void allocateShipOnField(const int shipIdx)
{
    int rowIdx, colIdx;
    int i = 0;

    rowIdx = bfield.ships[shipIdx].position[1]; //y-row
    colIdx = bfield.ships[shipIdx].position[0]; //x-col

    while(i < bfield.ships[shipIdx].length)
    {
        bfield.battleField[colIdx + bfield.fieldSize * rowIdx] = bfield.ships[shipIdx].id;

        if(bfield.ships[shipIdx].orientation == 0)
            colIdx++;
        else
            rowIdx++;
        
        i++;
    }
}


void restrictFreeFieldSpace(const int shipIdx)
{
    int i, j;
    int areaStartXPos, areaStartYPos;
    int areaEndXPos, areaEndYPos;
    int restrictAreaLength, restrictAreaWidth;

    if(bfield.ships[shipIdx].orientation == 0)
    {
        restrictAreaLength = bfield.ships[shipIdx].length + 2;
        restrictAreaWidth = 3;
    }
    else
    {
        restrictAreaLength = 3;
        restrictAreaWidth = bfield.ships[shipIdx].length + 2;
    }

    areaStartXPos = bfield.ships[shipIdx].position[0] - 1;
    areaStartYPos = bfield.ships[shipIdx].position[1] - 1;
    areaEndXPos = areaStartXPos + restrictAreaLength;
    areaEndYPos = areaStartYPos + restrictAreaWidth;

    for(i = areaStartYPos; i < areaEndYPos; i++)
    {
        if(i < 0 || i > (bfield.fieldSize - 1))
            continue;

        for(j = areaStartXPos; j < areaEndXPos; j++)
        {
            if(j < 0 || j > (bfield.fieldSize - 1))
                continue;

            if(bfield.battleField[j + bfield.fieldSize * i] == bfield.ships[shipIdx].id)
                continue;

            bfield.battleField[j + bfield.fieldSize * i] = -1;
        }
    }
}


void setShipsOnBattleField()
{
    int i;
    int delta, positionOffset, currOrient;

    for(i = 0; i < bfield.shipsAmount; i++)
    {
        do
        {
            currOrient = bfield.ships[i].orientation = rand() % 2;
            bfield.ships[i].position[0] = rand() % bfield.fieldSize;
            bfield.ships[i].position[1] = rand() % bfield.fieldSize;    

            delta = bfield.fieldSize - bfield.ships[i].position[currOrient];
        
            if(delta < bfield.ships[i].length)
            {
                positionOffset = bfield.ships[i].length - delta;
                bfield.ships[i].position[currOrient] -= positionOffset;
            }
        }
        while(checkShipCollision(i) == 1);
        allocateShipOnField(i);
        restrictFreeFieldSpace(i);
    }
}


void writeNum(const int num)
{
    int index = num;
    int digit, count;
    char digitsBuf[3];

    count = 1;

    if(index == 0)
        digitsBuf[1] = '0';
    else
    {
        while(index > 0)
        {
            digit = index % 10;
            digitsBuf[count] = (char)(digit + 48);
            index /= 10;
            count--;
        }
    }

    if(num < 10)
        digitsBuf[0] = ' ';

    digitsBuf[2] = '\0';
    write(1, digitsBuf, 3);
    write(1, " ", 2);
}


void renderBattleField()
{
    int i, j;
    char ch, rowIdx;
    
    write(1, "\x1b[3;H", 6);
    write(1, "   ", 3);    

    for(j = 0; j <= bfield.fieldSize; j++)
    {
        ch = bfield.cellMarksBuf[j];
        write(1, &ch, 1);
        write(1, " ", 2);
    }

    write(1, "\r\n", 3);

    for(i = 0; i < bfield.fieldSize; i++)
    {
        for(j = 0; j <= bfield.fieldSize; j++)
        {
            if(j == 0)
            {
                writeNum(i);
                continue;
            }

            if(bfield.battleField[(j - 1) + bfield.fieldSize * i] > 0)
                ch = '#';//(char)(bfield.battleField[(j - 1) + bfield.fieldSize * i] + 47);
            else if(bfield.battleField[(j - 1) + bfield.fieldSize * i] >= -1)
                ch = '~';
            else if(bfield.battleField[(j - 1) + bfield.fieldSize * i] < -1) //for rendering of damaged parts of ships
                ch = 'x';
            
            write(1, &ch, 1);
            write(1, " ", 2);
        }

        write(1, "\r\n", 3);
    }

    write(1, "\r\n", 3);
    write(1, "Current ships amount: ", 23);
    writeNum(bfield.shipsAmount);
    write(1, "\r\n", 3);
}


int checkCell(char *cell)
{
    int i;
    char buf[10] = "0123456789";
    short successFlag = 0;

    for(i = 0; i < bfield.fieldSize; i++)
    {
        if(cell[0] == bfield.cellMarksBuf[i])
        {
            successFlag = 1;
            break;
        }
    }

    if(successFlag)
    {
        for(i = 0; i < bfield.fieldSize; i++)
        {
            if(cell[1] == buf[i])
                return 0;
        }
    }

    write(1, "\x1b[17;H", 7);
    write(1, "You choose a wrong cell!", 25);
    write(1, "\r\n", 3);
    return 1;
}


short getDamage(char *cell)
{
    int xPos, yPos;
    int shipIdx;
    char ch;

    xPos = (int)(cell[0]) - 97;
    yPos = (int)(cell[1]) - 48;

    if((shipIdx = bfield.battleField[xPos + bfield.fieldSize * yPos]) > 0)
    {
	shipIdx -= 1;
        bfield.battleField[xPos + bfield.fieldSize * yPos] = -2;
        bfield.ships[shipIdx].health -= 1;
        
        write(1, "\x1b[20;H", 7);
        ch = (char)(bfield.ships[shipIdx].health + 48);
        write(1, &ch, 1);

        if(bfield.ships[shipIdx].health < 1)
        {
            bfield.shipsAmount--;
            return 2;
        }
        
        return 1;
    }

    return 0;   
}
