#ifndef ELEVATOR_H
#define ELEVATOR_H

#define FLOORS 10
#define PIT_LENGTH (FLOORS + 4)

typedef struct
{
    char   elvtr_name;              //'P' is for passenger, 'C' is for cargo
    char   elvtr_pit[PIT_LENGTH];   //"PL: ----------", "CL: ----------" 
    short  floor_mask[FLOORS];      //[0,1] array: 0 - pass floor by, 1 - stop at floor
    char   floor_list[FLOORS];      //"0123456789"
    int    current_floor;           //index in *floor_list (0-9)
    short  motion_direction;        //1 - up, -1 - down, 0 - idle
    int    passengers_amount;
    int    waiting_passengers_amount;
    short  waiting_floor_mask[FLOORS];
    int    elvtr_speed;
} Elevator;

#endif
