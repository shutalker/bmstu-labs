#include <stdio.h>
#include <error.h>
#include <stdlib.h>
#include <string.h>
#include <sys/select.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <signal.h>
#include <unistd.h>
#include <termios.h>
#include <fcntl.h>
#include <math.h>
#include "elevator.h"

#define PFD_READ 0
#define PFD_WRITE 1
#define PIPE_1 0
#define PIPE_2 1
#define PIPE_3 2
#define PIPE_4 3
#define MAX(a,b) (((a)>(b))?(a):(b))

/*
    pipe 1: parent --->(writes) pipe --->(reads) child_1
    pipe 2: parent --->(writes) pipe --->(reads) child_2
    pipe 3: parent <---(reads) pipe <---(writes) child_1
    pipe 4: parent <---(reads) pipe <---(writes) child_2
*/

static struct termios prev_tty;
static pid_t  child_pid[2];
static int    pipe_fds[4][2];

void restore_tty(int sig)
{
    write(1, "\x1b[2J", 4);               //clear screen
    write(1, "\x1b[;H", 4);               //set the cursor at left-top corner
    tcsetattr(0, TCSAFLUSH, &prev_tty);   //restoring canonical input
    kill(child_pid[0], SIGINT);
    kill(child_pid[1], SIGINT);
    printf("Killed by signal %d\n", sig);
    exit(0);
}


void request_lift_locations(){}


void set_restore_tty_handler(struct sigaction *act)
{
    sigset_t set;
    
    sigemptyset(&set);
    sigaddset(&set, SIGTERM);
    sigaddset(&set, SIGINT);
    sigaddset(&set, SIGSEGV);
    act->sa_handler = restore_tty;
    act->sa_mask = set;
    sigaction(SIGTERM, act, 0);
    sigaction(SIGINT, act, 0);
    sigaction(SIGSEGV, act, 0);
}


void set_request_lift_locations(struct sigaction *act)
{
    sigset_t set;
    
    sigemptyset(&set);
    sigaddset(&set, SIGUSR1);
    act->sa_handler = request_lift_locations;
    act->sa_mask = set;
}


void set_non_canonical_input(struct termios *cur_tty)
{
    tcgetattr(0, cur_tty);
    prev_tty = *cur_tty;               
    cur_tty->c_lflag = ~(ICANON | ECHO | ISIG); //making non-canonical input
    cur_tty->c_cc[VMIN] = 1;
    tcsetattr(0, TCSAFLUSH, cur_tty);
}


void init_elevator(Elevator *elvtr, char name, int speed)
{
    int i;

    elvtr->elvtr_name = name;
    elvtr->elvtr_pit[0] = name;
    elvtr->elvtr_pit[1] = 'L';
    elvtr->elvtr_pit[2] = ':';
    elvtr->elvtr_pit[3] = ' ';
    
    for(i = 4; i < PIT_LENGTH; i++)  //lenth of elvtr_pit = 14
        elvtr->elvtr_pit[i] = '-';

    for(i = 0; i < FLOORS; i++)
    {
        elvtr->floor_list[i] = (char) (48 + i); //48 is the code of '0'
        elvtr->floor_mask[i] = 0;
        elvtr->waiting_floor_mask[i] = 0;
    }

    elvtr->current_floor = 0;
    elvtr->motion_direction = 0;
    elvtr->passengers_amount = 0;
    elvtr->waiting_passengers_amount = 0;
    elvtr->elvtr_speed = speed; 
}


void render_elevator(const Elevator *elvtr)
{  
    char esc[5];
    char elevator_position;
    char passengers;
    int floor_idx;
    int esc_length;
    
    passengers = (char)(elvtr->passengers_amount + 48);
    floor_idx = elvtr->current_floor;
    elevator_position = (char)((int)(elvtr->floor_list[floor_idx]) + 1);
    esc[0] = 27;
    esc[1] = '[';

    if(elevator_position > '9') //esc-sequence for "10" symbol that does not exist
    {
        esc[2] = '1';
        esc[3] = '0';
        esc[4] = 'C';
        esc_length = 5;
    }
    else
    {
        esc[2] = elevator_position;
        esc[3] = 'C';
        esc[4] = 0;
        esc_length = 4;
    }

    usleep(30);
    write(1, "\x1b[;H", 4);

    switch(elvtr->elvtr_name)
    {
        case 'P': write(1, "\x1b[2;H", 5);  
                  break;

        case 'C': write(1, "\x1b[3;H", 5);
                  break;
    }

    write(1, &elvtr->elvtr_pit, PIT_LENGTH);
    write(1, "\x1b[4G", 4);
    write(1, &esc, esc_length);
    write(1, &passengers, 1);        
}


int check_floors_to_go(const short *floors)
{
    int i;
    int dest_floor_amount = 0;

    for(i = 0; i < FLOORS; i++)
    {
        if(floors[i] == 1)
            dest_floor_amount++;
    }
    
    return dest_floor_amount;
}


int search_nearest_floor_to_go(short *floors, int curr_floor)
{
    int i, j, nearest_floor;
    int up_search_limit, down_search_limit;
    int up_floor_idx, down_floor_idx;

    up_search_limit = (FLOORS - 1) - curr_floor;
    down_search_limit = curr_floor; //curr_floor - 0 

    for(i = 0, j = 0; i < up_search_limit || j < down_search_limit; i++, j++)
    {
        up_floor_idx = curr_floor + i + 1;
        down_floor_idx = curr_floor - j - 1;
        
        if((up_floor_idx < FLOORS) && (floors[up_floor_idx] == 1))
            nearest_floor = up_floor_idx;
        else if((down_floor_idx >= 0) && (floors[down_floor_idx] == 1))
            nearest_floor = down_floor_idx;
    }

    return  nearest_floor;
}


short exists_unvisited_floors_in_direction(const Elevator *elvtr)
{
    int i;
    int search_limit;
    short unvisited_floors_flag = 0;

    if(elvtr->motion_direction == 1)
    {
        search_limit = FLOORS;
        i = elvtr->current_floor;
    }
    else //elvtr->motion_direction == -1
    {
        search_limit = elvtr->current_floor;
        i = 0;
    }

    for(i; i < search_limit; i++)
        if(elvtr->floor_mask[i] == 1)
        {
            unvisited_floors_flag = 1;
            break;
        }

    return unvisited_floors_flag;
}


void move_elevator(Elevator *elvtr)
{
    int floor_to_go = -1;
    int wait_flag = 0;        

    if(check_floors_to_go(elvtr->floor_mask) == 0)
    {
        if(elvtr->motion_direction != 0)
            elvtr->motion_direction = 0;

        return;
    }
    else if(elvtr->motion_direction == 0)  //if the elevator was idle before the calling
    {
        floor_to_go = search_nearest_floor_to_go(elvtr->floor_mask, elvtr->current_floor);

        if(floor_to_go == elvtr->current_floor)
        {
            if(elvtr->passengers_amount > 0)
            {
                elvtr->passengers_amount--;
                render_elevator(elvtr);
            }    
            return;
        }

        elvtr->motion_direction = (floor_to_go > elvtr->current_floor)? 1 : -1;
    }
    else
    {
        if(!exists_unvisited_floors_in_direction(elvtr))
            elvtr->motion_direction = -(elvtr->motion_direction);
    }
    
    elvtr->current_floor += elvtr->motion_direction; //lift up or down to one floor
    sleep(elvtr->elvtr_speed);
    render_elevator(elvtr);

    if(elvtr->floor_mask[elvtr->current_floor] == 1)
    {
        elvtr->floor_mask[elvtr->current_floor] = 0;

        if(elvtr->waiting_floor_mask[elvtr->current_floor] == 1)
        {
            elvtr->waiting_floor_mask[elvtr->current_floor] = 0;
            elvtr->passengers_amount++;
            if(elvtr->waiting_passengers_amount > 0)
                elvtr->waiting_passengers_amount--;
        }
        else if(elvtr->passengers_amount > 0)
            elvtr->passengers_amount--;

        sleep(1);
        render_elevator(elvtr);
    }

}


void run_elevator(Elevator *elvtr)
{
    //set_act_lift_button();
    char call_mode;     //'.' - lift button was pressed, '=' - floor button was pressed
    char dest_floor;
    char elevator_motion_direction;
    int  floor_idx;
    int  pipe_idx;      //in which pipe will write the process
    fd_set rfds; 
    struct timeval tv = {0, 0};

    switch(elvtr->elvtr_name)
    {
        case 'P': pipe_idx = PIPE_3;  
                  break;

        case 'C': pipe_idx = PIPE_4;
                  break;
    }
    render_elevator(elvtr);    

    while(1)
    {  
        FD_ZERO(&rfds);
        FD_SET(0, &rfds);  // 0 - stdin fd
    
        if(select(1, &rfds, NULL, NULL, &tv) > 0)
        {
            if(FD_ISSET(0, &rfds))
            {
                read(0, &call_mode, 1);
                if(call_mode == '.')
                {   
                    read(0, &dest_floor, 1);
                    floor_idx = (int)(dest_floor) - 48; //48 - '0' code
                    if(elvtr->motion_direction == 0)
                    {   //lift is idle and someone come in
                        elvtr->floor_mask[floor_idx] = 1;
                        if(elvtr->passengers_amount == 0)
                        {
                            elvtr->passengers_amount++;
                            render_elevator(elvtr);
                        }
                    }
                    else if(elvtr->motion_direction != 0 && elvtr->passengers_amount > 0)
                        elvtr->floor_mask[floor_idx] = 1;
                }
                else if(call_mode == '?')
                {//request for elevator state from control block
                    write(pipe_fds[pipe_idx][PFD_WRITE], &elvtr->floor_list[elvtr->current_floor], 1);
                    switch(elvtr->motion_direction)
                    {
                        case 0: elevator_motion_direction = '0';
                                break;

                        case 1: elevator_motion_direction = '+';
                                break;
        
                        case -1: elevator_motion_direction = '-';
                                break;
                    }
                    write(pipe_fds[pipe_idx][PFD_WRITE], &elevator_motion_direction, 1);
                    kill(getppid(), SIGUSR1);
                }
                else if(call_mode == '=')
                {//call from the floor
                    read(0, &dest_floor, 1);
                    floor_idx = (int)(dest_floor) - 48;
                    elvtr->floor_mask[floor_idx] = 1;
                    elvtr->waiting_floor_mask[floor_idx] = 1;
                    elvtr->waiting_passengers_amount++;
                }
            }
        }

        move_elevator(elvtr);    
    } 
}


void recode_floor_button(char *btn)
{
    if(btn[0] > '0')
        btn[0] = (char)((int)(btn[0]) - 1);
    else
        btn[0] = '9';
}


short is_dest_floor_on_road(int curr_floor, int dest_floor, short motion_direction)
{
    int relative_floor_position; // = dest - curr; it can be < 0 or >= 0 
                                 //if < 0 then dest under curr and elevator should go down to be on one road with dest
                                 //if > 0 then dest above curr and elevator should go up to be on one road with dest
    relative_floor_position =  dest_floor - curr_floor;

    if((relative_floor_position * motion_direction) > 0)
        return 1;
    else
        return 0;
}


short choose_nearest_lift(char lift_state[][2], const char *dest)
{
    /*
      lift_state[i] - state of i-elevator
                 |
                 0 index is for passenger elevator
                 1 index is for cargo elevator
      lift_state[i][0] - current floor of i-elevator
      lift_state[i][1] - current direction of motion of i-elevator  
    */
    int     i;
    int     dest_floor;
    int     distance_to_dest[2];
    int     curr_floor[2];
    short   motion_direction[2];

    dest_floor = (int)(*dest) - 48;
    
    for(i = 0; i < 2; i++)
    {
        curr_floor[i] = (int)(lift_state[i][0]) - 48;
        distance_to_dest[i] = abs(curr_floor[i] - dest_floor);

        switch(lift_state[i][1])
        {
            case '0': motion_direction[i] = 0;
                      break;

            case '+': motion_direction[i] = 1;
                      break;

            case '-': motion_direction[i] = -1;
                      break;
        }
    }

    if((motion_direction[0] == 0) && (motion_direction[1] == 0))
    { //if both elevators are idle

        if(distance_to_dest[0] <= 2 * distance_to_dest[1])
            return 0;
        else
            return 1;
    }

    if((motion_direction[0] == 0) && (motion_direction[1] != 0))
    {//if cargo elevator is in motion
        //define the direction of motion of cargo elevator
        //if elevator goes to dest_floor then compare the distances

        if(is_dest_floor_on_road(curr_floor[1], dest_floor, motion_direction[1]))
            if(distance_to_dest[1] <= (distance_to_dest[0] / 2))
                return 1;
        else
            return 0;
    }

    if((motion_direction[0] != 0) && (motion_direction[1] == 0))
    {//if passenger elevator is in motion

        if(is_dest_floor_on_road(curr_floor[0], dest_floor, motion_direction[0]))
        {   
            if(distance_to_dest[0] <= (distance_to_dest[1] * 2)) 
                return 0;
        }
        else
            return 1;
    }

    if((motion_direction[0] != 0) && (motion_direction[1] != 0))
    {

        if(is_dest_floor_on_road(curr_floor[0], dest_floor, motion_direction[0]))
        {
            if(is_dest_floor_on_road(curr_floor[1], dest_floor, motion_direction[1]))
            {
                if(distance_to_dest[0] <= (distance_to_dest[1] * 2))
                    return 0;
                else
                    return 1;
            }
            else
                return 0;
        }
        else if(is_dest_floor_on_road(curr_floor[1], dest_floor, motion_direction[1]))
        {
            if(is_dest_floor_on_road(curr_floor[0], dest_floor, motion_direction[0]))
            {
                if(distance_to_dest[1] <= (distance_to_dest[0] / 2))
                    return 1;
                else
                    return 0;
            }
            else
                return 1;
        }
        else
            return 0;
    }

    return 0;
}


char is_lift_button_pressed(char *buttons, char *pressed_btn)
{
    int i;
    char floor_btn = '\0';

    for(i = 0; i < FLOORS; i++)
    {
        if(buttons[i] == *pressed_btn)
        {
            floor_btn = (char) (48 + i);
            break;
        }
    }
    
    return floor_btn;
}


int main()
{
    int     i, proc_counter = 0;
    char    *lifts_buttons[] = {
                                 "qwertyuiop",
                                 "asdfghjkl;"
                               };
    char    btn;
    char    lift_name[2] = {'P', 'C'};
    int     lift_speed[2] = {1, 2};
    char    lift_state[2][2];        //state of [i] elevator: current floor [0], direction of motion [1]
    char    pressed_lift_btn = '\0';
    short   nearest_lift_idx; 

    struct sigaction act_restore_tty;
    struct sigaction act_request_lift_locations;
    struct termios current_tty;
    sigset_t set;

    fd_set rfds; 
    struct timeval tv = {0, 0};

    Elevator lift[2]; 

    if(!isatty(0))
    {
        write(2, "stdin is not a terminal", 23);
        write(2, "\r\n", 2);
        exit(1);
    }

	for(i = 0; i < 2; i++, proc_counter++)    //creating child processes
	{
        pipe(pipe_fds[i]);
        pipe(pipe_fds[i+2]);

		if((child_pid[i]=fork()) == 0)        //child process
		{
            dup2(pipe_fds[i][PFD_READ], 0);   //assign child's stdin with pipe 1 (2)
            close(pipe_fds[i][PFD_WRITE]);    //close i-child on write in pipe 1 (2)
            close(pipe_fds[i+2][PFD_READ]);   //close i-child on read in pipe 3   (4)
			init_elevator(&lift[i], lift_name[i], lift_speed[i]);
            run_elevator(&lift[i]);
			exit(0);
		}
		else if (child_pid[i] == -1)
			perror("Fork");
        else                                 //parent process
        {
            close(pipe_fds[i][PFD_READ]);    //close parent on read in pipe 1 (2)
            close(pipe_fds[i+2][PFD_WRITE]); //close parent on write in pipe 3 (4)
        }
	}

    memset(&act_restore_tty, 0, sizeof(act_restore_tty));
    set_restore_tty_handler(&act_restore_tty);
    set_request_lift_locations(&act_request_lift_locations);
    sigemptyset(&set);

    set_non_canonical_input(&current_tty);

    write(1, "\x1b[?25l", 6);
    write(1, "\x1b[2J", 4);
    write(1, "\x1b[;H", 4);

    while(1)
    {
        read(0, &btn, 1);
        if(btn == 'z')
            break;


        if((btn >= '0') && (btn <= '9')) //control block buttons processing
        { //47 is code of '/' (next - '0') and 58 is code of ':' (previous - '9')
            recode_floor_button(&btn);

            FD_ZERO(&rfds);
            //FD_SET(pipe_fds[PIPE_3][PFD_READ], &rfds);
            //FD_SET(pipe_fds[PIPE_4][PFD_READ], &rfds);            

            for(i = 0; i < proc_counter; i++)
            {
                sigaction(SIGUSR1, &act_request_lift_locations, 0);
                write(pipe_fds[i][PFD_WRITE], "?", 1); //request elevators state
                sigsuspend(&set);
                read(pipe_fds[i+2][PFD_READ], &lift_state[i][0], 1);
                read(pipe_fds[i+2][PFD_READ], &lift_state[i][1], 1);
            }

            nearest_lift_idx = choose_nearest_lift(lift_state, &btn); //0 or 1 that matches with i-process index
            write(pipe_fds[nearest_lift_idx][PFD_WRITE], "=", 1);
            write(pipe_fds[nearest_lift_idx][PFD_WRITE], &btn, 1);
        }
        else
        {
            for(i = 0; i < proc_counter; i++)  //cabin lift buttons processing
            {
                pressed_lift_btn = is_lift_button_pressed(lifts_buttons[i], &btn);            
                if(pressed_lift_btn != '\0')
                {
                    write(pipe_fds[i][PFD_WRITE], ".", 1);   //indicates that lift button was pressed (passenger is inside the elevator)
                    write(pipe_fds[i][PFD_WRITE], &pressed_lift_btn, 1);
                    break;
                }
            }
        }
    }

    write(1, "\x1b[2J", 4);
    write(1, "\x1b[;H", 4);
    write(1, "\x1b[?25h", 6);

    tcsetattr(0, TCSAFLUSH, &prev_tty);       //restoring canonical input

    for(i = 0; i < proc_counter; i++)
    {
        kill(child_pid[i], SIGINT);
        waitpid(child_pid[i], 0, WUNTRACED);
    }

    exit(0);
}
