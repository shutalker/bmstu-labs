#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <memory.h>
#include <signal.h>
#include <error.h>
#include <stdio.h>
#include "seabattle.h"

int serverSock;
int commSock;


void terminate_handler(int sig)
{
    int exitCode = 0;

    write(1, "\r\n", 3);
    write(1, "Server is shutting down...\n", 28);
    shutdown(commSock, 2);
    shutdown(serverSock, 2);
    //unlink(SOCK_FILE);
    destroyBattleField();

    if(sig = SIGINT)
        exitCode = 2;
    sleep(1);
    exit(exitCode);
}


void set_act_terminate_server(struct sigaction *act)
{
    sigset_t sigMask;

    act->sa_handler = terminate_handler;

    sigemptyset(&sigMask);
    sigaddset(&sigMask, SIGTERM);
    sigaddset(&sigMask, SIGINT);
    sigaddset(&sigMask, SIGSEGV);

    act->sa_mask = sigMask;

    sigaction(SIGTERM, act, 0);
    sigaction(SIGINT, act, 0);
    sigaction(SIGSEGV, act, 0);
}


int main()
{
    socklen_t serverAddrLen, clientAddrLen;
    char cell[3];
    char requestBuf[3];
    short loseFlag = 0;
    short damageFlag = 0;
    short missFlag = 1; //means that server first wair for client's turn
    struct sockaddr_in serverAddr;
    struct sigaction actTerminateServer;
    struct hostent *hp;

    memset(&actTerminateServer, 0, sizeof(actTerminateServer));
    set_act_terminate_server(&actTerminateServer);

    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(9999);
    hp = gethostbyname("127.0.0.1");
    memcpy ((char *)&serverAddr.sin_addr, hp->h_addr, hp->h_length);
    serverAddrLen = sizeof(serverAddr);    

    serverSock = socket(AF_INET, SOCK_STREAM, 0);
    bind(serverSock, (struct sockaddr *)(&serverAddr), serverAddrLen);

    write(1, "\x1b[2J", 5);
    write(1, "\x1b[;H", 5);
    write(1, "Waiting for opponent...", 24);

    initBattleField();
    setShipsOnBattleField();
    renderBattleField();

    listen(serverSock, 1);
    commSock = accept(serverSock, NULL, NULL);

    if(commSock != -1)
    {
        write(1, "\x1b[;H", 5);
        write(1, "Opponent connected successfully", 32);
        write(1, "\x1b[17;H", 7);
    }
    else
    {
        write(1, "An error has occured while accepting:", 37);
        perror(" accept");
        kill(getpid(), SIGINT);   
    }

    while(bfield.shipsAmount > 0)
    {
        if(missFlag)
        {
            do
            {
                while(read(commSock, requestBuf, 3) == 0)
                    usleep(1); //waiting for client's turn
    
                damageFlag = getDamage(requestBuf);
                renderBattleField();
            
                if(damageFlag == 1) //if ship was damaged
                    write(commSock, "+", 2); // '+' means that opponent nanes uron
                else if(damageFlag == 2) //if ship was destroyed
                {
                    if(bfield.shipsAmount < 1) //if client lost
                    {
                        write(commSock, ".", 2);
                        write(1, "YOU LOSE!", 10);
                        loseFlag = 1;
                        break;
                    }
    
                    write(commSock, "!", 2);
                }
                else if(damageFlag == 0)
                    write(commSock, "?", 2);
            }   
            while(damageFlag > 0);
        
            missFlag = 0;
        }

        if(loseFlag)
            continue;
        
        do  //input processing
        {
            write(1, "\x1b[16;H", 7);
            write(1, "Select cell: ", 14);
            read(0, cell, 3);
            cell[2] = '\0';
            write(1, "\x1b[16;H", 7);
            write(1, "\x1b[0J", 5);
        }
        while(checkCell(cell) == 1);

        write(commSock, cell, 3);
        write(1, "\x1b[17;H", 7);

        while(read(commSock, requestBuf, 3) == 0)
            usleep(1); //waiting for response from client
        
        if(requestBuf[0] == '?') // '?' means that client missed
        {
            write(1, "MISSED!", 8);
            write(1, "\x1b[18;H", 7); 
            missFlag = 1;           
            continue; 
        }
        else if(requestBuf[0] == '+') //if opponent's ship was damaged
        {
            write(1, "DAMAGED!", 9);
            missFlag = 0;
            continue;
        }
        else if(requestBuf[0] == '!') //if opponent's ship was destroyed
        {
            write(1, "DESTROYED!", 11);
            missFlag = 0;
            continue;
        }
        else if(requestBuf[0] == '.') //if opponent lost
        {
            write(1, "YOU WIN!", 9);
            missFlag = 0;
            break;
        }      
    }

    shutdown(commSock, 2);
    shutdown(serverSock, 2);
    //unlink(SOCK_FILE);
    destroyBattleField();
    exit(0);
}
