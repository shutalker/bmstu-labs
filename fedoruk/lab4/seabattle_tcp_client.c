#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <memory.h>
#include <signal.h>
#include <string.h>
#include "seabattle.h"

int clientSock;


void terminate_handler(int sig)
{
    shutdown(clientSock, 2);
    destroyBattleField();
    write(1, "\x1b[2J", 5);
    write(1, "\x1b[;H", 5);
    write(1, "Client was unexpectedly terminated\r\n", 37);
    exit(0);
}


void set_act_terminate_client(struct sigaction *act)
{
    sigset_t sigMask;

    act->sa_handler = terminate_handler;

    sigemptyset(&sigMask);
    sigaddset(&sigMask, SIGTERM);
    sigaddset(&sigMask, SIGINT);

    act->sa_mask = sigMask;

    sigaction(SIGTERM, act, 0);
    sigaction(SIGINT, act, 0);
}


int main()
{
    socklen_t serverAddrLen, clientAddrLen;
    char cell[3];
    char responseBuf[3];
    short damageFlag = 0;
    struct sockaddr_in serverAddr, clientAddr;
    struct sigaction actTerminateClient;
    struct hostent *hp;

    memset(&actTerminateClient, 0, sizeof(actTerminateClient));
    set_act_terminate_client(&actTerminateClient);

    clientAddr.sin_family = AF_INET;
    clientAddr.sin_port = htons(9998);
    hp = gethostbyname("127.0.0.1");
    memcpy ((char *)&clientAddr.sin_addr, hp->h_addr, hp->h_length);
    clientAddrLen = sizeof(clientAddr);

    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(9999);
    hp = gethostbyname("127.0.0.1");
    memcpy ((char *)&serverAddr.sin_addr, hp->h_addr, hp->h_length);
    serverAddrLen = sizeof(serverAddr);    

    clientSock = socket(AF_INET, SOCK_STREAM, 0);
    bind(clientSock, (struct sockaddr *)(&clientAddr), clientAddrLen);

    write(1, "\x1b[2J", 5);
    write(1, "\x1b[;H", 5);

    if(connect(clientSock, (struct sockaddr *)(&serverAddr), serverAddrLen) == 0)
        write(1, "Connected to server successfully\r\n", 35);
    else
    {
        write(1, "Unable to connect to the server\r\n", 34);
        exit(2);
    }

    initBattleField();
    setShipsOnBattleField();
    renderBattleField();

    while(bfield.shipsAmount > 0)
    {

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

        write(clientSock, cell, 3);

        while(read(clientSock, responseBuf, 3) == 0)
            usleep(1); //waiting for response from server

        write(1, "\x1b[17;H", 7);

        if(responseBuf[0] == '?') // '?' means that client missed
        {
            write(1, "MISSED!", 8);
            write(1, "\x1b[18;H", 7);            

            do
            {
                while(read(clientSock, responseBuf, 3) == 0)
                    usleep(1); //waiting for server's turn
 
                damageFlag = getDamage(responseBuf);
                renderBattleField();
           
                if(damageFlag == 1) //if client's ship was damaged
                    write(clientSock, "+", 2); // '+' means that opponent nanes uron
                else if(damageFlag == 2) //if client's ship was destroyed
                {
                    if(bfield.shipsAmount < 1) //if client lost
                    {
                        write(clientSock, ".", 2);
                        write(1, "YOU LOSE!", 10);
                        break;
                    }

                    write(clientSock, "!", 2);
                }
                else if(damageFlag == 0)
                    write(clientSock, "?", 2);
            }
            while(damageFlag > 0);  
        }
        else if(responseBuf[0] == '+') //if opponent's ship was damaged
        {
            write(1, "DAMAGED!", 9);
            continue;
        }
        else if(responseBuf[0] == '!') //if opponent's ship was destroyed
        {
            write(1, "DESTROYED!", 11);
            continue;
        }
        else if(responseBuf[0] == '.') //if opponent lost
        {
            write(1, "YOU WIN!", 9);
            break;
        }
    }

    shutdown(clientSock, 2);
    destroyBattleField();
    exit(0);
}
