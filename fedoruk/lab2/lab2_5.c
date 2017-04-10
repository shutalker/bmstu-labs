#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <semaphore.h>
#include <math.h>
#define _REENTRANT
#include <sched.h>
#include <sys/time.h>

typedef struct {
    pthread_t threadId;
    int threadIdx;
    int step;
    int firstRowIdx;
    int lastRowIdx;
} Thread;

double *LAEkoeffMatrix, *rightPartVector;
int LAEmatrixDimention;
int *accordVec;     //vector of columns replacements (vec[current_column] = some_current_column)
int threadsAmount;
int semPostCallsAmount, semPostCount; //for organizing scheme of parallel threads
Thread *threads;
sem_t *sems;
pthread_barrier_t barr1, barr2;


void generate_matrix(double *matrix, const int LAEmatrixDimention)
{
    int i, j;

    srand(time(NULL));

    for(i = 0; i < LAEmatrixDimention; i++)
    {
        for(j = 0; j < LAEmatrixDimention; j++)
            LAEkoeffMatrix[j + LAEmatrixDimention * i] = (double) (rand() % 10) - 5; 
    }

    for(i = 0; i < LAEmatrixDimention; i++)
    {
        accordVec[i] = i;
    }
}


void print_matrix()
{
    int i, j;

    if(accordVec)
    {
        for(i = 0; i < LAEmatrixDimention; i++)
            printf("%4d\t", accordVec[i] + 1);
        printf("\n");
    }

    for(i = 0; i < LAEmatrixDimention; i++)
    {
        for(j = 0; j < LAEmatrixDimention; j++)
            printf("%2.2lf\t", LAEkoeffMatrix[j + LAEmatrixDimention * i]);
        printf("\n");
    }
    printf("\n\n");
}


void start_timer(struct timeval *tv, struct timezone *tz)
{
    gettimeofday(tv, tz);
}


double stop_timer(struct timeval *startTV, struct timeval *endTV, struct timeval *diffTV, struct timezone *tz)
{
    gettimeofday(endTV, tz);
    diffTV->tv_sec = endTV->tv_sec - startTV->tv_sec;
	diffTV->tv_usec = endTV->tv_usec - startTV->tv_usec;

	if(diffTV->tv_usec < 0) 
    {
        diffTV->tv_sec--; 
        diffTV->tv_usec = 1000000 - diffTV->tv_usec; 
    }

    return (double)(diffTV->tv_sec + (double)(diffTV->tv_usec) / 1000000);
}


void * gauss_solution(void *arg)
{
    int replacing_col;
    int i, j, k, sval;
    double curr_element, lead_element;
    double temp;
    Thread *trd;

    trd = (Thread *)(arg);
    pthread_barrier_wait(&barr1);

    for(i = trd->firstRowIdx; i < trd->lastRowIdx; i += trd->step)
    {
      
        if(i != 0)
        {//gauss' exclusion (starts from the 2-nd row)
            for(k = 0; k < i; k++)// in this case 'k' is row index
            {
                sem_wait(&sems[trd->threadIdx]);

                for(j = 0; j < LAEmatrixDimention; j++)
                    LAEkoeffMatrix[j + LAEmatrixDimention * i] -= LAEkoeffMatrix[j + LAEmatrixDimention * k] * LAEkoeffMatrix[k + LAEmatrixDimention * i];
            } 
        }

        lead_element = LAEkoeffMatrix[0 + LAEmatrixDimention * i];
        replacing_col = i;

        for(j = 0; j < LAEmatrixDimention; j++)
        {//searching for the leading element
            curr_element = LAEkoeffMatrix[j + LAEmatrixDimention * i];

            if(fabs(curr_element) > fabs(lead_element))
            {
                replacing_col = j;
                lead_element = curr_element;
            }
        }

        if(i != replacing_col)//if leading element is not diagonal ([i][i])
        {//rearrange two columns in matrix
         //column with the leading element <---> column with diagonal element
            for(k = 0; k < LAEmatrixDimention; k++)// in this case 'k' is row index
            {
                temp = LAEkoeffMatrix[i + LAEmatrixDimention * k]; //temp = matrix[k][i];
                LAEkoeffMatrix[i + LAEmatrixDimention * k] = LAEkoeffMatrix[replacing_col + LAEmatrixDimention * k]; //matrix[j][i] = matrix[j][replacing_col]
                LAEkoeffMatrix[replacing_col + LAEmatrixDimention * k] = temp;
            }

            temp = accordVec[i];  //associate swap colums with columns replacement vector
            accordVec[i] = accordVec[replacing_col];    
            accordVec[replacing_col] = temp;
        }

        if(lead_element)
        {
            for(j = 0; j < LAEmatrixDimention; j++)
            {//normalize the i-row (row with diagonal element [i][i])
                LAEkoeffMatrix[j + LAEmatrixDimention * i] = LAEkoeffMatrix[j + LAEmatrixDimention * i] / lead_element;
            }
        }

        //printf("THREAD: %d\n", trd->threadIdx);
        //print_matrix();

        if(semPostCount >= threadsAmount)
        {
            semPostCount = 0;
            semPostCallsAmount++;
        }

        for(j = 0; j < threadsAmount; j++)
        {
            for(k = 0; k < semPostCallsAmount; k++)
		        sem_post(&sems[j]);
        }
        
        semPostCount++;
    }

    pthread_barrier_wait(&barr2);
    pthread_exit(NULL);
}


int main(int argc, char *argv[])
{
    int i;
    double processingTime;
    struct timeval timeV1, timeV2, diffTV; //for calculating time of LAE system processing by threads
    struct timezone timeZone;
    pthread_attr_t threadAttr;    

    if(argc < 3)
    {
        fprintf(stderr, "Usage: %s #matrix_dimention #threads_amount\n", argv[0]);
        exit(1);
    }

    LAEmatrixDimention = atoi(argv[1]);

    if(LAEmatrixDimention > 0)
        LAEkoeffMatrix = (double *) malloc(sizeof(double) * LAEmatrixDimention * LAEmatrixDimention);
    else
    {
        fprintf(stderr, "Matrix dimention must be a positive integer value\n");
        exit(2);
    }

    threadsAmount = atoi(argv[2]);    

    if((threadsAmount > 0) && (LAEmatrixDimention % threadsAmount == 0))
    {
        threads = (Thread *) malloc(sizeof(Thread) * threadsAmount);
        sems = (sem_t *) malloc(sizeof(sem_t) * threadsAmount);
    }
    else
    {
        fprintf(stderr, "Threads amount must be a multiple of matrix dimention\n");
        exit(3);
    }

    pthread_attr_init(&threadAttr);
    pthread_attr_setscope(&threadAttr, PTHREAD_SCOPE_SYSTEM);
    pthread_attr_setdetachstate(&threadAttr, PTHREAD_CREATE_JOINABLE);

    pthread_barrier_init(&barr1, NULL, threadsAmount + 1);
    pthread_barrier_init(&barr2, NULL, threadsAmount + 1);

    accordVec = (int *) malloc(sizeof(int) * LAEmatrixDimention);
    generate_matrix(LAEkoeffMatrix, LAEmatrixDimention);
    //print_matrix();
    semPostCount = 0;
    semPostCallsAmount = 1;

    for(i = 0; i < threadsAmount; i++)
    {
        threads[i].threadIdx = i;
        threads[i].step = threadsAmount;
        threads[i].firstRowIdx = i;
        threads[i].lastRowIdx = LAEmatrixDimention - threadsAmount + i + 1;
        sem_init(&sems[i], 0, 0);
        pthread_create(&threads[i].threadId, &threadAttr, gauss_solution, (void *)(&threads[i]));
    }

    pthread_barrier_wait(&barr1);
    start_timer(&timeV1, &timeZone);
    pthread_barrier_wait(&barr2);
    processingTime = stop_timer(&timeV1, &timeV2, &diffTV, &timeZone);

    printf("Processing time: %2.6lf s\n", processingTime);

    free(LAEkoeffMatrix);
    free(accordVec);
    free(threads);

    exit(0);
}
