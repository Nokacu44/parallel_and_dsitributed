#include <stdio.h>
#include <omp.h>
#include <mpi.h>
#include <stdlib.h>

void laplace(float *A, float *B, float *daprev, float *danext, int N, int LD, int Niter) {

    int NP, id, i, j, iter;
    MPI_Status status;

    MPI_Comm_size(MPI_COMM_WORLD, &NP);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    for (iter = 0; iter < Niter; ++iter) {

        if (id != 0) {
            for (j = 0; j < N; ++j) daprev[j] = A[j];

            MPI_Send(daprev, N, MPI_FLOAT, id - 1, 1, MPI_COMM_WORLD);
            MPI_Recv(daprev, N, MPI_FLOAT, id - 1, 0, MPI_COMM_WORLD, &status);
        }

        if (id != NP - 1) {
            for (j = 0; j < N; ++j) danext[j] = A[(N/NP - 1) * LD + j];

            MPI_Send(danext, N, MPI_FLOAT, id + 1, 0, MPI_COMM_WORLD);
            MPI_Recv(danext, N, MPI_FLOAT, id + 1, 1, MPI_COMM_WORLD, &status);
        }

        // FASE DI CALCOLO ##########################
        if (id != 0) // il primo processo non deve calcolare la prima riga
            for (j = 1; j < N - 1; ++j) 
                B[j] = (daprev[j] + A[1 * LD + j] + A[j - 1] + A[j + 1]) * 0.25;

        for (i = 1; i < N/NP - 1; ++i) // Tutti i processi calcolano almeno la rige comprese tra la prima e l'ultima
            for (j = 1; j < N - 1; ++j) 
                B[(i * LD) + j] = (A[(i + 1) * LD + j]+ A[(i - 1) * LD + j] + A[(i * LD) + (j - 1)] + A[(i * LD) + (j + 1)]) * 0.25;

        if (id != NP - 1) // l'ultimo processo non deve calcolare l'ultima riga
            for (j = 1; j < N - 1; ++j) 
                B[((N/NP - 1) * LD) + j] = (danext[j]+ A[((N/NP - 1) - 1) * LD + j] + A[((N/NP - 1)  * LD) + (j - 1)] + A[((N/NP - 1)  * LD) + (j + 1)]) * 0.25;
        // ######################################

        // FASE DI COPIA (evita problemi di consistenza) ##########################
        if (id != 0) 
            for (j = 1; j < N - 1; ++j) 
                A[j] = B[j];

        for (i = 1; i < N/NP - 1; ++i) 
            for (j = 1; j < N - 1; ++j) 
                A[(i * LD) + j] = B[(i * LD) + j];

        if (id != NP - 1)
            for (j = 1; j < N - 1; ++j) 
                A[((N/NP - 1) * LD) + j] = B[((N/NP - 1) * LD) + j];
        // ######################################
    }

}

void laplace_nb(float *A, float *B, float *daprev, float *danext, int N, int LD, int Niter) {
    int NP, id, i, j, iter;
    MPI_Status status;
    MPI_Request req[4];

    MPI_Comm_size(MPI_COMM_WORLD, &NP);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);



    for (iter = 0; iter < Niter; ++iter) {

        if (id != 0) {
            for (j = 0; j < N; ++j) daprev[j] = A[j];

            MPI_Isend(daprev, N, MPI_FLOAT, id - 1, 1, MPI_COMM_WORLD, &req[0]);
        }
        if (id != NP - 1) {
            for (j = 0; j < N; ++j) danext[j] = A[(N / NP - 1) * LD + j];
            
            MPI_Isend(danext, N, MPI_FLOAT, id + 1, 0, MPI_COMM_WORLD, &req[2]);
        }
        
        if (id != 0)        { MPI_Irecv(daprev, N, MPI_FLOAT, id - 1, 0, MPI_COMM_WORLD, &req[1]);}
        if (id != NP - 1)   { MPI_Irecv(danext, N, MPI_FLOAT, id + 1, 1, MPI_COMM_WORLD, &req[3]);}

        // FASE DI CALCOLO (da qui non bloccante) ##########################

        // nessun wait, i dati da ricevere non sono usati in questo calcolo
        for (i = 1; i < N / NP - 1; ++i) {
            for (j = 1; j < N - 1; ++j) {
                B[(i * LD) + j] = (A[(i + 1) * LD + j] + A[(i - 1) * LD + j] +
                                   A[(i * LD) + (j - 1)] + A[(i * LD) + (j + 1)]) * 0.25;
            }
        }

        if (id != 0) {
            MPI_Wait(&req[1], &status); // aspetta la completa ricezione prima di lavorare su daprev
            for (j = 1; j < N - 1; ++j) {
                B[j] = (daprev[j] + A[1 * LD + j] + A[j - 1] + A[j + 1]) * 0.25;
            }
        }
        if (id != NP - 1) {
            MPI_Wait(&req[3], &status); // aspetta la completa ricezione prima di lavorare su danext
            for (j = 1; j < N - 1; ++j) {
                B[((N / NP - 1) * LD) + j] = (danext[j] + A[((N / NP - 1) - 1) * LD + j] +
                                              A[((N / NP - 1) * LD) + (j - 1)] + A[((N / NP - 1) * LD) + (j + 1)]) * 0.25;
            }
        }
        // ######################################

        // FASE DI COPIA (evita problemi di consistenza) ##########################
        if (id != 0)
            for (j = 1; j < N - 1; ++j) 
                A[j] = B[j];

        for (i = 1; i < N / NP - 1; ++i)
            for (j = 1; j < N - 1; ++j) 
                A[(i * LD) + j] = B[(i * LD) + j];

        if (id != NP - 1) 
            for (j = 1; j < N - 1; ++j) 
                A[((N / NP - 1) * LD) + j] = B[((N / NP - 1) * LD) + j];
        // ######################################

    }
}





