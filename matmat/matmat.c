#include <stdlib.h>
#include <omp.h>
/*

proc matmat do
    for _ = 1 to N1
        for _ = 1 to N2
            for _ = 1 to N3
                C(i,j) = C(i,j)+A(i,k)*B(k,j)
            endfor
        endfor
    endfor
endproc
*/

void matmatijk(int ldA, int ldB, int ldC, double *A, double *B, double *C, int N1, int N2, int N3)
{
    int i, j, k;
    for (i = 0; i < N1; ++i)
        for (j = 0; j < N2; ++j)
            for (k = 0; k < N3; ++k)
                C[ldC * i + j] = C[ldC * i + j] + A[ldA * i + k] * B[ldB * k + j];
}

void matmatkji(int ldA, int ldB, int ldC, double *A, double *B, double *C, int N1, int N2, int N3)
{
    int i, j, k;
    for (k = 0; k < N1; ++k)
        for (j = 0; j < N2; ++j)
            for (i = 0; i < N3; ++i)
                C[ldC * i + j] = C[ldC * i + j] + A[ldA * i + k] * B[ldB * k + j];
}

void matmatikj(int ldA, int ldB, int ldC, double *A, double *B, double *C, int N1, int N2, int N3)
{
    int i, j, k;
    for (i = 0; i < N1; ++i)
        for (k = 0; k < N2; ++k)
            for (j = 0; j < N3; ++j)
                C[ldC * i + j] = C[ldC * i + j] + A[ldA * i + k] * B[ldB * k + j];
}

void matmatjik(int ldA, int ldB, int ldC, double *A, double *B, double *C, int N1, int N2, int N3)
{
    int i, j, k;
    for (j = 0; j < N1; ++j)
        for (i = 0; i < N2; ++i)
            for (k = 0; k < N3; ++k)
                C[ldC * i + j] = C[ldC * i + j] + A[ldA * i + k] * B[ldB * k + j];
}

void matmatkij(int ldA, int ldB, int ldC, double *A, double *B, double *C, int N1, int N2, int N3)
{
    int i, j, k;
    for (k = 0; k < N1; ++k)
        for (i = 0; i < N2; ++i)
            for (j = 0; j < N3; ++j)
                C[ldC * i + j] = C[ldC * i + j] + A[ldA * i + k] * B[ldB * k + j];
}

void matmatjki(int ldA, int ldB, int ldC, double *A, double *B, double *C, int N1, int N2, int N3)
{
    int i, j, k;
    for (j = 0; j < N1; ++j)
        for (k = 0; k < N2; ++k)
            for (i = 0; i < N3; ++i)
                C[ldC * i + j] = C[ldC * i + j] + A[ldA * i + k] * B[ldB * k + j];
}

void matmatblock(int ldA, int ldB, int ldC, double *A, double *B, double *C, int N1, int N2, int N3, int dbA, int dbB, int dbC)
{
    int ii, jj, kk;
    for (ii = 0; ii < (N1 + dbA - 1) / dbA; ii++)
        for (jj = 0; jj < (N2 + dbB - 1) / dbB; jj++)
            for (kk = 0; kk < (N3 + dbC - 1) / dbC; kk++)
                matmatikj(
                    ldA, ldB, ldC,
                    &A[(ii * dbA * ldA) + (kk * dbA)],
                    &B[(kk * dbB * ldB) + (jj * dbB)],
                    &C[(ii * dbC * ldC) + (jj * dbC)],
                    dbA, dbB, dbC);
}

void matmatthread(int ldA, int ldB, int ldC, double *A, double *B, double *C, 
                  int N1, int N2, int N3, 
                  int dbA, int dbB, int dbC, 
                  int NTROW, int NTCOL) {
    int NT = NTROW * NTCOL, id, idi, idj;

    omp_set_num_threads(NT);

    #pragma omp parallel private(id, idi, idj)
    {
        id = omp_get_thread_num();
        idi = id / NTCOL;         
        idj = id % NTCOL;

        // Calcolare correttamente gli indici di inizio per i blocchi
        int start_i = idi * (N1 / NTROW) + (idi < N1 % NTROW ? idi : N1 % NTROW);
        int start_j = idj * (N3 / NTCOL) + (idj < N3 % NTCOL ? idj : N3 % NTCOL);

        // Calcolare la dimensione del blocco, tenendo conto dei possibili ultimi blocchi parziali
        int block_size_i = N1 / NTROW + (idi < N1 % NTROW ? 1 : 0);
        int block_size_j = N3 / NTCOL + (idj < N3 % NTCOL ? 1 : 0);
        
        // Ottimizzazione dell'uso della cache: operare su blocchi di matrice A, B e C
        for (int i = start_i; i < start_i + block_size_i; i += dbA) {
            for (int j = start_j; j < start_j + block_size_j; j += dbC) {
                // Chiamata alla funzione di blocco che lavora sui sotto-array
                matmatblock(ldA, ldB, ldC, &A[i * ldA], &B[j], &C[i * ldC + j], 
                            N1 / NTROW + (idi < N1 % NTROW ? 1 : 0), N2, 
                            N3 / NTCOL + (idj < N3 % NTCOL ? 1 : 0), dbA, dbB, dbC);
            }
        }
    }
}