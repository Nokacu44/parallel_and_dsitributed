#include <stdio.h>
#include <stdlib.h>

#define MEASURE_GFLOPS(FUNC, LABEL)                      \
  do                                                     \
  {                                                      \
    t1 = get_cur_time();                                 \
    FUNC(ldA, ldB, ldC, A, B, C, N1, N2, N3);            \
    t2 = get_cur_time();                                 \
    Gflops = (2.0 * (N1 * N2 * N3)) / ((t2 - t1) * 1e9); \
    printf("GFLOPS (%s) = %e \n", LABEL, Gflops);        \
  } while (0)

void set_matrix(double *MAT, int LD, int N, double value)
{
  int i, j, k;
  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      MAT[i * LD + j] = value;
}

void populate_matrix_random(double *MAT, int LD, int N)
{
  int i, j, k;
  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      MAT[i * LD + j] = (rand() % 100);
}

int main(void)
{
  double get_cur_time();
  void matmatijk(int, int, int, double *, double *, double *, int, int, int);
  void matmatkji(int, int, int, double *, double *, double *, int, int, int);
  void matmatikj(int, int, int, double *, double *, double *, int, int, int);
  void matmatjik(int, int, int, double *, double *, double *, int, int, int);
  void matmatkij(int, int, int, double *, double *, double *, int, int, int);
  void matmatjki(int, int, int, double *, double *, double *, int, int, int);
  void matmatblock(int ldA, int ldB, int ldC, double *A, double *B, double *C, int N1, int N2, int N3, int dbA, int dbB, int dbC);
  void matmatthread(int ldA, int ldB, int ldC, double *A, double *B, double *C,
                    int N1, int N2, int N3,
                    int dbA, int dbB, int dbC,
                    int NTROW, int NTCOL);
  int N1, N2, N3;
  int ldA, ldB, ldC;
  int iterations, iter;
  double *A, *B, *C;
  double Gflops;
  double t1, t2, total_time;

  ldA = ldB = ldC = 256;
  N1 = N2 = N3 = 256;

  A = (double *)malloc(sizeof(double) * ldA * ldA);
  B = (double *)malloc(sizeof(double) * ldB * ldB);
  C = (double *)malloc(sizeof(double) * ldC * ldC);

  populate_matrix_random(A, ldA, N1);
  populate_matrix_random(B, ldB, N2);
  set_matrix(C, ldC, N3, 0);

  iterations = 1536;

  // prova dati conformi
  set_matrix(A, ldA, N1, 1);
  set_matrix(B, ldB, N2, 1);
  set_matrix(C, ldC, N3, 1);
  int y;
  matmatijk(ldA, ldB, ldC, A, B, C, N1, N2, N3);
  for (y = 0; y < N3; ++y)
  {
    printf("%e ", C[ldC * y + y]);
  }
  //

  for (iter = 256; iter <= iterations; iter += 256)
  {
    // t1 = get_cur_time();
    // matmatijk(ldA, ldB, ldC, A, B, C, N1, N2, N3);
    // t2 = get_cur_time();
    // Gflops = (2.0 * (N1 * N2 * N3)) / ((t2 - t1) * 1e9);
    // printf("GFLOPS (i,j,k) = %e \n", Gflops);
    printf("Iterazioni: %d \n", iter);
    // MEASURE_GFLOPS(matmatijk, "i,j,k");
    // MEASURE_GFLOPS(matmatkji, "k,j,i");
    // MEASURE_GFLOPS(matmatikj, "i,k,j");
    // MEASURE_GFLOPS(matmatjik, "j,i,k");
    // MEASURE_GFLOPS(matmatkij, "k,i,j");
    // MEASURE_GFLOPS(matmatjki, "j,k,i");

    t1 = get_cur_time();
    matmatblock(ldA, ldB, ldC, A, B, C, N1, N2, N3, 64, 64, 64);
    t2 = get_cur_time();
    Gflops = (2.0 * (N1 * N2 * N3)) / ((t2 - t1) * 1e9);
    printf("GFLOPS matmatblock = %e \n", Gflops);

    t1 = get_cur_time();
    matmatthread(ldA, ldB, ldC, A, B, C, N1, N2, N3, 64, 64, 64, 2, 4);
    t2 = get_cur_time();
    Gflops = (2.0 * (N1 * N2 * N3)) / ((t2 - t1) * 1e9);
    printf("GFLOPS matmatthread = %e \n", Gflops);
  }
}