#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define M 1700
#define N 1000
#define K 800

int main()
{
    int i, j, k;
    double start_time, end_time;
    double **A, **B, **C;

    // Allocate memory for matrices
    A = (double **) malloc(M * sizeof(double *));
    B = (double **) malloc(K * sizeof(double *));
    C = (double **) malloc(M * sizeof(double *));
    for (i = 0; i < M; i++) {
        A[i] = (double *) malloc(K * sizeof(double));
        C[i] = (double *) malloc(N * sizeof(double));
    }
    for (i = 0; i < K; i++) {
        B[i] = (double *) malloc(N * sizeof(double));
    }

    // Initialize matrices
    for (i = 0; i < M; i++) {
        for (j = 0; j < K; j++) {
            A[i][j] = i + j;
        }
    }
    for (i = 0; i < K; i++) {
        for (j = 0; j < N; j++) {
            B[i][j] = i + j;
        }
    }
    omp_set_num_threads(4);
    // Perform matrix multiplication
    int h = 0;
    start_time = omp_get_wtime();
    #pragma omp parallel for private(i, j, k) shared(A, B, C)
    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            C[i][j] = 0.0;
            for (k = 0; k < K; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
        if (h == 0 && omp_get_thread_num() == 0) {
            printf("Number of used threads is %d.\n", omp_get_num_threads());
            h = 1;
        }
    }
    end_time = omp_get_wtime();

    // Print result
    printf("Matrix multiplication (OPENMP) completed in %f seconds\n", end_time - start_time);

    // Free memory
    for (i = 0; i < M; i++) {
        free(A[i]);
        free(C[i]);
    }
    for (i = 0; i < K; i++) {
        free(B[i]);
    }
    free(A);
    free(B);
    free(C);

    return 0;
}
