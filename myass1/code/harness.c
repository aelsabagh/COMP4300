#include <stdlib.h>
#include <stdio.h>
#include "mpi.h"
#include "ass1.h"
#include "summa.h"

#define ITERS 5

#define A(I,J) a[(I) + ((J))* ( m)]
#define B(I,J) b[(I) + ((J))* ( k)]
#define C(I,J) c[(I) + ((J))* ( m)]
#define D(I,J) d[(I) + ((J))* ( m)]
#define TIME
#define OUTPUT

int compare_dgemm(double* a, double* b, double*c, int m, int n, int k);

int main(int argc, char *argv[]) {
    if (argc < 4) {
        fprintf(stderr, "Usage: ass1 M N K\n");
        exit(1);
    }
    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int k = atoi(argv[3]);

    int rank, size;
    double t1=0,time=0,timeMin=0;

    MPI_Init( &argc, &argv );
    MPI_Comm_size( MPI_COMM_WORLD, &size );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    //if (!rank) printf("m %d n %d k %d\n", m, n, k);
    double *a = malloc(m*k*sizeof(double));
    double *b = malloc(k*n*sizeof(double));
    double *c = malloc(m*n*sizeof(double));
    int i, j;

    /* Naive Test Begin
    if (!rank) {
        for(i=0; i<m; i++) {
            for (j=0; j<k; j++) {
                A(i,j) = (double)(i+j);
            }
        }
        for(i=0; i<k; i++) {
            for (j=0; j<n; j++) {
                B(i,j) = (double)(i+j)-1;
            }
        }
        for(i=0; i<m; i++) {
            for (j=0; j<n; j++) {
                C(i,j) = 0.0;
            }
        }
    }
    MPI_Barrier( MPI_COMM_WORLD );
    timeMin = 0;
    t1 = MPI_Wtime();
    mult_naive(m, n, k, (double*)a, (double*)b, (double*)c);
    timeMin = MPI_Wtime()-t1;
    for(i = 1; i < ITERS; i++) {
      t1 = MPI_Wtime();
      mult_naive(m, n, k, (double*)a, (double*)b, (double*)c);
      time = MPI_Wtime()-t1;
      timeMin = (time < timeMin) ? time : timeMin;
    }
    if (!rank) {
#ifdef TIME
      printf("%-20s%-7d%-7d%-7d%-5d%g\n", "Naive", m, n, k, 1, timeMin);
#endif
#ifdef OUTPUT
      printf("mult_naive:     ");
#endif
      compare_dgemm((double*)a, (double*)b, (double*)c, m, n, k);
    }
    /* Naive Test End */


    /* Implementation Test Begin */
    if (!rank) {
        for(i=0; i<m; i++) {
            for (j=0; j<k; j++) {
                A(i,j) = (double)(i+j);
            }
        }
        for(i=0; i<k; i++) {
            for (j=0; j<n; j++) {
                B(i,j) = (double)(i+j)-1;
            }
        }
        for(i=0; i<m; i++) {
            for (j=0; j<n; j++) {
                C(i,j) = 0.0;
            }
        }
    }
    MPI_Barrier( MPI_COMM_WORLD );
    t1 = MPI_Wtime();
    mult_replicated(m, n, k, (double*)a, (double*)b, (double*)c);
    time = MPI_Wtime()-t1;
    timeMin = time;
    for(i = 1; i < ITERS; i++) {
      t1 = MPI_Wtime();
      mult_replicated(m, n, k, (double*)a, (double*)b, (double*)c);
      time = MPI_Wtime()-t1;
      timeMin = (time < timeMin) ? time : timeMin;
    }
    if (!rank) {
#ifdef TIME
      printf("%-20s%-7d%-7d%-7d%-5d%g\n", "Replicated", m, n, k, size, timeMin);
#endif
#ifdef OUTPUT
      //printf("mult_replicated:     ");
#endif
      //compare_dgemm((double*)a, (double*)b, (double*)c, m, n, k);
    }
    /* Implementation Test End */



    /* SUMMA Test Begin */
    if (!rank) {
        for(i=0; i<m; i++) {
            for (j=0; j<k; j++) {
                A(i,j) = (double)(i+j);
            }
        }
        for(i=0; i<k; i++) {
            for (j=0; j<n; j++) {
                B(i,j) = (double)(i+j)-1;
            }
        }
        for(i=0; i<m; i++) {
            for (j=0; j<n; j++) {
                C(i,j) = 0.0;
            }
        }
    }
    MPI_Barrier( MPI_COMM_WORLD );
    timeMin = 0;
    t1 = MPI_Wtime();
    mult_summa(m, n, k, (double*)a, (double*)b, (double*)c);
    timeMin = MPI_Wtime()-t1;
    for(i = 1; i < ITERS; i++) {
      t1 = MPI_Wtime();
      mult_summa(m, n, k, (double*)a, (double*)b, (double*)c);
      time = MPI_Wtime()-t1;
      timeMin = (time < timeMin) ? time : timeMin;
    }

    MPI_Barrier( MPI_COMM_WORLD );
    if (!rank) {
#ifdef TIME
        printf("%-20s%-7d%-7d%-7d%-5d%g\n", "SUMMA", m, n, k, size, timeMin);
#endif
        //printf("mult_summa:           ");
        //compare_dgemm((double*)a, (double*)b, (double*)c, m, n, k);
    }
    /* SUMMA Test End */


    free(a); free(b); free(c);
    MPI_Finalize();
    return 0;
}

// compare matrix C=AB against DGEMM D=AB
int compare_dgemm(double* a, double* b, double*c, int m, int n, int k) {
    double *d = malloc(m*n*sizeof(double));
    char transa = 'N';
    char transb = 'N';
    double alpha = 1.0;
    double beta = 0.0;
    int i,j;
    int different = 0;

    for(i=0; i<m; i++) {
        for (j=0; j<k; j++) {
            A(i,j) = (double)(i+j);
        }
    }
    for(i=0; i<k; i++) {
        for (j=0; j<n; j++) {
            B(i,j) = (double)(i+j)-1;
        }
    }
    double t1=0,time=0,timeMin=0;
    t1 = MPI_Wtime();
    dgemm_(&transa, &transb,
           &m, &n, &k,
           &alpha, a, &m,
                   b, &k,
           &beta,  d, &m);
    time = MPI_Wtime()-t1;
    timeMin = time;
    for(i = 1; i < ITERS; i++) {
      t1 = MPI_Wtime();
      dgemm_(&transa, &transb,
             &m, &n, &k,
             &alpha, a, &m,
                     b, &k,
             &beta,  d, &m);
      time = MPI_Wtime()-t1;
      timeMin = (time < timeMin) ? time : timeMin;
    }

    for (j = 0; j < n; j++) {
        for(i = 0; i < m; i++) {
            if(D(i,j) != C(i,j)) {
                fprintf(stderr, "Error: c[%d][%d] = %f d[%d][%d] = %f\n", i,j,c[j*m+i], i,j,d[j*m+i]);
                different = 1;
            }
        }
    }

    if (different) {
#ifdef OUTPUT
        printf("failed\n");
#endif
    } else {
#ifdef OUTPUT
        printf("passed\n");
#endif
    }
#ifdef TIME
    printf("%-20s%-7d%-7d%-7d%-5d%g\n", "DGEMM", m, n, k, 1, timeMin);
#endif
    free(d);

    return different;
}
