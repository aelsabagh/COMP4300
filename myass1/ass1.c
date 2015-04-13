#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "mpi.h"
#include "mkl.h"

#define A(I,J) a[(I) + ((J))* ( m)]
#define B(I,J) b[(I) + ((J))* ( k)]
#define C(I,J) c[(I) + ((J))* ( m)]

/*
#define PROGRESS
#define DEBUG
#define INPUT
#define OUTPUT
#define PRINTB
#define PRINTA
#define PRINTC
#define DETAILA
#define PRINT
*/


/**
Implementation of Naive (Sequential) Matrix Multiplication
Approach:
- Processor 0 multiplies matrices A and B to achieve C.
**/
int mult_naive(int m, int n, int k, double* a, double* b, double* c) {
#ifdef DEBUG
  printf("mult_naive: start\n");
#endif
  int i,j,l;
  for (i = 0; i < m; i++) {
      for (j = 0; j < n; j++) {
          C(i,j) = 0.0;
          for (l = 0; l < k; l++) {
              C(i,j) += A(i,l) * B(l,j);
          }
      }
  }
#ifdef DEBUG
  printf("mult_naive: end\n");
#endif
  return 0;
}



/**
Implementation of Distributed Matrix Multiplication
Approach:
- Columns of matrix B are distributed over (number of processors-1) by processor 0
- Matrix A is broadcast to all processors
- Processors 1..n multiply AB to find results for their assigned columns of C
- Processors 1..n send their results back to processor 0
**/
int mult_replicated(int m, int n, int k, double* a, double* b, double* c) {

    // Initialise
    int i, j, l;
    int std_work, low_bound, upper_bound, range;
    int rank, size, root=0;
    MPI_Status status;
    MPI_Request request;
    double alpha = 1.0;
    double beta  = 0.0;

    MPI_Comm_size( MPI_COMM_WORLD, &size );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    //If there's only one process, use it to compute and return (sequential)
    if (size == 1) {
      range = n;
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, range, k, alpha, a, m, b, k, beta, c, m);
      return 0;
    } //else, continue
#ifdef DEBUG
    if (rank==root) printf("mult_replicated: start\n");
#endif

    /* -------------- ALL PROCESSORS -------------- */
    /* Broadcast Matrix A to all slaves */
    MPI_Bcast(&a[0], m*k, MPI_DOUBLE, root, MPI_COMM_WORLD);

    if (rank==root) {
      /* -------------- MASTER -------------- */
#ifdef DEBUG
      printf("------------MASTER------------\n");
#endif
#ifdef INPUT
      printf("       ----Matrix A----\n");
      for (i = 0; i < m; i++) {
        if (i < 10) printf("row %d:  ", i);
        else printf("row %d: ", i);
          for (j = 0; j < k; j++) {
            if (A(i,j) < 10 && A(i,j) >= 0) {
              printf(" %0.f ", A(i,j));
            } else {
              printf("%0.f ", A(i,j));
            }
          }
          printf("\n");
      }
      printf("       ----Matrix B----\n");
      for (i = 0; i < k; i++) {
        if (i < 10) printf("row %d:  ", i);
        else printf("row %d: ", i);
          for (j = 0; j < n; j++) {
            if (B(i,j) < 10 && B(i,j) >= 0) {
              printf(" %0.f ", B(i,j));
            } else {
              printf("%0.f ", B(i,j));
            }
          }
          printf("\n");
      }
#endif
      for (i = 1; i < size; i++) {
        // Calculate each slave's portion of B's columns
        std_work = (n/(size-1));
        low_bound = ((i - 1) * std_work);
        if (((i + 1) == size) && ((n % (size - 1)) != 0)) {
          upper_bound = n;
        } else {
          upper_bound = low_bound + std_work;
        }

#ifdef OUTPUT
        printf("Master sending %d-%d to Slave %d\n", low_bound, upper_bound, i);
#endif
        // Send the lower & upper bounds
        MPI_Send(&low_bound, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
        MPI_Send(&upper_bound, 1, MPI_INT, i, 2, MPI_COMM_WORLD);
        range = upper_bound - low_bound;
        // Send the allocated columns of B to the intended slave
        MPI_Send(&b[low_bound*k], (range * k), MPI_DOUBLE, i, 3, MPI_COMM_WORLD);

#ifdef DETAILB
        printf("-sent low_bound = %d, upper_bound = %d\n", low_bound, upper_bound);
#endif
      }
#ifdef DEBUG
      printf("mult_replicated: Master finished partioning & send.\n");
#endif
    }
#ifdef DEBUG
    if (rank==root) printf("mult_replicated: %d finished bcast.\n", rank);
#endif

    if (rank > 0) {
      /* -------------- SLAVE -------------- */
#ifdef DEBUG
      printf("------------SLAVE %d------------\n", rank);
#endif
      // Receive lower & upper bounds
      MPI_Recv(&low_bound, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
      MPI_Recv(&upper_bound , 1, MPI_INT, 0, 2, MPI_COMM_WORLD, &status);
      range = upper_bound - low_bound;
      // Receive the allocated columns of B
      MPI_Recv(&b[0], (range * k), MPI_DOUBLE, 0, 3, MPI_COMM_WORLD, &status);

      /* Calculate */
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, range, k, alpha, a, m, b, k, beta, c, m);
      /* Send results back to Master */
      MPI_Send(&low_bound, 1, MPI_INT, 0, 4, MPI_COMM_WORLD);
      MPI_Send(&upper_bound, 1, MPI_INT, 0, 5, MPI_COMM_WORLD);
      range = upper_bound - low_bound;
      MPI_Send(&c[0], range * m, MPI_DOUBLE, 0, 6, MPI_COMM_WORLD);
#ifdef DETAIL
      printf("***Slave %d sending %d elements back\n", rank, range*m);
      printf("Matrix sizes\nA: %dx%d\nB: %dx%d\nC: %dx%d\n", m, k, k, range, m, n);
#endif
#ifdef PRINT
      printf("*Slave %d's calculations successful\n", rank);
#endif
#ifdef PRINTC
      printf("     ----Slave %d's C Matrix----\n", rank);
      for (i = 0; i < m; i++) {
        if (i < 10) printf("row %d:  ", i);
        else printf("row %d: ", i);
          for (j = 0; j < range; j++) {
            if (C(i,j) < 10 && C(i,j) >= 0) {
              printf(" %0.f ", C(i,j));
            } else {
              printf("%0.f ", C(i,j));
            }
          }
          printf("\n");
      }
#endif
#ifdef DETAILB
      printf("--Slave %d from %d-%d:%d--\n", rank, low_bound, upper_bound, num_elts);
#endif
#ifdef PRINTB
      printf("------ Slave %d's B Matrix from %d-%d:%d-------\n", rank, low_bound, upper_bound, range*k);
      for (i = 0; i < k; i++) {
        if (i < 10) printf("row %d:  ", i);
        else printf("row %d: ", i);
          for (j = 0; j < range; j++) {
            if (B(i,j) < 10 && B(i,j) >= 0) {
              printf(" %0.f ", B(i,j));
            } else {
              printf("%0.f ", B(i,j));
            }
          }
          printf("\n");
      }
#endif
#ifdef PRINTA
      printf("     ----Slave %d's A----\n", rank);
      for (i = 0; i < m; i++) {
        if (i < 10) printf("row %d:  ", i);
        else printf("row %d: ", i);
          for (j = 0; j < k; j++) {
            if (A(i,j) < 10 && A(i,j) >= 0) {
              printf(" %0.f ", A(i,j));
            } else {
              printf("%0.f ", A(i,j));
            }
          }
          printf("\n");
      }
#endif
#ifdef DETAILB
      //print the slave's matrix
      printf("b[el]:  ");
      for (el = 0; el < n*k; el++) {
        printf("%0.f ", b[el]);
      }
      printf("\n");
      //Print each slave's complete matrix
      printf("B(j,i): ");
      for (i = 0; i < n; i++) {
        for (j = 0; j < k; j++) {
          printf("%.0f ", B(j,i));
        }
      }
      printf("\n");
#endif
    }

    /* Gather matrix C at process 0 */
    if (rank == 0) {
#ifdef DEBUG
      printf("mult_replicated: Root waiting to receive results.\n");
#endif
        for (i = 1; i < size; i++) {// untill all slaves have handed back the processed data
          MPI_Recv(&low_bound, 1, MPI_INT, i, 4, MPI_COMM_WORLD, &status);
          MPI_Recv(&upper_bound, 1, MPI_INT, i, 5, MPI_COMM_WORLD, &status);
          range = upper_bound - low_bound;
          MPI_Recv(&c[low_bound*m], (range) * m, MPI_DOUBLE, i, 6, MPI_COMM_WORLD, &status);
        }
#ifdef OUTPUT
        printf("---------------OUTPUT---------------\n");
        printf("     ----Master's C----\n");
          for (i = 0; i < m; i++) {
            if (i < 10) printf("row %d:  ", i);
            else printf("row %d: ", i);
            for (j = 0; j < n; j++) {
            if (C(i,j) < 10 && C(i,j) >= 0) {
              printf("  %0.f ", C(i,j));
            } else if (C(i,j) >= 100) {
              printf("%0.f ", C(i,j));
            } else {
              printf(" %0.f ", C(i,j));
            }
          }
          printf("\n");
      }
#endif
    }

    return 0;
}


/**
Implementation of Distributed Matrix Multiplication using the SUMMA Algorithm
Approach:
- Blocks of matrix A and B are distributed over all processors
- Processors 0..n-1 calculate multiply AB to find results for their assigned blocks of C
- Processors 1..n send their results back to processor 0
**/
int mult_summa(int m, int n, int k, double* a, double* b, double* c) {

    // Initialise
    int i, j, l;
    int row_size, col_size;
    int mesh_width, my_row, my_col;
    int rows, cols;
    int block_size_a;
    int nprows, npcols;
    int maxIteration;
    int std_m_a, std_n_a, std_m_b, std_n_b, std_m_c, std_n_c;
    int rank, size;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &size );

#ifdef PROGRESS
    if (!rank) printf("-Initalising..\n");
#endif
    MPI_Comm comm_row, comm_col;
    MPI_Status status;
    int el;
    double val;
    int i_row, i_col;
    int *block_details_a = malloc (sizeof (int) * 9);
    int *block_details_b = malloc (sizeof (int) * 9);
    int *block_details_c = malloc (sizeof (int) * 9);
    int total;

    // Find the grid PxQ of processes
    maxIteration = sqrt(size);
    for (i = maxIteration; i >= 1; i--) {
      //printf("i = %d\n", i);
      if ((size%i) == 0) {
        nprows = i;
        break;
      }
    }
    npcols = size/nprows;
    assert(size == (nprows*npcols));

    /** Finding Block Sizes of A, B, C **/
    // Initialise
    int *m_a = malloc (sizeof (int) * nprows);
    int *n_a = malloc (sizeof (int) * npcols);
    int *m_b = malloc (sizeof (int) * nprows);
    int *n_b = malloc (sizeof (int) * npcols);
    int *m_c = malloc (sizeof (int) * nprows);
    int *n_c = malloc (sizeof (int) * npcols);
    std_m_a = m/nprows;
    std_n_a = k/npcols;
    std_m_b = k/nprows;
    std_n_b = n/npcols;
    std_m_c = m/nprows;
    std_n_c = n/npcols;

#ifdef PROGRESS
    if (!rank) printf("+Initialised!\n");
#endif
#ifdef PROGRESS
    if (!rank) printf("-Assigning block sizes to processes...\n");
#endif

    /* Calculating A Block Sizes */
    total = 0;
    for (i = 0; i < nprows; i++) {
      m_a[i] = std_m_a;
      total += std_m_a;
    }
    if (m%nprows != 0) {
      m_a[nprows-1] = std_m_a + m%nprows;
      total += m%nprows;
    }
    for (i = 0; i < npcols; i++) {
      n_a[i] = std_n_a;
      total += std_n_a;
    }
    if (k%npcols != 0) {
      n_a[npcols-1] = std_n_a + k%npcols;
      total += k%npcols;
    }
    assert(total == (m+k));

    /* Calculating B Block Sizes */
    total = 0;
    for (i = 0; i < nprows; i++) {
      m_b[i] = std_m_b;
      total += std_m_b;
    }
    if (k%nprows != 0) {
      m_b[nprows-1] = std_m_b + k%nprows;
      total += k%nprows;
    }
    for (i = 0; i < npcols; i++) {
      n_b[i] = std_n_b;
      total += std_n_b;
    }
    if (n%npcols != 0) {
      n_b[npcols-1] = std_n_b + n%npcols;
      total += n%npcols;
    }
    assert(total == (k+n));

    /* Calculating C Block Sizes */
    total = 0;
    for (i = 0; i < nprows; i++) {
      m_c[i] = std_m_c;
      total += std_m_c;
    }
    if (m%nprows != 0) {
      m_c[nprows-1] = std_m_c + m%nprows;
      total += m%nprows;
    }
    for (i = 0; i < npcols; i++) {
      n_c[i] = std_n_c;
      total += std_n_c;
    }
    if (n%npcols != 0) {
      n_c[npcols-1] = std_n_c + n%npcols;
      total += n%npcols;
    }
    assert(total == (m+n));
#ifdef PROGRESS
    if (!rank) printf("+Assigned block sizes to processes successfully!\n");
#endif

    /* Create communication meshes */
    mesh_width = npcols;
    my_row = rank / mesh_width;
    my_col = rank % mesh_width;
#ifdef DEBUG
    printf("mesh_width = %d\n", mesh_width);
    printf("rank= %d, my_row = %d, my_col = %d\n", rank, my_row, my_col);
#endif
    // Create communicators
    MPI_Comm_split( MPI_COMM_WORLD, my_row, my_col, &comm_row );
    MPI_Comm_split( MPI_COMM_WORLD, my_col, my_row, &comm_col );

    // Mesh info - acts as a barrier
    MPI_Comm_size ( comm_row, &row_size);
    MPI_Comm_size ( comm_col, &col_size);

    if (!rank) {
#ifdef PROGRESS
      printf("-Sending blocks to appropriate processes..!\n");
#endif
      // Get local blocks for all matrices for all processes
      for (i = size-1; i >= 0; i--) {
        i_row = i / mesh_width;
        i_col = i % mesh_width;
        block_details_a[0] = i_row;                        //process row
        block_details_a[1] = i_col;                        //process col
        block_details_a[2] = m_a[i_row];                   //no. of rows
        block_details_a[3] = n_a[i_col];                   //no. of cols
        block_details_a[4] = (m_a[i_row]*n_a[i_col]);      //block-size
        block_details_a[5] = (i_row*std_m_a);              //row lower bound
        block_details_a[6] = (i_row*std_m_a)+m_a[i_row];   //row upper bound
        block_details_a[7] = (i_col*std_n_a);              //col lower bound
        block_details_a[8] = (i_col*std_n_a)+n_a[i_col];   //col upper bound

        block_details_b[0] = i_row;                        //process row
        block_details_b[1] = i_col;                        //process col
        block_details_b[2] = m_b[i_row];                   //no. of rows
        block_details_b[3] = n_b[i_col];                   //no. of cols
        block_details_b[4] = (m_b[i_row]*n_b[i_col]);      //block-size
        block_details_b[5] = (i_row*std_m_b);              //row lower bound
        block_details_b[6] = (i_row*std_m_b)+m_b[i_row];   //row upper bound
        block_details_b[7] = (i_col*std_n_b);              //col lower bound
        block_details_b[8] = (i_col*std_n_b)+n_b[i_col];   //col upper bound
        if (i > 0) {
          double *localA = malloc (sizeof (double) * block_details_a[4]);
          double *localB = malloc (sizeof (double) * block_details_b[4]);

          /* Get Block of A of Process i */
          el = 0;
          for (j = block_details_a[7]; j < block_details_a[8]; j++) {
            for (l = block_details_a[5]; l < block_details_a[6]; l++) {
              val = a[(l) + ((j))* (m)];
              localA[el] = val;
              el++;
            }
          }

          /* Get Block of B of Process i */
          el = 0;
          for (j = block_details_b[7]; j < block_details_b[8]; j++) {
            for (l = block_details_b[5]; l < block_details_b[6]; l++) {
              val = b[(l) + (j*m)];
              localB[el] = val;
              el++;
            }
          }
          // Send Block A & B and their characteristics
          MPI_Send(&block_details_a[0], 9, MPI_INT, i, 1, MPI_COMM_WORLD);
          MPI_Send(&block_details_b[0], 9, MPI_INT, i, 2, MPI_COMM_WORLD);
          MPI_Send(&localA[0], block_details_a[4], MPI_DOUBLE, i, 3, MPI_COMM_WORLD);
          MPI_Send(&localB[0], block_details_b[4], MPI_DOUBLE, i, 4, MPI_COMM_WORLD);
        } else {
          /* Get Block of A for Process 0 */
          el = 0;
          for (j = block_details_a[7]; j < block_details_a[8]; j++) {
            for (l = block_details_a[5]; l < block_details_a[6]; l++) {
              val = a[(l) + (j*m)];
              a[el] = val;
              el++;
            }
          }
          /* Get Block of B for Process 0 */
          el = 0;
          for (j = block_details_b[7]; j < block_details_b[8]; j++) {
            for (l = block_details_b[5]; l < block_details_b[6]; l++) {
              val = b[(l) + (j*m)];
              b[el] = val;
              el++;
            }
          }
        }
      }

    } else {
      // Receive Block A & B and their characteristics
      MPI_Recv(&block_details_a[0], 9, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
      MPI_Recv(&block_details_b[0], 9, MPI_INT, 0, 2, MPI_COMM_WORLD, &status);
      MPI_Recv(&a[0], block_details_a[4], MPI_DOUBLE, 0, 3, MPI_COMM_WORLD, &status);
      MPI_Recv(&b[0], block_details_b[4], MPI_DOUBLE, 0, 4, MPI_COMM_WORLD, &status);
    }

#ifdef PROGRESS
    if (!rank) printf("+Sent blocks to appropriate processes successfully!\n");
#endif
    // Calculate C block details
    block_details_c[0] = my_row;                        //process row
    block_details_c[1] = my_col;                        //process col
    block_details_c[2] = m_c[my_row];                   //no. of rows
    block_details_c[3] = n_c[my_col];                   //no. of cols
    block_details_c[4] = (m_c[my_row]*n_c[my_col]);     //block-size
    block_details_c[5] = (my_row*std_m_c);              //row lower bound
    block_details_c[6] = (my_row*std_m_c)+m_c[my_row];  //row upper bound
    block_details_c[7] = (my_col*std_n_c);              //col lower bound
    block_details_c[8] = (my_col*std_n_c)+n_c[my_col];  //col upper bound
    double *localC = malloc (sizeof (double) * block_details_c[4]);

#ifdef DEBUG
    MPI_Barrier (MPI_COMM_WORLD);
    if (!rank) {
      printf("********* TESTING *********\n");
      printf("       ----Matrix A----\n");
      for (i = 0; i < m; i++) {
        printf("%5s%3d: ", "row ", i);
        for (j = 0; j < k; j++) {
          printf("%6.0f ", A(i,j));
        }
        printf("\n");
      }
      printf("     --- A Block Sizes ---\n");
      printf("%d:m_a[", m);
      for (i = 0; i < npcols-1; i++) printf("%d, ", m_a[i]);
      printf("%d]\n", m_a[npcols-1]);
      printf("%d:n_a[", k);
      for (i = 0; i < npcols-1; i++) printf("%d, ", n_a[i]);
      printf("%d]\n", n_a[npcols-1]);
    }

    if (!rank) {
      printf("       ----Matrix B----\n");
      for (i = 0; i < k; i++) {
        printf("%5s%3d: ", "row ", i);
        for (j = 0; j < n; j++) {
          printf("%6.0f ", B(i,j));
        }
        printf("\n");
      }
      printf("     --- B Block Sizes ---\n");
      printf("%d:m_b[", k);
      for (i = 0; i < nprows-1; i++) printf("%d, ", m_b[i]);
      printf("%d]\n", m_b[nprows-1]);
      printf("%d:n_b[", n);
      for (i = 0; i < npcols-1; i++) printf("%d, ", n_b[i]);
      printf("%d]\n", n_b[npcols-1]);
    }

    if (!rank) {
      printf("       ----Matrix C----\n");
      for (i = 0; i < m; i++) {
        printf("%5s%3d: ", "row ", i);
        for (j = 0; j < n; j++) {
          printf("%6.0f ", C(i,j));
        }
        printf("\n");
      }
      printf("     --- C Block Sizes ---\n");
      printf("%d:m_c[", m);
      for (i = 0; i < nprows-1; i++) printf("%d, ", m_c[i]);
      printf("%d]\n", m_c[nprows-1]);
      printf("%d:n_c[", n);
      for (i = 0; i < npcols-1; i++) printf("%d, ", n_c[i]);
      printf("%d]\n", n_c[npcols-1]);
    }
    MPI_Barrier (MPI_COMM_WORLD);
    printf("     ---- %d,%d: A Block: %dx%d ----\n", my_row, my_col, block_details_a[2], block_details_a[3]);
    for (i = 0; i < block_details_a[2]; i++) {
      printf("row %d: ", i);
      for (j = 0; j < block_details_a[3]; j++) {
        printf("%3.0f ", a[(i) + ((j))* (block_details_a[2])]);
      }
      printf("\n");
    }
    MPI_Barrier (MPI_COMM_WORLD);
    printf("     ---- %d,%d: B Block: %dx%d ----\n", my_row, my_col, block_details_b[2], block_details_b[3]);
    for (i = 0; i < block_details_b[2]; i++) {
      printf("row %d: ", i);
      for (j = 0; j < block_details_b[3]; j++) {
        printf("%3.0f ", b[(i) + ((j))* (block_details_b[2])]);
      }
      printf("\n");
    }
    printf("\n");
    MPI_Barrier (MPI_COMM_WORLD);
    printf("     ---- %d,%d: C Block: %dx%d ----\n", my_row, my_col, block_details_c[2], block_details_c[3]);
    for (i = 0; i < block_details_c[2]; i++) {
      printf("row %d: ", i);
      for (j = 0; j < block_details_c[3]; j++) {
        printf("%3.0f ", localC[(i) + ((j))* (block_details_c[2])]);
      }
      printf("\n");
    }
    printf("\n\n");
#endif

    /* PDGEMM */
    int nb = 20;
    double alpha = 1.0, beta = 0.0;
    double *work1 = malloc (sizeof (double) * (m*k));
    double *work2 = malloc (sizeof (double) * (k*n));
    pdgemm( m, n, k, nb, alpha, a, block_details_a[2], b, block_details_b[2],
            beta, localC, block_details_c[2], m_a, n_a, m_b, n_b, m_c, n_c,
            comm_row, comm_col, work1, work2 );

    if (rank) {
      // Processes 1..n send results back to Proc 0
      MPI_Send(&block_details_c[0], 9, MPI_INT, 0, 1, MPI_COMM_WORLD);
      MPI_Send(&localC[0], block_details_c[4], MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
    } else {
      // Proc 0 gets its results
      el = 0;
      for (j = block_details_c[7]; j < block_details_c[8]; j++) {
        for (l = block_details_c[5]; l < block_details_c[6]; l++) {
          c[(l) + ((j))* (m)] = localC[el];
          el++;
        }
      }
      // Proc 0 get the results of processes 1..n
      for (i = 1; i < size; i++) {
        /* Send results back */
        int *local_details_c = malloc (sizeof (int) * 9);
        MPI_Recv(&local_details_c[0], 9, MPI_INT, i, 1, MPI_COMM_WORLD, &status);

        double *localC = malloc (sizeof (double) * (local_details_c[4]));
        MPI_Recv(&localC[0], local_details_c[4], MPI_DOUBLE, i, 2, MPI_COMM_WORLD, &status);
        el = 0;
        // Order the results in C
        for (j = local_details_c[7]; j < local_details_c[8]; j++) {
          for (l = local_details_c[5]; l < local_details_c[6]; l++) {
            c[(l) + ((j))* (m)] = localC[el];
            el++;
          }
        }
        free(localC);
        free(local_details_c);
      }
    }

    // Free all allocated memory
    free(m_a);
    free(n_a);
    free(m_b);
    free(n_b);
    free(m_c);
    free(n_c);
    free(block_details_a);
    free(block_details_b);
    free(block_details_c);
    free(work1);
    free(work2);

    return 0;
}
