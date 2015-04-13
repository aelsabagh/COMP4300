// forward declarations - link against BLAS
int daxpy_(int *n, double *da, double *dx, 
	int *incx, double *dy, int *incy);
int dgemm_(char *transa, char *transb, int *m, int *n, 
    int *k, double *alpha, double *a, int *lda, 
	double *b, int *ldb, double *beta, double *c, int *ldc);

// forward declaration - link against LAPACK
int dlacpy_(char *uplo, int *m, int *n, 
            double *a, int *lda, 
            double *b, int *ldb);

void pdgemm(int m, int n, int k, 
            int nb, 
            double alpha, double *a, int lda, 
                          double *b, int ldb, 
            double beta,  double *c, int ldc, 
            int m_a[], int n_a[], 
            int m_b[], int n_b[], 
            int m_c[], int n_c[], 
            MPI_Comm comm_row, MPI_Comm comm_col, 
            double *work1, double *work2);





