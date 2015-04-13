int mult_naive(int m, int n, int k, double* a, double* b, double* c);
int mult_replicated(int m, int n, int k, double* a, double* b, double* c);

/* Perform distributed matrix multiplication using provided code for the SUMMA algorithm. */
int mult_summa(int m, int n, int k, double* a, double* b, double* c);
