/*

Implementation of SUMMA by Robert van de Geijn
http://www.cs.utah.edu/formal_verification/MPI_Tests/general_tests/advanced_mpi/03-summa/

Copyright (c) 2007-2007 The University of Tennessee and The University
                        of Tennessee Research Foundation.  All rights
                        reserved.
Copyright (c) 2007-2007 The University of Colorado at Denver and Health
                        Sciences Center.  All rights reserved.

with minor modifications by Josh Milthorpe
Copyright (c) 2013 Australian National University

$COPYRIGHT$

Additional copyrights may follow

$HEADER$

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

- Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

- Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer listed
  in this license in the documentation and/or other materials
  provided with the distribution.

- Neither the name of the copyright holders nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

The copyright holders provide no reassurances that the source code
provided does not infringe any patent, copyright, or any other
intellectual property rights of third parties.  The copyright holders
disclaim any liability to any recipient for claims brought against
recipient by any third party for infringement of that parties
intellectual property rights.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "mpi.h"
#include "summa.h"
                /* macro for column major indexing                 */
#define A( i,j ) (a[ j*lda + i ])
#define B( i,j ) (b[ j*ldb + i ])
#define C( i,j ) (c[ j*ldc + i ])

#define min( x, y ) ( (x) < (y) ? (x) : (y) )

void RING_SUM(double *buf, int count, MPI_Datatype type, int root, MPI_Comm comm, double *work);
void RING_Bcast(double *buf, int count, MPI_Datatype type, int root, MPI_Comm comm);

int    i_one=1;         /* used for constant passed to blas call   */
double d_one=1.0,
       d_zero=0.0;      /* used for constant passed to blas call   */

void pdgemm( m, n, k, nb, alpha, a, lda, b, ldb,
             beta, c, ldc, m_a, n_a, m_b, n_b, m_c, n_c,
             comm_row, comm_col, work1, work2 )
int    m, n, k,         /* global matrix dimensions                */
       nb,              /* panel width                             */
       m_a[], n_a[],    /* dimensions of blocks of A               */
       m_b[], n_b[],    /* dimensions of blocks of B               */
       m_c[], n_c[],    /* dimensions of blocks of C               */
       lda, ldb, ldc;   /* leading dimension of local arrays that
                           hold local portions of matrices A, B, C */
double *a, *b, *c,      /* arrays that hold local parts of A, B, C */
       alpha, beta,     /* multiplication constants                */
       *work1, *work2;  /* work arrays                             */
MPI_Comm comm_row,      /* Communicator for this row of nodes      */
       comm_col;        /* Communicator for this column of nodes   */
{
  int myrow, mycol,     /* my  row and column index                */
      i, j, kk, iwrk,   /* misc. index variables                   */
      icurrow, icurcol, /* index of row and column that hold current
                           row and column, resp., for rank-1 update*/
      ii, jj;           /* local index (on icurrow and icurcol, resp.)
                           of row and column for rank-1 update     */

                /* get myrow, mycol                                */
  MPI_Comm_rank( comm_row, &mycol );  MPI_Comm_rank( comm_col, &myrow );
                /* scale local block of C                          */
  for ( j=0; j<n_c[ mycol ]; j++ )
    for ( i=0; i<m_c[ myrow ]; i++ )
      C( i,j ) = beta * C( i,j );

  icurrow = 0;          icurcol = 0;
  ii = jj = 0;

  for ( kk=0; kk<k; kk+=iwrk) {
    iwrk = min( nb, m_b[ icurrow ]-ii );
    iwrk = min( iwrk, n_a[ icurcol ]-jj );
                /* pack current iwrk columns of A into work1       */
    if ( mycol == icurcol )
       dlacpy_( "General", &m_a[ myrow ], &iwrk, &A( 0, jj ), &lda, work1,
                &m_a[ myrow ] );
                /* pack current iwrk rows of B into work2          */
    if ( myrow == icurrow )
       dlacpy_( "General", &iwrk, &n_b[ mycol ], &B( ii, 0 ), &ldb, work2,
                &iwrk );
                /* broadcast work2                                 */
                /* broadcast work1 and work2                       */
    RING_Bcast( work1 , m_a[ myrow ]*iwrk, MPI_DOUBLE, icurcol, comm_row );
    RING_Bcast( work2 , n_b[ mycol ]*iwrk, MPI_DOUBLE, icurrow, comm_col );
                /* update local block                              */
    dgemm_( "No transpose", "No transpose", &m_c[ myrow ], &n_c[ mycol ],
            &iwrk, &alpha, work1, &m_a[ myrow ], work2, &iwrk, &d_one,
            c, &ldc );      
                 /* update icurcol, icurrow, ii, jj                 */
    ii += iwrk;           jj += iwrk;
    if ( jj>=n_a[ icurcol ] ) { icurcol++; jj = 0; };
    if ( ii>=m_b[ icurrow ] ) { icurrow++; ii = 0; };
  }
}


void RING_SUM( buf, count, type, root, comm, work )
double *buf, *work;
int count, root;
MPI_Datatype type;
MPI_Comm comm;
{
  int me, np;
  MPI_Status status;

  MPI_Comm_rank( comm, &me );    MPI_Comm_size( comm, &np );
  if ( me != (root+1)%np ) {
    MPI_Recv( work, count, type, (me-1+np)%np, MPI_ANY_TAG, comm, &status );
    daxpy_( &count, &d_one, work, &i_one, buf, &i_one );
  }
  if ( me != root )
    MPI_Send( buf, count, type, (me+1)%np, 0, comm );
}


void RING_Bcast( buf, count, type, root, comm )
double *buf;
int count, root;
MPI_Datatype type;
MPI_Comm comm;
{
  int me, np;
  MPI_Status status;

  MPI_Comm_rank( comm, &me );    MPI_Comm_size( comm, &np );
  if ( me != root)
    MPI_Recv( buf, count, type, (me-1+np)%np, MPI_ANY_TAG, comm, &status );
  if ( ( me+1 )%np != root )
    MPI_Send( buf, count, type, (me+1)%np, 0, comm );
}
