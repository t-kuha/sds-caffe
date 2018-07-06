#include <iostream>
//#include <stdio.h>
#include <stdlib.h>
#include <string.h>	// memset()

#ifdef __SDSCC__
#include "sds_lib.h"
#endif

#include "sds/sds_gemm.h"

//#include "sds/gemm_utils.h"
#include "sds/gemm_consts.h"
#include "sds/gemm_types.h"
//#include "sds/gemm_trans.h"
//#ifdef GEMM_BLOCK
//#include "sds/gemm_block.h"
//#endif
//#include "sds/gemm_plain.h"

// Prototype
void gemm_block_kernel(float ALPHA, float BETA, BlockedMatrix *A_blk, BlockedMatrix *B_blk, BlockedMatrix *C_blk);
void gemm_block(int TA, int TB, int M, int N, int K, float ALPHA,
    float *A, int lda, float *B, int ldb, float BETA, float *C, int ldc);
void gemm_block_calc_A(
    float ALPHA, float BETA, 
    BlockedMatrix *A_blk, 
    BlockedMatrix *B_blk, 
    BlockedMatrix *C_blk, 
    float *A_buf, float *B_buf, float *T_buf, 
    float *C_buf, float *R_buf);
void gemm_block_calc_B(
    float ALPHA, float BETA, 
    BlockedMatrix *A_blk, 
    BlockedMatrix *B_blk, 
    BlockedMatrix *C_blk, 
    float *A_buf, float *B_buf, float *T_buf, 
    float *C_buf, float *R_buf);
BlockedMatrix* flatten_matrix_to_blocked(int T, float *A, 
      int M, int N, int lda, int blk_m, int blk_n);
void blocked_matrix_to_flatten(BlockedMatrix* blk_mat, float *A);
void gemm_block_units_mplus(
        float T[BLK_M*BLK_N],
        float C[BLK_M*BLK_N],
        float R[BLK_M*BLK_N]);
void mplus_kernel(
        float T[BLK_M][BLK_N],
        float C[BLK_M][BLK_N],
        float R[BLK_M*BLK_N]);
void gemm_block_units_mmult(
        float A[BLK_M*BLK_K],
        float B[BLK_K*BLK_N],
        float ALPHA,
        float T[BLK_M*BLK_N]);
void mmult_kernel(
    float A[BLK_M][BLK_K],
    float B[BLK_K][BLK_N],
    float ALPHA,
    float T[BLK_M*BLK_N]);
int get_blocked_width(int orig, int blk_width);


// ------------------
void sds_gemm(int TA, int TB, int M, int N, int K,
    float ALPHA,
    float *A, int lda,
    float *B, int ldb,
    float BETA,
    float *C, int ldc)
{
//#ifdef GEMM_BLOCK
  gemm_block(TA,TB,M,N,K,ALPHA,A,lda,B,ldb,BETA,C,ldc);
//#else
//  gemm_plain(TA,TB,M,N,K,ALPHA,A,lda,B,ldb,BETA,C,ldc);
//#endif
}

// --------------------------
void gemm_block(int TA, int TB, int M, int N, int K, float ALPHA,
    float *A, int lda,
    float *B, int ldb,
    float BETA, 
    float *C, int ldc) 
{
  BlockedMatrix *A_blk = flatten_matrix_to_blocked(TA, A, M, K, lda, BLK_M, BLK_K);
  BlockedMatrix *B_blk = flatten_matrix_to_blocked(TB, B, K, N, ldb, BLK_K, BLK_N);
  BlockedMatrix *C_blk = flatten_matrix_to_blocked(0,  C, M, N, ldc, BLK_M, BLK_N);

  gemm_block_kernel(ALPHA,BETA,A_blk,B_blk,C_blk);

  blocked_matrix_to_flatten(C_blk, C);
}

void gemm_block_kernel(float ALPHA, float BETA, BlockedMatrix *A_blk, BlockedMatrix *B_blk, BlockedMatrix *C_blk)
{
#ifdef __SDSCC__
  float* A_buf = (float*)sds_alloc(sizeof(float)*NUM_DEPTH*PIPE_SIZE_MK);
  float* B_buf = (float*)sds_alloc(sizeof(float)*NUM_DEPTH*PIPE_SIZE_KN);
  float* T_buf = (float*)sds_alloc(sizeof(float)*NUM_DEPTH*PIPE_SIZE_MN);
  float* C_buf = (float*)sds_alloc(sizeof(float)*NUM_DEPTH*BLK_SIZE_MN);
  float* R_buf = (float*)sds_alloc(sizeof(float)*NUM_DEPTH*BLK_SIZE_MN);
#else
  float* A_buf = new float [NUM_DEPTH*PIPE_SIZE_MK];
  float* B_buf = new float [NUM_DEPTH*PIPE_SIZE_KN];
  float* T_buf = new float [NUM_DEPTH*PIPE_SIZE_MN];
  float* C_buf = new float [NUM_DEPTH*BLK_SIZE_MN];
  float* R_buf = new float [NUM_DEPTH*BLK_SIZE_MN];
#endif

  if (!A_buf || !B_buf || !C_buf || !T_buf || !R_buf) {
    std::cerr << "Error buffer allocation" << std::endl;
    exit(1);
  }

  memset(A_buf, 0, sizeof(float)*NUM_DEPTH*PIPE_SIZE_MK);
  memset(B_buf, 0, sizeof(float)*NUM_DEPTH*PIPE_SIZE_KN);
  memset(T_buf, 0, sizeof(float)*NUM_DEPTH*PIPE_SIZE_MN);
  memset(C_buf, 0, sizeof(float)*NUM_DEPTH*BLK_SIZE_MN);
  memset(R_buf, 0, sizeof(float)*NUM_DEPTH*BLK_SIZE_MN);
  // main calculation part
#ifdef AMAJOR
  gemm_block_calc_A(ALPHA,BETA,A_blk,B_blk,C_blk,A_buf,B_buf,T_buf,C_buf,R_buf);
#else
  gemm_block_calc_B(ALPHA,BETA,A_blk,B_blk,C_blk,A_buf,B_buf,T_buf,C_buf,R_buf);
#endif

#ifdef __SDSCC__
  sds_free(A_buf);
  sds_free(B_buf);
  sds_free(T_buf);
  sds_free(C_buf);
  sds_free(R_buf);
#else
  delete [] A_buf;
  delete [] B_buf;
  delete [] T_buf;
  delete [] C_buf;
  delete [] R_buf;
#endif
}

inline void gemm_block_calc_A(
    float ALPHA, float BETA, 
    BlockedMatrix *A_blk, 
    BlockedMatrix *B_blk, 
    BlockedMatrix *C_blk, 
    float *A_buf, float *B_buf, float *T_buf, 
    float *C_buf, float *R_buf) 
{
  int num_blk_M=A_blk->H/BLK_M;
  int num_blk_N=B_blk->W/BLK_N;
  int num_blk_K=A_blk->W/BLK_K;

  int i, j, k, d, p, t;
  int num_depth, num_pipes;

  for (j = 0; j < num_blk_N; j ++) {
    for (i = 0; i < num_blk_M; i += NUM_DEPTH) {
      num_depth = ((i+NUM_DEPTH) <= num_blk_M) ? NUM_DEPTH : num_blk_M-i;
      // initialize C_buf
      for (d = 0; d < num_depth; d ++)
        memcpy(&C_buf[d*BLK_SIZE_MN], 
               &C_blk->mat[((i+d)*num_blk_N+j)*BLK_SIZE_MN],
               sizeof(float)*BLK_SIZE_MN);
      for (t = 0; t < NUM_DEPTH*BLK_SIZE_MN; t ++)
        C_buf[t] *= BETA;

      // Here NUM_DEPTH number of C blocks will be computed
      for (k = 0; k < num_blk_K; k += NUM_PIPES) {
        num_pipes = ((k+NUM_PIPES) <= num_blk_K) ? NUM_PIPES : num_blk_K-k;
        // initialize A_buf
        for (d = 0; d < num_depth; d ++)
          memcpy(&A_buf[d*PIPE_SIZE_MK],
                 &A_blk->mat[((i+d)*num_blk_K+k)*BLK_SIZE_MK],
                 sizeof(float)*PIPE_SIZE_MK);
        
        // initialize B_buf
        for (p = 0; p < num_pipes; p ++) 
          for (d = 0; d < num_depth; d ++) 
            memcpy(&B_buf[d*PIPE_SIZE_KN+p*BLK_SIZE_KN], 
                   &B_blk->mat[((k+p)*num_blk_N+j)*BLK_SIZE_KN],
                   sizeof(float)*BLK_SIZE_KN);

        // compute
        gemm_block_units_mmult(A_buf,B_buf,ALPHA,T_buf);
        gemm_block_units_mplus(T_buf,C_buf,R_buf);

        // copy R_buf to C_buf
        memcpy(C_buf, R_buf, sizeof(float)*NUM_DEPTH*BLK_SIZE_MN);
      }

      // final write back
      for (d = 0; d < num_depth; d ++)
        memcpy(&C_blk->mat[((i+d)*num_blk_N+j)*BLK_SIZE_MN],
               &C_buf[d*BLK_SIZE_MN],
               sizeof(float)*BLK_SIZE_MN);
    }
  }
}

inline void gemm_block_calc_B(
    float ALPHA, float BETA, 
    BlockedMatrix *A_blk, 
    BlockedMatrix *B_blk, 
    BlockedMatrix *C_blk, 
    float *A_buf, float *B_buf, float *T_buf, 
    float *C_buf, float *R_buf) 
{
  int num_blk_M=A_blk->H/BLK_M;
  int num_blk_N=B_blk->W/BLK_N;
  int num_blk_K=A_blk->W/BLK_K;

  int i, j, k, d, p, t;
  int num_depth, num_pipes;
  for (i = 0; i < num_blk_M; i ++) {
    for (j = 0; j < num_blk_N; j += NUM_DEPTH) {
      num_depth = ((j+NUM_DEPTH) <= num_blk_N) ? NUM_DEPTH : num_blk_N-j;

      // initialize C_buf
      memcpy(C_buf, 
             &C_blk->mat[(i*num_blk_N+j)*BLK_SIZE_MN],
             sizeof(float)*NUM_DEPTH*BLK_SIZE_MN);
      for (t = 0; t < NUM_DEPTH*BLK_SIZE_MN; t ++)
        C_buf[t] *= BETA;

      // Here NUM_DEPTH number of C blocks will be computed
      for (k = 0; k < num_blk_K; k += NUM_PIPES) {
        memset(A_buf, 0, sizeof(float)*NUM_DEPTH*PIPE_SIZE_MK);
        memset(B_buf, 0, sizeof(float)*NUM_DEPTH*PIPE_SIZE_KN);
        memset(T_buf, 0, sizeof(float)*NUM_DEPTH*BLK_SIZE_MN);
        memset(R_buf, 0, sizeof(float)*NUM_DEPTH*BLK_SIZE_MN);

        num_pipes = ((k+NUM_PIPES) <= num_blk_K) ? NUM_PIPES : num_blk_K-k;
        // printf("(%3d %3d %3d) depth %3d pipes %3d\n", i, j, k, num_depth, num_pipes);
        // initialize A_buf
        for (d = 0; d < num_depth; d ++)
          memcpy(&A_buf[d*PIPE_SIZE_MK],
                 &A_blk->mat[(i*num_blk_K+k)*BLK_SIZE_MK],
                 sizeof(float)*num_pipes*BLK_SIZE_MK);
        
        // initialize B_buf
        for (d = 0; d < num_depth; d ++) 
          for (p = 0; p < num_pipes; p ++) 
            memcpy(&B_buf[d*PIPE_SIZE_KN+p*BLK_SIZE_KN], 
                   &B_blk->mat[((k+p)*num_blk_N+(j+d))*BLK_SIZE_KN],
                   sizeof(float)*BLK_SIZE_KN);

        // compute
        gemm_block_units_mmult(A_buf,B_buf,ALPHA,T_buf);
        gemm_block_units_mplus(T_buf,C_buf,R_buf);

        // copy R_buf to C_buf
        memcpy(C_buf, R_buf, sizeof(float)*NUM_DEPTH*BLK_SIZE_MN);
      }

      // final write back
      memcpy(&C_blk->mat[(i*num_blk_N+j)*BLK_SIZE_MN],
             C_buf,
             sizeof(float)*BLK_SIZE_MN*num_depth);
    }
  }
}

BlockedMatrix* flatten_matrix_to_blocked(int T, float *A, 
      int M, int N, int lda, int blk_m, int blk_n) {
  
  int M_align = get_blocked_width(M,blk_m);
  int N_align = get_blocked_width(N,blk_n);

  // generate a BlockedMatrix struct by given input parameters
  BlockedMatrix* blk_mat = (BlockedMatrix *) malloc(sizeof(BlockedMatrix));
  blk_mat->T   = T;
  // size params should be taken carefully
  blk_mat->H   = M_align;
  blk_mat->W   = N_align;
  blk_mat->bH  = blk_m;
  blk_mat->bW  = blk_n;
  blk_mat->oH  = (T == 0) ? M : N;
  blk_mat->oW  = (T == 0) ? N : M;
  blk_mat->ld  = lda;
#ifdef __SDSCC__
  if(blk_mat->mat){
	sds_free(blk_mat->mat);
  }
  blk_mat->mat = (float *) sds_alloc(sizeof(float)*blk_mat->H*blk_mat->W);
#else
  if(blk_mat->mat){
	delete [] blk_mat->mat;
  }
  blk_mat->mat = new float [blk_mat->H*blk_mat->W];
#endif

  if (!blk_mat->mat) {
    std::cerr << "Can't allocate memory for blocked matrix" << std::endl;
    exit(1);
  }

  // useful info
  int blk_id;
  int blk_size = blk_m * blk_n;
  int num_blk_M = M_align / blk_m;
  int num_blk_N = N_align / blk_n;
  
  // This is basically a linear algebra trick
  // First, we create a outer map from origin block(src_blk) to transposed block(dst_blk)
  for (blk_id = 0; blk_id < num_blk_M*num_blk_N; blk_id ++) {
    int src_blk_i = (T == 0) ? blk_id / num_blk_N : blk_id % num_blk_N;
    int src_blk_j = (T == 0) ? blk_id % num_blk_N : blk_id / num_blk_N;
    
    int src_base_i = src_blk_i * ((T==0) ? blk_m : blk_n);
    int src_base_j = src_blk_j * ((T==0) ? blk_n : blk_m);
    int blk_i, blk_j, src_i, src_j, blk_idx;
    
    for (blk_idx = 0; blk_idx < blk_size; blk_idx ++) {
      blk_i = (T == 0) ? blk_idx / blk_n : blk_idx % blk_n;
      blk_j = (T == 0) ? blk_idx % blk_n : blk_idx / blk_n;
      src_i = src_base_i + blk_i;
      src_j = src_base_j + blk_j;

      float src = 
        ((T == 0 && src_i < M && src_j < N) ||
         (T == 1 && src_i < N && src_j < M)) 
        ? A[src_i*lda+src_j] : 0.0;
      // printf("i %d j %d src %lf\n", src_i, src_j, src);
      blk_mat->mat[blk_id*blk_size+blk_idx] = src;
    }
  }

  return blk_mat;
}

void blocked_matrix_to_flatten(BlockedMatrix* blk_mat, float *A) {
  int blk_id;
  int blk_size  = blk_mat->bH * blk_mat->bW;
  int num_blk_W = blk_mat->W / blk_mat->bW;
  int blk_n = blk_mat->bW;
  int blk_m = blk_mat->bH;

  int i, j;
  int T = blk_mat->T;
  /* original shape */
  int M = (T) ? blk_mat->oW : blk_mat->oH;
  int N = (T) ? blk_mat->oH : blk_mat->oW;
  int lda = blk_mat->ld;
  for (i = 0; i < blk_mat->W*blk_mat->H; i += blk_size) {
    blk_id = i / blk_size;
    int blk_id_m = blk_id/num_blk_W;
    int blk_id_n = blk_id%num_blk_W;

    for (j = 0; j < blk_size; j++) {
      int x = (T) ? blk_id_n*blk_n+j%blk_n : blk_id_m*blk_m+j/blk_n;
      int y = (T) ? blk_id_m*blk_m+j/blk_n : blk_id_n*blk_n+j%blk_n;
      if (( T && x < N && y < M) ||
          (!T && x < M && y < N)) {
        A[x*lda+y] = blk_mat->mat[i+j];          
      }
    }
  }
}

void mmult_kernel(
    float A[BLK_M][BLK_K],
    float B[BLK_K][BLK_N],
    float ALPHA,
    float T[BLK_M*BLK_N])
{
#pragma HLS INLINE self
// scale for GEMM array partitions
#if GEMM_SCALE == 0
  #pragma HLS array_partition variable=A block factor=16 dim=2
  #pragma HLS array_partition variable=B block factor=16 dim=1
#elif GEMM_SCALE == 1
  #pragma HLS array_partition variable=A block factor=16 dim=2
  #pragma HLS array_partition variable=B block factor=16 dim=1
#elif GEMM_SCALE == 2
  #pragma HLS array_partition variable=A block factor=8 dim=2
  #pragma HLS array_partition variable=B block factor=8 dim=1
#endif

  int i, j, k;
  RowLoop: for (i = 0; i < BLK_M; i ++) {
    ColLoop: for (j = 0; j < BLK_N; j ++) {
    #pragma HLS pipeline II=1
      float sum = 0.0;
      ProductLoop: for (k = 0; k < BLK_K; k ++) {
        float temp = A[i][k] * B[k][j];
        sum += temp;
      }
      T[i*BLK_N+j] = sum;
    }
  }
}

void gemm_block_units_mmult(
        float A[BLK_M*BLK_K],
        float B[BLK_K*BLK_N],
        float ALPHA,
        float T[BLK_M*BLK_N])
{
  float A_buf[BLK_M][BLK_K];
  float B_buf[BLK_K][BLK_N];

  int i, j;
  RowCopy: for (i = 0; i < BLK_M; i ++)
    ColCopy: for (j = 0; j < BLK_K; j ++) {
    #pragma HLS pipeline
      A_buf[i][j] = ALPHA * A[i*BLK_K+j];
      // assume A and B has the same shape
      B_buf[i][j] = B[i*BLK_N+j];
    }

  mmult_kernel(A_buf,B_buf,ALPHA,T);
}

void mplus_kernel(
        float T[BLK_M][BLK_N],
        float C[BLK_M][BLK_N],
        float R[BLK_M*BLK_N])
{
#pragma HLS INLINE self
  int i, j;
  PlusRow: for (i = 0; i < BLK_M; i ++)
    PlusCol: for (j = 0; j < BLK_N; j ++)
    #pragma HLS pipeline II=1
      R[i*BLK_N+j] = C[i][j] + T[i][j];
}

void gemm_block_units_mplus(
        float T[BLK_M*BLK_N],
        float C[BLK_M*BLK_N],
        float R[BLK_M*BLK_N])
{
  float T_buf[BLK_M][BLK_N];
  float C_buf[BLK_M][BLK_N];

  int i, j;
  PlusRowCopy: for (i = 0; i < BLK_M; i ++)
    PlusColCopy: for (j = 0; j < BLK_N; j ++) {
    #pragma HLS pipeline
      T_buf[i][j] = T[i*BLK_N+j];
      C_buf[i][j] = C[i*BLK_N+j];
    }

  mplus_kernel(T_buf,C_buf,R);
}

//#ifdef __SDSCC__
//#include "hls_math.h"
//#else
#include <cmath>
//#endif
int get_blocked_width(int orig, int blk_width) {
//#ifdef __SDSCC__
//  return (int) hls::ceil((float)orig/blk_width)*blk_width;
//#else
  return (int) std::ceil((double)orig/blk_width)*blk_width;
//#endif
}