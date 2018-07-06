#ifndef GEMM_CONSTS_H__
#define GEMM_CONSTS_H__

/**
 * GEMM_SCALE: configure GEMM block size information
 * BLK_M, BLK_N and BLK_K will be set
 */
#if GEMM_SCALE >= 7 // irregular
  #define GEMM_IRREGULAR
  #define GEMM_RESOURCE_PARTITION
  #if GEMM_SCALE == 7
    #define DIM 68
    #define VEC 56
  #else
    #error "GEMM_SCALE doesn't exist."
  #endif
  // BLK_M == BLK_N,
  #define BLK_M DIM
  #define BLK_N BLK_M
  #define BLK_K VEC
#else 
  #if GEMM_SCALE == 0
  #define DIM 32
  #elif GEMM_SCALE == 1
  #define DIM 16
  #elif GEMM_SCALE == 2
  #define DIM 24
  #elif GEMM_SCALE == 3
  #define DIM 48
  #elif GEMM_SCALE == 4
  #define DIM 56
  #elif GEMM_SCALE == 5
  #define DIM 96
  #elif GEMM_SCALE == 6
  #define DIM 64 
  #endif
  // BLK_M == BLK_N == BLK_K
  #define BLK_M DIM
  #define BLK_N BLK_M
  #define BLK_K BLK_M
#endif

#define NUM_PIPES 1
#define NUM_DEPTH 1

#define BLK_SIZE_MN (BLK_M*BLK_N)
#define BLK_SIZE_KN (BLK_N*BLK_K)
#define BLK_SIZE_MK (BLK_M*BLK_K)

#define PIPE_SIZE_MN (BLK_SIZE_MN*NUM_PIPES)
#define PIPE_SIZE_KN (BLK_SIZE_KN*NUM_PIPES)
#define PIPE_SIZE_MK (BLK_SIZE_MK*NUM_PIPES)

#endif /* GEMM_CONSTS_H__ */
