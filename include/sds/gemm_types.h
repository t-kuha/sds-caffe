#ifndef GEMM_TYPES_H__
#define GEMM_TYPES_H__

/**
 * GEMM related types definitions
 */
typedef struct BlockedMatrix_tag{
  float *mat;     // matrix data
  int T;          // transpose flag
  int H, W;       // height and width
  int ld;         // leading dimension
  int bH, bW;     // block height and width
  int oH, oW;     // original shape
} BlockedMatrix;


#endif
