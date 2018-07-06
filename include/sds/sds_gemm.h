#ifndef __SDS_GEMM_H__
#define __SDS_GEMM_H__

//#include "sds/gemm_types.h"

void sds_gemm(int TA, int TB, int M, int N, int K,
    float ALPHA,
    float *A, int lda,
    float *B, int ldb,
    float BETA,
    float *C, int ldc);

#endif	// __SDS_GEMM_H__
