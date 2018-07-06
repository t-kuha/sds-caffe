#ifndef GEMV_SDS_HH__
#define GEMV_SDS_HH__

void gemv_sds(int TA, int M, int N, float ALPHA, 
    const float *A, int lda,
    const float *x, int ldx,
    float BETA, 
    float *y, int ldy);

//double gemv_get_compute_time();
#endif
