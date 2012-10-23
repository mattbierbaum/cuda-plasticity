/* plasticity.h
 */
#ifndef _PLASTICITY_H_
#define _PLASTICITY_H_

#define NUM_COMP 9    
#define NUM_SIG_COMP 9 

#define TVDstep simpleTVD
#define PYTHON_COMPATIBILITY_DIVIDE
#define CONTINUE_RUN

#define mu 0.5  
#define nu 0.3 

#define theta 1.0

#define DOUBLE
#define LLF

#include "EXTENDED_HEADER"

// From here on defines data type related definitions
#ifndef DOUBLE
#define ReadMatrixFileFunc ReadMatrixFile
#define PRECISION_STR "sp"
#define data_type float
#define cdata_type cuComplex
#define init_cdata(x,y) make_cuFloatComplex(x,y)
#define FORWARD_FFT CUFFT_R2C
#define BACKWARD_FFT CUFFT_C2R
#define fft_r2c cufftExecR2C
#define fft_c2r cufftExecC2R
#define fft_dtype_r cufftReal
#define fft_dtype_c cufftComplex
#define Iamax cublasIsamax
#define axpy cublasSaxpy
#define vec_copy cublasScopy
#define ME 1.0e-8
//#define ME 1.1920929e-7
#else
#define ReadMatrixFileFunc ReadDoubleMatrixFile
#define PRECISION_STR "dp"
#define data_type double
#define cdata_type cuDoubleComplex
#define init_cdata(x,y) make_cuDoubleComplex(x,y)
#define FORWARD_FFT CUFFT_D2Z
#define BACKWARD_FFT CUFFT_Z2D
#define fft_r2c cufftExecD2Z
#define fft_c2r cufftExecZ2D
#define fft_dtype_r cufftDoubleReal
#define fft_dtype_c cufftDoubleComplex
#define Iamax cublasIdamax
#define axpy cublasDaxpy
#define vec_copy cublasDcopy
#define ME 2.2204460492503131e-16
#endif

#ifndef DIMENSION3
typedef int2 d_dim_vector;
#define Lsize(L) (L.x*L.y)
#define LKsize(L) ((L.x/2+1)*L.y)
#define GridSize(L) ((L.x-1)/TILEX+1), L.y 
#define KGridSize(L) ((L.x/2)/TILEX)+1, L.y
#define locate(f,v,idx) (*(f+((idx)*L.y+((v.y+L.y)%L.y))*L.x+((v.x+L.x)%L.x)))
#define locateop(f,v,op,d,idx) (*(f+((idx)*L.y+((v.y op d.y +L.y)%L.y))*L.x+((v.x op d.x +L.x)%L.x)))
#else
typedef int3 d_dim_vector;
#define Lsize(L) (L.x*L.y*L.z)
#define LKsize(L) ((L.x/2+1)*L.y*L.z)
#define GridSize(L) ((L.x-1)/TILEX+1), L.y*L.z
#define KGridSize(L) ((L.x/2)/TILEX)+1, L.y*L.z
#define locate(f,v,idx) (*(f+(((idx)*L.z+((v.z+L.z)%L.z))*L.y+((v.y+L.y)%L.y))*L.x+((v.x+L.x)%L.x)))
#define locateop(f,v,op,d,idx) (*(f+(((idx)*L.z+(v.z op d.z +L.z)%L.z)*L.y+((v.y op d.y +L.y)%L.y))*L.x+((v.x op d.x +L.x)%L.x)))
#endif

#endif // _PLASTICITY_H_

