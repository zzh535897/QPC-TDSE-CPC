#pragma once

//do not include this class before support_avx.h

#ifdef support_avx1
        #define MKL_Complex16 qpc::avx128_complex
#else
        #error this code only works when avx is supported 
#endif

#include <mkl.h>

namespace qpc
{
	//=====================================================================================================
	template<auto avxl> static inline auto	pow	(realnum<avxl> const& x,realnum<avxl> const& y)noexcept
	{
		realnum<avxl> z;
		vdPow(avx_dvecn<avxl>::n,(const double*)&x,(const double*)&y,(double*)&z);
		return 	z;
	}
	template<auto avxl> static inline auto  atan2   (realnum<avxl> const& x,realnum<avxl> const& y)noexcept
        {
                realnum<avxl> z;
                vdAtan2(avx_dvecn<avxl>::n,(const double*)&x,(const double*)&y,(double*)&z);
                return  z;
        }
	template<auto avxl> static inline auto	powx	(realnum<avxl> const& x,double const y)noexcept
	{
		realnum<avxl> z;
		vdPowx(avx_dvecn<avxl>::n,(const double*)&x,y,(double*)&z);	
		return 	z;
	}
	template<auto avxl> static inline auto	sqrt    (realnum<avxl> const& x)noexcept
        {
                realnum<avxl> z;
                vdSqrt(avx_dvecn<avxl>::n,(const double*)&x,(double*)&z);
                return  z;
        }
	template<auto avxl> static inline auto  cbrt    (realnum<avxl> const& x)noexcept
        {
                realnum<avxl> z;
                vdCbrt(avx_dvecn<avxl>::n,(const double*)&x,(double*)&z);
                return  z;
        }
	template<auto avxl> static inline auto  isqrt   (realnum<avxl> const& x)noexcept
        {
                realnum<avxl> z;
                vdInvSqrt(avx_dvecn<avxl>::n,(const double*)&x,(double*)&z);
                return  z;
        }
	template<auto avxl> static inline auto  icbrt   (realnum<avxl> const& x)noexcept
        {
                realnum<avxl> z;
                vdInvCbrt(avx_dvecn<avxl>::n,(const double*)&x,(double*)&z);
                return  z;
        }
	template<auto avxl> static inline auto  pow32   (realnum<avxl> const& x)noexcept
        {
                realnum<avxl> z;
                vdPow3o2(avx_dvecn<avxl>::n,(const double*)&x,(double*)&z);
                return  z;
        }
	template<auto avxl> static inline auto  pow23   (realnum<avxl> const& x)noexcept
	{
		realnum<avxl> z;
		vdPow2o3(avx_dvecn<avxl>::n,(const double*)&x,(double*)&z);
		return	z;
	}
	template<auto avxl> static inline auto  exp    	(realnum<avxl> const& x)noexcept
        {
                realnum<avxl> z;
                vdExp(avx_dvecn<avxl>::n,(const double*)&x,(double*)&z);
                return  z;
        }
	template<auto avxl> static inline auto  log    	(realnum<avxl> const& x)noexcept
        {
                realnum<avxl> z;
                vdLn(avx_dvecn<avxl>::n,(const double*)&x,(double*)&z);
                return  z;
        }
	template<auto avxl> static inline auto  log10	(realnum<avxl> const& x)noexcept
        {
                realnum<avxl> z;
                vdLog10(avx_dvecn<avxl>::n,(const double*)&x,(double*)&z);
                return  z;
        }
	template<auto avxl> static inline auto  sin     (realnum<avxl> const& x)noexcept
        {
                realnum<avxl> z;
                vdSin(avx_dvecn<avxl>::n,(const double*)&x,(double*)&z);
                return  z;
        }
	template<auto avxl> static inline auto  cos     (realnum<avxl> const& x)noexcept
        {
                realnum<avxl> z;
                vdCos(avx_dvecn<avxl>::n,(const double*)&x,(double*)&z);
                return  z;
        }
	template<auto avxl> static inline auto  tan     (realnum<avxl> const& x)noexcept
        {
                realnum<avxl> z;
                vdTan(avx_dvecn<avxl>::n,(const double*)&x,(double*)&z);
                return  z;
        }
	template<auto avxl> static inline auto  asin    (realnum<avxl> const& x)noexcept
        {
                realnum<avxl> z;
                vdAsin(avx_dvecn<avxl>::n,(const double*)&x,(double*)&z);
                return  z;
        }
	template<auto avxl> static inline auto  acos    (realnum<avxl> const& x)noexcept
        {
                realnum<avxl> z;
                vdAcos(avx_dvecn<avxl>::n,(const double*)&x,(double*)&z);
                return  z;
        }
	template<auto avxl> static inline auto  atan    (realnum<avxl> const& x)noexcept
        {
                realnum<avxl> z;
                vdAtan(avx_dvecn<avxl>::n,(const double*)&x,(double*)&z);
                return  z;
        }
	template<auto avxl> static inline auto  sinh    (realnum<avxl> const& x)noexcept
        {
                realnum<avxl> z;
                vdSinh(avx_dvecn<avxl>::n,(const double*)&x,(double*)&z);
                return  z;
        }
	template<auto avxl> static inline auto  cosh    (realnum<avxl> const& x)noexcept
        {
                realnum<avxl> z;
                vdCosh(avx_dvecn<avxl>::n,(const double*)&x,(double*)&z);
                return  z;
        }
	template<auto avxl> static inline auto  tanh    (realnum<avxl> const& x)noexcept
        {
                realnum<avxl> z;
                vdTanh(avx_dvecn<avxl>::n,(const double*)&x,(double*)&z);
                return  z;
        }
	template<auto avxl> static inline auto  asinh    (realnum<avxl> const& x)noexcept
        {
                realnum<avxl> z;
                vdAsinh(avx_dvecn<avxl>::n,(const double*)&x,(double*)&z);
                return  z;
        }
	template<auto avxl> static inline auto  acosh    (realnum<avxl> const& x)noexcept
        {
                realnum<avxl> z;
                vdAcosh(avx_dvecn<avxl>::n,(const double*)&x,(double*)&z);
                return  z;
        }
	template<auto avxl> static inline auto  atanh    (realnum<avxl> const& x)noexcept
        {
                realnum<avxl> z;
                vdAtanh(avx_dvecn<avxl>::n,(const double*)&x,(double*)&z);
                return  z;
        }
	//=====================================================================================================
	template<auto avxl> static inline auto  pow    	(complex<avxl> const& x,complex<avxl> const& y)noexcept
        {
                complex<avxl> z;
                vzPow(avx_cvecn<avxl>::n,(const MKL_Complex16*)&x,(const MKL_Complex16*)&y,(MKL_Complex16*)&z);
                return  z;
        }
	template<auto avxl> static inline auto	abs	(complex<avxl> const& x)noexcept
	{
		complex<avxl> z		=	avx_normc(x);
		vzSqrt(avx_cvecn<avxl>::n,(const MKL_Complex16*)&z,(MKL_Complex16*)&z);
		return	z;
	}
	template<auto avxl> static inline auto 	arg	(complex<avxl> const& x)noexcept
	{
		static_assert(avx_cvecn<avxl>::n==1);
		double a;	
		vzArg(avx_cvecn<avxl>::n,(const MKL_Complex16*)&x,(double*)&a);
		return 	a;
	}
	template<auto avxl> static inline auto  conj    (complex<avxl> const& x)noexcept
        {
                complex<avxl> z;
                vzConj(avx_cvecn<avxl>::n,(const MKL_Complex16*)&x,(MKL_Complex16*)&z);
                return  z;
        }
	template<auto avxl> static inline auto  sqrt    (complex<avxl> const& x)noexcept
        {
                complex<avxl> z;
                vzSqrt(avx_cvecn<avxl>::n,(const MKL_Complex16*)&x,(MKL_Complex16*)&z);
                return  z;
        }
	template<auto avxl> static inline auto  exp    	(complex<avxl> const& x)noexcept
        {
                complex<avxl> z;
                vzExp(avx_cvecn<avxl>::n,(const MKL_Complex16*)&x,(MKL_Complex16*)&z);
                return  z;
        }
	template<auto avxl> static inline auto  log    	(complex<avxl> const& x)noexcept
        {
                complex<avxl> z;
                vzLn(avx_cvecn<avxl>::n,(const MKL_Complex16*)&x,(MKL_Complex16*)&z);
                return  z;
        }
	template<auto avxl> static inline auto  log10   (complex<avxl> const& x)noexcept
        {
                complex<avxl> z;
                vzLog10(avx_cvecn<avxl>::n,(const MKL_Complex16*)&x,(MKL_Complex16*)&z);
                return  z;
        }
	template<auto avxl> static inline auto  sin    	(complex<avxl> const& x)noexcept
        {
                complex<avxl> z;
                vzSin(avx_cvecn<avxl>::n,(const MKL_Complex16*)&x,(MKL_Complex16*)&z);
                return  z;
        }
	template<auto avxl> static inline auto  cos     (complex<avxl> const& x)noexcept
        {
                complex<avxl> z;
                vzCos(avx_cvecn<avxl>::n,(const MKL_Complex16*)&x,(MKL_Complex16*)&z);
                return  z;
        }
	template<auto avxl> static inline auto  tan     (complex<avxl> const& x)noexcept
        {
                complex<avxl> z;
                vzTan(avx_cvecn<avxl>::n,(const MKL_Complex16*)&x,(MKL_Complex16*)&z);
                return  z;
        }
	template<auto avxl> static inline auto  asin     (complex<avxl> const& x)noexcept
        {
                complex<avxl> z;
                vzAsin(avx_cvecn<avxl>::n,(const MKL_Complex16*)&x,(MKL_Complex16*)&z);
                return  z;
        }
	template<auto avxl> static inline auto  acos     (complex<avxl> const& x)noexcept
        {
                complex<avxl> z;
                vzAcos(avx_cvecn<avxl>::n,(const MKL_Complex16*)&x,(MKL_Complex16*)&z);
                return  z;
        }
	template<auto avxl> static inline auto  atan     (complex<avxl> const& x)noexcept
        {
                complex<avxl> z;
                vzAtan(avx_cvecn<avxl>::n,(const MKL_Complex16*)&x,(MKL_Complex16*)&z);
                return  z;
        }
	template<auto avxl> static inline auto  sinh     (complex<avxl> const& x)noexcept
        {
                complex<avxl> z;
                vzSinh(avx_cvecn<avxl>::n,(const MKL_Complex16*)&x,(MKL_Complex16*)&z);
                return  z;
        }
	template<auto avxl> static inline auto  cosh     (complex<avxl> const& x)noexcept
        {
                complex<avxl> z;
                vzCosh(avx_cvecn<avxl>::n,(const MKL_Complex16*)&x,(MKL_Complex16*)&z);
                return  z;
        }
	template<auto avxl> static inline auto  tanh     (complex<avxl> const& x)noexcept
        {
                complex<avxl> z;
                vzTanh(avx_cvecn<avxl>::n,(const MKL_Complex16*)&x,(MKL_Complex16*)&z);
                return  z;
        }
	template<auto avxl> static inline auto  asinh    (complex<avxl> const& x)noexcept
        {
                complex<avxl> z;
                vzAsinh(avx_cvecn<avxl>::n,(const MKL_Complex16*)&x,(MKL_Complex16*)&z);
                return  z;
        }
	template<auto avxl> static inline auto  acosh    (complex<avxl> const& x)noexcept
        {
                complex<avxl> z;
                vzAcosh(avx_cvecn<avxl>::n,(const MKL_Complex16*)&x,(MKL_Complex16*)&z);
                return  z;
        }
	template<auto avxl> static inline auto  atanh    (complex<avxl> const& x)noexcept
        {
                complex<avxl> z;
                vzAtanh(avx_cvecn<avxl>::n,(const MKL_Complex16*)&x,(MKL_Complex16*)&z);
                return  z;
        }

	//===============================================================================
	template<auto avxl> static inline auto	cis	 (realnum<avxl> const& x)noexcept
	{
		complex<avxl> z;
		vzCIS(avx_dvecn<avxl>::n,(const double*)&x,(MKL_Complex16*)& z);
		return 	z;
	}
	//===============================================================================
	//v?PackI(n,src,str,dst) note str describes src
	//v?PackV(n,src,idx,dst) note idx describes src
	//v?UnpackI(n,src,dst,str) note str describes dst
	//v?UnpackV(n,src,dst,idx) note str describes dst
	//===============================================================================
}//end of qpc


//===================================
//mkl sparse library call
//===================================

namespace qpc
{
	static constexpr char mkl_sparse_status_return[][32]=
	{
		{"SPARSE_STATUS_SUCCESS"},
		{"SPARSE_STATUS_NOT_INITIALIZED"},
		{"SPARSE_STATUS_ALLOC_FAILED"},
		{"SPARSE_STATUS_INVALID_VALUE"},
		{"SPARSE_STATUS_EXECUTION_FAILED"},
		{"SPARSE_STATUS_INTERNAL_ERROR"},
		{"SPARSE_STATUS_NOT_SUPPORTED"}
	};

	const char* mkl_sparse_error_handle(sparse_status_t err)
	{
		if(err==SPARSE_STATUS_SUCCESS)		return mkl_sparse_status_return[0];
		if(err==SPARSE_STATUS_NOT_INITIALIZED)	return mkl_sparse_status_return[1];
		if(err==SPARSE_STATUS_ALLOC_FAILED)	return mkl_sparse_status_return[2];
		if(err==SPARSE_STATUS_INVALID_VALUE)	return mkl_sparse_status_return[3];
		if(err==SPARSE_STATUS_EXECUTION_FAILED)	return mkl_sparse_status_return[4];
		if(err==SPARSE_STATUS_INTERNAL_ERROR)	return mkl_sparse_status_return[5];
		if(err==SPARSE_STATUS_NOT_SUPPORTED)	return mkl_sparse_status_return[6];
		return 0;
	}

	template<class errn_t>
	void 	mkl_sparse_check(int64_t n,int64_t* ia,int64_t *ja)
	{
		sparse_struct handle;
		sparse_matrix_checker_init(&handle);
		handle.n=n;
		handle.csr_ia=ia;
		handle.csr_ja=ja;
		handle.indexing=MKL_ZERO_BASED;
		handle.matrix_structure=MKL_GENERAL_STRUCTURE;
		handle.matrix_format=MKL_CSR;
		handle.message_level=MKL_PRINT;
		handle.print_style=MKL_C_STYLE;
		if(sparse_matrix_checker(&handle)!=MKL_SPARSE_CHECKER_SUCCESS)
		{
			throw errn_t{};
		}
	}
}//end of qpc
