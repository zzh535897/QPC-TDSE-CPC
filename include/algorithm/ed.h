#pragma once

#include <libraries/support_std.h>
#include <libraries/support_avx.h>

#include <structure/csrmat.h>		//exact diagonalization routines/linear system solver for csr matrix, by calling FEAST and DSS 

//====================================================================================================
//
//                      This file is a part of QPC Header Only Library
//
//====================================================================================================

//====================================================================================================
//			Standard Linear Algebraic Operation Wrappers.
//	Description:
//	(1) Exact diagonalization, LU factorization, etc
//
//	(2) For historical reasons, the namespace are called "impl_ed_simd". Now most of the calls 
//	have been tansferred to lapacke routines.
//
//Updated by: Zhao-Han Zhang(张兆涵)  Mar. 21th, 2023
//Updated by: Zhao-Han Zhang(张兆涵)  Dec. 31th, 2022
//
//Copyright © 2022-2023 Zhao-Han Zhang
//====================================================================================================

namespace qpc
{
	namespace impl_ed_simd
	{
		using   comp_t  =       avx128_complex;
		#ifdef support_avx3
		using   comp_v  =       avx512_complex;
		#endif

		//diagonalization routines (ordinary eigen value problem)
		template<size_t n_elvl,char opt,int mode=0>
		int	ev_dsy		(double* o,double* e)noexcept//double symmetric full
		{
			if constexpr(opt=='d')
			{
				//dsyevd is usually better for large matrix
				if constexpr(mode==0)	
				return	LAPACKE_dsyevd(LAPACK_COL_MAJOR,'V','L',n_elvl,o,n_elvl,e);//return in o is eigenvector
				if constexpr(mode==1)
				return	LAPACKE_dsyevd(LAPACK_ROW_MAJOR,'V','U',n_elvl,o,n_elvl,e);//return in o is eigenvector
			}else if constexpr(opt=='x')
			{
				if constexpr(mode==0)	
				return	LAPACKE_dsyev(LAPACK_COL_MAJOR,'V','L',n_elvl,o,n_elvl,e);//return in o is eigenvector
				if constexpr(mode==1)
				return	LAPACKE_dsyev(LAPACK_ROW_MAJOR,'V','U',n_elvl,o,n_elvl,e);//return in o is eigenvector
			}
		}	
		template<size_t n_elvl,int mode=0>
		int	ev_dsp		(double* o,double* e,double* z)noexcept//double symmetric packed
		{
			if constexpr(mode==0)
			return 	LAPACKE_dspev(LAPACK_COL_MAJOR,'V','L',n_elvl,o,e,z,n_elvl);//return in e is eigen values,z is eigen vector(in col major),o the intermediate values
			if constexpr(mode==1)
			return 	LAPACKE_dspev(LAPACK_ROW_MAJOR,'V','U',n_elvl,o,e,z,n_elvl);//return in e is eigen values,z is eigen vector(in row major),o the intermediate values
		}
		template<size_t n_elvl,int mode=0>
		int	ev_dst		(double* d,double* e,double* z)noexcept//double symmetric tridiagonal
		{
			if constexpr(mode==0)
			return 	LAPACKE_dstev(LAPACK_COL_MAJOR,'V',n_elvl,d,e,z,n_elvl);   //return in d is eigen values,z is eigen vector(in col major),e the intermediate values
			if constexpr(mode==1)
			return  LAPACKE_dstev(LAPACK_ROW_MAJOR,'U',n_elvl,d,e,z,n_elvl);   //return in d is eigen values,z is eigen vector(in row major),e the intermediate values
		}
		template<int mode=0>
		int	ev_dst		(double* d,double* e,double* z,size_t n_elvl)noexcept//double symmetric tridiagonal (dynamic length version)
		{
			if constexpr(mode==0)
                        return  LAPACKE_dstev(LAPACK_COL_MAJOR,'V',n_elvl,d,e,z,n_elvl);   //return in d is eigen values,z is eigen vector(in col major),e the intermediate values
                        if constexpr(mode==1)
                        return  LAPACKE_dstev(LAPACK_ROW_MAJOR,'U',n_elvl,d,e,z,n_elvl);   //return in d is eigen values,z is eigen vector(in row major),e the intermediate values
		}
		template<size_t n_elvl,size_t n_diag,int mode=0>
		int	ev_dsb		(double* o,double* e,double* z)noexcept//double symmetric banded
		{
			if constexpr(mode==0)
			return 	LAPACKE_dsbev(LAPACK_COL_MAJOR,'V','L',n_elvl,n_diag,o,n_diag+1,e,z,n_elvl);//return in e is eigen values,z is eigen vector(in col major),o the intermediate values
			if constexpr(mode==1)
			return  LAPACKE_dsbev(LAPACK_ROW_MAJOR,'V','U',n_elvl,n_diag,o,n_elvl  ,e,z,n_elvl);//return in e is eigen values,z is eigen vector(in row major),o the intermediate values
		}
		template<size_t n_elvl,size_t n_diag,int mode=0>
		int	ev_zhb		(comp_t* o,double* e,comp_t* z)noexcept//hermitian banded. !practically verified!
		{
			if constexpr(mode==0)
			return 	LAPACKE_zhbev(LAPACK_COL_MAJOR,'V','L',n_elvl,n_diag,o,n_diag+1,e,z,n_elvl);//return in e is eigen values,z is eigen vector(in col major),o the intermediate values
			if constexpr(mode==1)
			return 	LAPACKE_zhbev(LAPACK_ROW_MAJOR,'V','U',n_elvl,n_diag,o,n_elvl  ,e,z,n_elvl);//return in e is eigen values,z is eigen vector(in row major),o the intermediate values
		}
		//diagonalization routines (generalized eigen value problem)
		template<size_t n_elvl,size_t n_diag,int mode=0>
		int    	gv_dsp		(double* o,double* s,double* e,double* z)noexcept//double symmetric packed (generalized)
		{
			if constexpr(mode==0)
			return 	LAPACKE_dspgv(LAPACK_COL_MAJOR,1,'V','L',n_elvl,o,s,e,z,n_elvl);//return in e is eigen values,z is eigen vector(in col major),o&s the intermediate values
			if constexpr(mode==1)
			return  LAPACKE_dspgv(LAPACK_ROW_MAJOR,1,'V','U',n_elvl,o,s,e,z,n_elvl);//return in e is eigen values,z is eigen vector(in row major),o&s the intermediate values
		}
		template<size_t n_elvl,size_t o_diag,size_t s_diag=o_diag,int mode=0>
		int	gv_dsb		(double* o,double* s,double* e,double* z)noexcept//double symmetric banded (generalized). !practically verified!
		{
			if constexpr(mode==0)
			return 	LAPACKE_dsbgv(LAPACK_COL_MAJOR,'V','L',n_elvl,o_diag,s_diag,o,o_diag+1,s,s_diag+1,e,z,n_elvl);//return in e is eigen values,z is eigen vector(in col major),o&s the intermediate values
			if constexpr(mode==1)
			return  LAPACKE_dsbgv(LAPACK_ROW_MAJOR,'V','U',n_elvl,o_diag,s_diag,o,n_elvl  ,s,n_elvl  ,e,z,n_elvl);//return in e is eigen values,z is eigen vector(in row major),o&s the intermediate values
		}
		//linear system routines
		template<size_t n_elvl>
		int	sv_dgf		(double* a,double* b,MKL_INT* i)noexcept//double general full
		{
			return 	LAPACKE_dgesv(LAPACK_ROW_MAJOR,n_elvl,1,a,n_elvl,i,b,1);
		}
		template<size_t n_elvl>
                int     sv_zgf          (comp_t* a,comp_t* b,MKL_INT* i)noexcept//complex general full
                {
                        return  LAPACKE_zgesv(LAPACK_ROW_MAJOR,n_elvl,1,a,n_elvl,i,b,1);
                }
		template<size_t n_elvl,size_t n_diag>
		int	sv_dgb		(double* a,double* b,MKL_INT* p)noexcept//double general banded
		{
			return 	LAPACKE_dgbsv(LAPACK_ROW_MAJOR,n_elvl,n_diag,n_diag,1,a,3*n_diag+1,p,b,1);//lda == n_diag*3+1!
		}
		template<size_t n_elvl,size_t n_diag,int mode=0>
		int	sv_zgb		(comp_t* a,comp_t* b,MKL_INT* p)noexcept//complex general banded
		{
			//mkl mannual is wrong, lda should be n_elvl. the first n_elvl*n_diag is used as temp. in storage(where row is innermost in the figure)
			// X   X   X   X   X   X   ...
			// 0   0   A13 A24 A35 A46 ...
			// 0   A12 A23 A34 A45 A56
			// A11 A22 A33 A44 A55 A66
			// A21 A32 A43 A54 A65 A76
			// ... 
			if constexpr(mode==0)
			return 	LAPACKE_zgbsv(LAPACK_ROW_MAJOR,n_elvl,n_diag,n_diag,1,a,n_elvl,p,b,1);
			//the first n_diag of each line is used as temp. in storage(where row is innermost in the figure)
			// X   0   0   A11 A21 A31 ...
			// X   0   A12 A22 A32 A42 ...
			// X   A13 A23 A33 A43 A53
			// X   A24 A34 A44 A54 A64 
			else if constexpr(mode==1)
			return 	LAPACKE_zgbsv(LAPACK_COL_MAJOR,n_elvl,n_diag,n_diag,1,a,3*n_diag+1,p,b,n_elvl);//the first n_diag of each col is used as temp
		}
		template<size_t n_elvl>
		int	sv_dsf		(double* a,double* b)noexcept//double symmetric positive full
		{
			return 	LAPACKE_dposv(LAPACK_ROW_MAJOR,'U',n_elvl,1,a,n_elvl,b,1);
		}
		template<size_t n_elvl>
                int     sv_zsf          (comp_t* a,comp_t* b)noexcept//complex hermitian positive full
                {
                        return  LAPACKE_zposv(LAPACK_ROW_MAJOR,'U',n_elvl,1,a,n_elvl,b,1);
                }
		template<size_t n_elvl>
                int     sv_dsp          (double* a,double* b)noexcept//double symmetric positive packed
                {
                        return  LAPACKE_dppsv(LAPACK_ROW_MAJOR,'U',n_elvl,1,a,b,1);
                }
		template<size_t n_elvl>
		int     sv_zsp		(comp_t* a,comp_t* b)noexcept//complex symmetric positive packed
		{
			return 	LAPACKE_zppsv(LAPACK_ROW_MAJOR,'U',n_elvl,1,a,b,1);
		}
		
		//linear system routines (if factorization is done)
		template<size_t n_elvl,size_t n_diag>
		int	fsv_dsb		(double* a,double* b)noexcept//double symmetric positive banded, a is the cholesky L. !practically verified!
		{
			return 	LAPACKE_dpbtrs(LAPACK_COL_MAJOR,'L',n_elvl,n_diag,1,a,n_diag+1,b,n_elvl);
		}
		template<size_t n_elvl,size_t n_diag>
		int	fsv_zsb		(comp_t* a,comp_t* b)noexcept//complex hermitian positive banded, a is the cholesky L.
		{
			return 	LAPACKE_zpbtrs(LAPACK_COL_MAJOR,'L',n_elvl,n_diag,1,a,n_diag+1,b,n_elvl);
		}
		//lu factorization& ldlt factorization &cholesky factorization
		template<size_t n_elvl,size_t n_diag,bool clear=1>
		int	lu_dsb		(double* a)noexcept//double symmetric positive banded. !practically verified!
		{
			int info= LAPACKE_dpbtrf(LAPACK_COL_MAJOR,'L',n_elvl,n_diag,a,n_diag+1);
			if constexpr(clear)//last ones are padded with zero
			{
				double*	_a=a+n_elvl*(n_diag+1ul)-1ul;
				for(size_t j=0;j<n_diag  ;++j)
				for(size_t i=0;i<n_diag-j;++i)
				_a[-(j*(n_diag+1ul)+i)]=0.0;
			}
			return	info;
		}
		template<size_t n_elvl,size_t n_diag,bool clear=1>
		int     lu_zhb		(comp_t* a)noexcept//complex hermitian positive banded
		{
			int info= LAPACKE_zpbtrf(LAPACK_COL_MAJOR,'L',n_elvl,n_diag,a,n_diag+1);
			if constexpr(clear)//last ones are padded with zero
			{
				comp_t* _a=a+n_elvl*(n_diag+1ul)-1ul;
	                        for(size_t j=0;j<n_diag  ;++j)
        	                for(size_t i=0;i<n_diag-j;++i)
                	        _a[-(j*(n_diag+1ul)+i)]=0.0;
			}
			return	info;
		}
		template<size_t n_elvl,size_t n_diag>
		int	lu_zsb		(comp_t* a)noexcept//complex symmetric banded. !!practically verified!!
		{
			int info=0;
			constexpr size_t n_elem	=	n_diag+1ul;
			//kernel
			#ifdef support_avx3
			auto	loop_kernel	=	[&info](comp_t* _a,size_t kmax)
			{
				for(size_t k=1ul;k<kmax;++k)
				{
					comp_t*	as = _a-n_diag*k;
 					comp_t aa = as[0]*as[-k];
 					for(size_t i=0ul;i<n_elem-k;++i)
					_a[i]	-=	aa*as[i];
				}
				if(norm(*_a)<1e-31)info++;

				if constexpr(n_diag==4ul)
				{
					avx512_complex _zmm1 = {_mm512_broadcast_f64x2(1./(*_a))};
					avx512_complex _zmm2 = {_mm512_loadu_pd(++_a)};
					_mm512_storeu_pd(_a, _zmm1 * _zmm2);
				}else
				if constexpr(n_diag==8ul)
				{
					avx512_complex _zmm1 = {_mm512_broadcast_f64x2(1./(*_a))};
					avx512_complex _zmm2 = {_mm512_loadu_pd(++_a)};
					avx512_complex _zmm3 = {_mm512_loadu_pd(4+_a)};
					_mm512_storeu_pd(_a  , _zmm1 * _zmm2);
					_mm512_storeu_pd(_a+4, _zmm1 * _zmm3);
				}else
				{
					comp_t _a0 = 1./(*_a); 
					for(size_t i=1ul;i<n_elem;++i)
					{
						_a[i]	*=	_a0;
					}
				}
			};
			#else
			auto	loop_kernel	=	[&info](comp_t* _a,size_t kmax)
			{
				for(size_t k=1ul;k<kmax;++k)
				{
					comp_t*	as = _a-n_diag*k;
 					comp_t aa = as[0]*as[-k];
 					for(size_t i=0ul;i<n_elem-k;++i)
					_a[i]	-=	aa*as[i];
				}
				if(norm(*_a)<1e-31)info++;

				{
					comp_t _a0 = 1./(*_a); 
					for(size_t i=1ul;i<n_elem;++i)
					{
						_a[i]	*=	_a0;
					}
				}
			};
			#endif
			//first n_diag
			for(size_t j=0;j<n_elem;++j)
			{
				loop_kernel(a+j*n_elem,1ul+j);
			}
			//middle case
			for(size_t j=n_elem;j<n_elvl-n_diag;++j)
			{
				loop_kernel(a+j*n_elem,n_elem);
			}
			//last n_diag	
			for(size_t j=n_elvl-n_diag;j<n_elvl;++j)
			{
				for(size_t i=n_elvl-j;i<n_elem;++i)//pad with zero
				{
					a[j*n_elem+i]=0.0;
				}
				loop_kernel(a+j*n_elem,n_elem);
			}
			return 	info;
		}
		template<size_t n_elvl,size_t n_indt=0>//n_indt is the indent value of b
		int	lu_dst		(double* a,double* b)noexcept//double symmetric positive tridiag. !practically verified!
		{
			return 	LAPACKE_dpttrf(n_elvl,a,b+n_indt);
		}
		template<size_t n_elvl,size_t n_indt=0>//n_indt is the indent value of b
		int	lu_zht		(double* a,comp_t* b)noexcept//complex hermitian positive tridiag
		{
			return 	LAPACKE_zpttrf(n_elvl,a,b+n_indt);
		}

		//-----------------------------------------------helper functions----------------------------------------------------------
		/*template<size_t n1_elvl,size_t m1_diag,size_t n2_elvl,size_t m2_diag,int flag=0>
		void	dsb_dsb_to_dsb	(const double* a1,const double* a2,double* a)noexcept
		{//clear up a by yourself !!
			constexpr size_t m_diag	=	n2_elvl*m1_diag+m2_diag;
			for(size_t ir=0;ir<n1_elvl;++ir)
			{
				double const*	_a1=a1+ir*(m1_diag+1ul);
				size_t const icmax=std::min(m1_diag+1ul,n1_elvl-ir);
				for(size_t jr=0;jr<n2_elvl;++jr)
				{
					double const*	_a2 =	a2+jr*(m2_diag+1ul);
					double*	_a  =	a+(ir*n2_elvl+jr)*(m_diag+1ul);
					size_t jcmax=std::min(m2_diag+1ul,n2_elvl-jr);
					size_t jcmin=std::min(m2_diag+1ul,jr+1ul);
					//on diagonal blocks (ic=0)
					{
						for(size_t jc=0;jc<jcmax;++jc)
						{
							if constexpr(flag==0)_a[jc] =_a1[0]*_a2[jc];
							if constexpr(flag==1)_a[jc]+=_a1[0]*_a2[jc];
							if constexpr(flag==2)_a[jc]-=_a1[0]*_a2[jc];
						}
					}
					//off diagonal blocks
					for(size_t ic=1;ic<icmax;++ic)
					{
						_a+=	n2_elvl;
						for(size_t jc=0;jc<jcmax;++jc)
						{
							if constexpr(flag==0)_a[jc] =_a1[ic]*_a2[jc];
							if constexpr(flag==1)_a[jc]+=_a1[ic]*_a2[jc];
							if constexpr(flag==2)_a[jc]-=_a1[ic]*_a2[jc];
						}
						for(size_t jc=1;jc<jcmin;++jc)
						{
							if constexpr(flag==0)_a[-jc] =_a1[ic]*_a2[-jc*m2_diag];
							if constexpr(flag==1)_a[-jc]+=_a1[ic]*_a2[-jc*m2_diag];
							if constexpr(flag==2)_a[-jc]-=_a1[ic]*_a2[-jc*m2_diag];
						}
					}
				}
			}	
		}//end of dsb_dsb_to_dsb

		template<size_t n1_elvl,size_t m1_diag,size_t n2_elvl,size_t m2_diag,int flag=0>
		void	dsb_dsb_to_dsp	(const double* a1,const double* a2,double* a)noexcept
		{//clear up a by yourself !!!
			auto	dspidx	=	[&](const size_t irow)
			{
				return 	(2ul*n1_elvl*n2_elvl-1ul-irow)*irow/2ul;
			};
			for(size_t ir=0;ir<n1_elvl;++ir)
			{
				double const*	_a1=a1+ir*(m1_diag+1ul);
				size_t const icmax=std::min(m1_diag+1ul,n1_elvl-ir);
				for(size_t jr=0;jr<n2_elvl;++jr)
				{
					double const* _a2 =	a2+jr*(m2_diag+1ul);
					size_t const irow = 	ir*n2_elvl+jr;
					double*	      _a  =	a+dspidx(irow);
					size_t const jcmax=std::min(m2_diag+1ul,n2_elvl-jr);
					size_t const jcmin=std::min(m2_diag+1ul,jr+1ul);
					//on diagonal blocks (ic=0)
					{
						for(size_t jc=0;jc<jcmax;++jc)
						{
							size_t const jrow = irow+jc;
							if constexpr(flag==0)_a[jrow] =_a1[0]*_a2[jc];
							if constexpr(flag==1)_a[jrow]+=_a1[0]*_a2[jc];
							if constexpr(flag==2)_a[jrow]-=_a1[0]*_a2[jc];
						}
					}
					//off diagonal blocks
					for(size_t ic=1;ic<icmax;++ic)
					{
						for(size_t jc=0;jc<jcmax;++jc)
						{
							size_t const jrow = irow+ic*n2_elvl+jc;
							if constexpr(flag==0)_a[jrow] =_a1[ic]*_a2[jc];
							if constexpr(flag==1)_a[jrow]+=_a1[ic]*_a2[jc];
							if constexpr(flag==2)_a[jrow]-=_a1[ic]*_a2[jc];
						}
						for(size_t jc=1;jc<jcmin;++jc)
						{
							size_t const jrow = irow+ic*n2_elvl-jc;
							if constexpr(flag==0)_a[jrow] =_a1[ic]*_a2[-jc*m2_diag];
							if constexpr(flag==1)_a[jrow]+=_a1[ic]*_a2[-jc*m2_diag];
							if constexpr(flag==2)_a[jrow]-=_a1[ic]*_a2[-jc*m2_diag];
						}
					}
				}
			}		
		}//end of dsb_dsb_to_dsp*/

	}//end of impl_ed_simd
}//end of qpc
