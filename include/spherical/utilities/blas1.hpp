#pragma once

#ifdef support_avx3
	#define use_manual_simd
#else
	#define use_manual_auto
//	#define use_manual_blas
#endif

//--------------------------------------------------------------------------------------------//
	template<size_t n,size_t flag,bool aligned=false>
	static	inline	void	vecd_assg_vecd_noomp	(const comp_t* src,comp_t* dst)noexcept
	{// dst op src, op can be =,+=,-=
		static_assert(flag<3);
#ifdef use_manual_simd
		if constexpr(aligned)
		{
			for(size_t j=0ul;j<(n/4ul*4ul);j+=4ul)
			{
				if constexpr(flag==0)_mm512_store_pd(dst+j,_mm512_load_pd(src+j));
				if constexpr(flag==1)_mm512_store_pd(dst+j,_mm512_add_pd(_mm512_load_pd(dst+j),_mm512_load_pd(src+j)));
				if constexpr(flag==2)_mm512_store_pd(dst+j,_mm512_sub_pd(_mm512_load_pd(dst+j),_mm512_load_pd(src+j))); 
			}
		}else
		{
			for(size_t j=0ul;j<(n/4ul*4ul);j+=4ul)
			{
				if constexpr(flag==0)_mm512_storeu_pd(dst+j,_mm512_loadu_pd(src+j));
				if constexpr(flag==1)_mm512_storeu_pd(dst+j,_mm512_add_pd(_mm512_loadu_pd(dst+j),_mm512_loadu_pd(src+j)));
				if constexpr(flag==2)_mm512_storeu_pd(dst+j,_mm512_sub_pd(_mm512_loadu_pd(dst+j),_mm512_loadu_pd(src+j))); 
			}	
		}
		if constexpr(n%4ul!=0ul)
		{
			for(size_t j=(n/4ul*4ul);j<n;++j)
			{
				if constexpr(flag==0)dst[j] =src[j];
				if constexpr(flag==1)dst[j]+=src[j];
				if constexpr(flag==2)dst[j]-=src[j];
			}
		}
#else
#ifdef use_manual_auto

		for(size_t i=0;i<n;++i)
		{
			if constexpr(flag==0)
			{	
				dst[i][0] = src[i][0];
				dst[i][1] = src[i][1];
			}
			if constexpr(flag==1)
			{
				dst[i][0] += src[i][0];
				dst[i][1] += src[i][1];
			}
			if constexpr(flag==2)
			{
				dst[i][0] -= src[i][0];
				dst[i][1] -= src[i][1];
			}
		}
#else	
		if constexpr(flag==0)cblas_zcopy(n,src,1,dst,1);
		if constexpr(flag==1)cblas_daxpy(2*n, 1.0,(double*)src,1,(double*)dst,1);
		if constexpr(flag==2)cblas_daxpy(2*n,-1.0,(double*)src,1,(double*)dst,1);
#endif
#endif
	}//end of vecd_assg_vecd_noomp

	template<size_t n,size_t flag,bool aligned=false>
	static	inline	void	vecd_assg_vecd_noomp	(const comp_t* src,comp_t* dst,const comp_t& mul)noexcept
	{// dst op src*mul, op can be =,+=,-=
		static_assert(flag<3);
#ifdef use_manual_simd
		comp_v	_mul	=	{_mm512_broadcast_f64x2(mul)};
		if constexpr(aligned)
		{
			for(size_t j=0ul;j<(n/4ul*4ul);j+=4ul)
			{
				if constexpr(flag==0)_mm512_store_pd(dst+j,comp_v{_mm512_load_pd(src+j)}*_mul);
				if constexpr(flag==1)_mm512_store_pd(dst+j,comp_v{_mm512_load_pd(dst+j)}+comp_v{_mm512_load_pd(src+j)}*_mul);
				if constexpr(flag==2)_mm512_store_pd(dst+j,comp_v{_mm512_load_pd(dst+j)}-comp_v{_mm512_load_pd(src+j)}*_mul);
			}	
		}else
		{
			for(size_t j=0ul;j<(n/4ul*4ul);j+=4ul)
			{
				if constexpr(flag==0)_mm512_storeu_pd(dst+j,comp_v{_mm512_loadu_pd(src+j)}*_mul);
				if constexpr(flag==1)_mm512_storeu_pd(dst+j,comp_v{_mm512_loadu_pd(dst+j)}+comp_v{_mm512_loadu_pd(src+j)}*_mul);
				if constexpr(flag==2)_mm512_storeu_pd(dst+j,comp_v{_mm512_loadu_pd(dst+j)}-comp_v{_mm512_loadu_pd(src+j)}*_mul);
			}
		}
		if constexpr(n%4ul!=0ul)
		{
			for(size_t j=(n/4ul*4ul);j<n;++j)
			{
				if constexpr(flag==0)dst[j] =src[j]*mul;
				if constexpr(flag==1)dst[j]+=src[j]*mul;
				if constexpr(flag==2)dst[j]-=src[j]*mul;
			}
		}

#else
#ifdef use_manual_auto
		if constexpr(flag==0) for(size_t i=0;i<n;++i)dst[i] = src[i]*mul;
		if constexpr(flag==1) for(size_t i=0;i<n;++i)dst[i]+= src[i]*mul;
		if constexpr(flag==2) for(size_t i=0;i<n;++i)dst[i]-= src[i]*mul;
#else
		if constexpr(flag==0)
		{
			for(size_t i=0;i<n;++i)dst[i] = src[i]*mul;
		}else
		if constexpr(flag==1)
		{
			cblas_zaxpy(n,&mul,src,1,dst,1);
		}else 
		if constexpr(flag==2)
		{
			comp_t negmul = mul*(-1.);
			cblas_zaxpy(n,&negmul,src,1,dst,1);
		}	
#endif 
#endif
	}//end of vecd_assg_vecd_noomp

	template<size_t flag,bool aligned=false>
	static	inline	void	vecd_assg_vecd_noomp	(const size_t n,const comp_t* src,comp_t* dst,const comp_t& mul)noexcept
	{// dst op src*mul, op can be =,+=,-=
		static_assert(flag<3);
#ifdef use_manual_simd
		comp_v	_mul	=	{_mm512_broadcast_f64x2(mul)};
		if constexpr(aligned)
		{
			for(size_t j=0ul;j<(n/4ul*4ul);j+=4ul)
			{
				if constexpr(flag==0)_mm512_store_pd(dst+j,comp_v{_mm512_load_pd(src+j)}*_mul);
				if constexpr(flag==1)_mm512_store_pd(dst+j,comp_v{_mm512_load_pd(dst+j)}+comp_v{_mm512_load_pd(src+j)}*_mul);
				if constexpr(flag==2)_mm512_store_pd(dst+j,comp_v{_mm512_load_pd(dst+j)}-comp_v{_mm512_load_pd(src+j)}*_mul);
			}	
		}else
		{
			for(size_t j=0ul;j<(n/4ul*4ul);j+=4ul)
			{
				if constexpr(flag==0)_mm512_storeu_pd(dst+j,comp_v{_mm512_loadu_pd(src+j)}*_mul);
				if constexpr(flag==1)_mm512_storeu_pd(dst+j,comp_v{_mm512_loadu_pd(dst+j)}+comp_v{_mm512_loadu_pd(src+j)}*_mul);
				if constexpr(flag==2)_mm512_storeu_pd(dst+j,comp_v{_mm512_loadu_pd(dst+j)}-comp_v{_mm512_loadu_pd(src+j)}*_mul);
			}
		}
		for(size_t j=(n/4ul*4ul);j<n;++j)
		{
			if constexpr(flag==0)dst[j] =src[j]*mul;
			if constexpr(flag==1)dst[j]+=src[j]*mul;
			if constexpr(flag==2)dst[j]-=src[j]*mul;
		}

#else
#ifdef use_manual_auto
		if constexpr(flag==0) for(size_t i=0;i<n;++i)dst[i] = src[i]*mul;
		if constexpr(flag==1) for(size_t i=0;i<n;++i)dst[i]+= src[i]*mul;
		if constexpr(flag==2) for(size_t i=0;i<n;++i)dst[i]-= src[i]*mul;

#else
		if constexpr(flag==0)
		{
			for(size_t i=0;i<n;++i)dst[i] = src[i]*mul;
		}else
		if constexpr(flag==1)
		{
			cblas_zaxpy(n,&mul,src,1,dst,1);
		}else 
		if constexpr(flag==2)
		{
			comp_t negmul = mul*(-1.);
			cblas_zaxpy(n,&negmul,src,1,dst,1);
		}
#endif
#endif
	}//end of vecd_assg_vecd_noomp

	template<size_t n,size_t flag,bool aligned=false>
        static  inline  void    vecd_assg_vecd_noomp    (const comp_t* src,comp_t* dst,const double& mul)noexcept
        {// dst op src*mul, op can be =,+=,-=
                static_assert(flag<3);
#ifdef use_manual_simd
		if constexpr(aligned)
		{
			auto _mul = _mm512_set1_pd(mul);
			for(size_t i=0ul;i<n*2ul;i+=8ul)
			{
				if constexpr(flag==0)_mm512_store_pd(i+(double*)dst, _mm512_mul_pd(  _mul,_mm512_load_pd(i+(double*)src)));
				if constexpr(flag==1)_mm512_store_pd(i+(double*)dst, _mm512_fmadd_pd(_mul,_mm512_load_pd(i+(double*)src),_mm512_load_pd(i+(double*)dst)));
				if constexpr(flag==2)_mm512_store_pd(i+(double*)dst,-_mm512_fmsub_pd(_mul,_mm512_load_pd(i+(double*)src),_mm512_load_pd(i+(double*)dst)));
			}
		}else
		{//the compiler's auto vectorization are almost as good as manual codes
			const double* _src = (const double*)src;
			      double* _dst = (      double*)dst;
			for(size_t i=0ul;i<n*2ul;++i)
			{
				if constexpr(flag==0)_dst[i] =_src[i]*mul;
				if constexpr(flag==1)_dst[i]+=_src[i]*mul;
				if constexpr(flag==2)_dst[i]-=_src[i]*mul;
			}
                }

#else
#ifdef use_manual_auto
		if constexpr(flag==0) 
		{
			for(size_t i=0;i<n;++i) 
			{
				dst[i][0] = src[i][0]*mul;
				dst[i][1] = src[i][1]*mul;
			}
		}else 
		if constexpr(flag==1)
		{
			for(size_t i=0;i<n;++i)
			{
				dst[i][0] += src[i][0]*mul;
				dst[i][1] += src[i][1]*mul;
			}
		}else
		if constexpr(flag==2)
		{
			for(size_t i=0;i<n;++i)
			{
				dst[i][0] -= src[i][0]*mul;
				dst[i][1] -= src[i][1]*mul;
			}
		}
#else
		if constexpr(flag==0)
		{
			for(size_t i=0;i<n;++i) 
			{
				dst[i][0] = src[i][0]*mul;
				dst[i][1] = src[i][1]*mul;
			}
		}else if constexpr(flag==1) cblas_daxpy(2*n, mul,(double*)src,1,(double*)dst,1);
		 else if constexpr(flag==2) cblas_daxpy(2*n,-mul,(double*)src,1,(double*)dst,1);
#endif
#endif
        }//end of vecd_assg_vecd_noomp

	template<size_t n,size_t flag=0>
	static	inline	void	vecd_assg_vecd		(const comp_t* src,comp_t* dst)noexcept
	{// dst op src, op can be =,+=,-=
		static_assert(flag<3);
#ifdef use_manual_simd
		#pragma omp for //IMPROVE ME. add nowait?
		for(size_t j=0ul;j<(n/4ul*4ul);j+=4ul)
		{
			if constexpr(flag==0)_mm512_storeu_pd(dst+j,_mm512_loadu_pd(src+j));
			if constexpr(flag==1)_mm512_storeu_pd(dst+j,_mm512_loadu_pd(dst+j) + _mm512_loadu_pd(src+j));
			if constexpr(flag==2)_mm512_storeu_pd(dst+j,_mm512_loadu_pd(dst+j) - _mm512_loadu_pd(src+j)); 
		}
		if constexpr(n%4ul!=0ul)
		{
			#pragma omp master
			for(size_t j=(n/4ul*4ul);j<n;++j)
			{
				if constexpr(flag==0)dst[j] =src[j];
				if constexpr(flag==1)dst[j]+=src[j];
				if constexpr(flag==2)dst[j]-=src[j];
			}
			#pragma omp barrier
		}
#else
		#pragma omp for simd
		for(size_t i=0;i<n;++i)
		{
			if constexpr(flag==0)
			{
				dst[i][0] = src[i][0];
				dst[i][1] = src[i][1];
			}else
			if constexpr(flag==1)
			{
				dst[i][0] += src[i][0];
				dst[i][1] += src[i][1];
			}else
			if constexpr(flag==2)
			{
				dst[i][0] -= src[i][0];
				dst[i][1] -= src[i][1];
			}
		}
#endif
	}//end of vecd_assg_vecd

	template<size_t n,size_t flag,class mult_t>
	static	inline	void	vecd_assg_vecd		(const comp_t* src,comp_t* dst,const mult_t& mul)noexcept
	{// dst op src*mul, op can be =,+=,-=
		static_assert(flag<3);
#ifdef use_manual_simd
		#pragma omp for
		for(size_t j=0ul;j<(n/4ul*4ul);j+=4ul)
		{
			if constexpr(flag==0)avx512_dumpu(avx512_loadu(src,j)*mul,dst,j);
			if constexpr(flag==1)avx512_dumpu(avx512_loadu(dst,j)+(mul*avx512_loadu(src,j)),dst,j);
			if constexpr(flag==2)avx512_dumpu(avx512_loadu(dst,j)-(mul*avx512_loadu(src,j)),dst,j);
		}
		if constexpr(n%4ul!=0ul)
		{
			#pragma omp master
			for(size_t j=(n/4ul*4ul);j<n;++j)
			{
				if constexpr(flag==0)dst[j] =src[j]*mul;
				if constexpr(flag==1)dst[j]+=src[j]*mul;
				if constexpr(flag==2)dst[j]-=src[j]*mul;
			}
			#pragma omp barrier
		}
#else
		#pragma omp for
		for(size_t i=0;i<n;++i)
		{
			if constexpr(flag==0)dst[i] =src[i]*mul;
			if constexpr(flag==1)dst[i]+=src[i]*mul;
			if constexpr(flag==2)dst[i]-=src[i]*mul;
		}
#endif
	}//end of vecd_assg_vecd

	//---------------------------------------------------------------------------------------------------------------------------------------
	//						   vector inner product
	//---------------------------------------------------------------------------------------------------------------------------------------
	template<size_t n,bool aligned=false>
	static	inline	comp_t	vecd_proj_vecd	(const comp_t* lhs,const comp_t* rhs)noexcept
	{//return <lhs|rhs>
#ifdef use_manual_simd
		comp_v	sum = 	zero<comp_v>;
		if constexpr(aligned)
		{
			for(size_t j=0ul;j<(n/4ul*4ul);j+=4ul)
			{
				sum+=	avx_mulc(comp_v{_mm512_load_pd(rhs+j)},
						 comp_v{_mm512_load_pd(lhs+j)});
			}
		}else
		{
			for(size_t j=0ul;j<(n/4ul*4ul);j+=4ul)
			{
				sum+=	avx_mulc(comp_v{_mm512_loadu_pd(rhs+j)},
						 comp_v{_mm512_loadu_pd(lhs+j)});
			}
		}
		comp_t	ret =	{sum[0]+sum[2]+sum[4]+sum[6],sum[1]+sum[3]+sum[5]+sum[7]};
		if constexpr(n%4ul!=0ul)
		for(size_t j=(n/4ul*4ul);j<n;++j)
		{
			ret+=	avx_mulc(rhs[j],lhs[j]);
		}
		return 	ret;
#else
		comp_t 	ret;
		cblas_zdotc_sub(n,lhs,1,rhs,1,&ret);
		return	ret;
#endif
	}//end of vecd_proj_vecd (comp,comp)

	template<bool aligned=false>
	static	inline	comp_t	vecd_proj_vecd	(const size_t n,const comp_t* lhs,const comp_t* rhs)noexcept
	{//return <lhs|rhs>
#ifdef use_manual_simd
		comp_v	sum = 	zero<comp_v>;
		if constexpr(aligned)
		{
			for(size_t j=0ul;j<(n/4ul*4ul);j+=4ul)
			{
				sum+=	avx_mulc(comp_v{_mm512_load_pd(rhs+j)},
						 comp_v{_mm512_load_pd(lhs+j)});
			}
		}else
		{
			for(size_t j=0ul;j<(n/4ul*4ul);j+=4ul)
			{
				sum+=	avx_mulc(comp_v{_mm512_loadu_pd(rhs+j)},
						 comp_v{_mm512_loadu_pd(lhs+j)});
			}
		}
		comp_t	ret =	{sum[0]+sum[2]+sum[4]+sum[6],sum[1]+sum[3]+sum[5]+sum[7]};
		for(size_t j=(n/4ul*4ul);j<n;++j)
		{
			ret+=	avx_mulc(rhs[j],lhs[j]);
		}
		return 	ret;
#else
		comp_t 	ret;
		cblas_zdotc_sub(n,lhs,1,rhs,1,&ret);
		return	ret;
#endif
	}//end of vecd_proj_vecd (comp,comp)

	template<size_t n>
	static	inline	comp_t	vecd_proj_vecd	(const double* lhs,const comp_t* rhs)noexcept
	{
		//auto vectorization will take good care of it
		double	re = 0.0;
		double	im = 0.0;
		for(size_t i=0;i<n;++i) 
		{
			re += lhs[i] * rhs[i][0];
			im += lhs[i] * rhs[i][1];
		}
		return	comp_t{re,im};
	}//end of vecd_proj_vecd (real,comp)

#ifdef support_avx3
	#undef use_manual_simd
#else
	#undef use_manual_auto
//	#undef use_manual_blas
#endif
