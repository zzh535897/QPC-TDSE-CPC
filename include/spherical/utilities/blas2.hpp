#pragma once

#ifdef support_avx3
        #define use_manual_simd 1
#endif

#ifdef support_avx3
//+++++++++++++++++++++++++++++++++//
//multiplication of complex numbers
//+++++++++++++++++++++++++++++++++//
	static	inline	comp_v	avx512_zmulz	(__m512d x,__m512d y)noexcept//compute x*y
	{
		__m512d         zmm2    =       _mm512_permute_pd(y,255);//yi,yi
		__m512d         zmm3    =       _mm512_permute_pd(x, 85);//xi,xr
		__m512d         zmm4    =       _mm512_mul_pd(zmm2,zmm3);       //xi*yi,xr*yi
		__m512d         zmm1    =       _mm512_permute_pd(y,  0);//yr,yr
		return  {_mm512_fmaddsub_pd(zmm1,x,zmm4)};                      //(yr,yr)*(xr,xi)-+(xi*yi,xr*yi)
	}//end of avx512_zmulz

	static 	inline	comp_v	avx512_zmulz_c	(__m512d x,__m512d y)noexcept//compute x*conj(y)
	{
		__m512d         zmm2    =       _mm512_permute_pd(y,255);//yi,yi
		__m512d         zmm3    =       _mm512_permute_pd(x, 85);//xi,xr
		__m512d         zmm4    =       _mm512_mul_pd(zmm2,zmm3);       //xi*yi,xr*yi		
		__m512d		zmm1	=	_mm512_permute_pd(y,  0);//yr,yr
		return	{_mm512_fmsubadd_pd(zmm1,x,zmm4)};                      //(yr,yr)*(xr,xi)+-(xi*yi,xr*yi)
	}//end of avx512_zmulz_c

	static	inline	comp_v	avx512_dmulz	(__m512d x,__m512d y)noexcept//compute x*y
	{
		return	{_mm512_mul_pd(x,y)};	//(xr*yr,xr*yi)
	}//end of avx512_dmulz

//+++++++++++++++++++++++++++++++++//
//could be better than zsbmv?
//+++++++++++++++++++++++++++++++++//
	template<size_t n>
	static 	inline	comp_t	symb_prj_vecd_m09	(const comp_t* mat,const comp_t* lhs,const comp_t* rhs)noexcept
	{
		comp_t	sum	=	zero<comp_t>;

		static constexpr size_t	m	 = 	9ul;
		static constexpr __m512i vindex1 =	{-18*8+16,-18*8+17,-18*7+14,-18*7+15,-18*6+12,-18*6+13,-18*5+10,-18*5+11};
		static constexpr __m512i vindex2 = 	{-18*4+ 8,-18*4+ 9,-18*3+ 6,-18*3+ 7,-18*2+ 4,-18*2+ 5,-18*1+ 2,-18*1+ 3};
		static constexpr __m512d zeros   = 	{0.,0.,0.,0.,0.,0.,0.,0.};

		int mask_low,mask_hig;

		mask_low=65535<<16; //11 11 11 11 11 11 11 11 >>>>>>
		
		for(size_t i=0;i<m;++i)
		{
			const comp_t* _mat =  mat + i*m ;
			const comp_t* _rhs =  rhs + i;
			const comp_t  _lhs =*(lhs + i);
			mask_hig = mask_low >>8;
			comp_v	tmp=	avx512_zmulz(_mm512_mask_i64gather_pd(zeros,mask_low,vindex1,_mat,8), _mm512_maskz_loadu_pd(mask_low,_rhs-8))
				+	avx512_zmulz(_mm512_mask_i64gather_pd(zeros,mask_hig,vindex2,_mat,8), _mm512_maskz_loadu_pd(mask_hig,_rhs-4))
				+	avx512_zmulz(_mm512_loadu_pd(_mat+1),_mm512_loadu_pd(_rhs+1))
				+	avx512_zmulz(_mm512_loadu_pd(_mat+5),_mm512_loadu_pd(_rhs+5));
			sum	+=	avx_mulc( avx_sum(tmp) + *_mat**_rhs, _lhs );
			mask_low>>=2;
		}

		for(size_t i=m;i<n-m;++i)
                {
                        const comp_t* _mat =  mat + i*m ;
                        const comp_t* _rhs =  rhs + i;
                        const comp_t  _lhs =*(lhs + i);

                        comp_v  tmp=   	avx512_zmulz(_mm512_i64gather_pd(vindex1,_mat,8), _mm512_loadu_pd(_rhs-8))
                                +       avx512_zmulz(_mm512_i64gather_pd(vindex2,_mat,8), _mm512_loadu_pd(_rhs-4))
                                +       avx512_zmulz(_mm512_loadu_pd(_mat+1),_mm512_loadu_pd(_rhs+1))
                                +       avx512_zmulz(_mm512_loadu_pd(_mat+5),_mm512_loadu_pd(_rhs+5));
                        sum     +=      avx_mulc( avx_sum(tmp) + *_mat**_rhs, _lhs );
                }
	
		mask_low = 65535; // >>>>>>>> 11 11 11 11 11 11 11 11
		
		for(size_t i=n-m;i<n;++i)
                {
                        const comp_t* _mat =  mat + i*m ;
                        const comp_t* _rhs =  rhs + i;
                        const comp_t  _lhs =*(lhs + i);
                        mask_hig = mask_low >> 8;

                        comp_v  tmp=    avx512_zmulz(_mm512_i64gather_pd(vindex1,_mat,8), _mm512_loadu_pd(_rhs-8))
                                +       avx512_zmulz(_mm512_i64gather_pd(vindex2,_mat,8), _mm512_loadu_pd(_rhs-4))
                                +       avx512_zmulz(_mm512_maskz_loadu_pd(mask_low,_mat+1),_mm512_maskz_loadu_pd(mask_low,_rhs+1))
                                +       avx512_zmulz(_mm512_maskz_loadu_pd(mask_hig,_mat+5),_mm512_maskz_loadu_pd(mask_hig,_rhs+5));
                        sum     +=      avx_mulc( avx_sum(tmp) + *_mat**_rhs, _lhs );
                        mask_low>>=2;
                }
		return	sum;
	}//end of m09 (complex)

	template<size_t n,size_t flag,class...mult_t>//flag 0,1 :=,+=
	static 	inline	void	symb_mul_vecd_m09	(const comp_t* mat,const comp_t* src,comp_t* dst,const mult_t&...mult)noexcept
	{
		static constexpr size_t	m	 = 	9ul;
		static constexpr __m512i vindex1 =	{-18*8+16,-18*8+17,-18*7+14,-18*7+15,-18*6+12,-18*6+13,-18*5+10,-18*5+11};
		static constexpr __m512i vindex2 = 	{-18*4+ 8,-18*4+ 9,-18*3+ 6,-18*3+ 7,-18*2+ 4,-18*2+ 5,-18*1+ 2,-18*1+ 3};
		static constexpr __m512d zeros   = 	{0.,0.,0.,0.,0.,0.,0.,0.};

		int mask_low,mask_hig;

		mask_low=65535<<16; //11 11 11 11 11 11 11 11 >>>>>>
		
		for(size_t i=0;i<m;++i)
		{
			mask_hig = mask_low >>8;
			comp_v	tmp=	avx512_zmulz(_mm512_mask_i64gather_pd(zeros,mask_low,vindex1,mat,8), _mm512_maskz_loadu_pd(mask_low,src-8))
				+	avx512_zmulz(_mm512_mask_i64gather_pd(zeros,mask_hig,vindex2,mat,8), _mm512_maskz_loadu_pd(mask_hig,src-4))
				+	avx512_zmulz(_mm512_loadu_pd(mat+1),_mm512_loadu_pd(src+1))
				+	avx512_zmulz(_mm512_loadu_pd(mat+5),_mm512_loadu_pd(src+5));
			mask_low>>=2;
			if constexpr(flag==0ul)
			*dst	 =	((avx_sum(tmp) + *mat**src)*...*mult);
			if constexpr(flag==1ul)
			*dst	+=	((avx_sum(tmp) + *mat**src)*...*mult);

			src++;dst++;mat+=m;
		}

		for(size_t i=m;i<n-m;++i)
                {
                        comp_v  tmp=   	avx512_zmulz(_mm512_i64gather_pd(vindex1,mat,8), _mm512_loadu_pd(src-8))
                                +       avx512_zmulz(_mm512_i64gather_pd(vindex2,mat,8), _mm512_loadu_pd(src-4))
                                +       avx512_zmulz(_mm512_loadu_pd(mat+1),_mm512_loadu_pd(src+1))
                                +       avx512_zmulz(_mm512_loadu_pd(mat+5),_mm512_loadu_pd(src+5));
			if constexpr(flag==0ul)
                        *dst 	=	((avx_sum(tmp) + *mat**src)*...*mult);
			if constexpr(flag==1ul)
                        *dst	+=	((avx_sum(tmp) + *mat**src)*...*mult);

			src++;dst++;mat+=m;
                }
	
		mask_low = 65535; // >>>>>>>> 11 11 11 11 11 11 11 11
		
		for(size_t i=n-m;i<n;++i)
                {
                        mask_hig = mask_low >> 8;

                        comp_v  tmp=    avx512_zmulz(_mm512_i64gather_pd(vindex1,mat,8), _mm512_loadu_pd(src-8))
                                +       avx512_zmulz(_mm512_i64gather_pd(vindex2,mat,8), _mm512_loadu_pd(src-4))
                                +       avx512_zmulz(_mm512_maskz_loadu_pd(mask_low,mat+1),_mm512_maskz_loadu_pd(mask_low,src+1))
                                +       avx512_zmulz(_mm512_maskz_loadu_pd(mask_hig,mat+5),_mm512_maskz_loadu_pd(mask_hig,src+5));
                        mask_low>>=2;
			if constexpr(flag==0ul)
                        *dst  	=     	(( avx_sum(tmp) + *mat**src)*...*mult);
			if constexpr(flag==1ul)
                        *dst	+=     	(( avx_sum(tmp) + *mat**src)*...*mult);

			src++;dst++;mat+=m;
                }
	}//end of symb_mul_vecd_m09 (complex)

	template<size_t n,size_t flag,class...mult_t>//flag 0,1 :=,+=
	static 	inline	void	symb_cmul_vecd_m09	(const comp_t* mat,const comp_t* src,comp_t* dst,const mult_t&...mult)noexcept
	{
		static constexpr size_t	m	 = 	9ul;
		static constexpr __m512i vindex1 =	{-18*8+16,-18*8+17,-18*7+14,-18*7+15,-18*6+12,-18*6+13,-18*5+10,-18*5+11};
		static constexpr __m512i vindex2 = 	{-18*4+ 8,-18*4+ 9,-18*3+ 6,-18*3+ 7,-18*2+ 4,-18*2+ 5,-18*1+ 2,-18*1+ 3};
		static constexpr __m512d zeros   = 	{0.,0.,0.,0.,0.,0.,0.,0.};

		int mask_low,mask_hig;

		mask_low=65535<<16; //11 11 11 11 11 11 11 11 >>>>>>
		
		for(size_t i=0;i<m;++i)
		{
			mask_hig = mask_low >>8;
			comp_v	tmp=	avx512_zmulz_c(_mm512_maskz_loadu_pd(mask_low,src-8),_mm512_mask_i64gather_pd(zeros,mask_low,vindex1,mat,8))
				+	avx512_zmulz_c(_mm512_maskz_loadu_pd(mask_hig,src-4),_mm512_mask_i64gather_pd(zeros,mask_hig,vindex2,mat,8))
				+	avx512_zmulz_c(_mm512_loadu_pd(src+1),_mm512_loadu_pd(mat+1))
				+	avx512_zmulz_c(_mm512_loadu_pd(src+5),_mm512_loadu_pd(mat+5));
			mask_low>>=2;
			if constexpr(flag==0ul)
			*dst	 =	((avx_sum(tmp) + *mat**src)*...*mult);
			if constexpr(flag==1ul)
			*dst	+=	((avx_sum(tmp) + *mat**src)*...*mult);

			src++;dst++;mat+=m;
		}

		for(size_t i=m;i<n-m;++i)
                {
                        comp_v  tmp=   	avx512_zmulz_c(_mm512_loadu_pd(src-8),_mm512_i64gather_pd(vindex1,mat,8))
                                +       avx512_zmulz_c(_mm512_loadu_pd(src-4),_mm512_i64gather_pd(vindex2,mat,8))
                                +       avx512_zmulz_c(_mm512_loadu_pd(src+1),_mm512_loadu_pd(mat+1))
                                +       avx512_zmulz_c(_mm512_loadu_pd(src+5),_mm512_loadu_pd(mat+5));
			if constexpr(flag==0ul)
                        *dst   	=	((avx_sum(tmp) + *mat**src)*...*mult);
			if constexpr(flag==1ul)
                        *dst   	+=	((avx_sum(tmp) + *mat**src)*...*mult);

			src++;dst++;mat+=m;
                }
	
		mask_low = 65535; // >>>>>>>> 11 11 11 11 11 11 11 11
		
		for(size_t i=n-m;i<n;++i)
                {
                        mask_hig = mask_low >> 8;

                        comp_v  tmp=    avx512_zmulz_c(_mm512_loadu_pd(src-8),_mm512_i64gather_pd(vindex1,mat,8))
                                +       avx512_zmulz_c(_mm512_loadu_pd(src-4),_mm512_i64gather_pd(vindex2,mat,8))
                                +       avx512_zmulz_c(_mm512_maskz_loadu_pd(mask_low,src+1),_mm512_maskz_loadu_pd(mask_low,mat+1))
                                +       avx512_zmulz_c(_mm512_maskz_loadu_pd(mask_hig,src+5),_mm512_maskz_loadu_pd(mask_hig,mat+5));
                        mask_low>>=2;
			if constexpr(flag==0ul)
                        *dst  	=     	(( avx_sum(tmp) + *mat**src)*...*mult);
			if constexpr(flag==1ul)
                        *dst	+=     	(( avx_sum(tmp) + *mat**src)*...*mult);

			src++;dst++;mat+=m;
                }
	}//end of symb_cmul_vecd_m09 (complex)

	template<size_t n,size_t flag,class mult_t>//flag 0,1 :=,+=
	static  inline  comp_t 	symb_prj_mul_vecd_m09	(const comp_t* mat,const comp_t* lhs,const comp_t* rhs,comp_t* dst,const mult_t& mul)noexcept
	{//z op b*A*y, op can be =,+=  also compute x'*A*y, where ' is hermitian conjugate :)
		comp_t	sum	=	zero<comp_t>;

		static constexpr size_t	m	 = 	9ul;
		static constexpr __m512i vindex1 =	{-18*8+16,-18*8+17,-18*7+14,-18*7+15,-18*6+12,-18*6+13,-18*5+10,-18*5+11};
		static constexpr __m512i vindex2 = 	{-18*4+ 8,-18*4+ 9,-18*3+ 6,-18*3+ 7,-18*2+ 4,-18*2+ 5,-18*1+ 2,-18*1+ 3};
		static constexpr __m512d zeros   = 	{0.,0.,0.,0.,0.,0.,0.,0.};

		int mask_low,mask_hig;

		mask_low=65535<<16; //11 11 11 11 11 11 11 11 >>>>>>
		
		for(size_t i=0;i<m;++i)
		{
			const comp_t* _mat =  mat + i*m ;
			const comp_t* _rhs =  rhs + i;
			const comp_t  _lhs =*(lhs + i);
			mask_hig = mask_low >>8;
			comp_v	tmp=	avx512_zmulz(_mm512_mask_i64gather_pd(zeros,mask_low,vindex1,_mat,8), _mm512_maskz_loadu_pd(mask_low,_rhs-8))
				+	avx512_zmulz(_mm512_mask_i64gather_pd(zeros,mask_hig,vindex2,_mat,8), _mm512_maskz_loadu_pd(mask_hig,_rhs-4))
				+	avx512_zmulz(_mm512_loadu_pd(_mat+1),_mm512_loadu_pd(_rhs+1))
				+	avx512_zmulz(_mm512_loadu_pd(_mat+5),_mm512_loadu_pd(_rhs+5));
			comp_t	val= 	avx_sum(tmp) + *_mat**_rhs;
			sum	+=	avx_mulc( val, _lhs);
			val	*=	mul;
			if constexpr(flag==0)
			{
				dst[i]	=	val;
			}
			if constexpr(flag==1)
			{
				#pragma omp atomic
				dst[i][0] += val[0];
				#pragma omp atomic
				dst[i][1] += val[1];
			}
			mask_low>>=2;
		}

		for(size_t i=m;i<n-m;++i)
                {
                        const comp_t* _mat =  mat + i*m ;
                        const comp_t* _rhs =  rhs + i;
                        const comp_t  _lhs =*(lhs + i);

                        comp_v  tmp=   	avx512_zmulz(_mm512_i64gather_pd(vindex1,_mat,8), _mm512_loadu_pd(_rhs-8))
                                +       avx512_zmulz(_mm512_i64gather_pd(vindex2,_mat,8), _mm512_loadu_pd(_rhs-4))
                                +       avx512_zmulz(_mm512_loadu_pd(_mat+1),_mm512_loadu_pd(_rhs+1))
                                +       avx512_zmulz(_mm512_loadu_pd(_mat+5),_mm512_loadu_pd(_rhs+5));
			comp_t	val=	avx_sum(tmp) + *_mat**_rhs;
                        sum     +=      avx_mulc( val, _lhs );
			val	*=	mul;
			if constexpr(flag==0)
			{
				dst[i]	=	val;
			}
			if constexpr(flag==1)
			{
				#pragma omp atomic
				dst[i][0] += val[0];
				#pragma omp atomic
				dst[i][1] += val[1];
			}
                }
	
		mask_low = 65535; // >>>>>>>> 11 11 11 11 11 11 11 11
		
		for(size_t i=n-m;i<n;++i)
                {
                        const comp_t* _mat =  mat + i*m ;
                        const comp_t* _rhs =  rhs + i;
                        const comp_t  _lhs =*(lhs + i);
                        mask_hig = mask_low >> 8;

                        comp_v  tmp=    avx512_zmulz(_mm512_i64gather_pd(vindex1,_mat,8), _mm512_loadu_pd(_rhs-8))
                                +       avx512_zmulz(_mm512_i64gather_pd(vindex2,_mat,8), _mm512_loadu_pd(_rhs-4))
                                +       avx512_zmulz(_mm512_maskz_loadu_pd(mask_low,_mat+1),_mm512_maskz_loadu_pd(mask_low,_rhs+1))
                                +       avx512_zmulz(_mm512_maskz_loadu_pd(mask_hig,_mat+5),_mm512_maskz_loadu_pd(mask_hig,_rhs+5));
			comp_t	val=	avx_sum(tmp) + *_mat**_rhs;
                        sum     +=      avx_mulc( val, _lhs );
			val	*=	mul;
                        mask_low>>=2;
			if constexpr(flag==0)
			{
				dst[i]	=	val;
			}
			if constexpr(flag==1)
			{
				#pragma omp atomic
				dst[i][0] += val[0];
				#pragma omp atomic
				dst[i][1] += val[1];
			}
                }
		return	sum;
	}//end of m09 (complex)
#endif

//---------------------------------------------------------------------------------------------------------------------------------------
// loop kernels for raw-written version
//---------------------------------------------------------------------------------------------------------------------------------------
	template<size_t m,class type_a,class type_x>
	static	inline	auto	symb_mul_vecd_left	(const type_a* a,const type_x* x,size_t jmin,size_t jmax)noexcept
	{
		auto t  =       zero<decltype(*a**x)>;
		constexpr size_t q=m-1ul;
		for(size_t j=jmin;j<jmax;++j)t+=a[-j*q]*x[-j];
		return 	t;
	}
	template<size_t m,class type_a,class type_x>
	static	inline	auto	symb_mulc_vecd_left	(const type_a* a,const type_x* x,size_t jmin,size_t jmax)noexcept
	{
		auto t  =       zero<decltype(*a**x)>;
		constexpr size_t q=m-1ul;
		for(size_t j=jmin;j<jmax;++j)t+=avx_mulc(x[-j],a[-j*q]);
		return 	t;
	}
	template<size_t m,class type_a,class type_x>
        static  inline  auto    symb_mul_vecd_right    	(const type_a* a,const type_x* x,size_t kmin,size_t kmax)noexcept
	{
		auto t  =       zero<decltype(*a**x)>;
		for(size_t k=kmin;k<kmax;++k)t+=a[k]*x[k];
		return	t;
	}
	template<size_t m,class type_a,class type_x>
        static  inline  auto    symb_mulc_vecd_right    (const type_a* a,const type_x* x,size_t kmin,size_t kmax)noexcept
	{
		auto t  =       zero<decltype(*a**x)>;
		for(size_t k=kmin;k<kmax;++k)t+=avx_mulc(x[k],a[k]);;
		return	t;
	}
	template<size_t m,class type_a,class type_x>
	static	inline	auto	asyb_mul_vecd_left	(const type_a* a,const type_x* x,size_t jmin,size_t jmax)noexcept
	{
		auto t  =       zero<decltype(*a**x)>;
		constexpr size_t q=m-1ul;
		for(size_t j=jmin;j<jmax;++j)t-=a[-j*q]*x[-j];
		return 	t;
	}
	template<size_t m,class type_a,class type_x>
        static  inline  auto    asyb_mul_vecd_right    	(const type_a* a,const type_x* x,size_t kmin,size_t kmax)noexcept
	{
		auto t  =       zero<decltype(*a**x)>;
		for(size_t k=kmin;k<kmax;++k)t+=a[k]   *x[ k];
		return	t;
	}
	template<size_t m,class type_a,class type_x>
	static	inline	auto	symb_mul_vecd_loop	(const type_a* a,const type_x* x,size_t jmin,size_t jmax,size_t kmin,size_t kmax)noexcept
	{//kernel of computing a*x for symmetric banded matrix a, m is number of middle&upper elements in one line
		auto t	=	zero<decltype(*a**x)>;
		constexpr size_t q=m-1ul;
		for(size_t j=jmin;j<jmax;++j)t+=a[-j*q]*x[-j];
		for(size_t k=kmin;k<kmax;++k)t+=a[k]   *x[ k];
		return 	t;	
	}
	template<size_t m,class type_a,class type_x>
	static	inline	auto	symb_mulc_vecd_loop	(const type_a* a,const type_x* x,size_t jmin,size_t jmax,size_t kmin,size_t kmax)noexcept
	{//kernel of computing conj(a)*x for symmetric banded matrix a, m is number of middle&upper elements in one line
		auto t	=	zero<decltype(*a**x)>;
		constexpr size_t q=m-1ul;
		for(size_t j=jmin;j<jmax;++j)t+=avx_mulc(x[-j],a[-j*q]);//see support_avx.h
		for(size_t k=kmin;k<kmax;++k)t+=avx_mulc(x[ k],a[ k  ]);//see support_avx.h
		return 	t;	
	}
	
	template<size_t m,class type_a,class type_x>
	static  inline  auto    asyb_mul_vecd_loop      (const type_a* a,const type_x* x,size_t jmin,size_t jmax,size_t kmin,size_t kmax)noexcept
	{//kernel of computing a*x for asymmetric banded matrix a, m is number of middle&upper elements in one line
		auto t	=	zero<decltype(*a**x)>;
		constexpr size_t q=m-1ul;
		for(size_t j=jmin;j<jmax;++j)t-=a[-j*q]*x[-j];
		for(size_t k=kmin;k<kmax;++k)t+=a[k]   *x[ k];
		return  t;
	}
	template<size_t m,class type_a,class type_x>
	static	inline	auto	herb_mul_vecd_loop	(const type_a* a,const type_x* x,size_t jmin,size_t jmax,size_t kmin,size_t kmax)noexcept
	{//kernel of computing a*x for hermitian banded matrix a, m is number of non-repeative elements in one line
		auto t  =       zero<decltype(*a**x)>;
		constexpr size_t q=m-1ul;
                for(size_t j=jmin;j<jmax;++j)t+=avx_mulc(x[-j],a[-j*q]);//only accept double,and avx complex
                for(size_t k=kmin;k<kmax;++k)t+=a[k]*x[k];
                return  t;	
	}
	template<size_t m,class type_a,class type_x>
	static  inline  auto    ltrb_mul_vecd_loop      (const type_a* a,const type_x* x,size_t jmin,size_t jmax)noexcept
	{//kernel of computing a*x for lower triangular matrix a, m is number of non-repeative elements in one line
		auto t  =       zero<decltype(*a**x)>;
		constexpr size_t q=m-1ul;
		for(size_t j=jmin;j<jmax;++j)t+=a[-j*q]*x[-j];//note a is accessed from a symmetric righthand side band matrix
		return	t;
	}
	template<size_t m,class type_a,class type_x>
	static	inline	auto	utrb_mul_vecd_loop	(const type_a* a,const type_x* x,size_t kmin,size_t kmax)noexcept
	{//kernel of computing a*x for upper triangular matrix a, m is number of non-repeative elements in one line
		auto t  =       zero<decltype(*a**x)>;
		for(size_t k=kmin;k<kmax;++k)t+=a[k]*x[k];
		return 	t;
	}
	template<size_t m,class type_a,class type_x>
	static  inline  void    ltrb_inv_vecd_loop      (const type_a* a,type_x x,type_x* y,size_t jmin,size_t jmax)noexcept
	{//kernel of computing y=a\x for lower triangular matrix a, m is number of non-repeative elements in one line
		constexpr size_t q=m-1ul;
		for(size_t j=jmin;j<jmax;++j)x-=y[-j]*a[-j*q];//note a is accessed from a symmetric righthand side band matrix
		*y= x/ *a;	
	}
	template<size_t m,class type_a,class type_x>
	static	inline	void	utrb_inv_vecd_loop	(const type_a* a,type_x x,type_x* y,size_t kmin,size_t kmax)noexcept
	{//kernel of computing y=a\x for upper triangular matrix a, m is number of non-repeative elements in one line
		for(size_t k=kmin;k<kmax;++k)x-=y[k]*a[k];
		*y= x/ *a;
	}
	template<size_t m,class type_a,class type_x>
	static	inline	void	ltrb_ldl_vecd_loop	(const type_a* a,type_x x,type_x* y,size_t jmin,size_t jmax)noexcept
	{//kernel of computing Y=L\X for lower triangular matrix L with unity diagonal entry, m is number of non-repeative elements in one line
		constexpr size_t q=m-1ul;
             	for(size_t j=jmax;j-->jmin;)x-=y[-j]*a[-j*q];//note a is accessed from a symmetric righthand side band matrix
		*y=x;
	}
	template<size_t m,class type_a,class type_x>
	static	inline	void	utrb_ldl_vecd_loop	(const type_a* a,type_x x,type_x* y,size_t kmin,size_t kmax)noexcept
	{//kernel of computing Y=(D*L^T)\X for lower triangular matrix L, thus D*L^T an upper triangular one.  L(i,i) stores D. m is number of non-repeative elements in one line
		x/=*a;
		for(size_t k=kmin;k<kmax;++k)x-=y[k]*a[k];
		*y=x;
	}

//---------------------------------------------------------------------------------------------------------------------------------------
// BLAS-LIKE EXTENSION
//---------------------------------------------------------------------------------------------------------------------------------------
	template<size_t n,size_t m,size_t flag,class type_a,class type_t>//n=n_dims,m=n_elem
	static  inline  void    symb_mul_vecd           (const type_a* a,const type_t* x,type_t* y)noexcept //corresponds to blas ?sbmv
	{//y op a*x, op can be =,+=,-= ,where a is symmetric banded matrix, x,y are dense vectors
		if constexpr(2*m>n)
		{
#ifdef use_manual_simd
			if constexpr(m==9ul && std::is_same_v<type_a,comp_t> && std::is_same_v<type_t,comp_t>)
			{
				symb_mul_vecd_m09<n,flag>(a,x,y);
			}else
#endif
			{
				for(size_t i=0;i<n-m;++i)
				{
					if constexpr(flag==0)y[i] =symb_mul_vecd_right<m>(a+i*m,x+i,0,m);
					if constexpr(flag==1)y[i]+=symb_mul_vecd_right<m>(a+i*m,x+i,0,m);
					if constexpr(flag==2)y[i]-=symb_mul_vecd_right<m>(a+i*m,x+i,0,m);
				}
				for(size_t i=n-m;i<n;++i)
				{
					if constexpr(flag==0)y[i] =symb_mul_vecd_right<m>(a+i*m,x+i,0,n-i);
					if constexpr(flag==1)y[i]+=symb_mul_vecd_right<m>(a+i*m,x+i,0,n-i);
					if constexpr(flag==2)y[i]-=symb_mul_vecd_right<m>(a+i*m,x+i,0,n-i);
				}
				for(size_t i=1;i<m;++i)
				{
					if constexpr(flag<=1)y[i]+=symb_mul_vecd_left<m>(a+i*m,x+i,1,i+1);
					if constexpr(flag==2)y[i]-=symb_mul_vecd_left<m>(a+i*m,x+i,1,i+1);
				}
				for(size_t i=m;i<n;++i)
				{
					if constexpr(flag<=1)y[i]+=symb_mul_vecd_left<m>(a+i*m,x+i,1,m);
					if constexpr(flag==2)y[i]-=symb_mul_vecd_left<m>(a+i*m,x+i,1,m);
				}
			}
		}else
		{
			for(size_t i=0;i<m;++i)
			{
				if constexpr(flag==0)y[i] =symb_mul_vecd_loop<m>(a+i*m,x+i,1,i+1,0,m);
				if constexpr(flag==1)y[i]+=symb_mul_vecd_loop<m>(a+i*m,x+i,1,i+1,0,m);
				if constexpr(flag==2)y[i]-=symb_mul_vecd_loop<m>(a+i*m,x+i,1,i+1,0,m);
			}
			for(size_t i=m;i<n-m;++i)
			{
				if constexpr(flag==0)y[i] =symb_mul_vecd_loop<m>(a+i*m,x+i,1,m,0,m);
				if constexpr(flag==1)y[i]+=symb_mul_vecd_loop<m>(a+i*m,x+i,1,m,0,m);
				if constexpr(flag==2)y[i]-=symb_mul_vecd_loop<m>(a+i*m,x+i,1,m,0,m);
			}
			for(size_t i=n-m;i<n;++i)
			{
				if constexpr(flag==0)y[i] =symb_mul_vecd_loop<m>(a+i*m,x+i,1,m,0,n-i);
				if constexpr(flag==1)y[i]+=symb_mul_vecd_loop<m>(a+i*m,x+i,1,m,0,n-i);
				if constexpr(flag==2)y[i]-=symb_mul_vecd_loop<m>(a+i*m,x+i,1,m,0,n-i);	
			}
		}
	}

	template<size_t n,size_t m,size_t flag,class type_a,class type_t,class type_b>//n=n_dims,m=n_elem
	static  inline  void    symb_mul_vecd           (const type_a* a,const type_t* x,type_t* y,const type_b& b)noexcept
	{//y op b*a*x, op can be =,+=,-= ,where a is symmetric banded matrix, b is scalar, x,y are dense vectors
		if constexpr(2*m>n)
		{
#ifdef use_manual_simd
			if constexpr(m==9ul&&std::is_same_v<type_a,comp_t>&&std::is_same_v<type_t,comp_t>)
			{
				symb_mul_vecd_m09<n,flag>(a,x,y,b);
			}else
#endif
			{
				for(size_t i=0;i<n-m;++i)
				{
					if constexpr(flag==0)y[i] =symb_mul_vecd_right<m>(a+i*m,x+i,0,m)*b;
					if constexpr(flag==1)y[i]+=symb_mul_vecd_right<m>(a+i*m,x+i,0,m)*b;
					if constexpr(flag==2)y[i]-=symb_mul_vecd_right<m>(a+i*m,x+i,0,m)*b;
				}
				for(size_t i=n-m;i<n;++i)
				{
					if constexpr(flag==0)y[i] =symb_mul_vecd_right<m>(a+i*m,x+i,0,n-i)*b;
					if constexpr(flag==1)y[i]+=symb_mul_vecd_right<m>(a+i*m,x+i,0,n-i)*b;
					if constexpr(flag==2)y[i]-=symb_mul_vecd_right<m>(a+i*m,x+i,0,n-i)*b;
				}
				for(size_t i=1;i<m;++i)
				{
					if constexpr(flag<=1)y[i]+=symb_mul_vecd_left<m>(a+i*m,x+i,1,i+1)*b;
					if constexpr(flag==2)y[i]-=symb_mul_vecd_left<m>(a+i*m,x+i,1,i+1)*b;
				}
				for(size_t i=m;i<n;++i)
				{
					if constexpr(flag<=1)y[i]+=symb_mul_vecd_left<m>(a+i*m,x+i,1,m)*b;
					if constexpr(flag==2)y[i]-=symb_mul_vecd_left<m>(a+i*m,x+i,1,m)*b;
				}
			}
		}else
		{
			for(size_t i=0;i<m;++i)
			{
				if constexpr(flag==0)y[i] =symb_mul_vecd_loop<m>(a+i*m,x+i,1,i+1,0,m)*b;
				if constexpr(flag==1)y[i]+=symb_mul_vecd_loop<m>(a+i*m,x+i,1,i+1,0,m)*b;
				if constexpr(flag==2)y[i]-=symb_mul_vecd_loop<m>(a+i*m,x+i,1,i+1,0,m)*b;
			}
			for(size_t i=m;i<n-m;++i)
			{
				if constexpr(flag==0)y[i] =symb_mul_vecd_loop<m>(a+i*m,x+i,1,m,0,m)*b;
				if constexpr(flag==1)y[i]+=symb_mul_vecd_loop<m>(a+i*m,x+i,1,m,0,m)*b;
				if constexpr(flag==2)y[i]-=symb_mul_vecd_loop<m>(a+i*m,x+i,1,m,0,m)*b;
			}
			for(size_t i=n-m;i<n;++i)
			{
				if constexpr(flag==0)y[i] =symb_mul_vecd_loop<m>(a+i*m,x+i,1,m,0,n-i)*b;
				if constexpr(flag==1)y[i]+=symb_mul_vecd_loop<m>(a+i*m,x+i,1,m,0,n-i)*b;
				if constexpr(flag==2)y[i]-=symb_mul_vecd_loop<m>(a+i*m,x+i,1,m,0,n-i)*b;	
			}
		}
	}


	template<size_t n,size_t m,size_t flag,class type_a,class type_t,class type_b>//n=n_dims,m=n_elem
	static  inline  auto    symb_prj_mul_vecd_naive	(const type_a* a,const type_t* x,const type_t* y,type_t* z,const type_b& b)noexcept
	{//z op b*A*y, op can be =,+=  also compute x'*A*y, where ' is hermitian conjugate :)
	//where A is symmetric banded matrix, b is scalar, x,y are dense vectors
   		auto	sum 	=	zero<decltype((*x)*(*a)*(*y))>; 
		if constexpr(2*m>n)
		{
			for(size_t i=0;i<n-m;++i)
			{
				auto tmp = symb_mul_vecd_right<m>(a+i*m,y+i,0,m);
				sum	+= avx_mulc(tmp,x[i]);
				tmp*=b;
				if constexpr(flag==0)z[i] =tmp;
				if constexpr(flag==1)
				{
					#pragma omp atomic
					z[i][0]+=tmp[0];
					#pragma omp atomic
					z[i][1]+=tmp[1];
				}
			}
			for(size_t i=n-m;i<n;++i)
			{
				auto tmp = symb_mul_vecd_right<m>(a+i*m,y+i,0,n-i);
				sum	+= avx_mulc(tmp,x[i]);
				tmp*=b;
				if constexpr(flag==0)z[i] =tmp;
				if constexpr(flag==1)
				{
					#pragma omp atomic
					z[i][0]+=tmp[0];
					#pragma omp atomic
					z[i][1]+=tmp[1];
				}	
			}
			for(size_t i=1;i<m;++i)
			{
				auto tmp = symb_mul_vecd_left<m>(a+i*m,y+i,1,i+1);
				sum	+= avx_mulc(tmp,x[i]);
				tmp*=b;
				if constexpr(flag<=1)
				{
					#pragma omp atomic
					z[i][0]+=tmp[0];
					#pragma omp atomic
					z[i][1]+=tmp[1];
				}
			}
			for(size_t i=m;i<n;++i)
			{
				auto tmp = symb_mul_vecd_left<m>(a+i*m,y+i,1,m);
				sum	+= avx_mulc(tmp,x[i]);
				tmp*=b;
				if constexpr(flag<=1)
				{
					#pragma omp atomic
					z[i][0]+=tmp[0];
					#pragma omp atomic
					z[i][1]+=tmp[1];
				}
			}
		}else
		{
			for(size_t i=0;i<m;++i)
			{
				auto tmp = symb_mul_vecd_loop<m>(a+i*m,y+i,1,i+1,0,m);
				sum	+= avx_mulc(tmp,x[i]);
				tmp*=b;
				if constexpr(flag==0)z[i] =tmp;
				if constexpr(flag==1)
				{
					#pragma omp atomic
					z[i][0]+=tmp[0];
					#pragma omp atomic
					z[i][1]+=tmp[1];
				}
			}
			for(size_t i=m;i<n-m;++i)
			{
				auto tmp = symb_mul_vecd_loop<m>(a+i*m,y+i,1,m,0,m);
				sum	+= avx_mulc(tmp,x[i]);
				tmp*=b;
				if constexpr(flag==0)z[i] =tmp;
				if constexpr(flag==1)
				{
					#pragma omp atomic
					z[i][0]+=tmp[0];
					#pragma omp atomic
					z[i][1]+=tmp[1];
				}
			}
			for(size_t i=n-m;i<n;++i)
			{
				auto tmp = symb_mul_vecd_loop<m>(a+i*m,y+i,1,m,0,n-i);
				sum	+= avx_mulc(tmp,x[i]);
				tmp*=b;
				if constexpr(flag==0)z[i] =tmp;
				if constexpr(flag==1)
				{
					#pragma omp atomic
					z[i][0]+=tmp[0];
					#pragma omp atomic
					z[i][1]+=tmp[1];
				}
			}
		}
		return	sum;
	}//end of symb_prj_mul_vecd_naive

	template<size_t n,size_t m,size_t flag,class type_a,class type_t,class type_b>//n=n_dims,m=n_elem
	static  inline  auto    symb_prj_mul_vecd	(const type_a* a,const type_t* x,const type_t* y,type_t* z,const type_b& b)noexcept
	{//z op b*A*y, op can be =,+=  also compute x'*A*y, where ' is hermitian conjugate :)
	//where A is symmetric banded matrix, b is scalar, x,y are dense vectors
	
		if constexpr(std::is_same_v<type_a,comp_t>&&std::is_same_v<type_t,comp_t>)
		{
#ifdef use_manual_simd
			if constexpr(m==9ul) 
			{
				return symb_prj_mul_vecd_m09<n,flag>(a,x,y,z,b);
			}else
#endif
			{ 	
				return	symb_prj_mul_vecd_naive<n,m,flag>(a,x,y,z,b);
			}
		}else	return	symb_prj_mul_vecd_naive<n,m,flag>(a,x,y,z,b);

	}//end of symb_prj_mul_vecd

	template<size_t n,size_t m,size_t flag,class type_a,class type_t>//n=n_dims,m=n_elem
	static  inline  void    symb_cmul_vecd	(const type_a* a,const type_t* x,type_t* y)noexcept //corresponds to blas ?sbmv
	{//y op conj(a)*x, op can be =,+=,-= ,where a is symmetric banded matrix, x,y are dense vectors
		if constexpr(2*m>n)
		{
#ifdef use_manual_simd
			if constexpr(m==9ul&&std::is_same_v<type_a,comp_t>&&std::is_same_v<type_t,comp_t>)
			{
				symb_cmul_vecd_m09<n,flag>(a,x,y);
			}else
#endif
			{
				for(size_t i=0;i<n-m;++i)
				{
					if constexpr(flag==0)y[i] =symb_mulc_vecd_right<m>(a+i*m,x+i,0,m);
					if constexpr(flag==1)y[i]+=symb_mulc_vecd_right<m>(a+i*m,x+i,0,m);
					if constexpr(flag==2)y[i]-=symb_mulc_vecd_right<m>(a+i*m,x+i,0,m);
				}
				for(size_t i=n-m;i<n;++i)
				{
					if constexpr(flag==0)y[i] =symb_mulc_vecd_right<m>(a+i*m,x+i,0,n-i);
					if constexpr(flag==1)y[i]+=symb_mulc_vecd_right<m>(a+i*m,x+i,0,n-i);
					if constexpr(flag==2)y[i]-=symb_mulc_vecd_right<m>(a+i*m,x+i,0,n-i);
				}
				for(size_t i=1;i<m;++i)
				{
					if constexpr(flag<=1)y[i]+=symb_mulc_vecd_left<m>(a+i*m,x+i,1,i+1);
					if constexpr(flag==2)y[i]-=symb_mulc_vecd_left<m>(a+i*m,x+i,1,i+1);
				}
				for(size_t i=m;i<n;++i)
				{
					if constexpr(flag<=1)y[i]+=symb_mulc_vecd_left<m>(a+i*m,x+i,1,m);
					if constexpr(flag==2)y[i]-=symb_mulc_vecd_left<m>(a+i*m,x+i,1,m);
				}
			}
		}else
		{
			for(size_t i=0;i<m;++i)
			{
				if constexpr(flag==0)y[i] =symb_mulc_vecd_loop<m>(a+i*m,x+i,1,i+1,0,m);
				if constexpr(flag==1)y[i]+=symb_mulc_vecd_loop<m>(a+i*m,x+i,1,i+1,0,m);
				if constexpr(flag==2)y[i]-=symb_mulc_vecd_loop<m>(a+i*m,x+i,1,i+1,0,m);
			}
			for(size_t i=m;i<n-m;++i)
			{
				if constexpr(flag==0)y[i] =symb_mulc_vecd_loop<m>(a+i*m,x+i,1,m,0,m);
				if constexpr(flag==1)y[i]+=symb_mulc_vecd_loop<m>(a+i*m,x+i,1,m,0,m);
				if constexpr(flag==2)y[i]-=symb_mulc_vecd_loop<m>(a+i*m,x+i,1,m,0,m);
			}
			for(size_t i=n-m;i<n;++i)
			{
				if constexpr(flag==0)y[i] =symb_mulc_vecd_loop<m>(a+i*m,x+i,1,m,0,n-i);
				if constexpr(flag==1)y[i]+=symb_mulc_vecd_loop<m>(a+i*m,x+i,1,m,0,n-i);
				if constexpr(flag==2)y[i]-=symb_mulc_vecd_loop<m>(a+i*m,x+i,1,m,0,n-i);	
			}
		}
	}

	template<size_t n,size_t m,size_t flag,class type_a,class type_t,class mult_t>//n=n_dims,m=n_elem
	static  inline  void    symb_cmul_vecd	(const type_a* a,const type_t* x,type_t* y,const mult_t& b)noexcept //corresponds to blas ?sbmv
	{//y op b*conj(a)*x, op can be =,+=,-= ,where a is symmetric banded matrix, x,y are dense vectors
		if constexpr(2*m>n)
		{
#ifdef use_manual_simd
			if constexpr(m==9ul&&std::is_same_v<type_a,comp_t>&&std::is_same_v<type_t,comp_t>)
			{
				symb_cmul_vecd_m09<n,flag>(a,x,y,b);
			}else
#endif
			{
				for(size_t i=0;i<n-m;++i)
				{
					if constexpr(flag==0)y[i] =symb_mulc_vecd_right<m>(a+i*m,x+i,0,m)*b;
					if constexpr(flag==1)y[i]+=symb_mulc_vecd_right<m>(a+i*m,x+i,0,m)*b;
					if constexpr(flag==2)y[i]-=symb_mulc_vecd_right<m>(a+i*m,x+i,0,m)*b;
				}
				for(size_t i=n-m;i<n;++i)
				{
					if constexpr(flag==0)y[i] =symb_mulc_vecd_right<m>(a+i*m,x+i,0,n-i)*b;
					if constexpr(flag==1)y[i]+=symb_mulc_vecd_right<m>(a+i*m,x+i,0,n-i)*b;
					if constexpr(flag==2)y[i]-=symb_mulc_vecd_right<m>(a+i*m,x+i,0,n-i)*b;
				}
				for(size_t i=1;i<m;++i)
				{
					if constexpr(flag<=1)y[i]+=symb_mulc_vecd_left<m>(a+i*m,x+i,1,i+1)*b;
					if constexpr(flag==2)y[i]-=symb_mulc_vecd_left<m>(a+i*m,x+i,1,i+1)*b;
				}
				for(size_t i=m;i<n;++i)
				{
					if constexpr(flag<=1)y[i]+=symb_mulc_vecd_left<m>(a+i*m,x+i,1,m)*b;
					if constexpr(flag==2)y[i]-=symb_mulc_vecd_left<m>(a+i*m,x+i,1,m)*b;
				}
			}
		}else
		{
			for(size_t i=0;i<m;++i)
			{
				if constexpr(flag==0)y[i] =symb_mulc_vecd_loop<m>(a+i*m,x+i,1,i+1,0,m)*b;
				if constexpr(flag==1)y[i]+=symb_mulc_vecd_loop<m>(a+i*m,x+i,1,i+1,0,m)*b;
				if constexpr(flag==2)y[i]-=symb_mulc_vecd_loop<m>(a+i*m,x+i,1,i+1,0,m)*b;
			}
			for(size_t i=m;i<n-m;++i)
			{
				if constexpr(flag==0)y[i] =symb_mulc_vecd_loop<m>(a+i*m,x+i,1,m,0,m)*b;
				if constexpr(flag==1)y[i]+=symb_mulc_vecd_loop<m>(a+i*m,x+i,1,m,0,m)*b;
				if constexpr(flag==2)y[i]-=symb_mulc_vecd_loop<m>(a+i*m,x+i,1,m,0,m)*b;
			}
			for(size_t i=n-m;i<n;++i)
			{
				if constexpr(flag==0)y[i] =symb_mulc_vecd_loop<m>(a+i*m,x+i,1,m,0,n-i)*b;
				if constexpr(flag==1)y[i]+=symb_mulc_vecd_loop<m>(a+i*m,x+i,1,m,0,n-i)*b;
				if constexpr(flag==2)y[i]-=symb_mulc_vecd_loop<m>(a+i*m,x+i,1,m,0,n-i)*b;	
			}
		}
	}

	template<size_t n,size_t m,size_t flag,class type_a,class type_t>//n=n_dims,m=n_elem
	static	inline	void	asyb_mul_vecd		(const type_a* a,const type_t* x,type_t* y)noexcept
	{//y op a*x, op can be =,+=,-=, where a is asymmetric banded matrix, x,y are dense vectors
		static_assert(2*m<=n);
		for(size_t i=0;i<m;++i)
		{
			if constexpr(flag==0)y[i] =asyb_mul_vecd_loop<m>(a+i*m,x+i,1,i+1,1,m);
			if constexpr(flag==1)y[i]+=asyb_mul_vecd_loop<m>(a+i*m,x+i,1,i+1,1,m);
			if constexpr(flag==2)y[i]-=asyb_mul_vecd_loop<m>(a+i*m,x+i,1,i+1,1,m);
		}
		for(size_t i=m;i<n-m;++i)
		{
			if constexpr(flag==0)y[i] =asyb_mul_vecd_loop<m>(a+i*m,x+i,1,m,1,m);
			if constexpr(flag==1)y[i]+=asyb_mul_vecd_loop<m>(a+i*m,x+i,1,m,1,m);
			if constexpr(flag==2)y[i]-=asyb_mul_vecd_loop<m>(a+i*m,x+i,1,m,1,m);
		}
		for(size_t i=n-m;i<n;++i)
		{
			if constexpr(flag==0)y[i] =asyb_mul_vecd_loop<m>(a+i*m,x+i,1,m,1,n-i);
			if constexpr(flag==1)y[i]+=asyb_mul_vecd_loop<m>(a+i*m,x+i,1,m,1,n-i);
			if constexpr(flag==2)y[i]-=asyb_mul_vecd_loop<m>(a+i*m,x+i,1,m,1,n-i);
		}
	}
	template<size_t n,size_t m,size_t flag,class type_a,class type_t,class type_b>//n=n_dims,m=n_elem
	static  inline  void    asyb_mul_vecd           (const type_a* a,const type_t* x,type_t* y,const type_b& b)noexcept
	{//y op b*a*x, op can be =,+=,-=, where a is asymmetric banded matrix, b is scalar, x,y are dense vectors
		static_assert(2*m<=n);
		for(size_t i=0;i<m;++i)
		{
			if constexpr(flag==0)y[i] =asyb_mul_vecd_loop<m>(a+i*m,x+i,1,i+1,1,m)*b;
			if constexpr(flag==1)y[i]+=asyb_mul_vecd_loop<m>(a+i*m,x+i,1,i+1,1,m)*b;
			if constexpr(flag==2)y[i]-=asyb_mul_vecd_loop<m>(a+i*m,x+i,1,i+1,1,m)*b;
		}
		for(size_t i=m;i<n-m;++i)
		{
			if constexpr(flag==0)y[i] =asyb_mul_vecd_loop<m>(a+i*m,x+i,1,m,1,m)*b;
			if constexpr(flag==1)y[i]+=asyb_mul_vecd_loop<m>(a+i*m,x+i,1,m,1,m)*b;
			if constexpr(flag==2)y[i]-=asyb_mul_vecd_loop<m>(a+i*m,x+i,1,m,1,m)*b;
		}
		for(size_t i=n-m;i<n;++i)
		{
			if constexpr(flag==0)y[i] =asyb_mul_vecd_loop<m>(a+i*m,x+i,1,m,1,n-i)*b;
			if constexpr(flag==1)y[i]+=asyb_mul_vecd_loop<m>(a+i*m,x+i,1,m,1,n-i)*b;
			if constexpr(flag==2)y[i]-=asyb_mul_vecd_loop<m>(a+i*m,x+i,1,m,1,n-i)*b;
		}
	}
	template<size_t n,size_t m,size_t flag,class type_a,class type_t>//n=n_dims,m=n_elem
	static  inline  void    herb_mul_vecd           (const type_a* a,const type_t* x,type_t* y)noexcept//corresponds to blas ?hbmv
	{//y op a*x, op can be =,+=,-= ,where a is hermitian banded matrix, x,y are dense vectors
		static_assert(2*m<=n);
		for(size_t i=0;i<m;++i)
		{
			if constexpr(flag==0)y[i] =herb_mul_vecd_loop<m>(a+i*m,x+i,1,i+1,0,m);
			if constexpr(flag==1)y[i]+=herb_mul_vecd_loop<m>(a+i*m,x+i,1,i+1,0,m);
			if constexpr(flag==2)y[i]-=herb_mul_vecd_loop<m>(a+i*m,x+i,1,i+1,0,m);
		}
		for(size_t i=m;i<n-m;++i)
		{
			if constexpr(flag==0)y[i] =herb_mul_vecd_loop<m>(a+i*m,x+i,1,m,0,m);
			if constexpr(flag==1)y[i]+=herb_mul_vecd_loop<m>(a+i*m,x+i,1,m,0,m);
			if constexpr(flag==2)y[i]-=herb_mul_vecd_loop<m>(a+i*m,x+i,1,m,0,m);
		}
		for(size_t i=n-m;i<n;++i)
		{
			if constexpr(flag==0)y[i] =herb_mul_vecd_loop<m>(a+i*m,x+i,1,m,0,n-i);
			if constexpr(flag==1)y[i]+=herb_mul_vecd_loop<m>(a+i*m,x+i,1,m,0,n-i);
			if constexpr(flag==2)y[i]-=herb_mul_vecd_loop<m>(a+i*m,x+i,1,m,0,n-i);	
		}
	}
	template<size_t n,size_t m,size_t flag,class type_a,class type_t,class type_b>//n=n_dims,m=n_elem
	static  inline  void    herb_mul_vecd           (const type_a* a,const type_t* x,type_t* y,const type_b& b)noexcept
	{//y op b*a*x, op can be =,+=,-= ,where a is hermitian banded matrix, b is scalar, x,y are dense vectors
		static_assert(2*m<=n);
		for(size_t i=0;i<m;++i)
		{
			if constexpr(flag==0)y[i] =herb_mul_vecd_loop<m>(a+i*m,x+i,1,i+1,0,m)*b;
			if constexpr(flag==1)y[i]+=herb_mul_vecd_loop<m>(a+i*m,x+i,1,i+1,0,m)*b;
			if constexpr(flag==2)y[i]-=herb_mul_vecd_loop<m>(a+i*m,x+i,1,i+1,0,m)*b;
		}
		for(size_t i=m;i<n-m;++i)
		{
			if constexpr(flag==0)y[i] =herb_mul_vecd_loop<m>(a+i*m,x+i,1,m,0,m)*b;
			if constexpr(flag==1)y[i]+=herb_mul_vecd_loop<m>(a+i*m,x+i,1,m,0,m)*b;
			if constexpr(flag==2)y[i]-=herb_mul_vecd_loop<m>(a+i*m,x+i,1,m,0,m)*b;
		}
		for(size_t i=n-m;i<n;++i)
		{
			if constexpr(flag==0)y[i] =herb_mul_vecd_loop<m>(a+i*m,x+i,1,m,0,n-i)*b;
			if constexpr(flag==1)y[i]+=herb_mul_vecd_loop<m>(a+i*m,x+i,1,m,0,n-i)*b;
			if constexpr(flag==2)y[i]-=herb_mul_vecd_loop<m>(a+i*m,x+i,1,m,0,n-i)*b;	
		}
	}
	template<size_t n,size_t m,size_t flag,class type_a,class type_t>//n=n_dims,m=n_elem
	static  inline  void    ltrb_mul_vecd           (const type_a* a,const type_t* x,type_t* y)noexcept
	{//y op a*x, op can be =,+=,-=, where a is banded lower triangular matrix, x,y are dense vectors
		for(size_t i=0;i<m;++i)
		{
			if constexpr(flag==0)y[i] =ltrb_mul_vecd_loop<m>(a+i*m,x+i,0,i+1);
			if constexpr(flag==1)y[i]+=ltrb_mul_vecd_loop<m>(a+i*m,x+i,0,i+1);
			if constexpr(flag==2)y[i]-=ltrb_mul_vecd_loop<m>(a+i*m,x+i,0,i+1);
		}
		for(size_t i=m;i<n;++i)
		{
			if constexpr(flag==0)y[i] =ltrb_mul_vecd_loop<m>(a+i*m,x+i,0,m);
			if constexpr(flag==1)y[i]+=ltrb_mul_vecd_loop<m>(a+i*m,x+i,0,m);
			if constexpr(flag==2)y[i]-=ltrb_mul_vecd_loop<m>(a+i*m,x+i,0,m);
		}	
	}
	template<size_t n,size_t m,size_t flag,class type_a,class type_t,class type_b>//n=n_dims,m=n_elem
	static  inline  void    ltrb_mul_vecd           (const type_a* a,const type_t* x,type_t* y,const type_b& b)noexcept
	{//y op b*a*x, op can be =,+=,-=, where a is banded lower triangular matrix, b is scalar, x,y are dense vectors
		for(size_t i=0;i<m;++i)
		{
			if constexpr(flag==0)y[i] =ltrb_mul_vecd_loop<m>(a+i*m,x+i,0,i+1)*b;
			if constexpr(flag==1)y[i]+=ltrb_mul_vecd_loop<m>(a+i*m,x+i,0,i+1)*b;
			if constexpr(flag==2)y[i]-=ltrb_mul_vecd_loop<m>(a+i*m,x+i,0,i+1)*b;
		}
		for(size_t i=m;i<n;++i)
		{
			if constexpr(flag==0)y[i] =ltrb_mul_vecd_loop<m>(a+i*m,x+i,0,m)*b;
			if constexpr(flag==1)y[i]+=ltrb_mul_vecd_loop<m>(a+i*m,x+i,0,m)*b;
			if constexpr(flag==2)y[i]-=ltrb_mul_vecd_loop<m>(a+i*m,x+i,0,m)*b;
		}
		
	}
	template<size_t n,size_t m,size_t flag,class type_a,class type_t>//n=n_dims,m=n_elem
	static	inline	void	utrb_mul_vecd           (const type_a* a,const type_t* x,type_t* y)noexcept
	{//y op a*x, op can be =,+=,-=, where a is banded upper triangular matrix, x,y are dense vectors
		for(size_t i=0;i<n-m;++i)
		{
			if constexpr(flag==0)y[i] =utrb_mul_vecd_loop<m>(a+i*m,x+i,0,m);
			if constexpr(flag==1)y[i]+=utrb_mul_vecd_loop<m>(a+i*m,x+i,0,m);
			if constexpr(flag==2)y[i]-=utrb_mul_vecd_loop<m>(a+i*m,x+i,0,m);
		}
		for(size_t i=n-m;i<n;++i)
		{
			if constexpr(flag==0)y[i] =utrb_mul_vecd_loop<m>(a+i*m,x+i,0,n-i);
			if constexpr(flag==1)y[i]+=utrb_mul_vecd_loop<m>(a+i*m,x+i,0,n-i);
			if constexpr(flag==2)y[i]-=utrb_mul_vecd_loop<m>(a+i*m,x+i,0,n-i);
		}
	}
	template<size_t n,size_t m,size_t flag,class type_a,class type_t,class type_b>//n=n_dims,m=n_elem
	static  inline  void    utrb_mul_vecd           (const type_a* a,const type_t* x,type_t* y,const type_b& b)noexcept
	{//y op b*a*x, op can be =,+=,-=, where a is banded upper triangular matrix, b is scalar, x,y are dense vectors
		for(size_t i=0;i<n-m;++i)
		{
			if constexpr(flag==0)y[i] =utrb_mul_vecd_loop<m>(a+i*m,x+i,0,m)*b;
			if constexpr(flag==1)y[i]+=utrb_mul_vecd_loop<m>(a+i*m,x+i,0,m)*b;
			if constexpr(flag==2)y[i]-=utrb_mul_vecd_loop<m>(a+i*m,x+i,0,m)*b;
		}
		for(size_t i=n-m;i<n;++i)
		{
			if constexpr(flag==0)y[i] =utrb_mul_vecd_loop<m>(a+i*m,x+i,0,n-i)*b;
			if constexpr(flag==1)y[i]+=utrb_mul_vecd_loop<m>(a+i*m,x+i,0,n-i)*b;
			if constexpr(flag==2)y[i]-=utrb_mul_vecd_loop<m>(a+i*m,x+i,0,n-i)*b;
		}
	}
	template<size_t n,size_t m,class type_a,class type_t>//n=n_dims,m=n_elem
	static	inline	auto	utrb_nrm_vecd           (const type_a* a,const type_t* x)noexcept
	{//get (ax)^H*(ax), where a is banded upper triangular matrix, x is dense vector
		auto	sum	=	zero<decltype(norm(*a**x))>;
		for(size_t i=0;i<n-m;++i)
		{
			sum	+=	norm(utrb_mul_vecd_loop<m>(a+i*m,x+i,0,m));
		}
		for(size_t i=n-m;i<n;++i)
		{
			sum 	+=	norm(utrb_mul_vecd_loop<m>(a+i*m,x+i,0,n-i));
		}
		return	sum;
	}
	template<size_t n,size_t m,class type_a,class type_x,class type_y>///n=n_dims,m=n_elem 
	static	inline	auto	utrb_prj_vecd		(const type_a* a,const type_x* x,const type_y* y)noexcept
	{//get (ax)^H*(ay), where a is banded upper triangular matrix, x,y are dense vectors
		auto	sum	=	zero<decltype(*a**y**x)>;
		for(size_t i=0;i<n-m;++i)
		{
			auto const*	ai	=	a+i*m;
			sum	+=	avx_mulc( 
					utrb_mul_vecd_loop<m>(ai,y+i,0,m),
					utrb_mul_vecd_loop<m>(ai,x+i,0,m));
		}
		for(size_t i=n-m;i<n;++i)
		{
			auto const*	ai	=	a+i*m;
			sum 	+=	avx_mulc(
					utrb_mul_vecd_loop<m>(ai,y+i,0,n-i),
					utrb_mul_vecd_loop<m>(ai,x+i,0,n-i));
		}
		return	sum;
	}

	template<size_t n,size_t m,class type_a,class type_t>//n=n_dims,m=n_elem
	static	inline	void	ltrb_inv_vecd		(const type_a* a,const type_t* x,type_t* y)noexcept
	{//y = a\x, where a is banded lower triangular matrix, x,y are dense vectors
		for(size_t i=0;i<m;++i)
		{
			ltrb_inv_vecd_loop<m>(a+i*m,x[i],y+i,1,i+1);
		}
		for(size_t i=m;i<n;++i)
		{
			ltrb_inv_vecd_loop<m>(a+i*m,x[i],y+i,1,m);
		}
	}
	template<size_t n,size_t m,class type_a,class type_t,class type_b>//n=n_dims,m=n_elem
	static  inline  void    ltrb_inv_vecd           (const type_a* a,const type_t* x,type_t* y,const type_b& b)noexcept
	{//y = (b*a)\x, where a is banded lower triangular matrix, b is scalar, x,y are dense vectors
		for(size_t i=0;i<m;++i)
		{
			ltrb_inv_vecd_loop<m>(a+i*m,x[i]/b,y+i,1,i+1);
		}
		for(size_t i=m;i<n;++i)
		{
			ltrb_inv_vecd_loop<m>(a+i*m,x[i]/b,y+i,1,m);
		}
	}
	template<size_t n,size_t m,class type_a,class type_t>//n=n_dims,m=n_elem
	static  inline  void    utrb_inv_vecd		(const type_a* a,const type_t* x,type_t* y)noexcept
	{//y = a\x, where a is banded upper triangular matrix, x,y are dense vectors
		for(size_t i=0;i<m;++i)//IMPROVE ME. ADD CACHE HINT?
		{
			size_t  s=n-1-i;
			utrb_inv_vecd_loop<m>(a+s*m,x[s],y+s,1,i+1);
		}
		for(size_t i=m;i<n;++i)
		{
			size_t  s=n-1-i;
			utrb_inv_vecd_loop<m>(a+s*m,x[s],y+s,1,m);
		}
	}
	template<size_t n,size_t m,class type_a,class type_t,class type_b>//n=n_dims,m=n_elem
	static  inline  void    utrb_inv_vecd		(const type_a* a,const type_t* x,type_t* y,const type_b& b)noexcept
	{//y = (b*a)\x, where a is banded upper triangular matrix, b is scalar, x,y are dense vectors
		for(size_t i=0;i<m;++i)//IMPROVE ME. ADD CACHE HINT?
		{
			size_t  s=n-1-i;
			utrb_inv_vecd_loop<m>(a+s*m,x[s]/b,y+s,1,i+1);
		}
		for(size_t i=m;i<n;++i)
		{
			size_t  s=n-1-i;
			utrb_inv_vecd_loop<m>(a+s*m,x[s]/b,y+s,1,m);
		}
	}
	template<size_t n,size_t m,class type_a,class type_t>//n=n_dims,m=n_elem
	static	inline	void	ltrb_ldl_vecd		(const type_a* a,const type_t* x,type_t* y)noexcept
	{//Y = L\X, where L is banded lower triangular matrix from a LDLT factorization, x,y are dense vectors
		for(size_t i=0;i<m;++i)
		{
			ltrb_ldl_vecd_loop<m>(a+i*m,x[i],y+i,1,i+1);
		}
		for(size_t i=m;i<n;++i)
		{
			ltrb_ldl_vecd_loop<m>(a+i*m,x[i],y+i,1,m);
		}
	}
	template<size_t n,size_t m,class type_a,class type_t,class type_b>//n=n_dims,m=n_elem
	static  inline  void    ltrb_ldl_vecd           (const type_a* a,const type_t* x,type_t* y,const type_b& b)noexcept
	{//y = (b*L)\x, where L is banded lower triangular matrix from a LDLT factorization, b is scalar, x,y are dense vectors
		for(size_t i=0;i<m;++i)
		{
			ltrb_ldl_vecd_loop<m>(a+i*m,x[i]/b,y+i,1,i+1);
		}
		for(size_t i=m;i<n;++i)
		{
			ltrb_ldl_vecd_loop<m>(a+i*m,x[i]/b,y+i,1,m);
		}
	}
	template<size_t n,size_t m,class type_a,class type_t>//n=n_dims,m=n_elem
	static  inline  void    utrb_ldl_vecd		(const type_a* a,const type_t* x,type_t* y)noexcept
	{//y = (D*L^T)\x, where D*L^T is banded upper triangular matrix from a LDLT factorization, x,y are dense vectors
		for(size_t i=0;i<m;++i)//IMPROVE ME. ADD CACHE HINT?
		{
			size_t  s=n-1-i;
			utrb_ldl_vecd_loop<m>(a+s*m,x[s],y+s,1,i+1);
		}
		for(size_t i=m;i<n;++i)
		{
			size_t  s=n-1-i;
			utrb_ldl_vecd_loop<m>(a+s*m,x[s],y+s,1,m);
		}
	}
	template<size_t n,size_t m,class type_a,class type_t,class type_b>//n=n_dims,m=n_elem
	static  inline  void    utrb_ldl_vecd		(const type_a* a,const type_t* x,type_t* y,const type_b& b)noexcept
	{//y = (b*D*L^T)\x, where D*L^T is banded upper triangular matrix from a LDLT factorization, b is scalar, x,y are dense vectors
		for(size_t i=0;i<m;++i)//IMPROVE ME. ADD CACHE HINT?
		{
			size_t  s=n-1-i;
			utrb_ldl_vecd_loop<m>(a+s*m,x[s]/b,y+s,1,i+1);
		}
		for(size_t i=m;i<n;++i)
		{
			size_t  s=n-1-i;
			utrb_ldl_vecd_loop<m>(a+s*m,x[s]/b,y+s,1,m);
		}
	}
	template<size_t n,size_t m,class type_a,class type_x,class type_y>//n=n_dims,m=n_elem
	static  inline  auto    symb_prj_vecd_naive	(const type_a* a,const type_x* x,const type_y* y)noexcept
	{
		auto	tmp	=	zero<decltype(*a**x**y)>;
		for(size_t i=0;i<m;++i)
		{
			tmp+=avx_mulc(symb_mul_vecd_loop<m>(a+i*m,y+i,1,i+1,0,m),x[i]);
		}
		for(size_t i=m;i<n-m;++i)
		{
			tmp+=avx_mulc(symb_mul_vecd_loop<m>(a+i*m,y+i,1,m,0,m)  ,x[i]);
		}
		for(size_t i=n-m;i<n;++i)
		{
			tmp+=avx_mulc(symb_mul_vecd_loop<m>(a+i*m,y+i,1,m,0,n-i),x[i]);
		}
		return 	tmp;
	}

	template<size_t n,size_t m,class type_a,class type_x,class type_y>//n=n_dims,m=n_elem
	static  inline  auto    symb_prj_vecd		(const type_a* a,const type_x* x,const type_y* y)noexcept
	{//x^H*a*y, where a is symmetric banded matrix, ^H means hermitian conjugate
		if constexpr(std::is_same_v<type_a,comp_t>&&std::is_same_v<type_x,comp_t>&&std::is_same_v<type_y,comp_t>)
		{
#ifdef use_manual_simd
			if constexpr(m==9ul)
			{
				return symb_prj_vecd_m09<n>(a,x,y);
			}else
#endif
			{
			 	return	symb_prj_vecd_naive<n,m>(a,x,y);
			}
		}else	return	symb_prj_vecd_naive<n,m>(a,x,y);
	}
	template<size_t n,size_t m,class type_a,class type_x,class type_y>//n=n_dims,m=n_elem
	static	inline	auto	asyb_prj_vecd		(const type_a* a,const type_x* x,const type_y* y)noexcept
	{//x^H*a*y, where a is asymmetric banded matrix, ^H means hermitian conjugate
		auto    tmp     =       zero<decltype(*a**x**y)>;
		for(size_t i=0;i<m;++i)
		{
			tmp+=avx_mulc(asyb_mul_vecd_loop<m>(a+i*m,y+i,1,i+1,0,m),x[i]);
		}
		for(size_t i=m;i<n-m;++i)
		{
			tmp+=avx_mulc(asyb_mul_vecd_loop<m>(a+i*m,y+i,1,m,0,m)  ,x[i]);
		}
		for(size_t i=n-m;i<n;++i)
		{
			tmp+=avx_mulc(asyb_mul_vecd_loop<m>(a+i*m,y+i,1,m,0,n-i),x[i]);
		}
		return 	tmp;
	}


#undef use_manual_simd
