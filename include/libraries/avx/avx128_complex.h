#pragma once
//*==================================================================================================*//
//*avx(128bit) complex
//*==================================================================================================*//
/*(1) load and cast*/
	static inline	avx128_complex	avx128_loadr	(const double x)noexcept	
	{
		return 	avx128_complex{x,0.};
	}
	static inline	avx128_complex	avx128_loadi	(const double x)noexcept	
	{
		return 	avx128_complex{0.,x};
	}
	static inline	avx128_complex	avx128_loadu	(const avx128_complex* x,const size_t i)noexcept
	{
		return 	x[i];
	}
	static inline	avx128_complex	avx128_loadu	(const avx128_complex* x,const size_t*i)noexcept
	{
		return 	x[*i];
	}
	static inline	void		avx128_dumpu	(const avx128_complex& v,avx128_complex* x,const size_t i)noexcept
	{
		x[i]=v;
	}
	static inline	void		avx128_dumpu	(const avx128_complex& v,avx128_complex* x,const size_t*i)noexcept
	{
		x[*i]=v;
	}
/*(2) arithmetic*/
	static inline	avx128_complex	avx_add	(const avx128_complex& x,const avx128_complex& y)noexcept
	{
		return 	{_mm_add_pd(x,y)};
	}
	static inline	avx128_complex	avx_sub	(const avx128_complex& x,const avx128_complex& y)noexcept
	{
		return 	{_mm_sub_pd(x,y)};
	}
	static inline	avx128_complex	avx_mul	(const avx128_complex& x,const avx128_complex& y)noexcept
	{
		__m128d		xmm1	=	_mm_permute_pd(y,_0b00000000);
		__m128d		xmm2	=	_mm_permute_pd(y,_0b11111111);
		__m128d		xmm3	=	_mm_permute_pd(x,_0b01010101);
		__m128d		xmm4	=	_mm_mul_pd(xmm2,xmm3);
		return 	{_mm_fmaddsub_pd(xmm1,x,xmm4)};
	}
	static inline	avx128_complex	avx_mul	(const avx128_complex& x,const double& y)noexcept
	{
		return 	{_mm_mul_pd(x,avx128_double{y,y})};
	}
	static inline	avx128_complex	avx_mul	(const double& x,const avx128_complex& y)noexcept
	{
		return 	{_mm_mul_pd(y,avx128_double{x,x})};
	}
	static inline	avx128_complex	avx_mulc(const avx128_complex& x,const avx128_complex& y)noexcept
	{
		__m128d         xmm1    =       _mm_permute_pd(y,_0b00000000);
                __m128d         xmm2    =       _mm_permute_pd(y,_0b11111111);
                __m128d         xmm3    =       _mm_permute_pd(x,_0b01010101);
                __m128d         xmm4    =       _mm_mul_pd(xmm2,xmm3);
                return  {_mm_fmsubadd_pd(xmm1,x,xmm4)};
	}
	static inline	double		avx_mulc(const double& x,const double& y)noexcept
	{
		return 	x*y;
	}
	static inline	avx128_complex	avx_mulc(const avx128_complex& x,const double& y)noexcept
	{
		return 	{_mm_mul_pd(x,avx128_double{y,y})};
	}
	static inline	avx128_complex	avx_mulc(const double& x,const avx128_complex& y)noexcept
	{
		return 	{_mm_mul_pd(y,avx128_double{x,-x})};
	}	
	static inline	avx128_complex	avx_div	(const avx128_complex& x,const avx128_complex& y)noexcept
	{
		__m128d		xmm1	=	_mm_permute_pd(y,_0b00000000);	//yr,yr
		__m128d		xmm2	=	_mm_permute_pd(y,_0b11111111);	//yi,yi
		__m128d		xmm3	=	_mm_permute_pd(x,_0b01010101);	//xi,xr
		__m128d		xmm4	=	_mm_mul_pd(xmm2,xmm3);		//yi*xi,yi*xr
		__m128d	 	xmm5	=	_mm_fmsubadd_pd(xmm1,x,xmm4);	//(yr*xr,yr*xi)-+(yi*xi,yi*xr)
		__m128d		xmm6	=	_mm_mul_pd(xmm2,xmm2);		//yi*yi,yi*yi
		__m128d		xmm7	=	_mm_fmadd_pd(xmm1,xmm1,xmm6);	//yi*yi+yr*yr
		return 	{_mm_div_pd(xmm5,xmm7)};
	}
	static inline	avx128_complex	avx_div	(const avx128_complex& x,const double& y)noexcept
	{
		return 	{_mm_div_pd(x,avx128_double{y,y})};
	}
	static inline	avx128_complex	avx_div	(const double&x,const avx128_complex& y)noexcept
	{
		__m128d		xmm1	=	_mm_mul_pd(y,y);
		__m128d		xmm2	=	_mm_permute_pd(xmm1,_0b01010101);
		__m128d		xmm3	=	{x,-x};
		__m128d		xmm4	=	_mm_add_pd(xmm1,xmm2);
		__m128d		xmm5	=	_mm_mul_pd(xmm3,y);
		return	{_mm_div_pd(xmm5,xmm4)};
	}
/*(3) function*/
	static	inline	double		avx_norm (const avx128_complex& x)noexcept
	{
		__m128d		xmm0	=	_mm_mul_pd(x,x);
		return	xmm0[0]+xmm0[1];
	}
	static	inline	double		avx_redu (const avx128_complex& x)noexcept
	{
		return	x[0]+x[1];
	}
/*(4) operator*/
	inline	__declare_oper(+,avx128_complex,const avx128_complex&x,const avx128_complex&y,avx_add(x,y))
	inline	__declare_oper(+,avx128_complex,const avx128_complex&x,const double&y,avx_add(x,avx128_complex{y,0}))
	inline	__declare_oper(+,avx128_complex,const double&x,const avx128_complex&y,avx_add(avx128_complex{x,0},y))

	inline	__declare_oper(-,avx128_complex,const avx128_complex&x,const avx128_complex&y,avx_sub(x,y))
	inline	__declare_oper(-,avx128_complex,const avx128_complex&x,const double&y,avx_sub(x,avx128_complex{y,0}))
	inline	__declare_oper(-,avx128_complex,const double&x,const avx128_complex&y,avx_sub(avx128_complex{x,0},y))
	
	inline	__declare_oper(*,avx128_complex,const avx128_complex&x,const avx128_complex&y,avx_mul(x,y))
	inline	__declare_oper(*,avx128_complex,const avx128_complex&x,const double&y,avx_mul(x,y))
	inline	__declare_oper(*,avx128_complex,const double&x,const avx128_complex&y,avx_mul(x,y))

	inline	__declare_oper(/,avx128_complex,const avx128_complex&x,const avx128_complex&y,avx_div(x,y))
	inline	__declare_oper(/,avx128_complex,const avx128_complex&x,const double&y,avx_div(x,y))
	inline	__declare_oper(/,avx128_complex,const double&x,const avx128_complex&y,avx_div(x,y))

	inline	avx128_complex& avx128_complex::operator *=(const avx128_complex& rhs)noexcept	{(*this)=(*this)*rhs;return *this;}
	inline	avx128_complex& avx128_complex::operator /=(const avx128_complex& rhs)noexcept	{(*this)=(*this)/rhs;return *this;}
	inline	avx128_complex& avx128_complex::operator = (const double& rhs)noexcept {data[0]=rhs;data[1]=0.;return *this;}

/*(5) wrappers*/
	static	inline	double		norm		(const double& x)noexcept	{return x*x;}
	static	inline	double		norm		(const avx128_complex&x)noexcept{return avx_norm(x);}
	//mulr implements real(x*conj(y))
	static	inline	double		mulr		(const double& x,const double& y)noexcept{return x*y;}
	static	inline	double		mulr		(const double& x,const avx128_complex& y)noexcept{return x*y[0];}
	static	inline	double		mulr		(const avx128_complex&x,const double& y)noexcept{return	x[0]*y;}
	static 	inline	double		mulr		(const avx128_complex&x,const avx128_complex&y)noexcept{return x[0]*y[0]+x[1]*y[1];}
	//real or imag
	static	inline	double		real		(const double& x)noexcept	{return	x;}
	static 	inline	double		real		(const avx128_complex&x)noexcept{return x[0];}
	static	inline	double		imag		(const double& x)noexcept	{return 0.0;}
	static 	inline	double		imag		(const avx128_complex&x)noexcept{return	x[1];}
	/*conjugate*/
	static	inline	double		conj		(const double& x)noexcept	{return	x;}
