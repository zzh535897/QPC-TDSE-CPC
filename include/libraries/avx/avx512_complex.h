#pragma once

//*==================================================================================================*//
//*avx(512bit) complex number (interleave format)
//*==================================================================================================*//

/*(1) load and cast*/
	static 	inline	avx512_complex	avx512_loadr	(const double x)noexcept
	{
		return	avx512_complex{_mm512_broadcast_f64x2(__m128d{x,0.})};
	}
	static 	inline	avx512_complex	avx512_loadi	(const double x)noexcept
	{
		return	avx512_complex{_mm512_broadcast_f64x2(__m128d{0.,x})};
	}
	static 	inline	avx512_complex	avx512_load	(const avx128_complex& x)noexcept
	{
		return 	avx512_complex{_mm512_broadcast_f64x2(x)};
	}
/*(2) indexing*/
	static constexpr auto avx512_loadr_idx	=	__m512i
	{
		0,4,1,5,2,6,3,7
	};
	static constexpr auto avx512_loadi_idx	=	__m512i
	{
		4,0,5,1,6,2,7,3
	};
	template<size_t stride>
	static constexpr auto avx512_loadr_str	=	__m256i
	{
		0l*stride,1l*stride,2l*stride,3l*stride
	};
	template<size_t stride>
	static constexpr auto avx512_loadi_str	=	__m256i
	{
		0l*stride,1l*stride,2l*stride,3l*stride
	};
	template<size_t stride>
	static constexpr auto avx512_loadc_str	=	__m512i
	{
		0l*stride,0l*stride+1l,
		2l*stride,2l*stride+1l,
		4l*stride,4l*stride+1l,
		6l*stride,6l*stride+1l,	
	};
	template<size_t stride=1ul>
	static	inline	avx512_complex	avx512_loadr	(const double* x,const size_t i)noexcept//10 latency or 26 latency
	{
		if constexpr(stride==1ul)
		{
			return 	{_mm512_maskz_permutexvar_pd(85 ,avx512_loadr_idx,_mm512_castpd256_pd512(_mm256_loadu_pd(x+i)))};
		}else
		{
			return	{_mm512_maskz_permutexvar_pd(85 ,avx512_loadr_idx,_mm512_castpd256_pd512(_mm256_i64gather_pd(x+i,avx512_loadr_str<stride>,8)))};
		}
	}
	static  inline  avx512_complex  avx512_loadr    (const double* x,const size_t i,const avx8_mask m)noexcept//11 latency
	{
		return 	{_mm512_maskz_permutexvar_pd(85 ,avx512_loadr_idx,_mm512_castpd256_pd512(_mm256_maskz_loadu_pd(m,x+i)))};
	}
	template<size_t stride=1ul>
	static	inline	avx512_complex	avx512_loadi	(const double* x,const size_t i)noexcept//10 latency or 26 latency
	{
		if constexpr(stride==1ul)
		{
			return 	{_mm512_maskz_permutexvar_pd(170,avx512_loadi_idx,_mm512_castpd256_pd512(_mm256_loadu_pd(x+i)))};
		}else
		{
			return 	{_mm512_maskz_permutexvar_pd(170,avx512_loadi_idx,_mm512_castpd256_pd512(_mm256_i64gather_pd(x+i,avx512_loadi_str<stride>,8)))};
		}
	}
	static	inline	avx512_complex  avx512_loadi	(const double* x,const size_t i,const avx8_mask m)noexcept//11 latency
	{
		return 	{_mm512_maskz_permutexvar_pd(170,avx512_loadi_idx,_mm512_castpd256_pd512(_mm256_maskz_loadu_pd(m,x+i)))};
	}
	template<size_t stride=1ul>
	static	inline	avx512_complex	avx512_loadu	(const avx128_complex* x,const size_t i)noexcept//9 latency or 27 latency
	{
		if constexpr(stride==1ul)
		{
			return	{_mm512_loadu_pd(x+i)};
		}else
		{
			return 	{_mm512_i64gather_pd(avx512_loadc_str<stride>,x+i,8)};
		}
	}
	static	inline	avx512_complex	avx512_loadu	(const avx128_complex* x,const size_t i,const avx8_mask m)noexcept//9 latency
	{
		return 	{_mm512_maskz_loadu_pd(m,x+i)};
	}
	static	inline	avx512_complex	avx512_loadu	(const avx128_complex* x,const avx256_sint64& i)noexcept
	{
		__m256i	j	=	_mm256_add_epi64(i,__m256i{1,1,1,1});
		__m512i	k	=	{i[0],j[0],i[1],j[1],i[2],j[2],i[3],j[3]};
		return 	{_mm512_i64gather_pd(k,x,8)};
	}
	static	inline	avx512_complex	avx512_loadr	(const double* x,const avx256_sint64& i)noexcept//28 latency
	{
		constexpr __m512i id={0,4,1,5,2,6,3,7};
		return 	avx512_complex{_mm512_maskz_permutexvar_pd(85 ,id,_mm512_castpd256_pd512(_mm256_i64gather_pd(x,i,8)))};
	}
	static  inline  avx512_complex  avx512_loadi    (const double* x,const avx256_sint64& i)noexcept//28 latency
        {
		constexpr __m512i id={4,0,5,1,6,2,7,3};
                return  avx512_complex{_mm512_maskz_permutexvar_pd(170,id,_mm512_castpd256_pd512(_mm256_i64gather_pd(x,i,8)))};
        }

	template<size_t stride=1>
	static	inline	void		avx512_dumpu	(const avx512_complex& v,avx128_complex* x,const size_t i)noexcept//6 latency or 12 latency
	{
		if constexpr(stride==1)
		{
			_mm512_storeu_pd(x+i,v);
		}else
		{
			_mm512_i64scatter_pd(x+i,avx512_loadc_str<stride>,v,8);
		}
	}
	static	inline	void		avx512_dumpu	(const avx512_complex& v,avx128_complex* x,const size_t i,const avx8_mask m)noexcept//6 latency
	{
		_mm512_mask_storeu_pd(x+i,m,v);
	}
	static	inline	void		avx512_dumpu	(const avx512_complex& v,avx128_complex* x,const avx256_sint64& i)noexcept//14 latency
	{
		__m256i	j	=	_mm256_add_epi64(i,__m256i{1,1,1,1});
		__m512i	k	=	{i[0],j[0],i[1],j[1],i[2],j[2],i[3],j[3]};
		_mm512_i64scatter_pd(x,k,v,8);
	}	
/*(3) arithmetic*/
	static	inline	avx512_complex	avx_add		(const avx512_complex& x,const avx512_complex& y)noexcept//4 latency
	{
		return	{_mm512_add_pd(x,y)};
	}
	static 	inline	avx512_complex	avx_sub		(const avx512_complex& x,const avx512_complex& y)noexcept//4 latency
	{
		return	{_mm512_sub_pd(x,y)};
	}
	static 	inline	avx512_complex	avx_mul		(const avx512_complex& x,const avx512_complex& y)noexcept//11 latency
	{	
		__m512d		zmm1	=	_mm512_permute_pd(y,_0b00000000);//yr,yr	
		__m512d		zmm2	=	_mm512_permute_pd(y,_0b11111111);//yi,yi
		__m512d		zmm3	=	_mm512_permute_pd(x,_0b01010101);//xi,xr
		__m512d		zmm4	=	_mm512_mul_pd(zmm2,zmm3);	//xi*yi,xr*yi
		return 	{_mm512_fmaddsub_pd(zmm1,x,zmm4)};			//(yr,yr)*(xr,xi)-+(xi*yi,xr*yi)
	}
	static 	inline	avx512_complex	avx_mul		(const avx512_complex& x,const double y)noexcept//4 latency
	{
		return 	{_mm512_mul_pd(x,_mm512_set1_pd(y))};
	}
	static 	inline	avx512_complex	avx_mulc	(const avx512_complex& x,const avx512_complex& y)noexcept//11 latency
	{
		__m512d		zmm1	=	_mm512_permute_pd(y,_0b00000000);//yr,yr	
		__m512d		zmm2	=	_mm512_permute_pd(y,_0b11111111);//yi,yi
		__m512d		zmm3	=	_mm512_permute_pd(x,_0b01010101);//xi,xr
		__m512d		zmm4	=	_mm512_mul_pd(zmm2,zmm3);	//xi*yi,xr*yi
		return 	{_mm512_fmsubadd_pd(zmm1,x,zmm4)};			//(yr,yr)*(xr,xi)+-(xi*yi,xr*yi)
	}//conjugated
	static 	inline	avx512_complex	avx_div		(const avx512_complex& x,const avx512_complex& y)noexcept//42 latency
	{
		__m512d		zmm1	=	_mm512_permute_pd(y,_0b00000000);//yr,yr	
		__m512d		zmm2	=	_mm512_permute_pd(y,_0b11111111);//yi,yi
		__m512d		zmm3	=	_mm512_permute_pd(x,_0b01010101);//xi,xr
		__m512d		zmm4	=	_mm512_mul_pd(zmm2,zmm3);	//xi*yi,xr*yi
		__m512d	 	zmm5	=	_mm512_fmsubadd_pd(zmm1,x,zmm4);//(yr,yr)*(xr,xi)+-(xi*yi,yi*xr)
		__m512d		zmm6	=	_mm512_mul_pd(zmm2,zmm2);	//yi*yi,yi*yi
		__m512d		zmm7	=	_mm512_fmadd_pd(zmm1,zmm1,zmm6);//yr*yr+yi*yi
		return 	{_mm512_div_pd(zmm5,zmm7)};
	}
	static 	inline	avx512_complex	avx_div		(const avx512_complex& x,const double y)noexcept//23 latency
	{
		return 	{_mm512_div_pd(x,_mm512_set1_pd(y))};
	}
	static 	inline	avx512_complex	avx_div		(const double x,const avx512_complex& y)noexcept//37 latency
	{
		__m512d		zmm0	=	{x,-x,x,-x,x,-x,x,-x};
		__m512d		zmm1	=	_mm512_mul_pd(y,y);
		__m512d		zmm2	=	_mm512_permute_pd(zmm1,_0b01010101);
		__m512d		zmm3	=	_mm512_add_pd(zmm1,zmm2);
		__m512d		zmm4	=	_mm512_mul_pd(zmm0,y);
		return	{_mm512_div_pd(zmm4,zmm3)};
	}
/*(4) functions*/
	static 	inline	avx512_complex	avx_normc	(const avx512_complex& x)noexcept//12 latency
	{
		__m512d		zmm0	=	_mm512_mul_pd(x,x);
		__m512d		zmm1	=	_mm512_maskz_permute_pd(_0b01010101,zmm0,_0b10101010);
		__m512d		zmm2	=	_mm512_maskz_permute_pd(_0b10101010,zmm0,_0b10101010);
		return	{_mm512_add_pd(zmm1,zmm2)};
	}//norm
	static 	inline	double		avx_norm	(const avx512_complex& x)noexcept
	{
		return 	_mm512_reduce_add_pd(_mm512_mul_pd(x,x));
	}//norm sum (|x|^2)
	static 	inline	double		avx_normw	(const avx512_complex& x,const avx512_complex& y)noexcept
	{
		return 	_mm512_reduce_add_pd(_mm512_mul_pd(_mm512_mul_pd(x,x),y));
	}//norm sum (|x|^2*y)
	//redu sum
	static	inline	double		avx_redu	(const avx512_complex& x)noexcept
	{	
		return	_mm512_reduce_add_pd(x);
	}
	//sum
	static	inline	avx128_complex	avx_sum		(const avx512_complex& x)noexcept
	{
		return 	avx128_complex{x[0]+x[2]+x[4]+x[6],x[1]+x[3]+x[5]+x[7]};
	}	
/*(5) operator*/
	inline	__declare_oper(+,avx512_complex,const avx512_complex&x,const avx512_complex& y,avx_add(x,y))
	inline	__declare_oper(+,avx512_complex,const avx512_complex&x,const double& y,avx_add(x,avx512_loadr(y)))
	inline	__declare_oper(+,avx512_complex,const double&x,const avx512_complex& y,avx_add(y,avx512_loadr(x)))
	inline	__declare_oper(+,avx512_complex,const avx512_complex&x,const avx128_complex& y,avx_add(x,avx512_load(y)))
	inline	__declare_oper(+,avx512_complex,const avx128_complex&x,const avx512_complex& y,avx_add(avx512_load(x),y))
	
	inline	__declare_oper(-,avx512_complex,const avx512_complex&x,const avx512_complex& y,avx_sub(x,y))
	inline	__declare_oper(-,avx512_complex,const avx512_complex&x,const double& y,avx_sub(x,avx512_loadr(y)))
	inline	__declare_oper(-,avx512_complex,const double&x,const avx512_complex& y,avx_sub(avx512_loadr(x),y))
	inline	__declare_oper(-,avx512_complex,const avx512_complex&x,const avx128_complex& y,avx_sub(x,avx512_load(y)))
	inline	__declare_oper(-,avx512_complex,const avx128_complex&x,const avx512_complex& y,avx_sub(avx512_load(x),y))

	inline	__declare_oper(*,avx512_complex,const avx512_complex&x,const avx512_complex& y,avx_mul(x,y))
	inline	__declare_oper(*,avx512_complex,const avx512_complex&x,const double& y,avx_mul(x,y))
	inline	__declare_oper(*,avx512_complex,const double&x,const avx512_complex& y,avx_mul(y,x))
	inline	__declare_oper(*,avx512_complex,const avx512_complex&x,const avx128_complex& y,avx_mul(x,avx512_load(y)))
	inline	__declare_oper(*,avx512_complex,const avx128_complex&x,const avx512_complex& y,avx_mul(avx512_load(x),y))

	inline	__declare_oper(/,avx512_complex,const avx512_complex&x,const avx512_complex& y,avx_div(x,y))
	inline	__declare_oper(/,avx512_complex,const avx512_complex&x,const double& y,avx_div(x,y))
	inline	__declare_oper(/,avx512_complex,const double&x,const avx512_complex& y,avx_div(x,y))
	inline	__declare_oper(/,avx512_complex,const avx512_complex&x,const avx128_complex& y,avx_div(x,avx512_load(y)))
	inline	__declare_oper(/,avx512_complex,const avx128_complex&x,const avx512_complex& y,avx_div(avx512_load(x),y))

	inline	avx512_complex& avx512_complex::operator *=(const avx512_complex& rhs)noexcept	{(*this)=(*this)*rhs;return *this;}
	inline	avx512_complex& avx512_complex::operator /=(const avx512_complex& rhs)noexcept	{(*this)=(*this)/rhs;return *this;}
	inline	avx512_complex&	avx512_complex::operator = (const double& rhs)noexcept {data=data_t{rhs,0.,rhs,0.,rhs,0.,rhs,0.};return *this;}
	inline	avx512_complex&	avx512_complex::operator = (const avx128_complex& rhs)noexcept {data=data_t{rhs[0],rhs[1],rhs[0],rhs[1],rhs[0],rhs[1],rhs[0],rhs[1]};return *this;}

