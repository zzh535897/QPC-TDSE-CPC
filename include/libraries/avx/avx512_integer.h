#pragma once
//============================================================================
// avx(512bit) signed integer
//============================================================================
/*(1) cast and load*/
	static constexpr avx512_sint64	avx512_load	(const long long int&x)noexcept
	{
		return	avx512_sint64{x,x,x,x,x,x,x,x};
	}
	static constexpr avx512_sint64	avx512_load	(const long int& x)noexcept
	{
		return 	avx512_sint64{x,x,x,x,x,x,x,x};
	}
	static 	inline	avx512_sint64	avx512_loadu	(const long long int*x)noexcept
	{
		return 	avx512_sint64{_mm512_maskz_loadu_epi64(_0b11111111,x)};//FIXME loadu?
	}
	static 	inline	avx512_sint64	avx512_loadu	(const long int*x)noexcept
	{
		return	avx512_sint64{_mm512_maskz_loadu_epi64(_0b11111111,x)};//FIXME loadu?
	}
	static	inline	void		avx512_storeu	(long long int* x,const long long int* y)noexcept
	{
		_mm512_mask_storeu_epi64(x,_0b11111111,_mm512_maskz_loadu_epi64(_0b11111111,y));//FIXME loadu
	}
	static	inline	void		avx512_storeu	(long int* x,const long int* y)noexcept
	{
		_mm512_mask_storeu_epi64(x,_0b11111111,_mm512_maskz_loadu_epi64(_0b11111111,y));//FIXME loadu
	}
/*(2) arithmetic*/
	static 	inline	avx512_sint64	avx_add		(const avx512_sint64& x,const avx512_sint64& y)noexcept
	{
		return	{_mm512_add_epi64(x,y)};
	}
	static	inline	avx512_sint64	avx_addz	(const avx512_sint64& x,const avx512_sint64& y,const avx8_mask& m)noexcept
	{
		return	{_mm512_maskz_add_epi64(m,x,y)};	
	}
	static	inline	avx512_sint64	avx_sub		(const avx512_sint64& x,const avx512_sint64& y)noexcept
	{
		return	{_mm512_sub_epi64(x,y)};
	}
	static	inline	avx512_sint64	avx_subz	(const avx512_sint64& x,const avx512_sint64& y,const avx8_mask& m)noexcept
	{
		return  {_mm512_maskz_sub_epi64(m,x,y)};
	}
	static	inline	avx512_sint64	avx_mul		(const avx512_sint64& x,const avx512_sint64& y)noexcept
	{
		return	{_mm512_mullo_epi64(x,y)};
	}
	static	inline  avx512_sint64  avx_mulz    	(const avx512_sint64& x,const avx512_sint64& y,const avx8_mask& m)noexcept
	{
		return 	{_mm512_maskz_mullo_epi64(m,x,y)};
	}
	static	inline	avx512_sint64 	avx_mull	(const avx512_sint64& x,const avx512_sint64& y)noexcept
	{
		return 	{_mm512_mul_epi32(x,y)};
	}
	static 	inline	avx512_sint64	avx_div		(const avx512_sint64& x,const avx512_sint64& y)noexcept
	{
		return 	avx512_sint64{x[0]/y[0],x[1]/y[1],x[2]/y[2],x[3]/y[3],x[4]/y[4],x[5]/y[5],x[6]/y[6],x[7]/y[7]};
		//return	{_mm512_div_epi64(x,y)};//only intel compiler can compile this line
	}
/*(3) functions*/
	static	inline	avx512_sint64	avx_abs		(const avx512_sint64& x)noexcept
	{
		return 	{_mm512_abs_epi64(x)};
	}
	static	inline	avx512_sint64	avx_min		(const avx512_sint64& x,const avx512_sint64& y)noexcept
	{
		return 	{_mm512_min_epi64(x,y)};
	}
	static	inline	avx512_sint64	avx_max		(const avx512_sint64& x,const avx512_sint64& y)noexcept
	{
		return 	{_mm512_max_epi64(x,y)};
	}
	static	inline	long long int	avx_minr	(const avx512_sint64& x)noexcept
	{
		return 	_mm512_reduce_min_epi64(x);
	}
	static	inline	long long int	avx_maxr	(const avx512_sint64& x)noexcept
	{
		return 	_mm512_reduce_max_epi64(x);
	}
	static	inline	long long int	avx_sum		(const avx512_sint64& x)noexcept
	{
		return 	_mm512_reduce_add_epi64(x);
	}
/*(4) logistic*/
	static	inline	avx8_mask	avx_cmp_le	(const avx512_sint64& x,const avx512_sint64& y)noexcept
	{
		return	{_mm512_cmp_epi64_mask	(x,y,_MM_CMPINT_LE)};
	}/*<=*/
	static	inline	avx8_mask	avx_cmp_eq	(const avx512_sint64& x,const avx512_sint64& y)noexcept
	{
		return 	{_mm512_cmp_epi64_mask	(x,y,_MM_CMPINT_EQ)};
	}/*==*/
	static	inline	avx8_mask	avx_cmp_ge	(const avx512_sint64& x,const avx512_sint64& y)noexcept
	{
		return 	{_mm512_cmp_epi64_mask	(x,y,_MM_CMPINT_NLT)};
	}/*>=*/
	static	inline	avx8_mask	avx_cmp_ne	(const avx512_sint64& x,const avx512_sint64& y)noexcept
	{
		return 	{_mm512_cmp_epi64_mask	(x,y,_MM_CMPINT_NE)};
	}/*!=*/
	static	inline	avx8_mask	avx_cmp_lt	(const avx512_sint64& x,const avx512_sint64& y)noexcept
	{
		return 	{_mm512_cmp_epi64_mask	(x,y,_MM_CMPINT_LT)};
	}/*<*/
	static	inline	avx8_mask	avx_cmp_gt	(const avx512_sint64& x,const avx512_sint64& y)noexcept
	{
		return	{_mm512_cmp_epi64_mask	(x,y,_MM_CMPINT_NLE)};
	}/*>*/
/*(5) operator*/
	inline 	avx512_sint64::operator	avx512_double()noexcept		{return {_mm512_cvt_roundepi64_pd (data,_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC)};}
	inline	avx512_sint64::operator const	avx512_double()const noexcept	{return	{_mm512_cvt_roundepi64_pd (data,_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC)};}

	inline	__declare_oper(+,avx512_sint64,const avx512_sint64&x,const avx512_sint64&y,avx_add(x,y))
	inline	__declare_oper(+,avx512_sint64,const long long int& x,const avx512_sint64&y,avx_add(avx512_load(x),y))
	inline	__declare_oper(+,avx512_sint64,const avx512_sint64&x,const long long int& y,avx_add(x,avx512_load(y)))

	inline	__declare_oper(-,avx512_sint64,const avx512_sint64&x,const avx512_sint64&y,avx_sub(x,y))
	inline	__declare_oper(-,avx512_sint64,const long long int& x,const avx512_sint64&y,avx_sub(avx512_load(x),y))
	inline	__declare_oper(-,avx512_sint64,const avx512_sint64&x,const long long int& y,avx_sub(x,avx512_load(y)))

	inline	__declare_oper(*,avx512_sint64,const avx512_sint64&x,const avx512_sint64&y,avx_mul(x,y))
	inline	__declare_oper(*,avx512_sint64,const long long int& x,const avx512_sint64&y,avx_mul(avx512_load(x),y))
	inline	__declare_oper(*,avx512_sint64,const avx512_sint64&x,const long long int& y,avx_mul(x,avx512_load(y)))

	inline	__declare_oper(/,avx512_sint64,const avx512_sint64&x,const avx512_sint64&y,avx_div(x,y))
	inline	__declare_oper(/,avx512_sint64,const long long int& x,const avx512_sint64&y,avx_div(avx512_load(x),y))
	inline	__declare_oper(/,avx512_sint64,const avx512_sint64&x,const long long int& y,avx_div(x,avx512_load(y)))

	inline	__declare_oper(==,avx8_mask,const avx512_sint64&x,const avx512_sint64&y,avx_cmp_eq(x,y))
	inline	__declare_oper(==,avx8_mask,const long long int& x,const avx512_sint64&y,avx_cmp_eq(avx512_load(x),y))
	inline	__declare_oper(==,avx8_mask,const avx512_sint64&x,const long long int& y,avx_cmp_eq(x,avx512_load(y)))

	inline	__declare_oper(!=,avx8_mask,const avx512_sint64&x,const avx512_sint64&y,avx_cmp_ne(x,y))
	inline	__declare_oper(!=,avx8_mask,const long long int& x,const avx512_sint64&y,avx_cmp_ne(avx512_load(x),y))
	inline	__declare_oper(!=,avx8_mask,const avx512_sint64&x,const long long int& y,avx_cmp_ne(x,avx512_load(y)))
	
	inline	__declare_oper(<=,avx8_mask,const avx512_sint64&x,const avx512_sint64&y,avx_cmp_le(x,y))
	inline	__declare_oper(<=,avx8_mask,const long long int& x,const avx512_sint64&y,avx_cmp_le(avx512_load(x),y))
	inline	__declare_oper(<=,avx8_mask,const avx512_sint64&x,const long long int& y,avx_cmp_le(x,avx512_load(y)))

	inline	__declare_oper(>=,avx8_mask,const avx512_sint64&x,const avx512_sint64&y,avx_cmp_ge(x,y))
	inline	__declare_oper(>=,avx8_mask,const long long int& x,const avx512_sint64&y,avx_cmp_ge(avx512_load(x),y))
	inline	__declare_oper(>=,avx8_mask,const avx512_sint64&x,const long long int& y,avx_cmp_ge(x,avx512_load(y)))

	inline	__declare_oper(<,avx8_mask,const avx512_sint64&x,const avx512_sint64&y,avx_cmp_lt(x,y))
	inline	__declare_oper(<,avx8_mask,const long long int& x,const avx512_sint64&y,avx_cmp_lt(avx512_load(x),y))
	inline	__declare_oper(<,avx8_mask,const avx512_sint64&x,const long long int& y,avx_cmp_lt(x,avx512_load(y)))

	inline	__declare_oper(>,avx8_mask,const avx512_sint64&x,const avx512_sint64&y,avx_cmp_gt(x,y))
	inline	__declare_oper(>,avx8_mask,const long long int& x,const avx512_sint64&y,avx_cmp_gt(avx512_load(x),y))
	inline	__declare_oper(>,avx8_mask,const avx512_sint64&x,const long long int& y,avx_cmp_gt(x,avx512_load(y)))
