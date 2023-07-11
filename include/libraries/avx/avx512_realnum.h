#pragma once
//*==================================================================================================*//
//*avx(512bit) real number
//*==================================================================================================*//

/*(1) load and cast*/
	static 	inline	avx512_double	avx512_load	(const double x)noexcept
	{
		return	avx512_double{_mm512_set1_pd(x)};
	}
	static	inline	avx512_double	avx512_loadu	(const double*x,const size_t i)noexcept
	{
		return 	avx512_double{_mm512_loadu_pd(x+i)};
	}
	static 	inline	avx512_double	avx512_loadu	(const double*x,const avx512_sint64 i)noexcept
	{
		return 	avx512_double{_mm512_i64gather_pd(i,x,8)};
	}
        static  inline  avx512_double  	avx512_loadu    (const double*x,const size_t*i)noexcept
        {
                return  avx512_double{_mm512_i64gather_pd(_mm512_load_epi64(i),x,8)};//FIXME loadu
        }
	static	inline	void		avx512_dumpu	(const avx512_double& v,double*x,const size_t i)noexcept
	{
		_mm512_storeu_pd(x+i,v);
	}
	static	inline	void		avx512_dumpu	(const avx512_double& v,double*x,const size_t*i)noexcept
	{
		_mm512_i64scatter_pd(x,_mm512_load_epi64(i),v,8);//FIXME loadu
	}
/*(2) arithmatic*/
	static 	inline	avx512_double	avx_add		(const avx512_double& x,const avx512_double& y)noexcept
	{
		return	{_mm512_add_pd		(x,y)};
	}
	static	inline	avx512_double	avx_addz	(const avx512_double& x,const avx512_double& y,const avx8_mask& m)noexcept
	{
		return	{_mm512_maskz_add_pd	(m,x,y)};	
	}
	static	inline	avx512_double	avx_sub		(const avx512_double& x,const avx512_double& y)noexcept
	{
		return	{_mm512_sub_pd		(x,y)};
	}
	static	inline	avx512_double	avx_subz	(const avx512_double& x,const avx512_double& y,const avx8_mask& m)noexcept
	{
		return  {_mm512_maskz_sub_pd    (m,x,y)};
	}
	static	inline	avx512_double	avx_mul		(const avx512_double& x,const avx512_double& y)noexcept
	{
		return	{_mm512_mul_pd		(x,y)};
	}
	static	inline  avx512_double   avx_mulz     	(const avx512_double& x,const avx512_double& y,const avx8_mask& m)noexcept
	{
		return 	{_mm512_maskz_mul_pd    (m,x,y)};
	}
	static	inline	avx512_double	avx_div		(const avx512_double& x,const avx512_double& y)noexcept
	{
		return 	{_mm512_div_pd		(x,y)};
	}
	static	inline  avx512_double   avx_divz     	(const avx512_double& x,const avx512_double& y,const avx8_mask& m)noexcept
        {
                return  {_mm512_maskz_div_pd    (m,x,y)};
        }
/*(3) logistics*/
	static	inline	avx8_mask	avx_cmp_le	(const avx512_double& x,const avx512_double& y)noexcept
	{
		return	{_mm512_cmp_pd_mask	(x,y,_CMP_LE_OQ)};
	}/*<=*/
	static	inline	avx8_mask	avx_cmp_eq	(const avx512_double& x,const avx512_double& y)noexcept
	{
		return 	{_mm512_cmp_pd_mask	(x,y,_CMP_EQ_OQ)};
	}/*==*/
	static	inline	avx8_mask	avx_cmp_ge	(const avx512_double& x,const avx512_double& y)noexcept
	{
		return 	{_mm512_cmp_pd_mask	(x,y,_CMP_GE_OQ)};
	}/*>=*/
	static	inline	avx8_mask	avx_cmp_ne	(const avx512_double& x,const avx512_double& y)noexcept
	{
		return 	{_mm512_cmp_pd_mask	(x,y,_CMP_NEQ_UQ)};
	}/*!=*/
	static	inline	avx8_mask	avx_cmp_lt	(const avx512_double& x,const avx512_double& y)noexcept
	{
		return 	{_mm512_cmp_pd_mask	(x,y,_CMP_LT_OQ)};
	}/*<*/
	static	inline	avx8_mask	avx_cmp_gt	(const avx512_double& x,const avx512_double& y)noexcept
	{
		return	{_mm512_cmp_pd_mask	(x,y,_CMP_GT_OQ)};
	}/*>*/
/*(4) functions*/
	static	inline	avx512_double	avx_abs		(const avx512_double& x)noexcept
	{
		return 	{_mm512_abs_pd(x)};
	}
	static	inline	avx512_double	avx_min		(const avx512_double& x,const avx512_double& y)noexcept
	{
		return 	{_mm512_min_pd(x,y)};
	}
	static	inline	avx512_double	avx_max		(const avx512_double& x,const avx512_double& y)noexcept
	{
		return 	{_mm512_max_pd(x,y)};
	}
	static	inline	double		avx_minr	(const avx512_double& x)noexcept
	{
		return 	_mm512_reduce_min_pd(x);
	}
	static	inline	double		avx_maxr	(const avx512_double& x)noexcept
	{
		return 	_mm512_reduce_max_pd(x);
	}
	static	inline	double		avx_sum		(const avx512_double& x)noexcept
	{
		return 	_mm512_reduce_add_pd(x);
	}
	static	inline	avx512_double	avx_normc	(const avx512_double& x)noexcept
	{
		return 	{_mm512_mul_pd(x,x)};
	}
	static	inline	double		avx_norm	(const avx512_double& x)noexcept
	{
		return	_mm512_reduce_add_pd(_mm512_mul_pd(x,x));
	}
	
/*(5) operator*/
	inline	 avx512_double::operator 	avx512_sint64()noexcept		{return	{_mm512_cvt_roundpd_epi64(data,_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC)};}
	inline	 avx512_double::operator const	avx512_sint64()const noexcept	{return {_mm512_cvt_roundpd_epi64(data,_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC)};}

	inline	__declare_oper(+,avx512_double,const avx512_double&x,const avx512_double& y,avx_add(x,y))
	inline	__declare_oper(+,avx512_double,const        double&x,const avx512_double& y,avx_add(avx512_load(x),y))
	inline	__declare_oper(+,avx512_double,const avx512_double&x,const        double& y,avx_add(x,avx512_load(y)))

	inline	__declare_oper(-,avx512_double,const avx512_double&x,const avx512_double& y,avx_sub(x,y))
	inline	__declare_oper(-,avx512_double,const        double&x,const avx512_double& y,avx_sub(avx512_load(x),y))
	inline	__declare_oper(-,avx512_double,const avx512_double&x,const        double& y,avx_sub(x,avx512_load(y)))

	inline	__declare_oper(*,avx512_double,const avx512_double&x,const avx512_double& y,avx_mul(x,y))
	inline	__declare_oper(*,avx512_double,const        double&x,const avx512_double& y,avx_mul(avx512_load(x),y))
	inline	__declare_oper(*,avx512_double,const avx512_double&x,const        double& y,avx_mul(x,avx512_load(y)))
	
	inline	__declare_oper(/,avx512_double,const avx512_double&x,const avx512_double& y,avx_div(x,y))
	inline	__declare_oper(/,avx512_double,const        double&x,const avx512_double& y,avx_div(avx512_load(x),y))
	inline	__declare_oper(/,avx512_double,const avx512_double&x,const        double& y,avx_div(x,avx512_load(y)))

	inline	__declare_oper(<=,avx8_mask,const avx512_double&x,const avx512_double& y,avx_cmp_le(x,y))
	inline	__declare_oper(<=,avx8_mask,const avx512_double&x,const        double& y,avx_cmp_le(x,avx512_load(y)))	
	inline	__declare_oper(<=,avx8_mask,const        double&x,const avx512_double& y,avx_cmp_le(avx512_load(x),y))

	inline	__declare_oper(>=,avx8_mask,const avx512_double&x,const avx512_double& y,avx_cmp_ge(x,y))
	inline	__declare_oper(>=,avx8_mask,const avx512_double&x,const        double& y,avx_cmp_ge(x,avx512_load(y)))	
	inline	__declare_oper(>=,avx8_mask,const        double&x,const avx512_double& y,avx_cmp_ge(avx512_load(x),y))

	inline	__declare_oper(==,avx8_mask,const avx512_double&x,const avx512_double& y,avx_cmp_eq(x,y))
	inline	__declare_oper(==,avx8_mask,const avx512_double&x,const        double& y,avx_cmp_eq(x,avx512_load(y)))	
	inline	__declare_oper(==,avx8_mask,const        double&x,const avx512_double& y,avx_cmp_eq(avx512_load(x),y))
	
	inline	__declare_oper(!=,avx8_mask,const avx512_double&x,const avx512_double& y,avx_cmp_ne(x,y))
	inline	__declare_oper(!=,avx8_mask,const avx512_double&x,const        double& y,avx_cmp_ne(x,avx512_load(y)))	
	inline	__declare_oper(!=,avx8_mask,const        double&x,const avx512_double& y,avx_cmp_ne(avx512_load(x),y))

	inline	__declare_oper(>,avx8_mask,const avx512_double&x,const avx512_double& y,avx_cmp_gt(x,y))
	inline	__declare_oper(>,avx8_mask,const avx512_double&x,const        double& y,avx_cmp_gt(x,avx512_load(y)))	
	inline	__declare_oper(>,avx8_mask,const        double&x,const avx512_double& y,avx_cmp_gt(avx512_load(x),y))

	inline	__declare_oper(<,avx8_mask,const avx512_double&x,const avx512_double& y,avx_cmp_lt(x,y))
	inline	__declare_oper(<,avx8_mask,const avx512_double&x,const        double& y,avx_cmp_lt(x,avx512_load(y)))	
	inline	__declare_oper(<,avx8_mask,const        double&x,const avx512_double& y,avx_cmp_lt(avx512_load(x),y))

	inline	avx512_double& avx512_double::operator = (const double& rhs)noexcept {data=_mm512_set1_pd(rhs);return *this;}

/*(6) rounding*/
	static	inline	avx512_double	round		(const avx512_double& x)noexcept
	{
		return 	{_mm512_roundscale_pd(x,_MM_FROUND_TO_NEAREST_INT)};
	}
	static	inline	avx512_double	floor		(const avx512_double& x)noexcept
	{
		return 	{_mm512_roundscale_pd(x,_MM_FROUND_TO_NEG_INF)};
	}
	static	inline	avx512_double	ceil		(const avx512_double& x)noexcept
	{
		return 	{_mm512_roundscale_pd(x,_MM_FROUND_TO_POS_INF)};
	}	
	static	inline	avx512_double	trunc		(const avx512_double& x)noexcept
	{
		return 	{_mm512_roundscale_pd(x,_MM_FROUND_TO_ZERO)};
	}
	static	inline	avx512_sint64	lround		(const avx512_double& x)noexcept
	{
		return	{_mm512_cvt_roundpd_epi64(x,_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC)};
	}
	static	inline	avx512_sint64	lfloor		(const avx512_double& x)noexcept
	{
		return 	{_mm512_cvt_roundpd_epi64(x,_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)};
	}
	static	inline	avx512_sint64	lceil		(const avx512_double& x)noexcept
	{
		return 	{_mm512_cvt_roundpd_epi64(x,_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)};
	}
	static	inline	avx512_sint64	ltrunc		(const avx512_double& x)noexcept
	{
		return	{_mm512_cvt_roundpd_epi64(x,_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)};
	}
