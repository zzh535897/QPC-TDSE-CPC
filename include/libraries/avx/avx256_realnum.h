#pragma once
//*==================================================================================================*//
//*avx(256bit) real number
//*==================================================================================================*//

/*(1) load and cast*/
	static	inline	avx256_double	avx256_load	(const double x)noexcept
	{
		return 	avx256_double{_mm256_set1_pd(x)};
	}
	static  inline  avx256_double	avx256_loadu	(const double*x,const size_t i)noexcept
	{
		return  avx256_double{_mm256_loadu_pd(x+i)};
	}
/*(2) arithmatic*/
	static 	inline	avx256_double	avx_add		(const avx256_double& x,const avx256_double& y)noexcept
	{
		return	{_mm256_add_pd		(x,y)};
	}
	static	inline	avx256_double	avx_sub		(const avx256_double& x,const avx256_double& y)noexcept
	{
		return	{_mm256_sub_pd		(x,y)};
	}
	static	inline	avx256_double	avx_mul		(const avx256_double& x,const avx256_double& y)noexcept
	{
		return	{_mm256_mul_pd		(x,y)};
	}
	static	inline	avx256_double	avx_div		(const avx256_double& x,const avx256_double& y)noexcept
	{
		return 	{_mm256_div_pd		(x,y)};
	}
	static	inline	avx256_double	avx_mulc	(const avx256_double& x,const avx256_double& y)noexcept
	{
		return 	avx_mul(x,y);	
	}
	static	inline	avx256_double	avx_mulc	(const double x,const avx256_double& y)noexcept
	{
		return 	avx_mul(avx256_double{x,x,x,x},y);
	}
	static	inline	avx256_double	avx_mulc	(const avx256_double& x,const double y)noexcept
	{
		return 	avx_mul(avx256_double{y,y,y,y},x);
	}
/*(3) operator*/
	inline	__declare_oper(+,avx256_double,const avx256_double&x,const avx256_double&y,avx_add(x,y))
	inline	__declare_oper(+,avx256_double,const avx256_double&x,const double y,avx_add(x,avx256_double{y,y,y,y}))
	inline	__declare_oper(+,avx256_double,const double x,const avx256_double&y,avx_add(y,avx256_double{x,x,x,x}))

	inline	__declare_oper(-,avx256_double,const avx256_double&x,const avx256_double&y,avx_sub(x,y))
	inline	__declare_oper(-,avx256_double,const avx256_double&x,const double y,avx_sub(x,avx256_double{y,y,y,y}))
	inline	__declare_oper(-,avx256_double,const double x,const avx256_double&y,avx_sub(avx256_double{x,x,x,x},y))

	inline	__declare_oper(*,avx256_double,const avx256_double&x,const avx256_double&y,avx_mul(x,y))
	inline	__declare_oper(*,avx256_double,const avx256_double&x,const double y,avx_mul(x,avx256_double{y,y,y,y}))
	inline	__declare_oper(*,avx256_double,const double x,const avx256_double&y,avx_mul(avx256_double{x,x,x,x},y))
	
	inline	__declare_oper(/,avx256_double,const avx256_double&x,const avx256_double&y,avx_div(x,y))
	inline	__declare_oper(/,avx256_double,const avx256_double&x,const double y,avx_div(x,avx256_double{y,y,y,y}))
	inline	__declare_oper(/,avx256_double,const double x,const avx256_double&y,avx_div(avx256_double{x,x,x,x},y))

	inline	avx256_double& avx256_double::operator = (const double& rhs)noexcept {data[0]=rhs;data[1]=rhs;data[2]=rhs;data[3]=rhs;return *this;}
