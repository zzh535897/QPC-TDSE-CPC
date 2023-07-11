#pragma once

#include <libraries/support_std.h>

#include <algorithm/pd.h>

//====================================================================================================
//
//                      This file is part of QPC Header Only Library
//
//====================================================================================================

//===============================================================================================
//
//		 Gauss Hypergeometric function 2F1(a,b,c,z) for |z|<1.
//
//===============================================================================================

namespace qpc
{
	
	static constexpr size_t hypergeom_2f1_taylor_imax	=	4000ul;
	static constexpr size_t hypergeom_2f1_taylor_jmax	=	240ul;	
	static constexpr size_t hypergeom_2f1_taylor_wksz	=	hypergeom_2f1_taylor_imax*3ul;

	static constexpr double	hypergeom_2f1_taylor_rtol	=	1e-30;

	template<class data_t>
	inline 	int 	hypergeom_2f1_taylor	(const data_t& a,const data_t& b,const data_t& c,const data_t& z,data_t& f)noexcept
	{
		data_t	r	=	a*b/c*z;
			f	=	data_t(1)+r;
		size_t	j;
    		for(j=1;j<hypergeom_2f1_taylor_imax;++j)
		{
			r	*=	(a+data_t(j))*(b+data_t(j))/data_t(j+1)/(c+data_t(j))*z;
			f	+=	r;
			if(norm(r/f)<hypergeom_2f1_taylor_rtol)break;
		}
		if(j>=hypergeom_2f1_taylor_imax)return 1;
		else				return 0;
	}//end of hypergeom_2f1_taylor

	template<class data_t>
	inline 	int	hypergeom_2f1_acpade	(const data_t& a,const data_t& b,const data_t& c,const data_t& z,data_t& f,data_t* work)noexcept
	{
		data_t	r	=	data_t(1);
		auto	func	=	[&](const size_t j)//called once for each j
		{
			return r*=(a+data_t(j))*(b+data_t(j))/data_t(j+1)/(c+data_t(j))*z;
		};
		int	info	=	pade_approximant<data_t>(work,f,func,hypergeom_2f1_taylor_imax,hypergeom_2f1_taylor_jmax);
			f	+=	data_t(1);
		return	info;
	}//end of hypergeom_2f1_padeac

	template<class data_t>
	inline 	int	hypergeom_2f1_quotie	(const data_t& a,const data_t& b,const data_t& c,const data_t& z,data_t& f)noexcept//not a good method
	{
		data_t	p	=	data_t(0);
		data_t	q	=	data_t(1);
		data_t	r	=	data_t(1);
			f	=	data_t(1);
		data_t	t;
		size_t	j;
		for(j=1;j<hypergeom_2f1_taylor_imax;++j)
		{
			data_t jm	=	data_t(j-1);
			data_t cm	=	data_t(j)*(c+jm);
			p	=	cm*(p+q);
			q	*=	(a+jm)*(b+jm)*z;
			r	*=	cm;	
			t	=	(p+q)/r;
			if(norm(t/f-data_t(1))<hypergeom_2f1_taylor_rtol)break;
			f	=	t;
		}
		if(j>=hypergeom_2f1_taylor_imax)return 1;
                else                            return 0;
	}//end of hypergeom_2f1_quotie

	
	template<class data_t>
	inline 	int	hypergeom_2f1		(const data_t& a,const data_t& b,const data_t& c,const data_t& z,data_t& f,data_t* w)noexcept
	{
		auto zn=norm(z);
		if(zn>1.0)//XXX improve me
		{
			return 	-9999;
		}else if(zn>0.81)
		{
			return 	hypergeom_2f1_acpade<data_t>(a,b,c,z,f,w);
		}else
		{
			return	hypergeom_2f1_taylor<data_t>(a,b,c,z,f);
		}
	}//end of hypergeom_2f1
}//end of qpc
