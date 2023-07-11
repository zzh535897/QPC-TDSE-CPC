#pragma once

#include <libraries/support_std.h>
//====================================================================================================
//
//                      This file is part of QPC Header Only Library
//
//====================================================================================================

//====================================================================================================
//
//					Newton-Raphson Algorithm
//
//(1) The algorithm solves equations in the following form:
//	
//	f(x1,x2,...)=0;
//
// by the iteration
//
// 	xj:=xi-f'(xi)\f(xi);
//
//  For one dimensional case, f'\f is computed in a straight forward way, while for two dimensional 
//  case, it is evaluated by the inverse of Jacobian Matrix:
//
//  	f'\f:=[df1/dx1,df1/dx2;df2/dx1,df2/dx2;]^-1*[f1,f2];
//
//Updated by: Zhao-Han Zhang(张兆涵)  Dec. 15th, 2022
//
// Copyright © 2022 Zhao-Han Zhang
//====================================================================================================

namespace qpc
{
	
	template<class data_t>
	static constexpr data_t rsqrtinf()noexcept
	{
		if constexpr(std::is_same_v<data_t,float>)
		{
			return 	1.844674407370955e+19f;//2^64
		}
		if constexpr(std::is_same_v<data_t,double>)
		{
			return	1.340780792994260e+154;//2^512
		}
		if constexpr(std::is_same_v<data_t,long double>)
		{
			return  1.0907481356194159295e+2466L;//2^8192
		}
	}//end of rsqrtinf

	template<class data_t,class func_t,class updt_t>
	int	sv_newton	
	(
		const func_t& f,
		const updt_t& u,
		const data_t rtol,
		const size_t imax
	)noexcept
	{
		using comp_t	=	std::complex<data_t>;
		bool_t  flag    =       1;
               	size_t  step    =       0;
                data_t	rtol2 	=       rtol*rtol;
		while(flag&&step++<imax)
		{
			if constexpr(std::is_invocable_v<func_t,data_t&,data_t&>)//1D real case
			{
				data_t	vf1,df1;
				f(vf1,df1);
				auto    del1    =       -vf1;
				auto    det     =       df1;
				flag            =       u(del1/=det)>rtol2;
			}else
			if constexpr(std::is_invocable_v<func_t,comp_t&,comp_t&>)//1D complex case
			{
				comp_t	vf1,df1;
				f(vf1,df1);
				auto	del1	=	-vf1;
				auto	det	=	df1;
				if(std::isinf(norm(det)))//avoid overflow
				{
					det	=	det *rsqrtinf<data_t>();
					del1	=	del1*rsqrtinf<data_t>();
				}
				flag		=	u(del1/=det)>rtol2;
			}else
			if constexpr(std::is_invocable_v<func_t,data_t&,data_t&,data_t&,data_t&,data_t&,data_t&>)//2D real case
			{
				data_t	vf1,vf2,df11,df12,df21,df22;
				f(vf1,vf2,df11,df12,df21,df22);
				auto    del1    =       (vf2*df12-vf1*df22);
				auto    del2    =       (vf1*df21-vf2*df11);
				auto    det     =       (df11*df22-df12*df21);
				flag            =       u(del1/=det,del2/=det)>rtol2;				
			}else 
			if constexpr(std::is_invocable_v<func_t,comp_t&,comp_t&,comp_t&,comp_t&,comp_t&,comp_t&>)//2D complex case
			{
				comp_t	vf1,vf2,df11,df12,df21,df22;
				f(vf1,vf2,df11,df12,df21,df22);
				auto	del1	=	(vf2*df12-vf1*df22);	
				auto	del2	=	(vf1*df21-vf2*df11);
				auto	det	=	(df11*df22-df12*df21);
				if(std::isinf(norm(det)))//avoid overflow
				{
					det     =       det*rsqrtinf<data_t>();
					del1	=	del1*rsqrtinf<data_t>();
					del2	=	del2*rsqrtinf<data_t>();
				}
				flag		=	u(del1/=det,del2/=det)>rtol2;
			}//end of case 2
		}
		if(step>=imax)	return 1;
              	else 		return 0;
	}//end of sv_newton

	template<class data_t,class func_t>
	int	sv_secant	
	(
		const func_t& f,
		std::complex<data_t>& x0,
		std::complex<data_t>& x1,
		const data_t rtol,
		const size_t imax
	)noexcept
	{
		using comp_t	=	std::complex<data_t>;
		bool_t  flag    =       1;
               	size_t  step    =       0;
        	data_t	rtol2 	=       rtol*rtol;
		comp_t	f0,f1,del1;
		f(x0,f0);
		while(flag&&step++<imax)
		{
			f(x1,f1);
			del1	=	-f1*(x1-x0)/(f1-f0);
			x0	=	x1;
			f0	=	f1;
			x1	+=	del1;
			flag	=	norm(del1/x1)>rtol2;
		}
		if(step>=imax)  return 	1;
		else		return 	0;
	}//end of sv_secant
	
}//end of qpc

