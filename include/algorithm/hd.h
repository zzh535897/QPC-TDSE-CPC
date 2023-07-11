#pragma once

#include <libraries/support_std.h>

//====================================================================================================
//
//			This file is a part of QPC Header Only Library
//
//====================================================================================================

//====================================================================================================
//
//					  Hill Determiant	
//
//Description:
//	(1) The algorithm evaluates the Hill-type determinant for 3 term recurrence relation (3RR):
//
//		alpha_{i}*c_{i+1}+beta_{i}*c_{i}+gamma_{i}*c_{i-1} = 0;
//
// 	by simple LU factorization without pivioting. The inparsed functor f should return
//  
//   		f(i+nmax)=alpha(i)*gamma(i+1)/beta(i)/beta(i+1)
//
// 	where nmax is the indent value. For one-end case, n=nmax+1; For two-end case, n=2*nmax+1.
//
//Updated by: Zhao-Han Zhang(张兆涵)  Dec. 15th, 2022
//
//Copyright © 2022 Zhao-Han Zhang
//====================================================================================================

namespace qpc
{	
	template<class data_t,class func_t>
	data_t	hd_tridiag	(const func_t& f,long n)noexcept
	{
		data_t	d	=	data_t(1);
		data_t	r	=	data_t(1);
		for(long i=1;i<n;++i)
		{
			d	=	data_t(1)-f(i)/d;
			r	*=	d;
		}
		return	r;	
	}//end of hd_tridiag
}//end of qpc
