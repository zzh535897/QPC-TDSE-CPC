#pragma once

#include <libraries/support_std.h>
//====================================================================================================
//
//  			This file is part of QPC Header Only Library
//
//====================================================================================================

//====================================================================================================
//					
//					Lentz-Thompson Algorithm
//
//	References:
//	[1] A. R. Barnett, D. H. Feng, J. W. Steed & L. J. B. Goldfarb Comput. Phys. Commun, 8, (1974)
//	pp.377-395
//	[2] W. J. Lentz, Appl. Opt. 15,668 (1976) pp.668-671
//	[3] I. J. Thompson & A. R. Barnett, J. Comput. Phys. 64,2 (1986) pp.490-509
//
//	Description:
//	(1) The algorithm evaluates the following type of infinite continued fraction
//
//		H=b0+a1/b1+a2/b2+...
//
//	in a forward way by the iteration for j>=1,i=j-1
//
//		dj:=1/(bj+aj*di);
//		cj:=bj+aj/ci;
//		Hj:=Hi*cj*dj;
//
//	with initial value H0:=b0, d0:=0, c0=b0;
//
//	It may also evaluate H' at the same time by the extra iterations
//
//		dj':=-(bj'+aj'*di+aj*di')/(dj*dj);
//		cj':=bj'+(aj'*ci-aj*ci')/(ci*ci);
//		Hj':=Hi'*cj*dj+Hi*(cj'*dj+cj*dj');
//
//	(2) Cited from ref.[1-3], the content of this file should be considered as their 
//	C++ implementation or translation. 
//
//Created by: Zhao-Han Zhang(张兆涵)  Nov. 10th, 2020
//Updated by: Zhao-Han Zhang(张兆涵)  Dec. 15th, 2022
//Updated by: Zhao-Han Zhang(张兆涵)  Dec. 23th, 2022
//	- fix the bug. when |d0|<eps, the original method use d0:=eps*b0. however, if b0
//	happens to be exact 0, then the code will fail. 
//
//Copyright © 2022 Zhao-Han Zhang
//====================================================================================================

namespace qpc
{
	template<class data_t,class func_t,class ret0_t>
	int	cf_lentz_real	(const func_t& func,const data_t rtol,const size_t imax,ret0_t&& ret0)noexcept
	{
		data_t  eps=std::numeric_limits<data_t>::epsilon();
                data_t  eps2=eps*eps;
                data_t  rtol2=rtol*rtol;
                data_t  a0,b0,c0,d0,h0,delta;
                size_t  i;
               	func(0,a0,b0);//set b0
               	if(norm(b0)<eps2)b0=eps;
		h0=b0;d0=data_t(0);c0=b0;
		for(i=2;i<imax;++i)
		{
			func(i-1,a0,b0);//set ai,bi
			d0=b0+a0*d0;
			c0=b0+a0/c0;
			if(norm(d0)<eps2)d0=eps*(norm(b0)<eps2?eps:b0);
			if(norm(c0)<eps2)c0=eps*(norm(b0)<eps2?eps:b0);
			d0=data_t(1)/d0;
			delta=c0*d0;
			h0=h0*delta;
			if(norm(delta-data_t(1))<rtol2)break;
		}
		ret0=h0;
                if(i>=imax)     return  1;
                else            return  0;
	}//end of cf_lentz_real case 0D

	template<class data_t,class func_t,class ret0_t,class ret1_t>
	int	cf_lentz_real	(const func_t& func,const data_t rtol,const size_t imax,ret0_t&& ret0,ret1_t&& ret1)noexcept
	{
		data_t  eps=std::numeric_limits<data_t>::epsilon();
                data_t  eps2=eps*eps;
                data_t  rtol2=rtol*rtol;
                data_t  a0,b0,c0,d0,h0,a1,b1,c1,d1,h1,delta;
                size_t  i;	
		func(0,a0,a1,b0,b1);
		if(norm(b0)<eps2)b0=eps;
		h0=b0;d0=data_t(0);c0=b0;
		h1=b1;d1=data_t(0);c1=b1;
		for(i=2;i<imax;++i)
		{
			func(i-1,a0,a1,b0,b1);
			d1=-(b1+a1*d0+a0*d1);
			d0=b0+a0*d0;
			c1=b1+(a1*c0-a0*c1)/(c0*c0);
			c0=b0+a0/c0;
			if(norm(d0)<eps2)d0=eps*(norm(b0)<eps2?eps:b0);
			if(norm(c0)<eps2)c0=eps*(norm(b0)<eps2?eps:b0);
			d0=data_t(1)/d0;
			d1=d1*d0*d0;
			delta=c0*d0;
			h1=h1*delta+h0*(c1*d0+c0*d1);
			h0=h0*delta;
			if(norm(delta-data_t(1))<rtol2)break;
		}
		ret0=h0;
		ret1=h1;
		if(i>=imax)     return  1;
		else            return  0;
	}//end of cf_lentz_real case 1D

	template<class data_t,class func_t,class ret0_t,class ret1_t,class ret2_t>
	int	cf_lentz_real	(const func_t& func,const data_t rtol,const size_t imax,ret0_t&& ret0,ret1_t&& ret1,ret2_t&& ret2)noexcept
	{
		data_t  eps=std::numeric_limits<data_t>::epsilon();
                data_t  eps2=eps*eps;
                data_t  rtol2=rtol*rtol;
                data_t  a0,b0,c0,d0,h0,a1,b1,c1,d1,h1,a2,b2,c2,d2,h2,delta;
                size_t  i;	
		func(0,a0,a1,a2,b0,b1,b2);
		if(norm(b0)<eps2)b0=eps;
		h0=b0;d0=data_t(0);c0=b0;
		h1=b1;d1=data_t(0);c1=b1;
		h2=b2;d2=data_t(0);c2=b2;
		for(i=2;i<imax;++i)
		{
			func(i-1,a0,a1,a2,b0,b1,b2);
			d1=-(b1+a1*d0+a0*d1);
			d2=-(b2+a2*d0+a0*d2);
			d0=b0+a0*d0;
			c1=b1+(a1*c0-a0*c1)/(c0*c0);
			c2=b2+(a2*c0-a0*c2)/(c0*c0);
			c0=b0+a0/c0;
			if(norm(d0)<eps2)d0=eps*(norm(b0)<eps2?eps:b0);
			if(norm(c0)<eps2)c0=eps*(norm(b0)<eps2?eps:b0);
			d0=data_t(1)/d0;
			d1=d1*d0*d0;
			d2=d2*d0*d0;
			delta=c0*d0;
			h1=h1*delta+h0*(c1*d0+c0*d1);
			h2=h2*delta+h0*(c2*d0+c0*d2);
			h0=h0*delta;
			if(norm(delta-data_t(1))<rtol2)break;
		}
		ret0=h0;
		ret1=h1;
		ret2=h2;
		if(i>=imax)     return  1;
		else            return  0;
	}//end of cf_lentz_real case 2D

	template<class data_t,class func_t,class ret0_t,class ret1_t,class ret2_t,class ret3_t>
	int	cf_lentz_real	(const func_t& func,const data_t rtol,const size_t imax,ret0_t&& ret0,ret1_t&& ret1,ret2_t&& ret2,ret3_t&& ret3)noexcept
	{
		data_t  eps=std::numeric_limits<data_t>::epsilon();
                data_t  eps2=eps*eps;
                data_t  rtol2=rtol*rtol;
                data_t  a0,b0,c0,d0,h0,a1,b1,c1,d1,h1,a2,b2,c2,d2,h2,a3,b3,c3,d3,h3,delta;
                size_t  i;	
		func(0,a0,a1,a2,a3,b0,b1,b2,b3);
		if(norm(b0)<eps2)b0=eps;
		h0=b0;d0=data_t(0);c0=b0;
		h1=b1;d1=data_t(0);c1=b1;
		h2=b2;d2=data_t(0);c2=b2;
		h3=b3;d3=data_t(0);c3=b3;
		for(i=2;i<imax;++i)
		{
			func(i-1,a0,a1,a2,a3,b0,b1,b2,b3);
			d1=-(b1+a1*d0+a0*d1);
			d2=-(b2+a2*d0+a0*d2);
			d3=-(b3+a3*d0+a0*d3);
			d0=b0+a0*d0;
			c1=b1+(a1*c0-a0*c1)/(c0*c0);
			c2=b2+(a2*c0-a0*c2)/(c0*c0);
			c3=b3+(a3*c0-a0*c3)/(c0*c0);
			c0=b0+a0/c0;
			if(norm(d0)<eps2)d0=eps*(norm(b0)<eps2?eps:b0);
			if(norm(c0)<eps2)c0=eps*(norm(b0)<eps2?eps:b0);
			d0=data_t(1)/d0;
			d1=d1*d0*d0;
			d2=d2*d0*d0;
			d3=d3*d0*d0;
			delta=c0*d0;
			h1=h1*delta+h0*(c1*d0+c0*d1);
			h2=h2*delta+h0*(c2*d0+c0*d2);
			h3=h3*delta+h0*(c3*d0+c0*d3);
			h0=h0*delta;
			if(norm(delta-data_t(1))<rtol2)break;
		}
		ret0=h0;
		ret1=h1;
		ret2=h2;
		ret3=h3;
		if(i>=imax)     return  1;
		else            return  0;
	}//end of cf_lentz_real case 3D

	template<class data_t,class func_t,class ret0_t,class retb_t>
	int	cf_steed_real	(const func_t& func,const data_t rtol,const size_t imax,ret0_t&& ret0,retb_t&& retb)noexcept
	{
		data_t	rtol2=rtol*rtol;
		data_t	eps=std::numeric_limits<data_t>::epsilon();
		data_t	eps2=eps*eps;
		size_t	i;
		data_t 	a0,b0,c0,d0,h0,delta,cap_b;
		func(0,a0,b0);
		if(norm(b0)<eps2)b0=eps;
		h0=b0;d0=data_t(0);c0=b0;cap_b=data_t(1);
		for(i=2;i<imax;++i)
		{
			func(i-1,a0,b0);
			d0=b0+a0*d0;
			c0=b0+a0/c0;
			if(norm(d0)<eps2)d0=eps*(norm(b0)<eps2?eps:b0);
			if(norm(c0)<eps2)c0=eps*(norm(b0)<eps2?eps:b0);
			cap_b*=d0;
			d0=data_t(1)/d0;
			delta=c0*d0;
			h0=h0*delta;
			if(norm(delta-data_t(1))<rtol2)break;
		}
		ret0=h0;
		retb=cap_b;
		if(i>=imax)     return  1;
		else		return	0;
	}//end of cf_steed_real case 1D

	template<class data_t,class func_t,class ret0_t>
	int	cf_lentz_comp	(const func_t& func,const data_t rtol,const size_t imax,ret0_t&& ret0)noexcept
	{
		using   comp_t  =       std::complex<data_t>;	
		data_t	rtol2=rtol*rtol;
		data_t	eps=std::numeric_limits<data_t>::epsilon();
		data_t	eps2=eps*eps*data_t(2);
		size_t	i;
		comp_t 	a0,b0,c0,d0,h0,delta;
		func(0,a0,b0);
		if(norm(b0)<eps2)b0=eps;
		h0=b0;d0=data_t(0);c0=b0;
		for(i=2;i<imax;++i)
		{
			func(i-1,a0,b0);
			d0=b0+a0*d0;
			c0=b0+a0/c0;
			if(norm(d0)<eps2)d0=eps*(norm(b0)<eps2?eps:b0);
			if(norm(c0)<eps2)c0=eps*(norm(b0)<eps2?eps:b0);
			d0=data_t(1)/d0;
			delta=c0*d0;
			h0=h0*delta;
			if(norm(delta-data_t(1))<rtol2)break;
		}
		ret0=h0;
		if(i>=imax)     return  1;
		else		return	0;
	}//end of cf_lentz_comp case 0D

	template<class data_t,class func_t,class ret0_t,class ret1_t>
	int	cf_lentz_comp	(const func_t& func,const data_t rtol,const size_t imax,ret0_t&& ret0,ret1_t&& ret1)noexcept
	{
		using	comp_t	=	std::complex<data_t>;
		data_t	rtol2=rtol*rtol;
		data_t	eps=std::numeric_limits<data_t>::epsilon();
		data_t	eps2=eps*eps*data_t(2);
		size_t	i;
		comp_t 	a0,b0,c0,d0,h0,a1,b1,c1,d1,h1,delta;
		func(0,a0,a1,b0,b1);
		if(norm(b0)<eps2)b0=eps;
		h0=b0;d0=data_t(0);c0=b0;
		h1=b1;d1=data_t(0);c1=b1;
		for(i=2;i<imax;++i)
		{
			func(i-1,a0,a1,b0,b1);
			d1=-(b1+a1*d0+a0*d1);
			d0=b0+a0*d0;
			c1=b1+(a1*c0-a0*c1)/(c0*c0);
			c0=b0+a0/c0;
			if(norm(d0)<eps2)d0=eps*(norm(b0)<eps2?eps:b0);
			if(norm(c0)<eps2)c0=eps*(norm(b0)<eps2?eps:b0);
			d0=data_t(1)/d0;
			d1=d1*d0*d0;
			delta=c0*d0;
			h1=h1*delta+h0*(c1*d0+c0*d1);
			h0=h0*delta;
			if(norm(delta-data_t(1))<rtol2)break;
		}
		ret0=h0;
		ret1=h1;
		if(i>=imax)     return  1;
		else		return	0;
	}//end of cf_lentz_comp case 1D
	
	template<class data_t,class func_t,class ret0_t,class ret1_t,class ret2_t>
	int	cf_lentz_comp	(const func_t& func,const data_t rtol,const size_t imax,ret0_t&& ret0,ret1_t&& ret1,ret2_t&& ret2)noexcept
	{	
		using	comp_t	=	std::complex<data_t>;
		data_t	rtol2=rtol*rtol;
		data_t	eps=std::numeric_limits<data_t>::epsilon();
		data_t	eps2=eps*eps*data_t(2);
		size_t	i;
		comp_t 	a0,b0,c0,d0,h0,a1,b1,c1,d1,h1,a2,b2,c2,d2,h2,delta;
		func(0,a0,a1,a2,b0,b1,b2);
		if(norm(b0)<eps2)b0=eps;
		h0=b0;d0=data_t(0);c0=b0;
		h1=b1;d1=data_t(0);c1=b1;
		h2=b2;d2=data_t(0);c2=b2;
		for(i=2;i<imax;++i)
		{
			func(i-1,a0,a1,a2,b0,b1,b2);
			d1=-(b1+a1*d0+a0*d1);
			d2=-(b2+a2*d0+a0*d2);
			d0=b0+a0*d0;
			c1=b1+(a1*c0-a0*c1)/(c0*c0);
			c2=b2+(a2*c0-a0*c2)/(c0*c0);
			c0=b0+a0/c0;
			if(norm(d0)<eps2)d0=eps*(norm(b0)<eps2?eps:b0);
			if(norm(c0)<eps2)c0=eps*(norm(b0)<eps2?eps:b0);
			d0=data_t(1)/d0;
			d1=d1*d0*d0;
			d2=d2*d0*d0;
			delta=c0*d0;
			h1=h1*delta+h0*(c1*d0+c0*d1);
			h2=h2*delta+h0*(c2*d0+c0*d2);
			h0=h0*delta;
			if(norm(delta-data_t(1))<rtol2)break;
		}
		ret0=h0;
		ret1=h1;
		ret2=h2;
		if(i>=imax)     return  1;
		else		return	0;	
	}//end of cf_lentz_comp case 2D

	template<class data_t,class func_t,class ret0_t,class retb_t>
	int	cf_steed_comp	(const func_t& func,const data_t rtol,const size_t imax,ret0_t&& ret0,retb_t&& retb,size_t& reti)noexcept
	{
		using   comp_t  =       std::complex<data_t>;	
		data_t	rtol2=rtol*rtol;
		data_t	eps=std::numeric_limits<data_t>::epsilon();
		data_t	eps2=eps*eps*data_t(2);
		size_t	i;
		comp_t 	a0,b0,c0,d0,h0,delta,cap_b;
		func(0,a0,b0);
		if(norm(b0)<eps2)b0=eps;
		h0=b0;d0=data_t(0);c0=b0;cap_b=data_t(1);
		for(i=2;i<imax;++i)
		{
			func(i-1,a0,b0);
			d0=b0+a0*d0;
			c0=b0+a0/c0;
			if(norm(d0)<eps2)d0=eps*(norm(b0)<eps2?eps:b0);
			if(norm(c0)<eps2)c0=eps*(norm(b0)<eps2?eps:b0);
			cap_b*=d0;
			d0=data_t(1)/d0;
			delta=c0*d0;
			h0=h0*delta;
			if(norm(delta-data_t(1))<rtol2)break;
		}
		ret0=h0;
		retb=cap_b;
		reti=i;
		if(i>=imax)     return  1;
		else		return	0;
	}//end of cf_steed_comp case 1D
}//end of qpc
