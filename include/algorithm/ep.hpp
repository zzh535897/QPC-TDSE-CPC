#pragma once
#include <libraries/support_std.h>

//====================================================================================================
//
//                      This file is part of QPC Header Only Library
//
//====================================================================================================


//====================================================================================================
//
//
//			    		Wynn's epsilon algorithm
//
//	References:
//		[1] P. Wynn, SIAM J. Numer. Anal. 3(1), 91–122 (1966)
//
//	Description:
//		(1) ep(i,j+1)=ep(i+1,j-1)+1/(ep(i+1,j)-ep(i,j))
//
//		with ep(i, 0)= cumsum(a,i)
//		     ep(i,-1)= 0
//     		this would create a pade approximant table for sum(a):
//
//    		ep(0,0) -- ep(0,1) -- ep(0,2) -- ... ep(0,2*jmax) -- ep(0,2*jmax+1)
//	        	 /	   /	      /
//	       		/	  /	     /
//     		ep(1,0) -- ep(1,1) -- ep(1,2) -- ...  
//	        	 /	   /	      /
//	       		/	  /	     /
//     		ep(2,0) -- ep(2,1) -- ep(2,2) -- ...
//       	:
//       	:
//       	:
//       	:
//       	:	  ep(imax-2,1)...
//       	:	 /
//       	:	/
//     		ep(imax-1,0)
//
//Updated by: Zhao-Han Zhang(张兆涵)  Dec. 15th, 2022
//
//Copyright © 2022 Zhao-Han Zhang
//====================================================================================================

namespace qpc
{
	template<class data_t,class func_t>
	int 	pade_approximant
	(
		data_t*	work,	//the workspace. at least length imax*3
		data_t& retn,
		func_t&&func,  	//the expression for A. called only once for each i.
		size_t 	imax,
		size_t	jmax
	)noexcept
	{
		data_t*	ep0	=	work;
		data_t*	ep1	=	work+imax;
		data_t*	ep2	=	work+imax*2ul;
			ep0[0]	= 	func(0);
		data_t	tmp;
		for(size_t i=1;i<imax;++i)//fill initial values ep0=S and ep1=1/A
		{
			tmp	=	func(i);	
			ep0[i]	=	ep0[i-1]+tmp;//could be trunction error
			ep1[i-1]=	data_t(1)/tmp;
		}	
		size_t	len	=	imax-1ul;
		for(size_t i=0;i<len-1;++i)//the first ep2
		{
			tmp	=	data_t(1)/(ep1[i+1]-ep1[i]);
			if(std::isnan(real(tmp)))tmp=data_t(0);
			ep2[i]	=	ep0[i+1]+tmp;
		}	ep2[len-1]=	ep0[len];
		for(size_t j=1;j<jmax;++j)
		{
			std::swap(ep0,ep1);
			std::swap(ep1,ep2);
			--len;			//move to calc next ep (non-pade colomn)
			for(size_t i=0;i<len;++i)
                	{
                        	tmp     =       data_t(1)/(ep1[i+1]-ep1[i]);//ep1[len] is legal
	                        if(std::isnan(real(tmp)))tmp=data_t(0);
        	                ep2[i]  =       ep0[i+1]+tmp;//ep0[len] is legal
                	}
			std::swap(ep0,ep1);
			std::swap(ep1,ep2);	//move to calc next ep (pade colomn)
			for(size_t i=0;i<len-1;++i)
                	{
                        	tmp     =       data_t(1)/(ep1[i+1]-ep1[i]);
                        	if(std::isnan(real(tmp)))tmp=data_t(0);
                        	ep2[i]  =       ep0[i+1]+tmp;
                	}    	ep2[len-1]=     ep0[len];
		}
		retn	=	ep2[len-jmax*2ul>len?0:len-jmax*2ul];//empirically better
		return 	0;//FIXME add convergency check
	}//end of pade_approximant

	template<class data_t>
	int 	pade_approximant2
	(
		data_t*	work,	//the workspace. at least length imax*3, and the leading imax terms has already been filled with Sm
		data_t& retn,
		size_t 	imax,
		size_t	jmax
	)noexcept
	{
		data_t*	ep0	=	work;
		data_t*	ep1	=	work+imax;
		data_t*	ep2	=	work+imax*2ul;
		data_t	tmp;
		for(size_t i=1;i<imax;++i)//fill initial values ep1=1/A
		{
			ep1[i-1]=	data_t(1)/(ep0[i]-ep0[i-1]);//could be cancellation error
		}	
		size_t	len	=	imax-1ul;
		for(size_t i=0;i<len-1;++i)//the first ep2
		{
			tmp	=	data_t(1)/(ep1[i+1]-ep1[i]);
			if(std::isnan(real(tmp)))tmp=data_t(0);
			ep2[i]	=	ep0[i+1]+tmp;
		}	ep2[len-1]=	ep0[len];
		for(size_t j=1;j<jmax;++j)
		{
			std::swap(ep0,ep1);
			std::swap(ep1,ep2);
			--len;			//move to calc next ep (non-pade colomn)
			for(size_t i=0;i<len;++i)
                	{
                        	tmp     =       data_t(1)/(ep1[i+1]-ep1[i]);//ep1[len] is legal
	                        if(std::isnan(real(tmp)))tmp=data_t(0);
        	                ep2[i]  =       ep0[i+1]+tmp;//ep0[len] is legal
                	}
			std::swap(ep0,ep1);
			std::swap(ep1,ep2);	//move to calc next ep (pade colomn)
			for(size_t i=0;i<len-1;++i)
                	{
                        	tmp     =       data_t(1)/(ep1[i+1]-ep1[i]);
                        	if(std::isnan(real(tmp)))tmp=data_t(0);
                        	ep2[i]  =       ep0[i+1]+tmp;
                	}    	ep2[len-1]=     ep0[len];
		}
		retn	=	ep2[len-jmax*2ul>len?0:len-jmax*2ul];//empirically better
		return 	0;//FIXME add convergency check
	}//end of pade_approximant2

	template<class data_t>
	int 	pade_approximant3
	(
		data_t*	work,	//the workspace. at least length imax*3, and the leading 2*imax+1 terms has already been filled with S(m), 1/a(m+1)
		data_t& retn,
		size_t 	imax,
		size_t	jmax
	)noexcept
	{
		data_t*	ep0	=	work;
		data_t*	ep1	=	work+imax;
		data_t*	ep2	=	work+imax*2ul;
		data_t	tmp;
		size_t	len	=	imax-1ul;
		for(size_t i=0;i<len-1;++i)//the first ep2
		{
			tmp	=	data_t(1)/(ep1[i+1]-ep1[i]);
			if(std::isnan(real(tmp)))tmp=data_t(0);
			ep2[i]	=	ep0[i+1]+tmp;
		}	ep2[len-1]=	ep0[len];
		for(size_t j=1;j<jmax;++j)
		{
			std::swap(ep0,ep1);
			std::swap(ep1,ep2);
			--len;			//move to calc next ep (non-pade colomn)
			for(size_t i=0;i<len;++i)
                	{
                        	tmp     =       data_t(1)/(ep1[i+1]-ep1[i]);//ep1[len] is legal
	                        if(std::isnan(real(tmp)))tmp=data_t(0);
        	                ep2[i]  =       ep0[i+1]+tmp;//ep0[len] is legal
                	}
			std::swap(ep0,ep1);
			std::swap(ep1,ep2);	//move to calc next ep (pade colomn)
			for(size_t i=0;i<len-1;++i)
                	{
                        	tmp     =       data_t(1)/(ep1[i+1]-ep1[i]);
                        	if(std::isnan(real(tmp)))tmp=data_t(0);
                        	ep2[i]  =       ep0[i+1]+tmp;
                	}    	ep2[len-1]=     ep0[len];
		}
		retn	=	ep2[len-jmax*2ul>len?0:len-jmax*2ul];//empirically better
		return 	0;//FIXME add convergency check
	}//end of pade_approximant3
}//end of qpc
