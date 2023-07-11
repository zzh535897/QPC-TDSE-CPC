#pragma once
#include <libraries/support_std.h>
//====================================================================================================
//
//                      This file is part of QPC Header Only Library
//
//====================================================================================================

//====================================================================================================
//	
//				Complex-Value Log-Gamma Function
//
//	References:
//		[1] W. J. Cody and K. E. Hillstrom, 'Chebyshev Approximations for
//		the Natural Logarithm of the Gamma Function,' Math. Comp. 21,
//		1967, pp. 198-203.
//		[2] K. E. Hillstrom, ANL/AMD Program ANLC366S, DGAMMA/DLGAMA, May,
//		1969.
//		[3] Hart, Et. Al., Computer Approximations, Wiley and sons, New
//		York, 1968.
//====================================================================================================

namespace qpc
{

	template<class data_t>
	static constexpr auto	gammalnc_coef	=	std::array<data_t,10ul>
	{
		 data_t(1)	/	data_t(12),
		-data_t(1)	/	data_t(360),
		 data_t(1)	/	data_t(1260),
		-data_t(1)	/	data_t(1680),
		 data_t(1)	/	data_t(1188),
		-data_t(691)	/	data_t(360360),
		 data_t(1)	/	data_t(156),
		-data_t(4521241)/	data_t(153000000),
		 data_t(43867)	/	data_t(244188),
		-data_t(174611)	/	data_t(125400)
	};

	template<class data_t>
	static constexpr auto	halflog2pi	=	data_t(0.91893853320467274178);//0.5*log(2*pi)

	template<class data_t>
	void	gammalnc	(const std::complex<data_t>& z,data_t* lnr,data_t* arg)noexcept
	{
		data_t 	x0,x1,z1,th,tmp;
		long	na;
		data_t 	x	=	real(z);
		data_t 	y	=	imag(z);
		data_t& gr	=	*lnr;
		data_t&	gi	=	*arg;
		
    		if(y == data_t(0)&&floor(x) == x &&x <= data_t(0))//negative integer
		{
			gr	=	std::numeric_limits<data_t>::infinity();
			gi	=	data_t(0);
			return;
		}else if(x<data_t(0))
		{
        		x1	=	x;
        		x	=	-x;
			y	=	-y;
		}else	x1	=	data_t(0);//to make x1<0 false in the later code

    		if(x <= data_t(7))
		{
        		na	=	(long)(data_t(7)-x);//na will not be larger than 7
        		x0	=	x+na;
    		}else	x0	=	x;//leave na unset
		//-----------------------------------------------------
    			z1	=	sqrt(x0*x0+y*y);
    			th	=	atan(y/x0);
			tmp	=	log(z1);
    			gr	=	(x0-data_t(0.5))*tmp-th*y-x0+halflog2pi<data_t>;
    			gi	=	(x0-data_t(0.5))*th+y*tmp-y;
		//-----------------------------------------------------
		data_t	costh	=	cos(th);
		data_t	sinth	=	sin(th);
		data_t	cos2th	=	costh*costh-sinth*sinth;
		data_t	sin2th	=	costh*sinth*data_t(2);
		data_t	z1z1	=	z1*z1;
			tmp	=	data_t(1)/z1;
		for(int k=0;k<10;++k)
		{
			gr	+=	gammalnc_coef<data_t>[k]*tmp*costh;
			gi	-=	gammalnc_coef<data_t>[k]*tmp*sinth;
			tmp	/=	z1z1;
			data_t	costhtmp=costh*cos2th-sinth*sin2th;
			data_t	sinthtmp=costh*sin2th+sinth*cos2th;
			costh	=	costhtmp;
			sinth	=	sinthtmp;
		} 
		if(x<=data_t(7))
		{
			data_t gr1=data_t(0);
			data_t gi1=data_t(0);
			for(int j=0;j<na;++j)
			{
		    		gr1+=data_t(0.5)*log((x+j)*(x+j)+y*y);
		    		gi1+=atan(y/(x+j));
			}
			gr-=gr1;
			gi-=gi1;
		}
		if(x1<data_t(0))
		{
				z1	=	sqrt(x*x+y*y);
			data_t	th1	=	atan(y/x);
			data_t	sr	=	-sin(PI<data_t>*x)*cosh(PI<data_t>*y);
			data_t  si	=	-cos(PI<data_t>*x)*sinh(PI<data_t>*y);
			data_t	z2	=	sqrt(sr*sr+si*si);
			data_t	th2	=	atan2(si,sr);
			if(sr<data_t(0))
			{
				th2	+=	PI<data_t>;
			}
				gr	=	log(PI<data_t>/(z1*z2))-gr;
				gi	=	-th1-th2-gi;
		}
	}//end of gammalnc

	template<class data_t>
        void    gammac		(const std::complex<data_t>& z,std::complex<data_t>& g)noexcept
	{
		data_t lnr,arg;
		gammalnc(z,&lnr,&arg);
		data_t g0=exp(lnr);
		g={g0*cos(arg),g0*sin(arg)};
	}//end of gammac

}//end of qpc
