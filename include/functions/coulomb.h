#pragma once
#include <libraries/support_std.h>

#include <functions/gamma.h>//LogGamma

#include <algorithm/cf.h>
#include <algorithm/pd.h>
#include <algorithm/rk.h>

//=====================================================================================================
//
//					Coulomb Wave Function
//
// References:
// [1] Abramowitz and Stegun, Handbook of Mathematical Functions (1964)	
// [2] A. R. Barnett, D. H. Feng, J. W. Steed, L. J. B. Goldfarb, Commput. Phys. Commun. 8, (1974) 
// pp.377-395
// [3] A. R. Barnett, J. Comput. Phys. 46 (1982) pp.171-188
// [4] A. R. Barnett, Commput. Phys. Commun. 27 (1982) pp.147-166
//
// Description:
// (1) The algorithm computes arrays of usually normalized regular and irregular Coulomb wave function[1]
// F(L,eta,rho), G(L,eta,rho), F'(L,eta,rho), G'(L,eta,rho), where L, eta and rho are all real numbers,
// majorly based on the algorithm in ref.[2-4].
// 
// List:
//
// (0)ODE equation
// 	coulomb_ode			----  	the ode equation model for Coulomb wave function
// 
// (1)Recurrence relations
//
//	coulomb_recurrence_cf1 		----	recurrence relation for F
//	coulomb_recurrence_cf2  	----  	recurrence relation for H+,H-
//	coulomb_recurrence_cf1_real	----	recurrence relation for F, with pure real parameters
//	coulomb_recurrence_cf2_real  	----  	recurrence relation for H+,H-, with pure real parameters
//
// (2)Coutinued fractions
//
//	coulomb_ratio0		----	F'/F
//	coulomb_ratio1		----	F'/F, Steed's B	
//	coulomb_ratio2		----	H+'/H+
//	coulomb_ratio3		----	H-'/H-
//	coulomb_ratio0_real	----	F'/F for pure real parameters
//	coulomb_ratio1_real	----	F'/F, Steed's B for pure real parameters
//	coulomb_ratio2_real	----	H+'/H+ for pure real parameters
//	coulomb_ratio3_real	----	H-'/H- for pure real parameters
//
// (3)Utilities for general constants
//
// 	coulomb_sig		----	Coulomb Phase Shift, Sigma = 1/2I * [LogGamma(L+1+I*Eta)-LogGamma(L+1-I*Eta)]
// 	coulomb_sig_real	----	Coulomb Phase Shift for pure real parameters, Sigma = ArgLogGamma(L+1+I*Eta)	
// 	coulomb_chi		----	Sigma[L,Eta]-Sigma[-L-1,Eta]-Pi*(L+1/2)
// 	coulomb_chi_real	----	Sigma[L,Eta]-Sigma[-L-1,Eta]-Pi*(L+1/2) for real parameters
// 	coulomb_wronskian	----	examine whether F'*H-F*H' == 1
//
// (4)Utilities for Taylor expansions
//
//	coulomb_taylor_phi	----	Taylor expansion part at 0 for F or
//					Taylor expansion part at 0 for F, F'
//	coulomb_taylor_phi_real	----	Taylor Expansion part at 0 for F, with real parameters or
//					Taylor expansion part at 0 for F, F', with real parameters
//
//	coulomb_padeac_phi	----	Taylor expansion part at 0 for F, with pade acceleration
//	coulomb_padeac_phi_real	----	Taylor expansion part at 0 for F, with pade acceleration
//
//	coulomb_taylor_nrm	----	C=1/2*Exp(-Pi*Eta/2)*Sqrt(Gamma(L+1+iEta)*Gamma(L+1-iEta))/Gamma[2L+2]
//	coulomb_taylor_nrm_real	----	C=1/2*Exp(-Pi*Eta/2)*Abs(Gamma(L+1+iEta))/Gamma[2L+2], for real parameters
//	coulomb_taylor_pow	----	(2X)^(L+1)
//	coulomb_taylor_pow_real	----	(2X)^(L+1), for real parameters
//
//	coulomb_taylor		----	F=C*(2X)^(L+1)*Phi and F'
//	coulomb_taylor_real	----	F=C*(2X)^(L+1)*Phi and F' with real parameters
//
// (5)Utilities for asymptotic expansions
//
//	coulomb_asympt_phi	----	Asymptotic series part at +inf for H+,H-,H+',H-' or
//					Asymptotic series part at +inf for H+,H- only
//	coulomb_asympt_psi	----	Asymptotic series part at +inf for F, designed for calibration of sign
//
//	coulomb_asympt_exp	----	if flag==0, Exp(I*w*(X-Eta*Log(2*X)-L*Pi/2+Sigma))
//					if flag==1, Exp(I*w*(X-Eta*Log(2*X)-L*Pi/2))
//	coulomb_asympt_exp_real	----	if flag==0, Exp(I*w*(X-Eta*Log(2*X)-L*Pi/2+Sigma))
//	   				if flag==1, Exp(I*w*(X-Eta*Log(2*X)-L*Pi/2))
//
// (6)Interfaces for user
//
//	coulombsh1		----	the phase shift in coulombic+short-range potential calculated from known value of u(x), u'(x)
//	coulombsh2		---- 	the phase shift in coulombic+short-range potential calculated from known value of u(xa),u(xb)
//
//	coulombfg		----	the function evaluating the value of F,F',G,G' for real L, real Eta, real X, with usual normalization, or
//					the function evaluating the value of F,F' for real L, real Eta, real X, with usual normalization.
//
//
//=====================================================================================================	

namespace qpc
{

	template<class data_t=double>
	struct	coulomb_ode final //for numerov
	{
		using cord_t	=	data_t;

		data_t	e;	//eta = -mZ/k, m is mass, Z is nuclear charge, k = sqrt(2m*E), E is energy
		data_t	l;	//l can be non-integer

		//to describe y''+gy=0, where g = (1.-2*eta/rho-l*(l+1)/rho^2)	
		inline	void	operator() (const data_t& rho,data_t& g)const noexcept
		{
			g	=	(1.-2.*e/rho-l*(l+1.)/(rho*rho));	
		}
	};//end of coulomb_ode

	template<class data_t=double>
	struct	coulomb_ode2 final //for runge-kutta
	{
		using	cord_t	=	std::array<data_t,2ul>;
	
		data_t	e;
		data_t	l;
	
		inline	void	operator() (const data_t& X, const cord_t& Y, cord_t& dY)const noexcept
		{
			dY[0]	=	Y[1];
			dY[1]	=	-((data_t(1)-data_t(2)*e/X-(l+data_t(1))*l/(X*X)))*Y[0];
		}
	};//end of coulomb_ode2
	//--------------------------------------------------------------------------------------------------------------
	static constexpr size_t	coulomb_ratio0_imax	=	100000ul;
	static constexpr size_t	coulomb_ratio1_imax	=	100000ul;
	static constexpr size_t	coulomb_ratio2_imax	=	100000ul;
	static constexpr size_t	coulomb_ratio3_imax	=	100000ul;

	static constexpr size_t coulomb_taylor_imax	=	2000ul;

	static constexpr size_t coulomb_asympt_imax	=	100ul;	//do not be too large
	static constexpr size_t	coulomb_asympt_ipad	=	coulomb_asympt_imax*3ul;
	static constexpr size_t coulomb_asympt_jpad	=	30ul;	//must < 0.5 imax

	template<class data_t>
	static constexpr data_t	coulomb_ratio0_rtol	=	std::numeric_limits<data_t>::epsilon()*data_t(32);
	template<class data_t>
	static constexpr data_t	coulomb_ratio1_rtol	=	std::numeric_limits<data_t>::epsilon()*data_t(32);
	template<class data_t>
	static constexpr data_t	coulomb_ratio2_rtol	=	std::numeric_limits<data_t>::epsilon()*data_t(32);
	template<class data_t>
	static constexpr data_t	coulomb_ratio3_rtol	=	std::numeric_limits<data_t>::epsilon()*data_t(32);
	template<class data_t>
	static constexpr data_t coulomb_taylor_rtol2	=	std::numeric_limits<data_t>::epsilon()*data_t(64)
							*	std::numeric_limits<data_t>::epsilon();
	template<class data_t>
	static constexpr data_t coulomb_asympt_rtol2	=	std::numeric_limits<data_t>::epsilon()*data_t(64)
							*	std::numeric_limits<data_t>::epsilon();
							

	template<class data_t>
	static constexpr data_t	coulomb_wronskian_tol	=	data_t(1e-12);
	//--------------------------------------------------------------------------------------------------------------
	template<class data_t>
	struct	coulomb_recurrence_cf1
	{
		std::complex<data_t>	nu;//v+n
		std::complex<data_t>	et;//eta
		std::complex<data_t> 	ir;//1/rho

		//see, for example, in I. J. Thompson 1986
		inline	void	operator()	(size_t i,std::complex<data_t>& a,std::complex<data_t>& b)const noexcept
		{
			std::complex<data_t> l	=	data_t(i)+nu;
			std::complex<data_t> s	=	et/l;
			a= -(data_t(1)+s*s);
			b=(data_t(2)*l+data_t(1))*(ir+s/(l+data_t(1)));
		}
		inline	auto	coulomb_s	(size_t i)const noexcept
		{
			std::complex<data_t> l  =       data_t(i)+nu;
			return 	l*ir+et/l;
		}
		inline	auto	coulomb_t	(size_t i)const noexcept
		{
			std::complex<data_t> l  =       data_t(i)+nu;
			return 	(data_t(2)*l+data_t(1))*(ir+et/(l*(l+data_t(1))));
		}
		inline	auto	coulomb_r	(size_t i)const noexcept
		{
			std::complex<data_t> l  =       data_t(i)+nu;
			return 	data_t(1)+et*et/(l*l);
		}
	};//end of coulomb_recurrence_cf1

	template<class data_t,int omg>
	struct	coulomb_recurrence_cf2
	{
		std::complex<data_t>	nu;//v+n
		std::complex<data_t>	et;//eta
		std::complex<data_t>	rh;//rho

		static constexpr std::complex<data_t>	o	= omg>0?im<data_t>:jm<data_t>;

		//see, for example, in I. J. Thompson 1986
		inline	void	operator()	(size_t i,std::complex<data_t>& a,std::complex<data_t>& b)const noexcept
		{
			a=(o*et-nu+data_t(i-1))*(o*et+nu+data_t(i));
			b=data_t(2)*(rh-et+data_t(i)*o);
		}

	};//end of coulomb_recurrence_cf2

	template<class data_t>
	struct	coulomb_recurrence_cf1_real
	{
		data_t	nu;//v+n
		data_t	et;//eta
		data_t	ir;//1/rho

		//see, for example, in I. J. Thompson 1986
		inline	void	operator()	(size_t i,data_t& a,data_t& b)const noexcept
		{
			data_t l	=	data_t(i)+nu;
			data_t s	=	et/l;
			a= -(data_t(1)+s*s);
			b=(data_t(2)*l+data_t(1))*(ir+s/(l+data_t(1)));
		}
		inline	auto	coulomb_s	(size_t i)const noexcept
		{
			data_t 	l  	=       data_t(i)+nu;
			return 	l*ir+et/l;
		}
		inline	auto	coulomb_t	(size_t i)const noexcept
		{
			data_t 	l  	=       data_t(i)+nu;
			return 	(data_t(2)*l+data_t(1))*(ir+et/(l*(l+data_t(1))));
		}
		inline	auto	coulomb_r	(size_t i)const noexcept
		{
			data_t 	l  	=       data_t(i)+nu;
			return 	data_t(1)+et*et/(l*l);
		}
	};//end of coulomb_recurrence_cf1_real

	template<class data_t,int omg>
	struct	coulomb_recurrence_cf2_real
	{
		data_t	nu;//v+n
		data_t	et;//eta
		data_t	rh;//rho

		static constexpr std::complex<data_t>	o	= omg>0?im<data_t>:jm<data_t>;

		//see, for example, in I. J. Thompson 1986
		inline	void	operator()	(size_t i,std::complex<data_t>& a,std::complex<data_t>& b)const noexcept
		{
			a=(o*et-nu+data_t(i-1))*(o*et+nu+data_t(i));
			b=data_t(2)*(rh-et+data_t(i)*o);
		}
	};//end of coulomb_recurrence_cf2_real
	//--------------------------------------------------------------------------------------------------------------

	template<class data_t>
	int	coulomb_ratio0
	(
		const std::complex<data_t>& v,	//v+n
		const std::complex<data_t>& e,	//eta
		const std::complex<data_t>& x,	//rho
		std::complex<data_t>& 	f	//returned as F'/F
	)noexcept
	{
		std::complex<data_t>	cf;
		auto	recur	=	coulomb_recurrence_cf1<data_t>{v,e,data_t(1)/x};
		auto 	info	=	cf_lentz_comp(recur,coulomb_ratio0_rtol<data_t>,coulomb_ratio0_imax,cf);
			f	= 	cf-recur.coulomb_s(0);
		return 	info>0?1:0;//error code 0,1
	}//end of coulomb_ratio0

	template<class data_t>
	int	coulomb_ratio0_real
	(
		const data_t& v,	//v+n
		const data_t& e,	//eta
		const data_t& x,	//rho
		data_t& f		//returned as F'/F
	)noexcept
	{
		data_t	cf;
		auto	recur	=	coulomb_recurrence_cf1_real<data_t>{v,e,data_t(1)/x};
		auto 	info	=	cf_lentz_real(recur,coulomb_ratio0_rtol<data_t>,coulomb_ratio0_imax,cf);
			f	= 	cf-recur.coulomb_s(0);
		return 	info>0?2:0;//error code 0,2
	}//end of coulomb_ratio0_real
	
	template<class data_t>
	int	coulomb_ratio1	
	(
		const std::complex<data_t>& v,	//v+n
		const std::complex<data_t>& e,	//eta
		const std::complex<data_t>& x,	//rho
		std::complex<data_t>& 	f,	//returned as F'/F
		std::complex<data_t>&	s, 	//returned as Steed's B
		size_t& i			//returned as the termination index I
	)noexcept
	{
		std::complex<data_t>	cf;
		auto	recur	=	coulomb_recurrence_cf1<data_t>{v,e,data_t(1)/x};
		auto 	info	=	cf_steed_comp(recur,coulomb_ratio1_rtol<data_t>,coulomb_ratio1_imax,cf,s,i);
			f	= 	cf-recur.coulomb_s(0);
		return 	info>0?3:0;//error code 0,3
	}//end of coulomb_ratio1

	template<class data_t>
	int	coulomb_ratio1_real
	(
		const data_t& v,	//v+n
		const data_t& e,	//eta
		const data_t& x,	//rho
		data_t& f,	//returned as F'/F
		data_t&	s 	//returned as Steed's B
	)noexcept
	{
		data_t	cf;
		auto	recur	=	coulomb_recurrence_cf1_real<data_t>{v,e,data_t(1)/x};
		auto 	info	=	cf_steed_real(recur,coulomb_ratio1_rtol<data_t>,coulomb_ratio1_imax,cf,s);
			f	= 	cf-recur.coulomb_s(0);
		return 	info>0?4:0;//error code 0,4
	}//end of coulomb_ratio1_real

	template<class data_t>
	int	coulomb_ratio2	
	(
		const std::complex<data_t>& v,	//v+n
		const std::complex<data_t>& e,	//eta
		const std::complex<data_t>& x,	//rho
		std::complex<data_t>&	h	//returned as H+'/H+
	)noexcept
	{
		std::complex<data_t>	cf;
		auto	recur	=	coulomb_recurrence_cf2<data_t,+1>{v,e,x};//with omg=+1
		auto	info	=	cf_lentz_comp(recur,coulomb_ratio2_rtol<data_t>,coulomb_ratio2_imax,cf);
			h	= 	recur.o/recur.rh*(cf-x+e);
		return 	info>0?5:0;//error code 0,5
	}//end of coulomb_ratio2

	template<class data_t>
	int	coulomb_ratio2_real
	(
		const data_t& v,	//v+n
		const data_t& e,	//eta
		const data_t& x,	//rho
		std::complex<data_t>& h	//returned as H+'/H+
	)noexcept
	{
		std::complex<data_t>	cf;
		auto	recur	=	coulomb_recurrence_cf2_real<data_t,+1>{v,e,x};//with omg=+1
		auto	info	=	cf_lentz_comp(recur,coulomb_ratio2_rtol<data_t>,coulomb_ratio2_imax,cf);
			h	= 	recur.o/recur.rh*(cf-x+e);
		return 	info>0?6:0;//error code 0,6
	}//end of coulomb_ratio2_real
	
	template<class data_t>
        int 	coulomb_ratio3	
	(
		const std::complex<data_t>& v,	//v+n
		const std::complex<data_t>& e,	//eta
		const std::complex<data_t>& x,	//rho
		std::complex<data_t>& 	h	//returned as H-'/H-
	)noexcept
        {
                std::complex<data_t>    cf;
                auto    recur   =       coulomb_recurrence_cf2<data_t,-1>{v,e,x};//with omg=-1
		auto	info	=	cf_lentz_comp(recur,coulomb_ratio3_rtol<data_t>,coulomb_ratio3_imax,cf);
                	h	=	recur.o/recur.rh*(cf-x+e);
		return 	info>0?7:0;//error code 0,7
        }//end of coulomb_ratio3

	template<class data_t>
        int 	coulomb_ratio3_real
	(
		const data_t& v,	//v+n
		const data_t& e,	//eta
		const data_t& x,	//rho
		std::complex<data_t>& h	//returned as H-'/H-
	)noexcept
        {
                std::complex<data_t>    cf;
                auto    recur   =       coulomb_recurrence_cf2_real<data_t,-1>{v,e,x};//with omg=-1
		auto	info	=	cf_lentz_comp(recur,coulomb_ratio3_rtol<data_t>,coulomb_ratio3_imax,cf);
                	h	=	recur.o/recur.rh*(cf-x+e);
		return 	info>0?8:0;//error code 0,8
        }//end of coulomb_ratio3_real
	//-----------------------------------------------------------------------------------------------------------
	template<class data_t>
	auto	coulomb_sig
	(
		const std::complex<data_t>& v,
		const std::complex<data_t>& e
	)noexcept
	{
		data_t	zr,zi,r1,r2,a1,a2;
		//1st, gammaln(v+1+ie)
		zr	=	real(v)-imag(e)+data_t(1.);
		zi	=	imag(v)+real(e);
		gammalnc<data_t>({zr,zi},&r1,&a1);
		//2nd, gammaln(v+1-ie)
		zr	=	real(v)+imag(e)+data_t(1.);
		zi	=	imag(v)-real(e);
		gammalnc<data_t>({zr,zi},&r2,&a2);

		return 	std::complex<data_t>{(a1-a2)/data_t(2),(r2-r1)/data_t(2)};
	}//end of coulomb_sig

	template<class data_t>
	auto	coulomb_sig_real
	(
		const data_t& v,
		const data_t& e
	)noexcept
	{
		data_t	zr,zi,r1,a1;
		//1st, gammaln(v+1+ie)
		zr	=	real(v)-imag(e)+data_t(1.);
		zi	=	imag(v)+real(e);
		gammalnc<data_t>({zr,zi},&r1,&a1);
		return	a1;
	}//end of coulomb_sig_real

	template<class data_t>
	int	coulomb_chi
	(
		const std::complex<data_t>& v,
		const std::complex<data_t>& e,
		std::complex<data_t>& chi
	)noexcept
	{
		chi= 	coulomb_sig(v,e)
		-	coulomb_sig(-v-data_t(1),e)
		-	(v+data_t(0.5))*PI<data_t>;	
		return	0;//XXX any check?
	}//end of coulomb_chi
	
	template<class data_t>
	int	coulomb_chi_real
	(
		const data_t& v,
		const data_t& e,
		data_t& chi
	)noexcept
	{
		chi=  	coulomb_sig_real(v,e)
		-	coulomb_sig_real(-v-data_t(1),e)
		-	(v+data_t(0.5))*PI<data_t>;
		return 	0;//XXX any check?
	}//end of coulomb_chi_real

	template<class data_t>
	int	coulomb_wronskian
	(
		const data_t& f0,
		const data_t& f1,
		const data_t& g0,
		const data_t& g1
	)noexcept
	{
		if(fabs(f1*g0-f0*g1-data_t(1))<coulomb_wronskian<data_t>)
		{
			return 	0;
		}else
		{
			return 	1;
		}
	}//end of coulomb_wronskian
	//-----------------------------------------------------------------------------------------------------------

	template<class data_t,int flag=0>
	auto	coulomb_asympt_exp
	(
		const std::complex<data_t>& v,
		const std::complex<data_t>& e,
		const std::complex<data_t>& x
	)noexcept
	{
		if constexpr(flag==0)//with Sigma
		{
			auto	arg	=	x-e*log(data_t(2)*x)-(PI<data_t>/data_t(2))*v
					+	coulomb_sig(v,e);
			return 	exp(im<data_t>*arg);
		}
		if constexpr(flag==1)//without Sigma
		{
			auto    arg     =       x-e*log(data_t(2)*x)-(PI<data_t>/data_t(2))*v;
			return	exp(im<data_t>*arg);
		}
	}//end of coulomb_asympt_exp

	template<class data_t,int flag=0>
	auto	coulomb_asympt_exp_real
	(
		const data_t& v,
		const data_t& e,
		const data_t& x
	)noexcept
	{
		if constexpr(flag==0)//with Sigma
		{
			auto	arg	=	x-e*log(data_t(2)*x)-PI<data_t>*v/data_t(2)
					+	coulomb_sig_real(v,e);
			return 	exp(im<data_t>*arg);
		}
		if constexpr(flag==1)//without Sigma
		{
			auto    arg     =       x-e*log(data_t(2)*x)-PI<data_t>*v/data_t(2);
			return	exp(im<data_t>*arg);
		}
	}//end of coulomb_asympt_exp_real

	template<class data_t,class...work_t>
	int	coulomb_asympt_phi	
	(
		const std::complex<data_t>& v,
		const std::complex<data_t>& e,
		const std::complex<data_t>& x,
		std::complex<data_t>& h,//H+ or H-
		int w,//+1 or -1 to choose H+ or H-
		work_t...work//be of length coulomb_asympt_ipad
	)noexcept
	{
		//see Abramowitz& Stegun 1966, Chap14.5, Michel 2007
		using	comp_t	=	std::complex<data_t>;
			h	=	data_t(1);
		comp_t	b	=	data_t(1);	//h= sum b(i)
		comp_t	iw	=	data_t(w)*im<data_t>;	 	//to calculate b(i+1)=ratio(i)*b(i)
		comp_t	c0	=	e*(iw-e)-v*(v+data_t(1));	//to calculate b(i+1)=ratio(i)*b(i)
		comp_t	c1	=	data_t(1)+iw*data_t(2)*e;	//to calculate b(i+1)=ratio(i)*b(i)
		comp_t	c2	=	data_t(2)*iw*x;			//to calculate b(i+1)=ratio(i)*b(i)
		data_t	n; 
		size_t	i;
		for(i=0;i<coulomb_asympt_imax;++i)
		{
			n	=	data_t(i);
			b	*=	(n*(n+c1)+c0)/(c2*(n+data_t(1)));//ratio(i)
			h	+=	b;
			if(norm(b/h)<coulomb_asympt_rtol2<data_t>)break;
		}
		if(i<coulomb_asympt_imax)	
		{
			return 	0;
		}else if constexpr(sizeof...(work)==1)
		{
			b	=	data_t(1);
			auto f	=	[&](const size_t ii)
			{
				n       =       data_t(ii);
				return 	b*=(n*(n+c1)+c0)/(c2*(n+data_t(1)));//ratio(i)
			};
			pade_approximant<comp_t>(work...,h,f,coulomb_asympt_imax,coulomb_asympt_jpad);
			h	+=	data_t(1);
			return	0;
		}else
		{
			return 	2000;//does not converge
		}
	}//end of coulomb_asympt_phi

	template<class data_t,class...work_t>
	int	coulomb_asympt_phi	
	(
		const std::complex<data_t>& v,
		const std::complex<data_t>& e,
		const std::complex<data_t>& x,
		std::complex<data_t>& h0,//H+ or H-
		std::complex<data_t>& h1,//H+'or H-'
		int w,//+1 or -1
		work_t...work//of length coulomb_asympt_ipad
	)noexcept
	{
		//see Abramowitz& Stegun 1966, Chap14.5, and Michel 2007
		using	comp_t	=	std::complex<data_t>;
		comp_t	iw	=	data_t(w)*im<data_t>;
		comp_t	b0	=	data_t(1);
		comp_t	b1	=	iw*(data_t(1)-e/x);
			h0	=	b0;
			h1	=	b1;
		comp_t	c0	=	e*(iw-e)-v*(v+data_t(1));	//to calculate b(i+1)=ratio(i)*b(i)
		comp_t	c1	=	data_t(1)+iw*data_t(2)*e;	//to calculate b(i+1)=ratio(i)*b(i)
		comp_t	c2	=	data_t(2)*iw*x;			//to calculate b(i+1)=ratio(i)*b(i)
		data_t	n; 
		comp_t	r;
		size_t	i;
		for(i=0;i<coulomb_asympt_imax;++i)
		{
			n	=	data_t(i);
			r	=	(n*(n+c1)+c0)/(c2*(n+data_t(1)));//ratio(i)
			b0	*=	r;
			b1	=	b1*r-b0/x;
			h0	+=	b0;
			h1	+=	b1;
			if(norm(b0/h0)<coulomb_asympt_rtol2<data_t>)break;
		}
		if(i<coulomb_asympt_imax)	
		{
			return 	0;
		}else if constexpr(sizeof...(work)==1)
		{
			b0	=	data_t(1);
			auto f0	=       [&](const size_t ii)
                        {
                                n       =       data_t(ii);
                                b0	*=	(n*(n+c1)+c0)/(c2*(n+data_t(1)));//ratio(i)
				return	b0;
                        };
                        pade_approximant<comp_t>(work...,h0,f0,coulomb_asympt_imax,coulomb_asympt_jpad);
                        h0      +=      data_t(1);
		
			b0	=	data_t(1);
			b1	=	iw*(data_t(1)-e/x);
			auto f1	=	[&](const size_t ii)
			{
				n	=	data_t(ii);
				r       =       (n*(n+c1)+c0)/(c2*(n+data_t(1)));//ratio(i)
				b1	-=	b0/x;
				b0	*=	r;
				b1	*=	r;
				return	b1;
			};
                        pade_approximant<comp_t>(work...,h1,f1,coulomb_asympt_imax,coulomb_asympt_jpad);
			h1	+=	iw*(data_t(1)-e/x);
			return	0;	
		}else
		{
			return 	2000;//does not converge
		}
	}//end of coulomb_asympt_phi

	template<class data_t>
	int	coulomb_asympt_psi
	(
		const std::complex<data_t>& v,
		const std::complex<data_t>& e,
		const std::complex<data_t>& x,
		std::complex<data_t>& h0,
		int w
	)noexcept
	{
		//see Abramowitz& Stegun 1966, Chap14.5, and Michel 2007
		using	comp_t	=	std::complex<data_t>;
		comp_t	iw	=	data_t(w)*im<data_t>;
		comp_t	b0	=	data_t(1);
			h0	=	b0;
		comp_t	c0	=	e*(iw-e)-v*(v+data_t(1));	//to calculate b(i+1)=ratio(i)*b(i)
		comp_t	c1	=	data_t(1)+iw*data_t(2)*e;	//to calculate b(i+1)=ratio(i)*b(i)
		comp_t	c2	=	data_t(2)*iw*x;			//to calculate b(i+1)=ratio(i)*b(i)
		data_t	n; 
		comp_t	r;
		size_t	i;
		for(i=0;i<10;++i)
		{
			n	=	data_t(i);
			r	=	(n*(n+c1)+c0)/(c2*(n+data_t(1)));	//ratio(i)
			h0	+=	(b0*=r);
			if(norm(b0/h0)<data_t(0.1))break;
		}
		return 	i<10?0:90000;
	}//end of coulomb_asympt_psi
	//--------------------------------------------------------------------------------------------------------------
	//usual F normalization constant C= 1/2 Exp(-Pi*Eta/2)*Sqrt(Gamma(L+1+iEta)*Gamma(L+1-iEta))/Gamma[2L+2], such that F = C * Phi * (2X)^(L+1)
	template<class data_t>
	int	coulomb_taylor_nrm	
	(
		const std::complex<data_t>& v,
		const std::complex<data_t>& e,
		std::complex<data_t>& nrm
	)noexcept
	{
		data_t	r1,r2,r3,a1,a2,a3,amp,ang;
		//1st, gammaln(v+1+ie)
		gammalnc<data_t>(v+data_t(1)+im<data_t>*e,&r1,&a1);
		//2nd, gammaln(v+1-ie)
		gammalnc<data_t>(v+data_t(1)-im<data_t>*e,&r2,&a2);
		//3rd, gammaln(2*v+2)
		gammalnc<data_t>(v*data_t(2)+data_t(2),&r3,&a3);
		//result
		amp	=	(r1+r2)/data_t(2)-r3;
		ang	=	(a1+a2)/data_t(2)-a3;
		nrm	=	exp(amp+im<data_t>*ang-e*(PI<data_t>/data_t(2)))/data_t(2);
		//error info
		return	0;//any check?
	}//end of coulomb_taylor_nrm

	//usual F normalization constant C= 1/2 Exp(-Pi*Eta/2)*Abs(Gamma[L+1+iEta])/Gamma[2L+2], such that F = C * Phi * (2X)^(L+1)
	template<class data_t>
	int	coulomb_taylor_nrm_real
	(
		const data_t& v,
		const data_t& e,
		data_t&	nrm
	)noexcept
	{
		data_t	r1,a1,r2,a2, zr,zi;
		//1st gammaln(l+1+ieta)
		zr = 1. + v;
		zi = e;
		gammalnc<data_t>({zr,zi},&r1,&a1);
		//2nd gammaln(2l+2)
		zr = 2. * v + 2.;
		zi = 0.;
		gammalnc<data_t>({zr,zi},&r2,&a2);
		nrm	=	exp(-PI<data_t>*e/data_t(2)+r1-r2)/data_t(2)
			*	(cos(a2)>data_t(0.)?data_t(+1.):data_t(-1.));
		return	0;
	}//end of coulomb_taylor_nrm_real

	template<class data_t>
	int	coulomb_taylor_phi	
	(
		const std::complex<data_t>& v,
		const std::complex<data_t>& e,
		const std::complex<data_t>& x,
		std::complex<data_t>& phi
	)noexcept
	{
		//see Abramowitz& Stegun 1966, Chap14.1
		using	comp_t	=	std::complex<data_t>;
		comp_t	x2	=	x*x;
		comp_t	ex	=	data_t(2)*e*x;

		comp_t	a1	=	data_t(1);
    		comp_t	a2	=	e/(v+data_t(1))*x;
			phi	=	a1+a2;

		size_t	i;
		comp_t 	k,a3;

		for(i=3;i<=coulomb_taylor_imax;++i)
		{
			k	=	data_t(i)+v;
			a3	=	(ex*a2-x2*a1)/((k+v)*data_t(i-1));//the denom is inf iff v<=-1 and 2v is negative integer
			phi	+=	a3;
			if(norm(a3/phi)<coulomb_taylor_rtol2<data_t>)break;
			a1	=	a2;
			a2	=	a3;
		}
		if(i>=coulomb_taylor_imax)	return 	300;//does not converge
		else				return 	0;
	}//end of coulomb_taylor_phi

	template<class data_t>
	int	coulomb_taylor_phi	
	(
		const std::complex<data_t>& v,
		const std::complex<data_t>& e,
		const std::complex<data_t>& x,
		std::complex<data_t>& phi,
		std::complex<data_t>& phid
	)noexcept
	{
		//see Abramowitz& Stegun 1966, Chap14.1
		using	comp_t	=	std::complex<data_t>;
		comp_t	x2	=	x*x;
		comp_t	ex	=	data_t(2)*e*x;

		comp_t	a1	=	data_t(1);
    		comp_t	a2	=	e/(v+data_t(1))*x;
			phi	=	a1+a2;
			phid	=	a1*(v+data_t(1))+a2*(v+data_t(2));

		size_t	i;
		comp_t 	k,a3;

		for(i=3;i<=coulomb_taylor_imax;++i)
		{
			k	=	data_t(i)+v;
			a3	=	(ex*a2-x2*a1)/((k+v)*data_t(i-1));//the denom is inf iff v<=-1 and 2v is negative integer
			phi	+=	a3;
			phid	+=	a3*(v+data_t(i));
			if(norm(a3/phi)<coulomb_taylor_rtol2<data_t>)break;
			a1	=	a2;
			a2	=	a3;
		}
		phid/=x;
		if(i>=coulomb_taylor_imax)	return 	400;//does not converge
		else				return 	0;
	}//end of coulomb_taylor_phi
	
	template<class data_t>
	int	coulomb_taylor_phi_real
	(
		const data_t& v,
		const data_t& e,
		const data_t& x,
		data_t&	phi
	)noexcept
	{
		//see Abramowitz& Stegun 1966, Chap14.1
		data_t	x2	=	x*x;
		data_t	ex	=	data_t(2)*e*x;

		data_t	a1	=	data_t(1);
    		data_t	a2	=	e/(v+data_t(1))*x;
			phi	=	a1+a2;

		size_t	i;
		data_t 	k,a3;

		for(i=3;i<=coulomb_taylor_imax;++i)
		{
			k	=	data_t(i)+v;
			a3	=	(ex*a2-x2*a1)/((k+v)*data_t(i-1));//the denom is inf iff v<=-1 and 2v is negative integer
			phi	+=	a3;
			if
			(norm(a3/phi)<coulomb_taylor_rtol2<data_t>&&
			 norm(a2/phi)<coulomb_taylor_rtol2<data_t>)break;//to avoid fake convergency (important when e=0)
			a1	=	a2;
			a2	=	a3;
		}
		if(i>=coulomb_taylor_imax)	return 	500;//does not converge
		else				return 	0;
	}//end of coulomb_taylor_phi_real

	template<class data_t>
	int	coulomb_taylor_phi_real
	(
		const data_t& v,
		const data_t& e,
		const data_t& x,
		data_t&	phi,
		data_t& phid	
	)noexcept
	{
		//see Abramowitz& Stegun 1966, Chap14.1
		data_t	x2	=	x*x;
		data_t	ex	=	data_t(2)*e*x;

		data_t	a1	=	data_t(1);
    		data_t	a2	=	e/(v+data_t(1))*x;
			phi	=	a1+a2;
			phid	=	a1*(v+data_t(1))+a2*(v+data_t(2));
		size_t	i;
		data_t 	k,a3;

		for(i=3;i<=coulomb_taylor_imax;++i)
		{
			k	=	data_t(i)+v;
			a3	=	(ex*a2-x2*a1)/((k+v)*data_t(i-1));//the denom is inf iff v<=-1 and 2v is negative integer
			phi	+=	a3;
			phid	+=	a3*(v+data_t(i));
			if
			(norm(a3/phi)<coulomb_taylor_rtol2<data_t>&&
			 norm(a2/phi)<coulomb_taylor_rtol2<data_t>)break;//to avoid fake convergency (important when e=0)
			a1	=	a2;
			a2	=	a3;
		}
		phid/=x;
		if(i>=coulomb_taylor_imax)	return 	600;//does not converge
		else				return 	0;
	}//end of coulomb_taylor_phi_real

	template<class data_t>
	int	coulomb_padeac_phi_real//not significant
	(
		const data_t& v,
		const data_t& e,
		const data_t& x,
		data_t&	phi
	)noexcept
	{
		//see Abramowitz& Stegun 1966, Chap14.1
		auto	work	=	std::vector<data_t>(coulomb_taylor_imax*3ul);
		data_t	x2	=	x*x;
		data_t	ex	=	data_t(2)*e*x;

		data_t	a1	=	data_t(1);
    		data_t	a2	=	e/(v+data_t(1))*x;
			phi	=	a1+a2;
		data_t 	a3;
		auto	func	=	[&](const size_t i)
		{
			data_t k=	data_t(i+3)+v;
			a3	=	(ex*a2-x2*a1)/((k+v)*data_t(i+2));//the denom is inf iff v<=-1 and 2v is negative integer
			a1	=	a2;
			a2	=	a3;
			return	a3;
		};
		pade_approximant<data_t>(work.data(),work[0],func,coulomb_taylor_imax,100ul);//empirical
		phi+=work[0];
		return 	0;
	}//end of coulomb_padeac_phi_real

	template<class data_t>
	int	coulomb_taylor_pow	
	(
		const std::complex<data_t>& v,
		const std::complex<data_t>& x,
		std::complex<data_t>& p
	)noexcept
	{
		p =	std::pow(data_t(2)*x,v+data_t(1));
		return	std::isinf(real(p))||std::isinf(imag(p))?700:0;//overflow or 0^0 error
	}//end of coulomb_taylor_pow
	template<class data_t>
	int	coulomb_taylor_pow_real
	(
		const data_t& v,
		const data_t& x,
		data_t&	p
	)noexcept
	{
		p = 	std::pow(data_t(2)*x,v+data_t(1));
		return 	std::isinf(p)?800:0;//overflow or 0^0 error
	}//end of coulomb_taylor_pow_real
	
	template<class data_t>
	int	coulomb_taylor
	(
		const std::complex<data_t>& v,
		const std::complex<data_t>& e,
		const std::complex<data_t>& x,
		std::complex<data_t>& f0	//returned as F
	)noexcept
	{
		std::complex<data_t> _nrm,_phi0,_pow;
		int info=	coulomb_taylor_nrm(v,e,_nrm)
			+	coulomb_taylor_phi(v,e,x,_phi0)
			+	coulomb_taylor_pow(v,x,_pow);
		f0	=	_nrm*_phi0*_pow;
		return	info;
	}//end of coulomb_taylor (complex f0 only)

	template<class data_t>
	int	coulomb_taylor
	(
		const std::complex<data_t>& v,
		const std::complex<data_t>& e,
		const std::complex<data_t>& x,
		std::complex<data_t>& f0,	//returned as F
		std::complex<data_t>& f1	//returned as F'
	)noexcept
	{
		std::complex<data_t> _nrm,_phi0,_phi1,_pow;
		int info=	coulomb_taylor_nrm(v,e,_nrm)
			+	coulomb_taylor_phi(v,e,x,_phi0,_phi1)
			+	coulomb_taylor_pow(v,x,_pow);
		f0	=	_nrm*_phi0*_pow;
		f1	=	_nrm*_phi1*_pow;
		return	info;
	}//end of coulomb_taylor (complex f0,f1)

	template<class data_t>
	int	coulomb_taylor_real
	(
		const data_t& v,
		const data_t& e,
		const data_t& x,
		data_t& f0	//returned as F
	)noexcept
	{
		data_t _nrm,_phi0,_pow;
		int info=	coulomb_taylor_nrm_real(v,e,_nrm)
			+	coulomb_taylor_phi_real(v,e,x,_phi0)
			+	coulomb_taylor_pow_real(v,x,_pow);
		f0	=	_nrm*_phi0*_pow;
		return	info;
	}//end of coulomb_taylor_real (real f0 only)
	
	
	template<class data_t>
	int	coulomb_taylor_real
	(
		const data_t& v,
		const data_t& e,
		const data_t& x,
		data_t& f0,	//returned as F
		data_t& f1	//returned as F'
	)noexcept
	{
		data_t _nrm,_phi0,_phi1,_pow;
		int info=	coulomb_taylor_nrm_real(v,e,_nrm)
			+	coulomb_taylor_phi_real(v,e,x,_phi0,_phi1)
			+	coulomb_taylor_pow_real(v,x,_pow);
		f0	=	_nrm*_phi0*_pow;
		f1	=	_nrm*_phi1*_pow;
		return	info;
	}//end of coulomb_taylor_real (real f0,f1)	
	
	template<class data_t>
	int	coulomb_padeac_real
	(
		const data_t& v,
		const data_t& e,
		const data_t& x,
		data_t& f0	//returned as F
	)noexcept
	{
		data_t _nrm,_phi0,_pow;
		int info=	coulomb_taylor_nrm_real(v,e,_nrm)
			+	coulomb_padeac_phi_real(v,e,x,_phi0)
			+	coulomb_taylor_pow_real(v,x,_pow);
		f0	=	_nrm*_phi0*_pow;
		return	info;
	}//end of coulomb_padeac_real (real f0 only)
		
	//-----------------------------------------------------------------------------------------------------------
	//			functions for phase shift in coulombic+short-range potential
	//-----------------------------------------------------------------------------------------------------------
	static	inline	int	coulombsh1
	(
		const double& v,
		const double& e,
		const double& x,	//must be in asymptotic region
		const double& u0,	//the numerical value of regular solution u  at x, should be solved from ode solver
		const double& u1,	//the numerical value of regular solution u' at x, should be solved from ode solver
		double* sh,		//the 'coulombic' phase shift Delta = delta+ Sigma
		double*	nm		//the absolute normalization such that nm*u0: = cos(delta)* F + sin(delta)* G
	)noexcept
	{
		using comp_t	=	std::complex<double>;
		double	ratio;
		comp_t	h0,h1,ex;
		int 	info	=	coulomb_asympt_phi<double>(comp_t(v),comp_t(e),comp_t(x),h0,h1,+1);//we suppose pade acceleration is not needed
			ex	=	coulomb_asympt_exp_real<double,1>(v,e,x);//1 means the result does not contain Sigma
			h0	*=	ex;
			h1	*=	ex;
		if(u0*u0>0.)
		{
			ratio	=	u1/u0;
			*sh	=	atan(-(ratio*imag(h0)-imag(h1))/(ratio*real(h0)-real(h1)));
			*nm	=	imag(exp(im<double>**sh)*h0)/u0;
		}else
		{
			ratio	=	u0/u1;
			*sh	=	atan(-(ratio*imag(h1)-imag(h0))/(ratio*real(h1)-real(h0)));
			*nm	=	imag(exp(im<double>**sh)*h1)/u1;
		}
		return 	info;		
	}//end of coulombsh1

	static	inline	int	coulombsh2
	(
		const double& v,
		const double& e,
		const double& xa,	//must be in asymptotic region
		const double& xb,	//must be in asymptotic region
		const double& ua,	//the numerical value of regular solution u at xa, should be solved from ode solver
		const double& ub,	//the numerical value of regular solution u at xb, should be solved from ode solver
		double* sh,		//the 'coulombic' phase shift Delta = delta + Sigma
		double*	nm		//the absolute normalization such that nm*u0: = cos(delta)* F + sin(delta)* G
	)noexcept
	{
		using comp_t	=	std::complex<double>;
		int	info	=	0;
		double	ratio;
		comp_t	ha,hb,exa,exb;
			info	+=	coulomb_asympt_phi<double>(comp_t(v),comp_t(e),comp_t(xa),ha,+1);//we suppose pade acceleration is not needed
			info	+=	coulomb_asympt_phi<double>(comp_t(v),comp_t(e),comp_t(xb),hb,+1);
			exa	=	coulomb_asympt_exp_real<double,1>(v,e,xa);//1 means the result does not contain Sigma
			exb	=	coulomb_asympt_exp_real<double,1>(v,e,xb);//1 means the result does not contain Sigma
			ha	*=	exa;
			hb	*=	exb;
		if(ua*ua>0.)
		{
			ratio	=	ub/ua;
			*sh	=	atan(-(ratio*imag(ha)-imag(hb))/(ratio*real(ha)-real(hb)));
			*nm	=	imag(exp(im<double>**sh)*ha)/ua;
		}else
		{
			ratio	=	ua/ub;
			*sh	=	atan(-(ratio*imag(hb)-imag(ha))/(ratio*real(hb)-real(ha)));
			*nm	=	imag(exp(im<double>**sh)*hb)/ub;
		}
		return 	info;		
	}//end of coulombsh2

	//-----------------------------------------------------------------------------------------------------------
	//			      functions for F&G with real parameters
	//-----------------------------------------------------------------------------------------------------------
	template<class data_t>//float,double, or long double
	int	coulombfg //both F and G
	(
		const data_t& v,//v should not be a negative integer
		const data_t& e,
		const data_t& x,//x should not be a negative number
		data_t* f0,//valid index 0:nfwd, filled with       F((0:nfwd)+v,e,x)
		data_t* f1,//valid index 0:nfwd, filled with (d/dx)F((0:nfwd)+v,e,x)
		data_t* g0,//valid index 0:nfwd, filled with       G((0:nfwd)+v,e,x)
		data_t* g1,//valid index 0:nfwd, filled with (d/dx)G((0:nfwd)+v,e,x)
		size_t nfwd	//can be 0
	)noexcept
	{
		using	comp_t	=	std::complex<data_t>;
		if((std::floor(v)==v)&&(v<0))return -1;	//when v is negative integer
		if(x<0)return -2;			//when x is negative number
		bool	isvint2	=	std::floor(v*2.)==v*2.;
		data_t 	xc	=	e+sqrt(e*e+v*v+v);//exp-osc turning point
		data_t 	vn	=	v+data_t(nfwd);
		bool	in_osc	=	x>data_t(0.9)*xc
				&&	x>data_t(1e-3);
		bool	in_asy	=	x>data_t(1000)
				&&	x>data_t(3.0)*(fabs(v+nfwd+1)+e*e);
		int	info	=	0;
	
		//determine unnormalized Fn,Fn'
		if(!in_asy)//usually, by Fn'/Fn and sign(Fn)
		{
			data_t	fm,sb;
			if(vn!=0)//there is divided-by-0 error if vn=0
			{
				info	+=	coulomb_ratio1_real<data_t>(vn,e,x,fm,sb);//possible code: 0 or 4
				f0[nfwd]=	(sb>0?data_t(1e-290):data_t(-1e-290));
				f1[nfwd]=	(sb>0?data_t(1e-290):data_t(-1e-290))*fm;
			}else
			{
				data_t ir=	data_t(1)/x;
				info    +=      coulomb_ratio1_real<data_t>(vn+data_t(1),e,x,fm,sb);//possible code: 0 or 4
				f0[nfwd]=       (sb>0?data_t(1e-290):data_t(-1e-290))*(ir+e+fm);
				f1[nfwd]=	(sb>0?data_t(1e-290):data_t(-1e-290))*(-data_t(1)-e*e)+f0[nfwd]*(ir+e);
			}
		}else//in asymptotic region, by asymptotic expansion, where CF1=Fn'/Fn converges too slowly
		{
			comp_t	hn0,hn1,exp;
			info 	+=	coulomb_asympt_phi<data_t>(vn,e,x,hn0,hn1,+1);//possible code: 0 or 2000, pade acceleration is not used.
			exp	=	coulomb_asympt_exp_real<data_t,0>(vn,e,x);
			f0[nfwd]=	imag(exp*hn0);
			f1[nfwd]=	imag(exp*hn1);	
		}

		//backward iteration of F,F' using 2RR
		for(long i=long(nfwd);i>0;--i)
		{
			data_t l,tmr,tms,tmp;
			l	=	v+data_t(i);
			tmp	=	e/l;
			tmr	=	sqrt(data_t(1)+tmp*tmp);	//Rl=sqrt(1+e^2/l^2)
			tms	=	l/x+tmp;			//Sl=l/x+e/l
			f0[i-1]	=	(tms*f0[i  ]+    f1[i])/tmr;
			f1[i-1]	=	 tms*f0[i-1]-tmr*f0[i];
			g0[i]	=	tmr;//dump Rl to avoid sqrt twice
		}
		//determine absolute normalization of F0,F0',G0,G0'
		data_t	nm;
		if(in_osc)//if in oscillatory region, by p-iq=H-'/H- and FH'-HF'=1
		{
			comp_t	hm;
			data_t	fm,pm,qm;
			info	+=	coulomb_ratio3_real<data_t>(v,e,x,hm);//possible code: 0 or 8
			fm	=	f1[0]/f0[0];
			pm	=	real(hm);
			qm	=	-imag(hm);
			nm	=	sqrt(qm/((pm-fm)*(pm-fm)+qm*qm))/fabs(f0[0]);//the sign of f0[0] is correct. so only the ratio of norm is required.
			f0[0]	*=	nm;		
			f1[0]	*=	nm;
			g0[0]	=	(   f1[0]-pm*f0[0])/qm;
			g1[0]	=	 pm*g0[0]-qm*f0[0];
		}else//if in exponential region, F or F' calc by Taylor
		{
			info	+=	coulomb_taylor_real<data_t>(v,e,x,nm);//possible code: 0+200+800
			nm	/=	f0[0];
			f0[0]	*=	nm;		
			f1[0]	*=	nm;
			//FIXME when eta>100, taylor F is still not accurate enough between 0.75xc^2 and xc^2
			if(isvint2)//2v integer, {!! by p-iq=H0-'/H0- and FH'-HF'=1}
			{
				comp_t  hm;
                                data_t  pm,qm;
				info    +=      coulomb_ratio3_real<data_t>(v,e,x,hm);
				pm      =       real(hm);
				qm      =       -imag(hm);
				g0[0]   =       (   f1[0]-pm*f0[0])/qm;
				g1[0]   =        pm*g0[0]-qm*f0[0];			
				//FIXME when eta>100, CF2 is still not accurate enough inside exponential region
			}else//2v non-integer, use G = (FcosX-Fc)/sinX
			{
				data_t	_chi,_coschi,_sinchi,_f0c,_f1c;
				info	+=	coulomb_chi_real(v,e,_chi);
				_coschi	=	cos(_chi);
				_sinchi =	sin(_chi);
				info	+=	coulomb_taylor_real(-v-1,e,x,_f0c,_f1c);
				g0[0]	=	(_coschi*f0[0]-_f0c)/_sinchi;
				g1[0]	=	(_coschi*f1[0]-_f1c)/_sinchi;
			}
		}

		//forward iteration of G,G' using 2RR, and renormalization of F,F'
		for(long i=0;i<long(nfwd);++i)
		{
			data_t l,tmr,tms;
			l	=	v+data_t(i+1);
			tmr	=	g0[i+1];
			tms     =       l/x+e/l;
			g0[i+1]	=	(tms*g0[i]-    g1[i  ])/tmr;
			g1[i+1]	=	 tmr*g0[i]-tms*g0[i+1];
			f0[i+1]	*=	nm;
			f1[i+1]	*=	nm;
		}
		return	info;
	}//end of coulombfg

	template<class data_t>
	int	coulombfg //F only (do not update this function. copy-delete the previous one instead.)
	(
		const data_t& v,//v should not be a negative integer. we suggest |v| being as small as possible to achieve higher accuracy.
		const data_t& e,
		const data_t& x,
		data_t* f0,	//valid index 0:nfwd, filled with       F((0:nfwd)+v,e,x)
		data_t* f1,	//valid index 0:nfwd, filled with (d/dx)F((0:nfwd)+v,e,x)
		size_t nfwd
	)noexcept
	{
		bool 	isvint	=	std::floor(v)==v;
		bool	isvneg	=	v<data_t(0);
		data_t 	xc	=	e+sqrt(e*e+v*v+v);//turning point
		data_t	x2	=	x*x;
		bool	in_osc	=	x2>=xc*xc*data_t(0.75)&&x2>data_t(1e-6);
		bool	in_asy	=	x2>data_t(1000000)&&x2>=data_t(3)*(fabs(v+nfwd+1)+e*e);
		if(isvint&&isvneg)return -1;//negative integer

		using	comp_t	=	std::complex<data_t>;
		int	info	=	0;

		//determine unnormalized Fn,Fn'
		if(!in_asy)//usually, by Fn'/Fn and sign(Fn)
		{
			data_t	fm,sb,u;
			u	=	v+data_t(nfwd);
			if(u!=data_t(0))//there is divided-by-0 error if u=0
			{
				info	+=	coulomb_ratio1_real<data_t>(u,e,x,fm,sb);
				f0[nfwd]=	(sb>0?data_t(1e-290):data_t(-1e-290));
				f1[nfwd]=	(sb>0?data_t(1e-290):data_t(-1e-290))*fm;
			}else
			{
				data_t ir=	data_t(1)/x;
				info    +=      coulomb_ratio1_real<data_t>(u+data_t(1),e,x,fm,sb);
				f0[nfwd]=       (sb>0?data_t(1e-290):data_t(-1e-290))*(ir+e+fm);
				f1[nfwd]=	(sb>0?data_t(1e-290):data_t(-1e-290))*(-data_t(1)-e*e)+f0[nfwd]*(ir+e);
			}
		}else//in asymptotic region, by asymptotic expansion, where CF1=Fn'/Fn converges too slowly
		{
			comp_t	hn0,hn1,exp;
			data_t u=	v+data_t(nfwd);
			info 	+=	coulomb_asympt_phi<data_t>(comp_t(u),comp_t(e),comp_t(x),hn0,hn1,+1);//possible code:0, 2000. pade acceleration is not used
			exp	=	coulomb_asympt_exp_real<data_t,0>(u,e,x);
			f0[nfwd]=	imag(exp*hn0);
			f1[nfwd]=	imag(exp*hn1);		
		}
		//backward iteration of F,F' using 2RR
		for(long i=long(nfwd);i>0;--i)
		{
			data_t l,tmr,tms,tmp;
			l	=	v+data_t(i);
			tmp	=	e/l;
			tmr	=	sqrt(data_t(1)+tmp*tmp);	//Rl=sqrt(1+e^2/l^2)
			tms	=	l/x+tmp;			//Sl=l/x+e/l
			f0[i-1]	=	(tms*f0[i  ]+    f1[i])/tmr;
			f1[i-1]	=	 tms*f0[i-1]-tmr*f0[i];
		}
		//determine the absolute normalization of F0,F0'
		data_t	nm;
		if(!in_osc)//if in exponential region, F calc by Taylor
		{
			info	+=	coulomb_taylor_real(v,e,x,nm);
			nm	/=	f0[0];
		//FIXME when eta is positive large(>100), it's still not accurate enough between 0.75xc^2 and xc^2
		}else//if in oscillatory region, F calc by p-iq=H-'/H- and FH'-HF'=1
		{
			comp_t	hm;
			data_t	fm,qm,pm;
			info	+=	coulomb_ratio3_real<data_t>(v,e,x,hm);
			fm	=	f1[0]/f0[0];
			pm	=	real(hm);
			qm	=	-imag(hm);
			nm	=	sqrt(qm/((pm-fm)*(pm-fm)+qm*qm))/fabs(f0[0]);//the sign of f0[0] is correct. so only the ratio of norm is required.
		}
		//renormalization of F and F'
		for(long i=0;i<=long(nfwd);++i)
		{
			f0[i]	*=	nm;
			f1[i]	*=	nm;
		}
		return	info;//the sum of all runtime error code
	}//end of coulombfg	
}//end qpc
