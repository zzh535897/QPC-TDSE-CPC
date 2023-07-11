#pragma once

//require <libraries/support_std.h>
//require <libraries/support_avx.h>
//require <resources/arena.h>
//require <utilities/recid.h>
//require <functions/basis_bsplines.h>
//require <functions/basis_legendre.h>
//require <functions/basis_coulombf.h>


template<class dims_t>
struct	integrator_radi
{
	static constexpr size_t n_more = 0ul;

	using	base_t	=	typename bss::bsplines<dims_t::nr,dims_t::mr,dims_t::b_lfbd,dims_t::b_rtbd,n_more>::axis;//(0,0) for spherical, (1,0) for prolate spheroidal
	//note:
	//base_t::n_glin is the # gaussian points on each interval
	//base_t::n_knot is the # total knots (= # intervals +1)

	base_t	base;

	static constexpr size_t calc_n_upper(const size_t i)noexcept	
	{
		return	i<dims_t::n_dims-dims_t::n_diag?dims_t::n_elem:(dims_t::n_dims-i);
	}
	static constexpr size_t calc_n_lower(const size_t i)noexcept
	{
		return	i<dims_t::n_diag?i:dims_t::n_diag;
	}

	//to initialize the value of basis
	inline	void	initialize	(const double rmin,const double rmax)noexcept
	{
		base.initialize_axis_linear_step(rmin,rmax);
	}

	inline	void	initialize	(const double rmin,const double rmax,const double expm)noexcept
	{
		base.initialize_axis_exponential_step(rmin,rmax,expm);
	}

	template<size_t n_pole>
	inline	void	initialize	(const double rmin,const double rmax,const std::array<double,n_pole> pole,const std::array<size_t,n_pole> each)noexcept
	{
		if constexpr(n_pole==0)initialize(rmin,rmax);
		else base.initialize_axis_piecewise_step(rmin,rmax,pole,each);
	}

	template<size_t n_pole>
	inline	void	initialize	(const double rmin,const double rmax,const std::array<double,n_pole> pole,const std::array<size_t,n_pole> each,const double para)noexcept
	{
		if constexpr(n_pole==0)initialize(rmin,rmax,para);
		else base.initialize_axis_piecewise_step(rmin,rmax,pole,each,para);
	}
	
	//------------------------------------------------------------
	//		   one-electron integrals
	//------------------------------------------------------------
	
	//<Bi|f(r)|Bj>
	template<class retn_t,class...func_t>
	inline	void		integrate_0r0	(const retn_t& retn,const func_t&...func)const noexcept
	{
		for(size_t ir=0;ir<dims_t::n_dims;++ir)
		for(size_t ic=0;ic<dims_t::n_elem;++ic)
		{
			retn(ir,ic)	=	((base.template integrate<0,0>(func,ir,ir+ic))+...);
		}	
	}
	//<Bi|f(r)|Bj'>
	template<class retn_t,class...func_t>
	inline	void		integrate_0r1	(const retn_t& retn,const func_t&...func)const noexcept
	{
		for(size_t ir=0;ir<dims_t::n_dims;++ir)
		for(size_t ic=0;ic<dims_t::n_elem;++ic)
		{
			retn(ir,ic)	=	((base.template integrate<0,1>(func,ir,ir+ic))+...);
		}
	}
	//<Bi'|f(r)|Bj'>
	template<class retn_t,class...func_t>
	inline	void		integrate_1r1	(const retn_t& retn,const func_t&...func)const noexcept
	{
		for(size_t ir=0;ir<dims_t::n_dims;++ir)
		for(size_t ic=0;ic<dims_t::n_elem;++ic)
		{
			retn(ir,ic)	=	((base.template integrate<1,1>(func,ir,ir+ic))+...);
		}
	}	
	//<Bi|(-1/2m)*(d/dr)^2+V(r)+l(l+1)/(2mr^2)|Bj>
	template<class retn_t,class potn_t>
	inline	void		integrate_hc	(const retn_t& retn,const potn_t& potn,const double mass,const size_t l)const noexcept
	{
		double	coef	=	0.5/mass;
		auto	part1	=	[&](double r){return l*(l+1)/(r*r+1e-50);};
		auto	part2	=	[&](double r){return 1.;};
		for(size_t ir=0;ir<dims_t::n_dims;++ir)
		for(size_t ic=0;ic<dims_t::n_elem;++ic)
		{
			retn(ir,ic)	=	base.template integrate<0,0>(potn ,ir,ir+ic)
					+	base.template integrate<0,0>(part1,ir,ir+ic)*coef
					+	base.template integrate<1,1>(part2,ir,ir+ic)*coef;
		}
	}
	//(R/4/mass)*<Bi|-(d/dx)(x*x-1)(d/dx)+m*m/(x*x-1)+l(l+1)|Bj>
	//-(Z1+Z2)R^2/4*<Bi|x|Bj>
	template<class retn_t,class zadd_t>
	inline	void		integrate_hb0	(const retn_t& retn,const double dist,const double mass,const zadd_t& zadd,const long l,const long m)const noexcept
	{
		double	coef1	=	+dist/mass/4;// R/4/M
		double	coef2	=	-dist*dist/4;//-R^2/4
		double	coef3	=	l*(l+1);
		if(m%2==0)//even
		{
			auto	part1	=	[&](double x){return x*x-1.0;};
			auto	part2	=	[&](double x){return m*m/(x*x-1.0+1e-50)+coef3;};
			for(size_t ir=0;ir<dims_t::n_dims;++ir)
			for(size_t ic=0;ic<dims_t::n_elem;++ic)
			{
				retn(ir,ic)=(	base.template integrate<1,1>(part1,ir,ir+ic)
					+	base.template integrate<0,0>(part2,ir,ir+ic))*coef1
					+	base.template integrate<0,0>(zadd ,ir,ir+ic) *coef2;//for pure coulomb, zadd(x):=(Z1+Z2)*x
			}
		}else//odd
		{
			auto	square	=	[&](double x){return x*x;};
			auto	part1	=	[&](double x){return square(x-1.0);};
			auto	part3	=	[&](double x){return (x-1.0)/(x+1.0);};
			auto	part2	=	[&](double x){return (m*m+1.0)/square(x+1.0)+coef3*part3(x);};
			auto	part4	=	[&](double x){return part3(x)*zadd(x);};
			for(size_t ir=0;ir<dims_t::n_dims;++ir)
			for(size_t ic=0;ic<dims_t::n_elem;++ic)
			{
				retn(ir,ic)=(	base.template integrate<1,1>(part1,ir,ir+ic)
					+	base.template integrate<0,0>(part2,ir,ir+ic)
					+	base.template integrate<0,1>(part3,ir,ir+ic)
					+	base.template integrate<1,0>(part3,ir,ir+ic))*coef1
					+	base.template integrate<0,0>(part4,ir,ir+ic) *coef2;
			}
		}
	}
	
	//-(Z1-Z2)*R^2/4*<Bi|1|Bj>* <l,m|e|l+1,m>
	template<class retn_t,class zsub_t>
	inline	void		integrate_hb1	(const retn_t& retn,const double dist,const zsub_t& zsub,const long l,const long m)const noexcept
	{
		
		double	coef3	=	-dist*dist/4.0
				*	sqrt(double((l+1)*(l+1)-m*m)/double((2*l+1)*(2*l+3)));//-R^2/4*<l,m|eta|l+1,m>
		if(m%2==0)//even
		{
			for(size_t ir=0;ir<dims_t::n_dims;++ir)
			for(size_t ic=0;ic<dims_t::n_elem;++ic)
			{
				retn(ir,ic)	=	base.template integrate<0,0>(zsub,ir,ir+ic)*coef3;//for pure coulomb, zsub(x):=(Z2-Z1)*1
			}
		}else//odd
		{
			auto	part1	=	[&](double x){return (x-1.0)/(x+1.0)*zsub(x);};
			for(size_t ir=0;ir<dims_t::n_dims;++ir)
			for(size_t ic=0;ic<dims_t::n_elem;++ic)
			{
				retn(ir,ic)	=	base.template integrate<0,0>(part1,ir,ir+ic)*coef3;//for pure coulomb, zsub(x):=(Z2-Z1)*1
			}
		}
	}

	//<Bi,m|sqrt(x^2-1)(d/dx)sqrt(x^2-1)|Bj,m> ,for even m
	template<class retn_t>
	inline	void		integrate_bde	(const retn_t& retn)const noexcept
	{
		auto	part1	=	[&](double x){return x*x-1.0;};
		auto	part2	=	[&](double x){return x;};
		for(size_t ir=0;ir<dims_t::n_dims;++ir)
		for(size_t ic=0;ic<dims_t::n_elem;++ic)
		{
			retn(ir,ic)	=	base.template integrate<0,1>(part1,ir,ir+ic)
					+	base.template integrate<0,0>(part2,ir,ir+ic);
		}
	}
	//<Bi,m|sqrt(x^2-1)(d/dx)sqrt(x^2-1)|Bj,m> ,for odd m
	template<class retn_t>
	inline	void		integrate_bdo	(const retn_t& retn)const noexcept
	{
		auto	part1	=	[&](double x){return (x-1.0)*(x-1.0);}; 
		auto	part2	=	[&](double x){return (x-1.0);        };
		for(size_t ir=0;ir<dims_t::n_dims;++ir)
		for(size_t ic=0;ic<dims_t::n_elem;++ic)
		{
			retn(ir,ic)	=	base.template integrate<0,1>(part1,ir,ir+ic)
					+	base.template integrate<0,0>(part2,ir,ir+ic);
		}
	}
};//end of integrator_radi

template<class dims_t>
struct	integrator_angu
{
	//by default, |lm> here refers to the standard Ylm in Quantum Mechanics: 
	//	
	//	Ylm(θ,φ)= (-)^m sqrt([(2l+1)(l-m)!]/[4pi(l+m)!]) Plm(cosθ) exp(imφ)
	//
	//be careful with the phase (-)^m which are by-default excluded in GSL!
	//
	//until now, all angular integrals are explicitly written, there is no need to 
	//dump any data like the radial integrator

	template<class retn_t,class func_t>
	static inline void	integrate		(const retn_t& retn,const func_t& func)noexcept//func must be a callable function of m,l
	{
		long l,m;
		for(size_t im=0;im<dims_t::m_dims;++im)
		for(size_t il=0;il<dims_t::l_dims;++il)
		{
			dims_t::in(im,il,m,l);
			retn(im,il)=func(m,l);
		}
	}//end of integrate 
	//P10=<l1,m1|cos|l2,m2>, upper real part (symmetric)
	template<class retn_t>
	static inline void	integrate_cos		(const retn_t& retn)noexcept
	{
		auto	func	=	[](long m,long l)
		{
			return 	sqrt(double((l+1)*(l+1)-m*m)/double((2*l+1)*(2*l+3)));
		};integrate(retn,func);
	}//end of integrate_cos

	//Q10=<l1,m1|-sin*d/dth-cos|l2,m2>, upper real part (asymmetric)
	template<class retn_t>
	static	inline void	integrate_sin2dcos	(const retn_t& retn)noexcept
	{
		auto	func	=	[](long m,long l)
		{
			return 	sqrt(double((l+1)*(l+1)-m*m)/double((2*l+1)*(2*l+3)))*(l+1);
		};integrate(retn,func);
	}//end of integrate_sin2dcos

	//P1-1=<l1,m1|sin exp-|l2,m2> outer (m2=m1+1,l2=l1+1)
	template<class retn_t>
	static inline void	integrate_sinexp1	(const retn_t& retn)noexcept
	{
		auto	func	=	[](long m,long l)
		{
			return	-sqrt(double((l+m+2)*(l+m+1))/double((2*l+1)*(2*l+3)));
		};integrate(retn,func);
	}//end of integrate_sinexp1

	//P1-1=<l1,m1|sin exp-|l2,m2> inner (m2=m1+1,l2=l1-1)
	template<class retn_t>
	static inline void	integrate_sinexp2	(const retn_t& retn)noexcept
	{
		auto	func	=	[](long m,long l)
		{
			return	+sqrt(double((l-m)*(l-m-1))/double((2*l+1)*(2*l-1)));
		};integrate(retn,func);	
	}//end of integrate_sinexp2

	//P11=<l1,m1|sin exp+|l2,m2> outer
	template<class retn_t>
	static inline void	integrate_sinexp3	(const retn_t& retn)noexcept
	{
		auto	func	=	[](long m,long l)
		{
			return 	-sqrt(double((l+m)*(l+m-1))/double((2*l+1)*(2*l-1)));
		};integrate(retn,func);	
	}//end of integrate_sinexp3
	
	//P11=<l1,m1|sin exp+|l2,m2> inner
	template<class retn_t>
	static inline void      integrate_sinexp4       (const retn_t& retn)noexcept
	{	
		auto	func	=	[](long m,long l)
		{
			return 	+sqrt(double((l-m+2)*(l-m+1))/double((2*l+1)*(2*l+3)));
		};integrate(retn,func);	
	}//end of integrate_sinexp4

	//-Q1-1=<l1,m1|-costh*Lm-sinth*exp(-iph)*(1-Lz)|l2,m2> outer
	template<class retn_t>
	static inline void	integrate_coslm1	(const retn_t& retn)noexcept
	{
		auto	func	=	[](long m,long l)
		{
			return	-sqrt(double((l+m+1)*(l+m+2))/double((2*l+1)*(2*l+3)))*(l+1);
		};integrate(retn,func);	
	}//end of integrate_coslm1

	//-Q1-1=<l1,m1|-costh*Lm-sinth*exp(-iph)*(1-Lz)|l2,m2> inner
	template<class retn_t>
	static inline void	integrate_coslm2	(const retn_t& retn)noexcept
	{
		auto	func	=	[](long m,long l)
		{
			return 	-sqrt(double((l-m)*(l-m-1))/double((2*l+1)*(2*l-1)))*l;
		};integrate(retn,func);	
	}//end of integrate_coslm2

	//Q11=<l1,m1|costh*Lp-sinth*exp(+iph)*(1+Lz)|l2,m2> outer
	template<class retn_t>
	static inline void	integrate_coslp1	(const retn_t& retn)noexcept
	{
		auto	func	=	[](long m,long l)
		{
			return 	sqrt(double((l+m)*(l+m-1))/double((2*l+1)*(2*l-1)))*l;
		};integrate(retn,func);	
	}//end of integrate_coslp1

	//Q11=<l1,m1|costh*Lp-sinth*exp(+iph)*(1+Lz)|l2,m2> inner
	template<class retn_t>
	static inline void	integrate_coslp2	(const retn_t& retn)noexcept
	{
		auto	func	=	[](long m,long l)
		{
			return 	sqrt(double((l+2-m)*(l+1-m))/double((2*l+1)*(2*l+3)))*(l+1);
		};integrate(retn,func);	
	}//end of integrate_coslp2

	//P2-1=<l1,m1|costh*sinth*exp(-iph)|l2,m2> inner (l2=l1-2)
	template<class retn_t>
	static	inline	void	integrate_p21m_low	(const retn_t& retn)noexcept
	{
		auto	func	=	[](long m,long l)
		{
			return 	sqrt(double(((l-1)*(l-1)-(m+1)*(m+1))*(l-m-1)*(l-m))/double((2*l-3)*(2*l-1)*(2*l-1)*(2*l+1)));
		};integrate(retn,func);	
	}//end of integrate_p21m_mid
	
	//P2-1=<l1,m1|costh*sinth*exp(-iph)|l2,m2> middle (l2=l1)
	template<class retn_t>
	static	inline	void	integrate_p21m_mid	(const retn_t& retn)noexcept
	{
		auto	func	=	[](long m,long l)
		{
			return 	-double(2*m+1)/double((2*l+3)*(2*l-1))*sqrt((l-m)*(l+m+1));
		};integrate(retn,func);	
	}//end of integrate_p21m_mid
		
	//P2-1=<l1,m1|costh*sinth*exp(-iph)|l2,m2> outer (l2=l1+2)
	template<class retn_t>
	static	inline	void	integrate_p21m_upp	(const retn_t& retn)noexcept
	{
		auto	func	=	[](long m,long l)
		{
			return 	-sqrt(double(((l+2)*(l+2)-(m+1)*(m+1))*(l+m+2)*(l+m+1))/double((2*l+5)*(2*l+3)*(2*l+3)*(2*l+1)));
		};integrate(retn,func);	
	}//end of integrate_p21m_upp

	//<l1,m|eta|l2,m>
	template<class retn_t>
	static	inline	void	integrate_eta1_sh1	(const retn_t& retn)noexcept
	{
		auto	func	=	[](long m,long l)
		{
			return 	sqrt(double((l+1)*(l+1)-m*m)/double((2*l+1)*(2*l+3)));
		};integrate(retn,func);	
	}//end of integrate_eta1_sh1
	//<l1,m|eta^2|l2,m> sh2
	template<class retn_t>
	static	inline	void	integrate_eta2_sh2	(const retn_t& retn)noexcept
	{
		auto	func	=	[](long m,long l)
		{
			return 	sqrt(double(((l+1)*(l+1)-m*m)*((l+2)*(l+2)-m*m))/double((2*l+1)*(2*l+3)*(2*l+3)*(2*l+5)));
		};integrate(retn,func);	
	}//end of integrate_eta2_sh2

	//<l1,m|eta^2|l2,m> sh0
	template<class retn_t>
	static	inline	void	integrate_eta2_sh0	(const retn_t& retn)noexcept
	{
		auto	func	=	[](long m,long l)
		{
			return 	double(2*l*l+2*l-1-2*m*m)/double((2*l-1)*(2*l+3));
		};integrate(retn,func);	
	}//end of integrate_eta2_sh0

	//<l1,m|eta^3|l2,m> sh3
	template<class retn_t>	
	static	inline	void	integrate_eta3_sh3	(const retn_t& retn)noexcept
	{
		auto	func	=	[](long m,long l)
		{
			return 	sqrt(double(((l+1)*(l+1)-m*m)*((l+2)*(l+2)-m*m)*((l+3)*(l+3)-m*m))/double((2*l+1)*(2*l+3)*(2*l+3)*(2*l+5)*(2*l+5)*(2*l+7)));	
		};integrate(retn,func);	
	}//end of integrate_eta3_sh3
	
	//<l1,m|eta^3|l2,m> sh1
	template<class retn_t>	
	static	inline	void	integrate_eta3_sh1	(const retn_t& retn)noexcept
	{
		auto	func	=	[](long m,long l)
		{
			return	sqrt(double((l+1)*(l+1)-m*m)/double((2*l+1)*(2*l+3)))*3.0*double((l+1)*(l+1)-m*m-2)/double((2*l-1)*(2*l+5));
		};integrate(retn,func);	
	}//end of integrate_eta3_sh1

	//--------------------------------------------------
	//coupling constants (internal usage)
	static 	inline	double	coupling_3j		(const size_t l1,const size_t l2,const size_t l3)noexcept
	{
		return	gsl_sf_coupling_3j(2*l1,2*l2,2*l3,0,0,0);
	}//end of coupling_3j
	static 	inline	double	coupling_3j		(const size_t l1,const size_t l2,const size_t l3,const long m1,const long m2,const long m3)noexcept
	{
		return 	gsl_sf_coupling_3j(2*l1,2*l2,2*l3,2*m1,2*m2,2*m3);
	}//end of coupling_3j
};//end of integrator_angu

