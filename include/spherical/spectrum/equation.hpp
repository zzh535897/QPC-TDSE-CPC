#pragma once

//require <functions/coulomb.h>

//------------------------------------------------------------------------------------------------------------------
//							Equation
//
//(1) This section provides functors which are required by rk ODE solvers, see algorithm/rk.h, namely all classes
//
//have to be equipped with the concept ::cord_t and ::operator()(double const&,cord_t const&,cord_t&)const noexcept
//
//(2) All equations must implement the following ODEs where L = -(1/2M) ∇^2 + V(r) 
//
//	L[ψ] - E*ψ = 0 for positive E
//
//(3) Besides, to be used parallelly in "spherical", they must be trivially copyable objects.
//------------------------------------------------------------------------------------------------------------------
template<class type_t>
struct	equation_hydrogenic
{
	using	data_t	=	type_t;
	using	cord_t	=	std::array<data_t,2ul>;

	data_t	__z;
	data_t	__m;
	data_t	__l;
	data_t	__e;

	//original ODE: (d^2/dX^2)*Y+[1-2*Eta/X-L*(L+1)/X^2]Y == 0, X=k*r
	inline	void	operator()		(const data_t& X,const cord_t& Y,cord_t& dY)const noexcept
	{
		dY[0]	=	Y[1]; //Y[0] is Y, Y[1] is Y'
		dY[1]	=	-(data_t(1)-data_t(2)*__e/X-(__l+data_t(1))*__l/(X*X))*Y[0];
	}
	//constructor
	constexpr equation_hydrogenic		(const data_t& z,const data_t& m)noexcept:__z(z),__m(m){}
	
	//initialize the parameters for various k and l
	inline	void	initialize_param	(const data_t& k,const data_t& l)noexcept
	{
		__l	=	l;
		__e	=	-__m*__z/k;
	}		
	//initial condition for rk45 solver
	inline	bool	initial_condition	(data_t& x,cord_t& y)const noexcept//x should not be exact 0
	{
		double	init	=	__l==0.?1e-7:5e-7*(__l+1.)*__l;
		bool	from_x0	=	x<init;//to start at least from >=init
		if(!from_x0)x=init;
		coulombfg<data_t>(__l,__e,x,&(y[0]),&(y[1]),0);
		return	from_x0;
	}
	//phase shift from coulomb.h
	inline	void	compare_phaseshift	(data_t x,cord_t& y,data_t& nm,data_t& sh)const noexcept
	{
		coulombsh1(__l,__e,x,y[0],y[1],&sh,&nm);
	}
	//estimate the boundary of asymptotic region
	inline	data_t	asymptotic_region	()const noexcept
	{
		return 	std::max(__e*__e+std::fabs(__l),data_t(100));
	}
	
};//end of equation_hydrogenic

template<class type_t>
struct	equation_hydrogenic_yukawa
{
	using	data_t	=	type_t;
	using   cord_t  =       std::array<data_t,2ul>;

	data_t	__z;
	data_t	__m;
	data_t	__a;
	data_t	__s;
	data_t	__l;
	data_t	__e0;
	data_t	__e1;
	data_t	__d1;

	//original ODE: (d^2/dX^2)*Y+[1-2*Eta/X-L*(L+1)/X^2- Vs(X)/E]Y == 0, where Vs= -A*Exp[-S*r]/r
	inline	void	operator()	(const data_t& X,const cord_t& Y,cord_t& dY)const noexcept
	{
		dY[0]   =       Y[1]; //Y[0] is Y, Y[1] is Y'
                dY[1]   =       -(data_t(1)-data_t(2)*__e0/X-(__l+data_t(1))*__l/(X*X)-data_t(2)*__e1*std::exp(__d1*X)/X)*Y[0];
	}
	//constructor
	constexpr equation_hydrogenic_yukawa
	(
		const data_t& z,const data_t& m,const data_t& a,const data_t& s
	)noexcept:__z(z),__m(m),__a(a),__s(s){}

	//initialize the parameters for various k and l
	inline	void	initialize_param	(const data_t& k,const data_t& l)noexcept
	{
		__l	=	l;
		__e0	=	-__m*__z/k;
		__e1	=	-__m*__a/k;
		__d1	=	-    __s/k;
	}		
	//initial condition for rk45 solver
	inline	bool	initial_condition	(data_t& x,cord_t& y)const noexcept//x should not be exact 0
	{
		double	init	=	__l==0.?1e-7:5e-7*(__l+1.)*__l;
		bool	from_x0	=	x<init;//to start at least from >=init
		if(!from_x0)x=init;
		coulombfg<data_t>(__l,__e0+__e1,x,&(y[0]),&(y[1]),0);
		return	from_x0;
	}
	//phase shift from coulomb.h
	inline	void	compare_phaseshift	(data_t x,cord_t& y,data_t& nm,data_t& sh)const noexcept
	{
		data_t	__e;
		__e	=	__s!=data_t(0)?__e0:__e0+__e1;
		coulombsh1(__l,__e,x,y[0],y[1],&sh,&nm);
	}
	//estimate the boundary of asymptotic region
	inline	data_t	asymptotic_region	()const noexcept
	{
		return 	std::max(std::max(__e0*__e0+fabs(__l),data_t(100)),__s!=data_t(0)?data_t(-32)/__d1:0);
	}
};//end of equation_hydrogenic_yukawa


template<class type_t>
struct	equation_hydrogenic_xmtong
{
	using	data_t	=	type_t;
	using   cord_t  =       std::array<data_t,2ul>;
	
	data_t	__zc;
	data_t	__a1;
	data_t	__a2;
	data_t	__a3;
	data_t	__a4;
	data_t	__a5;
	data_t	__a6;
	data_t	__l;
	data_t	__e0;	//-zc/k
	data_t	__e1;	//-a1/k
	data_t	__e3;	//-a3/k/k
	data_t	__e5;	//-a5/k
	data_t	__d2;	//-a2/k
	data_t	__d4;	//-a4/k
	data_t	__d6;	//-a6/k

	//original ODE: (d^2/dX^2)*Y+[1-2*Eta/X-L*(L+1)/X^2- Vs(X)/E]Y == 0, where Vs= -(a1*Exp[-a2*r]+a3*r*Exp[-a4*r]+a5*Exp[-a6*r])/r
	inline	void	operator()	(const data_t& X,const cord_t& Y,cord_t& dY)const noexcept
	{
		dY[0]   =       Y[1]; //Y[0] is Y, Y[1] is Y'
                dY[1]   =       data_t(1)-(__l+data_t(1))*__l/(X*X)-data_t(2)*(
				__e0+
				__e1*std::exp(__d2*X)+
				__e3*std::exp(__d4*X)*X+
				__e5*std::exp(__d6*X))/X;
		dY[1]	=	-dY[1]*Y[0];
	}
	//constructor
	explicit equation_hydrogenic_xmtong
	(
		const data_t& zc,const data_t& a1,const data_t& a2,const data_t& a3,const data_t& a4,const data_t& a5,const data_t& a6
	):__zc(zc),__a1(a1),__a2(a2),__a3(a3),__a4(a4),__a5(a5),__a6(a6)
	{
		if(a1==data_t(0)||a2==data_t(0)||a3==data_t(0)||a4==data_t(0)||a5==data_t(0)||a6==data_t(0))//in principle, a's are not expected to be zero
		{
			throw	errn_t("In xmtong's potential, the ode model does not accept a's to be 0.\n");		
		}
	}
	//initialize the parameters for various k and l
	inline	void	initialize_param	(const data_t& k,const data_t& l)noexcept
	{
		__l	=	l;
		__e0	=	-__zc/k;
		__e1	=	-__a1/k;
		__e3	=	-__a3/k/k;
		__e5	=	-__a5/k;
		__d2	=	-__a2/k;
		__d4	=	-__a4/k;
		__d6	=	-__a6/k;
	}
	//initial condition for rk45 solver
	inline	bool	initial_condition	(data_t& x,cord_t& y)const noexcept//x should not be exact 0
	{
		double	init	=	__l==0.?1e-7:5e-7*(__l+1.)*__l;
		bool	from_x0	=	x<init;//to start at least from >=init
		if(!from_x0)x=init;
		coulombfg<data_t>(__l,__e0+__e1+__e5,x,&(y[0]),&(y[1]),0);
		return	from_x0;
	}
	//phase shift from coulomb.h
	inline	void	compare_phaseshift	(data_t x,cord_t& y,data_t& nm,data_t& sh)const noexcept
	{
		coulombsh1(__l,__e0,x,y[0],y[1],&sh,&nm);
	}
	//estimate the boundary of asymptotic region
	inline	data_t	asymptotic_region	()const noexcept
	{
		data_t	x1,x2;
		x1	=	std::max(data_t(-32)/__d2,data_t(-32)/__d4);
		x2	=	std::max(data_t(-32)/__d6,__e0*__e0+fabs(__l));
		return 	std::max(std::max(x1,x2),data_t(100));
	}	
};//end of equation_hydrogenic_xmtong

template<class type_t,class potn_t>
struct	equation_general
{
	using	data_t	=	type_t;
	using   cord_t  =       std::array<data_t,2ul>;

	data_t	_z;
	data_t	_k;
	data_t	_l;
	data_t	_e;
	data_t	_r;
	potn_t*	_v;

	//original ODE: (d^2/dX^2)*Y+[1-L*(L+1)/X^2 + 2V(X/k)/k^2]Y == 0
	inline	void	operator()	(const data_t& X,const cord_t& Y,cord_t& dY)const noexcept
	{
		dY[0]	=	Y[1];
		dY[1]	=	-(1. - _l*(_l+1.)/(X*X) - (*_v)(X/_k)*(2./(_k*_k))) * Y[0];
	}

	explicit equation_general(potn_t* potn, data_t zasy=0., data_t rasy=100.):_z(zasy),_r(rasy),_v(potn){}

	//initialize the parameters for various k and l
	inline	void	initialize_param	(const data_t& k,const data_t& l)noexcept
	{
		_k = k;
		_l = l;
		_e = -_z/k;
	}	
	//initial condition for rk45 solver
	inline	bool	initial_condition	(data_t& x,cord_t& y)const noexcept//x should not be exact 0
	{
		double	init	=	_l==0.?1e-7:5e-7*(_l+1.)*_l;
		bool	from_x0	=	x<init;//to start at least from >=init
		if(!from_x0)x=init;
		//type0: correct for debug
		//coulombfg<data_t>(_l,_e,x,&(y[0]),&(y[1]),0);
		//type1: less accurate
		//y[0]=data_t(0);
		//y[1]=data_t(1);
		//type2: taylor expansion from F/F'
		y[0]=data_t(1e-290);
		y[1]=y[0]*(data_t(1)+_l)/x;
		return	from_x0;
	}
	//phase shift from coulomb.h
	inline	void	compare_phaseshift	(data_t x,cord_t& y,data_t& nm,data_t& sh)const noexcept
	{
		coulombsh1(_l,_e,x,y[0],y[1],&sh,&nm);
	}
	//estimate the boundary of asymptotic region
	inline	data_t	asymptotic_region	()const noexcept
	{
		return 	std::max(3.0*(_e*_e+fabs(_l)),_r);
	}
};//end of equation_general

//------------------------------------------------------------------------------------------------------------------
//					   Equation Generator
//------------------------------------------------------------------------------------------------------------------

//the pre-defined way to create an ode-function object
template<class potn_t,class...args_t>
static constexpr auto	generate_odefun_from_potential	(const potn_t& _potn,const double mass,args_t&&...args)noexcept
{
	if constexpr(compare_v<potn_t,spherical::hydrogenic_xmtong>)
	{
		if(mass!=1.0)printf("warning! invalid mass for xmtong potential!\n");

		return  spherical::equation_hydrogenic_xmtong
		(
			_potn.zc,_potn.a1,_potn.a2,_potn.a3,_potn.a4,_potn.a5,_potn.a6 //mass defaults to 1
		);
	}else
	if constexpr(compare_v<potn_t,spherical::hydrogenic_yukawa>)  
	{
		return  spherical::equation_hydrogenic_yukawa
		(
			_potn.zc,mass,_potn.zs,_potn.sg
		);
	}else
	if constexpr(compare_v<potn_t,spherical::hydrogenic>)//note: we recommend using direct FG evaluation instead of numerical ODE solver
	{
		return  spherical::equation_hydrogenic
		(
			_potn.zc,mass
		);
	}else
	{
		if(mass!=1.0)printf("warning! invalid mass for general potential!\n");

		return	spherical::equation_general(&_potn,std::forward<args_t>(args)...);
	}	
}//end of generate_odefun_from_potential

//------------------------------------------------------------------------------------------------------------------
//					ODE solution container
//------------------------------------------------------------------------------------------------------------------
template<class data_t>
struct	solution
{
	private:
		data_t*	rh;	//an array of size n holding rho=kr, !!in rising order!!
		data_t*	u0;	//an array of size n holding one of the partial wave, i.e. the normalized u (kr;l,eta)
		data_t*	u1;	//an array of size n holding one of the partial wave, i.e. the normalized u'(kr;l,eta). (<k1|k2>=delta(k1-k2))
		data_t	sh;	//coulomb phase shift
		size_t	n;
		rsrc_t*	rsrc;

	public:	
		//access data
		inline	data_t*	_rh()const noexcept{return rh;}
		inline	data_t*	_u0()const noexcept{return u0;}
		inline	data_t*	_u1()const noexcept{return u1;}
		inline	data_t	_sh()const noexcept{return sh;}

		inline	data_t&	_rh(const size_t i)noexcept{return rh[i];}
		inline	data_t&	_u0(const size_t i)noexcept{return u0[i];}
		inline	data_t&	_u1(const size_t i)noexcept{return u1[i];}

		//memory allocation
		solution()=delete;
		solution(solution const&)=delete;
		solution& operator=(solution const&)=delete;
		solution& operator=(solution&&)=delete;

		explicit solution	(const size_t _n,rsrc_t* _rsrc=global_default_arena_ptr):n(_n),rsrc(_rsrc)
		{
			if(rsrc!=0)
			{
				rsrc->acquire((void**)&rh,sizeof(data_t)*n,align_val_avx3);
				rsrc->acquire((void**)&u0,sizeof(data_t)*n,align_val_avx3);
				rsrc->acquire((void**)&u1,sizeof(data_t)*n,align_val_avx3);
			}else
			{
				throw errn_t("invalid arena for constructing solution.\n");
			}
		}//end of constructor
		solution(solution&& rhs)noexcept
		{
			rh=rhs.rh;
			u0=rhs.u0;
			u1=rhs.u1;
			n =rhs.n;
			sh=rhs.sh;
			rsrc=rhs.rsrc;
			rhs.rsrc=0;	
		}//end of constructor (rvalue copy)
		~solution()noexcept
		{
			if(rsrc)
			{
				rsrc->release(rh);
				rsrc->release(u0);
				rsrc->release(u1);
			}
		}//end of destructor
		
		int static constexpr ode_solver 	=	1;	//1=rk45
									//2=rk5m

		//calculate the solution on given points by rk45?rk5m, and determine the phase shift
		template<class odef_t>
		void	initialize	(const odef_t& odef)noexcept
		{
			std::array<data_t,2> rtol	=	{data_t(5e-11),data_t(5e-11)};	
			std::array<data_t,2> temp;
			data_t	init=rh[0];
			bool	from_rho0=odef.initial_condition(init,temp);//@@require odef_t::initial_condition

			if(!from_rho0)
			{
				data_t	rh0 = rh[0];;
				if constexpr(ode_solver==1)rk45(odef,temp,temp,init,rh0,rtol);
				if constexpr(ode_solver==2)rk5m(odef,temp,temp,init,rh0,rtol);
			}
				u0[0]=temp[0];u1[0]=temp[1];
			for(size_t i=1;i<n;++i)
			{
				data_t  rh1 = rh[i-1];
				data_t 	rh2 = rh[i];
				if constexpr(ode_solver==1)rk45(odef,temp,temp,rh1,rh2,rtol);
				if constexpr(ode_solver==2)rk5m(odef,temp,temp,rh1,rh2,rtol);
				if(temp[0]>1e155)
				{
					temp[0]*=1e-290;
					temp[1]*=1e-290;
					for(size_t j=0;j<i;++j)
					{
						u0[j]*=1e-290;
						u1[j]*=1e-290;
					}
				}
				u0[i]=temp[0];u1[i]=temp[1];
			}
			data_t	nm;
			data_t	large=odef.asymptotic_region();//@@ require odef_t::asymptotic_region
			if(rh[n-1]<large)
			{	
				data_t	_temp0 = temp[0];
				data_t	_ratio = temp[1]/temp[0];
				temp[0]= 1e-290;
				temp[1]= 1e-290*_ratio;
				if constexpr(ode_solver==1)rk45(odef,temp,temp,rh[n-1],large,rtol);	
				if constexpr(ode_solver==2)rk5m(odef,temp,temp,rh[n-1],large,rtol);
				odef.compare_phaseshift(  large,temp,nm,sh);//@@require odef_t::compare_phaseshift
				nm/=(_temp0*1e290);
			}else
			{
				odef.compare_phaseshift(rh[n-1],temp,nm,sh);//@@require odef_t::compare_phaseshift
			}
			//renormalization
			for(size_t i=0;i<n;++i)
			{
				u0[i]*=nm;
				u1[i]*=nm;
			}
		}//end of initialize		
};//end of solution

