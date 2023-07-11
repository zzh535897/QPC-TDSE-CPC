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
//					Runge-Kutta Integrator
//
//	References:
//	[1] Süli, Endre; Mayers, David (2003), An Introduction to Numerical Analysis, 
//	Cambridge University Press, ISBN 0-521-00794-1 
//	[2] Fehlberg, E (1958). "Eine Methode zur Fehlerverkleinerung beim Runge-Kutta-Verfahren". 
//	Zeitschrift für Angewandte Mathematik und Mechanik. 38 (11/12): 421–426 
//
//	Description:
//	(1) The algorithm solves ordinary differential equation (arrays) by the traditional 
//	Runge-Kutta algorithm.
//
//	To call the routines, one should prepare a functor class which models the ODEs. An
//	example:
//
//		struct odefun
//		{
//			using cord_t 	=	std::array<double,N>;
//
//			void operator()(const double t,const cord_t& x,cord_t& dx);
//		};
//
//	which describes the ode
//
//		dxi/dt = fi(x0,x1,...), i=0,1,...,N-1
//
//	To solve it, one should then call rk44,rk45 or rk5m by:
//
//	rk44(const odef_t& odeobj,const cord_t& xi,cord_t& xf,const double ti,const double tf,const cord_t& tol);
//	rk45(const odef_t& odeobj,const cord_t& xi,cord_t& xf,const double ti,const double tf,const cord_t& tol);
//	rk5m(const odef_t& odeobj,const cord_t& xi,cord_t& xf,const double ti,const double tf,const cord_t& tol);
//
//Created by: Zhao-Han Zhang(张兆涵)  Apr. 20th, 2020
//Updated by: Zhao-Han Zhang(张兆涵)  Dec. 15th, 2022
//
//Copyright © 2022 Zhao-Han Zhang
//====================================================================================================

//====================================================================================================	
//					Numerov Integrator
//
//	Description:
//	(1) The algorithm solves second order ordinary differential equation (arrays) by Numerov method.
//
//Updated by: Zhao-Han Zhang(张兆涵)  Dec. 15th, 2022
//
//Copyright © 2022 Zhao-Han Zhang
//====================================================================================================	

namespace qpc
{
	
	template<class odef_t>struct solver_rk44;
        template<class odef_t>struct solver_rk45;
        template<class odef_t>struct solver_rk5m;
	template<class odef_t>struct solver_nv;
	//=================================================================================================
	//					interfaces
	//=================================================================================================
	template<class odef_t,class...args_t>
        inline	auto    rk44    (const odef_t& odefun,args_t&&...args)
        {
                return  solver_rk44<odef_t>::solve(odefun,std::forward<args_t>(args)...);
        }
        template<class odef_t,class...args_t>
        inline  auto    rk45    (const odef_t& odefun,args_t&&...args)
        {
                return  solver_rk45<odef_t>::solve(odefun,std::forward<args_t>(args)...);
        }
        template<class odef_t,class...args_t>
        inline  auto    rk5m    (const odef_t& odefun,args_t&&...args)
        {
                return  solver_rk5m<odef_t>::solve(odefun,std::forward<args_t>(args)...);
        }
	template<class odef_t,class...args_t>
	inline	auto	numerov	(const odef_t& odefun,args_t&&...args)
	{
		return 	solver_nv<odef_t>::solve(odefun,std::forward<args_t>(args)...);
	}
	//=================================================================================================
	//				  some useful constants
	//=================================================================================================
	struct	rk4
	{
		static constexpr std::array<double,4> a	=	{0.0,0.5,0.5,1.0};
		static constexpr std::array<double,4> b	=	{0.0,0.5,0.5,1.0};
		static constexpr std::array<double,4> c	=	{1.0/6.0,1.0/3.0,1.0/3.0,1.0/6.0};
	};
	struct	rk5
	{
		static constexpr std::array<double,5>	a	=	{1.0/4.0,3.0/8.0,12.0/13.0,1.0,1.0/2.0};
		static constexpr std::array<double,1> 	b0  	=       {1.0/4.0};
		static constexpr std::array<double,2>	b1	=	{3.0/32.0,9.0/32.0};
		static constexpr std::array<double,3>	b2	=	{1932.0/2197.0,-7200.0/2197.0,7296.0/2197.0};
		static constexpr std::array<double,4>	b3	=	{439.0/216.0,-8.0,3680.0/513.0,-845.0/4104.0};
		static constexpr std::array<double,5>	b4	=	{-8.0/27.0,2.0,-3544.0/2565.0,1859.0/4104.0,-11.0/40.0};

		static constexpr std::array<double,6>	c5	=	{16.0/135.0,0.0,6656.0/12825.0,28561.0/56430.0,-9.0/50.0,2.0/55.0};
		static constexpr std::array<double,6>	c4	=	{25.0/216.0,0.0,1408.0/2565.0,2197.0/4104.0,-1.0/5.0,0.0};
	};
	//=================================================================================================
	//				some useful functions
	//=================================================================================================
	struct	rk_helper
	{
		static constexpr double	dmax	=	0.2;
		static constexpr double dmin	=	1e-12;	
		
		static inline bool	is_hitting_boundary     (const double tnow,const double tend,const double step)noexcept
		{
			return  (tnow+1.0125*step-tend)*(tnow-tend)<=0.0;
		}
		static inline bool	cannot_double_more	(const double tnow,const double tend,const double step)noexcept
		{
			return 	fabs(step)>dmax||is_hitting_boundary(tnow,tend,step);
		}
		static inline bool	cannot_halfen_more	(const double step)noexcept
		{
			return 	fabs(step*0.5)<dmin;
		}
		template<class cord_t>
		static inline bool	is_within_tolerance     (const cord_t& tol,const cord_t& ful,const cord_t& hlf)noexcept
		{
			for(size_t i=0;i<tol.size();++i)if(fabs(1.-hlf[i]/ful[i])>tol[i])return 0;
			return 	1;
		}
		//
		template<class cord_t>
		static inline bool	is_proper_tolerance     (const cord_t& tol)noexcept
		{
			for(size_t i=0;i<tol.size();++i)if(tol[i]<=0)return 0;
			return 	1;
		}
		static inline double	generate_initial_dt	(const double tini,const double tend)noexcept
		{
			double  dt      =       (tend-tini)/4.0;
			while(fabs(dt)>0.5*dmax)dt=dt/2.0;
			return 	dt;
		}
		static inline bool	is_proper_initial_t	(const double tini,const double tend)noexcept
		{
			return	fabs(tend-tini)>dmin*4.0;
		}
	};
	//=================================================================================================
	//					implementations
	//=================================================================================================
	template<class odef_t>
	struct	solver_rk44:public rk4
	{
		using	cord_t	=	typename odef_t::cord_t;

		using rk4::a;
		using rk4::b;
		using rk4::c;

		static constexpr size_t	n	=	cord_t{}.size();
	
		static 	void	kernel	(const odef_t& odefun,const cord_t& valnow,cord_t& valret,const double t,const double h)noexcept
		{
			cord_t	valmid,valtmp;
			//STEP0
			odefun(t,valnow,valtmp);
			for(size_t i=0;i<n;++i)
			valret[i]	=	valnow[i]+h*c[0]*valtmp[i];
			//STEPi
			for(size_t s=1;s<4;++s){
			for(size_t i=0;i<n;++i)
			valmid[i]       =       valnow[i]+h*b[s]*valtmp[i];
			odefun(t+h*a[s],valmid,valtmp);
			for(size_t i=0;i<n;++i)
			valret[i]       =       valret[i]+h*c[s]*valtmp[i];}
		}//end of kernel
		
		static	int	solve	(const odef_t& odefun,const cord_t& xi,cord_t& xf,const double ti,const double tf,const cord_t& tol)noexcept
		{
			if(!rk_helper::is_proper_tolerance(tol))
			{
				return 	-1;//rk44 solver: invalid tolerance.
			}
			if(!rk_helper::is_proper_initial_t(ti,tf))//by lower rank
			{
				cord_t	dxi;
				odefun(ti,xi,dxi);
                                for(size_t i=0;i<n;++i)
				xf[i]=xi[i]+(tf-ti)*dxi[i];
				return 	0;//succeed
			}
			cord_t  valtmp[4];				//work as temporary
			cord_t* valfs1          =       &valtmp[0];	//full step
			cord_t* valhs1          =       &valtmp[1];     //half step 1
			cord_t* valhs2          =       &valtmp[2];     //half step 2
			cord_t* valnow          =       &valtmp[3];
			*valnow			=	xi;
			double	tn		=	ti;
			double  h               =       rk_helper::generate_initial_dt(ti,tf);//set dt as (tf-ti)/2^n until it's less than 0.5*dmax
			
			while(1)//this loop iterates all over the span [ti,tf]
			{
				if(rk_helper::is_hitting_boundary(tn,tf,h)){
				kernel(odefun,*valnow,*valfs1,tn,tf-tn);swap(valnow,valfs1);break;}
				kernel(odefun,*valnow,*valhs1,tn,h/2);
                        	kernel(odefun,*valhs1,*valhs2,tn+h/2,h/2);
                        	kernel(odefun,*valnow,*valfs1,tn,h);
				if(rk_helper::is_within_tolerance(tol,*valfs1,*valhs2))
				{//doubling
					do
					{
						swap(valfs1,valhs1);
						if(rk_helper::cannot_double_more(tn,tf,h*=2))break;
                                                kernel(odefun,*valhs1,*valhs2,tn+h/2,h/2);
                                                kernel(odefun,*valnow,*valfs1,tn    ,h  );
					}while(rk_helper::is_within_tolerance(tol,*valfs1,*valhs2));
					swap(valnow,valhs1);tn+=(h/=2);
				}else
				{//halfening
					do
					{
						if(rk_helper::cannot_halfen_more(h))break;
						h/=2;
						swap(valfs1,valhs1);
                                                kernel(odefun,*valnow,*valhs1,tn    ,h/2);
                                                kernel(odefun,*valhs1,*valhs2,tn+h/2,h/2);
					}while(!rk_helper::is_within_tolerance(tol,*valfs1,*valhs2));
					swap(valnow,valhs2);tn+=h;
				}
			}
			xf=*valnow;
			return	0;//succeed
		}//end of solve
	};//end of solver_rk44

	template<class odef_t>
        struct  solver_rk45
        {
                using   cord_t  =       typename odef_t::cord_t;

		static constexpr size_t n	=       cord_t{}.size();

		static	void      kernel	(const odef_t& odefun,const cord_t& valnow,cord_t& valrk4,cord_t& valrk5,const double t,const double h)noexcept
		{
			cord_t	valmid,valf[6];
			//STEP0
			odefun(t,valnow,valf[0]);
			//STEP1
			for(size_t i=0;i<n;++i)
			valmid[i]       =       valnow[i]+h*(rk5::b0[0]*valf[0][i]);
			odefun(t+h*rk5::a[0],valmid,valf[1]);
			//STEP2
			for(size_t i=0;i<n;++i)
			valmid[i]	=	valnow[i]+h*(rk5::b1[0]*valf[0][i]+rk5::b1[1]*valf[1][i]);
			odefun(t+h*rk5::a[1],valmid,valf[2]);
			//STEP3
			for(size_t i=0;i<n;++i)
			valmid[i]       =       valnow[i]+h*(rk5::b2[0]*valf[0][i]+rk5::b2[1]*valf[1][i]+rk5::b2[2]*valf[2][i]);
			odefun(t+h*rk5::a[2],valmid,valf[3]);
			//STEP4
			for(size_t i=0;i<n;++i)
                       	valmid[i]       =       valnow[i]+h*(rk5::b3[0]*valf[0][i]+rk5::b3[1]*valf[1][i]+rk5::b3[2]*valf[2][i]+rk5::b3[3]*valf[3][i]);
			odefun(t+h*rk5::a[3],valmid,valf[4]);
			//STEP5
			for(size_t i=0;i<n;++i)
			valmid[i]       =       valnow[i]+h*(rk5::b4[0]*valf[0][i]+rk5::b4[1]*valf[1][i]+rk5::b4[2]*valf[2][i]+rk5::b4[3]*valf[3][i]+rk5::b4[4]*valf[4][i]);
			odefun(t+h*rk5::a[4],valmid,valf[5]);
			//EVAL
			for(size_t i=0;i<n;++i)
			valrk4[i]       	=       valnow[i]+h*(rk5::c4[0]*valf[0][i]+rk5::c4[2]*valf[2][i]+rk5::c4[3]*valf[3][i]+rk5::c4[4]*valf[4][i]);
			for(size_t i=0;i<n;++i)
			valrk5[i]       	=       valnow[i]+h*(rk5::c5[0]*valf[0][i]+rk5::c5[2]*valf[2][i]+rk5::c5[3]*valf[3][i]+rk5::c5[4]*valf[4][i]+rk5::c5[5]*valf[5][i]);
		}//end of kernel

		static	int 	solve   (const odef_t& odefun,const cord_t& xi,cord_t& xf,const double ti,const double tf,const cord_t& tol)noexcept
		{
			if(!rk_helper::is_proper_tolerance(tol))
			{
				return 	-1;//rk45 solver: invalid tolerance
			}
                        if(!rk_helper::is_proper_initial_t(ti,tf))
			{
				return 	-2;//rk45 solver: invalid initial time
			}
			cord_t	valtmp[4];//FIXME what if cord_t is really big?
			cord_t	*valnow	=	&valtmp[0];
			cord_t	*valrk4	=	&valtmp[1];
			cord_t	*valrk5	=	&valtmp[2];
			cord_t	*valpre	=	&valtmp[3];
			*valnow		=	xi;
                        double  tn    	=       ti;
                        double  h     	=       rk_helper::generate_initial_dt(ti,tf);//set dt as (tf-ti)/2^n until it's less than 0.5*dmax

			while(1)//this loop iterates all over the span [ti,tf]
			{
				if(rk_helper::is_hitting_boundary(tn,tf,h)){
				kernel(odefun,*valnow,*valrk4,*valnow,tn,tf-tn);break;}
				kernel(odefun,*valnow,*valrk4,*valrk5,tn,h);
				if(rk_helper::is_within_tolerance(tol,*valrk4,*valrk5))
				{//doubling
					do
					{
						swap(valpre,valrk5);
						if(rk_helper::cannot_double_more(tn,tf,h*=2))break;
						kernel(odefun,*valnow,*valrk4,*valrk5,tn,h);
					}while(rk_helper::is_within_tolerance(tol,*valrk4,*valrk5));
					swap(valnow,valpre);tn+=(h/=2);
				}else
				{//halfening
					do
					{
						if(rk_helper::cannot_halfen_more(h))break;
						h/=2;
						kernel(odefun,*valnow,*valrk4,*valrk5,tn,h);
					}while(!rk_helper::is_within_tolerance(tol,*valrk4,*valrk5));
					swap(valnow,valrk5);tn+=h;
				}
			}
			xf=*valnow;
			return	0;//succeed
		}//end of solve
	};//end of solver_rk45

	template<class odef_t>
	struct	solver_rk5m
	{
		using   cord_t  =       typename odef_t::cord_t;
	
		static constexpr std::array<double,6> a		=	{1./5.,3./10.,4./5.,8./9.,1.,1.};
		static constexpr std::array<double,1> b0	=	{1./5.};
		static constexpr std::array<double,2> b1	=	{3./40.,9./40.};
		static constexpr std::array<double,3> b2	=	{44./45.,-56./15.,32./9.};
		static constexpr std::array<double,4> b3	=	{19372./6561.,-25360./2187.,64448./6561.,-212./729.};
		static constexpr std::array<double,5> b4	=	{9017./3168.,-355./33.,46732./5247.,49./176.,-5103./18656.};
		static constexpr std::array<double,6> c		=	{35./384.,0.,500./1113.,125./192.,-2187./6784.,11./84.};
		static constexpr std::array<double,7> e		=	{71./57600., 0., -71./16695.,71./1920., -17253./339200.,22./525.,-1./40.};
		
		static constexpr size_t n       =       cord_t{}.size();

		template<class lhst,class rhst>
		static constexpr auto	max(const lhst& lhs,const rhst& rhs)noexcept{return lhs>rhs?lhs:rhs;}	

		static	void      kernel  (const odef_t& odefun,const cord_t& valnow,cord_t& valnew,cord_t& valerr,const double t,const double h)noexcept
		{
			cord_t valmid,valf[7];
			//STEP0
			odefun(t,valnow,valf[0]);
			//STEP1
			for(size_t i=0;i<n;++i)
			valmid[i]       =       valnow[i]+h*(b0[0]*valf[0][i]);
			odefun(t+h*a[0],valmid,valf[1]);
			//STEP2
			for(size_t i=0;i<n;++i)
			valmid[i]	=	valnow[i]+h*(b1[0]*valf[0][i]+b1[1]*valf[1][i]);
			odefun(t+h*a[1],valmid,valf[2]);
			//STEP3
			for(size_t i=0;i<n;++i)
			valmid[i]       =       valnow[i]+h*(b2[0]*valf[0][i]+b2[1]*valf[1][i]+b2[2]*valf[2][i]);
			odefun(t+h*a[2],valmid,valf[3]);
			//STEP4
			for(size_t i=0;i<n;++i)
                       	valmid[i]       =       valnow[i]+h*(b3[0]*valf[0][i]+b3[1]*valf[1][i]+b3[2]*valf[2][i]+b3[3]*valf[3][i]);
			odefun(t+h*a[3],valmid,valf[4]);
			//STEP5
			for(size_t i=0;i<n;++i)
			valmid[i]       =       valnow[i]+h*(b4[0]*valf[0][i]+b4[1]*valf[1][i]+b4[2]*valf[2][i]+b4[3]*valf[3][i]+b4[4]*valf[4][i]);
			odefun(t+h*a[4],valmid,valf[5]);
			//EVAL
			for(size_t i=0;i<n;++i)
			valnew[i]       =       valnow[i]+h*(c[0]*valf[0][i]+c[2]*valf[2][i]+c[3]*valf[3][i]+c[4]*valf[4][i]+c[5]*valf[5][i]);
			odefun(t+h*a[5],valnew,valf[6]);//FIXME t+h may not exactly equal tf?
			//ERROR ESTIMATE
			for(size_t i=0;i<n;++i)
			valerr[i]	=	fabs(e[0]*valf[0][i]+e[2]*valf[2][i]+e[3]*valf[3][i]+e[4]*valf[4][i]+e[5]*valf[5][i]+e[6]*valf[6][i]);
		}//end of kernel

		static	int    	solve   (const odef_t& odefun,const cord_t& xi,cord_t& xf,const double ti,const double tf,const cord_t& tol)noexcept
		{
			if(!rk_helper::is_proper_tolerance(tol))
			{
				return	-1;//rk5m solver: invalid tolerance.
			}
                        if(!rk_helper::is_proper_initial_t(ti,tf))//by lower rank method
			{
				cord_t  dxi;
                                odefun(ti,xi,dxi);
                                for(size_t i=0;i<n;++i)xf[i]=xi[i]+(tf-ti)*dxi[i];
                                return  0;//succeed
			}
			cord_t  valtmp[2];
			cord_t	*valnow	=	&valtmp[0];
			cord_t	*valnew	=	&valtmp[1];
			cord_t	valerr;
			*valnow		=	xi;
			double  tn     	=       ti;
                        double  h       =       rk_helper::generate_initial_dt(ti,tf);//set dt as (tf-ti)/2^n until it's less than 0.5*dmax
			while(1)//this loop iterates all over the span [ti,tf]
                        {
				double	hmin	=	fabs(tn*std::numeric_limits<double>::epsilon())*64.;
				bool	done;
				double	merr	=	0.;
				double	temp;
                                if(rk_helper::is_hitting_boundary(tn,tf,h))
				{
                                	kernel(odefun,*valnow,*valnow,valerr,tn,tf-tn);break;
				}
				while(1)//this loop calculate one step
				{
                                	kernel(odefun,*valnow,*valnew,valerr,tn,h);
					done=1;
					merr=0.;
					for(size_t i=0;i<n;++i)
					{//check tol
						temp=valerr[i]/max(fabs((*valnow)[i]),0.01)/tol[i];//0.01 is newly modified
						if(temp>1.00)done=0;
						if(temp>merr)merr=temp;
					}
					if(done||fabs(0.5*h)<hmin)
        	                        {//finish this step, predict the next
						swap(valnow,valnew);
						tn+=h;
						temp=1.25*pow(merr,0.2);
						h=temp>0.2?h/temp:h*5.0;
						h=max(hmin,h);
						break;
                                	}else
	                                {//halfen h and redo this step
						h*=0.5;
					}
                                }
                        }
                        xf=*valnow;
			return	0;//succeed
		}//end of solve
	};//end of solver_rk5m

	
	template<class odef_t>
	struct	solver_nv
	{
		public:
			using	cord_t	=	typename odef_t::cord_t;

			static	void	kernel	
			(
				cord_t const& y0,cord_t const& y1,cord_t& y2,
				cord_t const& g0,cord_t const& g1,cord_t const& g2,
				double const  h2
			)noexcept
			{
				y2=(y1*(2.-5./6.*h2*g1)-y0*(1.+h2/12.*g0))/(1.+g2*h2/12.);
			}//end of kernel

			static	int       solve	
			(
				const odef_t& odefun,//calculate g
				cord_t* y,	//y0,y1 should be filled with proper initial values
				double  t0,	//init
				double	h,	//step
				size_t  n	//length of y and t
			)
			{
				cord_t g[3];
				cord_t* g0=&g[0];
				cord_t* g1=&g[1];
				cord_t* g2=&g[2];
				double h2=h*h;
				odefun(t0  ,*g0);
				odefun(t0+h,*g1);
				for(size_t i=2;i<n;++i)
				{
					odefun(t0+i*h,*g2);
					y[i]=	(y[i-1]*(2.- 5./6.*h2**g1)
						-y[i-2]*(1.+1./12.*h2**g0))
						/(1.+1./12.*h2**g2);
					std::swap(g0,g1);
					std::swap(g2,g1);
				}
				return 	0;		
			}//end of solve
	};//end of solver_nv
}//end of qpc
