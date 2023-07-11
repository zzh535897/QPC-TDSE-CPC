#pragma once

#include <libraries/support_std.h>

#include <algorithm/rk.h>
#include <algorithm/pd.h>
#include <algorithm/cf.h>
#include <algorithm/nr.h>
#include <algorithm/hd.h>
#include <algorithm/ed.h>
//====================================================================================================
//
//                      This file is part of QPC Header Only Library
//
//====================================================================================================

//====================================================================================================
//			  Generalized Spheroidal Wave Function (GSWF)
//
//References:
//[1] P. T. Greenland, Theoret. Chim. Acta (Berl.) 42, 273 (1976)
//[2] L. I. Ponomarev & L. N. Somov, J. Comput. Phys. 20 183 (1976)
//[3] D. I. Abramov & S. Y. Slavyanov, J. Phys. B: At. Mol. Phys. 11 2229 (1978)
//[4] J. Rankin& W. R. Thorson, J. Commput. Phys. 32 437 (1979) 
//[5] E. W. Leaver, J. Math. Phys. 27, 1238 (1985)
//[6] G. Hadinger et al, , J. Phys. B: At. Mol. Phys. 23 1625 (1990)
//[7] Y. S. Tergiman, Phys. Rev. A 48 1 (1993)
//
//Descriptions:				
//(0) naming conventions for variables.
//
//Let L denote lambda= (r1+r2)/rn
//    U denote mu    = (r2-r1)/rn, rn = nuclear distance, r1&r2 = distance from nucleus 1&2 to the electron
//    R denote the radial solution
//    S denote the angular solution
//
//    A denote the angular eigenvalue a
//    M denote the azimuthal eigenvalue m
//    K denote kappa = k*rn/2, where k=sqrt(2Ek)
//    E denote eta = -(Z1+Z2)/k or -(Z1-Z2)/k, Z1,Z2 are the positive charge
//
//(1) choice of transformation.
//
//	Type0: 	original R
//	Type1:	Ra, where R=(L^2-1)^(M/2)*Ra
//	Type2:  Rb, where R=[(L-1)/(L+1)]^(M/2)*Rb
//	Type3:  Rc, where R=[(L+1)/(L-1)]^(M/2)*Rc
//
//	Note that one may convert from Type2 to Type3 by changing the sign of M when evaluating Rb or Rc.
//
//(2) ordinary differential equation.
//
//	spheroidal_radi_odefun	----	the ode object for ode solver to solve
//	spheroidal_angu_odefun	----	the ode object for ode solver to solve
//	
//
//(3) evaluating radial functions.
//
//	spheroidal_taylor_phi	---- 	the taylor expansion part of Ra,Ra' 
//	spheroidal_taylor	----	calculate Ra,Ra' or Rb,Rb' or Rc,Rc' or R,R' by taylor expansion. 
//					only valid near 1.
//		
//	spheroidal_asympt_phi 	----	the asymptotic expansion part of Rb,Rb' or Rc,Rc' with coulomb
//					phase shift excluded.
//	spheroidal_asympt_exp	----	the exponential part of Rb,Rb' or Rc,Rc' with coulomb phase shift
//					excluded or included.
//	spheroidal_asympt	----	calculate R,R' or Rb,Rb' or Rc,Rc'. only valid near +inf.
//
//	spheroidal_asympt_sh1	----	compare the asymptotic expansion to a genuine solution to obtain
//					the coulomb phase shift and the normalization constant. Rx(L),
//					Rx'(L) should be provided.
//	spheroidal_asympt_sh2	---- 	compare the asymptotic expansion to a genuine solution to obtain
//					the coulomb phase shift and the normalization constant. Rx(L1),
//					Rx(L2) should be provided.
//
//	spheroidal_asympt_work	----	allocate a workspace for pade approximant.
//
//	spheroidal_direct	----	calculate R,R, or Rb,Rb' or Rc,Rc' by direct numerical integral. 
//					the initial values are set from taylor expansion near 1.
//					the normalization is done by asymptotic expansion.
//					(desiring L given by a single value as input.)
//
//	spheroidal_direct_arr	----	calculate R,R, or Rb,Rb' or Rc,Rc' by direct numerical integral.
//					the initial values are set from taylor expansion near 1.
//					the normalization is done by asymptotic expansion.
//					(desiring L given in an array as an input. with increasing order)
//
//	spheroidal_chaexp_hd	----	solve the characteristic exponent nu by Hill determiant method.
//
//	spheroidal_chaexp_nr	----	solve the characteristic exponent nu by Newton-Raphson method with
//					the CF evaluated by Lentz-Thompson algorithm.
//
//	
//(4) solving eigenvalue problem.
//
//[dep]	spheroidal_eigfun_angu0	----	for negative energy, evaluate the angular continued fraction f,
//					pf/pA,pf/pK as a function of A,K by Lentz method.
//				----	for positive energy, evaluate the angular continued fraction f,
//					pf/pA as a function of A,K by Lentz method. 
//[dep]	spheroidal_eigfun_angu1	----	for negative energy, evaluate the angular continued fraction pf/pR
//					as a function of A,K by Lentz method.
//
//[dep]	spheroidal_eigfun_radi0	----	for negative energy, evaluate the radial continued fraction g,
//					pg/pA,pg/pK as a function of A,K by Lentz method.
//
//[dep]	spheroidal_eigfun_radi1	----	for negative energy, evaluate the radial continued fraction pg/pR
//					as a function of A,K by Lentz method.
//					
//[dep]	spheroidal_eigval	----	solve the eigen values by Newton method (from given initial values) [for discrete states].
//
//	spheroidal_eigval_mat	----	solve the eigen values by matrix diagonalization techniques [for continuum].
//
//Updated by: Zhao-Han Zhang(张兆涵)  Dec. 15th, 2022
//Copyright © 2022 Zhao-Han Zhang
//=======================================================================================================

namespace qpc
{
	static constexpr size_t	spheroidal_taylor_imax	=	1000ul;

	static constexpr size_t spheroidal_asympt_imax	=	100ul;//number of terms to be summed up, do not be too large
	static constexpr size_t spheroidal_asympt_jmax	=	30ul; //number of transforms in epsilon algorithm, must < 0.5 imax
	static constexpr size_t spheroidal_asympt_pade	=	spheroidal_asympt_imax*6ul+4ul;

	static constexpr size_t spheroidal_eigfun_imax	=	10000ul;
	static constexpr size_t spheroidal_eigval_imax	=	500ul;	

	static constexpr long 	spheroidal_chaexp_imax	=	50000;	//size of hd evaluation
	static constexpr size_t	spheroidal_chaexp_imax_c=	1000ul;	//number of steps in cf evaluation
	static constexpr size_t	spheroidal_chaexp_imax_n=	250ul;	//number of steps in nr iterations

	template<class data_t>
	static constexpr data_t	spheroidal_taylor_rtol	=	std::numeric_limits<data_t>::epsilon()
							*	std::numeric_limits<data_t>::epsilon()*data_t(64);
	template<class data_t>
	static constexpr data_t spheroidal_asympt_rtol	=	std::numeric_limits<data_t>::epsilon()
							*	std::numeric_limits<data_t>::epsilon()*data_t(64);
	template<class data_t>
	static constexpr data_t spheroidal_eigfun_rtol	=	std::numeric_limits<data_t>::epsilon()*data_t(8);
	template<class data_t>
	static constexpr data_t spheroidal_eigval_rtol	=	std::numeric_limits<data_t>::epsilon()*data_t(8);

	template<class data_t>
	static constexpr data_t	spheroidal_chaexp_rtol_c=	std::numeric_limits<data_t>::epsilon()*data_t(8);
	template<class data_t>
	static constexpr data_t	spheroidal_chaexp_rtol_n=	std::numeric_limits<data_t>::epsilon()*data_t(8);
	//=======================================================================================================
	template<class type_t>
	struct	spheroidal_radi_odefun
	{
		using 	data_t	=	type_t;
		using	cord_t	=	std::array<data_t,2>;

		//choose Type 2 or 3 by setting m or -m
		data_t	M;
		data_t	K2;
		data_t	Dp;
		data_t	A;

		//-------------------------------------------
		//(x*x-1)y'' + 2(x+m)y' + (k*k*x*x+D*x-A)y=0
		//-------------------------------------------
		inline	void	operator()	(const data_t& x,const cord_t& y,cord_t& dy)const noexcept		
		{
			data_t	tmp=	x*x-data_t(1);
			dy[0]	=	y[1];
			dy[1]	=-	data_t(2)*(x+M)/tmp*y[1]
				 -	(K2*x*x+Dp*x-A)/tmp*y[0];
		}
	};//end of spheroidal_radi_odefun
	//=======================================================================================================
	template<class data_t>
	inline	auto	spheroidal_asympt_work()
	{
		return 	std::vector<std::complex<data_t>>(spheroidal_asympt_pade);
	}	
	//=======================================================================================================
	template<class data_t>
	inline	int	spheroidal_taylor_phi//Somov 1976
	(
		const data_t& M,	//m
		const data_t& A,	//A-kappa^2 (l,m,eta) !!!XXX causion XXX!!!
		const data_t& K,	//kappa (l,m,eta)
		const data_t& E,	//eta
		const data_t& L,	//lambda
		data_t&	phi0,		//returned as Ra
		data_t&	phi1		//returned as Ra'
	)noexcept
	{
		//parameters
		data_t	X	=	L-data_t(1);
		data_t	D	=	-data_t(2)*E*K;
		data_t	C	=	K*K;
		data_t	R	=	data_t(2)*C+D;
		//i-dependent functions
               	auto    P       =       [&](const size_t i){return data_t(2*(i+1))*(data_t(i+1)+M);};
              	auto    Q       =       [&](const size_t i){return (data_t(i)+M)*(data_t(i+1)+M)+D-A;};
		//taylor expansion for Ra,Ra', where R=(L^2-1)^(M/2)*Ra
		data_t	f0	=	data_t(1);
		data_t	f1	=	-(Q(0)*f0)/P(0);
		data_t	f2	=	-(Q(1)*f1+R*f0)/P(1);
		data_t	f3,tmp;
		data_t	pow	=	X*X;
			phi0	=	f0+f1*X+f2*X*X;	
			phi1	=	f1+f2*X*data_t(2);
		size_t	i;
		for(i=2;i<spheroidal_taylor_imax;++i)
		{
			f3	=	-(Q(i)*f2+R*f1+C*f0)/P(i);
			phi1	+=	pow*f3*data_t(i+1);
			pow	*=	X;
			tmp	=	pow*f3;
			phi0	+=	tmp;
			if(norm(tmp/phi0)<spheroidal_taylor_rtol<data_t>)break;
			f0	=	f1;
			f1	=	f2;
			f2	=	f3;	
		}
		if(i<spheroidal_taylor_imax)	return	0;
		else				return	1;//does not converge
	}//end of spheroidal_taylor_phi

	template<class data_t,int mode=0>
	inline 	int spheroidal_taylor
	(
		const data_t& M,	//m
		const data_t& A,	//A(l,m,eta)
		const data_t& K,	//kappa(l,m,eta)
		const data_t& E,	//eta
		const data_t& L,	//lambda
		data_t&	R0,		//returned the type0,1,2,3 R (L;M,A,E,K), specified by mode
		data_t&	R1		//returned the type0,1,2,3 R'(L;M,A,E,K), specified by mode
	)noexcept
	{
		int info=spheroidal_taylor_phi(M,A-K*K,K,E,L,R0,R1);
		if constexpr(mode==0)//convert phi,phi' to R,R'
		{
			data_t 	tmp	=	L>data_t(1)?(L*L-data_t(1)):(1e-50);
			data_t	pow	=	std::pow(tmp,M/data_t(2));
			R1	=	R1*pow+R0*pow*M*L/tmp;
			R0	=	R0*pow;
		}else 
		if constexpr(mode==1)//return Ra,Ra'
		{
			//just do nothing
		}else
		if constexpr(mode==2)//convert phi,phi' to Rb,Rb'
		{
			data_t	tmp	=	L+data_t(1);
			data_t	pow	=	std::pow(tmp,M);
			R1	=	R1*pow+R0*pow*M/tmp;
			R0	=	R0*pow;
		}else
		if constexpr(mode==3)//convert phi,phi' to Rc,Rc'
		{
			data_t	tmp	=	L>data_t(1)?(L-data_t(1)):(1e-50);
			data_t	pow	=	std::pow(tmp,M);
			R1	=	R1*pow+R0*pow*M/tmp;
			R0	=	R0*pow;	
		}
		return 	info;
	}//end of spheroidal_taylor

	//=======================================================================================================
	template<class data_t>
	inline 	auto spheroidal_asympt_exp
	(
		const data_t& X,
		const data_t& E
	)noexcept
	{
		return 	std::exp(im<data_t>*(X-E*std::log(data_t(2)*X)));
	}//end of spheroidal_asympt_exp
	template<class data_t>
	inline 	auto spheroidal_asympt_exp
	(
		const data_t& X,
		const data_t& E,
		const data_t& S	//if theta is known from other method. note when R->0, theta->sigma-l*pi/2
	)noexcept
	{
		return	std::exp(im<data_t>*(X-E*std::log(data_t(2)*X)+S));
	}//end of spheroidal_asympt_exp

	template<class data_t,int to_exclude_x=0>
	inline	int	spheroidal_asympt_phi//Rankin 1979, Hadinger 1996
	(
		const data_t& M,	//m
		const data_t& A,	//A(l,m,eta)
		const data_t& K,	//kappa(l,m,eta)
		const data_t& E,	//eta
		const data_t& X,	//kappa(l,m,eta)*(lambda+1)
		std::complex<data_t>& phi0,	//returned as phi  where R= (phi/X)*[(L-1)/(L+1)]^(M/2)
		std::complex<data_t>& phi1 	//returned as phi'
	)noexcept
	{
		using	comp_t	=	std::complex<data_t>;
		//parameters
		data_t	D	=	-data_t(2)*E*K;
		data_t	KOX	=	K/X;
		comp_t	E2K	=	im<data_t>*(E+K*data_t(2));
		comp_t	TMP	=	E*E-K*K-D+A-im<data_t>*M*K*data_t(2);
		comp_t	TMQ	=	E*E-im<data_t>*M*E;
		comp_t	TMR	=	M+im<data_t>*data_t(2)*E;
		//asymptotic expansion for phi,phi' where R= (phi/X)*[(L-1)/(L+1)]^(M/2); 
		comp_t	work[8];
		comp_t*	f0	=	work;
		comp_t*	f1	=	work+1;
		comp_t*	f2	=	work+2;
		comp_t*	f3	=	work+3;
		comp_t*	g0	=	work+4;
		comp_t*	g1	=	work+5;
		comp_t*	g2	=	work+6;
		comp_t*	g3	=	work+7;
			*f0	=	data_t(0);
			*f1	=	data_t(0);
			*f2	=	data_t(1);
			*g0	=	data_t(0);
			*g1	=	data_t(0);
			*g2	=	K*(im<data_t>*(data_t(1)-E/X)-data_t(1)/X);
		size_t	k;
		for(k=0;k<spheroidal_asympt_imax;++k)
		{
			comp_t a=	im<data_t>*data_t(2*k+2);
			comp_t b=	-data_t(k*(k+1))+TMP-E2K*data_t(2*k+1);
			comp_t c=	data_t(2)*K*(data_t(k*k)+data_t(k)*TMR-TMQ);
			comp_t tmp1=	b/a/X;
			comp_t tmp2=	c/a/(X*X);
			comp_t tmp3=	tmp1-tmp2;
			comp_t tmp4=	data_t(1)-tmp1;
			*f3	=	tmp4**f2+tmp3**f1+tmp2**f0;
			*g3	=	tmp4**g2+tmp3**g1+tmp2**g0
				+	KOX*tmp1*(*f2-*f1)+data_t(2)*KOX*tmp2*(*f1-*f0);
			if(norm(data_t(1)-*f3/ *f2)<spheroidal_asympt_rtol<data_t>
			&& norm(data_t(1)-*f2/ *f1)<spheroidal_asympt_rtol<data_t>)break;
			std::swap(f0,f1);std::swap(f1,f2);std::swap(f2,f3);
			std::swap(g0,g1);std::swap(g1,g2);std::swap(g2,g3);
		}
		comp_t	expx	=	spheroidal_asympt_exp<data_t>(X,E);//coulomb phase shift not included 
		if constexpr(to_exclude_x==0)
		{	
			expx	/=	X;
			phi0	=	*f3*expx;
			phi1	=	*g3*expx;
		}else
		{
			phi0	=	*f3*expx;
			phi1	=	*g3*expx+phi0/X;
		}
		if(k<spheroidal_asympt_imax)	return 0;
		else				return 1;//does not converge
	}//end of spheroidal_asympt_phi
	
	template<class data_t,int to_exclude_x=0>
	inline	int	spheroidal_asympt_phi//Rankin 1979, Hadinger 1996
	(
		const data_t& M,	//m
		const data_t& A,	//A(l,m,eta)
		const data_t& K,	//kappa(l,m,eta)
		const data_t& E,	//eta
		const data_t& X,	//kappa(l,m,eta)*(lambda+1)
		std::complex<data_t>& phi0,	//returned as phi  or (phi/X)  where R= (phi/X)*[(L-1)/(L+1)]^(M/2)
		std::complex<data_t>& phi1,	//returned as phi' or (phi/X)'
		std::complex<data_t>* workf,	//work space for pade approximant, if necessary. at least of length imax*3+2
		std::complex<data_t>* workg	//work space for pade approximant, if necessary. at least of length imax*3+2
	)noexcept
	{
		using	comp_t	=	std::complex<data_t>;
		//parameters
		data_t	D	=	-data_t(2)*E*K;
		data_t	KOX	=	K/X;
		comp_t	E2K	=	im<data_t>*(E+K*data_t(2));
		comp_t	TMP	=	E*E-K*K-D+A-im<data_t>*M*K*data_t(2);
		comp_t	TMQ	=	E*E-im<data_t>*M*E;
		comp_t	TMR	=	M+im<data_t>*data_t(2)*E;
		//asymtotic expansion for phi,phi' where R= (phi/X)*[(L-1)/(L+1)]^(M/2); 
		comp_t*	f0	=	workf;
		comp_t*	f1	=	workf+1;
		comp_t*	f2	=	workf+2;
		comp_t*	f3	=	workf+3;
		comp_t*	g0	=	workg;
		comp_t*	g1	=	workg+1;
		comp_t*	g2	=	workg+2;
		comp_t*	g3	=	workg+3;
			*f0	=	data_t(0);
			*f1	=	data_t(0);
			*f2	=	data_t(1);
			*g0	=	data_t(0);
			*g1	=	data_t(0);
			*g2	=	K*(im<data_t>*(data_t(1)-E/X)-data_t(1)/X);
		size_t	k;
		for(k=0;k<spheroidal_asympt_imax;++k)
		{
			comp_t a=	im<data_t>*data_t(2*k+2);
			comp_t b=	-data_t(k*(k+1))+TMP-E2K*data_t(2*k+1);
			comp_t c=	data_t(2)*K*(data_t(k*k)+data_t(k)*TMR-TMQ);
			comp_t tmp1=	b/a/X;
			comp_t tmp2=	c/a/(X*X);
			comp_t tmp3=	tmp1-tmp2;
			comp_t tmp4=	data_t(1)-tmp1;
			*f3	=	tmp4**f2+tmp3**f1+tmp2**f0;
			*g3	=	tmp4**g2+tmp3**g1+tmp2**g0
				+	KOX*tmp1*(*f2-*f1)+data_t(2)*KOX*tmp2*(*f1-*f0);
			if(norm(data_t(1)-*f3/ *f2)<spheroidal_asympt_rtol<data_t>
			&& norm(data_t(1)-*f2/ *f1)<spheroidal_asympt_rtol<data_t>)break;
			++f0,++f1,++f2,++f3,++g0,++g1,++g2,++g3;
		}
		comp_t	expx	=	spheroidal_asympt_exp<data_t>(X,E);//coulomb phase shift not included 
		if(k<spheroidal_asympt_imax)	
		{
			if constexpr(to_exclude_x==0)
			{
				expx/=X;
				phi0=*f3*expx;
				phi1=*g3*expx;
			}else
			{
				phi0=*f3*expx;
				phi1=*g3*expx+phi0/X;
			}
		}else
		{
			pade_approximant2<comp_t>(workf+2,phi0,spheroidal_asympt_imax,spheroidal_asympt_jmax);
			pade_approximant2<comp_t>(workg+2,phi1,spheroidal_asympt_imax,spheroidal_asympt_jmax);
			if constexpr(to_exclude_x==0)
			{
				expx/=X;
				phi0*=expx;
				phi1*=expx;	
			}else
			{
				phi0*=expx;
				phi1*=expx;
				phi1+=phi0/X;
			}
		}return 0;
	}//end of spheroidal_asympt_phi

	template<class data_t,int mode=0>
	inline int spheroidal_asympt//Rankin 1979, Hadinger 1996
	(
		const data_t& M,	//m
		const data_t& A,	//A(l,m,eta)
		const data_t& K,	//kappa(l,m,eta)
		const data_t& E,	//eta
		const data_t& L,	//lambda
		std::complex<data_t>& R0,	//returned as type0,2,3  R (L;M,A,E,K)
		std::complex<data_t>& R1,	//returned as type0,2,3  R'(L;M,A,E,K)
		std::complex<data_t>* work=0	//be of length spheroidal_asympt_pade
	)noexcept
	{
		int info;
		data_t	X	=	K*(L+data_t(1));
		if constexpr(mode==0)
		{
			if(work)
			info	=	spheroidal_asympt_phi(M,A,K,E,X,R0,R1,work,work+spheroidal_asympt_imax*3ul+2ul);
			else
			info	=	spheroidal_asympt_phi(M,A,K,E,X,R0,R1);
			data_t tmp=	(L-data_t(1))/(L+data_t(1));
			data_t tmq=	M*L/(L*L-data_t(1));
			data_t pow=	std::pow(tmp,M/data_t(2));
			R1	=	R1*pow+R0*tmq*pow;
			R0	=	R0*pow;
		}else
		if constexpr(mode==2)
		{
			if(work)
			info	=	spheroidal_asympt_phi(M,A,K,E,X,R0,R1,work,work+spheroidal_asympt_imax*3ul+2ul);
			else
			info    =       spheroidal_asympt_phi(M,A,K,E,X,R0,R1);
		}else 
		if constexpr(mode==3)
		{
			if(work)
			info	=	spheroidal_asympt_phi(-M,A,K,E,X,R0,R1,work,work+spheroidal_asympt_imax*3ul+2ul);
			else
			info    =       spheroidal_asympt_phi(-M,A,K,E,X,R0,R1);
		}
		return 	info;
	}//end of spheroidal_asympt

	template<class data_t,int mode=0>
	inline int spheroidal_asympt_sh1//Rankin 1979, Hadinger 1996
	(
		const data_t& M,        //m
                const data_t& A,        //A(l,m,eta)
                const data_t& K,        //kappa(l,m,eta)
                const data_t& E,        //eta
                const data_t& L,        //lambda
                const data_t& R0,       //genuine solution of type0,2,3  R (L;M,A,E,K)
                const data_t& R1,       //genuine solution of type0,2,3  R'(L;M,A,E,K)
		data_t&	sh,		//returned as phase shift
		data_t&	nm,		//returned as normalization constant
                std::complex<data_t>* work=0
	)noexcept
	{
		int info;
		std::complex<data_t> H0,H1;
		info	=	spheroidal_asympt<data_t,mode>(M,A,K,E,L,H0,H1,work);	
		sh	=	atan(-(R1*imag(H0)-R0*imag(H1))/(R1*real(H0)-R0*real(H1)));
		nm	=	imag(exp(im<data_t>*sh)*H0)/R0;//normalize to 1
		return 	info;
	}//end of spheroidal_asympt_sh1

	template<class data_t,int mode=0>
	inline int spheroidal_asympt_sh1//Rankin 1979, Hadinger 1996
	(
		const data_t& M,        //m
                const data_t& A,        //A(l,m,eta)
                const data_t& K,        //kappa(l,m,eta)
                const data_t& E,        //eta
                const data_t& L,        //lambda
                const data_t& R0,       //genuine solution of type0,2,3  R (L;M,A,E,K)
                const data_t& R1,       //genuine solution of type0,2,3  R'(L;M,A,E,K)
		data_t&	sh,		//returned as phase shift
		data_t&	F0,		//returned as regular solution F
		data_t&	F1,		//returned as regular solution F'
		data_t&	G0,		//returned as irregular solution G
		data_t&	G1,		//returned as irregular solution G'
                std::complex<data_t>* work=0
	)noexcept
	{
		int info;
		std::complex<data_t> H0,H1,expish;
		info	=	spheroidal_asympt<data_t,mode>(M,A,K,E,L,H0,H1,work);	
		sh	=	atan(-(R1*imag(H0)-R0*imag(H1))/(R1*real(H0)-R0*real(H1)));
		expish	=	exp(im<data_t>*sh);
		F0	=	real(expish)*imag(H0)+imag(expish)*real(H0);//already normalized as ~ exp/x
		F1	=	real(expish)*imag(H1)+imag(expish)*real(H1);//already normalized as ~ exp/x
		G0	=	real(expish)*real(H0)-imag(expish)*imag(H0);//already normalized as ~ exp/x
		G1	=	real(expish)*real(H1)-imag(expish)*imag(H1);//already normalized as ~ exp/x
		return 	info;
	}//end of spheroidal_asympt_sh1
		
	template<class data_t,int mode=0>
	inline int spheroidal_asympt_sh2//Rankin 1979, Hadinger 1996
	(
		const data_t& M,        //m
                const data_t& A,        //A(l,m,eta)
                const data_t& K,        //kappa(l,m,eta)
                const data_t& E,        //eta
                const data_t& Li,       //lambda1
		const data_t& Lj,	//lambda2
                const data_t& Ri,       //genuine solution of type0,2,3  R(Li;M,A,E,K)
                const data_t& Rj,       //genuine solution of type0,2,3  R(Lj;M,A,E,K)
		data_t&	sh,		//returned as phase shift
		data_t&	nm,		//returned as normalization constant
                std::complex<data_t>* work=0
	)noexcept
	{
		int info;
		std::complex<data_t> Hi,Hj,H1;
		info	=	spheroidal_asympt<data_t,mode>(M,A,K,E,Li,Hi,H1,work);	
		info	+=	spheroidal_asympt<data_t,mode>(M,A,K,E,Lj,Hj,H1,work);
		sh	=	atan(-(Rj*imag(Hi)-Ri*imag(Hj))/(Rj*real(Hi)-Ri*real(Hj)));
		nm	=	imag(exp(im<data_t>*sh)*Hi)/Ri;//normalize to 1 by nm*Ri
		return 	info;
	}//end of spheroidal_asympt_sh2	

	template<class data_t,int mode=2>
	inline int spheroidal_direct
	(
		const data_t& M,	//m
		const data_t& A,	//A
		const data_t& K,	//K
		const data_t& D,	//Dp=(z1+z2)*rn
		const data_t& L,	//lamdba
		const data_t& tol,	//tolerence
		data_t& sh,		//returned as coulomb phase shift
		data_t&	R0,		//returned as R or Rb  or Rc
		data_t&	R1,		//returned as R or Rb' or Rc'
		std::complex<data_t>* work=0
	)noexcept
	{
		static_assert(mode==2||mode==3);
		const data_t E	=	-D/K/data_t(2);
		if(L<=1.001)return spheroidal_taylor<data_t,mode>(M,A,K,E,L,R0,R1);
		auto	odefun	=	spheroidal_radi_odefun<data_t>();
		odefun.M	=	mode==2?M:-M;
		odefun.K2	=	K*K;
		odefun.Dp	=	D;
		odefun.A	=	A;
		auto	worksp	=	richardson<decltype(odefun),20>{odefun};//richardson workspace
		worksp.tol[0]	=	tol;
		worksp.tol[1]	=	tol;
		worksp.xf	=	data_t(1.001);
		const data_t S	=	PI<data_t>/data_t(4)/K;
		int info	=	spheroidal_taylor<data_t,mode>(M,A,K,E,worksp.xf,worksp.yf[0],worksp.yf[1]);
		while(worksp.xf+S<L)
		{
			worksp.xi=	worksp.xf;
			worksp.xf+=	S;
			worksp.yi=	worksp.yf;
			info	+=	richardson_ex(worksp);
		}	
		worksp.xi	=	worksp.xf;
		worksp.xf	=	L;
		worksp.yi	=	worksp.yf;
		info		+=	richardson_ex(worksp);
		data_t	nm;
		info		-=	spheroidal_asympt_sh1<data_t,mode>(M,A,K,E,worksp.xf,worksp.yf[0],worksp.yf[1],sh,nm,work)*100000;
		R0		=	worksp.yf[0]*nm;
		R1		=	worksp.yf[1]*nm;
		return 	info;
	}//end of spheroidal_direct

	template<class data_t,int mode=2>
	inline int spheroidal_direct_all
	(
		const data_t& M,	//m
		const data_t& A,	//A
		const data_t& K,	//K
		const data_t& D,	//Dp=(z1+z2)*rn
		const data_t& L,	//lamdba, must be in asymptotic region!
		const data_t& tol,	//tolerence
		data_t& sh,		//returned as coulomb phase shift
		data_t&	F0,
		data_t&	F1,
		data_t&	G0,
		data_t&	G1,
		std::complex<data_t>* work=0
	)noexcept
	{
		static_assert(mode==2||mode==3);
		const data_t E	=	-D/K/data_t(2);
		auto	odefun	=	spheroidal_radi_odefun<data_t>();
		odefun.M	=	mode==2?M:-M;
		odefun.K2	=	K*K;
		odefun.Dp	=	D;
		odefun.A	=	A;
		auto	worksp	=	richardson<decltype(odefun),20>{odefun};//richardson workspace
		worksp.tol[0]	=	tol;
		worksp.tol[1]	=	tol;
		worksp.xf	=	data_t(1.001);
		const data_t S	=	PI<data_t>/data_t(4)/K;
		int info	=	spheroidal_taylor<data_t,mode>(M,A,K,E,worksp.xf,worksp.yf[0],worksp.yf[1]);
		while(worksp.xf+S<L)
		{
			worksp.xi=	worksp.xf;
			worksp.xf+=	S;
			worksp.yi=	worksp.yf;
			info	+=	richardson_ex(worksp);
		}	
		worksp.xi	=	worksp.xf;
		worksp.xf	=	L;
		worksp.yi	=	worksp.yf;
		info		+=	richardson_ex(worksp);
		info		-=	spheroidal_asympt_sh1<data_t,mode>(M,A,K,E,worksp.xf,worksp.yf[0],worksp.yf[1],sh,F0,F1,G0,G1,work)*100000;
		return 	info;
	}//end of spheroidal_direct

	template<class data_t,int mode=2>
	inline int spheroidal_direct_arr
	(
		const data_t& M,	//m
		const data_t& A,	//A
		const data_t& K,	//K
		const data_t& D,	//D=(Z1+Z2)*rn
		const data_t* L,	//lamdba (array)
		const data_t& tol,	//tolerence
		data_t& sh,		//returned as coulomb phase shift
		data_t*	R0,		//returned as array of R  or Rb  or Rc
		data_t*	R1,		//returned as array of R' or Rb' or Rc'
		const size_t nL,	//size of L,R0,R1
		std::complex<data_t>* work=0//for asympt
	)noexcept
	{
		int 	info	=	0;
		size_t 	i	=	0;
		static_assert(mode==2||mode==3);
		data_t const E	=	-D/K/data_t(2);
		data_t const S	=	PI<double>/data_t(4)/K;
		while(L[i]<1.001&&i<nL)
		{
			info	+=	spheroidal_taylor<data_t,mode>(M,A,K,E,L[i],R0[i],R1[i]);
			++i;
		}
		auto	odefun	=	spheroidal_radi_odefun<data_t>();
		odefun.M	=	mode==2?M:-M;
		odefun.K2	=	K*K;
		odefun.Dp	=	D;
		odefun.A	=	A;
		auto	worksp	=	richardson<decltype(odefun),16>{odefun};//richardson workspace
		worksp.tol[0]	=	tol;
		worksp.tol[1]	=	tol;
		worksp.xf	=	data_t(1.001);
		info		+=	spheroidal_taylor<data_t,mode>(M,A,K,E,worksp.xf,worksp.yf[0],worksp.yf[1]);
		while(i<nL)
		{
			worksp.xi	=	worksp.xf;
			worksp.xf	=	L[i];
        	        worksp.yi	=      	worksp.yf;
                	info    	+=      richardson_ex(worksp);
			R0[i]		=	worksp.yf[0];
			R1[i]		=	worksp.yf[1];
			++i;
		}
		data_t	x_asy,y0_asy,y1_asy;
		data_t	xc_asy		=	std::max(data_t(3.0)*(sqrt(fabs(A))+E*E),data_t(50));
		if(L[nL-1]>xc_asy)
		{
			x_asy		=	L[nL-1];
			y0_asy		=	R0[nL-1];
			y1_asy		=	R1[nL-1];
		}else
		{
			while(worksp.xf+S<xc_asy)
			{
				worksp.xi=	worksp.xf;
				worksp.xf+=	S;
				worksp.yi=	worksp.yf;
				info	+=	richardson_ex(worksp);
			}	
				worksp.xi=	worksp.xf;
				worksp.xf=	xc_asy;
				worksp.yi=	worksp.yf;
				info	+=	richardson_ex(worksp);
			x_asy		=	xc_asy;
			y0_asy		=	worksp.yf[0];
			y1_asy		=	worksp.yf[1];
		}
		data_t	nm;
		if(spheroidal_asympt_sh1<data_t,mode>(M,A,K,E,x_asy,y0_asy,y1_asy,sh,nm,work))
		{
			info		=	std::numeric_limits<int>::infinity();
		}
		for(size_t ii=0;ii<nL;++ii)
		{
			R0[ii]		*=	nm;
			R1[ii]		*=	nm;
		}
		return 	info;	
	}//end of spheroidal_direct_arr
	
	//=======================================================================================================
	template<class data_t>
	inline int spheroidal_eigfun_angu0//this routine is designed for energy<0
	(
		const data_t& M,	//m
		const data_t& A,	//A(l,m,eta)
		const data_t& K,	//-I*kappa(l,m,eta)
		const data_t& E,	//-I*eta
		data_t& F0,		//returned as the continued fraction f
		data_t& FA,		//returned as pf/pA
		data_t& FK,		//returned as pf/pK
		const data_t& accr
	)noexcept
	{
		data_t const K2	=	-K*K;
		data_t const E2	=	-E*E;
		data_t const M2	=	M*M;
		data_t const KA	=	K2-A;
		auto	func	=	[&](const size_t i,data_t& aO,data_t& aA,data_t& aK,data_t& bO,data_t& bA,data_t& bK)
		{
			data_t l=	M+data_t(i);
			data_t q=	(l*l-M2)/(data_t(4)*l*l-data_t(1));
			aO	=	-data_t(4)*K2*(l*l+E2)*q;
			aA	=	data_t(0);
			aK	=	data_t(8)*K*l*l*q;
			bO	=	KA+l*(l+data_t(1));
			bA	=	-data_t(1);
			bK	=	-data_t(2)*K;
		};
		int info	=	cf_lentz_real<data_t>(func,accr,spheroidal_eigfun_imax,F0,FA,FK);
		return 	info>0?100:0;
	}//end of spheroidal_eigfun_angu0

	template<class data_t>
	inline int spheroidal_eigfun_angu1//this routine is designed for energy<0
	(
		const data_t& M,	//m
		const data_t& A,	//A(l,m,eta)
		const data_t& K,	//-I*kappa(l,m,eta)
		const data_t& E,	//-I*eta
		const data_t& r,	//parameter rn
		data_t& F0,		//returned as the continued fraction f
		data_t& FA,		//returned as pf/pA
		data_t& FK,		//returned as pf/pK
		data_t&	Fr,		//returned as pf/pr
		const data_t& accr
	)noexcept
	{
		data_t const K2	=	-K*K;
		data_t const E2	=	-E*E;
		data_t const M2	=	M*M;
		data_t const KA	=	K2-A;
		auto	func	=	[&](const size_t i,data_t& aO,data_t& aA,data_t& aK,data_t& ar,data_t& bO,data_t& bA,data_t& bK,data_t& br)
		{
			data_t l=	M+data_t(i);
			data_t q=	(l*l-M2)/(data_t(4)*l*l-data_t(1));
			aO	=	-data_t(4)*K2*(l*l+E2)*q;
			aA	=	data_t(0);
			aK	=	data_t(8)*K*l*l*q;
			ar	=	-data_t(8)*K2*E2*q/r;
			bO	=	KA+l*(l+data_t(1));
			bA	=	-data_t(1);
			bK	=	-data_t(2)*K;
			br	=	data_t(0);
		};
		int info	=	cf_lentz_real<data_t>(func,accr,spheroidal_eigfun_imax,F0,FA,FK,Fr);
		return 	info>0?300:0;
		return 	info;
	}//end of spheroidal_eigfun_angu1

	template<class data_t>
	inline int spheroidal_eigfun_radi0//this routine is designed for energy<0
	(
		const data_t& M,	//m
		const data_t& A,	//A(l,m,eta)
		const data_t& K,	//-I*kappa(l,m,eta)
		const data_t& E,	//-I*eta
		data_t& F0,		//returned as the continued fraction f
		data_t& FA,		//returned as pf/pA
		data_t& FK,		//returned as pf/pK
		const data_t& accr
	)noexcept
	{
		data_t const EoK=	E/K;
		data_t const EsK=	E-data_t(2)*K;
		data_t const KDA=	-K*K+data_t(2)*E*K-A;
		auto	func	=	[&](const size_t i,data_t& aO,data_t& aA,data_t& aK,data_t& bO,data_t& bA,data_t& bK)
		{
			data_t n=	data_t(i);
			data_t nE=	n-E;
			data_t nM=	n*(n+M);
			data_t n2=	n*data_t(2)+data_t(1);
			aO	=	-nM*(          nE+M)*nE;
			aA	=	data_t(0);
			aK	=	-nM*(data_t(2)*nE+M)*EoK;
			bO	=	KDA-data_t(2)*n*n-(M+data_t(1))*n2+EsK*(n2+M);
			bA	=	-data_t(1);
			bK	=	-data_t(2)*K-(EoK+data_t(2))*(n2+M);
		};	
		int info	=	cf_lentz_real<data_t>(func,accr,spheroidal_eigfun_imax,F0,FA,FK);
		return	info>0?200:0;
	}//end of spheroidal_eigfun_radi0

	template<class data_t>
	inline int spheroidal_eigfun_radi1//this routine is designed for energy<0
	(
		const data_t& M,	//m
		const data_t& A,	//A(l,m,eta)
		const data_t& K,	//-I*kappa(l,m,eta)
		const data_t& E,	//-I*eta
		const data_t& r,	//parameter rn
		data_t& F0,		//returned as the continued fraction f
		data_t& FA,		//returned as pf/pA
		data_t& FK,		//returned as pf/pK
		data_t& Fr,		//returned as pf/pr
		const data_t& accr
	)noexcept
	{
		data_t const EoK=	E/K;
		data_t const Eor=	E/r;
		data_t const Dor=	E*K*data_t(2)/r;
		data_t const EsK=	E-data_t(2)*K;
		data_t const KDA=	-K*K+data_t(2)*E*K-A;
		auto	func	=	[&](const size_t i,data_t& aO,data_t& aA,data_t& aK,data_t& ar,data_t& bO,data_t& bA,data_t& bK,data_t& br)
		{
			data_t n=	data_t(i);
			data_t nE=	n-E;
			data_t nM=	n*(n+M);
			data_t n2=	n*data_t(2)+data_t(1);
			data_t n2M=	n2+M;
			aO	=	-nM*(          nE+M)*nE;
			aA	=	data_t(0);
			aK	=	-nM*(data_t(2)*nE+M)*EoK;
			ar	=	+nM*(data_t(2)*nE+M)*Eor;
			bO	=	KDA-data_t(2)*n*n-(M+data_t(1))*n2+EsK*n2M;
			bA	=	-data_t(1);
			bK	=	-data_t(2)*K-(EoK+data_t(2))*n2M;
			br	=	Dor+n2M*Eor;
		};	
		int info	=	cf_lentz_real<data_t>(func,accr,spheroidal_eigfun_imax,F0,FA,FK,Fr);
		return	info>0?400:0;
	}//end of spheroidal_eigfun_radi1
	
		
	template<class data_t>
	inline int spheroidal_eigval//this routine is designed for energy<0
	(
		const data_t& M,	//m
		const data_t& A,	//A(l,m,eta), 		set by initial guess
		const data_t& K,	//-I*kappa(l,m,eta)	set by initial guess
		const data_t& Ds,	//(Z1-Z2)*rn
		const data_t& Dp,	//(Z1+Z2)*rn
		data_t& retA0,		//returned as convergent A
		data_t& retK0,		//returned as convergent K
		const data_t& accr1=spheroidal_eigval_rtol<data_t>,	//accuracy of nr solver
		const data_t& accr2=spheroidal_eigfun_rtol<data_t>	//accuracy of cf solver
	)noexcept
	{	
		int info=0;
		data_t	_A=A;
		data_t	_K=K;
		//solve _A,_K
		auto	equation=	[&](data_t& fa,data_t& fr,data_t& dfa_da,data_t& dfa_dk,data_t& dfr_da,data_t& dfr_dk)
		{
			info	+=	spheroidal_eigfun_angu0<data_t>(M,_A,_K,Ds/data_t(2)/_K,fa,dfa_da,dfa_dk,accr2);
			info	+=	spheroidal_eigfun_radi0<data_t>(M,_A,_K,Dp/data_t(2)/_K,fr,dfr_da,dfr_dk,accr2);
		};//info only+=100 or 200
		auto	modifier=	[&](const data_t& del_a,const data_t& del_k)
		{
			_A+=del_a;
			_K+=del_k;
			return 	norm(del_a/_A)+norm(del_k/_K);
		};
		info+=sv_newton<data_t>(equation,modifier,accr1,spheroidal_eigfun_imax);
		retA0=_A;retK0=_K;
		return 	info;	
	}//end of spheroidal_eigval, diff0 case

	template<class data_t>
	inline int spheroidal_eigval//this routine is designed for energy<0
	(
		const data_t& M,	//m
		const data_t& A,	//A(l,m,eta), 		set by initial guess
		const data_t& K,	//-I*kappa(l,m,eta)	set by initial guess
		const data_t& Ds,	//(Z1-Z2)*rn
		const data_t& Dp,	//(Z1+Z2)*rn
		const data_t& r,	//rn
		data_t& retA0,		//returned as convergent A
		data_t& retK0,		//returned as convergent K
		data_t& retA1,		//returned as convergent dA/dr
		data_t&	retK1,		//returned as convergent dK/dr
		const data_t& accr1=spheroidal_eigval_rtol<data_t>,	//accuracy of nr solver
		const data_t& accr2=spheroidal_eigfun_rtol<data_t>	//accuracy of cf solver
	)noexcept
	{	
		//solve _A, _K
		int info	=	spheroidal_eigval(M,A,K,Ds,Dp,retA0,retK0,accr1,accr2);
		//solve _A',_K'
		data_t	Fa,Fr,dFa_dA,dFa_dK,dFa_dR,dFr_dA,dFr_dK,dFr_dR;
		info		+=	spheroidal_eigfun_angu1<data_t>(M,retA0,retK0,Ds/data_t(2)/retK0,r,Fa,dFa_dA,dFa_dK,dFa_dR,accr2);
		info		+=	spheroidal_eigfun_radi1<data_t>(M,retA0,retK0,Dp/data_t(2)/retK0,r,Fr,dFr_dA,dFr_dK,dFr_dR,accr2);
		data_t	del1	=	dFr_dR*dFa_dK-dFa_dR*dFr_dK;
		data_t	del2	=	dFa_dR*dFr_dA-dFr_dR*dFa_dA;
		data_t	det 	=	dFa_dA*dFr_dK-dFa_dK*dFr_dA;
		retA1=del1/det;retK1=del2/det;
		return 	info;	
	}//end of spheroidal_eigval, diff1 case

	template<class data_t,size_t nL,bool parseK2=false>
	inline	int spheroidal_eigval_mat//this routine is designed for energy>0
	(
		data_t const& M,	//m, must take integer values
		data_t const& K,	//kappa=k*rn/2
		data_t const& Ds,	//(Z1-Z2)*rn
		data_t* A,		//returned as an array of eigenvalue A (of size nL)
		data_t* C,		//returned as an array of eigenstate C (of size nL*nL, in col-major format), in SPH-NRM
		data_t*	D		//workspace of size nL*3
	)noexcept
	{
		long	_M	=	std::lround(M);
		data_t  K2;
		if constexpr(parseK2)
		{
			K2	=	K;
		}else	K2	=	K*K;
		//set matrix value
		for(long i=0;i<long(nL);++i)
		{
			long L	=	_M+i;
			D[3*i]	=	K2*data_t(2*L*(L+1)-2*_M*_M-1)/data_t((2*L-1)*(2*L+3))+data_t(L*(L+1));
			D[3*i+1]=	Ds*std::sqrt(data_t( (L+1)*(L+1)-_M*_M                     )/data_t((2*L+3)*(2*L+1))                );
			D[3*i+2]=	K2*std::sqrt(data_t(((L+1)*(L+1)-_M*_M)*((L+2)*(L+2)-_M*_M))/data_t((2*L+5)*(2*L+3)*(2*L+3)*(2*L+1)));
		}
		//call lapacke
		int	info	=	impl_ed_simd::ev_dsb<nL,2>(D,A,C);
		return	info;	
	}//end of spheroidal_eigval_mat
	//=======================================================================================================
	
	template<class data_t>
	inline	auto	spheroidal_chaexp_hd		
	(
		const data_t& 	M,	//usual m
		const data_t& 	A,	//a-w^2 !!!XXX!!!
		const data_t&	K,	//usual w (real kappa)
		const data_t& 	E,	//usual eta+
		const long	imax=spheroidal_chaexp_imax
	)noexcept
	{
		using 	comp_t	=	std::complex<data_t>;
		data_t const M2	=	M*M;
		data_t const E2	=	E*E;
		data_t const _4K2=	K*K*data_t(4);
		auto	func	=	[&](long i)
		{
			i	-=	imax;
			data_t N2=	data_t(i*i);
			data_t An=	A-N2;
			return 	_4K2*(N2+E2)*(N2-M2)/((data_t(4)*N2-data_t(1))*(An*An-N2));
		};
		data_t const dH	=	hd_tridiag<data_t>(func,2*imax+1);
		//asin(sqrt(|H|)*cos(pi/2*sqrt(1+4(A-K2))))/pi;
		return	im<data_t>*asinh
		(
			im<data_t>*sqrt(comp_t(dH))*
                        cos(data_t(0.5)*PI<data_t>*sqrt(comp_t(data_t(1)+data_t(4)*A)))
		)/PI<data_t>;
	}//end of spheroidal_chaexp_hd

	template<class data_t>
	inline	int	spheroidal_chaexp_nr
	(
		const data_t&	M,	//usual m
		const data_t&	A,	//!!!a-w2!!!
		const data_t&	K,	//usual w (real kappa)
		const data_t&	E,	//usual eta+
		const size_t	Q,	//the index for a. when R->0, a->(Q+M)(Q+M+1)
		std::complex<data_t>& V,//return value of nu. 
		const data_t&	rtol_cf	=spheroidal_chaexp_rtol_c<data_t>,	//relative tolerence for cf evaluation
		const size_t	imax_cf	=spheroidal_chaexp_imax_c,		//maximum number of steps in cf evaluation
		const data_t&	rtol_nr	=spheroidal_chaexp_rtol_n<data_t>,	//relative tolerence for nr iterations
		const size_t 	imax_nr	=spheroidal_chaexp_imax_n 		//maximum number of steps in nr iterations
	)noexcept
	{
		using	comp_t	=	std::complex<data_t>;
		int		info	=	0;
		//set intial value by hill determinant
				V	=	spheroidal_chaexp_hd<data_t>(M,A,K,E);
		if(norm(imag(V))<data_t(1e-28))//when pure real, it is better to shift v to minimize cancellation. q=l-m must be correctly set.
		{
			if(real(V)<data_t(-0.1))//-0.5 case
			{
                        	V	+=	data_t(1);
                        }       V	+=	data_t(Q)+M;
		}
		//prepare all objects needed by NR solver
		const data_t	E2	=	E*E;
		const data_t	M2	=	M*M;
		const data_t 	K2	=	K*K;
		auto	cf_func_fwd	=	[&](size_t n,auto& a0,auto& a1,auto& b0,auto& b1)
		{
			auto	L	=	V+data_t(n);
			auto	L2	=	L*L;
			auto	T	=	L2*data_t(4)-data_t(1);
				a0	=	data_t( -4)*K2*(L2+E2)*(L2-M2)/T;
				a1	=	data_t(-32)*K2*(L2+E2*M2+(M2-E2)/data_t(4)-L/data_t(2))*L/(T*T);
				b0	=	L2+L-A;
				b1	=	L *data_t(2)+data_t(1);
		};//end of cf_func_fwd
		auto	cf_func_bwd	=	[&](size_t n,auto& a0,auto& a1,auto& b0,auto& b1)
		{
			auto	L	=	V-data_t(n)+data_t(1);
			auto	L2	=	L*L;
			auto	T	=	L2*data_t(4)-data_t(1);
				a0	=	data_t( -4)*K2*(L2+E2)*(L2-M2)/T;
				a1	=	data_t(-32)*K2*(L2+E2*M2+(M2-E2)/data_t(4)-L/data_t(2))*L/(T*T);
				b0	=	L2-L-A;
				b1	=	L *data_t(2)-data_t(1);
		};//end of cf_func_bwd
		auto	equation	=	[&](comp_t& fx0,comp_t& fx1)
		{
			comp_t 	res0_fwd,res1_fwd,res0_bwd,res1_bwd,res0_b0,res1_b0;
                     	     	info   	+=   	cf_lentz_comp<data_t>(cf_func_fwd,rtol_cf,imax_cf,res0_fwd,res1_fwd);
                		info   	+=   	cf_lentz_comp<data_t>(cf_func_bwd,rtol_cf,imax_cf,res0_bwd,res1_bwd);
                        			cf_func_fwd(0,res0_b0,res0_b0,res0_b0,res1_b0);
                                fx0	=       res0_fwd+res0_bwd-res0_b0;//f(v)
                                fx1	=       res1_fwd+res1_bwd-res1_b0;//df(v)/dv
		};//end of equation
		auto	modifier	=	[&](const comp_t& dx)
		{
			if (norm(imag(V))>data_t(1e-28))
                        {//imag case
                                V	+=	imag(dx)*im<data_t>;
                                return  norm(imag(dx)/V);
                        }else
                        {//real case
                                V	+=	dx;
                                return  norm(dx/V);
                        }	
		};//end of modifier
		return		info	+=	sv_newton<data_t>(equation,modifier,rtol_nr,imax_nr);
	}//end of spheroidal_chaexp_nr
}//end of qpc
