#pragma once

#include <algorithm/cf.h>
#include <algorithm/pd.h>

#include <functions/hypergeom.h>
#include <functions/gamma.h>
namespace qpc
{
	//=====================================================================================================
	//					Legendre Function
	//
	//				   See Abramowitz & Stegun 1964 
	//
	//(1) The algorithm evaluates the value of the Legendre Functions of the first and second kind, 
	//i.e. Plm(z), Qlm(z) for real z>1 and integer l,m. The results are returned in two arrays.
	//
	//(2) The algorithm evaluates the value of the Legendre Functions of the first and second kind,
	//i.e. Plm(z), Qlm(z) for real z>1 and complex l,m. The results are returned in two arrays.
	//
	//
	//The list of struct/class and functions:
	//
	//(1)
	//	legendre_recurrence 		----		cf object for Q(n+1)/Q(n) for integer m,l,z
	//	legendre_recurrence_real	----		cf object for Q(n+1)/Q(n) for real m,l,z
	//	legendre_recurrence_comp	----		cf object for Q(n+1)/Q(n) for complex m,l,z
	//
	//(2)
	//	legendre_ratio			----		Q(n+1)/Q(n) for integer m,l,z
	//	legendre_ratio_real		----		Q(n+1)/Q(n) for real m,l,z
	//	legendre_ratio_comp		----		Q(n+1)/Q(n) for complex m,l,z
	//
	//(3)
	//	legendre_qvalue_real		----		Q(v,m,z) for real v,m,z	
	//	legendre_qvalue_comp		----		Q(v,m,z) for complex v,m,z	
	//	legendre_pvalue_real		----		P(v,m,z) for real v,m,z	
	//	legendre_pvalue_comp		----		P(v,m,z) for complex v,m,z
	//
	//(4) 	
	//	legendre_pq			----		an array of P, Q for integer m,l and real z>1
	//	legendre_q			----		an array of    Q for integer m,l and real z>1
	//
	//=====================================================================================================
	
	template<class data_t>
	struct  legendre_recurrence
      	{
		data_t  _n1;//n
		data_t  _n2;//n+2m-1
		data_t  _n3;//2n+2m+1
		data_t  _n4;//n+2m
		data_t  _z;

		inline  void  operator()(size_t i,data_t& a,data_t& b)const noexcept
		{
			a= -(data_t(i)+_n1)/(data_t(i)+_n2);
			b= (data_t(2ul*i)+_n3)/(data_t(i)+_n4)*_z;
		}
	};//end of legendre_recurrence

	template<class data_t>
	struct	legendre_recurrence_real
	{
		data_t	m2;
		data_t	v;
		data_t	z;

		inline	void operator()(size_t i, data_t& a, data_t& b)const noexcept//for Q(m,v+1,z)/Q(m,v,z)
		{
			data_t l=v+data_t(i);
			a= (m2-l*l);
			b=-(data_t(2)*l+data_t(1))*z;
		}
	};//end of legendre_recurrence_real

	template<class data_t>
	struct	legendre_recurrence_comp
	{
		std::complex<data_t> m2;
		std::complex<data_t> v;
		std::complex<data_t> z;
	
		inline  void operator()(size_t i, std::complex<data_t>& a, std::complex<data_t>& b)const noexcept//for Q(m,v+1,z)/Q(m,v,z)
                {
                        std::complex<data_t> l=v+data_t(i);
                        a= (m2-l*l);
                        b=-(data_t(2)*l+data_t(1))*z;
                }
	};//end of legendre_recurrence_comp
	
	//----------------------------------------------------------------------------------------

	static constexpr size_t legendre_ratio_imax	=	4000ul;//maximum iteration number for cf

	template<class data_t>
	static constexpr data_t	legendre_ratio_rtol	=	data_t(16)*std::numeric_limits<data_t>::epsilon();

	//----------------------------------------------------------------------------------------
	template<class data_t>
	static	inline	int	legendre_ratio    	
	(
		const data_t m,
		const data_t n,
		const data_t z, 
		data_t& cf	//returned as Q(z;m,m+n+1)/Q(z;m,m+n)
	)noexcept
	{
		int info=cf_lentz_real
		(
			legendre_recurrence<data_t>
			{
				n,n+data_t(2)*m-data_t(1),data_t(2)*(n+m)+data_t(1),data_t(2)*m+n,z
			},
			legendre_ratio_rtol<data_t>,
			legendre_ratio_imax,cf
		);
		cf	=	data_t(1)/cf;
		return  info;
	}//end of legendre_ratio

	template<class data_t>
	static	inline	int	legendre_ratio_real	
	(
		const data_t m,
		const data_t v,
		const data_t z, 
		data_t& cf	//returned as Q(z;m,v+1)/Q(z;m,v)
	)noexcept
	{
		int info=cf_lentz_real
		(
			legendre_recurrence_real<data_t>{m*m,v,z},
			legendre_ratio_rtol<data_t>,
                        legendre_ratio_imax,cf
		);
		cf	+=	(data_t(2)*v+data_t(1))*z;	
		cf	/=	data_t(1)+v-m;
		return 	info;
	}//end of legendre_ratio_real

	template<class data_t>
	static	inline	int	legendre_ratio_comp	
	(
		const std::complex<data_t> m,
		const std::complex<data_t> v,
		const std::complex<data_t> z, 
		std::complex<data_t>& cf	//returned as Q(z;m,v+1)/Q(z;m,v)
	)noexcept
	{
		int info=cf_lentz_comp
		(
			legendre_recurrence_comp<data_t>{m*m,v,z},
			legendre_ratio_rtol<data_t>,
                        legendre_ratio_imax,cf
		);
		cf	+=	(data_t(2)*v+data_t(1))*z;	
		cf	/=	data_t(1)+v-m;
		return 	info;
	}//end of legendre_ratio_comp

	template<class data_t>
	static	inline	data_t	legendre_dfact          (const data_t m)noexcept
	{
		data_t  y=data_t(1);
        	for(data_t t=data_t(2)*m-data_t(1);t>data_t(0.5);t-=data_t(2))
           	{
               		y*=t;
             	}
          	return  y;
	}//end of legendre_dfact

	template<class data_t>
	static	inline	data_t	legendre_rfact          (const data_t m,const data_t n,     data_t y)noexcept
	{
		data_t	t=data_t(2)*m-data_t(1);
               	if(t<0)return y/n;
            	for(;t>data_t(0.5);t-=data_t(1))
            	{
                	y*=(n+t);
            	}
           	return  lround(m)%2?-y:+y;
	}//end of legendre_rfact
	//----------------------------------------------------------------------------------------
	template<class data_t>
	static	inline	int	legendre_qvalue_real	
	(
		const data_t m,
		const data_t v,
		const data_t z, 
		std::complex<data_t>& q, //returned as Q(z;m,v)
		data_t* w 		//workspace, at least longer than hypergeom_2f1_taylor_wksz
	)noexcept
	{
		int	info;
		data_t	z2r	=	data_t(1)/(z*z);
		data_t	f21,r1,r2,a1,a2,power1,power2;
		//1f2(a,b,c,z)
		info+=hypergeom_2f1<data_t>
		(
			(data_t(2)+v+m)/data_t(2),
			(data_t(1)+v+m)/data_t(2),
			v+data_t(1.5),z2r,f21,w
		);
		//gammaln(v+m+1)
		gammalnc<data_t>(v+m+data_t(1),&r1,&a1);
		//gammaln(v+3/2)
		gammalnc<data_t>(v+data_t(1.5),&r2,&a2);
		//(1-1/z^2)^(m/2), (2z)^(v+1)
		power1	=	std::pow(data_t(1)-z2r,m/data_t(2));
		power2	=	std::pow(data_t(2)*z  ,v+data_t(1));
		//return results
		q	=	std::exp(im<data_t>*(a1-a2+m*PI<data_t>)+(r1-r2))
			*	(power1/power2*f21*data_t(1.7724538509055160273L));
		//error message
		return	info>0?10000:0;
	}//end of legendre_qvalue_real

	template<class data_t>
	static	inline	int	legendre_qvalue_comp	
	(
		const std::complex<data_t> m,
		const std::complex<data_t> v,
		const std::complex<data_t> z, 
		std::complex<data_t>& q,	//returned as Q(z;m,v)
		std::complex<data_t>* w		//workspace, at least longer than hypergeom_2f1_taylor_wksz
	)noexcept
	{
		using comp_t	=	std::complex<data_t>;
		int	info;
		comp_t	z2r	=	data_t(1)/(z*z);
		comp_t	f21,power1,power2;
		data_t	r1,r2,a1,a2;
		//1f2(a,b,c,z)
		info+=hypergeom_2f1<comp_t>
		(
			(data_t(2)+v+m)/data_t(2),
			(data_t(1)+v+m)/data_t(2),
			v+data_t(1.5),z2r,f21,w
		);
		//gammaln(v+m+1)
		gammalnc<data_t>(v+m+data_t(1),&r1,&a1);
		//gammaln(v+3/2)
		gammalnc<data_t>(v+data_t(1.5),&r2,&a2);
		//(1-1/z^2)^(m/2), (2z)^(v+1)
		power1	=	std::pow(data_t(1)-z2r,m/data_t(2));
		power2	=	std::pow(data_t(2)*z  ,v+data_t(1));
		//return results
		q	=	std::exp(im<data_t>*(a1-a2+m*PI<data_t>)+(r1-r2))
			*	(power1/power2*f21*data_t(1.7724538509055160273L));
		//error message
		return	info>0?15000:0;
	}//end of legendre_qvalue_comp

	template<class data_t>
	static	inline	int	legendre_pvalue_real	
	(
		const data_t m,
		const data_t v,
		const data_t z, 
		std::complex<data_t>& p,	//returned as P(z;m,v)
		data_t*	w			//workspace, at least longer than hypergeom_2f1_taylor_wksz
	)noexcept
	{
		int	info;
		data_t	z2r	=	data_t(1)/(z*z);
		data_t	f21p,f21m,r1,r2,r3,r4,a1,a2,a3,a4,power1,power2,power3;
		//1f2(a,b,c,z)
		info+=hypergeom_2f1<data_t>
		(
			(data_t(1)+v-m)/data_t(2),
			(data_t(2)+v-m)/data_t(2),
			data_t(1.5)+v,z2r,f21p,w
		);
		info+=hypergeom_2f1<data_t>
		(
			(         -v-m)/data_t(2),
			(data_t(1)-v-m)/data_t(2),
			data_t(0.5)-v,z2r,f21m,w
		);
		//gammaln(v+1/2)
		gammalnc<data_t>(v+data_t(0.5),&r3,&a3);
		//gammaln(v-m+1)
		gammalnc<data_t>(v-m+data_t(1),&r4,&a4);
		//gammaln(-v-1/2)
		gammalnc<data_t>(data_t(-0.5)-v,&r1,&a1);
		//gammaln(-v-m)
		gammalnc<data_t>(-v-m,&r2,&a2);
		//(1-1/z^2)^(m/2), (2z)^(v+1) (2z)^(-v)
		power1	=	std::pow(data_t(1)-z2r,m/data_t(2));
		power2	=	std::pow(data_t(2)*z  ,-v-data_t(1));
		power3	=	std::pow(data_t(2)*z  ,v);
		//return results
		p	=	std::exp(im<data_t>*(a1-a2)+(r1-r2))
			*	(power2/power1*f21p)
			+	std::exp(im<data_t>*(a3-a4)+(r3-r4))
			*	(power3/power1*f21m);
		p	/=	data_t(1.7724538509055160273L);//v should not be half integer
		//error message
		return	info>0?20000:0;
	}//end of legendre_pvalue_real
	
	template<class data_t>
	static	inline	int	legendre_pvalue_comp	
	(
		const std::complex<data_t> m,
		const std::complex<data_t> v,
		const std::complex<data_t> z, 
		std::complex<data_t>& p,	//returned as P(z;m,v)
		std::complex<data_t>* w		//workspace, at least longer than hypergeom_2f1_taylor_wksz
	)noexcept
	{
		using	comp_t	=	std::complex<data_t>;
		int	info;
		comp_t	z2r	=	data_t(1)/(z*z);
		comp_t	f21p,f21m,power1,power2,power3;
		data_t	r1,r2,r3,r4,a1,a2,a3,a4;
		//1f2(a,b,c,z)
		info+=hypergeom_2f1<comp_t>
		(
			(data_t(1)+v-m)/data_t(2),
			(data_t(2)+v-m)/data_t(2),
			data_t(1.5)+v,z2r,f21p,w
		);
		info+=hypergeom_2f1<comp_t>
		(
			(         -v-m)/data_t(2),
			(data_t(1)-v-m)/data_t(2),
			data_t(0.5)-v,z2r,f21m,w
		);
		//gammaln(v+1/2)
		gammalnc<data_t>(v+data_t(0.5),&r3,&a3);
		//gammaln(v-m+1)
		gammalnc<data_t>(v-m+data_t(1),&r4,&a4);
		//gammaln(-v-1/2)
		gammalnc<data_t>(-v-data_t(0.5),&r1,&a1);
		//gammaln(-v-m)
		gammalnc<data_t>(-v-m,&r2,&a2);
		//(1-1/z^2)^(m/2), (2z)^(v+1) (2z)^(-v)
		power1	=	std::pow(data_t(1)-z2r,m/data_t(2));
		power2	=	std::pow(data_t(2)*z  ,-v-data_t(1));
		power3	=	std::pow(data_t(2)*z  ,v);
		//return results
		p	=	std::exp(im<data_t>*(a1-a2)+(r1-r2))
			*	(power2/power1*f21p)
			+	std::exp(im<data_t>*(a3-a4)+(r3-r4))
			*	(power3/power1*f21m);
		p	/=	data_t(1.7724538509055160273L);//v should not be half integer
		//error message
		return	info>0?25000:0;
	}//end of legendre_pvalue_comp
	//----------------------------------------------------------------------------------------
	template<int mode,class data_t>//the input p and q are used as arrays of length nmax
	static	int	legendre_pq		
	(
		const data_t m,
		const data_t z,
		data_t* p,	//returned as P(m:m+nmax-1,m,z)
		data_t* q,	//returned as Q(m:m+nmax-1,m,z)
		const size_t nmax//at least 2
	)noexcept
	{
		int	info;
		size_t	n=nmax-1ul;
		data_t  tmp1,tmp2,tmp3,tmp4;
		tmp1    =       data_t(2)*m+data_t(1);
		tmp2    =       data_t(2)*m;
		info    =       legendre_ratio(m,data_t(n),z,tmp3);//evaluate P_l+1/P_l

		if constexpr(mode==0)//original normalization
		tmp4    =      	std::pow(z*z-data_t(1),m/data_t(2))*legendre_dfact(m);
		if constexpr(mode==1)//in this mode, Pm*Qm for different z are still correct
		tmp4    =       std::pow(z*z-data_t(1),m/data_t(2));
		if constexpr(mode==2)//in this mode, Pm*Qm for different z are still correct
		tmp4    =      	std::pow(z*z-data_t(1),m/data_t(2))*data_t(1e-290);

		//initial values for P0,P1
		p[0]    =       tmp4;
		p[1]    =       z*tmp1*p[0];
		//iterate for P upwardly by 3RR
		for(size_t i=1;i<n;++i)
		{
			p[i+1]=((data_t(2ul*i)+tmp1)*z*p[i]-(i+tmp2)*p[i-1])/data_t(i+1ul);
		}
		//determine Qn,Qn-1 from Qm/Qn-1 and Wronskian
		q[n-1]  =       legendre_rfact(m,data_t(n),data_t(1)/(p[n]-p[n-1]*tmp3));
		q[n]    =       tmp3*q[n-1];
		//iterate for Q downwardly by 3RR
		for(size_t i=n;i-->1;)
		{
			q[i-1]=((data_t(2ul*i)+tmp1)*z*q[i]-(i+1)*q[i+1])/(i+tmp2);
		}
		//error message
		return 	info;
	}//end of legendre_pq	

	template<class data_t>
	static	int	legendre_q
	(
		const std::complex<data_t>& m,
		const std::complex<data_t>& v,
		const std::complex<data_t>& z,
		std::complex<data_t>* q0,	//Q(z;m,v+(-nbwd:nfwd))
		size_t nfwd,			//>0
		size_t nbwd,			//>0
		std::complex<data_t>* wk	//workspace
	)noexcept
	{
		using comp_t	=	std::complex<data_t>;
		int 	info	=	0;
		comp_t	qm,nm;
		//do absolute normalization by at v
		info		+=	legendre_qvalue_comp<data_t>(m,v,z,nm,wk);
		if(nfwd==0&&nbwd==0)return info;
		//set initial value for iteration by Q/Q
		info		+=	legendre_ratio_comp<data_t>(m,v+data_t(nfwd-1ul),z,qm);
		q0[nfwd]	=	data_t(1e-250);
		q0[nfwd-1]	=	data_t(1e-250)/qm;
		//backward iterations using 3RR
		for(long i=long(nfwd)-1;i>-long(nbwd);--i)
		{
			comp_t n=	v+data_t(i);
			q0[i-1]	=	q0[i+1]*(n+data_t(1)-m)
				-	q0[i  ]*(n*data_t(2)+data_t(1))*z;
			q0[i-1]	/=	-(n+m);
		}	nm	/=	q0[0];
		//renormalization
		for(long i=-long(nbwd);i<=long(nfwd);++i)
		{
			q0[i]	*=	nm;
		}
		return 	info;
	}//end of legendre_q

}//end of qpc
