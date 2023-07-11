#pragma once

#include <libraries/support_avx.h>
#include <libraries/support_gsl.h>	//for gaussian-legendre quadrature
#include <libraries/support_std.h>

#include <utilities/error.h>
//====================================================================================================
//
//                      This file is part of QPC Header Only Library
//
//====================================================================================================
//====================================================================================================
//
//				B-spline Integrator & Evaluator
//
//	References:
//	[1] C. de Boor, A Practical Guide to Splines, New York: Springer (1978)
//	[2] H. Bachau, E. Cormier, P. Decleva, J. E. Hansen, F. Martín, Rep, Prog, Phys, 64 (2001)
//
//	Description:
//	(1) See the text later
//
//Updated by: Zhao-Han Zhang(张兆涵)  Dec. 29th, 2022
//Copyright © 2022 Zhao-Han Zhang
//====================================================================================================

namespace qpc{
namespace bss{
	//=================================================================================================
 	// 				@@@@@@@@@@@@@@@ IN BRIEF @@@@@@@@@@@@@@@
 	//
 	// n_leng is the number of the b-spline functions, not the number of knots
 	// n_rank is the rank of the the b-spline functions
 	// b_lfbd is the flag enabling(when b_lfbd=true) the left-most non-zero B-spline function.
 	// b_rtbd is the flag enabling(when b_rtbd=true) the right-most non-zero B-spline function.
 	//
 	// bsplines<...>::axis is the B-spline integrator, storing knots, values of Bp at gaussian points, 
 	//  gaussian weights to perform integrals upon Bp
 	// bsplines<...>::disp is the B-spline displayer, storing values of Bp at wanted points, 
 	//  for calculating the summations once given coefficients
 	//
	//=================================================================================================
 	// 				@@@@@@@@@@@@@@@ IN DETAIL @@@@@@@@@@@@@@@
	//n+1 unequal knots: 
	//	x0,x1,x2...,xn-1,xn
	//
	//n nonzero-length intervals:
	//	[x0,x1),[x1,x2),...[xn-1,xn)
	//
	//x0,xn are both repeated for p times, thus:
	//	(1)totally n+1+2p knots.
	//	(2)the first/last p+1 have the same value, forming p intervals with 0 length on each side:
	//		x0,...,x0,x1,x2,...,xn-1,xn,...,xn
	//	   thus totally n+2p intervals.
	//
	//n+p Bp-s (including the discarded ones, seen later), each is only non-zero on p+1 intervals:
	//	{[x0,x0),...,[x0,x0),[x0,x1)},
	//	{[x0,x0),...,[x0,x1),[x1,x2)},
	//	...
	//	{[xn-2,xn-1),[xn-1,xn)...,[xn,xn)},
	//	{[xn-1,xn  ),[xn  ,xn)...,[xn,xn)}
	//
	//note	(1)the first/last Bp equals 1 at x0/xn, whose last/first domain is [x0,x1]/[xn-1,xn]. 
	//	they can be discarded by setting <lfbd> and <rtbd> in the template arguments. they are discarded
	//	by default.
	//	(2)the remaining n+p-# Bp satisfy reflective boundary conditions (Bp=0 at x0&xn).
	//	(3)let N=n+p-# denote the number of active basis, thus n=N+#-p>0. there are n+2p-#=N+p active 
	//	intervals, and N+p+1 active knots.
	//
	//	!!!(N,p,# are named as n_leng,n_rank,n_lfbd,n_rtbd in the code)!!!
	//
	//storage format:
	//	(0) n+1+2p = N+p+1+# knots are stored in an indented array (by #left, since it's inactive):
	//		knot (0,1,2,...,p-1)-#left  	equal x0,x0,...
	//		knot (p,p+1,...,N+#)-#left   	equal x0,x1,x2,...xn-1,xn
	//		knot (N+#+1,...,N+#+p)-#left 	equal xn,xn,...
	//	    now the i-th interval refers to [knot(i),knot(i+1)).
	//	(1) there are q gauss points on each interval. (q is set to 2p+2 for empirical reason)
	//	(2) there are p+1 non-zero Bp-s on each interval. 
	//	(3) on each interval, p+1 arrays (of size q) store the value of p+1 Bp-s on q gauss points.
	//	(4) the k-th (0<=k<=p) array on the i-th (0<=i<=N+p-1) active interval stores
	//	the value from the (i-p+k)-th basis. for those i-p+k<0 or i-p+k>=N, they are padded with 0. 
	//	namely on i-th interval:
	//	index of array	0	...	k	...	p
	//	index of basis 	i-p     ...	i-p+k   ...	i
	//
	//utilities:
	//	(1) knot(i) returns the i-th knot (left-hand side inactive basis has index -1, if exists) defined above.
	//
	//	(2) axis()  returns the array of gaussian points on all active intervals.
	//
	//	    axis(i) returns the array of gaussian points on i-th active intervals.
	//
	//	    leng()  returns the length of array returned by calling axis()
	//
	//	(3) cal<p,m>(i,x) returns the value of (d^m/dx^m)B[i,p](x) evaluated by recurrence relation.
	//	
	//	(4) dump<m>(i,b) returns the value of (d^m/dx^m)B[i,p] on gauss points into b.
	//
	//	(5) integrate<mi,mj>(func,i,j) returns the numerical value of the integral by gauss-legendre quadrature
	//		I = (d^mi/dx^mi)B[i,p]   *    func(x)   *    (d^mj/dx^mj)B[j,p] (mi,mj = 0,1,2)
	//
	//	(6) integrate<mi>(func,i) returns the numerical value of the integral by gauss-legendre quadrature
	//		I = (d^mi/dx^mi)B[i,p]   *    func(x) 				(mi=0,1,2)
	//=========================================================================================================	
	template<size_t n_leng,size_t n_rank,bool b_lfbd=0,bool b_rtbd=0,size_t glinfix=0ul>
	struct	bsplines
	{
		static constexpr size_t n_glin 	=	2ul*n_rank+2ul+glinfix;	//the empirical rank for gaussian-legendre quadrature
		static constexpr size_t n_lfbd	=	b_lfbd?0ul:1ul;	//the number of inactive knots to the left of zeroth active knot.
		static constexpr size_t n_rtbd	=	b_rtbd?0ul:1ul; //the number of inactive knots to the right of last active knot.
		static constexpr size_t n_knot	=	n_leng+n_rank+1ul+n_lfbd+n_rtbd;//number of all knots, i.e. n

		static_assert(n_leng+n_lfbd+n_rtbd>n_rank);


		using	cach_t	=	std::array<double,n_glin>;	//the array type to store the value of one b-spline function on gaussian point on each interval.
		using	cach_v	=	std::array<cach_t,n_rank+1ul>;	//the array type to store the value of all b-spline function on gaussian point on each interval. there are n_rank+1 non-zero b-splines on each interval

		//------------------------------------------------------------------------------------------------------------------------------
		struct	axis_data
		{
			protected:
				double* knt;    //array of knot
                                cach_v* val[3]; //array of basis value(for 0,1,2 order derivative),on i-th interval. val[.][i][n_rank-s] will belong to B{i-s}.
                                cach_t* gsv;    //array of gaussian points, on i-th interval
                                cach_t* gsw;    //array of gaussian weight, on i-th interval
			public:
				//access knot,axis,weight(for debug or direct access)
				inline	double*	knot	()const noexcept
				{
					return	knt;
				}
				inline	double*	axis	()const noexcept
				{
					return 	reinterpret_cast<double*>(gsv);//used as an array of axis value (in rising order)
				}
				template<size_t m>
				inline	double*	base	()const noexcept
				{
					return	reinterpret_cast<double*>(val[m]);//used as an array of base values
				}
				inline	double*	weight	()const noexcept
				{
					return 	reinterpret_cast<double*>(gsw);//used as an array of gaussian weights
				}

				inline	double	knot	(const size_t i)const noexcept
				{
					return 	knt[i];
				}		
				inline	double*	axis	(const size_t i)const noexcept
				{
					return 	gsv[i].data();			//for std::array, use .data(); for qpc::array, use .view()
				}
				template<size_t m>
				inline	cach_t*	base	(const size_t i)const noexcept
				{
					return	val[m][i].data();		//for std::array, use .data(); for qpc::array, use .view()
				}
				inline	double*	weight	(const size_t i)const noexcept
				{	
					return	gsw[i].data();			//for std::array, use .data(); for qpc::array, use .view()
				}

				static constexpr size_t	leng()noexcept
				{
					return 	n_glin*(n_leng+n_rank);		//length of the array returned by calling axis() and weight(), (is the # of gauss points on all active intervals)
				}
				static constexpr size_t rank()noexcept
				{
					return 	n_rank;				//length -1 of the array returned by calling base(i)
				}
				static constexpr size_t size()noexcept
				{
					return	n_leng+n_rank;			//index range to loop over all active intervals.
				}
				static constexpr size_t glin()noexcept
				{
					return 	n_glin;				//length of the array returned by calling axis(i) and weight(i)
				}
				//do integration manually or for debug
				template<size_t m>
				[[deprecated]] inline	void	dump	(const size_t i,double* b)const noexcept//for debug or display, return the value of i-th basis
				{
					size_t	j=0;
					cach_v*	v=val[m];
					for(size_t s=0;s<=n_rank;++s)
					{
						auto&	_v	=	v[i+s][n_rank-s];
						for(size_t k=0;k<n_glin;++k)
						{
							b[j++]	=	_v[k];//b should be longer than (n_rank+1)*n_glin
						}
					}
				}//end of dump

				//memory allocation
				explicit axis_data():
				knt(new (std::nothrow) double[n_knot]+n_lfbd),//indent if the first knot is inactive
				val{new (std::nothrow) cach_v[n_knot]+n_lfbd,
				    new (std::nothrow) cach_v[n_knot]+n_lfbd,
				    new (std::nothrow) cach_v[n_knot]+n_lfbd},
				gsv(new (std::nothrow) cach_t[n_knot]+n_lfbd),
				gsw(new (std::nothrow) cach_t[n_knot]+n_lfbd)
				{
					if(knt-n_lfbd==0 || val[0]-n_lfbd==0 || val[1]-n_lfbd==0 || val[2]-n_lfbd==0 || gsv-n_lfbd==0 || gsw-n_lfbd==0)
					{
						throw qpc::runtime_error_t<std::string,void>("allocation error of B-spline basic data.");
					}
				}//end of constructor
				virtual ~axis_data()noexcept
				{
					delete[] (val[0]-n_lfbd);
					delete[] (val[1]-n_lfbd);
					delete[] (val[2]-n_lfbd);
					delete[] (knt-n_lfbd);
					delete[] (gsv-n_lfbd);
					delete[] (gsw-n_lfbd);
				}//end of destructor
		};//end of axis_data

		//------------------------------------------------------------------------------------------------------------------------------
		struct	axis_type:public axis_data
		{
			protected:
				using	axis_data::val;
				using	axis_data::knt;
				using	axis_data::gsw;
				using	axis_data::gsv;
			public:

				template<size_t p=n_rank,size_t m=0ul>
				inline	double	cal	(const size_t i,const double x)const noexcept//calculate (d/dx)^m B^p_i(x), where N>i>=0
				{
					static_assert(p<=n_rank);
					if constexpr(m==0){//calc by definition
						if constexpr(p==0){
							if(x>=knt[i]&&x<knt[i+1])    	return  1.0;
							else                            return  0.0;
						}else{
							double	wl=knt[i+p    ]-knt[i    ];
							double	wr=knt[i+p+1ul]-knt[i+1ul];
							double	xl= wl==0.?0.:(x-knt[i      ])/wl*cal<p-1ul>(i    ,x);//recursively call cal
							double	xr= wr==0.?0.:(x-knt[i+1ul+p])/wr*cal<p-1ul>(i+1ul,x);//recursively call cal
							return	xl-xr;
						}
					}else if constexpr(m>=1){//calc by recurrence relation for derivative of B-spline function
						if constexpr(p==0){
							return  0.0;
						}else{
							double	wl=knt[i+p    ]-knt[i    ];
							double	wr=knt[i+p+1ul]-knt[i+1ul];
							double	xl= wl==0.?0.:cal<p-1ul,m-1ul>(i    ,x)/wl;//recursively call cal
							double	xr= wr==0.?0.:cal<p-1ul,m-1ul>(i+1ul,x)/wr;//recursively call cal
							return 	(xl-xr)*p;
						}
					}
				}//end of cal

				template<size_t mi,size_t mj,class func_t>
				inline	double	integrate (const func_t& mid,const size_t i,const size_t j)const noexcept
				{
					static_assert(mi<=2&&mj<=2);
					double retn=0.;
					if(i>n_leng||j>n_leng)return retn;//pad with 0
					size_t imin=max(i,j);		//min non-zero interval
					size_t imax=min(i,j)+n_rank;	//max non-zero interval,avoid error when minus unsigned integer are passed in
					auto*	lhs=val[mi];
					auto*	rhs=val[mj];
					for(size_t s=imin;s<=imax;++s)//on s-th interval
					{
						for(size_t k=0;k<n_glin;++k)//for the k-th gauss point
						{
							if constexpr(std::is_invocable_v<func_t,double>)
							{//for functor accepting double
								retn+=gsw[s][k]*mid(gsv[s][k])*lhs[s][n_rank+i-s][k]*rhs[s][n_rank+j-s][k];
							}else if constexpr(std::is_invocable_v<func_t,size_t,size_t>)
							{//for functor accepting size_t
								retn+=gsw[s][k]*mid(s,k) *lhs[s][n_rank+i-s][k]*rhs[s][n_rank+j-s][k];
							}else	
							{//for 2d-array-like object
								retn+=gsw[s][k]*mid[s][k]*lhs[s][n_rank+i-s][k]*rhs[s][n_rank+j-s][k];
							}
						}
					}
					return 	retn;
				}//end of integrate

				template<size_t mi,class func_t>
				inline	double	integrate (const func_t& mid,const size_t i)const noexcept
				{
					static_assert(mi<=2);
					double 	retn=0.;
					if(i>n_leng)return 0.;//pad with 0
					auto*	lhs=val[mi];
					for(size_t s=i;s<=i+n_rank;++s)//on s-th interval
					{
						for(size_t k=0;k<n_glin;++k)//for the k-th gauss point
						{
							if constexpr(std::is_invocable_v<func_t,double>)
							{//for functor accepting double
								retn+=gsw[s][k]*mid(gsv[s][k])*lhs[s][n_rank+i-s][k];
							}else if constexpr(std::is_invocable_v<func_t,size_t,size_t>)
							{//for functor accepting size_t
								retn+=gsw[s][k]*mid(s,k)*lhs[s][n_rank+i-s][k];
							}else if constexpr(std::is_invocable_v<func_t,size_t const&>)
							{//for 1d-array like object
								retn+=gsw[s][k]*mid(s*n_glin+k)*lhs[s][n_rank+i-s][k];
							}else
							{//for 1d-array
								retn+=gsw[s][k]*mid[s*n_glin+k]*lhs[s][n_rank+i-s][k];
							}
						}
					}
					return 	retn;
				}//end of integrate

				inline	void	overlap3 (const size_t i,double* res)const noexcept//#res= packed storage (n_rank+1,n_rank+1)
				{
					if(i>n_leng)return;	

					auto* 	_val = val[0];

					size_t count = 0ul;
					for(size_t j=i;j<=i+n_rank;++j)//only j>=i 
					for(size_t k=j;k<=i+n_rank;++k)//only k>=j
					{
						double sum = 0.0;
						for(size_t s=k;s<=i+n_rank;++s)//on s-th interval
						{
							size_t ish = n_rank+i-s;
							size_t jsh = n_rank+j-s;
							size_t ksh = n_rank+k-s;
							auto&__val = _val[s];
							auto&__gsv = gsv[s];
							auto&__gsw = gsw[s];
							for(size_t x=0;x<n_glin;++x)//for the x-th gauss point
							{
								double __xval = __gsv[x]; if(__xval==0.)continue;
								double __wval = __gsw[x];
								sum	+= 	__wval/__xval 	//1/r
									*  	__val[ish][x]
									*  	__val[jsh][x]
									*	__val[ksh][x];
							}
						}
						if(i==k)sum*=1./6.;
						else if(i==j||k==j)sum*=1./2.;
						res[count++]=sum;
					}
				}//end of overlap3
					
				template<size_t mi,size_t mj,class func_t>
				inline	void	integrate_interval	(const func_t& mid,const size_t s,double* retn)const noexcept//on s-th active interval, 0 <= s < n_leng+n_rank
				{
					static_assert(mi<=2&&mj<=2);
					auto&	lhs	=	val[mi][s];
					auto&	rhs	=	val[mj][s];
					auto&	wat	=	gsw[s];
					size_t	c=0ul;
					for(size_t i=0;i<=n_rank;++i)
					for(size_t j=i;j<=n_rank;++j)
					{
						double	sum=0.0;
						for(size_t k=0;k<n_glin;++k)//for the k-th gauss point
						{
							sum+=mid(k)*wat[k]*lhs[i][k]*rhs[j][k];
						}
						retn[c++]=sum;
					}
				}//end of integrate_interval

				inline	void	initialize()noexcept//to be called after setting all knots. note GSL error may be invoked from this call.
				{
					gsl_integration_glfixed_table* glt	=	gsl_integration_glfixed_table_alloc(n_glin);
					for(size_t id=0ul;id<n_knot-1ul;++id)
					{//on i-th interval. note there are N+p active intervals, # inactive intervals
						size_t	i =id-n_lfbd; //shift id to include the inactive intervals
						double	xl=knt[i];
						double	xr=knt[i+1ul];
						if(xl!=xr)
						{//knot[i]!=knot[i+1]
							for(size_t k=0;k<n_glin;++k)
							{//the k-th gaussian point on i-th interval
								double xi,wi;
								gsl_integration_glfixed_point(xl,xr,k,&xi,&wi,glt);
								gsv[i][k]=xi;
								gsw[i][k]=wi;
								for(size_t s=0;s<=n_rank;++s)//[i][n_rank][:] is the value for i-th base
								val[0][i][n_rank-s][k]=cal<n_rank,0>(i-s,xi);
								for(size_t s=0;s<=n_rank;++s)
								val[1][i][n_rank-s][k]=cal<n_rank,1>(i-s,xi);
								for(size_t s=0;s<=n_rank;++s)
								val[2][i][n_rank-s][k]=cal<n_rank,2>(i-s,xi);
							}
						}else
						{//knot[i]==knot[i+1]
							for(size_t k=0;k<n_glin;++k)
							{//the k-th gaussian point on i-th interval
								gsv[i][k]=xl;
								gsw[i][k]=0.;
								for(size_t s=0;s<=n_rank;++s)
								val[0][i][n_rank-s][k]=0.0;
								for(size_t s=0;s<=n_rank;++s)
								val[1][i][n_rank-s][k]=0.0;
								for(size_t s=0;s<=n_rank;++s)
								val[2][i][n_rank-s][k]=0.0;
							}
						}
					}
					gsl_integration_glfixed_table_free(glt);
				}//end of initialize


				inline	void	repeatively_set_knots			(const double xval,const size_t init,const size_t last)const noexcept
				{
					for(size_t i=init;i<last;++i)
					{
						knt[i-n_lfbd] 	=       xval;   	//shift by n_lfbd to include the inactive knots
					}
				}//end of repeatively_set_knots(1)

				template<class func_t>
				inline	void	repeatively_set_knots			(const func_t& func,const size_t init,const size_t last)const noexcept
				{
					for(size_t i=init;i<last;++i)
					{
						knt[i-n_lfbd]	=	func(i-init);	//shift by n_lfbd to include the inactive knots
					}
				}//end of repeatively_set_knots(2)

				inline	void	repeatively_set_heads			(const double xmin)const noexcept
				{
					this->repeatively_set_knots(xmin,0,n_rank);
				}//end of repeatively_set_heads

				inline	void	repeatively_set_tails			(const double xmax)const noexcept
				{
					this->repeatively_set_knots(xmax,n_leng+n_lfbd+n_rtbd,n_leng+n_lfbd+n_rtbd+n_rank+1ul);
				}//end of repeatively_set_tails

				template<class func_t>
				inline	void	repeatively_set_elses			(const func_t& func)const noexcept
				{
					this->repeatively_set_knots(func,n_rank,n_leng+n_lfbd+n_rtbd);
				}//end of repeatively_set_elses

				inline	void	initialize_axis_linear_step		(const double xmin,const double xmax)noexcept
				{
					double	step	=	(xmax-xmin)/(n_leng+n_lfbd+n_rtbd-n_rank);//n=N+#-p, x1-x0=(xn-x0)/n
					auto	func	=	[&](size_t id){return xmin+id*step;};

					this->repeatively_set_heads(xmin);
					this->repeatively_set_elses(func);
					this->repeatively_set_tails(xmax);
					initialize();
				}//end of initialize_axis_linear_step 


				inline	void	initialize_axis_sine2_mapping		(const double xmin,const double xmax)noexcept
				{
					double  step    =       1.0/(n_leng+n_lfbd+n_rtbd-n_rank);//n=N+#-p, x1-x0=(xn-x0)/n
					auto	func	=	[&](size_t id)
					{
						double	tmp	=	id*step;
						double	val  	= 	sin(0.5*PI<double>*tmp);//you should avoid tmp exceeding 1.0
						return	val*val*(xmax-xmin) + xmin;
					};
					this->repeatively_set_heads(xmin);
					this->repeatively_set_elses(func);
					this->repeatively_set_tails(xmax);
					initialize();
				}//end of initialize_axis_sine2_mapping
	
				template<size_t n_pole>
				inline	void	initialize_axis_piecewise_step		(const double xmin,const double xmax,const std::array<double,n_pole> xmid,const std::array<size_t,n_pole> each,const double para=0.0)noexcept
				{
					static_assert(n_rank>=2ul);
	
					this->repeatively_set_heads(xmin);
					size_t constexpr n_rept = n_rank-1ul;
	
					size_t	part	=	n_rank;
					double	xval	=	xmin;
					for(size_t j=0;j<n_pole;++j)
					{
						double 	step	=	(xmid[j]-xval)/each[j];
						auto	func	=	[&](size_t id){return xval+id*step;};
						this->repeatively_set_knots(func,part,each[j]+part);
						//repeat n_rank-2 times, such that B,B' is continuous, and B'' is not
						this->repeatively_set_knots(xmid[j],each[j]+part,part+each[j]+n_rept);

						part	+=	each[j]+n_rept;
						xval	=	xmid[j];
						//if(part>=n_leng+n_lfbd+n_rtbd)return; trust the caller
					}
					if(para==0.0)//reduces to linear sequence
					{
						double	step	=	(xmax-xval)/(n_leng+n_lfbd+n_rtbd-part);
						auto	func	=	[&](size_t id){return xval+id*step;};
						this->repeatively_set_knots(func,part,n_leng+n_lfbd+n_rtbd);
					}else
					{
						double  temp    =       (xmax-xval)/expm1(para);
						double  step    =       para/(n_leng+n_lfbd+n_rtbd-part);
						auto	func	=	[&](size_t id){return xval+temp*expm1(step*id);};
						this->repeatively_set_knots(func,part,n_leng+n_lfbd+n_rtbd);
					}
					this->repeatively_set_tails(xmax);
					initialize();
				}//end of initialize_axis_piecewise_step

				inline 	void	initialize_axis_exponential_step	(const double xmin,const double xmax,const double para)noexcept
				{
					double	temp	=	(xmax-xmin)/expm1(para);
					double	step	=	para/(n_leng+n_lfbd+n_rtbd-n_rank);//n=N+#-p
					auto	func	=	[&](const size_t id){return xmin+temp*expm1(step*id);};

					this->repeatively_set_heads(xmin);
					this->repeatively_set_elses(func);
					this->repeatively_set_tails(xmax);

					initialize();
				}//end of initialize_axis_exponential_step
		};//end of axis_type

		//------------------------------------------------------------------------------------------------------------------------------
		struct	disp_data
		{
			public:
						//0<=k<n_leng is the index of base function
				double**v;	//basis value, v[k][...] is the non-zero values of the k-th base
				size_t**i;	//index value, i[k][...] is the non-zero values of the k-th base (base[k] only takes non-zero values on x[i[k][...]])
				double*	x;	//axis value, x[j] is the value of j-th axis point, 0<=j<m
				size_t*	n;	//leng value, n[k] is the length of i[k]&v[k]
				size_t	m;	//number of axis point	

				
				explicit disp_data():x(0)
				{
					v=new (std::nothrow) double*[n_leng];
					i=new (std::nothrow) size_t*[n_leng];
					n=new (std::nothrow) size_t [n_leng];
					if(v==0||i==0||n==0)
					{
						throw qpc::runtime_error_t<std::string,void>("allocation error in B-spline disp data.");
					}
					#pragma GCC ivdep
					for(size_t k=0;k<n_leng;++k)
					{
						v[k]=0;
						i[k]=0;
					}
				}//end of constructor
				virtual ~disp_data()noexcept
				{
					for(size_t k=0;k<n_leng;++k)
					{
						if(v[k])delete[] v[k];
						if(i[k])delete[] i[k];
					}
					if(x)delete[] x;
					delete[] n;
					delete[] v;
					delete[] i;
				}//end of destructor
		};//end of disp_data

		//------------------------------------------------------------------------------------------------------------------------------
		struct	disp_type:public disp_data
		{
			public:
				using	disp_data::v;		
				using	disp_data::i;		
				using	disp_data::x;		
				using	disp_data::n;		
				using	disp_data::m;		

				template<class coef_t,class wave_t>
				inline 	void	accumulate	(const coef_t* _cf,wave_t* _wf)noexcept
				{//implements _wf+= sum(_cf[i]*base[i])
					for(size_t k=0;k<n_leng;++k)//loops over basis
					{
						auto    coef    =       _cf[k];
						auto*   val     =       v[k];
						auto*   idx     =       i[k];
						for(size_t s=0;s<n[k];++s)
						{
							_wf[idx[s]]+=   coef*val[s];
						}
					}	
				}//end of accumulate

				template<class coef_t,class wave_t,class ampd_t>
				inline 	void	accumulate	(const coef_t* _cf,wave_t* _wf,const ampd_t& _ef)noexcept
				{//implements _wf+= sum(_cf[i]*base[i]*_ef[i])or _wf+=_ef*sum(_cf[i]*base[i])
					if constexpr(std::is_invocable_v<ampd_t,size_t>)//when _ef[i] is ok 
					{
						for(size_t k=0;k<n_leng;++k)//loops over basis
						{
							auto	coef	=	_cf[k];
							auto*	val	=	v[k];
							auto*	idx	=	i[k];
							for(size_t s=0;s<n[k];++s)
							{
								_wf[idx[s]]+=   coef*val[s]*_ef(idx[s]);
							}
						}	
					}else//suppose _ef is constant
					{
						for(size_t k=0;k<n_leng;++k)//loops over basis
						{
							auto	coef	=	_cf[k]*_ef;
							auto*	val	=	v[k];
							auto*	idx	=	i[k];
							for(size_t s=0;s<n[k];++s)
							{
								_wf[idx[s]]+=	coef*val[s];
							}
						}
					}
				}//end of accumulate

				template<class coef_t,class wave_t,class ampd_t>
				inline 	void	accumulate2d	(const coef_t* _cf,wave_t* _wf,const ampd_t& _ef)noexcept
				{//implements _wf+= sum(_cf[i,j]*base[i]*base[j]*_ef[i,j])or _wf+=_ef*sum(_cf[i,j]*base[i]*base[j])
					if constexpr(std::is_invocable_v<ampd_t,size_t>)//when _ef[i] is ok 
					{
						for(size_t j=0;j<n_leng;++j)//loops over basis
						for(size_t k=0;k<n_leng;++k)//loops over basis
						{
							auto	coef	=	_cf[j*n_leng+k];
							auto*	val1	=	v[j];
							auto*	val2	=	v[k];
							auto*	idx1	=	i[j];
							auto*	idx2	=	i[k];
							for(size_t r=0;r<n[j];++r)
							for(size_t s=0;s<n[k];++s)
							{
								size_t i=	idx1[r]*m+idx2[s];
								_wf[i]	+=   	coef*val1[r]*val2[s]*_ef(i);
							}
						}	
					}else//suppose _ef is constant
					{
						for(size_t j=0;j<n_leng;++j)//loops over basis
						for(size_t k=0;k<n_leng;++k)//loops over basis
						{
							auto	coef	=	_cf[j*n_leng+k]*_ef;
							auto*   val1    =       v[j];
							auto*   val2    =       v[k];
							auto*   idx1    =       i[j];
							auto*   idx2    =       i[k];
							for(size_t r=0;r<n[j];++r)
							for(size_t s=0;s<n[k];++s)
							{
								size_t i=	idx1[r]*m+idx2[s];
								_wf[i]	+=	coef*val1[r]*val2[s];
							}
						}
					}//end
				}//end of accumulate2d

				template<class base_t,class axis_t>
				inline  void	initialize	(const base_t& _bs,const axis_t& _ax,const size_t _sz)//the input _bs should be the b-spline basis object, _ax the axis value functor,_sz the number of required points
				{
					m=_sz;//_sz should be larger than 0
					if(x)delete[] x;
					for(size_t k=0;k<n_leng;++k)
					{
						if(v[k]){delete[] v[k];v[k]=0;}
						if(i[k]){delete[] i[k];i[k]=0;}
					}
					x=new double[m];
					for(size_t j=0;j<m;++j)//assign value of axis
					{
						x[j]=_ax(j);
					}
					for(size_t k=0;k<n_leng;++k)//for k-th base, search non-zero points
					{
						size_t	counter=0;
						for(size_t j=0;j<m;++j)//count the number of non-zero terms
						{
							double	tmp=_bs.template cal<n_rank,0>(k,x[j]);
							if(fabs(tmp)>0.)counter++;
						}
						n[k]=counter;
						if(counter>0)
						{
							v[k]=new (std::nothrow) double[counter];
							i[k]=new (std::nothrow) size_t[counter];
							if(v[k]==0||i[k]==0)throw qpc::runtime_error_t<std::string,void>("allocation error in B-spline disp data.");
						}
						counter=0;
						for(size_t j=0;j<m;++j)//dump the values&indice of non-zero terms
						{
							double	tmp=_bs.template cal<n_rank,0>(k,x[j]);
							if(fabs(tmp)>0.)
							{
								v[k][counter]=tmp;
								i[k][counter]=j;
								counter++;
							}
						}
					}
				}//end of initialize
		};//end of disp_type

		//------------------------------------------------------------------------------------------------------------------------------
		struct	show_type final//for a single point
		{
			double*	val0;
			double*	val1;
			size_t*	indx;
			size_t	size;
			double	axis;

			template<class base_t>
			inline	void	initialize	(const base_t& base,const double _axis)
			{
				if(val0){delete[] val0;val0=0;}
				if(val1){delete[] val1;val1=0;}
				if(indx){delete[] indx;indx=0;}
				size=0ul;
				axis=_axis;
				for(size_t i=0;i<n_leng;++i)
				{
					if(std::fabs(base.cal(i,axis))>0.)size++;
				}
				val0=new (std::nothrow) double[size];				
				val1=new (std::nothrow) double[size];				
				indx=new (std::nothrow) size_t[size];
				if(val0==0||val1==0||indx==0)
				{
					throw qpc::runtime_error_t<std::string,void>("allocation error in B-spline show data.");
				}
				size=0ul;
				for(size_t i=0;i<n_leng;++i)
				{
					double _val0=base.template cal<n_rank,0ul>(i,axis);
					if(std::fabs(_val0)>0.)
					{
					double _val1=base.template cal<n_rank,1ul>(i,axis);
						val0[size]=_val0;
						val1[size]=_val1;
						indx[size]=i;
						++size;
					}
				}
			}//end of initialize

			template<size_t m=0>
			inline	double	accumulate	(const double* coef)const noexcept
			{
				double	sum=0.0;
				for(size_t i=0;i<size;++i)
				{
					if constexpr(m==0)sum+=coef[indx[i]]*val0[i];
					if constexpr(m==1)sum+=coef[indx[i]]*val1[i];
				}
				return 	sum;
			}//end of accumulate

			explicit show_type()noexcept:
			val0(0),
			val1(0),
			indx(0){}

			~show_type()noexcept
			{
				if(val0)delete[] val0;
				if(val1)delete[] val1;
				if(indx)delete[] indx;
			}
		};//end of show_type

		//-------------------------------------------------
		using	disp	=	disp_type;
		using	axis	=	axis_type;
		using	show	=	show_type;

	};//end of bsplines
}//end of bss
}//end of qpc
