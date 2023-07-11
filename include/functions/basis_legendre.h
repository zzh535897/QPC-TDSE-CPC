#pragma once
#include <libraries/support_avx.h>
#include <libraries/support_gsl.h>
#include <libraries/support_std.h>

#include <utilities/const.h>
#include <utilities/error.h>
//====================================================================================================
//
//                      This file is part of QPC Header Only Library
//
//====================================================================================================

//====================================================================================================
//									Spherical Harmonics
//	Descrition:
//	(1) By calling GSL, values and derivatives of (regular) spherical harmonics Ylm(θ,φ) with 
//	integer m and l>=|m| are computed. The domain of azimuthal angule should be 0<=θ<=pi,0<=φ<2pi,
//	which is neccessary for QPC spherical TDSE codes.
//
//	For details, one should refer to GSL mannual.
//
//====================================================================================================
namespace qpc{
namespace bss{

	struct	spherical_harmonics
	{
		static constexpr double cphase=-1.;
		//----------------------------------------------------------------------------
		//if mode=0, derivatives are dP(x)/d(x)
		//if mode=1, derivatives are dP(cosx)/dx
		//
		//	max() 	returns the length of legendre data array (as gsl format)
		//	idx(l,m) returns the index for l,m inside one legendre data array
		//	cal(x,f) returns the value at x into legendre data arrays f
		//
		//the maximum l is lmax, for any m.
		//----------------------------------------------------------------------------
		template<size_t lmax,int mode=1,int norm=0>
		class	legendre//this class implements numerical integral of the legendre function part of the spherical harmonics
		{
			public:
				static constexpr auto	normflag	=	norm==0?GSL_SF_LEGENDRE_SPHARM:
										norm==1?GSL_SF_LEGENDRE_FULL:
											GSL_SF_LEGENDRE_NONE;

				//--------------------------------------------------------------------------------------
				static	inline	size_t	max		()noexcept
				{
					return 	gsl_sf_legendre_array_n(lmax);
				}//end of max
				static	inline	size_t	idx		(const size_t l,const size_t m)noexcept//note that m>=0. for negative m, use Y(l,-m)=conj[Y(l,m)]*(-)^m
				{
					return 	gsl_sf_legendre_array_index(l,m);
				}//end of idx
				static 	inline	double	get		(const double*y,const long l,const long m)noexcept
				{
					return	m>=0?y[idx(l,m)]:(y[idx(l,-m)]*(m%2?-1.:1.));
				}//end of get
				template<class...args_t>
				static	inline	void	cal		(const double x,args_t&&...f)noexcept//f is the array to store the computed data
				{
					constexpr size_t n_deri=(sizeof...(args_t))-1ul;
					static_assert(n_deri<=2ul);
					if constexpr(mode==0)
					{
						if constexpr(n_deri==0)
						gsl_sf_legendre_array_e    	(normflag,lmax,x,cphase,f...);
						if constexpr(n_deri==1)
						gsl_sf_legendre_deriv1_array_e  (normflag,lmax,x,cphase,f...);
						if constexpr(n_deri==2)
						gsl_sf_legendre_deriv2_array_e  (normflag,lmax,x,cphase,f...);
					}
					if constexpr(mode==1)
					{
						if constexpr(n_deri==0)
						gsl_sf_legendre_array_e		  (normflag,lmax,x,cphase,f...);
						if constexpr(n_deri==1)
						gsl_sf_legendre_deriv1_alt_array_e(normflag,lmax,x,cphase,f...);
						if constexpr(n_deri==2)
						gsl_sf_legendre_deriv2_alt_array_e(normflag,lmax,x,cphase,f...);
					}
				}//end of cal
	
				//--------------------------------------------------------------------------------------
		};//end of legendre


		template<size_t mmax,size_t ldim,int mode=0>
		class	disp 	final
		{
			public:
				static_assert(ldim>0);
				static constexpr size_t lmax	=	mmax+ldim-1ul;//mmax should be max |M|

				using	comp_t	=	avx128_complex;
				double*	_th;//value of theta (mode=0) or cos(theta) (mode=1)
				double*	_ph;//value of phi

				double*	vth;//value of Plm(cos(th)), vth[ith+idx(l,m)*nth]=Plm(cos(_th[ith])), where idx(l,m) given by gsl
				double*	vpr;//value of cos(m*ph), vpr[iph+im*nph]=cos(m(im)*_ph[iph])
				double*	vpi;//value of sin(m*ph), vpi[iph+im*nph]=sin(m(im)*_ph[iph]), where m(im)= im-mmax

				size_t	nth;//number of theta points
				size_t	nph;//number of phi points
				size_t	max;//range of integers given by idx(l,m) given by gsl

				//get Ylm(th,ph)
				inline	comp_t	operator()	(long_t m,long_t l,size_t iph,size_t ith)const noexcept//note Y(l,-m)=(-1)^m conj[Y*(l,m)]
				{
					if(m>=0)
					{
						size_t	im=size_t(+m)*nph+iph;
						size_t 	it= idx(l,m)*nth+ith;
						return	comp_t{vpr[im], vpi[im]}*vth[it];
					}else
					{
						size_t	im=size_t(-m)*nph+iph;
						size_t	it= idx(l,-m)*nth+ith;
						return	comp_t{vpr[im],-vpi[im]}*vth[it]*(m%2?-1.:1.);
					}
				}
                                inline  size_t  idx             (size_t l,size_t m)const noexcept
                                {
                                        return  gsl_sf_legendre_array_index(l,m);
                                }
				//dump values
				template<class axis_t,class axis_p>
				void	initialize	(const axis_t& axis_th,const axis_p& axis_ph,const size_t _nth,const size_t _nph)
				{
					//free if already allocated
					if(_th){delete[] _th;_th=0;}
					if(_ph){delete[] _ph;_ph=0;}
					if(vth){delete[] vth;vth=0;}
                                        if(vpr){delete[] vpr;vpr=0;}
                                        if(vpi){delete[] vpi;vpi=0;}
					//set size parameters
					max=gsl_sf_legendre_array_n(lmax);
					nph=_nph;
					nth=_nth;
					//allocate memory
					_th=new (std::nothrow) double[nth];
					_ph=new (std::nothrow) double[nph];
					vth=new (std::nothrow) double[nth*max];
					vpr=new (std::nothrow) double[nph*(mmax+1ul)];
					vpi=new (std::nothrow) double[nph*(mmax+1ul)];
					double* tmp = new (std::nothrow) double[max];//plm work array
					if(_th==0||_ph==0||vth==0||vpr==0||vpi==0||tmp==0)//turn bad alloc to QPC error
					{
						throw qpc::runtime_error_t<std::string,void>("bad alloc in spherical_harmonics::disp.");
					}
					//set axis value
					for(size_t i=0;i<nth;++i){_th[i]=axis_th(i);}
					for(size_t i=0;i<nph;++i){_ph[i]=axis_ph(i);}
					//set exp value
					for(size_t im=0;im<=mmax;++im)//range of m = 0:mmax
					{
						for(size_t iph=0;iph<nph;++iph)
						{
							vpr[iph+im*nph]=gsl_sf_cos(im*_ph[iph]);
							vpi[iph+im*nph]=gsl_sf_sin(im*_ph[iph]);
						}
					}
					//set plm value
					for(size_t ith=0;ith<nth;++ith)
					{
						if constexpr(mode==0)
						{
							gsl_sf_legendre_array_e(GSL_SF_LEGENDRE_SPHARM,lmax,gsl_sf_cos(_th[ith]),cphase,tmp);
						}else
						{
							gsl_sf_legendre_array_e(GSL_SF_LEGENDRE_SPHARM,lmax,           _th[ith] ,cphase,tmp);
						}
						for(size_t l=0;l<mmax+ldim;++l)
						for(size_t m=0;m<=l;++m)
						{
							auto _id=idx(l,m);
							vth[ith+_id*nth]=tmp[_id];
						}
					}//end of plm value*/

					//deallocation
					delete[] tmp;
				}//end of initialize	
				explicit disp():_th(0),_ph(0),vth(0),vpr(0),vpi(0)
				{
					//do allocation in 'initialize'
				}
				~disp()noexcept
				{
					if(_th)delete[] _th;
					if(_ph)delete[] _ph;
					if(vth)delete[] vth;
					if(vpr)delete[] vpr;
					if(vpi)delete[] vpi;
				}
				
		};//end of disp
	};//end of spherical_harmonics
}//end of bss
}//end of qpc
