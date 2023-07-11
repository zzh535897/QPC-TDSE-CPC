#pragma once

//require <spherical/operator.h>
//require <spherical/basisfunc.h>
//require <spherical/intrinsic.h>

template<class dims_t>
struct	observable_z //f(r)*costh
{
	operator_rsym<dims_t>	oper_r;
	operator_y10<dims_t>	oper_y;	

	inline	void	initialize	(const integrator_radi<dims_t>& radi,const integrator_angu<dims_t>& angu)noexcept
	{
		radi.integrate_0r0(oper_r,[](double r){return r;});
		oper_y.initialize(angu);
	}//end of initialize

	template<class func_t>
	inline  void    initialize      (const integrator_radi<dims_t>& radi,const integrator_angu<dims_t>& angu,const func_t& func)noexcept
        {
		radi.integrate_0r0(oper_r,func);
                oper_y.initialize(angu);
        }//end of initialize

	inline	auto	observe		(const comp_t* lhsw,const comp_t* rhsw)noexcept
	{
		comp_t	obsv	=	zero<comp_t>;
		#pragma omp parallel for collapse(2) reduction(+:obsv)
		for(size_t im=0;im<dims_t::m_dims;++im)
		for(size_t il=0;il<dims_t::l_dims;++il)
		{
			auto* _lhsw=lhsw+dims_t::n_dims*dims_t::l_dims*im+dims_t::n_dims*il;
			auto* _rhsw=rhsw+dims_t::n_dims*dims_t::l_dims*im;
			if(il>0)
			obsv	+=	oper_r.observe_rsub
			(
				_lhsw,
				_rhsw+dims_t::n_dims*(il-1)
			)*oper_y(im,il-1);
			if(il+1ul<dims_t::l_dims)
			obsv	+=	oper_r.observe_rsub
			(
				_lhsw,
				_rhsw+dims_t::n_dims*(il+1)		
			)*oper_y(im,il);
		}
		return 	obsv;
	}//end of observe			
};//end of observable_z

template<class dims_t>
struct  observable_r //f(r)*sinth*cosph or f(r)*sinth*sinph
{
	operator_rsym<dims_t>  	oper_r;
	operator_y11<dims_t,+1> oper_yp;
	operator_y11<dims_t,-1> oper_ym;

	inline  void    initialize      (const integrator_radi<dims_t>& radi,const integrator_angu<dims_t>& angu)noexcept
        {
		radi.integrate_0r0(oper_r,[](double r){return r;});
                oper_yp.initialize(angu);
                oper_ym.initialize(angu);
        }//end of initialize

	template<class func_t>
	inline  void    initialize      (const integrator_radi<dims_t>& radi,const integrator_angu<dims_t>& angu,const func_t& func)noexcept
        {
                radi.integrate_0r0(oper_r,func);
                oper_yp.initialize(angu);
                oper_ym.initialize(angu);
        }//end of initialize

	//<x> = (<rp>+<rm>)/2
	//<y> = (<rp>-<rm>)/2I

	inline  auto    observe_rm    	(const comp_t* lhsw,const comp_t* rhsw)noexcept
	{
		comp_t  obsv    =       zero<comp_t>;
                #pragma omp parallel for collapse(2) reduction(+:obsv)
                for(size_t im=0;im<dims_t::m_dims-1;++im)
                for(size_t il=0;il<dims_t::l_dims;++il)
		{
			auto* _lhsw=lhsw+dims_t::n_dims*dims_t::l_dims*im+dims_t::n_dims*il;
			auto* _rhsw=rhsw+dims_t::n_dims*dims_t::l_dims*(im+1);
			size_t im2,il2;
			if(dims_t::template in_check<+1ul,+1ul>(im,il,im2,il2))
			obsv    +=      oper_r.observe_rsub
                        (
                                _lhsw,
                                _rhsw+dims_t::n_dims*il2
                        )*oper_ym.outer(im,il);
			if(dims_t::template in_check<+1ul,-1ul>(im,il,im2,il2))
			obsv    +=      oper_r.observe_rsub
                        (
                                _lhsw,
                                _rhsw+dims_t::n_dims*il2
                        )*oper_ym.inner(im,il);		
		}
		return 	obsv;
	}//end of observe_rm
	
	inline  auto    observe_rp      (const comp_t* lhsw,const comp_t* rhsw)noexcept
        {
                comp_t  obsv    =       zero<comp_t>;
                #pragma omp parallel for collapse(2) reduction(+:obsv)
                for(size_t im=1;im<dims_t::m_dims;++im)
                for(size_t il=0;il<dims_t::l_dims;++il)
                {
			auto* _lhsw=lhsw+dims_t::n_dims*dims_t::l_dims*im+dims_t::n_dims*il;
			auto* _rhsw=rhsw+dims_t::n_dims*dims_t::l_dims*(im-1);
			size_t im2,il2;
                        if(dims_t::template in_check<-1ul,+1ul>(im,il,im2,il2))
			obsv    +=      oper_r.observe_rsub
                        (
                                _lhsw,
                                _rhsw+dims_t::n_dims*il2
                        )*oper_yp.inner(im,il);
			if(dims_t::template in_check<-1ul,-1ul>(im,il,im2,il2))
			obsv    +=      oper_r.observe_rsub
                        (
                                _lhsw,
                                _rhsw+dims_t::n_dims*il2
                        )*oper_yp.outer(im,il);	
                }
                return  obsv;
        }//end of observe_rp

};//end of observable_r

template<class dims_t>
struct  observable_zz 	//f(r)*costh^2
{
	operator_rsym<dims_t>	oper_r;
	operator_eta2<dims_t>	oper_y;	

	template<class func_t>
	inline  void    initialize      (const integrator_radi<dims_t>& radi,const integrator_angu<dims_t>& angu,const func_t& func)noexcept
        {
		radi.integrate_0r0(oper_r,func);
                oper_y.initialize(angu);
        }//end of initialize

	inline	auto	observe		(const comp_t* lhsw,const comp_t* rhsw)noexcept
	{
		comp_t	obsv	=	zero<comp_t>;
		#pragma omp parallel for collapse(2) reduction(+:obsv)
		for(size_t im=0;im<dims_t::m_dims;++im)
		for(size_t il=0;il<dims_t::l_dims;++il)
		{
			auto* _lhsw=lhsw+dims_t::n_dims*dims_t::l_dims*im+dims_t::n_dims*il;
			auto* _rhsw=rhsw+dims_t::n_dims*dims_t::l_dims*im;
			obsv	+=	oper_r.observe_rsub
			(
				_lhsw,
				_rhsw+dims_t::n_dims*il
			)*oper_y.sh0(im,il);
			if(il>1ul)
			obsv	+=	oper_r.observe_rsub
			(
				_lhsw,
				_rhsw+dims_t::n_dims*(il-2ul)
			)*oper_y.sh2(im,il-2ul);
			if(il+2ul<dims_t::l_dims)
			obsv	+=	oper_r.observe_rsub
			(
				_lhsw,
				_rhsw+dims_t::n_dims*(il+2ul)		
			)*oper_y.sh2(im,il);
		}
		return 	obsv;
	}//end of observe			
};//end

template<class dims_t>
struct	observable_molec final
{
	operator_rsym<dims_t>	r0;
	operator_rsym<dims_t>	r1;

	operator_y10 <dims_t>	cos1;
	operator_eta2<dims_t>	cos2;

	inline	void	initialize	(const integrator_radi<dims_t>& radi,const integrator_angu<dims_t>& angu)noexcept
	{
		radi.integrate_0r0(r0,[](double r){return 1.0;});
		radi.integrate_0r0(r1,[](double r){return r;});

		cos1.initialize(angu);
		cos2.initialize(angu);
	}

	inline  void    initialize      (const integrator_radi<dims_t>& radi,const integrator_angu<dims_t>& angu,double rmin,double rmax)noexcept
        {
                radi.integrate_0r0(r0,[&](double r){return r>rmin&&r<rmax?1.0:0.0;});
                radi.integrate_0r0(r1,[&](double r){return r>rmin&&r<rmax?r  :0.0;});

                cos1.initialize(angu);
                cos2.initialize(angu);
        }
	
	using	coef_t	=	recvec<comp_t,recidx<dims_t::m_dims,dims_t::l_dims,dims_t::n_dims>>;

	inline	auto	observe_r0cos0	(const comp_t* w)noexcept
	{
		double	obsv	=	zero<double>;
		#pragma omp parallel for reduction(+:obsv)
		for(size_t is=0;is<dims_t::n_leng;is+=dims_t::n_dims)
		{
			obsv	+=	r0.observe_rsub(w+is,w+is)[0];
		}
		return	obsv;
	}

	inline	auto	observe_r0cos1	(const comp_t* w)noexcept
	{
		double	obsv	=	zero<double>;
		#pragma omp parallel for collapse(2) reduction(+:obsv)
		for(size_t im=0;im<dims_t::m_dims;++im)
		for(size_t il=0;il<dims_t::l_dims;++il)
		{
			auto* _lhsw=w+dims_t::n_dims*dims_t::l_dims*im+dims_t::n_dims*il;
			auto* _rhsw=w+dims_t::n_dims*dims_t::l_dims*im;
			if(il>0)
			obsv	+=	r0.observe_rsub
			(
				_lhsw,
				_rhsw+dims_t::n_dims*(il-1)
			)[0]*cos1(im,il-1);
			if(il+1ul<dims_t::l_dims)
			obsv	+=	r0.observe_rsub
			(
				_lhsw,
				_rhsw+dims_t::n_dims*(il+1)		
			)[0]*cos1(im,il);
		}
		return	obsv;
	}
	
	inline	auto	observe_r0cos2	(const comp_t* w)noexcept
	{
		double	obsv	=	zero<double>;
		#pragma omp parallel for collapse(2) reduction(+:obsv)
		for(size_t im=0;im<dims_t::m_dims;++im)
		for(size_t il=0;il<dims_t::l_dims;++il)
		{
			auto* _lhsw=w+dims_t::n_dims*dims_t::l_dims*im+dims_t::n_dims*il;
			auto* _rhsw=w+dims_t::n_dims*dims_t::l_dims*im;
			obsv	+=	r0.observe_rsub
			(
				_lhsw,
				_rhsw+dims_t::n_dims*il
			)[0]*cos2.sh0(im,il);
			if(il>1ul)
			obsv	+=	r0.observe_rsub
			(
				_lhsw,
				_rhsw+dims_t::n_dims*(il-2ul)
			)[0]*cos2.sh2(im,il-2ul);
			if(il+2ul<dims_t::l_dims)
			obsv	+=	r0.observe_rsub
			(
				_lhsw,
				_rhsw+dims_t::n_dims*(il+2ul)		
			)[0]*cos2.sh2(im,il);
		}	
		return  obsv;
	}
};//end of observable_mol


template<class dims_t>
struct  observable_prolate_z
{
	public:
		using	xmat_t	=	operator_xsub<dims_t,2ul>;
		using	eta1_t	=	operator_eta1<dims_t>;
		using	eta3_t	=	operator_eta3<dims_t>;

		xmat_t	xmat;
		eta1_t	eta1;
		eta3_t	eta3;
		double	coef;

		inline	void	initialize(const integrator_radi<dims_t>& radi,const integrator_angu<dims_t>& angu,const double dist)noexcept
		{
			xmat.initialize(radi,[](const double x){return x;},    0ul);
			xmat.initialize(radi,[](const double x){return x*x*x;},1ul);
			eta1.initialize(angu);
			eta3.initialize(angu);
			coef=dist*dist*dist*dist/16.0;
		}//end of initialize

		inline	auto	observe(const comp_t* lhsw,const comp_t* rhsw)const noexcept				
		{
			static_assert(dims_t::l_dims>3);
			auto	obsv	=	zero<comp_t>;
			#pragma omp parallel for simd collapse(2) reduction(+:obsv)
			for(size_t im=0;im<dims_t::m_dims;++im)
			for(size_t il=0;il<dims_t::l_dims;++il)
			{
				size_t	od	=	dims_t::od_m(im);
				auto* 	_lhsw	=	lhsw+dims_t::n_dims*dims_t::l_dims*im+dims_t::n_dims*il;	
				auto* 	_rhsw	=	rhsw+dims_t::n_dims*dims_t::l_dims*im+dims_t::n_dims*il;
				auto  	_ksi1	=	xmat.sub(od,0);
				auto  	_ksi3	=	xmat.sub(od,1);
				if(il>2)
				{
					obsv	-=	intrinsic::symb_prj_vecd<dims_t::n_dims,dims_t::n_elem>(_ksi1.data,_lhsw,_rhsw-dims_t::n_dims*3ul)*eta3.sh3(im,il-3ul);
				}
				if(il>0)
				{
					obsv	-=	intrinsic::symb_prj_vecd<dims_t::n_dims,dims_t::n_elem>(_ksi1.data,_lhsw,_rhsw-dims_t::n_dims)*eta3.sh1(im,il-1ul);
					obsv	+=	intrinsic::symb_prj_vecd<dims_t::n_dims,dims_t::n_elem>(_ksi3.data,_lhsw,_rhsw-dims_t::n_dims)*eta1.sh1(im,il-1ul);
				}
				if(il<dims_t::l_dims-1ul)
				{
					obsv	+=	intrinsic::symb_prj_vecd<dims_t::n_dims,dims_t::n_elem>(_ksi3.data,_lhsw,_rhsw+dims_t::n_dims)*eta1.sh1(im,il);
					obsv	-=	intrinsic::symb_prj_vecd<dims_t::n_dims,dims_t::n_elem>(_ksi1.data,_lhsw,_rhsw+dims_t::n_dims)*eta3.sh1(im,il);
				}
				if(il<dims_t::l_dims-3ul)
				{
					obsv	-=	intrinsic::symb_prj_vecd<dims_t::n_dims,dims_t::n_elem>(_ksi1.data,_lhsw,_rhsw+dims_t::n_dims*3ul)*eta3.sh3(im,il);
				}
			}
			return 	obsv*coef;
		}//end of observe
};//end of observable_z

