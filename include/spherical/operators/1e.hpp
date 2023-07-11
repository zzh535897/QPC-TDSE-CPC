#pragma once

//require <libraries/support_std.h>
//require <libraries/support_omp.h>
//require <libraries/support_avx.h>

//require <resources/arena.h>
//require <utilities/error.h>
//require <utilities/recid.h>

//require <algorithm/ed.h>


//================================================================================================================================
//
//						Radial SubMatrix (Spherical)
//
//================================================================================================================================
template<class dims_t>
struct	operator_rsub:public recvec<double,recidx<dims_t::n_dims,dims_t::n_elem>>//data[irow*dims_t::n_elem+icol]
{
	protected:	
		rsrc_t*	rsrc;

	public:
		//constructor
		operator_rsub(operator_rsub const&)=delete;
		operator_rsub& operator=(operator_rsub const&)=delete;
		operator_rsub& operator=(operator_rsub&&)=delete;

		explicit operator_rsub(rsrc_t* _rsrc=global_default_arena_ptr):rsrc(_rsrc)
		{
			if(rsrc!=0)
			{
				rsrc->acquire((void**)&(this->data),sizeof(double)*dims_t::n_rsub,align_val_avx3);
			}else
			{
				throw	errn_t("null arena parsed in when constructing operator_rsub.");
			}
		}
		operator_rsub(operator_rsub&& rhs)noexcept
		{
			this->data=rhs.data;
			rsrc=rhs.rsrc;
			rhs.rsrc=0;
		}		
		//destructor
		virtual ~operator_rsub()noexcept
		{
			if(rsrc!=0)
			{
				rsrc->release(this->data);
			}
		}
};//end of operator_rsub

template<class dims_t>
struct	operator_rsym:public operator_rsub<dims_t>
{
	public:
		using	coef_t	=	coefficient_view<dims_t>;

		explicit operator_rsym(rsrc_t* _rsrc=global_default_arena_ptr):operator_rsub<dims_t>(_rsrc){}

		inline	auto	observe_rsub	(const comp_t* w1,const comp_t* w2)const noexcept
		{
			return 	intrinsic::symb_prj_vecd<dims_t::n_dims,dims_t::n_elem>(this->data,w1,w2);
		}//end of observe_rsub
		inline	void	observe		(const comp_t* w1,double& norm)const noexcept
		{//set initial value of norm by yourself
			#pragma omp for collapse(2) reduction(+:norm)
			for(size_t im=0;im<dims_t::m_dims;++im)
			for(size_t il=0;il<dims_t::l_dims;++il)
			{
				auto	wave	=	w1+dims_t::n_dims*dims_t::l_dims*im+dims_t::n_dims*il;
				norm            +=      observe_rsub(wave,wave)[0];
			}
		}//end of observe
		inline	void	observe		(const comp_t* w1,const comp_t* w2,comp_t& proj)const noexcept
		{//set initial value of proj by yourself
			#pragma omp for collapse(2) reduction(+:proj)
			for(size_t im=0;im<dims_t::m_dims;++im)
			for(size_t il=0;il<dims_t::l_dims;++il)
			{
				auto	wave1	=	w1+dims_t::n_dims*dims_t::l_dims*im+dims_t::n_dims*il;
				auto	wave2	=	w2+dims_t::n_dims*dims_t::l_dims*im+dims_t::n_dims*il;
				proj		+=	observe_rsub(wave1,wave2);
			}
		}//end of observe
};//end of operator_rsym

template<class dims_t>
struct	operator_rasy:public operator_rsub<dims_t>
{
	public:
		using	coef_t	=	coefficient_view<dims_t>;

		explicit operator_rasy(rsrc_t* _rsrc=global_default_arena_ptr):operator_rsub<dims_t>(_rsrc){}

		inline	auto	observe_rsub	(const comp_t* w1,const comp_t* w2)const noexcept
		{
			return 	intrinsic::asyb_prj_vecd<dims_t::n_dims,dims_t::n_elem>(this->data,w1,w2);
		}//end of observe_rsub
		inline	void	observe		(const comp_t* w1,double& norm)const noexcept
		{
			#pragma omp for collapse(2) reduction(+:norm)
			for(size_t im=0;im<dims_t::m_dims;++im)
			for(size_t il=0;il<dims_t::l_dims;++il)
			{
				auto	wave	=	w1+dims_t::n_dims*dims_t::l_dims*im+dims_t::n_dims*il;
				norm            +=      observe_rsub(wave,wave)[0];
			}
		}//end of observe
		inline	void	observe		(const comp_t* w1,const comp_t* w2,comp_t& proj)const noexcept
		{
			#pragma omp for collapse(2) reduction(+:proj)
			for(size_t im=0;im<dims_t::m_dims;++im)
			for(size_t il=0;il<dims_t::l_dims;++il)
			{
				auto	wave1	=	w1+dims_t::n_dims*dims_t::l_dims*im+dims_t::n_dims*il;
				auto	wave2	=	w2+dims_t::n_dims*dims_t::l_dims*im+dims_t::n_dims*il;
				proj		+=	observe_rsub(wave1,wave2);
			}
		}//end of observe
};//end of operator_rasy

template<class dims_t>
struct	operator_rpow0:public operator_rsym<dims_t>
{
	public:
		explicit operator_rpow0(rsrc_t* _rsrc=global_default_arena_ptr):operator_rsym<dims_t>(_rsrc){}

		inline	void	initialize(const integrator_radi<dims_t>& base)noexcept
		{
			base.integrate_0r0(*this,[](double r){return 1.0;});
		}
};//end of operator_rpow0
template<class dims_t>
struct	operator_rpow1:public operator_rsym<dims_t>
{
	public:
		explicit operator_rpow1(rsrc_t* _rsrc=global_default_arena_ptr):operator_rsym<dims_t>(_rsrc){}

		inline	void	initialize(const integrator_radi<dims_t>& base)noexcept
		{
			base.integrate_0r0(*this,[](double r){return r;});
		}
};//end of operator_rpow1
template<class dims_t>
struct	operator_rpow2:public operator_rsym<dims_t>
{
	public:
		explicit operator_rpow2(rsrc_t* _rsrc=global_default_arena_ptr):operator_rsym<dims_t>(_rsrc){}

		inline	void	initialize(const integrator_radi<dims_t>& base)noexcept
		{
			base.integrate_0r0(*this,[](double r){return r*r;});
		}
};//end of operator_rpow2
template<class dims_t>
struct	operator_rpow3:public operator_rsym<dims_t>
{
	public:
		explicit operator_rpow3(rsrc_t* _rsrc=global_default_arena_ptr):operator_rsym<dims_t>(_rsrc){}

		inline	void	initialize(const integrator_radi<dims_t>& base)noexcept
		{
			base.integrate_0r0(*this,[](double r){return r*r*r;});
		}
};//end of operator_rpow3
template<class dims_t>
struct	operator_rinv1:public operator_rsym<dims_t>
{
	public:
		explicit operator_rinv1(rsrc_t* _rsrc=global_default_arena_ptr):operator_rsym<dims_t>(_rsrc){}

		inline	void	initialize(const integrator_radi<dims_t>& base)noexcept
		{
			base.integrate_0r0(*this,[](double r){return 1./(r+1e-50);});
		}
};//end of operator_rinv1

template<class dims_t>
struct	operator_rdif1:public operator_rasy<dims_t>
{
	public:
		explicit operator_rdif1(rsrc_t* _rsrc=global_default_arena_ptr):operator_rasy<dims_t>(_rsrc){}

		inline	void	initialize(const integrator_radi<dims_t>& base)noexcept
		{
			base.integrate_0r1(*this,[](double r){return 1.0;});
		}
};//end of operator_rdif1
//================================================================================================================================
//
//						Radial SubMatrix (Spheroidal)
//
//================================================================================================================================
template<class dims_t,size_t n_subs>
struct	operator_xsub:public recvec<double,recidx<2ul,n_subs,dims_t::n_dims,dims_t::n_elem>>//2ul for odd or even
{	
	private:
		rsrc_t*	rsrc;

	public:
		operator_xsub(operator_xsub const&)=delete;
		operator_xsub& operator=(operator_xsub const&)=delete;
		operator_xsub& operator=(operator_xsub &&)=delete;

		explicit operator_xsub(rsrc_t* _rsrc=global_default_arena_ptr):rsrc(_rsrc)
		{
			if(rsrc)
			{
				rsrc->acquire((void**)&this->data,sizeof(double)*this->dim.n_leng,align_val_avx3);
			}
		}
		operator_xsub(operator_xsub && rhs)noexcept
		{
			this->data=rhs.data;
			this->rsrc=rhs.rsrc;
			rhs.rsrc=0;
		}
		virtual	~operator_xsub()noexcept
		{
			if(rsrc)
			{
				rsrc->release(this->data);
			}
		}

		template<class func_t>
		inline	void	initialize	(const integrator_radi<dims_t>& base,const func_t& func0,const size_t id)noexcept
		{
			//0=even
			base.integrate_0r0(this->sub(0ul,id),func0);
			//1=odd
			base.integrate_0r0(this->sub(1ul,id),[&](double x){return func0(x)*(x-1.0)/(x+1.0);});
		}//end of initialize (<Bi,m|f(x)|Bj,m>)

		inline	void	initialize	(const integrator_radi<dims_t>& base,const size_t id)noexcept
		{
			//0=even
			base.integrate_bde(this->sub(0ul,id));
			//1=odd
			base.integrate_bdo(this->sub(1ul,id));
		}//end of initialize (<Bi,m|sqrt(x^2-1)(d/dx)sqrt(x^2-1)|Bj,m>)
};//end of operator_xsub
//================================================================================================================================
//
//							Angular  SubMatrix
//
//================================================================================================================================
template<class dims_t>
struct	operator_angu:public recvec<double,recidx<dims_t::m_dims,dims_t::l_dims>>//data[im*dims_t::l_dims+il]
{
	protected:
		rsrc_t*	rsrc;		

	public:
		//constructor
		operator_angu(operator_angu const&)=delete;
		operator_angu& operator=(operator_angu const&)=delete;
		operator_angu& operator=(operator_angu &&)=delete;

		explicit operator_angu(rsrc_t* _rsrc=global_default_arena_ptr):rsrc(_rsrc)
		{
			if(rsrc!=0)
			{
				rsrc->acquire((void**)&(this->data),sizeof(double)*dims_t::m_dims*dims_t::l_dims,align_val_avx3);
			}else
			{
				throw	errn_t{"null arena parsed in when constructing operator_angu."};
			}
		}
		operator_angu(operator_angu&& rhs)noexcept
		{
			this->data=rhs.data;
			this->rsrc=rhs.rsrc;
			rhs.rsrc=0;
		}
		//destructor
		virtual ~operator_angu()noexcept
		{
			if(rsrc!=0)
			{
				rsrc->release(this->data);
			}
		}	
};//end of operator_angu

template<class dims_t>
struct	operator_y10:public operator_angu<dims_t>
{
	public:
		explicit operator_y10(rsrc_t* _rsrc=global_default_arena_ptr):operator_angu<dims_t>(_rsrc){}

		inline	void	initialize(const integrator_angu<dims_t>& base)noexcept
		{
			base.integrate_cos(*this);
		}//end of initialize

		static constexpr size_t m_diag	=	1ul;

		template<class yeig_t,class yprj_t>
		inline	void	solve_eigen(const comp_t& mult,const yeig_t& yeig,const yprj_t& yprj)
		{
			comp_t* mem = new (std::nothrow) comp_t[dims_t::l_dims*(m_diag+1ul)*omp_get_max_threads()];
			if(mem==0)
			{
				throw errn_t("allocation error in solve_eigen in Y10.");
			}

			int info=0;
			#pragma omp parallel reduction(+:info)
			{
				comp_t* ymat = mem + dims_t::l_dims*(m_diag+1ul)*omp_get_thread_num(); //to hold the lower triangular part of Y in lapack col major
				#pragma omp for
				for(size_t im=0;im<dims_t::m_dims;++im)
				{
					for(size_t il=0;il<dims_t::l_dims;++il)
					{
						ymat[2*il  ]=	0.0;
						ymat[2*il+1]=	(*this)(im,il)*mult;
					}
					info	+=	impl_ed_simd::ev_zhb<dims_t::l_dims,m_diag>(ymat,yeig(im).data,yprj(im).data);
				}
			}
			delete[] mem;
			if(info)
			{
				throw  errn_t("LAPACKE error in solve_eigen of Y10.");
			}
		}//end of solve_eigen
};//end of operator_y10

template<class dims_t,int sign=-1>
struct	operator_y11
{
	public:
		operator_angu<dims_t> outer;
		operator_angu<dims_t> inner;

		explicit operator_y11(rsrc_t* _rsrc=global_default_arena_ptr):
		outer(_rsrc),
		inner(_rsrc){}

		inline	void	initialize(const integrator_angu<dims_t>& base)noexcept
		{
			if constexpr(sign<0)//sin*exp(-iph)
			{
				base.integrate_sinexp1(outer);
				base.integrate_sinexp2(inner);
			}
			if constexpr(sign>0)//sin*exp(+iph)
			{
				base.integrate_sinexp3(outer);
                                base.integrate_sinexp4(inner);	
			}
		}//end of initialize
	
		static constexpr size_t m_diag	=	dims_t::l_dims+2ul;

		template<class zeig_t,class zprj_t>
		inline	void	solve_eigen(const comp_t& mult,const zeig_t& zeig,const zprj_t& zprj)
		{
			comp_t*	zmat	=	new (std::nothrow) comp_t[dims_t::m_dims*dims_t::l_dims*(m_diag+1ul)];
			if(zmat==0)
			{
				throw errn_t("allocation error in solve_eigen in Y11.");
			}
			
			#pragma omp parallel for
			for(size_t im=0;im<dims_t::m_dims;++im)//to fill zmat's lower pannel
			for(size_t il=0;il<dims_t::l_dims;++il)
			{
				size_t	iml	=	im*dims_t::l_dims+il;
				size_t	jml,jm,jl;
				comp_t*	mat	=	zmat+iml*(m_diag+1ul);
				for(size_t j=0;j<m_diag+1ul;++j)
				{
					mat[j]	=	0.0;//pad with zero
				}
				if(dims_t::template in_check<+1ul,+1ul>(im,il,jm,jl))
				{
					jml	=	jm*dims_t::l_dims+jl;
					mat[jml-iml]=	outer(im,il)*mult;
				}
				if(dims_t::template in_check<+1ul,-1ul>(im,il,jm,jl))
				{
					jml	=	jm*dims_t::l_dims+jl;
					mat[jml-iml]=   inner(im,il)*mult;
				}
			}

			if(int err=impl_ed_simd::ev_zhb<dims_t::m_dims*dims_t::l_dims,m_diag>(zmat,zeig.data,zprj.data))//could be time-costly
			{
				throw errn_t("LAPACKE error in solve_eigen in Y11.");
			}
			delete[] zmat;
			if(int err=converter::catagorize<dims_t>(zeig.data,zprj.data))
			{
				throw errn_t("unexpected results from diagonalization of Y11, error="+to_string(err));
			}
		}//end of solve_eigen

};//end of operator_y11

template<class dims_t>
struct	operator_q10:public operator_angu<dims_t>
{
	public:
		explicit operator_q10(rsrc_t* _rsrc=global_default_arena_ptr):operator_angu<dims_t>(_rsrc){}

		inline	void	initialize(const integrator_angu<dims_t>& base)noexcept
		{
			base.integrate_sin2dcos(*this);
		}//end of initialize

		static constexpr size_t m_diag	=	1ul;

		template<class yeig_t,class yprj_t>
		inline	void	solve_eigen(const comp_t& mult,const yeig_t& yeig,const yprj_t& yprj)
		{
			comp_t* mem = new (std::nothrow) comp_t[dims_t::l_dims*(m_diag+1ul)*omp_get_max_threads()];
			if(mem==0)
			{
				throw errn_t("allocation error in solve_eigen in Q10.");
			}
			int info=0;
			#pragma omp parallel reduction(+:info)
			{
				comp_t*	ymat	=	mem + dims_t::l_dims*(m_diag+1ul)*omp_get_thread_num(); //to hold the lower triangular part of Y in lapack col major.
				#pragma omp for
				for(size_t im=0;im<dims_t::m_dims;++im)
				{
					for(size_t il=0;il<dims_t::l_dims;++il)
					{
						ymat[2*il  ]=	0.0;
						ymat[2*il+1]=	(*this)(im,il)*mult;
					}
					info	+=	impl_ed_simd::ev_zhb<dims_t::l_dims,m_diag>(ymat,yeig(im).data,yprj(im).data);
				}
			}
			if(info)
			{
				throw  errn_t("LAPACKE error in solve_eigen of Q10.");
			}
		}//end of solve_eigen
};//end of operator_q10

template<class dims_t,int sign=-1>
struct	operator_q11
{
	public:
		operator_angu<dims_t> outer;
		operator_angu<dims_t> inner;

		explicit operator_q11(rsrc_t* _rsrc=global_default_arena_ptr):
		outer(_rsrc),
		inner(_rsrc){}

		inline	void	initialize(const integrator_angu<dims_t>& base)noexcept
		{
			if constexpr(sign<0)//-cos*Lm-sin*exp-*(1-Lz) note this implementation differs a sign from the standard Q1m defined in QPC mannual
			{
				base.integrate_coslm1(outer);
				base.integrate_coslm2(inner);
			}
			if constexpr(sign>0)// cos*Lp-sin*exp+*(1+Lz)
			{
				base.integrate_coslp1(outer);
                                base.integrate_coslp2(inner);
			}
		}

		static constexpr size_t	m_diag	=	dims_t::l_dims+2ul;

		template<class zeig_t,class zprj_t>
		inline	void	solve_eigen(const comp_t& mult,const zeig_t& zeig,const zprj_t& zprj)
		{
			comp_t*	zmat	=	new (std::nothrow) comp_t[dims_t::m_dims*dims_t::l_dims*(m_diag+1ul)];
			if(zmat==0)
			{
				throw errn_t("allocation error in solve_eigen in Q11.");
			}
			#pragma omp parallel for
			for(size_t im=0;im<dims_t::m_dims;++im)//to fill zmat
			for(size_t il=0;il<dims_t::l_dims;++il)
			{
				size_t	iml	=	im*dims_t::l_dims+il;
				size_t	jml,jm,jl;
				comp_t*	mat	=	zmat+iml*(m_diag+1ul);
				for(size_t j=0;j<m_diag+1ul;++j)
				{
					mat[j]	=	0.0;//pad with zero
				}
				if(dims_t::template in_check<+1ul,+1ul>(im,il,jm,jl))
				{
					jml	=	jm*dims_t::l_dims+jl;
					if constexpr(sign<0)//the implementation of q1m in <basisfunc.h> differs a sign from the definition in qpc mannal
					mat[jml-iml]=	-outer(im,il)*mult;
					if constexpr(sign>0)
					mat[jml-iml]=	 outer(im,il)*mult;
				}
				if(dims_t::template in_check<+1ul,-1ul>(im,il,jm,jl))
				{
					jml	=	jm*dims_t::l_dims+jl;
					if constexpr(sign<0)//the implementation of q1m in <basisfunc.h> differs a sign from the definition in qpc mannal
					mat[jml-iml]=   -inner(im,il)*mult;
					if constexpr(sign>0)
					mat[jml-iml]=    inner(im,il)*mult;	
				}
			}
			if(int err=impl_ed_simd::ev_zhb<dims_t::m_dims*dims_t::l_dims,m_diag>(zmat,zeig.data,zprj.data))//could be time-costly
			{
				throw errn_t("LAPACKE error in solve_eigen in Q11.");
			}
			if(int err=converter::catagorize<dims_t>(zeig.data,zprj.data))
			{
				throw errn_t("unexpected results from diagonalization of Q11, error="+to_string(err));
			}
		}//end of solve_eigen
};//end of operator_q11

template<class dims_t>
struct	operator_eta1
{
	public:
		operator_angu<dims_t> sh1;
	
		inline	void	initialize(const integrator_angu<dims_t>& base)noexcept
		{
			base.integrate_eta1_sh1(sh1);
		}

		explicit operator_eta1(rsrc_t* _rsrc=global_default_arena_ptr):
		sh1(_rsrc){}
};//end of operator_eta1

template<class dims_t>
struct	operator_eta2
{
	public:
		operator_angu<dims_t> sh0;
		operator_angu<dims_t> sh2;
		
		inline	void	initialize(const integrator_angu<dims_t>& base)noexcept
		{
			base.integrate_eta2_sh0(sh0);
			base.integrate_eta2_sh2(sh2);
		}

		explicit operator_eta2(rsrc_t* _rsrc=global_default_arena_ptr):
		sh0(_rsrc),
		sh2(_rsrc){}
};//end of operator_eta2

template<class dims_t>
struct  operator_eta3
{
        public:
                operator_angu<dims_t> sh1;
                operator_angu<dims_t> sh3;

                inline  void    initialize(const integrator_angu<dims_t>& base)noexcept
                {
                        base.integrate_eta3_sh1(sh1);
                        base.integrate_eta3_sh3(sh3);
                }

		explicit operator_eta3(rsrc_t* _rsrc=global_default_arena_ptr):
		sh1(_rsrc),
		sh3(_rsrc){}
};//end of operator_eta3

template<class dims_t>
struct	operator_chi1
{
	public:
		operator_angu<dims_t> sh1;

		inline	void	initialize(const integrator_angu<dims_t>& base)noexcept
		{
			base.integrate_sin2dcos(sh1);
		}

		explicit operator_chi1(rsrc_t* _rsrc=global_default_arena_ptr):
		sh1(_rsrc){}
};//end of operator_chi1
//================================================================================================================================
//
//							Hamiltonian  Matrix
//
//================================================================================================================================
template<class dims_t>
struct	operator_hc
{
	protected:
		double*	data;	//will be arranged in squeezed format i.e. only 0,1,...,lmax are stored.
		rsrc_t*	rsrc;

	public:
		//get the indented pointer for (im,il)-th loop
		inline	double*		sub	(const size_t im,const size_t il)noexcept	{return data+dims_t::in_l(im,il)*dims_t::n_rsub;}
		inline	double*		sub	(const size_t im,const size_t il)const noexcept	{return data+dims_t::in_l(im,il)*dims_t::n_rsub;}

		inline	double*		operator[](const size_t l)noexcept	{return data+l*dims_t::n_rsub;}
		inline	double const*	operator[](const size_t l)const noexcept{return data+l*dims_t::n_rsub;}

		//constructor
		operator_hc(operator_hc const&)=delete;
		operator_hc(operator_hc&& rhs)noexcept
		{
			data=rhs.data;
			rsrc=rhs.rsrc;
			rhs.rsrc=0;
		}
		operator_hc& operator=(operator_hc const&)=delete;
		operator_hc& operator=(operator_hc &&)=delete;	


		explicit operator_hc(rsrc_t* _rsrc=global_default_arena_ptr):rsrc(_rsrc)
              	{
                	if(rsrc!=0)
			{
                      		rsrc->acquire((void**)&data,sizeof(double)*dims_t::n_rsub*(dims_t::n_lmax+1ul),align_val_avx3);
			}else
			{
				throw	errn_t("null arena parsed in when constructing operator_hc.");
			}
              	}//end of constructor
              	virtual ~operator_hc()noexcept
               	{
                	if(rsrc!=0)
			{
				rsrc->release(data);
			}
		}//end of destructor
		template<class potn_t>
		inline	void	initialize	(const integrator_radi<dims_t>& radi,const double mass,const potn_t& potn)noexcept
		{
			#pragma omp parallel for
			for(size_t l=0;l<=dims_t::n_lmax;++l)
			{
				radi.integrate_hc(integral_result_radi<dims_t>{data+l*dims_t::n_rsub},potn,mass,l);
			}
		}//end of initialize
};//end of operator_hc

template<class dims_t,size_t n_type>
struct  operator_hb:
protected recvec<double,recidx<dims_t::m_dims,dims_t::l_dims,n_type,dims_t::n_dims,dims_t::n_elem>>
{
	static_assert(n_type==1||n_type==2,"invalid charge symmetry type.");//1= charge symmetric, 2=charge non-symmetric
	protected:
		rsrc_t*	rsrc;

	public:
		using	recvec<double,recidx<dims_t::m_dims,dims_t::l_dims,n_type,dims_t::n_dims,dims_t::n_elem>>::sub;//allow user to directly call sub
		//constructor
		operator_hb(operator_hb const&)=delete;
		operator_hb(operator_hb&& rhs)noexcept
		{
			this->data=rhs.data;
			this->rsrc=rhs.rsrc;
			rhs.rsrc=0;
		}
		operator_hb& operator=(operator_hb const&)=delete;
		operator_hb& operator=(operator_hb &&)=delete;

		explicit operator_hb(rsrc_t* _rsrc=global_default_arena_ptr):rsrc(_rsrc)
		{
			if(rsrc!=0)
                        {
				rsrc->acquire((void**)&this->data,sizeof(double)*this->dim.n_leng,align_val_avx3);	
                        }else
                        {
                                throw   errn_t("null arena parsed in when constructing operator_hb.");
                        }
		}//end of constructor
		virtual	~operator_hb()noexcept
		{
			if(rsrc!=0)
                        {
				rsrc->release(this->data);
                        }
		}//end of destructor
		template<class zadd_t>
		inline	void	initialize	
		(
			const integrator_radi<dims_t>& radi,
			const integrator_angu<dims_t>& angu,
			const double dist,const double mass,const zadd_t& zadd
		)noexcept
		{
			static_assert(n_type==1,"this function can only be called when nuclear charge is symmetric.");
			#pragma omp parallel for collapse(2)
			for(size_t im=0;im<dims_t::m_dims;++im)
			for(size_t il=0;il<dims_t::l_dims;++il)
			{
				long 	m,l;
				dims_t::in(im,il,m,l);
				radi.integrate_hb0(this->sub(im,il,_ul<0>),dist,mass,zadd,l,m);//call hb0 to do (i,i) integral
			}	
		}//end of initialize (homonuclear)
	
		template<class zadd_t,class zsub_t>
                inline  void    initialize
                (
                        const integrator_radi<dims_t>& radi,
                        const integrator_angu<dims_t>& angu,
                        const double dist,const double mass,const zadd_t& zadd,const zsub_t& zsub
                )noexcept
                {
			static_assert(n_type==2,"this function can only be called when nuclear charge is asymmetric.");
                        #pragma omp parallel for collapse(2)
                        for(size_t im=0;im<dims_t::m_dims;++im)
                        for(size_t il=0;il<dims_t::l_dims;++il)
                        {
                                long    m,l;
                                dims_t::in(im,il,m,l);
				radi.integrate_hb0(this->sub(im,il,_ul<0>),dist,mass,zadd,l,m);//call hb0 to do (i,i) integral
				radi.integrate_hb1(this->sub(im,il,_ul<1>),dist,     zsub,l,m);//call hb1 to do (i,i+1) integral
                        }
                }//end of initialize(heteronuclear)

		inline	auto	create_csrmat_homo	(const size_t im)const
		{
			static_assert(n_type==1,"this function can only be called when nuclear charge is symmetric.");
			auto	mat	=	csrmat<double,1>();
			convert_nested_dsb_to_csr<dims_t::n_diag,dims_t::n_dims,0ul,dims_t::l_dims>(mat,this->sub(im).data);//see csrmat.h
			return 	mat;	//nrvo or move
		}//end of create_csrmat_homo

		inline	auto	create_csrmat_hete	(const size_t im)const 
		{
			static_assert(n_type==2,"this function can only be called when nuclear charge is non-symmetric.");
			auto	mat	=	csrmat<double,1>();
			convert_nested_dsb_to_csr<dims_t::n_diag,dims_t::n_dims,1ul,dims_t::l_dims>(mat,this->sub(im).data);//see csrmat.h
			return 	mat;	//nrvo or move
		}//end of create_csrmat_hete

		inline	auto	create_csrmat_real	(const size_t im)const
		{
			if constexpr(n_type==1)//just a wrapper call
			{
				return 	create_csrmat_homo(im);
			}else if constexpr(n_type==2)
			{
				return 	create_csrmat_hete(im);
			}	
		}//end of create_csrmat_real
};//end of operator_hb

template<class dims_t,size_t n_pole>
struct	operator_hm final
{
	private:
		static constexpr size_t lmax	=	dims_t::n_lmax;
		//static constexpr size_t qmax	=	cmin(dims_t::n_lmax*2ul,40ul);	//maximum "l angular momentum" in multipole expansion XXX due to GSL wigner3j accuracy problem, this cannot be large.
		static constexpr size_t qmax	=	dims_t::n_lmax*2ul;	
		static constexpr size_t udim	=	dims_t::m_dims;			//maximum "m angular momentum" in multipole expansion

		static constexpr double	eps2	=	1e-10*1e-10;			//cval with norm2 less than this value will be negelected.

		using	kntc_t	=	recvec<double,recidx<       lmax+1ul,dims_t::n_dims,dims_t::n_elem>>;//integral of kinetic operator -d^2/2m+l(l+1)/2mr^2
		using	potn_t	=	recvec<double,recidx<n_pole,qmax+1ul,dims_t::n_dims,dims_t::n_elem>>;//integral of multipole potential r(>)^q/r(<)^q+1
		using	ylmr_t	=	recvec<comp_t,recidx<udim,qmax+1ul,n_pole>>;//sum_p conj{ Ymq(th[p],ph[p]) }
		using	cptr_t	=	recvec<size_t,recidx<dims_t::m_dims*dims_t::l_dims,dims_t::m_dims*dims_t::l_dims>>;//(-)^m1 sqrt(4pi(2l1+1)(2l2+1)/(2q+1)) (l1,0,q,0,l2,0)*(...)

		using	pole_t	=	std::tuple<double,std::vector<std::array<double,3>>>;//Rc, {th,ph,zc}
		using	pole_l	=	std::array<pole_t,n_pole>;
		using	buff_t	=	std::tuple<double,size_t>; //{value of c(m1l1m2l2,q),value of q}

		using	hmat_t	=	sparse_matrix_t;	//CSR3 format for diagonalization and factorization purpose
		using	desc_t	=	matrix_descr;

		static constexpr size_t n_hmul	=	dims_t::m_dims*dims_t::l_dims*(dims_t::m_dims*dims_t::l_dims+1ul)/2ul;

		using	hmul_t	=	recvec<comp_t,recidx<n_hmul,dims_t::n_dims,dims_t::n_elem>>;//for implementing H*psi


		using	lgd	=	bss::spherical_harmonics::legendre<qmax>;

		kntc_t	kntc;
		potn_t	potn;
		pole_l	pole;
		ylmr_t	ylmr;
		cptr_t	cptr;
		buff_t*	buff;

		hmat_t	hmat;
		desc_t	desc;
	
		hmul_t	hmul;

		rsrc_t*	rsrc;

		//helper
		inline	double	get_rc		(const size_t i)const noexcept{return std::get<0>(pole[i]);}
		inline	size_t	get_np		(const size_t i)const noexcept{return std::get<1>(pole[i]).size();}
		inline	double	get_th		(const size_t i,const size_t j)const noexcept{return std::get<1>(pole[i])[j][0];}
		inline	double	get_ph		(const size_t i,const size_t j)const noexcept{return std::get<1>(pole[i])[j][1];}
		inline	double	get_zc		(const size_t i,const size_t j)const noexcept{return std::get<1>(pole[i])[j][2];}
	public:	
		//memory affairs
		explicit operator_hm(rsrc_t* _rsrc=global_default_arena_ptr):hmat(0),hmul{0},buff{0},rsrc(_rsrc)
		{
			if(rsrc)
			{
				rsrc->acquire((void**)&kntc.data,sizeof(double)*kntc.dim.n_leng,align_val_avx3);
				rsrc->acquire((void**)&potn.data,sizeof(double)*potn.dim.n_leng,align_val_avx3);
				rsrc->acquire((void**)&ylmr.data,sizeof(comp_t)*ylmr.dim.n_leng,align_val_avx3);
				rsrc->acquire((void**)&cptr.data,sizeof(size_t)*(cptr.dim.n_leng+1ul),align_val_avx3);//CSR3-like format
			}else
			{
				 throw errn_t{"null arena parsed in when constructing operator_hm."};
			}
		}
		~operator_hm()noexcept
		{
			if(rsrc)
			{
				rsrc->release(kntc.data);
				rsrc->release(potn.data);
				rsrc->release(ylmr.data);
				rsrc->release(cptr.data);
				if(buff)
				rsrc->release(buff);
			}
			destroy_hmat();
			destroy_hmul();
		}	

	private:
		//-------------------------------------------------------------
		// subfunctions for initialization (should not call by users)
		//-------------------------------------------------------------
		inline	void	cal_pole	(const size_t iml,const size_t jml,comp_t* tmp)const noexcept//suppose jml>=iml
		{//can only be called after kntc,cptr,ylmr and potn has been prepared
			long m1,l1;dims_t::in(iml/dims_t::l_dims,iml%dims_t::l_dims,m1,l1);
			long m2,l2;dims_t::in(jml/dims_t::l_dims,jml%dims_t::l_dims,m2,l2);
			size_t	uval=m2-m1;//u=m1-m2, uval records its index
			if(iml==jml)
			{
				#pragma GCC ivdep
				for(size_t i=0;i<dims_t::n_rsub;++i)tmp[i]=kntc(l1)[i];
			}else
			{
				#pragma GCC ivdep
				for(size_t i=0;i<dims_t::n_rsub;++i)tmp[i]=0.0;
			}
			size_t	idc_ini	=	cptr(iml,jml);
			size_t	idc_end	=	cptr(iml,jml+1ul);
			for(size_t idc=idc_ini;idc<idc_end;++idc)//for (m1l1,m2l2) block, find all nonzero (u,q) CI terms
			{
				auto [cval,qval]	=	buff[idc];
				for(size_t i_pole=0;i_pole<n_pole;++i_pole)
				{
					auto cyval	=	cval*ylmr(uval,qval,i_pole);
					auto vval	=	potn(i_pole,qval);
					for(size_t i=0;i<dims_t::n_rsub;++i)tmp[i]+=cyval*vval[i];
				}
			}
		}//end of cal_pole

		//to initialize operator data
		template<class func_t>
		inline	void	initialize_kntc
		(
			const integrator_radi<dims_t>& radi,
			const func_t& func, //centrifugal part
			const double mass
		)noexcept//prepare kinetic operator
		{
			#pragma omp parallel for
                        for(size_t l=0;l<=dims_t::n_lmax;++l)
                        {
                               	radi.integrate_hc(kntc(l),func,mass,l);
                        }
		}//end of initialize_kntc
	
		inline	void	initialize_potn
		(
			const integrator_radi<dims_t>& radi
		)noexcept//prepare potential (radial)
		{
			for(size_t p=0;p<n_pole;++p)//loop over all poles
			{
				double 	rc	=	this->get_rc(p);
				#pragma omp parallel for
				for(size_t q=0;q<=qmax;++q)
				{
					auto 	func	=	[=](double r)
					{
						double rmin	=	std::min(r,rc);
						double rmax	=	std::max(r,rc);
						return std::pow(rmin/rmax,q)/rmax;
					};
					radi.integrate_0r0(potn(p,q),func);
				}
			}	
		}//end of initialize_potn

		inline	void	initialize_ylmr()//parepare potential (angular)
		{
			size_t 	nthread = omp_get_max_threads();
			double*	plm_all;
			comp_t*	exm_all;

			rsrc->acquire((void**)&plm_all,nthread*lgd::max()*sizeof(double),align_val_avx3);
			rsrc->acquire((void**)&exm_all,nthread*udim*sizeof(comp_t),align_val_avx3);

			#pragma omp parallel for num_threads(nthread)
			for(size_t p=0;p<n_pole;++p)//loop over all poles
			{
				size_t	np	=	this->get_np(p);
				for(size_t iu=0;iu< udim;++iu)// u=0:udim-1
				for(size_t iq=0;iq<=qmax;++iq){ylmr(iu,iq,p)=0.0;}//init for summation

				size_t	tid	=	omp_get_thread_num();
				double*	plm	=	plm_all + tid*lgd::max();
				comp_t*	exm	=	exm_all + tid*udim;
				for(size_t ip=0;ip<np;++ip)
				{
					//prepare conj{exp(I*u*ph)}, u=0,-1,-2,...;  we only calculate the upper triag, where u=m1-m2 always <=0
					for(size_t iu=0;iu<udim;++iu)
					{
					double	uph	=	iu*get_ph(p,ip);
						exm[iu]	=	comp_t{cos(uph),sin(uph)}//conjugation and minus cancels each other!
							*	(iu%2?-1.:1.);//ylm conjugate property
					}		
					//prepare plm(costh)
					double	costh	=	cos(get_th(p,ip));
					lgd::cal(costh,plm);
					//add this pole to the sum
					double	zc	=	this->get_zc(p,ip);
					for(size_t iu=0 ;iu< udim;++iu)
					for(size_t iq=iu;iq<=qmax;++iq)
					{
						ylmr(iu,iq,p)-=plm[lgd::idx(iq,iu)]*exm[iu]*zc;//suppose electron charge = -1.0
					}
				}
			}

			rsrc->release(plm_all);
			rsrc->release(exm_all);
		}//end of initialize_ylmr

		inline	void	initialize_cptr
		(
			const integrator_angu<dims_t>& angu
		)//evaluate angular prefactors
		{
			if(buff){rsrc->release(buff);buff=0;}

			auto	temp	=	std::vector<std::vector<std::tuple<double,size_t>>>(dims_t::m_dims*dims_t::l_dims);

			#pragma omp parallel for
			for(size_t iml=0;iml<dims_t::m_dims*dims_t::l_dims;++iml)//loop over ml-blocks (row index)
			{
				long m1,l1;dims_t::in(iml/dims_t::l_dims,iml%dims_t::l_dims,m1,l1);
				auto&	_temp = temp[iml];
				for(size_t jml=0;jml<dims_t::m_dims*dims_t::l_dims;++jml) //loop over ml-blocks (col index)
				{
					long m2,l2;dims_t::in(jml/dims_t::l_dims,jml%dims_t::l_dims,m2,l2);
					
					bool	usgn	=	m2>=m1;	
					size_t	uabs	=	usgn  ?m2-m1:m1-m2;
					size_t	qini	=	l2>=l1?l2-l1:l1-l2;
					size_t 	qend	=	l1+l2;
					if(qini<uabs)qini+=(uabs-qini+1ul)/2ul*2ul;//(1,2) += 2; (3,4) += 4; etc 
					if(qend>qmax)qend=qmax;	//if qmax is not set by max(l1+l2), this would avoid out-of-bound error.
	
					cptr(iml,jml) 	=	_temp.size();
					for(size_t q=qini;q<=qend;q+=2ul)//loop over all q in laplace expansion
					{
						double v=	sqrt(4*PI<double>*(2*l1+1)*(2*l2+1)/(2*q+1))
							*	angu.coupling_3j(l1,q,l2)
							*	angu.coupling_3j(l1,q,l2,-m1,m1-m2,m2)
							*	(m1%2?-1.:1.);
						if(v*v>eps2)_temp.emplace_back(v,q);
					}
				}
			}
			size_t count_size = temp[0ul].size();
			for(size_t iml=1;iml<dims_t::m_dims*dims_t::l_dims;++iml)
			{
				for(size_t jml=0;jml<dims_t::m_dims*dims_t::l_dims;++jml)
				{
					cptr(iml,jml)+=count_size;
				}
				count_size+=temp[iml].size();
			}
			rsrc->acquire((void**)&buff,sizeof(buff_t)*count_size,align_val_avx3);

			cptr[cptr.dim.n_leng]=count_size;//a CSR3-like format
	
			count_size=0ul;
			for(size_t iml=0;iml<dims_t::m_dims*dims_t::l_dims;++iml)
			{
				for(auto iter:temp[iml])buff[count_size++]=iter;
			}
		}//end of initialize_cptr

		inline	void	initialize_hmat	(sparse_matrix_t& _h)
		{
			std::vector<MKL_INT> 	row;	
			std::vector<MKL_INT> 	col;
			std::vector<comp_t> 	val;
			std::vector<comp_t> 	tmp;
			
			try{
				tmp.resize(dims_t::n_elem*dims_t::n_dims);//size of an r-subspace
				for(size_t iml=0;iml<dims_t::m_dims*dims_t::l_dims;++iml)	
				{
					{//diagonal blocks (jml==iml)
						this->cal_pole(iml,iml,tmp.data());
						//calculate matrix index
						for(size_t ir=0;ir<dims_t::n_dims;++ir)
						{
							size_t row_index=iml*dims_t::n_dims+ir+1ul;//one-based indexing
							size_t nc=std::min(dims_t::n_elem,dims_t::n_dims-ir);
							for(size_t ic=0;ic<nc;++ic)//upper half panel
							{
								row.push_back(row_index);
								col.push_back(row_index+ic);
								val.push_back(tmp[ir*dims_t::n_elem+ic]);
							}
						}
					}
					for(size_t jml=iml+1;jml<dims_t::m_dims*dims_t::l_dims;++jml)
					{//off-diagonal blocks
						this->cal_pole(iml,jml,tmp.data());
						for(size_t ir=0;ir<dims_t::n_dims;++ir)
						{
							size_t row_index=iml*dims_t::n_dims+ir+1ul;//one-based indexing
							size_t col_index=jml*dims_t::n_dims+ir+1ul;//one-based indexing
							size_t mc=std::min(ir,dims_t::n_diag);
							for(size_t ic=1;ic<=mc;++ic)//lower half panel
							{
								row.push_back(row_index);
								col.push_back(col_index-ic);
								val.push_back(tmp[(ir-ic)*dims_t::n_elem+ic]);
							}
							size_t nc=std::min(dims_t::n_elem,dims_t::n_dims-ir);
							for(size_t ic=0;ic<nc;++ic)//upper half panel
							{
								row.push_back(row_index);
								col.push_back(col_index+ic);
								val.push_back(tmp[ir*dims_t::n_elem+ic]);
							}	
						}
					}//end of off diagonal*/	
				}
			}catch (const std::bad_alloc& e)
			{
				throw errn_t("std::bad alloc in initialize_hmat.");
			}

			sparse_matrix_t	hmat_coo;
			sparse_status_t info;
			info    =       mkl_sparse_z_create_coo
			(
				&hmat_coo,SPARSE_INDEX_BASE_ONE,
				dims_t::n_leng,dims_t::n_leng,val.size(),
				row.data(),col.data(),val.data()
			);
			if(info!=SPARSE_STATUS_SUCCESS)throw errn_t{"fail to generate hmat_coo when initializing operator_hm."};
			info    =       mkl_sparse_convert_csr
			(
				hmat_coo,SPARSE_OPERATION_NON_TRANSPOSE,&_h
			);
			if(info!=SPARSE_STATUS_SUCCESS)throw errn_t{"fail to convert hmat_coo when initializing operator_hm."};
			info	=	mkl_sparse_destroy
			(
				hmat_coo
			);
			if(info!=SPARSE_STATUS_SUCCESS)throw errn_t{"fail to destroy hmat_coo when initializing operator_hm."};
		}//end of initialize_hmat

		inline	void	initialize_hmul()
		{
			destroy_hmul();
			rsrc->acquire((void**)&hmul.data,sizeof(comp_t)*hmul.dim.n_leng,align_val_avx3);

			size_t	k=0ul;
			for(size_t iml=0ul;iml<dims_t::m_dims*dims_t::l_dims;++iml)
			{
				for(size_t jml=iml;jml<dims_t::m_dims*dims_t::l_dims;++jml)
				{
					this->cal_pole(iml,jml,hmul(k++).data);
				}
			}	
		}//end of initialize_hmul

		inline	void	destroy_hmat()noexcept//you can mannually call this
		{
			if(hmat)
			{
				mkl_sparse_destroy(hmat);hmat=0;
			}
		}//end of destroy_hmat

		inline	void	destroy_hmul()noexcept
		{
			if(hmul.data)
			{
				rsrc->release(hmul.data);hmul.data=0;
			}
		}//end of destroy_hmul
	public:
		//---------------------------------------------------------------
		//	    initializer. called by user ahead.
		//---------------------------------------------------------------
		inline	void	set_pole// interfaces to manipulate the pole list
		(
			const size_t i,const std::vector<double>& para
		)
		{
			if(para.size()%3ul!=1ul)
			{
				throw errn_t("invalid para in 1e set_pole!\n");
			}
			std::get<0>(pole[i]) = para[0];
			auto& _vec = std::get<1>(pole[i]);
			_vec.resize(para.size()-1ul);
			size_t count=0;
			for(size_t j=1;j<para.size();j+=3)
			{
				_vec[count][2] = para[j+0];	//zc. this class store zc, not e*zc
				_vec[count][0] = para[j+1];	//th
				_vec[count][1] = para[j+2];	//ph
				++count;
			}
		}//end of set_pole
	
		inline	void	clr_pole	()noexcept// interfaces to manipulate the pole list
		{
			for(size_t p=0;p<n_pole;++p)
			std::get<1>(pole[p]).clear();
		}//end of clr_pole

		template<int flag,class func_t>
		inline	void	initialize//call set_pole before calling me!
		(
			const integrator_radi<dims_t>& radi,
			const integrator_angu<dims_t>& angu,
			const func_t& func,
			const double mass
		)
		{
			initialize_kntc(radi,func,mass);
			initialize_potn(radi);
			initialize_ylmr();
			initialize_cptr(angu);	

			if constexpr(flag&1)//check bit0
			{
				initialize_hmat(hmat);
				desc.type=SPARSE_MATRIX_TYPE_HERMITIAN;
				desc.mode=SPARSE_FILL_MODE_UPPER;
				desc.diag=SPARSE_DIAG_NON_UNIT;
			}
			if constexpr(flag&2)//check bit1
			{
				initialize_hmul();
			}
		}//end of initialize
		//---------------------------------------------------------------
		//		 access to hmat, hmul and desc
		//---------------------------------------------------------------
		inline	auto	create_csrmat_comp()const
		{
			auto	mat	=	csrmat_view<comp_t,1>();
			mat.convert_from(hmat);//use mat.check() to check me if necessary
			return 	mat;
		}//end of create_csrmat_comp
	
		inline	auto const&	get_hmat()const noexcept{return	hmat;}
		inline	auto const&	get_desc()const noexcept{return desc;}
	
		inline	auto 		get_hmul()const noexcept{return hmul;}//hmul.const_cast not yet added. XXX
};//end of operator_hm
//================================================================================================================================
//
//								Overlap Matrix
//
//================================================================================================================================
template<class dims_t>
struct	operator_sc:public operator_rsym<dims_t>//centrifugal
{
	private:
		using	operator_rsym<dims_t>::data;
		using	operator_rsym<dims_t>::rsrc;

		double*	chol;	//store cholesky factorization
	public:
		//access cholesky factorization
		inline	double*		cholesky()noexcept	{return chol;}		
		inline	double const*	cholesky()const noexcept{return chol;}		

		//constructor
		operator_sc(operator_sc const&)=delete;
		operator_sc(operator_sc && rhs)noexcept:
		operator_rsym<dims_t>(std::move(rhs))
		{
			chol=rhs.chol;
		}
		operator_sc& operator=(operator_sc const&)=delete;
		operator_sc& operator=(operator_sc &&)=delete;

		explicit operator_sc(rsrc_t* _rsrc=global_default_arena_ptr):operator_rsym<dims_t>(_rsrc)
		{
			if(rsrc!=0)
			{
				rsrc->acquire((void**)&chol,sizeof(double)*dims_t::n_rsub,align_val_avx3);
			}else
			{
				throw	errn_t{"null arena parsed in when constructing operator_sc.\n"};
			}
		}
		//destructor
		virtual ~operator_sc()noexcept override
		{
			if(rsrc!=0)
			{
				rsrc->release(chol);
			}
		}
		//initializer
		inline	void	initialize(const integrator_radi<dims_t>& base)
		{
			base.integrate_0r0(*this,[](double){return 1.0;});
			#pragma GCC ivdep
			for(size_t i=0;i<dims_t::n_rsub;++i)
			{
				chol[i]=data[i];//copy to chol
			}
                    	int err=impl_ed_simd::lu_dsb<dims_t::n_dims,dims_t::n_diag>(chol);//call lapacke to do in-place cholesky factorization
                	if(err)
			{
				throw errn_t("lapcke fail in cholesky decomposition. errcode:"+to_string(err)+"\n");
			}
		}
		//solve linear system S*X=Y
		inline	void	inverse_sub(comp_t* u)const noexcept
		{
			intrinsic::ltrb_inv_vecd<dims_t::n_dims,dims_t::n_elem>(chol,u,u);
			intrinsic::utrb_inv_vecd<dims_t::n_dims,dims_t::n_elem>(chol,u,u);
		}

		inline	void	inverse_all(comp_t* u)const noexcept
		{
			#pragma omp for
			for(size_t i=0ul;i<dims_t::n_leng;i+=dims_t::n_dims)
			{
				inverse_sub(u+i*dims_t::n_dims);
			}
		}

		//create a sparse matrix version
		template<class type_t>
		inline	auto	create_csrmat()const //in full lm space
		{
			auto 	mat	=	csrmat<type_t,1>();
			for(size_t iml=0;iml<dims_t::m_dims*dims_t::l_dims;++iml)
			{
				size_t indent	=	iml*dims_t::n_dims+1ul;
				for(size_t ir=0;ir<dims_t::n_dims;++ir)
				{
					mat.push_irow();
					size_t nc=std::min(dims_t::n_elem,dims_t::n_dims-ir);
					for(size_t ic=0;ic<nc;++ic)
					{
						type_t val;	
						val=data[ir*dims_t::n_elem+ic];
						mat.val.push_back(val);
						mat.col.push_back(indent+ir+ic);
					}
				}
			}mat.push_irow();
			return 	mat;
		}//end of create_csrmat

		inline	auto	create_csrmat_real()const noexcept{return create_csrmat<double>();}
		inline	auto	create_csrmat_comp()const noexcept{return create_csrmat<comp_t>();}
};//end of operator_sc

template<class dims_t>
struct	operator_sb//bicentral
{
	public:
		using	data_t	=	csrmat<comp_t,1>;
		using	esub_t	=	operator_eta2<dims_t>;
		using	xsub_t	=	operator_xsub<dims_t,2>;//there are two integrals, x^2 and x^0

		xsub_t		pown;//store xi2 and xi0 integral. both odd and even are stored even if m_dims=1
		esub_t		eta2;//store et2 integral.
		double		r3o8;//store R^3/8	
		data_t*		data;//store matrix element for each m subspace, in csr format
		csrinv*		fact;//store handle of DSS to solve S\Y, for each m subspace

		//constructor
		operator_sb(operator_sb const&)=delete;
		operator_sb(operator_sb&& rhs)=delete;
		operator_sb& operator=(operator_sb const&)=delete;
		operator_sb& operator=(operator_sb &&)=delete;

		explicit operator_sb(rsrc_t* _rsrc=global_default_arena_ptr):pown(_rsrc)
		{
			data = new(std::nothrow) data_t[dims_t::m_dims];
			fact = new(std::nothrow) csrinv[dims_t::m_dims];//it seems that arena cannot be used to create DSS handle object for DSS handle is not trivially constructed
		}//end of constructor	
		virtual	~operator_sb()noexcept
		{
			delete[] data;
			delete[] fact;
		}//end of destructor

		//initializer, should only be called once
		template<int flag_to_init_inv=1>
		inline	void	initialize
		(
			const integrator_radi<dims_t>& radi,
			const integrator_angu<dims_t>& angu,
			const double dist
		)noexcept
		{//S=R^3/8*(e0@x2-e2@x0)
			r3o8	=	dist*dist*dist/8.;
			pown.initialize(radi,[&](double x){return x*x;},0ul);//id=0 is xi^2
			pown.initialize(radi,[&](double x){return 1.0;},1ul);//id=1 is xi^0
			eta2.initialize(angu);

			#pragma omp parallel for
			for(size_t im=0;im<dims_t::m_dims;++im)//for different m, <l,m|eta^2|l',m> varies, the global term varies
			{
				auto&	temp	=	data[im];
				auto	x2	=	pown(dims_t::od_m(im),0ul);	
				auto	x0	=	pown(dims_t::od_m(im),1ul);	
				for(size_t il=0;il<dims_t::l_dims;++il)
				{
					auto	e2sh0		=	eta2.sh0(im,il);
					auto	e2sh2		=	eta2.sh2(im,il);
					for(size_t ir=0;ir<dims_t::n_dims;++ir)
					{
						temp.push_irow();
						size_t n1	=      ir<dims_t::n_dims-dims_t::n_elem?dims_t::n_elem:dims_t::n_dims-ir;
                        			size_t n2	=      ir<dims_t::n_elem?ir:dims_t::n_elem-1;
	                        		size_t is	=      ir+il*dims_t::n_dims;
						for(size_t ic=0;ic<n1;++ic)
						{
							temp.val.push_back(comp_t{r3o8*(x2(ir,ic)-e2sh0*x0(ir,ic)),0.0});
							temp.col.push_back(is+ic+1);//one based indexing
						}
						if constexpr(dims_t::l_dims>1ul)
						if (il<dims_t::l_dims-2ul)//XXX better way?
						{
							for(size_t ic=n2;ic>0;--ic)
							{
								temp.val.push_back(comp_t{-r3o8*e2sh2*x0(ir-ic,ic),0.0});
								temp.col.push_back(is-ic+dims_t::n_dims*2ul+1);//one based indexing
							}
							for(size_t ic=0;ic<n1;++ic)
        	                			{
				                                temp.val.push_back(comp_t{-r3o8*e2sh2*x0(ir,ic),0.0});
                        	        			temp.col.push_back(is+ic+dims_t::n_dims*2ul+1);//one based indexing
                        				}
						}
					}
				}temp.push_irow();//row sensor for CSR format.
			}
			if constexpr(flag_to_init_inv==1)
			for(size_t im=0;im<dims_t::m_dims;++im)//prepare csr-dss handle from csr matrix data
			{
				fact[im].initialize(data[im]);
			}
		}//end of initialize

		inline	auto	create_csrmat_real	(const size_t im)const//by simply creating a new copy
		{
			auto	mat	=	csrmat<double,1>();
			mat.row.resize(data[im].row.size());
			mat.col.resize(data[im].col.size());
			mat.val.resize(data[im].val.size());
			for(size_t i=0;i<mat.row.size();++i)mat.row[i]=data[im].row[i];
			for(size_t i=0;i<mat.col.size();++i)mat.col[i]=data[im].col[i];
			for(size_t i=0;i<mat.val.size();++i)mat.val[i]=real(data[im].val[i]);
			return 	mat;	//use NRVO or move to avoid extra copy
		}//end of create_csrmat_real

		template<class lhsw_t,class rhsw_t>
		inline	comp_t	observe_rsub		(const lhsw_t* w1,const rhsw_t* w2,const size_t im,const size_t il)const noexcept
		{
			comp_t	prj	=	comp_t{0.0,0.0};
			double*	x2	=	pown(dims_t::od_m(im),0ul).data;
			double*	x0	=	pown(dims_t::od_m(im),1ul).data;
				prj	=	prj+intrinsic::symb_prj_vecd<dims_t::n_dims,dims_t::n_elem>(x2,w1,w2);
				prj	=	prj-intrinsic::symb_prj_vecd<dims_t::n_dims,dims_t::n_elem>(x0,w1,w2)*eta2.sh0(im,il);
			if constexpr(dims_t::l_dims>2)
			{
				if(il<dims_t::l_dims-2)
				prj	=	prj-intrinsic::symb_prj_vecd<dims_t::n_dims,dims_t::n_elem>(x0,w1,w2+2ul*dims_t::n_dims)*eta2.sh2(im,il);
				if(il>=2)
				prj	=	prj-intrinsic::symb_prj_vecd<dims_t::n_dims,dims_t::n_elem>(x0,w1,w2-2ul*dims_t::n_dims)*eta2.sh2(im,il-2);
			}
			return 	prj*r3o8;	
		}//end of observe_rsub

		template<class lhsw_t,class rhsw_t>
		inline	comp_t	observe_lsub		(const lhsw_t* w1,const rhsw_t* w2,const size_t im)const noexcept
		{
			comp_t	prj	=	comp_t{0.0,0.0};
			double*	x2	=	pown(dims_t::od_m(im),0ul).data;
			double*	x0	=	pown(dims_t::od_m(im),1ul).data;
			for(size_t il=0;il<dims_t::l_dims;++il)
			{
				lhsw_t const*	_w1	=	w1+il*dims_t::n_dims;
				rhsw_t const*	_w2	=	w2+il*dims_t::n_dims;
				prj	=	prj+intrinsic::symb_prj_vecd<dims_t::n_dims,dims_t::n_elem>(x2,_w1,_w2);
				prj	=	prj-intrinsic::symb_prj_vecd<dims_t::n_dims,dims_t::n_elem>(x0,_w1,_w2)*eta2.sh0(im,il);
				if constexpr(dims_t::l_dims>2){

				if(il<dims_t::l_dims-2)
				prj	=	prj-intrinsic::symb_prj_vecd<dims_t::n_dims,dims_t::n_elem>(x0,_w1,_w2+2ul*dims_t::n_dims)*eta2.sh2(im,il);
				if(il>=2)
				prj	=	prj-intrinsic::symb_prj_vecd<dims_t::n_dims,dims_t::n_elem>(x0,_w1,_w2-2ul*dims_t::n_dims)*eta2.sh2(im,il-2);}
			}
			return 	prj*r3o8;	
		}//end of observe_lsub

		//evaluate <lhs|S|rhs>
		template<class lhsw_t,class rhsw_t>
		inline	void	observe			(const lhsw_t* w1,const rhsw_t* w2,comp_t& proj)const noexcept
		{
			#pragma omp for collapse(2) reduction(+:proj)
			for(size_t im=0;im<dims_t::m_dims;++im)
			for(size_t il=0;il<dims_t::l_dims;++il)
			{
				size_t sh=	dims_t::l_dims*dims_t::n_dims*im+dims_t::n_dims*il;
				proj	+=	observe_rsub(w1+sh,w2+sh,im,il);
			}
		}//end  of observe

		template<class lhsw_t>
		inline	void	observe			(const lhsw_t* w1,double& norm)const noexcept
		{	
			#pragma omp for collapse(2) reduction(+:norm)
			for(size_t im=0;im<dims_t::m_dims;++im)
			for(size_t il=0;il<dims_t::l_dims;++il)
			{
				size_t sh=	dims_t::l_dims*dims_t::n_dims*im+dims_t::n_dims*il;
				norm	+=	observe_rsub(w1+sh,w1+sh,im,il)[0];
			}
		}//end of observe

		//solve linear system S*X=Y
		inline	void	inverse_all		(const comp_t* w1,comp_t* w2)noexcept
		{
			for(size_t im=0;im<dims_t::m_dims;++im)
			{
				fact[im].solve(w1+im*dims_t::n_dims*dims_t::l_dims,w2+im*dims_t::n_dims*dims_t::l_dims);
			}	
		}//end of inverse_all
		
};//end of operator_sb


