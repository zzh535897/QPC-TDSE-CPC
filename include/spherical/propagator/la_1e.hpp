#pragma once

//require <libraries/support_std.h>
//require <libraries/support_avx.h>
//require <libraries/support_omp.h>
//require <resources/arena.h>
//require <utilities/error.h>
//require <algorithm/ed.h>

//-----------------------------------------------------------------------------------------
template<class oper_s>
struct	type_of_inverse_all_t:public std::integral_constant<int,-1>{};//oper unregistered

template<class dims_t>
struct	type_of_inverse_all_t<operator_sc<dims_t>>:public std::integral_constant<int,0>{};//oper has inside-place "inverse_all"

template<class dims_t>
struct	type_of_inverse_all_t<operator_sb<dims_t>>:public std::integral_constant<int,1>{};//oper has out-of-place "inverse_all"

template<class oper_s>
static constexpr int 	type_of_inverse_all	 =	type_of_inverse_all_t<oper_s>{};
//-----------------------------------------------------------------------------------------
template<class dims_t>
struct	arnoldi_workspace
{
	protected:
		//the workspace for reduced matrix in krylov subspace 
		double*	arnoldi_a;	//diagonal entry, and eigen values after diagonalization
		double*	arnoldi_b;	//off-diagonal entry, and b[0] stores normalization constant of w
		//the workspace for reduced vector in krylov subspace
		comp_t*	arnoldi_w;	//the reduced vector. after propagation, w=exp(-idt*a)*z^T*[1,0,...]^T
		comp_t*	arnoldi_x;	//the reduced vector. after propagation, x=z*w; also used as projection values in GS orthogonalization
		double*	arnoldi_z;	//the eigen vector of reduced matrix, i.e. a=z'hz (generated after the krylov-reduction)
		//the workspace for projector into krylov subspace
		comp_t*	arnoldi_u;	//the projector into krylov subspace, i.e. H=u'hu (generated in the krylov-reduction)
		//stoping criterion
		size_t	arnoldi_kmax;
		double	arnoldi_emin;
		double	arnoldi_emax;
		rsrc_t*	rsrc;
	
	public:
		//access the workspace
		inline	auto		u(const size_t i)noexcept	{return arnoldi_u+i*dims_t::n_leng;}//require dims_t::n_leng
		inline	auto const	u(const size_t i)const noexcept	{return arnoldi_u+i*dims_t::n_leng;}//require dims_t::n_leng

		//memory allocation
		arnoldi_workspace(arnoldi_workspace const&)=delete;
		arnoldi_workspace(arnoldi_workspace&&)=delete;
		arnoldi_workspace& operator=(arnoldi_workspace const&)=delete;
		arnoldi_workspace& operator=(arnoldi_workspace&&)=delete;

		explicit arnoldi_workspace(size_t kmax,double emin,double emax,rsrc_t* _rsrc=global_default_arena_ptr):rsrc(_rsrc)
		{
			arnoldi_kmax	=	kmax;  
			arnoldi_emin	=	emin;
			arnoldi_emax	=	emax;
			if(emin>emax||emin<=0||emax<=0)
			{
				throw 	errn_t("invalid parameters for arnoldi propagator.\n");
			}
			if(rsrc!=0)
			{
				rsrc->acquire((void**)&arnoldi_a,(kmax+2)*sizeof(double),align_val_avx3);
				rsrc->acquire((void**)&arnoldi_b,(kmax+2)*sizeof(double),align_val_avx3);
				rsrc->acquire((void**)&arnoldi_w,(kmax+2)*sizeof(comp_t),align_val_avx3);
				rsrc->acquire((void**)&arnoldi_x,(kmax+2)*sizeof(comp_t),align_val_avx3);
				rsrc->acquire((void**)&arnoldi_z,(kmax*kmax)*sizeof(double),align_val_avx3);
				rsrc->acquire((void**)&arnoldi_u,(kmax+2)*dims_t::n_leng*sizeof(comp_t),align_val_avx3);
			}else
			{
				throw 	errn_t("null arena parsed in when constructing propagator_arnoldi.\n");
			}
		}//end of constructor
		
		virtual	~arnoldi_workspace()noexcept
		{
			if(rsrc!=0)
			{
				rsrc->release(arnoldi_a);
				rsrc->release(arnoldi_b);
				rsrc->release(arnoldi_w);
				rsrc->release(arnoldi_x);
				rsrc->release(arnoldi_z);
				rsrc->release(arnoldi_u);
			}
		}//end of destructor
};//end of arnoldi_workspace

template<class dims_t>
struct	propagator_arnoldi:public arnoldi_workspace<dims_t>
{
	protected:	
		using	base_t	=	arnoldi_workspace<dims_t>;
		using	base_t::arnoldi_a;
		using	base_t::arnoldi_b;
		using	base_t::arnoldi_z;
		using	base_t::arnoldi_w;
		using	base_t::arnoldi_x;
		using	base_t::arnoldi_u;
		using	base_t::arnoldi_kmax;
		using	base_t::arnoldi_emin;
		using	base_t::arnoldi_emax;
	public:
		using	base_t::u;
	
		template<class...args_t>
		explicit propagator_arnoldi(args_t&&...args):base_t(std::forward<args_t>(args)...){}
		//end of constructor
		virtual ~propagator_arnoldi()noexcept{}
		//end of destructor

		template<class coef_t,class oper_t>
		inline	void	construct_sub	(coef_t& w,const oper_t&s_mat,const size_t nthreads)noexcept;

		inline	void	propagate_sub	(double dt,const size_t m_sub,const size_t nthreads);
	
		template<class coef_t>
		inline	void	eliminate_sub	(coef_t& w,const size_t m_sub,const size_t nthreads)noexcept;

		template<class oper_t>
		inline	bool	orthogonalize_ip(size_t  k,oper_t&s_mat,const size_t nthreads)noexcept;	//smat has inside-place version "inverse_all"
		template<class oper_t>
		inline	bool	orthogonalize_op(size_t  k,oper_t&s_mat,const size_t nthreads)noexcept;	//smat has out-of-place version "inverse_all"

		inline	int	criterion	(const size_t m_sub)const noexcept;

		template<class deri_t,class coef_t>
		inline	int	propagate1	(coef_t& w,double dt,const size_t nthreads);
};//end of propagator_arnoldi

template<class dims_t>template<class coef_t,class oper_t>
inline	void	propagator_arnoldi<dims_t>::construct_sub	(coef_t& w,const oper_t&s_mat,const size_t nthreads)noexcept
{
	auto	norm	=	zero<double>;
	auto	uvec	=	u(0);
	#pragma omp parallel num_threads(nthreads)
	{ 
		intrinsic::vecd_assg_vecd<dims_t::n_leng,0>(w(),uvec);//copy w to u0. automatically barriered at the end of the call

		s_mat.observe(uvec,norm);//note the function usually uses omp reduction, thus may give indeterminstic results between different runs

		#pragma omp master
		{
			norm	=	sqrt(norm);
		}
		#pragma omp barrier
		intrinsic::vecd_assg_vecd<dims_t::n_leng,0>(uvec,uvec,1./norm);//do renormalization to 1
	}//end of omp parallel
	arnoldi_b[0]=norm;
}//end of construct_sub

template<class dims_t>
inline	void	propagator_arnoldi<dims_t>::propagate_sub	(double dt,const size_t m_sub,const size_t nthreads)
{
	//do diagonalization in krylov subspace, h=zEz', z stored as lapacke-row-major,
	int err=impl_ed_simd::ev_dst<0>(arnoldi_a,arnoldi_b+1,arnoldi_z,m_sub);//indent b by 1 due to lapacke format, see algorithm/ed.h
	if(err)
	{
		throw errn_t("lapacke fail in ev_dst. errcode:"+to_string(err)+"\n");
	}
	//do propagation by w:=exp(-I*dt*E)*z'*(1,0,0,....)'
	comp_t  _idt=-unim<comp_t>*dt;
	for(size_t i=0;i<m_sub;++i)
	{
		arnoldi_w[i]=arnoldi_z[i*m_sub]*exp(_idt*arnoldi_a[i]);
	}
	//do reconstruction by x:=z*w
	for(size_t j=0;j<m_sub;++j)arnoldi_x[j] =arnoldi_w[0]*arnoldi_z[j];
	for(size_t i=1;i<m_sub;++i){size_t sh  =i*m_sub;
	for(size_t j=0;j<m_sub;++j)arnoldi_x[j]+=arnoldi_w[i]*arnoldi_z[j+sh];}
	//do renormalization by x:=x*b[0]
	for(size_t j=0;j<m_sub;++j)arnoldi_x[j] =arnoldi_x[j]*arnoldi_b[0];
}//end of propagate_sub

template<class dims_t>template<class coef_t>
inline	void	propagator_arnoldi<dims_t>::eliminate_sub	(coef_t& w,const size_t m_sub,const size_t nthreads)noexcept
{
	#pragma omp parallel num_threads(nthreads)
	{
		comp_t*	ui=arnoldi_u;
		comp_t	xi=arnoldi_x[0];
		{
			intrinsic::vecd_assg_vecd<dims_t::n_leng,0>(ui,w(),xi);//w=ui*xi. this function is internally omp parallellized 
		}
		for(size_t i=1;i<m_sub;++i)
		{
			ui=arnoldi_u+i*dims_t::n_leng;
			xi=arnoldi_x[i];
			intrinsic::vecd_assg_vecd<dims_t::n_leng,1>(ui,w(),xi);//w+=ui*xi. this function is internally omp parallellized
		}
	}	
}//end of eliminate_sub

template<class dims_t>template<class oper_t>
inline	bool	propagator_arnoldi<dims_t>::orthogonalize_ip	(size_t k,oper_t&s_mat,const size_t nthreads)noexcept//make sure k>0
{
	auto	uk	=	u(k);
	auto	_flag	=	zero<bool>;
	auto	_norm	=	zero<double>;
	
	#pragma omp parallel num_threads(nthreads)
	{
		//evaluate P(i):=<u(i)|S*u(k)>
		#pragma omp for
		for(size_t i=0;i<k;++i)
		{
			arnoldi_x[i]	=	intrinsic::vecd_proj_vecd<dims_t::n_leng>(u(i),uk);
		}
		//evaluate |u(k)>:=S\|S*u(k)> XXX require oper_s::inverse_all
		s_mat.inverse_all(uk);//internally omp parallellized
		//evaluate |u(k)>:=|u(k)>-P(i)|u(i)> if P(i) is large. u(k-1) is always projected out regardless P(k-1)
		for(size_t i=0;i<k;++i)
		{
			if(i==k-1||norm(arnoldi_x[i])>4e-27)
			{
				intrinsic::vecd_assg_vecd<dims_t::n_leng,2>(u(i),uk,arnoldi_x[i]);//u(k)-=u(i)*P(i)
			}
		}
		//evaluate b(k):=<u(k)|S|u(k)> XXX require oper_s::observe
		s_mat.observe(uk,_norm);
		#pragma omp master
		if(_norm<1e-13)//lucky cancellation, stop orthogonalization
		{
			_norm	=	0;
			_flag	=	1;
		}else//does not cancel. do renormalization by u(k):=u(k)/sqrt(_norm);
		{
			_norm	=	sqrt(_norm);
		}
		#pragma omp barrier
		if(!_flag)
		{
			intrinsic::vecd_assg_vecd<dims_t::n_leng,0>(uk,uk,1./_norm);	
		}
	}//end of omp parallel
	arnoldi_a[k-1]	=	arnoldi_x[k-1][0];	//set lanczos a
	arnoldi_b[k]	=	_norm;			//set lanczos b
	return	_flag; 	
}//end of orthogonalize_ip

template<class dims_t>template<class oper_t>
inline	bool	propagator_arnoldi<dims_t>::orthogonalize_op	(size_t  k,oper_t&s_mat,const size_t nthreads)noexcept//make sure k>0
{
	auto	uk	=	u(k);
	auto	ut	=	u(k+1ul);//k+1 used as temp for storing S\|S*u(k)> since S\ is out-of-place
	auto	_flag	=	zero<bool>;
	auto	_norm	=	zero<double>;
	
	#pragma omp parallel num_threads(nthreads)
	{
		//evaluate P(i):=<u(i)|S*u(k)>
		#pragma omp for
		for(size_t i=0;i<k;++i)
		{
			arnoldi_x[i]	=	intrinsic::vecd_proj_vecd<dims_t::n_leng>(u(i),uk);
		}
	}
	{
		//evaluate |u(t)>:=S\|S*u(k)> XXX require oper_s::inverse_all
		s_mat.inverse_all(uk,ut);//internally omp parallellized, src,dst
	}
	#pragma omp parallel num_threads(nthreads)
	{
		//evaluate |u(k)>:=|u(k)>-P(i)|u(i)> if P(i) is large. u(k-1) is always projected out regardless P(k-1)
		for(size_t i=0;i<k;++i)
		{
			if(i==k-1||norm(arnoldi_x[i])>4e-27)
			{
				intrinsic::vecd_assg_vecd<dims_t::n_leng,2>(u(i),ut,arnoldi_x[i]);//uk-=ui*pi
			}
		}
		//evaluate b(k):=<u(k)|S|u(k)> XXX require oper_s::observe
		s_mat.observe(ut,_norm);

		#pragma omp master
		if(_norm<1e-13)//lucky cancellation, stop orthogonalization
		{
			_norm	=	0;
			_flag	=	1;
		}else//does not cancel. do renormalization by u(k):=u(k)/sqrt(_norm);
		{
			_norm	=	sqrt(_norm);
		}
		#pragma omp barrier
		if(!_flag)
		{
			intrinsic::vecd_assg_vecd<dims_t::n_leng,0>(ut,uk,1./_norm);//src,dst
		}
	}//end of omp parallel
	arnoldi_a[k-1]	=	arnoldi_x[k-1][0];	//set lanczos a
	arnoldi_b[k]	=	_norm;			//set lanczos b
	return	_flag; 	
}//end of orthogonalize_op

template<class dims_t>
inline	int	propagator_arnoldi<dims_t>::criterion		(const size_t m_sub)const noexcept
{
	double  eps=m_sub>1?norm(this->arnoldi_x[m_sub-1ul]):0.0;//norm, see support_avx.h
	if(eps>arnoldi_emax)
	{
		return  0;//>emax, dt shoule be smaller
	}else if(eps>arnoldi_emin)
	{
		return	1;//dt can remain unmodified
	}else
	{
		return 	2;//<emin, dt can be larger
	}
}//end of criterion

template<class dims_t>template<class deri_t,class coef_t>
inline	int	propagator_arnoldi<dims_t>::propagate1		(coef_t& w,double dt,const size_t nthreads)
{
	deri_t*	_this	=	static_cast<deri_t*>(this);	//cast to the derived class. this is a CRTP trick.
	bool	_flag	=	1;
	size_t	m_sub	=	0ul;			
	//the constructon of krylov subspace
	construct_sub(w,_this->oper_s,nthreads);//require deri_t::oper_s
	while(_flag&&m_sub<arnoldi_kmax)
	{
		_this	->	multiply(u(m_sub),u(m_sub+1),nthreads);//require deri_t::multiply
		if constexpr((type_of_inverse_all<decltype(deri_t::oper_s)>)==0)
		_flag	=!	
		 this	->	orthogonalize_ip(++m_sub,_this->oper_s,nthreads);//require deri_t::oper_s
		if constexpr((type_of_inverse_all<decltype(deri_t::oper_s)>)==1)
		_flag	=!	
		 this	->	orthogonalize_op(++m_sub,_this->oper_s,nthreads);//require deri_t::oper_s
	}
	//propagate in krylov subspace
	propagate_sub(dt,m_sub,nthreads);
	//the elimination of krylov subspace
	eliminate_sub(w ,m_sub,nthreads);
	return 	criterion(m_sub);
}//end of propagate1

//================================================================================================================================
//
//						Propagator User Class (Sph)
//
//================================================================================================================================
template<class dims_t>
struct	propagator_hc:public propagator_arnoldi<dims_t>
{
	public:
		static_assert(dims_t::i_type==0,"this propagator only applies to spherical symmtric system.");

		operator_hc<dims_t>	oper_h;
		operator_sc<dims_t>	oper_s;	

		template<class...args_t>
		explicit propagator_hc(args_t&&...args):propagator_arnoldi<dims_t>(std::forward<args_t>(args)...){}

		template<class potn_t>
		inline	void	initialize	
		(	
			const integrator_radi<dims_t>& radi,
			const integrator_angu<dims_t>& angu,
			const double mass,
			const potn_t&potn
		)noexcept
		{	
			oper_s.initialize(radi);
			oper_h.initialize(radi,mass,potn);
		}//end of initialize

		inline	void	multiply	(const comp_t* u0,comp_t* u1,const size_t nthreads)noexcept
		{
			#pragma omp parallel for collapse(2) num_threads(nthreads)
			for(size_t im=0;im<dims_t::m_dims;++im)
			for(size_t il=0;il<dims_t::l_dims;++il)
			{
				size_t 	shift	=	dims_t::n_dims*dims_t::l_dims*im+dims_t::n_dims*il;
				auto 	u0vec	=	u0+shift;	
				auto 	u1vec	=	u1+shift;
				intrinsic::symb_mul_vecd<dims_t::n_dims,dims_t::n_elem,0>(oper_h.sub(im,il),u0vec,u1vec);
			}
		}//end of multiply
	
		template<int mode,class...args_t>
		inline	int	propagate	(args_t&&...args)
		{
			if constexpr(mode==1)
			{
				return 	this->template propagate1<propagator_hc<dims_t>>(std::forward<args_t>(args)...);
			}
		}//end of propagate
};//end of propagator_hc

template<class dims_t>
struct	propagator_hc_lgwz:public propagator_arnoldi<dims_t>
{
	public:
		static_assert(dims_t::i_type==0,"this propagator only applies to spherical symmtric system.");
	
		operator_hc<dims_t>	oper_h;
		operator_sc<dims_t>	oper_s;
		operator_rpow1<dims_t>	oper_r;
		operator_y10<dims_t>	oper_y;
		double			ez;
	
		template<class...args_t>
		explicit propagator_hc_lgwz(args_t&&...args):propagator_arnoldi<dims_t>(std::forward<args_t>(args)...){}

		template<class potn_t>
		inline	void	initialize	
		(	
			const integrator_radi<dims_t>& radi,
			const integrator_angu<dims_t>& angu,
			const double mass,
			const potn_t potn
		)noexcept
		{	
			oper_h.initialize(radi,mass,potn);
			oper_s.initialize(radi);
			oper_r.initialize(radi);
			oper_y.initialize(angu);
		}//end of initialize

		inline	void	multiply	(const comp_t* u0,comp_t* u1,const size_t nthreads)noexcept
		{
			#pragma omp parallel for collapse(2) num_threads(nthreads)
			for(size_t im=0;im<dims_t::m_dims;++im)
			for(size_t il=0;il<dims_t::l_dims;++il)
			{
				auto u0vec	=	u0+dims_t::n_dims*dims_t::l_dims*im+dims_t::n_dims*il;	
				auto u1vec	=	u1+dims_t::n_dims*dims_t::l_dims*im+dims_t::n_dims*il;
				intrinsic::symb_mul_vecd<dims_t::n_dims,dims_t::n_elem,0>(oper_h.sub(im,il),u0vec,u1vec);
				if(il>0)
				intrinsic::symb_mul_vecd<dims_t::n_dims,dims_t::n_elem,1>(oper_r.sub(),u0vec-dims_t::n_dims,u1vec,ez*oper_y(im,il-1));
				if(il<dims_t::l_dims-1)
				intrinsic::symb_mul_vecd<dims_t::n_dims,dims_t::n_elem,1>(oper_r.sub(),u0vec+dims_t::n_dims,u1vec,ez*oper_y(im,il));
			}
		}//end of multiply

		template<int mode,class...args_t>
		inline	int	propagate	(args_t&&...args)
		{
			if constexpr(mode==1)
			{
				return 	this->template propagate1<propagator_hc_lgwz<dims_t>>(std::forward<args_t>(args)...);
			}
		}
};//end of propagator_hc_lgwz

template<class dims_t>
struct 	propagator_hc_vgwz:public propagator_arnoldi<dims_t>
{
	public:
		static_assert(dims_t::i_type==0,"this propagator only applies to spherical symmtric system.");

		operator_hc<dims_t>	oper_h;
		operator_sc<dims_t>	oper_s;
		operator_rinv1<dims_t> 	oper_f;	//1/r
		operator_rdif1<dims_t>	oper_g;	//d/dr
		operator_y10<dims_t>	oper_y;	
		operator_q10<dims_t>	oper_q;
		double			az;

		template<class...args_t>
		explicit propagator_hc_vgwz(args_t&&...args):propagator_arnoldi<dims_t>(std::forward<args_t>(args)...){}

		template<class potn_t>
		inline	void	initialize	
		(	
			const integrator_radi<dims_t>& radi,
			const integrator_angu<dims_t>& angu,
			const double mass,
			const potn_t&potn
		)noexcept
		{	
			oper_h.initialize(radi,mass,potn);
			oper_s.initialize(radi);
			oper_f.initialize(radi);
			oper_g.initialize(radi);
			oper_y.initialize(angu);
			oper_q.initialize(angu);
		}//end of initialize

		inline	void	multiply	(const comp_t* u0,comp_t* u1,const size_t nthreads)noexcept
		{
			comp_t	iaz	=	az*unim<comp_t>;
			#pragma omp parallel for collapse(2) num_threads(nthreads)
			for(size_t im=0;im<dims_t::m_dims;++im)
			for(size_t il=0;il<dims_t::l_dims;++il)
			{
				auto u0vec	=	u0+dims_t::n_dims*dims_t::l_dims*im+dims_t::n_dims*il;	
				auto u1vec	=	u1+dims_t::n_dims*dims_t::l_dims*im+dims_t::n_dims*il;
				intrinsic::symb_mul_vecd<dims_t::n_dims,dims_t::n_elem,0>(oper_h.sub(im,il),u0vec,u1vec);
				if(il>0){
				intrinsic::symb_mul_vecd<dims_t::n_dims,dims_t::n_elem,1>(oper_f.sub(),u0vec-dims_t::n_dims,u1vec, iaz*oper_q(im,il-1));//note: q is asymmetric
				intrinsic::asyb_mul_vecd<dims_t::n_dims,dims_t::n_elem,1>(oper_g.sub(),u0vec-dims_t::n_dims,u1vec,-iaz*oper_y(im,il-1));}
				if(il<dims_t::l_dims-1){
				intrinsic::asyb_mul_vecd<dims_t::n_dims,dims_t::n_elem,1>(oper_g.sub(),u0vec+dims_t::n_dims,u1vec,-iaz*oper_y(im,il));
				intrinsic::symb_mul_vecd<dims_t::n_dims,dims_t::n_elem,1>(oper_f.sub(),u0vec+dims_t::n_dims,u1vec,-iaz*oper_q(im,il));}//note: q is asymmetric
			}
		}//end of multiply

		template<int mode,class...args_t>
		inline	int	propagate	(args_t&&...args)
		{
			if constexpr(mode==1)
			{
				return 	this->template propagate1<propagator_hc_vgwz<dims_t>>(std::forward<args_t>(args)...);
			}
		}
};//end of propagator_hc_vgwz

template<class dims_t>
struct	propagator_hc_lgwr:public propagator_arnoldi<dims_t>
{
	public:
		static_assert(dims_t::i_type==0,"this propagator only applies to spherical symmtric system.");

		operator_hc<dims_t>	oper_h;
		operator_sc<dims_t>	oper_s;
		operator_rpow1<dims_t>	oper_r;
		operator_y11<dims_t>	oper_y;//ym
		double			ex;
		double			ey;

		template<class...args_t>
		explicit propagator_hc_lgwr(args_t&&...args):propagator_arnoldi<dims_t>(std::forward<args_t>(args)...){}

		template<class potn_t>
		inline	void	initialize	
		(	
			const integrator_radi<dims_t>& radi,
			const integrator_angu<dims_t>& angu,
			const double mass,
			const potn_t&potn
		)noexcept
		{	
			oper_h.initialize(radi,mass,potn);
			oper_s.initialize(radi);
			oper_r.initialize(radi);
			oper_y.initialize(angu);
		}//end of initialize

		inline	void	multiply	(const comp_t* u0,comp_t* u1,const size_t nthreads)noexcept
		{
			comp_t	ep	=	-(ex+ey*unim<comp_t>)/2.;//(Ex+iEy)/(-2),the prefactor of sin*exp+
			comp_t	em	=	-(ex-ey*unim<comp_t>)/2.;//(Ex-iEy)/(-2),the prefactor of sin*exp-
			#pragma omp parallel for num_threads(nthreads)
			for(size_t im=0;im<dims_t::m_dims;++im)
			{
				auto	u0vecm	=	u0+dims_t::n_dims*dims_t::l_dims*im;//im-th m-subspace in u0, work as rhs
				auto	u1vecm	=	u1+dims_t::n_dims*dims_t::l_dims*im;//im-th m-subspace in u1, work as lhs
				auto	u0vecml	=	u0vecm-dims_t::n_dims*dims_t::l_dims;//im-1 -th m-subspace in u0, 'l' means left
				auto	u0vecmr	=	u0vecm+dims_t::n_dims*dims_t::l_dims;//im+1 -th m-subspace in u0, 'r' means right
				
				for(size_t il=0;il<dims_t::l_dims;++il)
				{
					long m1,l1;
					size_t im2,il2;
					dims_t::in(im,il,m1,l1);
					auto u0vec	=	u0vecm+dims_t::n_dims*il;//il-th l-subspace in u0(im)
					auto u1vec	=	u1vecm+dims_t::n_dims*il;//il-th l-subspace in u1(im)
					intrinsic::symb_mul_vecd<dims_t::n_dims,dims_t::n_elem,0>(oper_h[l1],u0vec,u1vec);//Hc
				
					if(dims_t::template in_check<+1ul,+1ul>(im,il,im2,il2))
					{
						intrinsic::symb_mul_vecd<dims_t::n_dims,dims_t::n_elem,1>(oper_r.sub(),u0vecmr+il2*dims_t::n_dims,u1vec,ep*oper_y.outer(im,il));
					}
					if(dims_t::template in_check<+1ul,-1ul>(im,il,im2,il2))
					{
						intrinsic::symb_mul_vecd<dims_t::n_dims,dims_t::n_elem,1>(oper_r.sub(),u0vecmr+il2*dims_t::n_dims,u1vec,ep*oper_y.inner(im,il));
					}
					if(dims_t::template in_check<-1ul,+1ul>(im,il,im2,il2))
					{
						intrinsic::symb_mul_vecd<dims_t::n_dims,dims_t::n_elem,1>(oper_r.sub(),u0vecml+il2*dims_t::n_dims,u1vec,em*oper_y.inner(im2,il2));//TODO CHECKME
					}
					if(dims_t::template in_check<-1ul,-1ul>(im,il,im2,il2))
					{
						intrinsic::symb_mul_vecd<dims_t::n_dims,dims_t::n_elem,1>(oper_r.sub(),u0vecml+il2*dims_t::n_dims,u1vec,em*oper_y.outer(im2,il2));//TODO CHECKME
					}
				}
			}
		}//end of multiply
		
			
		template<int mode,class...args_t>
		inline	int	propagate	(args_t&&...args)
		{
			if constexpr(mode==1)
			{
				return 	this->template propagate1<propagator_hc_lgwr<dims_t>>(std::forward<args_t>(args)...);
			}
		}//end of propagate
};//end of propagator_hc_lgwr

template<class dims_t>
struct	propagator_hc_vgwr:public propagator_arnoldi<dims_t>
{
	public:
		static_assert(dims_t::i_type==0,"this propagator only applies to spherical symmtric system.");

		operator_hc<dims_t>     oper_h;
                operator_sc<dims_t>     oper_s;
		operator_rdif1<dims_t>	oper_dr;
                operator_rinv1<dims_t>  oper_ir;
                operator_y11<dims_t,+1> oper_yp;
                operator_y11<dims_t,-1> oper_ym;
                operator_q11<dims_t,+1> oper_qp;
                operator_q11<dims_t,-1> oper_qm;
                double                  ax;
                double                  ay;
		
		template<class...args_t>
		explicit propagator_hc_vgwr(args_t&&...args):propagator_arnoldi<dims_t>(std::forward<args_t>(args)...){}

		template<class potn_t>
                inline  void    initialize
                (
              		const integrator_radi<dims_t>& radi,
        		const integrator_angu<dims_t>& angu,
         		const double mass,
          		const potn_t&potn
                )noexcept
                {
                        oper_h.initialize(radi,mass,potn);
                        oper_s.initialize(radi);
                        oper_dr.initialize(radi);
                        oper_ir.initialize(radi);
                        oper_yp.initialize(angu);
                        oper_ym.initialize(angu);
                        oper_qp.initialize(angu);
                        oper_qm.initialize(angu);
                }//end of initialize
		
		inline	void	multiply	(const comp_t* u0,comp_t* u1,const size_t nthreads)noexcept
		{
			comp_t	iap	=	-unim<comp_t>*(ax+ay*unim<comp_t>)/2.;//the prefactor
			comp_t	iam	=	-unim<comp_t>*(ax-ay*unim<comp_t>)/2.;//the prefactor
			#pragma omp parallel for num_threads(nthreads)
			for(size_t im=0;im<dims_t::m_dims;++im)
			{
				auto	u0vecm	=	u0+dims_t::n_dims*dims_t::l_dims*im;//im-th m-subspace in u0
				auto	u1vecm	=	u1+dims_t::n_dims*dims_t::l_dims*im;//im-th m-subspace in u1
				auto	u0vecml	=	u0vecm-dims_t::n_dims*dims_t::l_dims;//im-1 -th m-subspace in u0, 'l' means left
				auto	u0vecmr	=	u0vecm+dims_t::n_dims*dims_t::l_dims;//im+1 -th m-subspace in u0, 'r' means right
				
				for(size_t il=0;il<dims_t::l_dims;++il)
				{
					long m1,l1;
					size_t im2,il2;
					dims_t::in(im,il,m1,l1);
					auto u0vec	=	u0vecm+dims_t::n_dims*il;//il-th l-subspace in u0(im);
					auto u1vec	=	u1vecm+dims_t::n_dims*il;//il-th l-subspace in u1(im);
					intrinsic::symb_mul_vecd<dims_t::n_dims,dims_t::n_elem,0>(oper_h[l1],u0vec,u1vec);//Hc
				
					if(dims_t::template in_check<+1ul,+1ul>(im,il,im2,il2))
					{
						intrinsic::asyb_mul_vecd<dims_t::n_dims,dims_t::n_elem,1>(oper_dr.sub(),u0vecmr+il2*dims_t::n_dims,u1vec,iap*oper_ym.outer(im,il));
						intrinsic::symb_mul_vecd<dims_t::n_dims,dims_t::n_elem,1>(oper_ir.sub(),u0vecmr+il2*dims_t::n_dims,u1vec,iap*oper_qm.outer(im,il));
					}
					if(dims_t::template in_check<+1ul,-1ul>(im,il,im2,il2))
					{
						intrinsic::symb_mul_vecd<dims_t::n_dims,dims_t::n_elem,1>(oper_ir.sub(),u0vecmr+il2*dims_t::n_dims,u1vec,iap*oper_qm.inner(im,il));
						intrinsic::asyb_mul_vecd<dims_t::n_dims,dims_t::n_elem,1>(oper_dr.sub(),u0vecmr+il2*dims_t::n_dims,u1vec,iap*oper_ym.inner(im,il));
					}
					if(dims_t::template in_check<-1ul,+1ul>(im,il,im2,il2))
					{
						intrinsic::asyb_mul_vecd<dims_t::n_dims,dims_t::n_elem,1>(oper_dr.sub(),u0vecml+il2*dims_t::n_dims,u1vec,iam*oper_yp.inner(im,il));
						intrinsic::symb_mul_vecd<dims_t::n_dims,dims_t::n_elem,1>(oper_ir.sub(),u0vecml+il2*dims_t::n_dims,u1vec,iam*oper_qp.inner(im,il));
					}
					if(dims_t::template in_check<-1ul,-1ul>(im,il,im2,il2))
					{
						intrinsic::symb_mul_vecd<dims_t::n_dims,dims_t::n_elem,1>(oper_ir.sub(),u0vecml+il2*dims_t::n_dims,u1vec,iam*oper_qp.outer(im,il));
						intrinsic::asyb_mul_vecd<dims_t::n_dims,dims_t::n_elem,1>(oper_dr.sub(),u0vecml+il2*dims_t::n_dims,u1vec,iam*oper_yp.outer(im,il));
					}
				}
			}
		}//end of multiply

		template<int mode,class...args_t>
                inline  int     propagate       (args_t&&...args)
                {
                        if constexpr(mode==1)
                        {
                                return  this->template propagate1<propagator_hc_vgwr<dims_t>>(std::forward<args_t>(args)...);
                        }
                }//end of propagate
};//end of propagator_hc_vgwr
//================================================================================================================================
//
//						Propagator User Class (Multipolar)
//
//================================================================================================================================
template<class dims_t,size_t n_pole>
struct	propagator_hm:public propagator_arnoldi<dims_t>
{
	public:
		operator_hm<dims_t,n_pole> 	oper_h;
		operator_sc<dims_t>		oper_s;

		template<class...args_t>
		explicit propagator_hm(args_t&&...args):propagator_arnoldi<dims_t>(std::forward<args_t>(args)...){} 

		template<class...init_t>
		inline	void	initialize	//you should call oper_h::set_pole first, see operator.hpp
		(
			const integrator_radi<dims_t>& radi,
			const integrator_angu<dims_t>& angu,
			const double mass
		)
		{
			oper_s.initialize(radi);
			oper_h.template initialize<3>(radi,angu,mass);//3dec=11bin= box hmat and hmul are prepared
		}//end of initialize

		inline	void	multiply	(const comp_t* u0,comp_t* u1,const size_t nthreads)
		{
			auto 	h	=	oper_h.get_hmul();
			#pragma omp parallel for num_threads(nthreads)
			for(size_t iml=0;iml<dims_t::m_dims*dims_t::l_dims;++iml)
			{
				comp_t* dst =	u1+iml*dims_t::n_dims;
				size_t  ish = 	(2ul*dims_t::m_dims*dims_t::l_dims-1ul-iml)*iml/2ul;
				//diagonal entry
				{
					intrinsic::symb_mul_vecd<dims_t::n_dims,dims_t::n_elem,0>(h(ish+iml).data,u0+iml*dims_t::n_dims,dst);	
				}
				//lower panel (utilize zhp format)
				for(size_t jml=0;jml<iml;++jml)
				{
					size_t jsh=(2ul*dims_t::m_dims*dims_t::l_dims-1ul-jml)*jml/2ul;
					intrinsic::symb_cmul_vecd<dims_t::n_dims,dims_t::n_elem,1>(h(jsh+iml).data,u0+jml*dims_t::n_dims,dst);
				}
				//upper panel
				for(size_t jml=1+iml;jml<dims_t::m_dims*dims_t::l_dims;++jml)
				{
					intrinsic::symb_mul_vecd<dims_t::n_dims,dims_t::n_elem,1>(h(ish+jml).data,u0+jml*dims_t::n_dims,dst);
				}
			}
		}//end of multiply

		template<int mode,class...args_t>
                inline  int     propagate       (args_t&&...args)
                {
                        if constexpr(mode==1)
                        {
                                return  this->template propagate1<propagator_hm<dims_t>>(std::forward<args_t>(args)...);
                        }
                }//end of propagate
};//end of propagator_hm

//================================================================================================================================
//
//						Propagator User Class (Bic)
//
//================================================================================================================================
template<class dims_t,size_t n_asym>
struct	propagator_hb:public propagator_arnoldi<dims_t>
{
	public:
		static_assert(dims_t::i_type==1,"this propagator only applies to spheroidal symmtric system.");

		operator_hb<dims_t,n_asym>	oper_h;
		operator_sb<dims_t>		oper_s;	

		template<class...args_t>
		explicit propagator_hb(args_t&&...args):propagator_arnoldi<dims_t>(std::forward<args_t>(args)...){}

		template<class zadd_t,class...zsub_t>
		inline	void	initialize	
		(	
			const integrator_radi<dims_t>& radi,
			const integrator_angu<dims_t>& angu,
			const double dist,
			const double mass,
			const zadd_t&zadd,
			const zsub_t&...zsub
		)noexcept
		{	
			oper_s.initialize(radi,angu,dist);
			if constexpr(n_asym==1)
			oper_h.initialize(radi,angu,dist,mass,zadd);
			if constexpr(n_asym==2)
			oper_h.initialize(radi,angu,dist,mass,zadd,zsub...);	
		}//end of initialize

		inline	void	multiply	(const comp_t* u0,comp_t* u1,const size_t nthreads)noexcept
		{
			if constexpr(n_asym==1)//Z1==Z2 version
			{
				#pragma omp parallel for collapse(2) num_threads(nthreads)
				for(size_t im=0;im<dims_t::m_dims;++im)
				for(size_t il=0;il<dims_t::l_dims;++il)
				{
					size_t	shift	=	dims_t::n_dims*dims_t::l_dims*im+dims_t::n_dims*il;
					auto 	u0vec	=	u0+shift;	
					auto 	u1vec	=	u1+shift;
					intrinsic::symb_mul_vecd<dims_t::n_dims,dims_t::n_elem,0>(oper_h.sub(im,il,_ul<0>).data,u0vec,u1vec);
				}
			}else//Z1!=Z2 version
			{
				#pragma omp parallel for collapse(2) num_threads(nthreads)
				for(size_t im=0;im<dims_t::m_dims;++im)
				for(size_t il=0;il<dims_t::l_dims;++il)
				{
					size_t  shift   =       dims_t::n_dims*dims_t::l_dims*im+dims_t::n_dims*il;
					auto    u0vec   =       u0+shift;
					auto    u1vec   =       u1+shift;
					intrinsic::symb_mul_vecd<dims_t::n_dims,dims_t::n_elem,0>(oper_h.sub(im,il  ,_ul<0>).data,u0vec,u1vec);
					if(il>0ul)
					intrinsic::symb_mul_vecd<dims_t::n_dims,dims_t::n_elem,1>(oper_h.sub(im,il-1,_ul<1>).data,u0vec-dims_t::n_dims,u1vec);
					if(il+1ul<dims_t::l_dims)
					intrinsic::symb_mul_vecd<dims_t::n_dims,dims_t::n_elem,1>(oper_h.sub(im,il  ,_ul<1>).data,u0vec+dims_t::n_dims,u1vec);
				}
			}
		}//end of multiply
	
		template<int mode,class...args_t>
		inline	int	propagate	(args_t&&...args)
		{
			if constexpr(mode==1)
			{
				return 	this->template propagate1<propagator_hb<dims_t,n_asym>>(std::forward<args_t>(args)...);
			}
		}//end of propagate
};//end of propagator_hb

template<class dims_t,size_t n_asym>
struct	propagator_hb_lgwz:public propagator_arnoldi<dims_t>
{
	public:
		static_assert(dims_t::i_type==1,"this propagator only applies to spheroidal symmtric system.");

		operator_hb<dims_t,n_asym>	oper_h;
		operator_sb<dims_t>		oper_s;	

		operator_xsub<dims_t,2>		oper_xn;

		operator_eta1<dims_t>		oper_e1;
		operator_eta3<dims_t> 		oper_e3;

		double	ez;

		template<class...args_t>
		explicit propagator_hb_lgwz(args_t&&...args):propagator_arnoldi<dims_t>(std::forward<args_t>(args)...){}

		template<class zadd_t,class...zsub_t>
		inline	void	initialize	
		(	
			const integrator_radi<dims_t>& radi,
			const integrator_angu<dims_t>& angu,
			const double dist,
			const double mass,
			const zadd_t&zadd,
			const zsub_t&...zsub
		)noexcept
		{
			double	ro=dist*dist*dist*dist/16.;//R^4/16	
			oper_s.initialize(radi,angu,dist);
			if constexpr(n_asym==1)
			oper_h.initialize(radi,angu,dist,mass,zadd);
			if constexpr(n_asym==2)
			oper_h.initialize(radi,angu,dist,mass,zadd,zsub...);
			oper_xn.initialize(radi,[&](double x){return x    ;},0ul);//<xi^1>*R^4/16
			oper_xn.initialize(radi,[&](double x){return x*x*x;},1ul);//<xi^3>*R^4/16
			oper_xn*=ro;
			oper_e1.initialize(angu);//<eta^1>
			oper_e3.initialize(angu);//<eta^3>
		}//end of initialize

		inline	void	multiply	(const comp_t* u0,comp_t* u1,const size_t nthreads)noexcept
		{
			#pragma omp parallel for collapse(2) num_threads(nthreads)
			for(size_t im=0;im<dims_t::m_dims;++im)
			for(size_t il=0;il<dims_t::l_dims;++il)
			{
				size_t	shift	=	dims_t::n_dims*dims_t::l_dims*im+dims_t::n_dims*il;
				auto 	u0vec	=	u0+shift;	
				auto 	u1vec	=	u1+shift;
				auto	x1mat	=	oper_xn.sub(dims_t::od_m(im),_ul<0>);
				auto	x3mat	=	oper_xn.sub(dims_t::od_m(im),_ul<1>);
				intrinsic::symb_mul_vecd<dims_t::n_dims,dims_t::n_elem,0>(oper_h.sub(im,il,0).data,u0vec,u1vec);
				if(il>2ul)
				{
					intrinsic::symb_mul_vecd<dims_t::n_dims,dims_t::n_elem,2>(x1mat.data,u0vec-3ul*dims_t::n_dims,u1vec,ez*oper_e3.sh3(im,il-3ul));//-=e^3*x^1
				}
				if(il>0ul)
				{
					intrinsic::symb_mul_vecd<dims_t::n_dims,dims_t::n_elem,2>(x1mat.data,u0vec-dims_t::n_dims,u1vec,ez*oper_e3.sh1(im,il-1ul));//-=e^3*x^1
					intrinsic::symb_mul_vecd<dims_t::n_dims,dims_t::n_elem,1>(x3mat.data,u0vec-dims_t::n_dims,u1vec,ez*oper_e1.sh1(im,il-1ul));//+=e^1*x^3
					if constexpr(n_asym==2)
					intrinsic::symb_mul_vecd<dims_t::n_dims,dims_t::n_elem,1>(oper_h.sub(im,il-1ul,1).data,u0vec-dims_t::n_dims,u1vec);
				}
				if(il+1ul<dims_t::l_dims)
				{
					intrinsic::symb_mul_vecd<dims_t::n_dims,dims_t::n_elem,1>(x3mat.data,u0vec+dims_t::n_dims,u1vec,ez*oper_e1.sh1(im,il));//+=e^1*x^3
					intrinsic::symb_mul_vecd<dims_t::n_dims,dims_t::n_elem,2>(x1mat.data,u0vec+dims_t::n_dims,u1vec,ez*oper_e3.sh1(im,il));//-=e^3*x^1
					if constexpr(n_asym==2)
					intrinsic::symb_mul_vecd<dims_t::n_dims,dims_t::n_elem,1>(oper_h.sub(im,il,1).data,u0vec+dims_t::n_dims,u1vec);
				}
				if(il+3ul<dims_t::l_dims)
				{
					intrinsic::symb_mul_vecd<dims_t::n_dims,dims_t::n_elem,2>(x1mat.data,u0vec+3ul*dims_t::n_dims,u1vec,ez*oper_e3.sh3(im,il));//-=e^3*x^1
				}
			}
		}//end of multiply
	
		template<int mode,class...args_t>
		inline	int	propagate	(args_t&&...args)
		{
			if constexpr(mode==1)
			{
				return 	this->template propagate1<propagator_hb_lgwz<dims_t,n_asym>>(std::forward<args_t>(args)...);
			}
		}//end of propagate
};//end of propagator_hb_lgwz

template<class dims_t,size_t n_asym>
struct	propagator_hb_vgwz:public propagator_arnoldi<dims_t>
{
	public:
		static_assert(dims_t::i_type==1,"this propagator only applies to spheroidal system.");

		operator_hb<dims_t,n_asym>	oper_h;
                operator_sb<dims_t>     	oper_s;

                operator_xsub<dims_t,2> oper_xn;

                operator_eta1<dims_t>   oper_e1;
                operator_chi1<dims_t>   oper_c1;	
	
		double	az;

		template<class...args_t>
		explicit propagator_hb_vgwz(args_t&&...args):propagator_arnoldi<dims_t>(std::forward<args_t>(args)...){}

		template<class zadd_t,class...zsub_t>
		inline	void	initialize	
		(	
			const integrator_radi<dims_t>& radi,
			const integrator_angu<dims_t>& angu,
			const double dist,
			const double mass,
			const zadd_t&zadd,
			const zsub_t&...zsub
		)noexcept
		{	
			double	ro=-dist*dist/4.;//-R^2/4
			oper_s.initialize(radi,angu,dist);
			if constexpr(n_asym==1)
			oper_h.initialize(radi,angu,dist,mass,zadd);
			if constexpr(n_asym==2)
			oper_h.initialize(radi,angu,dist,mass,zadd,zsub...);
			oper_xn.initialize(radi,[&](double x){return x;},0ul);//<x>*(-R^2/4)
			oper_xn.initialize(radi,                         1ul);//<sqrt(x^2-1)(d/dx)sqrt(x^2-1)>*(-R^2/4)
			oper_xn*=ro;
			oper_e1.initialize(angu);//<eta>
			oper_c1.initialize(angu);//<chi>
		}//end of initialize

		inline	void	multiply	(const comp_t* u0,comp_t* u1,const size_t nthreads)noexcept
		{
			comp_t	iaz=unim<comp_t>*az;
			#pragma omp parallel for collapse(2) num_threads(nthreads)
			for(size_t im=0;im<dims_t::m_dims;++im)
			for(size_t il=0;il<dims_t::l_dims;++il)
			{
				size_t	shift	=	dims_t::n_dims*dims_t::l_dims*im+dims_t::n_dims*il;
				auto 	u0vec	=	u0+shift;	
				auto 	u1vec	=	u1+shift;
				auto	x1mat	=	oper_xn.sub(dims_t::od_m(im),_ul<0>);
				auto	d1mat	=	oper_xn.sub(dims_t::od_m(im),_ul<1>);
				intrinsic::symb_mul_vecd<dims_t::n_dims,dims_t::n_elem,0>(oper_h.sub(im,il,0).data,u0vec,u1vec);
				if(il>0ul)
				{
					intrinsic::asyb_mul_vecd<dims_t::n_dims,dims_t::n_elem,1>(d1mat.data,u0vec-dims_t::n_dims,u1vec, iaz*oper_e1.sh1(im,il-1ul));
					intrinsic::symb_mul_vecd<dims_t::n_dims,dims_t::n_elem,1>(x1mat.data,u0vec-dims_t::n_dims,u1vec,-iaz*oper_c1.sh1(im,il-1ul));
					if constexpr(n_asym==2)
					intrinsic::symb_mul_vecd<dims_t::n_dims,dims_t::n_elem,1>(oper_h.sub(im,il-1ul,1).data,u0vec-dims_t::n_dims,u1vec);
				}
				if(il+1ul<dims_t::l_dims)
				{
					intrinsic::symb_mul_vecd<dims_t::n_dims,dims_t::n_elem,1>(x1mat.data,u0vec+dims_t::n_dims,u1vec, iaz*oper_c1.sh1(im,il));
					intrinsic::asyb_mul_vecd<dims_t::n_dims,dims_t::n_elem,1>(d1mat.data,u0vec+dims_t::n_dims,u1vec, iaz*oper_e1.sh1(im,il));
					if constexpr(n_asym==2)
					intrinsic::symb_mul_vecd<dims_t::n_dims,dims_t::n_elem,1>(oper_h.sub(im,il,1).data,u0vec+dims_t::n_dims,u1vec);
				}
			}
		}//end of multiply
		

		template<int mode,class...args_t>
                inline  int     propagate       (args_t&&...args)
                {
                        if constexpr(mode==1)
                        {
                                return  this->template propagate1<propagator_hb_vgwz<dims_t,n_asym>>(std::forward<args_t>(args)...);
                        }
                }//end of propagate
};//end of propagator_hb_vgwz



