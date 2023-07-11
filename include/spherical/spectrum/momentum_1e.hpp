#pragma once

//#include <functions/basis_bsplines.h>
//#include <functions/basis_legendre.h>
//#include <functions/basis_coulombf.h>
//#include <functions/spheroidal.h>

//---------------------------------------------------------------------------------------------------------------------------------------------
// some helper functions
//---------------------------------------------------------------------------------------------------------------------------------------------
template<class dims_t,class spec_t>
static	void	initialize_ph	(spec_t& spec)noexcept//require spec.{nkp,kp,exm(*,*)}
{//component exm
	#pragma omp parallel
	{
		#pragma omp for simd
		for(size_t iph=0;iph<spec.nkp;++iph)
		{
			double	ph	=	spec.kp[iph];
			for(size_t im=0;im<dims_t::m_dims;++im)
			{
				double mph=ph*dims_t::in_m(im);
				spec.exm(iph,im)=comp_t{cos(mph),sin(mph)};
			}
		}
	}
}//end of initialize_ph

template<class dims_t,class spec_t>
static	void	initialize_th	(spec_t& spec)noexcept//require spec.{nkt,kt,plm(*,*,*)}
{//component plm, by calling GSL
	using	lgd	=	bss::spherical_harmonics::legendre<dims_t::n_lmax>;
	#pragma omp parallel
	{
		auto 	val	=	std::vector<double>(lgd::max());
		#pragma omp for simd
		for(size_t ith=0;ith<spec.nkt;++ith)
		{
			double	costh	=	cos(spec.kt[ith]);
			lgd::cal(costh,val.data());//note: the normalization is the spherical harmonic one, see GSL mannual
			for(size_t im=0;im<dims_t::m_dims;++im)
			for(size_t il=0;il<dims_t::l_dims;++il)
			{
				long l,m;dims_t::in(im,il,m,l);
				spec.plm(ith,im,il)=lgd::get(val.data(),l,m);
			}
		}	
	}//end of pragma omp
}//end of initialize_th

template<class dims_t,class spec_t>
static	void	initialize_et	(spec_t& spec,const integrator_angu<dims_t>& angu,double z1,double z2,double rn)//require spec.{nkr,kr,nkt,plm,aqm,sqm,j0mql,j2mql}
{
	static_assert(dims_t::l_dims>=2ul,"initialize_et: l-dimension must be at least 2.");

	int 	info	=	0;
	auto	sh0	=	operator_angu<dims_t>();//to cache diagonal part of <eta^2>
	auto	sh2	=	operator_angu<dims_t>();//to cache off-diagonal part of <eta^2>	
	angu.integrate_eta2_sh0(sh0);
	angu.integrate_eta2_sh2(sh2);
	#pragma omp parallel reduction(+:info)
	{
		auto	work	=	std::vector<double>(dims_t::l_dims*3ul);
		auto	eigv	=	std::vector<double>(dims_t::l_dims*dims_t::l_dims);
		#pragma omp for collapse(2)
		for(size_t ikr=0;ikr<spec.nkr;++ikr)
		for(size_t im=0;im<dims_t::m_dims;++im)
		{
			double	_m	=	double(dims_t::in_m(im));//M
			double	_k	=	spec.kr[ikr]*rn/2.0;	//Kappa=kR/2
			double	_d	=	(z1-z2)*rn;		//Dsub
				info	+=	spheroidal_eigval_mat<double,dims_t::l_dims>//obtain A, a by diagonalizing Da=Aa
			(
				_m,_k,_d,
				spec.aqm(ikr,im).data,
				eigv.data(),
				work.data()
			);
			auto	j0ql	=	spec.j0mql(ikr,im);
			auto	j2ql	=	spec.j2mql(ikr,im);
			for(size_t iq=0;iq<dims_t::l_dims;++iq)
			{
				auto	temp	=	eigv.data()+iq*dims_t::l_dims;//q-th eigen vector
				//j0_ql=a_ql
				for(size_t il=0;il<dims_t::l_dims;++il)
				j0ql(iq,il)	=	temp[il];
				//j2_ql=T_ll'a_ql'
				for(size_t il=0;il<2;++il)//ensure l_dims>2
				j2ql(iq,il)	=	temp[     il]*sh0(im,il)
						+	temp[+2ul+il]*sh2(im,il);
				for(size_t il=2;il<dims_t::l_dims-2;++il)
				j2ql(iq,il)	=	temp[     il]*sh0(im,il)
						+	temp[+2ul+il]*sh2(im,il)
						+	temp[-2ul+il]*sh2(im,il-2);
				for(size_t il=dims_t::l_dims-2;il<dims_t::l_dims;++il)
				j2ql(iq,il)	=	temp[     il]*sh0(im,il)
						+	temp[-2ul+il]*sh2(im,il-2);
			}//end of j0ql,j2ql*/

			for(size_t ikt=0;ikt<spec.nkt;++ikt)
			{
				auto	_sqm	=	spec.sqm(ikr,ikt,im);
				double	_sum;
				double*	_aql;
				for(size_t iq=0;iq<dims_t::l_dims;++iq)
				{	
					_sum	=	0.0;
					_aql	=	eigv.data()+iq*dims_t::l_dims;
					for(size_t il=0;il<dims_t::l_dims;++il)
					_sum	+=	_aql[il]*spec.plm(ikt,im,il);//you should calculate plm first
					_sqm(iq)=	_sum;
				}
			}//end of sqm*/
		}
	}
	if(info)throw errn_t{"lapacke failure in initialize_et."};
}//end of initialize_et

template<class dims_t,class spec_t>
static	void	initialize_kr	(spec_t& spec,const integrator_radi<dims_t>& radi,const double zc,const double rb=0.)noexcept//require spec.{nkr,kr,ijl(*,*,*),cps(*,*)}
{//component ijl (hydrogenic), use coulomb.h and basis_coulombf.h
	int info=0;
	#pragma omp parallel reduction(+:info)
	{
		auto	fgl	=	bss::coulomb<long double>(0.L,dims_t::n_lmax,radi.base.leng());
		#pragma omp for schedule(dynamic)
		for(size_t ikr=0;ikr<spec.nkr;++ikr)
		{
			info+=fgl.evaluate(zc,spec.kr[ikr],radi.base.axis());
			if(rb>0.)
			for(size_t i=0;i<fgl.leng();++i)
			{
				fgl.f[i] *= 1./(1.+exp(-5.*(radi.base.axis()[i%radi.base.leng()]-rb)));//project onto planewave by setting zc=0, rb>0 
			}
			for(size_t il=0;il<=dims_t::n_lmax;++il)
			for(size_t ij=0;ij< dims_t::n_dims;++ij)
			{
				spec.ijl(ikr,il,ij)=radi.base.template integrate<0>(fgl._fc(il),ij)/spec.kr[ikr];
			}
			double	eta=-zc/spec.kr[ikr];
			for(size_t l=0;l<=dims_t::n_lmax;++l)
			{
				spec.cps(ikr,l)=exp(unim<comp_t>*(coulomb_sig_real<double>(l,eta)-l*PI<double>/2.0));//exp(I*(Sigma-Pi*L/2))
			}
		}
	}
	if(info)printf("non-fatal warning: initialize_kr info=%d\n",info);//to warn possible error*/
}//end of initialize_kr (hydrogenic)
		
template<class dims_t,class spec_t,class odef_t>
static	void	initialize_kr	(spec_t& spec,const integrator_radi<dims_t>& radi,const odef_t& odef)noexcept//require spec.{nkr,kr,ijl(*,*,*),cps(*,*)}
{//component ijl (centrifugal)
	static_assert(std::is_trivially_copyable_v<odef_t>,"odefun object should be trivially copyable.");
	double*	r	=	radi.base.axis();

	size_t 	nzero	=	0ul;
	for(size_t i=0;i<radi.base.leng();++i)//count lowerside boundary points
	{
		if(r[i]==r[0])nzero++;
	}

	#pragma omp parallel
	{
		auto	eq	=	odef;//trivially copied to each thread
		auto	ul	=	solution<double>(radi.base.leng());//container and computational class of the radial solution, for each thread
		#pragma omp for schedule(dynamic)
		for(size_t ik=0;ik<spec.nkr;++ik)
		{
			double k=	spec.kr[ik];
			for(size_t ir=0;ir<nzero;++ir)
			{
				ul._rh(ir)=k*r[nzero];	//boundary points have zero weight. skip their evaluation.
			}
			for(size_t ir=nzero;ir<radi.base.leng();++ir)//prepare rho=k*r, r on gauss point
			{
				ul._rh(ir)=k*r[ir];
			}
			for(size_t l=0;l<=dims_t::n_lmax;++l)//solve for different L partial waves
			{
				eq.initialize_param(k,double(l));
				ul.initialize(eq);//evaluate solution by solving ode
				spec.cps(ik,l)=exp(unim<comp_t>*(ul._sh()-l*PI<double>/2.0));//phase shift
				for(size_t ij=0;ij<dims_t::n_dims;++ij)//do integral by summing up the integrands on gauss points
				spec.ijl(ik,l,ij)=radi.base.template integrate<0>(ul._u0(),ij)/k;//int<B_j(r)*u_l(kr)*dr>/k
			}
		}
	}
}//end of initialize_kr (centrifugal analytic)

template<class dims_t,class spec_t>
static	void	initialize_kx	(spec_t& spec,const integrator_radi<dims_t>& radi,double z1,double z2,double rn)noexcept//require spec.{nkr,kr,aqm,i0mqn,i2mqn,cqm}
{
	double	r3o8	=	rn*rn*rn/8;
	double	dadd	=	rn*(z1+z2);
	auto	leng	=	radi.base.leng();
	auto	axis	=	radi.base.axis();
	auto	tempm	=	std::vector<double>(leng*dims_t::m_dims);
	for(size_t im=0;im<dims_t::m_dims;++im)
	{
		long m	=	dims_t::in_m(im);
		for(size_t ii=0;ii<leng;++ii)
		{
			//if m even, [(Ld-1)/(Ld+1)]^(m/2)
			//if m odd , [(Ld-1)/(Ld+1)]^((m+1)/2)
			tempm[im*leng+ii]=pow((axis[ii]-1.0+1e-50)/(axis[ii]+1.0),(m+1)/2);
		}
	}
	#pragma omp parallel
	{
		auto	work	=	spheroidal_asympt_work<double>();//workspace for each thread
		auto	temp0	=	std::vector<double>(leng);
		auto	temp1	=	std::vector<double>(leng);
		#pragma omp for simd schedule(dynamic)
		for(size_t ikr=0;ikr<spec.nkr;++ikr)
		for(size_t im=0;im<dims_t::m_dims;++im)
		{
			double	kappa	=	spec.kr[ikr]*rn/2.0;
			double	m	=	dims_t::in_m(im);
			auto 	am	=	spec.aqm.sub(ikr,im);
			auto	cm	=	spec.cqm.sub(ikr,im);
			auto	i0m	=	spec.i0mqn.sub(ikr,im);
			auto	i2m	=	spec.i2mqn.sub(ikr,im);
			for(size_t iq=0;iq<dims_t::l_dims;++iq)
			{
				double	sh;//computed as Y. S. Tergiman's theta.
				int	info	=	spheroidal_direct_arr<double,2>
				(
					m,am(iq),kappa,dadd,axis,1e-14,sh,
					temp0.data(),temp1.data(),leng,work.data()
				);
				sh+=kappa;// theta+kappa. this step could actually be ignored if only absolute value of PMD is needed
				if(std::isinf(info))printf("initialize_kx: spheroidal error\n");//FIXME
				cm(iq)	=	exp(unim<comp_t>*sh);//note now sh=sigma-(m+q)*pi/2, where sigma is the conventional 'coulomb phase shift'
				for(size_t ii=0;ii<leng;++ii)
	 			{
					temp1[ii]=	temp0[ii]*tempm[im*leng+ii];
					temp0[ii]=	temp1[ii]*axis[ii]*axis[ii];
				}
				for(size_t in=0;in<dims_t::n_dims;++in)
				{
					i0m(iq,in)=	radi.base.template integrate<0>(temp1.data(),in)*r3o8;
					i2m(iq,in)=	radi.base.template integrate<0>(temp0.data(),in)*r3o8;
				}	
			}
		}
	}//e*/	
}//end of initialize_kx

//------------------------------------------------------------------------------------------------------
static	void	self_check(const double* target,size_t length,const char* name)noexcept
{//check NAN/INF for prepared real data
	for(size_t i=0;i<length;++i)
	{
		if(std::isnan(target[i])||std::isinf(target[i]))
		{
			printf("nan or inf detected in %ld-th element of %s\n",i,name);return;
		}
	}
}//end of self_check
static 	inline  void	self_check(const comp_t* target,size_t length,const char* name)noexcept
{//check NAN/INF for prepared comp data
	for(size_t i=0;i<length;++i)
	{
		if(std::isnan(target[i][0])||std::isinf(target[i][0]))
		{
			printf("nan or inf detected in real %ld-th element of %s\n",i,name);return;
		}
		if(std::isnan(target[i][1])||std::isinf(target[i][1]))
		{
			printf("nan or inf detected in imag %ld-th element of %s\n",i,name);return;
		}
	}
}//end of self_check

//------------------------------------------------------------------------------------------------------
template<class dims_t,size_t _nkr,size_t _nkt,size_t _nkp>
struct	spectrum_c	//base class of spectrum_c_pcs
{
	public:
		using	work_t	=	std::vector<comp_t>;	
		using	coef_t	=	coefficient_view<dims_t>;

		static constexpr size_t	nkr=_nkr;		
		static constexpr size_t	nkt=_nkt;		
		static constexpr size_t	nkp=_nkp;

		double*	kr;	//kr[ikr]
		double*	kt;	//kt[ith]
		double*	kp;	//kp[iph]

		recvec<double,recidx<nkr,dims_t::n_lmax+1ul,dims_t::n_dims>>	ijl;
		recvec<double,recidx<nkt,dims_t::m_dims    ,dims_t::l_dims>> 	plm;//with SPH normalization
		recvec<comp_t,recidx<nkp,dims_t::m_dims>>			exm;//with SPH normalization
		recvec<comp_t,recidx<nkr,dims_t::n_lmax+1ul>>			cps;//for hydrogenic potential, it is exp[I*(argGamma(l+1+1i*eta)-l*pi/2)]

		work_t	spec;	//a workspace holding the result spectrum of PMD or PAD. only intialized when 'measure_xxx' functions are called.
		rsrc_t*	rsrc;

		//access axis
		inline	double*	axis_kr()const noexcept{return kr;}
		inline	double*	axis_kt()const noexcept{return kt;}
		inline	double*	axis_kp()const noexcept{return kp;}

		inline	size_t	size()const noexcept{return spec.size();}
		inline	double*	dptr()const noexcept{return (double*)spec.data();}

		//memory allocation
		spectrum_c(rsrc_t* _rsrc=global_default_arena_ptr):rsrc(_rsrc)
		{
			if(rsrc!=0)
			{
				rsrc->acquire((void**)&kr,sizeof(double)*nkr,align_val_avx3);
				rsrc->acquire((void**)&kt,sizeof(double)*nkt,align_val_avx3);
				rsrc->acquire((void**)&kp,sizeof(double)*nkp,align_val_avx3);
				rsrc->acquire((void**)&ijl.data,sizeof(double)*ijl.dim.n_leng,align_val_avx3);
				rsrc->acquire((void**)&plm.data,sizeof(double)*plm.dim.n_leng,align_val_avx3);
				rsrc->acquire((void**)&exm.data,sizeof(comp_t)*exm.dim.n_leng,align_val_avx3);
				rsrc->acquire((void**)&cps.data,sizeof(comp_t)*cps.dim.n_leng,align_val_avx3);
			}
		}//end of constructor
		virtual ~spectrum_c()noexcept
		{
			if(rsrc!=0)
			{
				rsrc->release(kr);
				rsrc->release(kt);
				rsrc->release(kp);
				rsrc->release(ijl.data);
				rsrc->release(plm.data);
				rsrc->release(exm.data);
				rsrc->release(cps.data);
			}
		}//end of destructor

		
		//calculate pmd spectrum
		void	measure_prj	(const coef_t& c,const size_t nthreads)noexcept//pmd stored as (ikr,ikt,ikp)
		{
			spec.resize(nkr*nkt*nkp);
			#pragma omp parallel for collapse(3) num_threads(nthreads)
			for(size_t ikr=0;ikr<nkr;++ikr)	
			for(size_t ikt=0;ikt<nkt;++ikt)	
			for(size_t ikp=0;ikp<nkp;++ikp)
			{
				comp_t	tmp	=	zero<comp_t>;
				for(size_t im=0;im<dims_t::m_dims;++im)
				for(size_t il=0;il<dims_t::l_dims;++il)
				{
					size_t l=	dims_t::in_l(im,il);

					tmp	+=	plm(ikt,im,il)*exm(ikp,im)*cps(ikr,l)
						*	intrinsic::vecd_proj_vecd<dims_t::n_dims>(ijl(ikr,l),c(im,il));
				}
				spec[(ikr*nkt+ikt)*nkp+ikp]=tmp*  0.797884560802865;//sqrt(2/pi)
			}
		}//end of measure_prj

		//calculate pad spectrum
		void	measure_pad	(const coef_t& c,const size_t nthreads)noexcept//pad stored as (ikr,ikt)
		{
			spec.resize(nkr*nkt);
			#pragma omp parallel for collapse(2) num_threads(nthreads)
			for(size_t ikr=0;ikr<nkr;++ikr)	
			for(size_t ikt=0;ikt<nkt;++ikt)//kp subspace is integrated out.
			{
				comp_t	tmp	=	zero<comp_t>;
				comp_t	tmq	=	zero<comp_t>;
				for(size_t im=0;im<dims_t::m_dims;++im)
				for(size_t il=0;il<dims_t::l_dims;++il)
				{
					size_t l=	dims_t::in_l(im,il);
					tmq	=	zero<comp_t>;
					auto cj	=	c(im,il);
					for(size_t ij=0;ij<dims_t::n_dims;++ij)
					tmq	+=	cj[ij]*ijl(ikr,l,ij);

					tmp	+=	plm(ikt,im,il)*cps(ikr,l)*tmq;//note plm is sph-nrm. Milosevic's formula is not correct for m!=0
				}
				spec[ikr*nkt+ikt]=	tmp*sqrt(kr[ikr])*2.0;//only the first nkt*nkr elements are used	
			}
		}//end of measure_pad

		//calculate partial-pmd
		void	measure_lmd	(const coef_t& c,const size_t nthreads)noexcept//ppmd stored as (ikr,im,il)
		{
			spec.resize(nkr*dims_t::m_dims*dims_t::l_dims);
			#pragma omp parallel for num_threads(nthreads)
			for(size_t ikr=0;ikr<nkr;++ikr)
			for(size_t im=0;im<dims_t::m_dims;++im)
			for(size_t il=0;il<dims_t::l_dims;++il)
			{
				size_t 	l	=	dims_t::in_l(im,il);
				auto  	cj	=	c(im,il);
				comp_t	tmq	=	zero<comp_t>;
				for(size_t ij=0;ij<dims_t::n_dims;++ij)
				{
					tmq	+=	cj[ij]*ijl(ikr,l,ij);
				}	tmq	*=	cps(ikr,l)*0.797884560802865;//sqrt(2/pi)
				spec[ikr*dims_t::l_dims*dims_t::m_dims+im*dims_t::l_dims+il] = tmq;	
			}
		}//end of measure_lmd

		template<class file_t>
		void	export_database	(file_t& file,const std::string& prefix)
		{
			file.save(kr,nkr,(prefix+"kr").c_str());
			file.save(kt,nkt,(prefix+"kt").c_str());
			file.save(kp,nkp,(prefix+"kp").c_str());

			file.save(&nkr,1ul,(prefix+"nkr").c_str());
			file.save(&nkt,1ul,(prefix+"nkt").c_str());
			file.save(&nkp,1ul,(prefix+"nkp").c_str());

			file.save(ijl.data,ijl.dim.n_leng,(prefix+"ijl").c_str());
			file.save(plm.data,plm.dim.n_leng,(prefix+"plm").c_str());
		
			file.save((double*)exm.data,exm.dim.n_leng*2ul,(prefix+"exm").c_str());
			file.save((double*)cps.data,cps.dim.n_leng*2ul,(prefix+"cps").c_str());
		}//end of export_database

		template<class file_t>
		int	import_database	(file_t& file,const std::string& prefix)
		{
			file.load(kr,nkr,(prefix+"kr").c_str());
                        file.load(kt,nkt,(prefix+"kt").c_str());
                        file.load(kp,nkp,(prefix+"kp").c_str());

			size_t check;
			file.load(&check,1ul,(prefix+"nkr").c_str());if(nkr!=check)return -1;
			file.load(&check,1ul,(prefix+"nkt").c_str());if(nkt!=check)return -2;
			file.load(&check,1ul,(prefix+"nkp").c_str());if(nkp!=check)return -3;

                        file.load(ijl.data,ijl.dim.n_leng,(prefix+"ijl").c_str());
                        file.load(plm.data,plm.dim.n_leng,(prefix+"plm").c_str());

                        file.load((double*)exm.data,exm.dim.n_leng*2ul,(prefix+"exm").c_str());
                        file.load((double*)cps.data,cps.dim.n_leng*2ul,(prefix+"cps").c_str());

			return 	0;
		}//end of import_database

};//end of spectrum_c

template<class dims_t,size_t _nkr,size_t _nkt,size_t _nkp>
struct	spectrum_c_pcs:public spectrum_c<dims_t,_nkr,_nkt,_nkp>
{
	public:
		//forward the parameter to base class
		template<class...args_t>
		explicit spectrum_c_pcs(args_t&&...args):spectrum_c<dims_t,_nkr,_nkt,_nkp>(std::forward<args_t>(args)...){}
	
		virtual	~spectrum_c_pcs()noexcept{}

		//prepare required coutinuum data from analytic hydrogenic coulomb wave function
		inline	void	initialize	
		(
			const integrator_radi<dims_t>& radi,
			double krmin,double krmax,	//|k|
			double thmin,double thmax,	//theta
			double phmin,double phmax,	//phi
			double zc,double rb,int flag		//nuclear charge, dugging boundary
		)noexcept
		{
			//axis, see spectrum/inventory.h
			set_sequence_lin(this->kp,phmin,phmax,this->nkp);
			set_sequence_lin(this->kt,thmin,thmax,this->nkt);
			if(flag==0)//->pmd
			set_sequence_lin(this->kr,krmin,krmax,this->nkr);
			else//->pad
			set_sequence_sqr(this->kr,0.5*krmin*krmin,0.5*krmax*krmax,this->nkr);//P=sqrt(2*mass*E)
			//exm
			initialize_ph<dims_t>(*this);
			//plm
			initialize_th<dims_t>(*this);
			//ijl and cps
			initialize_kr<dims_t>(*this,radi,zc,rb);
			//check
			self_check(this->kr,this->nkr,"kr");
			self_check(this->kt,this->nkt,"kt");
			self_check(this->kp,this->nkp,"kp");
			self_check(this->ijl.data,this->ijl.dim.n_leng,"ijl");
			self_check(this->plm.data,this->plm.dim.n_leng,"plm");
			self_check(this->exm.data,this->exm.dim.n_leng,"exm");
			self_check(this->cps.data,this->cps.dim.n_leng,"cps");
		}//end of initialize (hydrogenic)

		//prepare required coutinuum data from numerical integration
		template<class odef_t>
		inline  void    initialize
                (
                        const integrator_radi<dims_t>& radi,
                        double krmin,double krmax,      //|k|
                        double thmin,double thmax,      //theta
                        double phmin,double phmax,      //phi
			const odef_t& odef,		//ode function class as defined in equation.h
                        int flag              		
                )noexcept
		{
			//axis
			set_sequence_lin(this->kp,phmin,phmax,this->nkp);
			set_sequence_lin(this->kt,thmin,thmax,this->nkt);
			if(flag==0)//->pmd
			set_sequence_lin(this->kr,krmin,krmax,this->nkr);
			else//->pad
			set_sequence_sqr(this->kr,0.5*krmin*krmin,0.5*krmax*krmax,this->nkr);//P=sqrt(2*mass*E)
			//expph
			initialize_ph<dims_t>(*this);
			//plm
			initialize_th<dims_t>(*this);
			//ijl and cps
			initialize_kr<dims_t>(*this,radi,odef);
			//check
			self_check(this->kr,this->nkr,"kr");
			self_check(this->kt,this->nkt,"kt");
			self_check(this->kp,this->nkp,"kp");
			self_check(this->ijl.data,this->ijl.dim.n_leng,"ijl");
			self_check(this->plm.data,this->plm.dim.n_leng,"plm");
			self_check(this->exm.data,this->exm.dim.n_leng,"exm");
			self_check(this->cps.data,this->cps.dim.n_leng,"cps");
		}//end of initialize (centrifugal)
};//end of spectrum_c_pcs


template<class dims_t,size_t _nkr,size_t _nkt,size_t _nkp>
struct  spectrum_c_tsf
{
	public://being public for debug
		using   radi_t  =       integrator_radi<dims_t>;
		using	angu_t	=	integrator_angu<dims_t>;
		using   coef_t  =       coefficient_view<dims_t>;
		
		using 	lgd 	= 	bss::spherical_harmonics::legendre<dims_t::n_lmax>;

		using	jval_t	=	recvec<double,recidx<_nkr,dims_t::n_lmax+1ul>>;
		using	pval_t	=	recvec<double,recidx<_nkt,dims_t::m_dims,dims_t::l_dims>>;
		using 	eval_t	=	recvec<comp_t,recidx<_nkp,dims_t::m_dims>>;
		using	pmat_t	=	recvec<double,recidx<dims_t::m_dims,dims_t::l_dims>>;
		using	temp_t	=	recvec<comp_t,recidx<dims_t::m_dims,dims_t::l_dims>>;
		using	work_t	=	recvec<comp_t,recidx<_nkr,_nkt,_nkp>>;

		static constexpr std::array<comp_t,4>	power_of_minus_i	=	
		{
			-unim<comp_t>,-idty<comp_t>,unim<comp_t>,idty<comp_t>
		};//pow(-i, l+1)=pow(-i,(l+1)%4)
		
		double*	kr;	//size=nkr
		double*	kt;	//size=nkt
		double*	kp;	//size=nkp

		double*	coskt;	
		double*	sinkt;
		double*	coskp;
		double*	sinkp;

		double*	b0;	//value of Bn(r0)
		double*	br;	//value of Bn'(r0)
		size_t*	ib;	//index of Bn(r0)
		size_t	nb;	//size of b0,br,ib

		jval_t	j0;	//value of jl(kr0)	
		jval_t	jr;	//value of (d/dr)jl(kr0)
		pval_t	y0;	//value of Plm(kt), with SPH normalization
		eval_t	e0;	//value of exp(I*m*kp)

		pmat_t	pmat0;	//<ml|cos|ml'>
		pmat_t	pmat1;	//<ml|expiphsin|m'l'> outer
		pmat_t	pmat2;	//<ml|expiphsin|m'l'> inner

		temp_t	temp;	//the temp value when tsurff-ing (for b0)
		temp_t	temq;	//the temp value when tsurff-ing (for br)
		double	time;	//record time
		double	disx;	//displacement of x = / Ax(t) dt
		double	disy;	//displacement of y = / Ay(t) dt
		double	disz;	//displacement of z = / Az(t) dt
		work_t	work;	//the workspace of size nkr*nkt*nkp
		rsrc_t*	rsrc;	

	public:
		//-------------------------------------------------------
		inline	double*	dptr()const noexcept{return (double*)work.data;}
		inline	size_t 	size()const noexcept{return work.dim.n_leng;}

		inline	double*	axis_kr()const noexcept{return kr;}
		inline	double*	axis_kt()const noexcept{return kt;}
		inline	double*	axis_kp()const noexcept{return kp;}
		//-------------------------------------------------------
		inline	void	clearwork	()noexcept//reset workspace
		{
			time=0.0;
			disx=0.0;
			disy=0.0;
			disz=0.0;
			for(size_t i=0;i<temp.dim.n_leng;++i)temp[i]=0.0;
			for(size_t i=0;i<temq.dim.n_leng;++i)temq[i]=0.0;
			for(size_t i=0;i<work.dim.n_leng;++i)work[i]=0.0;
		}
		inline	comp_t	sumup_b0	(const comp_t* cnlm)const noexcept//just a helper function
		{
			comp_t  sum     =       zero<comp_t>;
			for(size_t i=0;i<nb;++i)sum+=cnlm[ib[i]]*b0[i];	
			return	sum;		
		}
		inline  comp_t  sumup_br        (const comp_t* cnlm)const noexcept//just a helper function
        	{
                	comp_t  sum     =       zero<comp_t>;
	                for(size_t i=0;i<nb;++i)sum+=cnlm[ib[i]]*br[i];
        	        return  sum;
        	}
		inline	void	prepare		(const comp_t* coef)noexcept//can be called in omp block, but usually unnecessary
		{
			#pragma omp for
			for(size_t im=0;im<dims_t::m_dims;++im)	
			for(size_t il=0;il<dims_t::l_dims;++il)
			{
				comp_t const*	_coef;
					_coef	=	coef+dims_t::l_dims*dims_t::n_dims*im+dims_t::n_dims*il;
				comp_t 	b0val	=	sumup_b0(_coef);
				comp_t 	brval	=	sumup_br(_coef);
				temp(im,il)	=	b0val;
				temq(im,il)	=	brval;
			}
		}
		inline	void	prepare		(const double ax,const double ay,const double az,const double dt)noexcept//set volkov phase
		{
			time	+=	dt;
			disx	+=	ax*dt;	//if ax, ay, az given by midpoint value, this integral has the highest accuracy
			disy	+=	ay*dt;
			disz	+=	az*dt;
		}
		//-------------------------------------------------------
		inline	void	accumulate
		(
			const double dt,
			const comp_t* coef
		)noexcept
		{
			prepare(coef);
			prepare(0.0,0.0,0.0,dt);
			#pragma omp parallel for simd
			for(size_t ikr=0;ikr<_nkr;++ikr)
			{
				auto 	_j0	=	j0(ikr);
				auto 	_jr	=	jr(ikr);
				for(size_t ikt=0;ikt<_nkt;++ikt){
				auto 	_y0	=	y0(ikt);
				for(size_t ikp=0;ikp<_nkp;++ikp){	
				auto 	_e0	=	e0(ikp);
				auto 	sum	=	zero<comp_t>;
					for(size_t im=0;im<dims_t::m_dims;++im)	
					for(size_t il=0;il<dims_t::l_dims;++il)
					{
						auto l		=	dims_t::in_l(im,il);
						comp_t 	_y0val	=	_y0(im,il)*_e0(im)*power_of_minus_i[l%4];
						auto 	_j0val	=	_j0(l);
						auto	_jrval	=	_jr(l);
						comp_t 	_b0_this=	temp(im,il);
						comp_t	_br_this=	temq(im,il);
						comp_t	jdb	=	-0.5*(_j0val*_br_this-_jrval*_b0_this);
						sum		+=	_y0val*jdb;
					}
					double	phase		=	0.5*kr[ikr]*kr[ikr]*time
								+	kr[ikr]*sinkt[ikt]*coskp[ikp]*disx
								+	kr[ikr]*sinkt[ikt]*sinkp[ikp]*disy
								+	kr[ikr]*coskt[ikt]*disz;
					work(ikr,ikt,ikp)	+=	sum*exp(unim<comp_t>*phase)*dt*0.797884560802865;
				}}
			}
		}//end of accumulate (ff)

		inline	void	accumulate
		(
			const double az0,
			const double az1,
			const double dt,
			const comp_t* coef
		)noexcept
		{
			comp_t	iaz	=	az1*unim<comp_t>;
			prepare(coef);
			prepare(0.0,0.0,0.5*(az0+az1),dt);
			#pragma omp parallel for simd
			for(size_t ikr=0;ikr<_nkr;++ikr)
			{
				auto 	_j0	=	j0(ikr);
				auto 	_jr	=	jr(ikr);
				for(size_t ikt=0;ikt<_nkt;++ikt){
				auto 	_y0	=	y0(ikt);
				for(size_t ikp=0;ikp<_nkp;++ikp){	
				auto 	_e0	=	e0(ikp);
				auto 	sum	=	zero<comp_t>;
					for(size_t im=0;im<dims_t::m_dims;++im)	
					for(size_t il=0;il<dims_t::l_dims;++il)
					{
						auto l		=	dims_t::in_l(im,il);
						comp_t 	_y0val	=	_y0(im,il)*_e0(im)*power_of_minus_i[l%4];
						auto 	_j0val	=	_j0(l);
						auto	_jrval	=	_jr(l);
						comp_t	_b0_prev=	il>0?temp (im,il-1ul):zero<comp_t>;
						double	_pm_prev=	il>0?pmat0(im,il-1ul):zero<double>;
						comp_t 	_b0_this=	temp(im,il);
						comp_t	_br_this=	temq(im,il);
						comp_t	_b0_next=	il+1ul<dims_t::l_dims?temp (im,il+1ul):zero<comp_t>;
						double	_pm_next=	il+1ul<dims_t::l_dims?pmat0(im,il    ):zero<double>;
						comp_t	jdb	=	-0.5*(_j0val*_br_this-_jrval*_b0_this);
						comp_t	jbp	=	_j0val*_b0_prev;
						comp_t	jbn	=	_j0val*_b0_next;
						sum		+=	_y0val*(jdb-iaz*jbp*_pm_prev-iaz*jbn*_pm_next);
					}
					double	phase		=	0.5*kr[ikr]*kr[ikr]*time
								+	kr[ikr]*coskt[ikt]*disz;
					work(ikr,ikt,ikp)	+=	sum*exp(unim<comp_t>*phase)*dt*0.797884560802865;
				}}
			}
		}//end of accumulate (z)

		inline	void	accumulate
		(
			const double ax0,
			const double ax1,
			const double ay0,
			const double ay1,
			const double dt,
			const comp_t* coef
		)noexcept
		{
			comp_t	iap	=	(ax1+ay1*unim<comp_t>)*unim<comp_t>*0.5;
			comp_t	ias	=	(ax1-ay1*unim<comp_t>)*unim<comp_t>*0.5;
			prepare(coef);
			prepare(0.5*(ax0+ax1),0.5*(ay0+ay1),0.0,dt);
			#pragma omp parallel for simd
			for(size_t ikr=0;ikr<_nkr;++ikr)
			{
				auto 	_j0	=	j0(ikr);
				auto 	_jr	=	jr(ikr);
				for(size_t ikt=0;ikt<_nkt;++ikt){
				auto 	_y0	=	y0(ikt);
				for(size_t ikp=0;ikp<_nkp;++ikp){	
				auto 	_e0	=	e0(ikp);
				auto 	sum	=	zero<comp_t>;
					for(size_t im=0;im<dims_t::m_dims;++im)	
					for(size_t il=0;il<dims_t::l_dims;++il)
					{
						auto l		=	dims_t::in_l(im,il);
						comp_t 	_y0val	=	_y0(im,il)*_e0(im)*power_of_minus_i[l%4];
						auto 	_j0val	=	_j0(l);
						auto	_jrval	=	_jr(l);
						comp_t 	_b0_00	=	temp(im,il);
						comp_t	_br_00	=	temq(im,il);
						comp_t	_tmpsum	=	-0.5*(_j0val*_br_00-_jrval*_b0_00);	//jdb
						
						size_t	jm,jl;
						if(dims_t::template in_check<-1ul,-1ul>(im,il,jm,jl))
						{
							_tmpsum	-=	ias*_j0val*temp(jm,jl)*pmat1(jm,jl); //CHECKME ias?
						}
						if(dims_t::template in_check<-1ul,+1ul>(im,il,jm,jl))
						{
							_tmpsum	-=	ias*_j0val*temp(jm,jl)*pmat2(jm,jl); //CHECKME ias?
						}
						if(dims_t::template in_check<+1ul,-1ul>(im,il,jm,jl))
						{
							_tmpsum	-=	iap*_j0val*temp(jm,jl)*pmat2(im,il); //CHECKME iap?
						}
						if(dims_t::template in_check<+1ul,+1ul>(im,il,jm,jl))
						{
							_tmpsum	-=	iap*_j0val*temp(jm,jl)*pmat1(im,il); //CHECKME iap?	
						}
						sum		+=	_y0val*_tmpsum;
					}
					double	phase		=	0.5*kr[ikr]*kr[ikr]*time
								+	kr[ikr]*sinkt[ikt]*coskp[ikp]*disx
								+	kr[ikr]*sinkt[ikt]*sinkp[ikp]*disy;
					work(ikr,ikt,ikp)	+=	sum*exp(unim<comp_t>*phase)*dt*0.797884560802865;
				}}
			}
		}//end of accumulate (xy)
		//-------------------------------------------------------
		inline	void	initialize	
		(
			const radi_t& radi,
			const angu_t& angu,
			double r0,			//the tsurff boundary
			double zc,			//the coulomb correction. parse 0 to turn-off this correction
                        double krmin,double krmax,      //|k|
                        double ktmin,double ktmax,      //theta
                        double kpmin,double kpmax,       //phi
			int flag
		)noexcept
		{
			//set zeros
			clearwork();
			//prepare axis
			set_sequence_lin(this->kp,kpmin,kpmax,_nkp);
			set_sequence_lin(this->kt,ktmin,ktmax,_nkt);
			if(flag==0)
			set_sequence_lin(this->kr,krmin,krmax,_nkr);
			else
			set_sequence_sqr(this->kr,0.5*krmin*krmin,0.5*krmax*krmax,_nkr);//P=sqrt(2*mass*E)
			for(size_t ikt=0;ikt<_nkt;++ikt)
			{
				coskt[ikt]	=	cos(kt[ikt]);
				sinkt[ikt]	=	sin(kt[ikt]);
			}
			for(size_t ikp=0;ikp<_nkp;++ikp)
			{
				coskp[ikp]	=	cos(kp[ikp]);
				sinkp[ikp]	=	sin(kp[ikp]);
			}
			//calculate Bn(r0)
			nb=0;
			for(size_t i=0;i<dims_t::n_dims;++i)//to count number of nonzero Bn(r0), usually not large
			{
				if(radi.base.cal(i,r0))++nb;
			}
			rsrc->acquire((void**)&b0,sizeof(double)*nb,align_val_avx3);
			rsrc->acquire((void**)&br,sizeof(double)*nb,align_val_avx3);
			rsrc->acquire((void**)&ib,sizeof(size_t)*nb,align_val_avx3);
			size_t nb_temp=0;
			for(size_t i=0;i<dims_t::n_dims;++i)//to cache values of Bn, Bn', n
                	{
                        	double  b       =radi.base.template cal<dims_t::mr,0>(i,r0);
	                        double  d;
        	                if(b>0.)//Bspline are either 0 or >0
                	        {
                        	        d       =radi.base.template cal<dims_t::mr,1>(i,r0);
                                	b0[nb_temp]=b;
	                                br[nb_temp]=d;
        	                        ib[nb_temp]=i;
                	                ++nb_temp;
                       	 	}
                	}
			//calculate jl(kr0). see function/coulomb.h
			for(size_t ikr=0;ikr<_nkr;++ikr)
			{	
				double	k	=	kr[ikr];
				double 	x	=	k*r0;
				auto	j0k	=	j0(ikr);
				auto	jrk	=	jr(ikr);
				coulombfg<double>(0.0,-zc/k,x,j0k.data,jrk.data,dims_t::n_lmax);//jl(x)=Fl(x,eta=0)/x
				#pragma GCC ivdep
				for(size_t il=0;il<=dims_t::n_lmax;++il)
				{
					j0k(il)/=k;	//we store (r*jl) = Fl/k
							//and (d/dr)(r*jl)'= Fl'
				}
			}
			//calculate Ylm(kt)
			auto	tmp	=	std::vector<double>(lgd::max());
			for(size_t ikt=0;ikt<_nkt;++ikt)
			{	
				lgd::cal(cos(kt[ikt]),tmp.data());
				for(size_t im=0;im<dims_t::m_dims;++im)
				for(size_t il=0;il<dims_t::l_dims;++il)
				{
					long m,l;
					dims_t::in(im,il,m,l);//cal (m,l)
					y0(ikt,im,il)	=	lgd::get(tmp.data(),l,m);
				}
			}
			//calculate exp
			for(size_t ikp=0;ikp<_nkp;++ikp)
			{
				for(size_t im=0;im<dims_t::m_dims;++im)
				{	
					double arg	=	dims_t::in_m(im)*kp[ikp];
					e0(ikp,im)	=	comp_t{gsl_sf_cos(arg),gsl_sf_sin(arg)};
				}
			}
			//calculate matrix element
			angu.integrate_cos(pmat0);
			angu.integrate_sinexp1(pmat1);
			angu.integrate_sinexp2(pmat2);
		}//end of initialize	

		explicit spectrum_c_tsf(rsrc_t* _rsrc=global_default_arena_ptr):b0(0),br(0),ib(0),rsrc(_rsrc)
		{
			if(rsrc)
			{
				rsrc->acquire((void**)&kr,sizeof(double)*_nkr,align_val_avx3);
				rsrc->acquire((void**)&kt,sizeof(double)*_nkt,align_val_avx3);
				rsrc->acquire((void**)&kp,sizeof(double)*_nkp,align_val_avx3);
				rsrc->acquire((void**)&coskt,sizeof(double)*_nkt,align_val_avx3);
				rsrc->acquire((void**)&sinkt,sizeof(double)*_nkt,align_val_avx3);
				rsrc->acquire((void**)&coskp,sizeof(double)*_nkp,align_val_avx3);
				rsrc->acquire((void**)&sinkp,sizeof(double)*_nkp,align_val_avx3);
				rsrc->acquire((void**)&j0.data,sizeof(double)*j0.dim.n_leng,align_val_avx3);
				rsrc->acquire((void**)&jr.data,sizeof(double)*jr.dim.n_leng,align_val_avx3);
				rsrc->acquire((void**)&y0.data,sizeof(double)*y0.dim.n_leng,align_val_avx3);
				rsrc->acquire((void**)&e0.data,sizeof(comp_t)*e0.dim.n_leng,align_val_avx3);
				rsrc->acquire((void**)&pmat0.data,sizeof(double)*pmat0.dim.n_leng,align_val_avx3);
				rsrc->acquire((void**)&pmat1.data,sizeof(double)*pmat1.dim.n_leng,align_val_avx3);
				rsrc->acquire((void**)&pmat2.data,sizeof(double)*pmat2.dim.n_leng,align_val_avx3);
				rsrc->acquire((void**)&temp.data,sizeof(comp_t)*temp.dim.n_leng,align_val_avx3);
				rsrc->acquire((void**)&temq.data,sizeof(comp_t)*temq.dim.n_leng,align_val_avx3);
				rsrc->acquire((void**)&work.data,sizeof(comp_t)*work.dim.n_leng,align_val_avx3);
			}
		}//end of constructor
		virtual ~spectrum_c_tsf()noexcept
		{
			if(rsrc)
			{
				rsrc->release(kr);
				rsrc->release(kt);
				rsrc->release(kp);
				rsrc->release(coskt);
				rsrc->release(sinkt);
				rsrc->release(coskp);
				rsrc->release(sinkp);
				rsrc->release(j0.data);
				rsrc->release(jr.data);
				rsrc->release(y0.data);
				rsrc->release(e0.data);
				rsrc->release(pmat0.data);
				rsrc->release(pmat1.data);
				rsrc->release(pmat2.data);
				rsrc->release(temp.data);
				rsrc->release(temq.data);
				rsrc->release(work.data);
				if(b0)rsrc->release(b0);
				if(br)rsrc->release(br);
				if(ib)rsrc->release(ib);
			}
		}//end of destructor
};//end of spectrum_c_tsf

//------------------------------------------------------------------------------------------------------
template<class dims_t,size_t _nkr,size_t _nkt,size_t _nkp>
struct	spectrum_b_pcs
{
	public:
		static constexpr size_t nkr=_nkr;
		static constexpr size_t nkt=_nkt;
		static constexpr size_t nkp=_nkp;

		double*	kr;	//kr[ikr] store xi
		double*	kt;	//kt[ikt] store theta=arccos(eta)
		double*	kp;	//kp[ikp] store phi

		recvec<double,recidx<    nkt,dims_t::m_dims,dims_t::l_dims>>		plm;	//with SPH normalization Nlm
		recvec<double,recidx<nkr,nkt,dims_t::m_dims,dims_t::l_dims>>		sqm;	//with SPH normalization Nlm
		recvec<double,recidx<nkr,    dims_t::m_dims,dims_t::l_dims>>		aqm;	//cache the angular eigenvalue
		recvec<comp_t,recidx<nkp,dims_t::m_dims>>				exm;	//with SPH normalization 
		recvec<comp_t,recidx<nkr,dims_t::m_dims,dims_t::l_dims>>		cqm;				

		recvec<double,recidx<nkr,dims_t::m_dims,dims_t::l_dims,dims_t::l_dims>>	j0mql;
		recvec<double,recidx<nkr,dims_t::m_dims,dims_t::l_dims,dims_t::l_dims>>	j2mql;
	
		recvec<double,recidx<nkr,dims_t::m_dims,dims_t::l_dims,dims_t::n_dims>>	i0mqn;
		recvec<double,recidx<nkr,dims_t::m_dims,dims_t::l_dims,dims_t::n_dims>>	i2mqn;

		comp_t*	spec;
		rsrc_t*	rsrc;

		using	coef_t	=	coefficient_view<dims_t>;

		inline	double*	dptr()const noexcept{return (double*)spec;}	
		inline	size_t	size()const noexcept{return nkt*nkp*nkr*2ul;}	

		explicit spectrum_b_pcs(rsrc_t* _rsrc=global_default_arena_ptr):rsrc(_rsrc)
		{
			if(rsrc)
			{
				rsrc->acquire((void**)&kr,sizeof(double)*nkr,align_val_avx3);
				rsrc->acquire((void**)&kt,sizeof(double)*nkt,align_val_avx3);
				rsrc->acquire((void**)&kp,sizeof(double)*nkp,align_val_avx3);
				rsrc->acquire((void**)&plm.data,sizeof(double)*plm.dim.n_leng,align_val_avx3);
				rsrc->acquire((void**)&sqm.data,sizeof(double)*sqm.dim.n_leng,align_val_avx3);
				rsrc->acquire((void**)&aqm.data,sizeof(double)*aqm.dim.n_leng,align_val_avx3);
				rsrc->acquire((void**)&exm.data,sizeof(comp_t)*exm.dim.n_leng,align_val_avx3);
				rsrc->acquire((void**)&cqm.data,sizeof(comp_t)*cqm.dim.n_leng,align_val_avx3);
				rsrc->acquire((void**)&j0mql.data,sizeof(double)*j0mql.dim.n_leng,align_val_avx3);
				rsrc->acquire((void**)&j2mql.data,sizeof(double)*j2mql.dim.n_leng,align_val_avx3);
				rsrc->acquire((void**)&i0mqn.data,sizeof(double)*i0mqn.dim.n_leng,align_val_avx3);
				rsrc->acquire((void**)&i2mqn.data,sizeof(double)*i2mqn.dim.n_leng,align_val_avx3);
				rsrc->acquire((void**)&spec,sizeof(comp_t)*nkr*nkt*nkp,align_val_avx3);
			}	
		}//end of constructor
		virtual	~spectrum_b_pcs()noexcept
		{
			if(rsrc)
			{
				rsrc->release(kr);
				rsrc->release(kt);
				rsrc->release(kp);
				rsrc->release(plm.data);
				rsrc->release(sqm.data);
				rsrc->release(aqm.data);
				rsrc->release(exm.data);
				rsrc->release(cqm.data);
				rsrc->release(j0mql.data);
				rsrc->release(j2mql.data);
				rsrc->release(i0mqn.data);
				rsrc->release(i2mqn.data);
				rsrc->release(spec);
			}
		}//end of destructor

		//calculate pmd spectrum
		void	measure_prj	(const coef_t& c)noexcept//pmd stored as (ikr,ikt,ikp)
		{
			#pragma omp parallel for collapse(3)
			for(size_t ikr=0;ikr<nkr;++ikr)
			for(size_t ikt=0;ikt<nkt;++ikt)
			for(size_t ikp=0;ikp<nkp;++ikp)
			{
				auto	allsum	=	zero<comp_t>;
				auto	stemp	=	sqm.sub(ikr,ikt);
				auto	dtemp	=	cqm.sub(ikr);
				auto	etemp	=	exm.sub(ikp);
				for(size_t im=0;im<dims_t::m_dims;++im)
				for(size_t iq=0;iq<dims_t::l_dims;++iq)
				{
					auto	i0temp	=	i0mqn.sub(ikr,im,iq);
					auto	i2temp	=	i2mqn.sub(ikr,im,iq);
					auto	j0temp	=	j0mql.sub(ikr,im,iq);
					auto	j2temp	=	j2mql.sub(ikr,im,iq);	
					comp_t	jqsum	=	zero<comp_t>;

					for(size_t il=0;il<dims_t::l_dims;++il)
					{
						comp_t const* ctemp=	c(im,il);
						jqsum	+=	intrinsic::vecd_proj_vecd<dims_t::n_dims>(i2temp.data,ctemp)*j0temp(il)
							-	intrinsic::vecd_proj_vecd<dims_t::n_dims>(i0temp.data,ctemp)*j2temp(il);
					}
						allsum	+=	jqsum*stemp(im,iq)*dtemp(im,iq)*etemp(im);
				}
				spec[nkp*nkt*ikr+nkp*ikt+ikp]	=	allsum*0.797884560802865;//sqrt(2/pi);
			}
		}//end of measure_prj

		//prepare required coutinuum data from bicenter coulomb wave function
		inline	void	initialize	
		(
			const integrator_radi<dims_t>& radi,
			const integrator_angu<dims_t>& angu,
			double krmin,double krmax,	//|k|
			double thmin,double thmax,	//theta
			double phmin,double phmax,	//phi
			double z1,double z2,double rn,	//nuclear charge, distance
			int flag	
		)noexcept
		{
			//axis
			set_sequence_lin(this->kp,phmin,phmax,this->nkp);
			set_sequence_lin(this->kt,thmin,thmax,this->nkt);

			if(flag==0)//->pmd
			set_sequence_lin(this->kr,krmin,krmax,this->nkr);
			else//->pad
			set_sequence_sqr(this->kr,0.5*krmin*krmin,0.5*krmax*krmax,this->nkr);//P=sqrt(2*mass*E)
			//exm
			initialize_ph<dims_t>(*this);
			//plm,sqm,j0,j2
			initialize_th<dims_t>(*this);
			initialize_et<dims_t>(*this,angu,z1,z2,rn);
			//i0,i2,cps
			initialize_kx<dims_t>(*this,radi,z1,z2,rn);
			//check
			self_check(this->kr,this->nkr,"kr");
			self_check(this->kt,this->nkt,"kt");
			self_check(this->kp,this->nkp,"kp");
			self_check(this->plm.data,this->plm.dim.n_leng,"plm");
			self_check(this->sqm.data,this->sqm.dim.n_leng,"sqm");
			self_check(this->exm.data,this->exm.dim.n_leng,"exm");
			self_check(this->cqm.data,this->cqm.dim.n_leng,"cqm");
			self_check(this->j0mql.data,this->j0mql.dim.n_leng,"j0mql");
			self_check(this->j2mql.data,this->j2mql.dim.n_leng,"j2mql");
			self_check(this->i0mqn.data,this->i0mqn.dim.n_leng,"i0mqn");
			self_check(this->i2mqn.data,this->i2mqn.dim.n_leng,"i2mqn");//end*/
		}//end of initialize
};//end of spectrum_b_pcs

