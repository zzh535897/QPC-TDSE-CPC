#pragma once

//require <libraries/support_avx.h>
//require <libraries/support_std.h>
//require <libraries/support_omp.h>
//require <utilities/error.h>
//require <utilities/recid.h>
//require <resources/arena.h>
//require <algorithm/ed.h>

template<class dims_t>
class	coefficient_data:public coefficient_view<dims_t>
{
	protected:
		rsrc_t*	rsrc;

	public:
		//expose as 1d double array (for save data)
		inline  double const*   dptr()const noexcept 	{return (double const*)(void*)this->data;}
		inline  double*         dptr()noexcept          {return (double      *)(void*)this->data;}
		//obtain the number of elements in 1d array (for save data)
		static constexpr size_t size()noexcept{return dims_t::n_leng;}

		//constructor
		coefficient_data(coefficient_data const&)=delete;
		coefficient_data& operator=(coefficient_data const&)=delete;
		coefficient_data& operator=(coefficient_data&&)=delete;

		explicit coefficient_data(rsrc_t* _rsrc=global_default_arena_ptr):rsrc(_rsrc)
		{
			if(rsrc)
			{
				rsrc->acquire((void**)&(this->data),sizeof(comp_t)*dims_t::n_leng,align_val_avx3);
			}else
			{
				throw errn_t("initializing coefficient_data with null rsrc.\n");
			}
		}
		coefficient_data(coefficient_data&& rhs)
		{
			this->rsrc=rhs.rsrc;
			this->data=rhs.data;
			rhs.rsrc=0;
		}	
		//destructor
		virtual ~coefficient_data()noexcept
		{
			if(rsrc)
			{
				rsrc->release((void*)this->data);
			}
		}
};//end of coefficient_data

template<class dims_t>
class	coefficient:public coefficient_data<dims_t>
{
	public:
		coefficient(coefficient const&)=delete;
		coefficient(coefficient&&)=default;
		coefficient& operator=(coefficient const&)=delete;
		coefficient& operator=(coefficient&&)=delete;
	
		explicit coefficient(rsrc_t* _rsrc=global_default_arena_ptr):coefficient_data<dims_t>(_rsrc){}

		//fill all elements by zero
		inline	void	fillzero()noexcept
		{
			for(size_t i=0;i<dims_t::n_leng;++i)
			{
				(*this)[i]=zero<comp_t>;//fill wave with 0
			}
		}

		inline	double	normalize(const operator_sc<dims_t>& s)noexcept
		{
			double	_nrm	= 	0.0;
			s.observe(this->data,&_nrm);
			_nrm	=	1./sqrt(_nrm);
			#pragma omp parallel for
			for(size_t i=0;i<dims_t::n_leng;++i)
			{
				this->data[i][0]*=_nrm;
				this->data[i][1]*=_nrm;
			}
			return	_nrm;
		}

		//initialize by solving ESc=Hc in the (im,il)-th r-subblock, for centrifugal H
		template<size_t n_state>
		inline	auto    initialize	
		(
			const operator_sc<dims_t>& _s,	//the operator S
			const operator_hc<dims_t>& _h,	//the operator H
			std::array<size_t,n_state> im,	//index of m-subspace
			std::array<size_t,n_state> il,	//index of l-subspace
			std::array<size_t,n_state> ir,	//index of r-subspace
			std::array<comp_t,n_state> cf	//superposition coefficients
		)
		{
			//auto	clc	=	timer();clc.tic();
			auto	h	=	std::vector<double>(dims_t::n_rsub);	//temp workspace for H in il-th r-subspace.
			auto	s	=	std::vector<double>(dims_t::n_rsub);	//temp workspace for S
			auto	e	=	std::vector<double>(dims_t::n_dims);	//temp workspace for holding eigen value
			auto	z	=	std::vector<double>(dims_t::n_dims*dims_t::n_dims);//temp workspace for eigen vector 
			auto	r	=	std::array<double,n_state>();		//return value, filled with energies if wanted

			this->fillzero();
			//!!!if lapacke is not correctly compiled with multi-threaded version, omp should be switched off to avoid random errors!!!

			for(size_t j=0;j<n_state;++j)//fool prevention
			{
				if(im[j]>=dims_t::m_dims)throw 	errn_t("invalid argument im: "+to_string(im[j])+"\n");
				if(il[j]>=dims_t::l_dims)throw 	errn_t("invalid argument il: "+to_string(il[j])+"\n");
				if(ir[j]>=dims_t::n_dims)throw	errn_t("invalid argument ir: "+to_string(ir[j])+"\n");
			}
			for(size_t j=0;j<n_state;++j)//loop over cf array
			{
				auto	h_view	=	_h.sub(im[j],il[j]);
				auto	s_view	=	_s.sub();
				#pragma GCC ivdep
				for(size_t i=0;i<dims_t::n_rsub;++i)//copy h,s to workspace	
				{
					h[i]	=	h_view[i];
					s[i]	=	s_view[i];
				}
				//call lapacke to solve gv problem, see ed.h
				int err		=	impl_ed_simd::gv_dsb<dims_t::n_dims,dims_t::n_diag>(h.data(),s.data(),e.data(),z.data());
				if(err)
				{
					throw	errn_t("lapacke fail in gv_dsb. errcode:"+to_string(err)+"\n");
				}
				auto	_psi	=	(*this)(im[j],il[j]);
				auto	_eig	=	z.data()+ir[j]*dims_t::n_dims;
					r[j]	=	e[ir[j]];//record the j-th energy as return value
				for(size_t i=0;i<dims_t::n_dims;++i)
				{
					_psi[i]+=	_eig[i]*cf[j];//superposition
				}
			}
			//clc.toc();printf("time=%lf\n",clc.get());
			return 	r;
		}//end of initialize(centrifugal)

		//initialize by solving ESc=Hc in the full space, for multipolar H
		template<size_t n_state,int n_state_expect=16,size_t n_pole=1>
		inline	auto    initialize	
		(
			const operator_sc<dims_t>& 	  _s,	//the operator S
			const operator_hm<dims_t,n_pole>& _h,	//the operator H
			std::array<size_t,n_state> id,	//index of eigen-value
			std::array<double,n_state> e1,	//minimun accepted energy
			std::array<double,n_state> e2,	//maximum accepted energy
			std::array<comp_t,n_state> cf	//superposition coefficients
		)
		{
			int info=0;
			this->fillzero();

			auto	eig	=	csreig<comp_t>();//see structure/csrmat.h
			auto	eng	=	std::array<double,n_state>();

			for(size_t i=0;i<n_state;++i)
			{
				auto	_hmat	=	_h.create_csrmat_comp();//generate a csrmat object for H
				auto	_smat	=	_s.create_csrmat_comp();//generate a csrmat object for S
				info	=	gv_zh_csr
				(
					_hmat,//_h.create_csrmat_comp(),
					_smat(),//_s.create_csrmat_comp(),
					eig,n_state_expect,e1[i],e2[i]
				);//call csr FEAST to do diagonalization*/
				if(info)
				{
					throw errn_t{"feast error in coefficient initialization. errorcode="+to_string(info)};
				}
				comp_t*	vec	=	eig.vec.data()+id[i]*dims_t::n_leng;
				for(size_t j=0;j<dims_t::n_leng;++j)
				{
					this->data[j]+=cf[i]*vec[j];
				}
				eng[i]=eig.val[id[i]];
			}
			return eng;
		}//end of initialize(multipolar)

		//initialize by solving ESc=Hc in the (im)-th r-subblock, for prolate spheroidal H
		template<size_t n_state,int n_state_expect=16,size_t hb_flag=1>
		inline	auto    initialize	
		(
			const operator_sb<dims_t>& 		_s,	//the operator S
			const operator_hb<dims_t,hb_flag>& 	_h,	//the operator H
			std::array<size_t,n_state> im,	//index of m-subspace
			std::array<size_t,n_state> ir,	//index of rl-subspace (inside range)
			std::array<double,n_state> e1,	//minimun accepted energy
			std::array<double,n_state> e2,	//maximum accepted energy
			std::array<comp_t,n_state> cf	//superposition coefficients
		)
		{
			int info=0;
			this->fillzero();

			auto	eig	=	csreig<double>();//see structure/csrmat.h
			auto	eng	=	std::array<double,n_state>();
			for(size_t i=0;i<n_state;++i)
			{
				info	=	gv_ds_csr
				(
					_h.create_csrmat_real(im[i]),//generate a csrmat object for H. auto-choose flag
					_s.create_csrmat_real(im[i]),//generate a csrmat object for S.
					eig,n_state_expect,e1[i],e2[i]
				);//call csr FEAST to do diagonalization
				if(info)
				{
					throw errn_t{"feast error in coefficient initialization:"+to_string(info)};
				}
				if(ir[i]>=size_t(eig.fnd))
				{
					throw errn_t{"ir exceeds number of eigenstates found."};
				}
					eng[i]	=	eig.val[ir[i]];
				auto 	_psi	=	(*this)(im[i],0);
				size_t	_sh	=	ir[i]*dims_t::n_dims*dims_t::l_dims;
				for(size_t j=0;j<dims_t::l_dims*dims_t::n_dims;++j)
				{
					_psi[j]	+=	eig.vec[j+_sh]*cf[i];	
				}
			}
			return	eng;
		}//end of initialize(prolate spheroidal)
		
};//end of coefficient

