#pragma once

//require <utilities/recid.h>
//

template<class dims_t>
struct	eigenstate_sph final//to store all required eigenstates and eigenvalues (the left handside orthogonal basis)
{
	recvec<std::vector<double>,recidx<dims_t::n_lmax+1ul>> vec;
	recvec<std::vector<double>,recidx<dims_t::n_lmax+1ul>> val;

	//memory affairs
	eigenstate_sph(eigenstate_sph const&)=delete;
	eigenstate_sph(eigenstate_sph&&)=delete;
	eigenstate_sph& operator=(eigenstate_sph const&)=delete;
	eigenstate_sph& operator=(eigenstate_sph&&)=delete;

	eigenstate_sph()
	{
		vec.data=new (std::nothrow) std::vector<double>[vec.dim.n_leng];
		val.data=new (std::nothrow) std::vector<double>[val.dim.n_leng];
	}//end of constructor
	~eigenstate_sph()noexcept
	{
		delete[] vec.data;
		delete[] val.data;	
	}//end of destructor

	inline	void	projection	(const coefficient_view<dims_t>& coef,std::vector<comp_t>& proj,std::vector<size_t>& size)
	{
		size.resize(dims_t::m_dims*dims_t::l_dims);
		proj.resize(dims_t::m_dims*dims_t::l_dims*dims_t::n_dims);
		auto	_size	=	recvec<size_t,recidx<dims_t::m_dims,dims_t::l_dims               >>{size.data()};
		auto	_proj	=	recvec<comp_t,recidx<dims_t::m_dims,dims_t::l_dims,dims_t::n_dims>>{proj.data()};
		#pragma omp parallel for simd collapse(2) schedule(dynamic)
		for(size_t im=0;im<dims_t::m_dims;++im)
		for(size_t il=0;il<dims_t::l_dims;++il)
		{
			size_t	l	=	dims_t::in_l(im,il);
			auto	__coef	=	 coef(im,il);
			auto	__proj	=	_proj(im,il);
			auto&	_basis	=	vec(l);
			auto&	_value	=	val(l);
			_size(im,il)	=	_value.size();//the number of states
			for(size_t in=0;in<_value.size();++in)	
			{	
				comp_t&	_temp	=	(__proj(in)=0.0);
				double*	_base	=	_basis.data()+in*dims_t::n_dims;
				for(size_t ir=0;ir<dims_t::n_dims;++ir)
				{
					_temp	+=	_base[ir]*__coef[ir];
				}	
			}
		}
	}//end of projection

	template<class crit_t>
	inline	void	initialize	(const operator_sc<dims_t>& sc,const operator_hc<dims_t>& hc,const crit_t& crit)
	{
		#pragma omp parallel
		{
			//temp for each thread
			auto	temp_mat_s	=	std::vector<double>(dims_t::n_rsub);
			auto	temp_mat_h	=	std::vector<double>(dims_t::n_rsub);
			auto	temp_vec	=	std::vector<double>(dims_t::n_dims*dims_t::n_dims);
			auto	temp_val	=	std::vector<double>(dims_t::n_dims);
			auto	crit_result	=	std::vector<size_t>(dims_t::n_dims);

			#pragma omp for schedule(dynamic)
			for(size_t l=0;l<=dims_t::n_lmax;++l)
			{
				//copy h,s to workspace
				auto  h_view	=	hc[l];
				auto  s_view	=	sc();
				#pragma GCC ivdep
				for(size_t i=0;i<dims_t::n_rsub;++i)
                              	{
                                        temp_mat_s[i]    =       s_view[i];
                                 	temp_mat_h[i]    =       h_view[i];
                        	}
				//call lapacke to solve gv problem, see ed.h
				int 	err 	=    	impl_ed_simd::gv_dsb<dims_t::n_dims,dims_t::n_diag>
				(
					temp_mat_h.data(),
					temp_mat_s.data(),
					temp_val.data(),
					temp_vec.data()
				);
                              	if(err)	throw errn_t("lapacke fail in gv_dsb. errcode:"+to_string(err)+"\n");
				//find all required states
				crit_result.clear();
				for(size_t i=0;i<dims_t::n_dims;++i)
				{	
					if(crit(l,temp_val[i],temp_vec.data()+i*dims_t::n_dims))
					crit_result.push_back(i);//requirement: crit(l,eig,vec)
				}
				//dump all required states
				auto& vec_this	=	vec(l);
				auto& val_this	=	val(l);
				vec_this.resize(crit_result.size()*dims_t::n_dims);
				val_this.resize(crit_result.size());
				for(size_t i=0;i<crit_result.size();++i)
				{
					size_t s=crit_result[i];
					val_this[i]=temp_val[s];
					double sig = temp_vec[s*dims_t::n_dims]>0.? 1.:-1.;
					//to store the left hand side basis, multiply by S
					intrinsic::symb_mul_vecd<dims_t::n_dims,dims_t::n_elem,0>
					(
						sc.sub(),
						temp_vec.data()+s*dims_t::n_dims,
						vec_this.data()+i*dims_t::n_dims
					);
					
				}
			}
		}
	}//end of initialize
};//end of eigenstate_sph

template<class dims_t>
struct	eigenstate_bic final
{
	recvec<std::vector<double>,recidx<dims_t::m_dims>> vec;
	recvec<std::vector<double>,recidx<dims_t::m_dims>> val;
	recvec<int                ,recidx<dims_t::m_dims>> num;
	recvec<std::vector<comp_t>,recidx<dims_t::m_dims>> prj;
		
	eigenstate_bic(eigenstate_bic const&)=delete;
	eigenstate_bic(eigenstate_bic&&)=delete;

	eigenstate_bic()
	{
		vec.data=new (std::nothrow) std::vector<double>[vec.dim.n_leng];
		val.data=new (std::nothrow) std::vector<double>[val.dim.n_leng];
		num.data=new (std::nothrow) int                [num.dim.n_leng];
		prj.data=new (std::nothrow) std::vector<comp_t>[prj.dim.n_leng];
	}//end of constructor
	~eigenstate_bic()noexcept
	{
		delete[] vec.data;
		delete[] val.data;
		delete[] num.data;
		delete[] prj.data;
	}//end of destructor

	inline	void	projection
	(
		const coefficient_view<dims_t>& coef
	)noexcept
	{
		for(size_t im=0;im<dims_t::m_dims;++im)
		{
			for(size_t is=0;is<size_t(num(im));++is)
			{
				double*	lhs= vec[im].data()+is*dims_t::n_dims*dims_t::l_dims;
				comp_t* rhs=coef(im).data;
				comp_t	sum=zero<comp_t>;
				for(size_t i=0;i<dims_t::n_dims*dims_t::l_dims;++i)
				{
					sum+=lhs[i]*rhs[i];
				}
				prj(im)[is]=sum;
			}
		}
	}//end of projection	

	template<size_t hb_flag>
	inline	void	initialize
	(
		const operator_sb<dims_t>&	  s,	//the operator S
		const operator_hb<dims_t,hb_flag>&h,	//the operator H	
		const double emin,
		const double emax,
		const size_t neig
	)
	{
		if(emin>emax)return;

		auto	eig	=	csreig<double>();//see structure/csrmat.h
		auto	dsc	=	matrix_descr();
		auto	scsr	=	sparse_matrix_t();
		dsc.type=SPARSE_MATRIX_TYPE_SYMMETRIC;
		dsc.mode=SPARSE_FILL_MODE_UPPER;
		dsc.diag=SPARSE_DIAG_NON_UNIT;

		for(size_t im=0;im<dims_t::m_dims;++im)
		{
			auto sm	=	std::move(s.create_csrmat_real(im));
			auto hm	=	std::move(h.create_csrmat_real(im));
			int info=	gv_ds_csr
			(
				hm(),sm(),eig,neig,emin,emax//find all states between (Emin,Emax)
			);//call csr FEAST to do diagonalization
			if(info)
			{
				printf("feast error:%d\n",info);return;
			}
			num[im]=eig.fnd;
			vec[im].resize(eig.fnd*dims_t::l_dims*dims_t::n_dims);
			val[im].resize(eig.fnd);
			prj[im].resize(eig.fnd);
			std::copy(eig.val.data(),eig.val.data()+eig.fnd,val[im].data());

			mkl_sparse_d_create_csr
			(
				&scsr,SPARSE_INDEX_BASE_ONE,sm.nrow(),sm.nrow(),sm.row.data(),sm.row.data()+1,sm.col.data(),sm.val.data()
			);//call mkl to create a sparse handle

			for(size_t i=0;i<size_t(eig.fnd);++i)
			{
				double* src	=	eig.vec.data()+i*dims_t::l_dims*dims_t::n_dims;
				double* dst	=	vec[im].data()+i*dims_t::l_dims*dims_t::n_dims;
				mkl_sparse_d_mv
				(
					SPARSE_OPERATION_NON_TRANSPOSE,1.0,scsr,dsc,src,0.0,dst
				);//call mkl to implement S*v
			}
			
			mkl_sparse_destroy(scsr);
		}	
	}//end of initialize
		
};//end of eigenstate_bic
