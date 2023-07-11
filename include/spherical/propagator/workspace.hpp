#pragma once

//COMMENTS:
//
//(0) The classes here are only used for building propagators for Crank-Nicolson Method in sphTDSE, molTDSE and oceTDSE.
// 
// They are not user classes, i.e. user code should never explicitly operate on them, in principle.
//
//(1) The usage of different species of workspace:
//
//	- cn_workspace_so is used for performing U = [S+a*Y@R]\[S-a*Y@R], where Y is pre-diagonalized.
//
//	- cn_workspace_sp is used for performing U = [S+a*Hsp]\[S-a*Hsp], where Hsp is sparsely distributed. PARDISO is 
//	called for.
//
//(2) Variables named "nthd" always means # threads in CPU context
//


template<class dims_t>
struct	cn_workspace_so final
{
	size_t 	nthd;	//the actual number of threads to be used for Crank-Nicolson Kernels
	comp_t*	work;	//store the temporary value of the coefficient C'.
	comp_t*	tmpc;	//store the temporary value of the coefficient C'.
	comp_t*	ldlt;	//store the temporary value of an LDLT or LU factorization of a banded matrix in r-space.
	rsrc_t*	rsrc;	

	static constexpr size_t n_simd = 4ul;	//for radial sub-array per time

	//access the workspace for it-th thread
	inline	comp_t*	tmp	(const size_t it)const noexcept{return work+it*dims_t::n_dims*n_simd;}
	inline	comp_t*	ldl	(const size_t it)const noexcept{return ldlt+it*dims_t::n_dims*(dims_t::n_diag+1ul)*n_simd;}

	//access tmpc as a coefficient array
	inline	comp_t* operator()(const size_t im,const size_t il)const noexcept//in l space
	{
		return 	tmpc+dims_t::l_dims*dims_t::n_dims*im+dims_t::n_dims*il;
	}
	inline	comp_t*	operator()(const size_t iml)const noexcept//in ml united space
	{
		return 	tmpc+dims_t::n_dims*iml;
	}
	inline	comp_t*	operator()()const noexcept
	{
		return	tmpc;
	}
	//memory affairs
	cn_workspace_so()noexcept:rsrc(0){}
	cn_workspace_so(cn_workspace_so const&)=delete;
	cn_workspace_so(cn_workspace_so &&)=delete;
	cn_workspace_so& operator=(cn_workspace_so const&)=delete;
	cn_workspace_so& operator=(cn_workspace_so &&)=delete;

	~cn_workspace_so()noexcept
	{
		if(rsrc!=0)
		{
			rsrc->release(work);
			rsrc->release(tmpc);
			rsrc->release(ldlt);
		}
	}

	void	allocation	(rsrc_t* _rsrc,size_t _nthd)//call this function before use
	{
		if(rsrc!=0)
		{
                	rsrc->release(work);
			rsrc->release(tmpc);
			rsrc->release(ldlt);
		}
		rsrc=_rsrc;
		nthd=_nthd;
		if(rsrc!=0)
		{
			rsrc->acquire((void**)&work,sizeof(comp_t)*(dims_t::n_dims*n_simd                       )*nthd,align_val_avx3);
			rsrc->acquire((void**)&tmpc,sizeof(comp_t)*(dims_t::n_dims*dims_t::l_dims*dims_t::m_dims)     ,align_val_avx3);
			rsrc->acquire((void**)&ldlt,sizeof(comp_t)*(dims_t::n_dims*(dims_t::n_diag+1ul)*n_simd  )*nthd,align_val_avx3);
		}else
		{
			throw errn_t("null arena parsed in when initializing cn_workspace_so.\n");
		}
	}//end of allocation
		
};//end of cn_workspace_so


template<class dims_t>
struct	cn_workspace_sp final
{
	using	mint_t	=	MKL_INT;

	static constexpr mint_t	neqn	=	mint_t(dims_t::n_leng);
	static constexpr mint_t	nfct	=	1;	//number of factors
	static constexpr mint_t	nrhs	=	1;	//number of right hand sides
	static constexpr mint_t	nmat	=	1;	//number of matrices
	static constexpr mint_t	fmsg	=	0;	//PARDISO message level
	static constexpr mint_t	imat	=	3;	//3=complex, structrally symmetric							

	//local workspace
	size_t	nthd;
	comp_t*	work;	

	//pardiso workspace
	void**	temp;	
	mint_t*	parm;

	rsrc_t*	rsrc;

	//memory affairs
	cn_workspace_sp()noexcept:rsrc(0){}
	cn_workspace_sp(cn_workspace_sp const&)=delete;
	cn_workspace_sp(cn_workspace_sp&&)=delete;
	
	cn_workspace_sp& operator=(cn_workspace_sp const&)=delete;
	cn_workspace_sp& operator=(cn_workspace_sp&&)=delete;
	
	~cn_workspace_sp()noexcept
	{
		if(rsrc)
		{
			rsrc->release(work);
			rsrc->release(temp);
			rsrc->release(parm);
		}
	}

	void	allocation	(rsrc_t* _rsrc,size_t _nthd,bool enable_lowrank=1)//call this function before use
	{
		if(rsrc)
		{
			rsrc->release(work);
			rsrc->release(temp);
			rsrc->release(parm);
		}
		nthd=_nthd;
		rsrc=_rsrc;
		if(rsrc)
		{
			rsrc->acquire((void**)&work,sizeof(comp_t)*dims_t::n_leng,align_val_avx3);
			rsrc->acquire((void**)&temp,sizeof(void**)*64ul,align_val_none);
			rsrc->acquire((void**)&parm,sizeof(mint_t)*64ul,align_val_none);
		}else
		{
			throw errn_t("null arena parsed in when initializing cn_workspace_sp.\n");
		}
		//do pardiso setup
		pardisoinit(temp,&imat,parm);
	
		parm[0]	=	1;	//use non-default iparm[1:63]
		parm[1]	=	2;	//fill-in reordering from METIS
	
		parm[5] =	0;    	//solutions are out-of-place
		parm[23]=	enable_lowrank?10:0;
		parm[26]=	0;	//disable matrix check
		parm[34]=	1;	//zero-based indexing
	}//end of allocation

	//phase: 
	//	11=symbolic factorization 
	//	22=numeric factorization
	//	33=forward&backward substitution
	//	23=22+33
	//	12=11+22
	//	13=11+22+33
	//	0 =free the memory of LU matrices,
	//	-1=free the memory of all matrices
	//
	//note: for lowrank update (parm[38]=1), you should use (11,23), not (11,22,33)
	
	template<int _phase>
	inline	int	call_pardiso	(comp_t* a,mint_t* ia,mint_t* ja,comp_t* rhs,comp_t* dst)noexcept
	{
		mint_t	phase=_phase;
		mint_t	error;
		parm[38]=0;	//disable lowrank update
		PARDISO(temp,&nfct,&nmat,&imat,&phase,&neqn,a,ia,ja,NULL,&nrhs,parm,&fmsg,rhs,dst,&error);
		return 	error;
	}//end of call_pardiso

	template<int _phase>
	inline	int	call_pardiso	(comp_t* a,mint_t* ia,mint_t* ja,comp_t* rhs,comp_t* dst,mint_t* updt)noexcept
	{
		mint_t	phase=_phase;
		mint_t	error;
		parm[38]=1;	//use lowrank update. note the flag in allocation should be true
		PARDISO(temp,&nfct,&nmat,&imat,&phase,&neqn,a,ia,ja,updt,&nrhs,parm,&fmsg,rhs,dst,&error);
		return	error;
	}//end of call_pardiso
};//end of cn_workspace_sp
