#pragma once

//=======================================================================================================================
//				workspace for doing propagation
//=======================================================================================================================
#include <spherical/propagator/workspace.hpp>

//=======================================================================================================================
//				functions for propagation in the diagonalized space
//=======================================================================================================================
#include <spherical/propagator/propagate.hpp>

//=======================================================================================================================
//				databases for transform into/from the diagonalized space
//=======================================================================================================================
template<class dims_t,class data_t>
struct	cn_transform_hl //note: this class does not own the data unless 'allocation' is called and rsrc_t is set to nonzero
{
	using	temp_t  =       cn_workspace_so<dims_t>;
	using	coef_t	=	coefficient<dims_t>;
	using	yprj_t	=	recvec<data_t,recidx<dims_t::m_dims,dims_t::l_dims,dims_t::l_dims>>;

	yprj_t	yprj;	//refer to the projector matrix (onto Y representation, Y = P^H * Yd * P) (!!P stored in row major!!)
	rsrc_t*	rsrc;

	//-------------------------------------------------------------------------------------------------------
	explicit cn_transform_hl():yprj{0},rsrc(0){}			//could be a member (allocate memory by 'allocation')
	explicit cn_transform_hl(data_t* _yprj):yprj{_yprj},rsrc(0){}	//could be a viewer (does not need 'allocation')
	cn_transform_hl(const cn_transform_hl&)=delete;
	cn_transform_hl(cn_transform_hl&& rhs)noexcept:yprj(rhs.yprj),rsrc(rhs.rsrc){rhs.rsrc=0;}
	~cn_transform_hl()noexcept{if(rsrc)rsrc->release(yprj.data);}
	//-------------------------------------------------------------------------------------------------------
	void	allocation	(rsrc_t* _rsrc)
	{
		if(rsrc)rsrc->release(yprj.data);
		rsrc=_rsrc;//to erase data without reallocation, just let _rsrc=0.
		if(rsrc)rsrc->acquire((void**)&yprj.data,sizeof(data_t)*yprj.dim.n_leng,align_val_avx3);
	}//end of allocation
	//-------------------------------------------------------------------------------------------------------
	void	conjugate	()noexcept
	{
		if constexpr(std::is_same_v<data_t,comp_t>)
		{
			#pragma omp for
			for(size_t i=0ul;i<yprj.dim.n_leng;++i)
			{
				yprj[i][1] = - yprj[i][1];
			}
		}//otherwise, do nothing
	}//end of conjugate
		
	void	copy_real	(recvec<comp_t,recidx<dims_t::m_dims,dims_t::l_dims,dims_t::l_dims>> const ysrc)noexcept
	{
		if constexpr(std::is_same_v<data_t,double>)
		{
			#pragma omp for
			for(size_t i=0ul;i<yprj.dim.n_leng;++i)
			{
				yprj[i]   =  ysrc[i][0];
			}
		}
	}//end of copy_real

	void	copy_imag	(recvec<comp_t,recidx<dims_t::m_dims,dims_t::l_dims,dims_t::l_dims>> const ysrc)noexcept
	{
		if constexpr(std::is_same_v<data_t,double>)
		{
			#pragma omp for
			for(size_t i=0ul;i<yprj.dim.n_leng;++i)
			{
				yprj[i]   =  ysrc[i][1];
			}
		}
	}//end of copy_real

	void	merge_two	(recvec<data_t,recidx<dims_t::m_dims,dims_t::l_dims,dims_t::l_dims>> const ylhs,
				 recvec<data_t,recidx<dims_t::m_dims,dims_t::l_dims,dims_t::l_dims>> const yrhs)noexcept
	{//compute and store Plhs * Prhs^H
		if constexpr(std::is_same_v<data_t,comp_t>)
		{
			cblas_zgemm_batch_strided(CblasRowMajor,CblasNoTrans,CblasConjTrans,
			dims_t::l_dims,dims_t::l_dims,dims_t::l_dims, //m,n,k
			&idty<comp_t>,
			ylhs.data,dims_t::l_dims,dims_t::l_dims*dims_t::l_dims,//A,ldA,strA
			yrhs.data,dims_t::l_dims,dims_t::l_dims*dims_t::l_dims,//B,ldB,strB
			&zero<comp_t>,
			yprj.data,dims_t::l_dims,dims_t::l_dims*dims_t::l_dims,dims_t::m_dims);//C,ldC,strC
		}else
		if constexpr(std::is_same_v<data_t,double>)
		{
			cblas_dgemm_batch_strided(CblasRowMajor,CblasNoTrans,CblasTrans,
			dims_t::l_dims,dims_t::l_dims,dims_t::l_dims, //m,n,k
			idty<double>,
			ylhs.data,dims_t::l_dims,dims_t::l_dims*dims_t::l_dims,//A,ldA,strA
			yrhs.data,dims_t::l_dims,dims_t::l_dims*dims_t::l_dims,//B,ldB,strB
			zero<double>,
			yprj.data,dims_t::l_dims,dims_t::l_dims*dims_t::l_dims,dims_t::m_dims);//C,ldC,strC	
		}
	}//end of merge_two
	//-------------------------------------------------------------------------------------------------------
	void	construct	(coef_t& coef,temp_t& temp)noexcept
	{//construct by C'=P*C, from coef to tmpc, 
		if constexpr(std::is_same_v<data_t,comp_t>)//for yprj complex
		{
			cblas_zgemm_batch_strided(CblasRowMajor,CblasNoTrans,CblasNoTrans,//C=mn,op(A)=mk,op(B)=kn
			dims_t::l_dims,dims_t::n_dims,dims_t::l_dims,//m,n,k
			&idty<comp_t>,
			yprj.data,dims_t::l_dims,dims_t::l_dims*dims_t::l_dims,//A,ldA,strA
			coef.data,dims_t::n_dims,dims_t::l_dims*dims_t::n_dims,//B,ldB,strB
			&zero<comp_t>,
			temp(),dims_t::n_dims,dims_t::l_dims*dims_t::n_dims,dims_t::m_dims);//C,ldC,strC,nbatch
		}else
		if constexpr(std::is_same_v<data_t,double>)//for yprj purereal
		{
			cblas_dgemm_batch_strided(CblasRowMajor,CblasNoTrans,CblasNoTrans,//C=mn,op(A)=mk,op(B)=kn
			dims_t::l_dims,dims_t::n_dims*2ul,dims_t::l_dims,//m,n,k
			idty<double>,
			(double*)yprj.data,    dims_t::l_dims,    dims_t::l_dims*dims_t::l_dims,//A,ldA,strA
			(double*)coef.data,2ul*dims_t::n_dims,2ul*dims_t::l_dims*dims_t::n_dims,//B,ldB,strB
			zero<double>,
			(double*)temp(),2ul*dims_t::n_dims,2ul*dims_t::l_dims*dims_t::n_dims,dims_t::m_dims);//C,ldC,strC,nbatch
		}
	}//end of construct2

	void	eliminate	(coef_t& coef,temp_t& temp)noexcept
	{//eliminate by C=P^H*C',  from tmpc to coef
		if constexpr(std::is_same_v<data_t,comp_t>)//for yprj complex
		{
			cblas_zgemm_batch_strided(CblasRowMajor,CblasConjTrans,CblasNoTrans,//C=mn,op(A)=mk,op(B)=kn
			dims_t::l_dims,dims_t::n_dims,dims_t::l_dims,//m,n,k
			&idty<comp_t>,
			yprj.data,dims_t::l_dims,dims_t::l_dims*dims_t::l_dims,//A,ldA,strA
			temp(),dims_t::n_dims,dims_t::l_dims*dims_t::n_dims,//B,ldB,strB
			&zero<comp_t>,
			coef.data,dims_t::n_dims,dims_t::l_dims*dims_t::n_dims,dims_t::m_dims);//C,ldC,strC
		}else
		if constexpr(std::is_same_v<data_t,double>)//for yprj purereal
		{
			cblas_dgemm_batch_strided(CblasRowMajor,CblasTrans,CblasNoTrans,//C=mn,op(A)=mk,op(B)=kn
			dims_t::l_dims,dims_t::n_dims*2ul,dims_t::l_dims,//m,n,k
			idty<double>,
			(double*)yprj.data,dims_t::l_dims,dims_t::l_dims*dims_t::l_dims,//A,ldA,strA
			(double*)temp(),2ul*dims_t::n_dims,2ul*dims_t::l_dims*dims_t::n_dims,//B,ldB,strB
			zero<double>,
			(double*)coef.data,2ul*dims_t::n_dims,2ul*dims_t::l_dims*dims_t::n_dims,dims_t::m_dims);//C,ldC,strC	
		}
	}//end of eliminate
	//-------------------------------------------------------------------------------------------------------
};//end of cn_transform_hl

template<class dims_t,class data_t>
struct	cn_transform_hm//note: this class does not own the data unless 'allocation' is called and rsrc_t is set to nonzero
{
	using	temp_t  =       cn_workspace_so<dims_t>;
	using	coef_t	=	coefficient<dims_t>;
	using	zprj_t	=	recvec<data_t,recidx<dims_t::m_dims*dims_t::l_dims,dims_t::m_dims*dims_t::l_dims>>;

	zprj_t	zprj;	//refer to the projector matrix (onto Z representation, Z = P^H * Zd * P) (!!!P stored in row major!!!)
	rsrc_t*	rsrc;	
	//-------------------------------------------------------------------------------------------------------
	explicit cn_transform_hm():zprj{0},rsrc(0){}			//could be a member (allocate memory by 'allocation')
	explicit cn_transform_hm(data_t* _zprj):zprj{_zprj},rsrc(0){}	//could be a viewer (does not need 'allocation')
	cn_transform_hm(const cn_transform_hm&)=delete;
	cn_transform_hm(cn_transform_hm&& rhs)noexcept:zprj(rhs.zprj),rsrc(rhs.rsrc){rhs.rsrc=0;}
	~cn_transform_hm()noexcept{if(rsrc)rsrc->release(zprj.data);}
	//-------------------------------------------------------------------------------------------------------
	void	allocation	(rsrc_t* _rsrc)
	{
		if(rsrc)rsrc->release(zprj.data);
		rsrc=_rsrc;//to erase data without reallocation, just let _rsrc=0
		if(rsrc)rsrc->acquire((void**)&zprj.data,sizeof(comp_t)*zprj.dim.n_leng,align_val_avx3);
	}//end of allocation
	//-------------------------------------------------------------------------------------------------------
	void	conjugate	()noexcept
	{
		if constexpr(std::is_same_v<data_t,comp_t>)
		{
			#pragma omp for
			for(size_t i=0ul;i<zprj.dim.n_leng;++i)
			{
				zprj[i][1] = - zprj[i][1];
			}
		}//otherwise, do nothing
	}//end of conjugate
		
	void	copy_real	(recvec<comp_t,recidx<dims_t::m_dims*dims_t::l_dims,dims_t::m_dims*dims_t::l_dims>> const zsrc)noexcept
	{
		if constexpr(std::is_same_v<data_t,double>)
		{
			#pragma omp for
			for(size_t i=0ul;i<zprj.dim.n_leng;++i)
			{
				zprj[i]   =  zsrc[i][0];
			}
		}
	}//end of copy_real

	void	copy_imag	(recvec<comp_t,recidx<dims_t::m_dims*dims_t::l_dims,dims_t::m_dims*dims_t::l_dims>> const zsrc)noexcept
	{
		if constexpr(std::is_same_v<data_t,double>)
		{
			#pragma omp for
			for(size_t i=0ul;i<zprj.dim.n_leng;++i)
			{
				zprj[i]   =  zsrc[i][1];
			}
		}
	}//end of copy_real

	void	merge_two	(recvec<data_t,recidx<dims_t::m_dims*dims_t::l_dims,dims_t::m_dims*dims_t::l_dims>> const zlhs,
				 recvec<data_t,recidx<dims_t::m_dims*dims_t::l_dims,dims_t::m_dims*dims_t::l_dims>> const zrhs)noexcept
	{//compute and store Plhs * Prhs^H
		if constexpr(std::is_same_v<data_t,comp_t>)
		{
			cblas_zgemm(CblasRowMajor,CblasNoTrans,CblasConjTrans, //C=mn,op(A)=mk,op(B)=kn, C:=alpha*op(A)*op(B)+beta*C
			dims_t::m_dims*dims_t::l_dims,dims_t::m_dims*dims_t::l_dims,dims_t::m_dims*dims_t::l_dims,  //m,n,k
			&idty<comp_t>,//alpha
			zlhs.data, dims_t::m_dims*dims_t::l_dims, //A,ldA
			zrhs.data, dims_t::m_dims*dims_t::l_dims, //B,ldB
			&zero<comp_t>,//beta
			zprj.data, dims_t::m_dims*dims_t::l_dims);//C,ldC
		}else
		if constexpr(std::is_same_v<data_t,double>)
		{
			cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasConjTrans, //C=mn,op(A)=mk,op(B)=kn, C:=alpha*op(A)*op(B)+beta*C
			dims_t::m_dims*dims_t::l_dims,dims_t::m_dims*dims_t::l_dims,dims_t::m_dims*dims_t::l_dims,  //m,n,k
			idty<double>,//alpha
			zlhs.data, dims_t::m_dims*dims_t::l_dims, //A,ldA
			zrhs.data, dims_t::m_dims*dims_t::l_dims, //B,ldB
			zero<double>,//beta
			zprj.data, dims_t::m_dims*dims_t::l_dims);//C,ldC	
		}
	}//end of merge_two
	//---------------------------------------------------------------------------------------------------------
	template<int isrc,int idst>
	void	construct	(coef_t& coef,temp_t& temp)noexcept
	{//construct by C'=P*C
		size_t constexpr src_min = dims_t::msub_ilow(isrc);
		size_t constexpr src_max = dims_t::msub_iupp(isrc);
		size_t constexpr dst_min = dims_t::msub_ilow(idst);
		size_t constexpr dst_max = dims_t::msub_iupp(idst);
		if constexpr(std::is_same_v<data_t,comp_t>)
		{
			cblas_zgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,//C=mn,op(A)=mk,op(B)=kn, C:=alpha*op(A)*op(B)+beta*C
			dst_max-dst_min,dims_t::n_dims,src_max-src_min,  //m,n,k
			&idty<comp_t>,//alpha
			zprj.data+src_min+dims_t::m_dims*dims_t::l_dims*dst_min,dims_t::m_dims*dims_t::l_dims, 	//A,ldA
			coef.data+src_min*dims_t::n_dims,dims_t::n_dims,                               		//B,ldB
			&zero<comp_t>,//beta
			temp()   +dst_min*dims_t::n_dims,dims_t::n_dims);//C,ldC
		}else
		if constexpr(std::is_same_v<data_t,double>)
		{
			cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,//C=mn,op(A)=mk,op(B)=kn, C:=alpha*op(A)*op(B)+beta*C
			dst_max-dst_min,2ul*dims_t::n_dims,src_max-src_min,  //m,n,k
			idty<double>,//alpha
			zprj.data+dst_min*dims_t::m_dims*dims_t::l_dims,dims_t::m_dims*dims_t::l_dims, //A,ldA
			(double*)(coef.data+src_min*dims_t::n_dims),2ul*dims_t::n_dims,                //B,ldB
			zero<double>,//beta
			(double*)(temp()   +dst_min*dims_t::n_dims),2ul*dims_t::n_dims);//C,ldC
		}
	}//end of construct
	
	template<int isrc,int idst>
	void	eliminate	(coef_t& coef,temp_t& temp)noexcept
	{//eliminate by C'=P^H*C
		size_t constexpr src_min = dims_t::msub_ilow(isrc);
		size_t constexpr src_max = dims_t::msub_iupp(isrc);
		size_t constexpr dst_min = dims_t::msub_ilow(idst);
		size_t constexpr dst_max = dims_t::msub_iupp(idst);
		if constexpr(std::is_same_v<data_t,comp_t>)
		{
			cblas_zgemm(CblasRowMajor,CblasConjTrans,CblasNoTrans,//C=mn,op(A)=mk,op(B)=kn, C:=alpha*op(A)*op(B)+beta*C
			dst_max-dst_min,dims_t::n_dims,src_max-src_min,  //m,n,k
			&idty<comp_t>,//alpha
			zprj.data+dst_min+src_min*dims_t::m_dims*dims_t::l_dims,dims_t::m_dims*dims_t::l_dims, //A,ldA
			temp()   +src_min*dims_t::n_dims,dims_t::n_dims, //B,ldB
			&zero<comp_t>,//beta
			coef.data+dst_min*dims_t::n_dims,dims_t::n_dims);
		}else
		if constexpr(std::is_same_v<data_t,double>)
		{
			cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,//C=mn,op(A)=mk,op(B)=kn, C:=alpha*op(A)*op(B)+beta*C
			dst_max-dst_min,2ul*dims_t::n_dims,src_max-src_min,  //m,n,k
			idty<double>,//alpha
			zprj.data+dst_min+src_min*dims_t::m_dims*dims_t::l_dims,dims_t::m_dims*dims_t::l_dims, //A,ldA
			(double*)(temp()   +src_min*dims_t::n_dims),2ul*dims_t::n_dims, //B,ldB
			zero<double>,//beta
			(double*)(coef.data+dst_min*dims_t::n_dims),2ul*dims_t::n_dims);//C,ldC	
		}	
	}//end of eliminate 
	//---------------------------------------------------------------------------------------------------------
};//end of cn_transform_hm

//=======================================================================================================================
//				databases for propagtion in the diagonalized space
//=======================================================================================================================
template<class dims_t>
struct	cn_component_hc
{
	using	temp_t	=	cn_workspace_so<dims_t>;
	using	umat_t	=	recvec<comp_t,recidx<dims_t::n_lmax+1ul,dims_t::n_dims,dims_t::n_elem>>;
	using	lmat_t	=	recvec<comp_t,recidx<dims_t::n_lmax+1ul,dims_t::n_dims,dims_t::n_elem>>;

	umat_t	upper;	//store (1@S)-idt/2*(1@H),each of size n_rsub, n_lmax in total.
	lmat_t	lower;	//store (1@S)+idt/2*(1@H),each of size n_rsub, n_lmax in total. (LDLT factorized)
	rsrc_t*	rsrc;	

	//memory affairs
	cn_component_hc()noexcept:rsrc(0){}
	cn_component_hc(cn_component_hc const&)=delete;
	cn_component_hc(cn_component_hc&&)=delete;
	cn_component_hc& operator=(cn_component_hc const&)=delete;
	cn_component_hc& operator=(cn_component_hc &&)=delete;
	~cn_component_hc()noexcept
	{
		if(rsrc!=0)
		{
			rsrc->release(upper.data);
			rsrc->release(lower.data);
		}
	}
	void	allocation	(rsrc_t* _rsrc)
	{
		if(rsrc!=0)
		{
			rsrc->release(upper.data);
                        rsrc->release(lower.data);
		}
		rsrc=_rsrc;
		if(rsrc!=0)
		{
			rsrc->acquire((void**)&upper.data,sizeof(comp_t)*upper.dim.n_leng,align_val_avx3);
			rsrc->acquire((void**)&lower.data,sizeof(comp_t)*upper.dim.n_leng,align_val_avx3);
		}else
		{
			throw errn_t("null arena parsed in when initializing cn_component_hc.\n");
		}
	}	
	//end of memory affairs

	int	initialize	(const operator_sc<dims_t>& s,const operator_hc<dims_t>& h,const double dt)
	{
		int err=0;
		#pragma omp parallel for reduction(+:err)
		for(size_t l=0;l<=dims_t::n_lmax;++l)
		{
			//copy
			for(size_t i=0;i<dims_t::n_rsub;++i)
			{
				double  si      =       s[i];
				comp_t  hi      =       h[l][i]*unim<comp_t>*dt*0.5;
				upper(l).data[i]=si-hi;//upper=S-H*idt/2
				lower(l).data[i]=si+hi;//lower=S+H*idt/2
			}
			//do in-place LDLT factorization to lower
			err+=impl_ed_simd::lu_zsb<dims_t::n_dims,dims_t::n_diag>(lower(l).data);//lower=LDL^T
		}
		return 	err;
	}//end of initialize

	void	propagate0	(coefficient_view<dims_t>& coef,temp_t& temp)noexcept
	{
		#pragma omp parallel num_threads(temp.nthd)
		{
			cn_hc_propagate0<dims_t>(upper,lower,coef,temp.tmp(omp_get_thread_num()));
		}
	}//end of propagate
};//end of cn_component_hc

template<class dims_t>
struct	cn_component_hl
{
	using	temp_t	=	cn_workspace_so<dims_t>;
	using	yeig_t	=	recvec<double,recidx<dims_t::m_dims,dims_t::l_dims>>;
	using	rmat_t	=	recvec<double,recidx<dims_t::n_dims,dims_t::n_elem>>;

	//upper = (1@S)-idt/2*f(t)*(Y@R)
	//lower = (1@S)+idt/2*f(t)*(Y@R)
	
	rmat_t	smat;	//just borrow	
	rmat_t	rmat;	//just borrow
	yeig_t	yeig;	//store the eigenvalue of Y, Yd
	double	step;	//step of time
	rsrc_t*	rsrc;

	//-------------------------------------------------------------------------------------------------------
	cn_component_hl()noexcept:yeig{0},step(0.),rsrc(0){}
	cn_component_hl(cn_component_hl const&)=delete;
	cn_component_hl(cn_component_hl&&)=delete;
	cn_component_hl& operator=(cn_component_hl const&)=delete;
	cn_component_hl& operator=(cn_component_hl&&)=delete;
	~cn_component_hl()noexcept{if(rsrc!=0)rsrc->release(yeig.data);}
	//-------------------------------------------------------------------------------------------------------	
	void	allocation	(rsrc_t* _rsrc)
	{
		if(rsrc!=0)rsrc->release(yeig.data);
		rsrc=_rsrc;//to erase data without reallocation, just let _rsrc=0.
		if(rsrc!=0)rsrc->acquire((void**)&yeig.data,sizeof(double)*yeig.dim.n_leng,align_val_avx3);
	}//end of allocation
	//-------------------------------------------------------------------------------------------------------	
	void	initialize	(rmat_t& s,rmat_t& r,const double dt)noexcept
	{
		smat=s;
		rmat=r;
		step=dt;
	}//end of initialize
	//-------------------------------------------------------------------------------------------------------
	void	propagate1	(temp_t& temp,double fz)noexcept
	{//Y hermitian and R real symmetric. possible usage: H=E*y10*r or H=-A*(I*q10)*(1/r). fz=E or -A.
		#pragma omp parallel num_threads(temp.nthd)
		{
			cn_hl_propagate1<dims_t>(yeig,smat,rmat,temp,fz*step);
		}
	}//end of propagate1

	void	propagate2	(temp_t& temp,double fz)noexcept
	{//Y hermitian and R imag asymmetric. possible usage: H=-A*y10*(I*d/dr). fz=E or -A
		#pragma omp parallel num_threads(temp.nthd)
		{
			cn_hl_propagate2<dims_t>(yeig,smat,rmat,temp,fz*step);
		}
	};//end of propagate2
	//---------------------------------------------------------------------------------------------------------
};//end of cn_component_hl

template<class dims_t>
struct	cn_component_hm
{
	using	temp_t	=	cn_workspace_so<dims_t>;
	using	zeig_t	=	recvec<double,recidx<dims_t::m_dims*dims_t::l_dims>>;
	using	rmat_t	=	recvec<double,recidx<dims_t::n_dims,dims_t::n_elem>>;

	//upper = (1@S)-idt/2*f(t)*(Z@R)
	//lower = (1@S)+idt/2*f(t)*(Z@R)
	
	//initialize to Z@R
	//Z:
	//	for   (y1m+y1p), c= 1. usually in x direction
	//	for I*(y1m-y1p), c=-i. usually in y direction
	//	for   (q1m+q1p), c= 1. usually in y direction
	//	for I*(q1m-q1p), c=-i. usually in x direction
	//	for   (p2m+p2p), c= 1. usually in x direction
	//	for I*(p2m-p2p), c=-i. usually in y direction
	//R:
	//	for 1/r or r or r^2, provide itself
	//	for I*d/dr, provide d/dr

	//initialize to Z@S
	//Z:
	//	for   (l0m+l0p), c= 1. usually in y direction
	//	for I*(l0m-l0p), c=-i. usually in x direction
	
	rmat_t	smat;	//just borrow	
	rmat_t	rmat;	//just borrow
	zeig_t	zeig;	//point to the eigenvalue of Z, i.e. Zd (borrowed)
	double	step;	//step of time
	rsrc_t*	rsrc;

	//-------------------------------------------------------------------------------------------------------
	cn_component_hm()noexcept:zeig{0},step(0.),rsrc(0){}
	cn_component_hm(cn_component_hm const&)=delete;
	cn_component_hm(cn_component_hm&&)=delete;
	cn_component_hm& operator=(cn_component_hm const&)=delete;
	cn_component_hm& operator=(cn_component_hm&&)=delete;

	~cn_component_hm()noexcept{if(rsrc!=0)rsrc->release(zeig.data);}
	//-------------------------------------------------------------------------------------------------------
	void	allocation	(rsrc_t* _rsrc)//call me before use
	{
		if(rsrc!=0)rsrc->release(zeig.data);
		rsrc=_rsrc;//to erase data without reallocation, just let _rsrc=0.
		if(rsrc!=0)rsrc->acquire((void**)&zeig.data,sizeof(double)*zeig.dim.n_leng,align_val_avx3);
	}//end of allocation
	//-------------------------------------------------------------------------------------------------------
	void	initialize	(rmat_t& s,rmat_t& r,const double dt)noexcept
	{
		smat=s;
		rmat=r;
		step=dt;
	}//end of initialize (radial)

	//---------------------------------------------------------------------------------------------------------
	template<int isub=-1>
	void	propagate1	(temp_t& temp,double f)noexcept	
	{//Z hermitian and R real symmetric.
		#pragma omp parallel num_threads(temp.nthd)
		{
			cn_hm_propagate1<dims_t,isub>(zeig,smat,rmat,temp,f*step);
		}
	}//end of propagate1

	template<int isub=-1>
	void	propagate2	(temp_t& temp,double f)noexcept	
	{//Z hermitian and R pure imag anti-symmetric.
		#pragma omp parallel num_threads(temp.nthd)
		{
			cn_hm_propagate2<dims_t,isub>(zeig,smat,rmat,temp,f*step);
		}
	}//end of propagate2

	template<int isub=-1>
	void	propagate3	(temp_t& temp,double f)noexcept
	{//Z hermitian and R equals S.
		#pragma omp parallel num_threads(temp.nthd)
		{
			cn_hm_propagate3<dims_t,isub>(zeig,f*step);
		}
	}//end of propagate3
	//---------------------------------------------------------------------------------------------------------
};//end of cn_component_hm

//=======================================================================================================================
//=======================================================================================================================
template<class dims_t>
struct	propagator_cn_hc
{
	public:
		cn_workspace_so<dims_t>	work;
		cn_component_hc<dims_t>	prop_hc;

		operator_sc<dims_t>	s;
		operator_hc<dims_t>	h;
	
		template<class potn_t>
		inline	void	initialize	
		(
			const size_t _nthd,
			const integrator_radi<dims_t>& radi,
			const integrator_angu<dims_t>& angu,
			const double  mass,
			const potn_t& potn,
			const double  step
		)
		{
			s.initialize(radi);
			h.initialize(radi,mass,potn);
			work.   allocation(global_default_arena_ptr,_nthd);
			prop_hc.allocation(global_default_arena_ptr);
			prop_hc.initialize(s,h,step);
		}//end of initialize

		inline	void	propagate	(coefficient<dims_t>& coef)noexcept
		{
			prop_hc.propagate0(coef,work);
		}//end of propagate
};//end of propagator_cn_hc

template<class dims_t>
struct	propagator_cn_hc_lgwz
{
	public:	
		cn_workspace_so<dims_t>	work;

		cn_component_hc<dims_t>	prop_hc;
		cn_component_hl<dims_t> prop_z1;

		cn_transform_hl<dims_t,double> tran_z1;

		operator_sc<dims_t>	s;
		operator_hc<dims_t>	h;
		operator_rpow1<dims_t>	r;
		operator_y10<dims_t>	y;

		template<class potn_t>
                inline  void    initialize      
		(
			const size_t _nthd,
			const integrator_radi<dims_t>& radi,
			const integrator_angu<dims_t>& angu,
			const double mass,
			const potn_t&potn,
			const double step
		)
                {
			cn_transform_hl<dims_t,comp_t> temp;

			s.initialize(radi);
			r.initialize(radi);
			y.initialize(angu);
			h.initialize(radi,mass,potn);
                        temp   .allocation(global_default_arena_ptr);
			work   .allocation(global_default_arena_ptr,_nthd);
                        prop_hc.allocation(global_default_arena_ptr);
			prop_z1.allocation(global_default_arena_ptr);
			tran_z1.allocation(global_default_arena_ptr);
                        prop_hc.initialize(s,h,step);
			prop_z1.initialize(s,r,step*0.5);
	
			y.solve_eigen(idty<comp_t>,prop_z1.yeig,temp.yprj);//Y@R, Y is real-symmetric

			tran_z1.copy_real(temp.yprj);
                }//end of initialize

		inline	void	propagate	
		(
			coefficient<dims_t>& coef,
			double ez_ti,
			double ez_tf
		)noexcept
		{
			//+++++++++++++++++++++++++++++++++++++++++//
			tran_z1.construct(coef,work);//purereal
			prop_z1.propagate1(work,ez_ti);
			tran_z1.eliminate(coef,work);//purereal
			//+++++++++++++++++++++++++++++++++++++++++//
			prop_hc.propagate0(coef,work);
			//+++++++++++++++++++++++++++++++++++++++++//
			tran_z1.construct(coef,work);//purereal
			prop_z1.propagate1(work,ez_tf);
			tran_z1.eliminate(coef,work);//purereal
			//+++++++++++++++++++++++++++++++++++++++++//
		}//end of propagate

		inline	void	propagate	(coefficient<dims_t>& coef)noexcept
		{
			prop_hc.propagate0(coef,work);
		}//end of propagate
};//end of propagator_cn_hc_lgwz

template<class dims_t>
struct	propagator_cn_hc_vgwz
{
	public:	
		cn_workspace_so<dims_t>	work;

		cn_component_hc<dims_t>	prop_hc;
		cn_component_hl<dims_t> prop_z1;
		cn_component_hl<dims_t> prop_z2;

		cn_transform_hl<dims_t,comp_t> tran_z1;
		cn_transform_hl<dims_t,double> tran_z2;
		cn_transform_hl<dims_t,comp_t> z2_2_z1;	

		operator_sc<dims_t>	s;
		operator_hc<dims_t>	h;
		operator_rinv1<dims_t>	r;
		operator_rdif1<dims_t> 	d;
		operator_y10<dims_t>	y;
		operator_q10<dims_t>	q;

		template<class potn_t>
                inline  void    initialize      
		(
			const size_t _nthd,
			const integrator_radi<dims_t>& radi,
			const integrator_angu<dims_t>& angu,
			const double mass,
			const potn_t&potn,
			const double step
		)
                {
			cn_transform_hl<dims_t,comp_t> temp;

			s.initialize(radi);
			r.initialize(radi);
			d.initialize(radi);
			y.initialize(angu);
			q.initialize(angu);
			h.initialize(radi,mass,potn);
			work   .allocation(global_default_arena_ptr,_nthd);
			temp   .allocation(global_default_arena_ptr);
                        prop_hc.allocation(global_default_arena_ptr);
			prop_z1.allocation(global_default_arena_ptr);
			prop_z2.allocation(global_default_arena_ptr);
			tran_z1.allocation(global_default_arena_ptr);
			z2_2_z1.allocation(global_default_arena_ptr);
			tran_z2.allocation(global_default_arena_ptr);
                        prop_hc.initialize(s,h,step);
			prop_z1.initialize(s,r,step*0.5);
			prop_z2.initialize(s,d,step*0.5);

			y.solve_eigen( idty<comp_t>,prop_z2.yeig,temp.yprj);//  Y@i*D,  Y real-symmetric
			tran_z2.copy_real(temp.yprj);	
	
			q.solve_eigen(-unim<comp_t>,prop_z1.yeig,tran_z1.yprj);//i*Q@1/R, iQ imag-anti-symmetric. ev_zhb uses the lower half panel, thus need an extra minus.
			tran_z1.conjugate();//convert from colmajor conjtran to rowmajor nontran
			
			z2_2_z1.merge_two(tran_z1.yprj,temp.yprj);
                }//end of initialize
			
		inline	void	propagate
		(
			coefficient<dims_t>& coef,
			double az_ti,//Az/M
			double az_tf
		)noexcept
		{
			az_ti=-az_ti;//required by propagate
			az_tf=-az_tf;//required by propagate

			//----------------------------------------//
			tran_z2.construct(coef,work);	//purereal
			prop_z2.propagate2(work,az_ti);
			std::swap(work.tmpc,coef.data);	//complex
			z2_2_z1.construct(coef,work);
			prop_z1.propagate1(work,az_ti);
			tran_z1.eliminate(coef,work);	//complex
			//----------------------------------------//
			prop_hc.propagate0(coef,work);
			//----------------------------------------//
			tran_z1.construct(coef,work);	//complex
			prop_z1.propagate1(work,az_tf);
			z2_2_z1.eliminate(coef,work);	//complex
			std::swap(work.tmpc,coef.data);
			prop_z2.propagate2(work,az_tf);
			tran_z2.eliminate(coef,work);	//purereal
			//----------------------------------------//
		}//end of propagate

		inline	void	propagate	(coefficient<dims_t>& coef)noexcept
		{
			prop_hc.propagate0(coef,work);
		}//end of propagate
};//end of propagator_cn_hc_vgwz

template<class dims_t>
struct	propagator_cn_hc_lgwr
{
	public:	
		cn_workspace_so<dims_t>	work;

		cn_component_hc<dims_t>	prop_hc;
		cn_component_hm<dims_t> prop_x1;
		cn_component_hm<dims_t> prop_y1;

		cn_transform_hm<dims_t,double>	tran_x1;
		cn_transform_hm<dims_t,comp_t>	x1_2_y1;
		cn_transform_hm<dims_t,comp_t>	tran_y1;
	
		operator_sc<dims_t>	s;
		operator_hc<dims_t>	h;
		operator_rpow1<dims_t>	r;
		operator_y11<dims_t,-1>	y;
		
		template<class potn_t>
                inline  void    initialize      
		(
			const size_t _nthd,
			const integrator_radi<dims_t>& radi,
			const integrator_angu<dims_t>& angu,
			const double mass,
			const potn_t&potn,
			const double step
		)
                {
			cn_transform_hm<dims_t,comp_t> temp_x1;

			s.initialize(radi);
			r.initialize(radi);
			y.initialize(angu);
			h.initialize(radi,mass,potn);
			work   .allocation(global_default_arena_ptr,_nthd);
                        prop_hc.allocation(global_default_arena_ptr);
			prop_x1.allocation(global_default_arena_ptr);
			prop_y1.allocation(global_default_arena_ptr);
			tran_x1.allocation(global_default_arena_ptr);
			x1_2_y1.allocation(global_default_arena_ptr);
			tran_y1.allocation(global_default_arena_ptr);
			temp_x1.allocation(global_default_arena_ptr);
			
                        prop_hc.initialize(s,h,step);
			prop_x1.initialize(s,r,step*0.5);
			prop_y1.initialize(s,r,step*0.5);
			y.solve_eigen( idty<comp_t>,prop_x1.zeig,temp_x1.zprj);//(Ex/2)*   (Ym+Yp) @ R.
			y.solve_eigen(-unim<comp_t>,prop_y1.zeig,tran_y1.zprj);//(Ey/2)* I*(ym-yp) @ R. note the lower half panel is used, extra minus is added.

			temp_x1.conjugate();//convert from colmajor conjtran to rowmajor nontran
			tran_y1.conjugate();//convert from colmajor conjtran to rowmajor nontran

			tran_x1.copy_real(temp_x1.zprj);
			x1_2_y1.merge_two(tran_y1.zprj,temp_x1.zprj);//U[y1]*U[x1]^H
                }//end of initialize

		template<int isub=-1>
		inline	void	propagate	
		(
			coefficient<dims_t>& coef,
			double 	ex_ti, double ex_tf, 
			double	ey_ti, double ey_tf
		)noexcept
		{
			ex_ti/=2.0;
			ey_ti/=2.0;
			ex_tf/=2.0;
			ey_tf/=2.0;
			//+++++++++++++++++++++++++++++++++++++++++++++++++++//
			tran_x1.template construct<-1,isub>(coef,work);//pure real construct
			prop_x1.template propagate1<isub>(work,ex_ti);
			std::swap(work.tmpc,coef.data);
			x1_2_y1.template construct<isub,isub>(coef,work);//purereal eliminate then complex construct
			prop_y1.template propagate1<isub>(work,ey_ti);
			tran_y1.template eliminate<isub,-1>(coef,work);//complex eliminate
			//+++++++++++++++++++++++++++++++++++++++++++++++++++//
			prop_hc.propagate0(coef,work);
			//+++++++++++++++++++++++++++++++++++++++++++++++++++//
			tran_y1.template construct<-1,isub>(coef,work);//complex construct
			prop_y1.template propagate1<isub>(work,ey_tf);
			x1_2_y1.template eliminate<isub,isub>(coef,work);//complex eliminate then purereal construct
			std::swap(work.tmpc,coef.data);
			prop_x1.template propagate1<isub>(work,ex_tf);
			tran_x1.template eliminate<isub,-1>(coef,work);//purereal eliminate
			//+++++++++++++++++++++++++++++++++++++++++++++++++++//
		}//end of propagate

		inline	void	propagate	(coefficient<dims_t>& coef)noexcept
		{
			prop_hc.propagate0(coef,work);
		}//end of propagate
};//end of propagator_cn_hc_lgwr

template<class dims_t>
struct	propagator_cn_hc_vgwr
{
	public:
		cn_workspace_so<dims_t>	work;

		cn_component_hc<dims_t>	prop_hc;
		cn_component_hm<dims_t> prop_x1;
		cn_component_hm<dims_t> prop_x2;
		cn_component_hm<dims_t> prop_y1;
		cn_component_hm<dims_t> prop_y2;

		cn_transform_hm<dims_t,double> tran_x1;
		cn_transform_hm<dims_t,double> tran_y2;
		cn_transform_hm<dims_t,comp_t> x1_2_x2;
		cn_transform_hm<dims_t,comp_t> x2_2_y1;
		cn_transform_hm<dims_t,comp_t> y1_2_y2;

		operator_sc<dims_t>	s;
		operator_hc<dims_t>	h;
		operator_rinv1<dims_t>	r;
		operator_rdif1<dims_t>	d;
		operator_y11<dims_t,-1>	y;
		operator_q11<dims_t,-1> q;
		
		template<class potn_t>
                inline  void    initialize      
		(
			const size_t _nthd,
			const integrator_radi<dims_t>& radi,
			const integrator_angu<dims_t>& angu,
			const double mass,
			const potn_t&potn,
			const double step
		)
                {
			s.initialize(radi);
			r.initialize(radi);
			d.initialize(radi);	
			y.initialize(angu);
			q.initialize(angu);
			h.initialize(radi,mass,potn);
	
			cn_transform_hm<dims_t,comp_t> temp_x1;
			cn_transform_hm<dims_t,comp_t> temp_x2;
			cn_transform_hm<dims_t,comp_t> temp_y1;
			cn_transform_hm<dims_t,comp_t> temp_y2;
			
			work   .allocation(global_default_arena_ptr,_nthd);
                        prop_hc.allocation(global_default_arena_ptr);
			prop_x1.allocation(global_default_arena_ptr);
			prop_x2.allocation(global_default_arena_ptr);
			prop_y1.allocation(global_default_arena_ptr);
			prop_y2.allocation(global_default_arena_ptr);

			tran_x1.allocation(global_default_arena_ptr);
			tran_y2.allocation(global_default_arena_ptr);

			x1_2_x2.allocation(global_default_arena_ptr);
			x2_2_y1.allocation(global_default_arena_ptr);
			y1_2_y2.allocation(global_default_arena_ptr);

			temp_x1.allocation(global_default_arena_ptr);
			temp_x2.allocation(global_default_arena_ptr);
			temp_y1.allocation(global_default_arena_ptr);
			temp_y2.allocation(global_default_arena_ptr);	

                       	prop_hc.initialize(s,h,step);
			prop_x1.initialize(s,d,step*0.5);//-Ax*  (Ym+Yp)@(S\I*d/dr)
			prop_x2.initialize(s,r,step*0.5);// Ax*I*(Qm-Qp)@(S\  1/r)
			prop_y1.initialize(s,d,step*0.5);//-Ay*I*(Ym-Yp)@(S\I*d/dr)
			prop_y2.initialize(s,r,step*0.5);//-Ay*  (Qm+Qp)@(S\  1/r)

			y.solve_eigen( idty<comp_t>,prop_x1.zeig,temp_x1.zprj);
			q.solve_eigen(-unim<comp_t>,prop_x2.zeig,temp_x2.zprj);//-1 since lower
			y.solve_eigen(-unim<comp_t>,prop_y1.zeig,temp_y1.zprj);//-1 since lower
			q.solve_eigen( idty<comp_t>,prop_y2.zeig,temp_y2.zprj);

			tran_x1.copy_real(temp_x1.zprj);
			tran_y2.copy_real(temp_y2.zprj);

			temp_x1.conjugate(); //convert from colmajor conjtran to rowmajor nontran
			temp_x2.conjugate(); //convert from colmajor conjtran to rowmajor nontran
			temp_y1.conjugate(); //convert from colmajor conjtran to rowmajor nontran
			temp_y2.conjugate(); //convert from colmajor conjtran to rowmajor nontran

			x1_2_x2.merge_two(temp_x2.zprj,temp_x1.zprj);//U[x2]*U[x1]^H
			x2_2_y1.merge_two(temp_y1.zprj,temp_x2.zprj);//U[y1]*U[x2]^H
			y1_2_y2.merge_two(temp_y2.zprj,temp_y1.zprj);//U[y2]*U[y1]^H
                }//end of initialize

		template<int isub=-1>	
		inline	void	propagate
		(
			coefficient<dims_t>& coef,
			double 	ax_ti, double ax_tf, //AX/M
			double	ay_ti, double ay_tf  //Ay/M
		)noexcept
		{
			ax_ti/=2.0;
			ay_ti/=2.0;
			ax_tf/=2.0;
			ay_tf/=2.0;

			//++++++++++++++++++++++++++++++++++++++++++++++++++++//
			tran_x1.template construct<-1,isub>(coef,work);//purereal construct
			prop_x1.template propagate2<isub>(work,-ax_ti);
			std::swap(work.tmpc,coef.data);
			x1_2_x2.template construct<isub,isub>(coef,work);//purereal eliminate then complex construct
			prop_x2.template propagate1<isub>(work, ax_ti);
                        std::swap(work.tmpc,coef.data);	
			x2_2_y1.template construct<isub,isub>(coef,work);//complex eliminate then complex construct
			prop_y1.template propagate2<isub>(work,-ay_ti);
			std::swap(work.tmpc,coef.data);
			y1_2_y2.template construct<isub,isub>(coef,work);//complex eliminate then purereal construct
			prop_y2.template propagate1<isub>(work,-ay_ti);
			tran_y2.template eliminate<isub,-1>(coef,work);
			//++++++++++++++++++++++++++++++++++++++++++++++++++++//
			prop_hc.propagate0(coef,work);//atomic
			//++++++++++++++++++++++++++++++++++++++++++++++++++++//
			tran_y2.template construct<-1,isub>(coef,work);		//purereal construct
			prop_y2.template propagate1<   isub>(work,-ay_tf);
			y1_2_y2.template eliminate<isub,isub>(coef,work);	//purereal eliminate then complex construct
			std::swap(work.tmpc,coef.data);
			prop_y1.template propagate2<isub>(work,-ay_tf);
			x2_2_y1.template eliminate<isub,isub>(coef,work);	//complex eliminate then complex construct
                       	std::swap(work.tmpc,coef.data);
			prop_x2.template propagate1<isub>(work, ax_tf);
			x1_2_x2.template eliminate<isub,isub>(coef,work);	//complex eliminate then purereal construct
                        std::swap(work.tmpc,coef.data);
			prop_x1.template propagate2<isub>(work,-ax_tf);
			tran_x1.template eliminate<isub,-1>(coef,work);		//pure real eliminate
			//++++++++++++++++++++++++++++++++++++++++++++++++++++//
		}//end of propagate
	
		inline	void	propagate	(coefficient<dims_t>& coef)noexcept
		{
			prop_hc.propagate0(coef,work);
		}//end of propagate
};//end of propagator_cn_hc_vgwr

template<class dims_t,size_t n_pole>
struct	propagator_cn_hm
{
	cn_workspace_sp<dims_t>		work;		
	
	csridx<0>			idx;	//store the index part of csr3
	std::vector<comp_t>		val;	//store the value part of csr3

	operator_sc<dims_t>		oper_s;
	operator_hm<dims_t,n_pole>	oper_h;
	comp_t				idt_hf;
	
	private:
	inline	void	initialize_idx()
	{
		idx.clear();
		for(size_t iml=0ul;iml<dims_t::m_dims*dims_t::l_dims;++iml)
		{
			for(size_t irow=0;irow<dims_t::n_dims;++irow)
			{
				size_t icol_min =       irow>=dims_t::n_diag?irow-dims_t::n_diag:0;
				size_t icol_max =       dims_t::n_dims-1ul-irow<dims_t::n_diag?dims_t::n_dims-1ul:irow+dims_t::n_diag;
				idx.push_irow();
				for(size_t jml=0ul;jml<dims_t::m_dims*dims_t::l_dims;++jml)
				{
					for(size_t icol=icol_min;icol<=icol_max;++icol)
					{
						idx.col.push_back(icol+jml*dims_t::n_dims);
					}
				}
			}
		}idx.push_irow();//last sensor
	}//end of initialize_idx

	inline	void	initialize_val()
	{
		val.resize(idx.col.size());
		auto&	_s	=	oper_s;
		auto	_h	=	oper_h.get_hmul();
		#pragma omp parallel for num_threads(work.nthd)
		for(size_t i=0;i<idx.row.size()-1ul;++i)
		{
			size_t iml=i/dims_t::n_dims;
			size_t ir =i%dims_t::n_dims;
	
			size_t	smin = idx.row[i];
			size_t 	smax = idx.row[i+1];
			for(size_t s=smin;s<smax;++s)
			{
				size_t j=idx.col[s];
				size_t jml=j/dims_t::n_dims;
				size_t jc =j%dims_t::n_dims;
				
				size_t idr,idc;
				if(jc>=ir)      {idr=ir;idc=jc-ir;}
				else            {idr=jc;idc=ir-jc;}
				
				if(iml==jml)
				{
					size_t ih=(2ul*dims_t::m_dims*dims_t::l_dims-1ul-iml)*iml/2ul+iml;
					val[s]=_s(idr,idc)+idt_hf*_h(ih,idr,idc);
				}else if(iml>jml)//lower
				{
					size_t ih=(2ul*dims_t::m_dims*dims_t::l_dims-1ul-jml)*jml/2ul+iml;
					val[s]=idt_hf*conj(_h(ih,idr,idc));
				}else//upper
				{
					size_t ih=(2ul*dims_t::m_dims*dims_t::l_dims-1ul-iml)*iml/2ul+jml;
					val[s]=idt_hf*_h(ih,idr,idc);
				}
			}
		}
	}//end of initialize_val

	public:
	template<class func_t>
	inline	void	initialize	//you should call oper_h::set_pole first, see operator.hpp
	(
		const size_t nthd,
		const integrator_radi<dims_t>& radi,
		const integrator_angu<dims_t>& angu,
		const comp_t step,
		const double mass,
		const func_t& func =[](double r){return 0.0;}//the centrifugal part potential
	)
	{
		work.allocation(global_default_arena_ptr,nthd,false);//disable lowrank;  pre-factorize S+idt/2*H.

		oper_s.initialize(radi);
		oper_h.template initialize<3>(radi,angu,func,mass);//3dec=11bin= both hmat and hmul are prepared

		idt_hf=0.5*step*unim<comp_t>;		
	
		this->initialize_idx();
		this->initialize_val();

		work.template call_pardiso<12>(val.data(),idx.row.data(),idx.col.data(),NULL,NULL);//symbolic factorization 
												   //+numeric factorization
	}//end of initialize

	inline	void	apply_upper	(comp_t* psi)noexcept
	{//w1:=[S-idt/2*H]*w0
		auto	_s	=	oper_s();
		auto	_h	=	oper_h.get_hmul();
		#pragma omp parallel for num_threads(work.nthd)
		for(size_t iml=0;iml<dims_t::m_dims*dims_t::l_dims;++iml)
		{
			comp_t* _dst 	=   	work.work+iml*dims_t::n_dims;
			size_t  ish 	= 	(2ul*dims_t::m_dims*dims_t::l_dims-1ul-iml)*iml/2ul;	
			size_t	jsh;
			//diagonal entry
			{
				comp_t*	_src	=	psi+iml*dims_t::n_dims;
				intrinsic::symb_mul_vecd<dims_t::n_dims,dims_t::n_elem,0>(_s,_src,_dst);
				intrinsic::symb_mul_vecd<dims_t::n_dims,dims_t::n_elem,1>(_h(ish+iml).data,_src,_dst,-idt_hf);
			}
			//upper panel
			for(size_t jml=1+iml;jml<dims_t::m_dims*dims_t::l_dims;++jml)
			{
				comp_t*	_src	=	psi+jml*dims_t::n_dims;
				intrinsic::symb_mul_vecd<dims_t::n_dims,dims_t::n_elem,1>(_h(ish+jml).data,_src,_dst,-idt_hf);	
			}	
			//lower panel (utilize zhp format)
			for(size_t jml=0;jml<iml;++jml)
			{
				jsh	=	(2ul*dims_t::m_dims*dims_t::l_dims-1ul-jml)*jml/2ul;
				comp_t*	_src	=	psi+jml*dims_t::n_dims;
				intrinsic::symb_cmul_vecd<dims_t::n_dims,dims_t::n_elem,1>(_h(jsh+iml).data,_src,_dst,-idt_hf);
			}
		}
	}//end of apply_upper

	inline	int	apply_lower	(comp_t* psi)noexcept
	{//w0:=[S+idt/2*H]\w1
		int info=work.template call_pardiso<33>(val.data(),idx.row.data(),idx.col.data(),work.work,psi);//apply substitution only
		return	info;
	}//end of apply_lower

	inline	int	propagate	(coefficient<dims_t>& coef)noexcept
	{
		//auto clc=timer(); clc.tic();
		this->apply_upper(coef());
		return
		this->apply_lower(coef());
		//clc.toc();double time2=clc.get();printf("upper %lf lower %lf\n",time1,time2);exit(-1);return 0;
	}//end of propagate
};//end of propagator_cn_hm


//XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
//	the followings are some propagators without atomic hamiltonian
//XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

template<class dims_t>
struct	propagator_cn_lgwz
{
	public:	
		cn_workspace_so<dims_t>	work;

		cn_component_hl<dims_t> prop_z1;
		cn_transform_hl<dims_t,double> tran_z1;

		operator_sc<dims_t>     s;
		operator_rpow1<dims_t>	r;
		operator_y10<dims_t>	y;

                inline  void    initialize      
		(
			const size_t _nthd,
			const integrator_radi<dims_t>& radi,
			const integrator_angu<dims_t>& angu,
			const double step
		)
                {
			cn_transform_hl<dims_t,comp_t> temp;

			s.initialize(radi);
			r.initialize(radi);
			y.initialize(angu);
			work   .allocation(global_default_arena_ptr,_nthd);
                        temp   .allocation(global_default_arena_ptr);
			prop_z1.allocation(global_default_arena_ptr);
			tran_z1.allocation(global_default_arena_ptr);
			prop_z1.initialize(s,r,step*0.5);
	
			y.solve_eigen(idty<comp_t>,prop_z1.yeig,temp.yprj);//Y@R, Y is real-symmetric

			tran_z1.copy_real(temp.yprj);
                }//end of initialize

		template<class dipo_t>
		inline	void	initialize
		(
			const size_t _nthd,
			const integrator_radi<dims_t>& radi,
			const integrator_angu<dims_t>& angu,
			const double step,
			const dipo_t&dipo
		)
		{
			cn_transform_hl<dims_t,comp_t> temp;

			s.initialize(radi);	
			radi.integrate_0r0(r,dipo);	
			y.initialize(angu);
			work   .allocation(global_default_arena_ptr,_nthd);
			prop_z1.allocation(global_default_arena_ptr);
			tran_z1.allocation(global_default_arena_ptr);
			prop_z1.initialize(s,r,step*0.5);

			y.solve_eigen(idty<comp_t>,prop_z1.yeig,temp.yprj);//Y@R, Y is real-symmetric

			tran_z1.copy_real(temp.yprj);
		}//end of initialize
			
		inline	void	propagate	
		(
			coefficient<dims_t>& coef,
			double ez
		)noexcept
		{
			tran_z1.construct(coef,work);//purereal
			prop_z1.propagate1(work,ez);
			tran_z1.eliminate(coef,work);//purereal
		}//end of propagate

		inline	void	propagate_fwd	(coefficient<dims_t>& coef, double ez)noexcept {this->propagate(coef,ez);}
		inline	void	propagate_bwd	(coefficient<dims_t>& coef, double ez)noexcept {this->propagate(coef,ez);}
};//end of propagator_cn_lgwz

template<class dims_t>
struct	propagator_cn_lgwr
{
	public:	
		cn_workspace_so<dims_t>	work;

		cn_component_hm<dims_t> prop_x1;
		cn_component_hm<dims_t> prop_y1;

		cn_transform_hm<dims_t,double> tran_x1;
		cn_transform_hm<dims_t,comp_t> x1_2_y1;
		cn_transform_hm<dims_t,comp_t> tran_y1;
	
		operator_sc<dims_t>	s;
		operator_rpow1<dims_t>	r;
		operator_y11<dims_t,-1>	y;

                inline  void    initialize      
		(
			const size_t _nthd,
			const integrator_radi<dims_t>& radi,
			const integrator_angu<dims_t>& angu,
			const double step
		)
                {
			cn_transform_hm<dims_t,comp_t> temp_x1;

			s.initialize(radi);
			r.initialize(radi);
			y.initialize(angu);
			work   .allocation(global_default_arena_ptr,_nthd);
			prop_x1.allocation(global_default_arena_ptr);
			prop_y1.allocation(global_default_arena_ptr);
			tran_x1.allocation(global_default_arena_ptr);
			x1_2_y1.allocation(global_default_arena_ptr);
			tran_y1.allocation(global_default_arena_ptr);
			temp_x1.allocation(global_default_arena_ptr);
			
			prop_x1.initialize(s,r,step*0.5);
			prop_y1.initialize(s,r,step*0.5);
			y.solve_eigen( idty<comp_t>,prop_x1.zeig,temp_x1.zprj);//(Ex/2)*   (Ym+Yp) @ R.
			y.solve_eigen(-unim<comp_t>,prop_y1.zeig,tran_y1.zprj);//(Ey/2)* I*(ym-yp) @ R. note the lower half panel is used, extra minus is added.

			temp_x1.conjugate();//convert from colmajor conjtran to rowmajor nontran
			tran_y1.conjugate();//convert from colmajor conjtran to rowmajor nontran

			tran_x1.copy_real(temp_x1.zprj);
			x1_2_y1.merge_two(tran_y1.zprj,temp_x1.zprj);//U[y1]*U[x1]^H
                }//end of initialize
		
		template<int isub=-1>
		inline	void	propagate_fwd
		(
			coefficient<dims_t>& coef,
			double 	ex, double ey
		)noexcept
		{
			ex/=2.0;
			ey/=2.0;
	
			tran_x1.template construct<-1,isub>(coef,work);//pure real construct
			prop_x1.template propagate1<isub>(work,ex);
			std::swap(work.tmpc,coef.data);
			x1_2_y1.template construct<isub,isub>(coef,work);//purereal eliminate then complex construct
			prop_y1.template propagate1<isub>(work,ey);
			tran_y1.template eliminate<isub,-1>(coef,work);//complex eliminate
		}//end of propagate_fwd

		template<int isub=-1>
		inline	void	propagate_bwd
		(
			coefficient<dims_t>& coef,
			double 	ex, double ey
		)noexcept
		{
			ex/=2.0;
			ey/=2.0;

			tran_y1.template construct<-1,isub>(coef,work);//complex construct
			prop_y1.template propagate1<isub>(work,ey);
			x1_2_y1.template eliminate<isub,isub>(coef,work);//complex eliminate then purereal construct
			std::swap(work.tmpc,coef.data);
			prop_x1.template propagate1<isub>(work,ex);
			tran_x1.template eliminate<isub,-1>(coef,work);//purereal eliminate
		}//end of propagate_bwd
};//end of propagator_cn_lgwr

template<class dims_t>
struct	propagator_cn_vgwz
{
	public:	
		cn_workspace_so<dims_t>	work;

		cn_component_hl<dims_t> prop_z1;
		cn_component_hl<dims_t> prop_z2;

		cn_transform_hl<dims_t,comp_t> tran_z1;
		cn_transform_hl<dims_t,double> tran_z2;
		cn_transform_hl<dims_t,comp_t> z2_2_z1;	

		operator_sc<dims_t>	s;
		operator_rinv1<dims_t>	r;
		operator_rdif1<dims_t> 	d;
		operator_y10<dims_t>	y;
		operator_q10<dims_t>	q;

                inline  void    initialize      
		(
			const size_t _nthd,
			const integrator_radi<dims_t>& radi,
			const integrator_angu<dims_t>& angu,
			const double step
		)
                {
			cn_transform_hl<dims_t,comp_t> temp;

			s.initialize(radi);
			r.initialize(radi);
			d.initialize(radi);
			y.initialize(angu);
			q.initialize(angu);
			work   .allocation(global_default_arena_ptr,_nthd);
			temp   .allocation(global_default_arena_ptr);
			prop_z1.allocation(global_default_arena_ptr);
			prop_z2.allocation(global_default_arena_ptr);
			tran_z1.allocation(global_default_arena_ptr);
			z2_2_z1.allocation(global_default_arena_ptr);
			tran_z2.allocation(global_default_arena_ptr);
			prop_z1.initialize(s,r,step*0.5);
			prop_z2.initialize(s,d,step*0.5);

			y.solve_eigen( idty<comp_t>,prop_z2.yeig,temp.yprj);//  Y@i*D,  Y real-symmetric
			tran_z2.copy_real(temp.yprj);	
	
			q.solve_eigen(-unim<comp_t>,prop_z1.yeig,tran_z1.yprj);//i*Q@1/R, iQ imag-anti-symmetric. ev_zhb uses the lower half panel, thus need an extra minus.
			tran_z1.conjugate();//convert from colmajor conjtran to rowmajor nontran
			
			z2_2_z1.merge_two(tran_z1.yprj,temp.yprj);
                }//end of initialize

		inline	void	propagate_fwd
		(
			coefficient<dims_t>& coef,
			double az//Az/M
		)noexcept
		{
			az=-az;//required by propagate
                        
			tran_z2.construct(coef,work);   //purereal
                	prop_z2.propagate2(work,az);
                        std::swap(work.tmpc,coef.data); //complex
                        z2_2_z1.construct(coef,work);
                   	prop_z1.propagate1(work,az);
                        tran_z1.eliminate(coef,work);   //complex
		}//end of propagate

		inline	void	propagate_bwd
		(
			coefficient<dims_t>& coef,
			double az//Az/M
		)noexcept
		{
			az=-az;//required by propagate
                 
			tran_z1.construct(coef,work);   //complex
			prop_z1.propagate1(work,az);
                        z2_2_z1.eliminate(coef,work);   //complex
                        std::swap(work.tmpc,coef.data);
                	prop_z2.propagate2(work,az);
                        tran_z2.eliminate(coef,work);   //purereal
		}//end of propagate	
};//end of propagator_cn_vgwz

template<class dims_t>
struct	propagator_cn_vgwr
{
	public:
		cn_workspace_so<dims_t>	work;

		cn_component_hm<dims_t> prop_x1;
		cn_component_hm<dims_t> prop_x2;
		cn_component_hm<dims_t> prop_y1;
		cn_component_hm<dims_t> prop_y2;

		cn_transform_hm<dims_t,double> tran_x1;
		cn_transform_hm<dims_t,double> tran_y2;

		cn_transform_hm<dims_t,comp_t> x1_2_x2;
		cn_transform_hm<dims_t,comp_t> x2_2_y1;
		cn_transform_hm<dims_t,comp_t> y1_2_y2;
	
		operator_sc<dims_t>	s;
		operator_rinv1<dims_t>	r;
		operator_rdif1<dims_t>	d;
		operator_y11<dims_t,-1>	y;
		operator_q11<dims_t,-1> q;

                inline  void    initialize      
		(
			const size_t _nthd,
			const integrator_radi<dims_t>& radi,
			const integrator_angu<dims_t>& angu,
			const double step
		)
                {
			s.initialize(radi);
			r.initialize(radi);
			d.initialize(radi);	
			y.initialize(angu);
			q.initialize(angu);
	
			cn_transform_hm<dims_t,comp_t> temp_x1;
			cn_transform_hm<dims_t,comp_t> temp_x2;
			cn_transform_hm<dims_t,comp_t> temp_y1;
			cn_transform_hm<dims_t,comp_t> temp_y2;
			
			work   .allocation(global_default_arena_ptr,_nthd);
			prop_x1.allocation(global_default_arena_ptr);
			prop_x2.allocation(global_default_arena_ptr);
			prop_y1.allocation(global_default_arena_ptr);
			prop_y2.allocation(global_default_arena_ptr);

			tran_x1.allocation(global_default_arena_ptr);
			tran_y2.allocation(global_default_arena_ptr);

			x1_2_x2.allocation(global_default_arena_ptr);
			x2_2_y1.allocation(global_default_arena_ptr);
			y1_2_y2.allocation(global_default_arena_ptr);

			temp_x1.allocation(global_default_arena_ptr);
			temp_x2.allocation(global_default_arena_ptr);
			temp_y1.allocation(global_default_arena_ptr);
			temp_y2.allocation(global_default_arena_ptr);	

			prop_x1.initialize(s,d,step*0.5);//-Ax*  (Ym+Yp)@(S\I*d/dr)
			prop_x2.initialize(s,r,step*0.5);// Ax*I*(Qm-Qp)@(S\  1/r)
			prop_y1.initialize(s,d,step*0.5);//-Ay*I*(Ym-Yp)@(S\I*d/dr)
			prop_y2.initialize(s,r,step*0.5);//-Ay*  (Qm+Qp)@(S\  1/r)

			y.solve_eigen( idty<comp_t>,prop_x1.zeig,temp_x1.zprj);
			q.solve_eigen(-unim<comp_t>,prop_x2.zeig,temp_x2.zprj);//-1 since lower
			y.solve_eigen(-unim<comp_t>,prop_y1.zeig,temp_y1.zprj);//-1 since lower
			q.solve_eigen( idty<comp_t>,prop_y2.zeig,temp_y2.zprj);

			tran_x1.copy_real(temp_x1.zprj);
			tran_y2.copy_real(temp_y2.zprj);

			temp_x1.conjugate(); //convert from colmajor conjtran to rowmajor nontran
			temp_x2.conjugate(); //convert from colmajor conjtran to rowmajor nontran
			temp_y1.conjugate(); //convert from colmajor conjtran to rowmajor nontran
			temp_y2.conjugate(); //convert from colmajor conjtran to rowmajor nontran

			x1_2_x2.merge_two(temp_x2.zprj,temp_x1.zprj);//U[x2]*U[x1]^H
			x2_2_y1.merge_two(temp_y1.zprj,temp_x2.zprj);//U[y1]*U[x2]^H
			y1_2_y2.merge_two(temp_y2.zprj,temp_y1.zprj);//U[y2]*U[y1]^H
                }//end of initialize
	
		template<int isub=-1>
		inline	void	propagate_fwd	
		(
			coefficient<dims_t>& coef,
			double 	ax, //AX/M
			double	ay  //Ay/M
		)noexcept
		{
			ax/=2.0;
			ay/=2.0;
	
			tran_x1.template construct<-1,isub>(coef,work);//purereal construct
			prop_x1.template propagate2<isub>(work,-ax);
                        std::swap(work.tmpc,coef.data);
                        x1_2_x2.template construct<isub,isub>(coef,work);//purereal eliminate then complex construct
                        prop_x2.template propagate1<isub>(work, ax);
                        std::swap(work.tmpc,coef.data);
                        x2_2_y1.template construct<isub,isub>(coef,work);//complex eliminate then complex construct
                        prop_y1.template propagate2<isub>(work,-ay);
                        std::swap(work.tmpc,coef.data);
                        y1_2_y2.template construct<isub,isub>(coef,work);//complex eliminate then purereal construct
                        prop_y2.template propagate1<isub>(work,-ay);
			tran_y2.template eliminate<isub,-1>(coef,work);
		}//end of propagate

		template<int isub=-1>
		inline	void	propagate_bwd
		(
			coefficient<dims_t>& coef,
			double 	ax, //AX/M
			double	ay  //Ay/M
		)noexcept
		{
			ax/=2.0;
			ay/=2.0;

			tran_y2.template construct<-1,isub>(coef,work);         //purereal construct
                        prop_y2.template propagate1<isub>(work,-ay);
                        y1_2_y2.template eliminate<isub,isub>(coef,work);       //purereal eliminate then complex construct
                        std::swap(work.tmpc,coef.data);
                        prop_y1.template propagate2<isub>(work,-ay);
                        x2_2_y1.template eliminate<isub,isub>(coef,work);       //complex eliminate then complex construct
                        std::swap(work.tmpc,coef.data);
                        prop_x2.template propagate1<isub>(work, ax);
                        x1_2_x2.template eliminate<isub,isub>(coef,work);       //complex eliminate then purereal construct
                        std::swap(work.tmpc,coef.data);
                        prop_x1.template propagate2<isub>(work,-ax);
                        tran_x1.template eliminate<isub,-1>(coef,work);         //pure real eliminate
		}//end of propagate
};//end of propagator_cn_vgwr