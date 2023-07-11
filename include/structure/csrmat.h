#pragma once
#include <libraries/support_std.h>
#include <libraries/support_avx.h>

#include <utilities/chain.h>
#include <utilities/error.h>

//csrmat stores the data of a csr matrix
//csreig stores the eigen value& eigen vector of a ev/gv problem from csr matrix.(Ax=lambda*x, Ax=lambda*Bx)
//csrinv stores the data required to solve a linear system for csr matrix. (Ax=B, A is csrmat.)

namespace qpc
{
//---------------------------------------------------------------------------------------------------------------------
//trait class. (to obtain the FP type of the eigen value once the FP type of the matrix elements are given)
	template<class data_t>
	struct	csr_eigval_type;
	template<>
	struct	csr_eigval_type<double>{using type_t=double;};
	template<>
	struct	csr_eigval_type<MKL_Complex16>{using type_t=double;};
	template<>
	struct	csr_eigval_type<float>{using type_t=float;};
	template<>
	struct	csr_eigval_type<MKL_Complex8>{using type_t=float;};

//---------------------------------------------------------------------------------------------------------------------
	template<int indexing=1>
	struct	csridx_view final
	{
		static_assert(indexing==0||indexing==1,"indexing should be either 0 or 1.");
		using   int_t   =       MKL_INT;//def as int by macro, see support_mkl.h

		int_t const* row;	//size = nrow+1
		int_t const* col;	//size = nval
		size_t	nrow;	
		size_t	nval;

		template<bool is_symmetric=1>
		inline	void	check	()noexcept//just for debug
		{
			sparse_struct handle;
			sparse_matrix_checker_init(&handle);
			handle.n=nrow;
			handle.csr_ia=(int_t*)row;
			handle.csr_ja=(int_t*)col;
			handle.indexing=indexing?MKL_ONE_BASED:MKL_ZERO_BASED;
			handle.matrix_structure=is_symmetric?MKL_STRUCTURAL_SYMMETRIC:MKL_GENERAL_STRUCTURE;
			handle.matrix_format=MKL_CSR;
			handle.message_level=MKL_PRINT;
			handle.print_style=MKL_C_STYLE;
			sparse_matrix_checker(&handle);
		}//end of check		
	};

	template<int indexing=1>
	struct	csridx final
	{
		static_assert(indexing==0||indexing==1,"indexing should be either 0 or 1.");
		using	int_t	=	MKL_INT;//def as int by macro, see support_mkl.h

		std::vector<int_t>      row;    //size = nrow+1
		std::vector<int_t>      col;    //size = nval

		inline	size_t	nval	()const noexcept{return col.size();}
		inline	size_t	nrow	()const noexcept{return row.size()-1ul;}

		inline	auto operator()()const noexcept{return csridx_view<indexing>{row.data(),col.data(),nrow(),nval()};}

		inline	void	push_irow	()//you should call me before starting a new row, and after all rows have been created. 
		{
			row.push_back(col.size()+indexing);
		}//end of push_irow

		inline	void	clear		()
		{
			row.clear();
			col.clear();
		}//end of clear
	};//end of csridx

	template<class data_t>
	using	csrval	=	std::vector<data_t>;
	//end of csrval
//---------------------------------------------------------------------------------------------------------------------
	template<class data_t,int indexing=1>
	struct	csrmat_view final
	{
		static_assert(indexing==0||indexing==1,"indexing should be either 0 or 1.");

		using   int_t   =       MKL_INT;//def as int by macro, see support_mkl.h

		int_t const*	row;
		int_t const*	col;
		data_t const*	val;
		size_t	nrow;
		size_t	nval;

		inline	void	print	()const noexcept//only for debug
		{
			printf("row size=%zu,\n",nrow+1ul);
			for(size_t i=0;i<=nrow;++i)
			{
				printf("row[%zu]=%d\n",i,row[i]);
			}
			printf("col size=%zu,\n",nval);
			for(size_t i=0;i< nval;++i)
			{
				printf("val[%zu]=%lf,col[%zu]=%d\n",i,val[i],i,col[i]);
			}
		}//end of print

		template<bool is_symmetric=1>
		inline	void	check	()const noexcept//only for debug
		{	
			csridx_view<indexing>{row,col,nrow,nval}.check();
		}//end of check

		inline	auto	convert_from(sparse_matrix_t src)noexcept
		{
			sparse_index_base_t bas;
			sparse_status_t info;
			int_t _nrow;
			int_t _ncol;
			int_t* _row_start;
			int_t* _row_end;
			int_t* _col;
			data_t* _val;
			if constexpr(std::is_same_v<double,data_t>)
			{
				info=mkl_sparse_d_export_csr(src,&bas,&_nrow,&_ncol,&_row_start,&_row_end,&_col,&_val);
			}
			if constexpr(std::is_same_v<MKL_Complex16,data_t>)
			{
				info=mkl_sparse_z_export_csr(src,&bas,&_nrow,&_ncol,&_row_start,&_row_end,&_col,&_val);
			}
			row=_row_start;
			col=_col;
			val=_val;
			nrow=_nrow;
			nval=_row_start[nrow]-_row_start[0];
			return 	info;
		}//end of convert_from
	};

	template<class data_t,int indexing=1>//indexing is either 0 or 1
	struct	csrmat final
	{
		static_assert(indexing==0||indexing==1,"indexing should be either 0 or 1.");

		using	int_t	=	MKL_INT;//def as int by macro, see support_mkl.h
		std::vector<int_t>	row;	//size = nrow+1
		std::vector<int_t>	col;	//size = nval
		std::vector<data_t>	val;	//size = nval

		//----------------------------------------------------------------------------------------
		inline	size_t	nval	()const noexcept{return col.size();}
		inline	size_t	nrow	()const noexcept{return row.size()-1ul;}

		inline	auto	operator()()const noexcept
		{
			return csrmat_view<data_t,indexing>{row.data(),col.data(),val.data(),nrow(),nval()};
		}
		//----------------------------------------------------------------------------------------
		//				to fill up/ clear up values
		//----------------------------------------------------------------------------------------
		inline	void	push_irow	()//you should call me after all rows have been pushed back. 
		{
			row.push_back(col.size()+indexing);
		}//end of push_irow

		template<class line_t,class icol_t>
		inline	void	push_line	(const size_t ncol,const line_t& line,const icol_t& icol)
		{
			push_irow();
			for(size_t i=0;i<ncol;++i)
                	{
                        	if constexpr(std::is_invocable_v<line_t,const size_t>)
                        	{val.push_back(line(i));}else
        	                {val.push_back(line[i]);}
	                        if constexpr(std::is_invocable_v<icol_t,const size_t>)//make sure the first must be the diagonal term, even it is zero.
        	                {col.push_back(icol(i)+indexing);}else
	                        {col.push_back(icol[i]+indexing);}
                	}	
		}//end of push_line
	};//end of csrmat

//---------------------------------------------------------------------------------------------------------------------
	template<class data_t>
	struct	csreig final//the resultant class of solving eigen value problem by FEAST (now only valid for symmetric/hermitian, ev/gv)
	{
		using	int_t	=	MKL_INT;
		using	val_t	=	typename csr_eigval_type<data_t>::type_t;
		using	vec_t	=	data_t;

		std::vector<int_t>	fpm;	//FEAST options
		std::vector<val_t>	val;	//to be filled with eigen value
		std::vector<vec_t>	vec;	//to be filled with eigen matrix (in col major)
		std::vector<val_t>	res;
		val_t			min;	//record the requirement of lower limit for val
		val_t			max;	//record the requirement of upper limit for val
		int_t			fnd;	//number of eigen values found

		template<int to_print=0>
		inline	void	initialize	(const int_t nrow,const int_t eignum,const val_t eigmin,const val_t eigmax)
		{
			fpm.resize(128);
			val.resize(eignum);
			vec.resize(eignum*nrow);
			res.resize(eignum);
			min=eigmin;
			max=eigmax;
        		feastinit(fpm.data());
			//################################# FEAST PARAM #################################//
        		fpm[0]=to_print;//enable runtime print
			//fpm[1]=16;	//number of contour points
			//fpm[2]=12;	//precision 
			//fpm[3]=30;	//maximum loop number
			//fpm[26]=1;	//enable check of CSR matrix
			//fpm[27]=1;	//enable check of positive definite of S matrix
			//fpm[63]=1;	//enable fpm[64+i] being used as pardiso iparam[i]
		}//end of initialize
	};//end of csreig
//---------------------------------------------------------------------------------------------------------------------
	struct	csrinv final
	{
		using	comp_t	=	MKL_Complex16;
		using	int_t	=	MKL_INT;

		_MKL_DSS_HANDLE_t handle;
		MKL_INT	opt;
		MKL_INT	sym;	//symmetric, hermitian, or none?
		MKL_INT	sgn;	//(hermitian)positive definite/indefinite?
		MKL_INT	err;	//error message
		

		template<bool is_sym=1,bool is_pos=1>
		inline	void	initialize	(const int_t* row,const int_t* col,const double* val,const int_t nrow,const int_t nval)
		{
			sym	=	is_sym?MKL_DSS_SYMMETRIC:MKL_DSS_NON_SYMMETRIC;
			sgn	=	is_pos?MKL_DSS_POSITIVE_DEFINITE:MKL_DSS_INDEFINITE;

			err	=	dss_define_structure(handle,sym,row,nrow,nrow,col,nval);
			if(err!=MKL_DSS_SUCCESS)goto throw_error;
			err	=	dss_reorder(handle,opt,NULL);
			if(err!=MKL_DSS_SUCCESS)goto throw_error;
			err	=	dss_factor_real(handle,sgn,val);
			if(err!=MKL_DSS_SUCCESS)goto throw_error;
			return;	
			throw_error:
			throw err;
		}//end of initialize (symmetric positive definite)

		template<bool is_her=1,bool is_pos=1>
		inline	void	initialize	(const int_t* row,const int_t* col,const comp_t* val,const int_t nrow,const int_t nval)
		{
			sym	=	is_her?MKL_DSS_SYMMETRIC_COMPLEX:MKL_DSS_NON_SYMMETRIC_COMPLEX;
			sgn	=	is_pos?MKL_DSS_HERMITIAN_POSITIVE_DEFINITE:MKL_DSS_HERMITIAN_INDEFINITE;

                        err     =       dss_define_structure(handle,sym,row,nrow,nrow,col,nval);
                        if(err!=MKL_DSS_SUCCESS)goto throw_error;
                        err     =       dss_reorder(handle,opt,NULL);
                        if(err!=MKL_DSS_SUCCESS)goto throw_error;
                        err     =       dss_factor_complex(handle,sgn,val);
                        if(err!=MKL_DSS_SUCCESS)goto throw_error;
                        return;
                        throw_error:
                        throw err;
		}//end of initialize (hermitian positive definite)

		//interfaces
		template<bool is_sym=1,bool is_pos=1>
		inline	void	initialize	(const csrmat<double,1>& mat)
		{initialize(mat.row.data(),mat.col.data(),mat.val.data(),mat.nrow(),mat.nval());}

		template<bool is_her=1,bool is_pos=1>
		inline	void	initialize	(const csrmat<comp_t,1>& mat)
		{initialize(mat.row.data(),mat.col.data(),mat.val.data(),mat.nrow(),mat.nval());}

		template<bool is_her=1,bool is_pos=1>
		inline	void	initialize	(const csrval<double>& val,const csridx<1>& idx){initialize(idx.row.data(),idx.col.data(),val.data(),idx.nrow(),idx.nval());}
		template<bool is_her=1,bool is_pos=1>
		inline	void	initialize	(const csrval<comp_t>& val,const csridx<1>& idx){initialize(idx.row.data(),idx.col.data(),val.data(),idx.nrow(),idx.nval());}


		//others
		inline	void	update		(const csrmat<double,1>& mat)
		{
			err     =       dss_factor_real(handle,sgn,mat.val.data());
                        if(err!=MKL_DSS_SUCCESS)throw runtime_error_t<std::string,void>{"csrinv::factor error code:"+to_string(err)};
		}//end of update (real)

		inline	void	update		(const csrmat<comp_t,1>& mat)
		{
			err     =       dss_factor_complex(handle,sgn,mat.val.data());
                        if(err!=MKL_DSS_SUCCESS)throw runtime_error_t<std::string,void>{"csrinv::factor error code:"+to_string(err)};	
		}//end of update (comp)

		inline	void	solve		(const double* rhs,double* sol,const int_t nrhs=1)
		{
			err	=	dss_solve_real(handle,opt,rhs,nrhs,sol);
			if(err!=MKL_DSS_SUCCESS)throw runtime_error_t<std::string,void>{"csrinv::solve error code:"+to_string(err)};
		}//end of solve (real)
	
		inline	void	solve		(const comp_t* rhs,comp_t* sol,const int_t nrhs=1)
		{
			err	=	dss_solve_complex(handle,opt,rhs,nrhs,sol);
			if(err!=MKL_DSS_SUCCESS)throw runtime_error_t<std::string,void>{"csrinv::solve error code:"+to_string(err)};
		}//end of solve (comp)
		
		//constructor&destructor (use RAII)
		csrinv()
		{
			opt	=	MKL_DSS_DEFAULTS;
			err	=	dss_create(handle,opt);
			if(err!=MKL_DSS_SUCCESS)throw runtime_error_t<std::string,void>{"csrinv::csrinv error code:"+to_string(err)};
		}
		~csrinv()noexcept
		{
			err     =       dss_delete(handle,opt);
		}
	};//end of csrinv

	
//----------------------------------------------------------------------------------------------------------------------------------------
//
//						 CSR Diagonalization using FEAST
//
//----------------------------------------------------------------------------------------------------------------------------------------	
	int	ev_ds_csr	(const csrmat_view<double,1>& csra,csreig<double>& retn,MKL_INT eignum,const double eigmin,const double eigmax)
	{//real, symmetric, Ax=lambda*x
		const char	uplo	=	'U';
		const MKL_INT	size	=	csra.nrow;
		double	epsout;
		MKL_INT	loop,info;	
		retn.initialize(size,eignum,eigmin,eigmax);
		dfeast_scsrev
        	(
                	&uplo,&size,
                	csra.val,csra.row,csra.col,
                	retn.fpm.data(),&epsout,&loop,&eigmin,&eigmax,&eignum,
			retn.val.data(),retn.vec.data(),&retn.fnd,retn.res.data(),
                	&info
        	);
		return	info;
	}
	int	ev_ds_csr	(const csrmat<double,1>& csra,csreig<double>& retn,MKL_INT eignum,const double eigmin,const double eigmax)
	{
		return 	ev_ds_csr(csra(),retn,eignum,eigmin,eigmax);
	}//end of ev_ds_csr

	int	ev_zh_csr	(const csrmat_view<MKL_Complex16,1>& csra,csreig<MKL_Complex16>& retn,MKL_INT eignum,const double eigmin,const double eigmax)
	{//complex, hermitian, Ax=lambda*x
		const char	uplo	=	'U';
		const MKL_INT	size	=	csra.nrow;	
		double	epsout;
		MKL_INT	loop,info;
		retn.initialize(size,eignum,eigmin,eigmax);
		zfeast_hcsrev
		(
			&uplo,&size,
			csra.val,csra.row,csra.col,
			retn.fpm.data(),&epsout,&loop,&eigmin,&eigmax,&eignum,
                        retn.val.data(),retn.vec.data(),&retn.fnd,retn.res.data(),
                        &info
                );
		return	info;
	}
	int	ev_zh_csr	(const csrmat<MKL_Complex16,1>& csra,csreig<MKL_Complex16>& retn,MKL_INT eignum,const double eigmin,const double eigmax)
	{
		return 	ev_zh_csr(csra(),retn,eignum,eigmin,eigmax);
	}//end of ev_zh_csr

	int     gv_ds_csr       (const csrmat_view<double,1>& csra,const csrmat_view<double,1>& csrb,csreig<double>& retn,MKL_INT eignum,const double eigmin,const double eigmax)
        {//real, symmetric, Ax=lambda*Bx
                const char      uplo    =       'U';
                const MKL_INT    	size    =       csra.nrow;//should be same to csrb.nrow()
                double  epsout;
                MKL_INT     loop,info;
                retn.initialize(size,eignum,eigmin,eigmax);
                dfeast_scsrgv
                (
                        &uplo,&size,
                        csra.val,csra.row,csra.col,
			csrb.val,csrb.row,csrb.col,
                        retn.fpm.data(),&epsout,&loop,&eigmin,&eigmax,&eignum,
                        retn.val.data(),retn.vec.data(),&retn.fnd,retn.res.data(),
                        &info
                );
                return  info;
        }
	int     gv_ds_csr       (const csrmat<double,1>& csra,const csrmat<double,1>& csrb,csreig<double>& retn,MKL_INT eignum,const double eigmin,const double eigmax)
	{
		return 	gv_ds_csr(csra(),csrb(),retn,eignum,eigmin,eigmax);
	}//end of gv_ds_csr
	int	gv_ds_csr	(const csridx<1>& idx,const csrval<double>& vala,const csrval<double>& valb,csreig<double>& retn,MKL_INT eignum,const double eigmin,const double eigmax)
	{
		auto view_a	=	csrmat_view<double>{idx.row.data(),idx.col.data(),vala.data(),idx.nrow(),idx.nval()};
		auto view_b	=	csrmat_view<double>{idx.row.data(),idx.col.data(),valb.data(),idx.nrow(),idx.nval()};
		return 	gv_ds_csr(view_a,view_b,retn,eignum,eigmin,eigmax);
	}//end of gv_ds_csr

	int	gv_zh_csr	(const csrmat_view<MKL_Complex16,1>& csra,const csrmat_view<MKL_Complex16,1>& csrb,csreig<MKL_Complex16>& retn,MKL_INT eignum,const double eigmin,const double eigmax)
	{//complex, hermitian, Ax=lambda*Bx
		const char	uplo	=	'U';
		const MKL_INT	size	=	csra.nrow;
		double	epsout;
		MKL_INT	loop,info;
                retn.initialize(size,eignum,eigmin,eigmax);
		zfeast_hcsrgv
		(
			&uplo,&size,
                        csra.val,csra.row,csra.col,
                        csrb.val,csrb.row,csrb.col,
                        retn.fpm.data(),&epsout,&loop,&eigmin,&eigmax,&eignum,
                        retn.val.data(),retn.vec.data(),&retn.fnd,retn.res.data(),
                        &info
		);
		return	info;
	}
	int	gv_zh_csr	(const csrmat<MKL_Complex16,1>& csra,const csrmat<MKL_Complex16,1>& csrb,csreig<MKL_Complex16>& retn,MKL_INT eignum,const double eigmin,const double eigmax)
	{
		return 	gv_zh_csr(csra(),csrb(),retn,eignum,eigmin,eigmax);
	}//end of gv_zh_csr
//======================================================================================================================
//
//				
//
//======================================================================================================================

//======================================================================================================================
//
//			  To expand a Band of Lapack-Band Matrix into a CSR Matrix
//
//======================================================================================================================

template<size_t n_diag,size_t n_dims,size_t l_diag,size_t l_dims>//n is the dsb dimension, l is the nested dimension
static inline void convert_nested_dsb_to_csr_loop	
(
	csrmat<double,1>& mat,
	const double* dsbmat,
	size_t i_row,	//=n_elem*ir+l_elem*jr , row index in lapmat, with banded indexing
	size_t i_col,	//=n_dims*jr+ir+1ul    , col index in csrmat, with one-based indexing
	size_t n_upp,	//the number of elements in row ir, in the upper half panel (including the diagonal ones)
	size_t n_low,	//the number of elements in row ir, in the lower half panel (excluding the diagonal ones)
	size_t m_upp	//the number of off-diagonal l-blocks in the upper half panel (including the diagonal ones)
)
{
	size_t constexpr n_elem	= n_diag+1ul;
	size_t constexpr n_subs	= n_elem*n_dims;

	mat.push_irow();
	{//pushback diagonal block
		for(size_t ic=0;ic<n_upp;++ic)
		{
			mat.val.push_back(dsbmat[i_row+ic]);
			mat.col.push_back(       i_col+ic );
		}
	}
	for(size_t jc=1;jc<m_upp;++jc)
	{//pushback off-diagonal block
		i_row+=n_subs;
		i_col+=n_dims;
		for(size_t ic=n_low;ic>0;--ic)
		{
			mat.val.push_back(dsbmat[i_row-ic*n_diag]);
			mat.col.push_back(       i_col-ic);
		}
		for(size_t ic=0;ic<n_upp;++ic)
		{
			mat.val.push_back(dsbmat[i_row+ic]);
			mat.col.push_back(       i_col+ic);
		}
	}
}//end of loop

template<size_t n_diag,size_t n_dims,size_t l_diag,size_t l_dims>//n is the dsb dimension, l is the nested dimension
void	convert_nested_dsb_to_csr	(csrmat<double,1>& mat,const double* dsbmat)
{
	size_t constexpr n_elem	= n_diag+1ul;
	size_t constexpr n_subs	= n_elem*n_dims;//number of elem in one n-block
	size_t constexpr l_elem	= (l_diag+1ul)*n_subs;//number of elem in a row of nl-block
	static_assert(n_diag*2ul<n_dims,"invalid banded matrix format.");
	static_assert(l_diag*2ul<l_dims,"invalid banded matrix format.");
	for(size_t jr=0;jr<l_dims;++jr)
	{
		size_t	i_row	=	l_elem*jr;
		size_t	i_col	=	n_dims*jr+1ul;//one-based indexing	
		size_t	m_upp	=	(jr<l_dims-l_diag)?(l_diag+1ul):(l_dims-jr);
		for(size_t ir=0;ir<n_diag;++ir)
		{//first few
			size_t	n_upp	=	n_elem;
			size_t	n_low	=	ir;
			convert_nested_dsb_to_csr_loop<n_diag,n_dims,l_diag,l_dims>(mat,dsbmat,i_row,i_col,n_upp,n_low,m_upp);
			i_row	+=	n_elem;
			i_col	+=	1ul;
		}
		for(size_t ir=n_diag;ir<n_dims-n_diag;++ir)
		{//middle
			size_t	n_upp	=	n_elem;	
			size_t	n_low	=	n_diag;
			convert_nested_dsb_to_csr_loop<n_diag,n_dims,l_diag,l_dims>(mat,dsbmat,i_row,i_col,n_upp,n_low,m_upp);
			i_row	+=	n_elem;
			i_col	+=	1ul;
		}
		for(size_t ir=n_dims-n_diag;ir<n_dims;++ir)
		{//last few
			size_t	n_upp	=	n_dims-ir;
			size_t	n_low	=	n_diag;
			convert_nested_dsb_to_csr_loop<n_diag,n_dims,l_diag,l_dims>(mat,dsbmat,i_row,i_col,n_upp,n_low,m_upp);
			i_row	+=	n_elem;
			i_col	+=	1ul;
		}
	}
	mat.push_irow();//sensor
}//end of convert_nested_dsb_to_csr

//--------------------------------------------------------------------------------------------------
template<int is_symmetric=1,int indexing=1>//0=general,1=symmetric,2=hermitian
static 	inline	void	dcsr_mul_zvec		(const double* a,const int* row,const int *col,const int nrow,const MKL_Complex16* src,MKL_Complex16* dst)noexcept//a naive implementation
{
	for(int i=0;i<nrow;++i)
	{
		auto sum=zero<MKL_Complex16>;
		for(int j=row[i]-indexing;j<row[i+1]-indexing;++j)
		{
			sum+=a[j]*src[col[j]-indexing];
		}
		dst[i]=sum;	
	}
	if constexpr(is_symmetric==1||is_symmetric==2)
	for(int i=0;i<nrow;++i)
	{
		auto tmp=src[i];
		for(int j=row[i]-indexing+1;j<row[i+1]-indexing;++j)
		{
			dst[col[j]-indexing]+=a[j]*tmp;
		}
	}
}//end of dcsr_mul_zvec

template<int is_symmetric=1,int indexing=1>//0=general,1=symmetric,2=hermitian
static 	inline	void	zcsr_mul_zvec		(const MKL_Complex16* a,const int* row,const int *col,const int nrow,const MKL_Complex16* src,MKL_Complex16* dst)noexcept//a naive implementation
{
	for(int i=0;i<nrow;++i)
	{
		auto sum=zero<MKL_Complex16>;
		for(int j=row[i]-indexing;j<row[i+1]-indexing;++j)
		{
			sum+=a[j]*src[col[j]-indexing];
		}
		dst[i]=sum;	
	}
	if constexpr(is_symmetric==1)//1=symmetric
	for(int i=0;i<nrow;++i)
	{
		auto tmp=src[i];
		for(int j=row[i]-indexing+1;j<row[i+1]-indexing;++j)
		{
			dst[col[j]-indexing]+=a[j]*tmp;
		}
	}
	if constexpr(is_symmetric==2)//2=hermitian
	for(int i=0;i<nrow;++i)
	{
		auto tmp=src[i];
		for(int j=row[i]-indexing+1;j<row[i+1]-indexing;++j)
		{
			dst[col[j]-indexing]+=avx_mulc(tmp,a[j]);
		}
	}
}//end of zcsr_mul_zvec
//==========================================================================================
//
//			to create the direct product of two dsb matrix
//
//(1) Let C=A@B, A=A(i,j), B=B(k,l).then C(i*nB+k,j*mB+l) = A(i,j)*B(k,l)
//==========================================================================================
template<size_t n1_diag,size_t n1_dims,size_t n2_diag,size_t n2_dims,bool is_symmetric=1>//n1,n2 are the dsb dimension
void	create_dsb_direct_product_csridx	
(
	csridx<1>& idx	//the indice object for csr3 matrix
)
{
	size_t constexpr n1_elem=       n1_diag+1ul;  
	size_t constexpr n2_elem=       n2_diag+1ul; 

	auto	push_back_upper	=	[&](const size_t vc1,const size_t ir2)noexcept//vc1 is the col indent value
	{
		size_t 	ic2max	=	ir2<n2_dims-n2_diag?n2_elem:(n2_dims-ir2);
		for(size_t ic2=0;ic2<ic2max;++ic2)
		{
			idx.col.push_back(vc1+ir2+ic2+1);//one based indexing
		}
	};
	auto	push_back_lower	=	[&](const size_t vc1,const size_t ir2)noexcept//vc1 is the col indent value
	{
		size_t 	ic2min	=	ir2<n2_diag?ir2:n2_diag;
		for(size_t ic2=ic2min;ic2>0;--ic2)
		{
			idx.col.push_back(vc1+ir2-ic2+1);//one based indexing
		}
	};

	//row id = ir1*n2_dims+ir2
	for(size_t ir1=0;ir1<n1_dims;++ir1)
	{
		size_t	m_upp,m_low;

		if constexpr(is_symmetric)
		m_low	=	0;
		else
		m_low	=	(ir1<n1_diag)?ir1:n1_diag;//number of lower blocks
		m_upp	=	(ir1<n1_dims-n1_diag)?n1_elem:(n1_dims-ir1);//number of upper blocks
		for(size_t ir2=0;ir2<n2_dims;++ir2)
		{
			//begin a new line
			idx.push_irow();
			//lower off-diagonal blocks
			for(size_t ic1=m_low;ic1>0;--ic1)
			{
				push_back_lower((ir1-ic1)*n2_dims,ir2);
				push_back_upper((ir1-ic1)*n2_dims,ir2);
			}
			//diagonal block
			{
				if constexpr(!is_symmetric)
				push_back_lower(ir1*n2_dims,ir2);
				push_back_upper(ir1*n2_dims,ir2);	
			}
			//upper off-diagonal blocks
			for(size_t ic1=1;ic1<m_upp;++ic1)
			{
				push_back_lower((ir1+ic1)*n2_dims,ir2);
				push_back_upper((ir1+ic1)*n2_dims,ir2);
			}
		}
	}
	idx.push_irow();//end the last line

}//end of create_dsb_direct_product_csridx 

template<size_t n1_diag,size_t n1_dims,size_t n2_diag,size_t n2_dims,class data_t>//n1,n2 are the dsb dimension
void	create_dsb_direct_product_csrval
(
	const double*	dsb1,//described by n1
	const double*   dsb2,//described by n2
	csrval<data_t>& val
)	
{
	size_t constexpr n1_elem=	n1_diag+1ul;
	size_t constexpr n2_elem=	n2_diag+1ul;

	auto	push_back_upper	=	[&](const size_t ir2,const double val1)noexcept
	{
		size_t 	ic2max	=	ir2<n2_dims-n2_diag?n2_elem:(n2_dims-ir2);
		for(size_t ic2=0;ic2<ic2max;++ic2)
		{
			val.push_back(dsb2[ic2+ir2*n2_elem]*val1);
		}
	};
	auto	push_back_lower	=	[&](const size_t ir2,const double val1)noexcept
	{
		size_t 	ic2min	=	ir2<n2_diag?ir2:n2_diag;
		for(size_t ic2=ic2min;ic2>0;--ic2)
		{
			val.push_back(dsb2[ir2*n2_elem-ic2*n2_diag]*val1);
		}
	};

	for(size_t ir1=0;ir1<n1_dims;++ir1)
	{
		size_t	m_upp	=	(ir1<n1_dims-n1_diag)?n1_elem:(n1_dims-ir1);
		for(size_t ir2=0;ir2<n2_dims;++ir2)
		{
			//diagonal block
			size_t ic1=0;
			{
				double	val1	=	dsb1[ir1*n1_elem+ic1];
				push_back_upper(ir2,val1);	
			}
			//off-diagonal block
			for(ic1=1;ic1<m_upp;++ic1)
			{
				double	val1	=	dsb1[ir1*n1_elem+ic1];
				push_back_lower(ir2,val1);
				push_back_upper(ir2,val1);
			}
		}
	}
}//end of create_dsb_direct_product_csrval
	
template<size_t n1_diag,size_t n1_dims,size_t n2_diag,size_t n2_dims>//n1,n2 are the dsb dimension
void	create_dsb_direct_product_csr
(
	const double*	dsb1,	//described by n1
	const double*	dsb2,	//described by n2
	csrmat<double,1>& csr	//to store the result in csr3 format	
)
{
	size_t constexpr n1_elem=	n1_diag+1ul;
	size_t constexpr n2_elem=	n2_diag+1ul;

	auto	push_back_upper	=	[&](const size_t vc1,const size_t ir2,const double val1)noexcept//vc1 is the col indent value
	{
		size_t 	ic2max	=	ir2<n2_dims-n2_diag?n2_elem:(n2_dims-ir2);
		for(size_t ic2=0;ic2<ic2max;++ic2)
		{
			csr.val.push_back(dsb2[ic2+ir2*n2_elem]*val1);
			csr.col.push_back(vc1+ir2+ic2+1);//one based indexing
		}
	};
	auto	push_back_lower	=	[&](const size_t vc1,const size_t ir2,const double val1)noexcept//vc1 is the col indent value
	{
		size_t 	ic2min	=	ir2<n2_diag?ir2:n2_diag;
		for(size_t ic2=ic2min;ic2>0;--ic2)
		{
			csr.val.push_back(dsb2[ir2*n2_elem-ic2*n2_diag]*val1);
			csr.col.push_back(vc1+ir2-ic2+1);//one based indexing
		}
	};

	for(size_t ir1=0;ir1<n1_dims;++ir1)
	{
		size_t	m_upp	=	(ir1<n1_dims-n1_diag)?n1_elem:(n1_dims-ir1);
		for(size_t ir2=0;ir2<n2_dims;++ir2)
		{
			//begin a new line
			csr.push_irow();
			//diagonal block
			size_t ic1=0;
			{
				double	val1	=	dsb1[ir1*n1_elem+ic1];
				push_back_upper((ir1+ic1)*n2_dims,ir2,val1);	
			}
			//off-diagonal block
			for(ic1=1;ic1<m_upp;++ic1)
			{
				double	val1	=	dsb1[ir1*n1_elem+ic1];
				push_back_lower((ir1+ic1)*n2_dims,ir2,val1);
				push_back_upper((ir1+ic1)*n2_dims,ir2,val1);
			}
		}
	}
	csr.push_irow();//end the last line
}//end of create_dsb_direct_product*/

}//end of qpc
