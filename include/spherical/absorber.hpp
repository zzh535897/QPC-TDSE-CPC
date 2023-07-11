#pragma once

template<class dims_t>
struct	absorber_mask
{
	protected:
		double*	data;
		comp_t*	work;
		double	rmin;
		double	rmax;
		size_t	nthd;
		rsrc_t*	rsrc;

	public:
		//constructor
		absorber_mask(absorber_mask const&)=delete;
		absorber_mask(absorber_mask&&)=delete;
		absorber_mask& operator=(absorber_mask const&)=delete;
		absorber_mask& operator=(absorber_mask&&)=delete;
		explicit absorber_mask(size_t _nthd,rsrc_t* _rsrc=global_default_arena_ptr):nthd(_nthd),rsrc(_rsrc)
		{
			if(rsrc)
			{
				rsrc->acquire((void**)&data,sizeof(double)*dims_t::n_rsub     ,align_val_avx3);
				rsrc->acquire((void**)&work,sizeof(comp_t)*dims_t::n_dims*nthd,align_val_avx3);
			}else
			{
				throw	errn_t("null arena parsed in when constructing absorber_mask.\n");
			}
		}
		//destructor
		virtual ~absorber_mask()noexcept
		{
			if(rsrc)
			{
				rsrc->release(data);
				rsrc->release(work);
			}
		}

		//cos^(1/n)
		inline	void	initialize	(const integrator_radi<dims_t>& radi,const double _rmin,const double _rmax,const double index=1./6.)noexcept
		{
			auto	maskfunc	=	[&](double r)
			{
				return r<_rmin?1.0:r>_rmax?0.0:pow(fabs(cos(0.5*pi*(r-_rmin)/(_rmax-_rmin))),index);
			};	
			radi.integrate_0r0(integral_result_radi<dims_t>{data},maskfunc);
		}
		

		inline	void	operator()	(coefficient_view<dims_t>& w_vec,const operator_sc<dims_t>& s_mat)noexcept
		{
			#pragma omp parallel for num_threads(nthd)	
			for(size_t im=0;im<dims_t::m_dims;++im)
			for(size_t il=0;il<dims_t::l_dims;++il)
			{
				auto*t	=	work+omp_get_thread_num()*dims_t::n_dims;
				auto w	=	w_vec(im,il).data;
				intrinsic::symb_mul_vecd<dims_t::n_dims,dims_t::n_elem,0>(data,w,t);
				intrinsic::ltrb_inv_vecd<dims_t::n_dims,dims_t::n_elem>(s_mat.cholesky(),t,t);
				intrinsic::utrb_inv_vecd<dims_t::n_dims,dims_t::n_elem>(s_mat.cholesky(),t,w);
			}
		}

		inline	void	operator()	(comp_t* w_vec,double* s_fac,const size_t n_level)noexcept
		{
			#pragma omp parallel for num_threads(nthd)
			for(size_t i=0ul;i<dims_t::n_leng*n_level;i+=dims_t::n_dims)
			{
				auto*t	=	work+omp_get_thread_num()*dims_t::n_dims;
				auto*w	=	w_vec+i;
				intrinsic::symb_mul_vecd<dims_t::n_dims,dims_t::n_elem,0>(data,w,t);
				intrinsic::ltrb_inv_vecd<dims_t::n_dims,dims_t::n_elem>(s_fac,t,t);
				intrinsic::utrb_inv_vecd<dims_t::n_dims,dims_t::n_elem>(s_fac,t,w);
			}
		}

};//end of absorber_mask