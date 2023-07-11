#pragma once

//require <algorithm/ed.h>

template<class dims_t>
struct	ionization_sph
{
	public:
		using	coef_t	=	coefficient_view<dims_t>;
		using	oper_s	=	operator_sc<dims_t>;
		using	oper_h	=	operator_hc<dims_t>;

		double	data;

		int	measure_ioyd	(const coef_t& c,const oper_s& s,const oper_h& h,const double thres=0.0)
		{
			int info=	0;
			data	=	0.;
			#pragma omp parallel reduction(+:data)
			{
				auto	temp_h	=	std::vector<double>(dims_t::n_rsub);
				auto	temp_s	=	std::vector<double>(dims_t::n_rsub);
				auto	temp_e	=	std::vector<double>(dims_t::n_dims);
				auto	temp_z	=	std::vector<double>(dims_t::n_dims*dims_t::n_dims);
				#pragma omp for collapse(2)
				for(size_t im=0;im<dims_t::m_dims;++im)
				for(size_t il=0;il<dims_t::l_dims;++il)
				{
					double const* h_ptr	=	h.sub(im,il);
					double const* s_ptr	=	s.sub();
					comp_t const* w_ptr	=	c(im,il);
					#pragma GCC ivdep
					for(size_t i=0;i<dims_t::n_rsub;++i)//copy to workspace
					{
						temp_h[i]=h_ptr[i];
						temp_s[i]=s_ptr[i];
					}
					int err         =       impl_ed_simd::gv_dsb<dims_t::n_dims,dims_t::n_diag>(temp_h.data(),temp_s.data(),temp_e.data(),temp_z.data());
                                        if(err)info=1;
					for(size_t ir=0;ir<dims_t::n_dims;++ir)
					{
						if(temp_e[ir]<thres)//bound states criteria
						{
							data+=norm(intrinsic::symb_prj_vecd<dims_t::n_dims,dims_t::n_elem>(s_ptr,temp_z.data()+ir*dims_t::n_dims,w_ptr));
						}
					}
				}
			}
			data=1.-data;
			return	info;
		}//end of measure	
};//end of ionization_sph

template<class dims_t>
struct	ionization_bic
{	
	public:
		using	coef_t	=	coefficient_view<dims_t>;
		using	oper_s	=	operator_sb<dims_t>;

		double	data;

		template<class oper_h>
		int	measure_ioyd	(const coef_t& c,const oper_s& s,const oper_h& h,const double emin=-2.,const size_t neig=dims_t::l_dims*dims_t::n_dims/8ul)
		{
			if(emin>0.0)return -1;
				data	=	0.0;
			int	info	=	0;
			auto 	eig	=	csreig<double>();
			for(size_t im=0;im<dims_t::m_dims;++im)	
			{
				info	+=	gv_ds_csr
				(
					h.create_csrmat_real(im),
					s.create_csrmat_real(im),
					eig,neig,emin,0.0//find all states between (Emin,0)
				);//call csr FEAST to do diagonalization
				comp_t const* tmp=c(im);
				#pragma omp parallel for schedule(dynamic) reduction(+:data)
				for(size_t is=0;is<size_t(eig.fnd);++is)
				{
					data+=norm(s.observe_lsub(eig.vec.data()+is*dims_t::l_dims*dims_t::n_dims,tmp,im));	
				}
			}
			data	=	1.-data;
			return	info;
		}//end of measure_ioyd
};//end of ionization_bic

