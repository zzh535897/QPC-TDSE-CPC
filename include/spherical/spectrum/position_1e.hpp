#pragma once

//#include <functions/basis_bsplines.h>
//#include <functions/basis_legendre.h>
//#include <functions/basis_coulombf.h>
//#include <functions/spheroidal.h>

template<class dims_t>
struct	spectrum_x
{	
	public:
		using	type_of_disp_r	=	typename bss::bsplines<dims_t::nr,dims_t::mr,dims_t::b_lfbd,dims_t::b_rtbd>::disp;
		using	type_of_disp_a	=	typename bss::spherical_harmonics::disp<dims_t::n_mmax,dims_t::l_dims>;

		type_of_disp_r  disp_r;//displayer of b-splines functions, see basis_bsplines.h
		type_of_disp_a  disp_a;//displayer of spherical harmonics, see basis_legendre.h

		std::vector<comp_t> buff_w;	//store the results after calling 'accumulate'

		inline  size_t  _nr()const noexcept{return disp_r.m;}	//number of radial    points to display	(r or xi)
		inline  size_t  _nt()const noexcept{return disp_a.nth;}	//number of angular   points to display (th or eta)
		inline  size_t  _np()const noexcept{return disp_a.nph;}	//number of azimuthal points to display (phi)
		inline  size_t          size()const noexcept{return _nr()*_nt()*_np();}			//the size of array (to be saved in a file)
		inline  double const*   dptr()const noexcept{return (double*)(void*)buff_w.data();}	//the pointer to the array (to be saved in a file)
	
		//to initialize the value of read-out axis and of the basis function. XXX only linear sequence are implemented. XXX
		void	initialize
		(
			const integrator_radi<dims_t>& radi,
			const double ri,const double rf,const size_t nr, //r		or	xi
			const double ai,const double af,const size_t na, //theta	or	arccos(eta)
			const double pi,const double pf,const size_t np  //phi		or	phi
		)
		{
			double  dr      =       nr>1?(rf-ri)/(nr-1):0.0;//to reach both two endians
			double  da      =       na>1?(af-ai)/(na-1):0.0;//to reach both two endians
			double  dp      =       np>1?(pf-pi)/(np-1):0.0;//to reach both two endians
			disp_r.initialize(radi.base,[&](const size_t i){return ri+i*dr;},nr);
			disp_a.initialize(          [&](const size_t i){return min(ai+i*da,af);},	//not to exceed af, which may cause fatal error
						    [&](const size_t i){return min(pi+i*dp,pf);},na,np);//not to exceed pf, which may cause fatal error
			
			buff_w.resize(nr*na*np);
		}//end of initialize
	
		//"accumulate" computes the exact value of wave function on given axis points
		//note: 
		//for B(r)Ylm(th,ph)*G(r), G(r)=1/r is set as 1/(r+eps) to avoid singularity 
		//for B(x)Ylm(et,ph)*G(r), G(r)=sqrt{(x-1)/(x+1)} or 1

		template<size_t i_type=dims_t::i_type>
		void    accumulate      (const comp_t* w,size_t nthreads=1)noexcept
		{
			auto	term	=	std::vector<double>(disp_r.m);//global term
			if constexpr(i_type==0)//spherical
			{
				#pragma GCC ivdep
				for(size_t ir=0;ir<disp_r.m;++ir)
				{
					double r=disp_r.x[ir];
					term[ir]=1./(r+1e-16);//eps =1e-50
				}
			}
			if constexpr(i_type==1)//prolate spheroidal
			{
				#pragma GCC ivdep
				for(size_t ix=0;ix<disp_r.m;++ix)
				{
					double x=disp_r.x[ix];
					term[ix]=sqrt((x-1.0)/(x+1.0));
				}
			}	
			#pragma omp parallel num_threads(nthreads)
			{
				#pragma omp for
				for(size_t i=0;i<size();++i)
				{
					buff_w[i]=zero<comp_t>;
				}
				#pragma omp for collapse(2)
				for(size_t iph=0;iph<_np();++iph)
				for(size_t ith=0;ith<_nt();++ith)
				{
					comp_t* 	_buff   =       buff_w.data()+(iph*_nt()+ith)*_nr();
					comp_t const* 	_wave;
					comp_t		_coef;
					if constexpr(i_type==0)//spherical
					{
						for(size_t im=0;im<dims_t::m_dims;++im)
						for(size_t il=0;il<dims_t::l_dims;++il)//summing up all r-subspaces
						{
							long_t  m,l;dims_t::in(im,il,m,l);//get l,m
							_coef	=	disp_a(m,l,iph,ith);//cal Ylm(th,ph)
							_wave	=	w+dims_t::n_dims*dims_t::l_dims*im+dims_t::n_dims*il;//the pointer to (im,il)-th r-subspace
							disp_r.accumulate(_wave,_buff,_coef);//do _buff+=R(:)*Ylm(th,ph)
						}
						for(size_t ir=0;ir<disp_r.m;++ir)
						{
							_buff[ir]=	_buff[ir]*term[ir];//do *= 1/r
						}
					}//end
					if constexpr(i_type==1)//prolate spheroidal
					{
						for(size_t im=0;im<dims_t::m_dims;++im)
						for(size_t il=0;il<dims_t::l_dims;++il)//summing up all x-subspaces
						{
							long_t m,l;dims_t::in(im,il,m,l);
	                                              	_coef   =       disp_a(m,l,iph,ith);//calc Ylm(arccos(et),ph)
							_wave	=	w+dims_t::n_dims*dims_t::l_dims*im+dims_t::n_dims*il;//the pointer to (im,il)-th r-subspace
							if(dims_t::od_m(im)==0)//even m
							disp_r.accumulate(_wave,_buff,[&](size_t ik){return _coef;});//do _buff+=R(:)*Ylm(arccos(et),ph)
							else//odd m
							disp_r.accumulate(_wave,_buff,[&](size_t ik){return _coef*term[ik];});//do _buff+=G(:)*R(:)*Ylm(arccos(et),ph)
						}
					}//end
				}
			}//end of omp*/	
		}//end of accumulate
};//end of spectrum_x



