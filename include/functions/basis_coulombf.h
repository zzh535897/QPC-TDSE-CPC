#pragma once

#include <libraries/support_std.h>

#include <resources/arena.h>
#include <utilities/error.h>

#include <functions/coulomb.h>
//====================================================================================================
//
//                      This file is part of QPC Header Only Library
//
//====================================================================================================
//====================================================================================================
//
//							Wrapper Class for Coulomb-Wave Function 
//	References:
//	(1) See functions/coulomb.h for details.
//
//	Description:
//	(1) See the text later.
//
//Updated by: Zhao-Han Zhang(张兆涵)  Dec. 31th, 2022
//Copyright © 2022 Zhao-Han Zhang
//====================================================================================================

namespace qpc{
namespace bss{

	// lmin,lmin+1,...lmin+kmax
	template<class data_t>
	struct	coulomb final
	{
		using	rsrc_t	=	arena;

		data_t*	f;		//to be filled with F
		data_t*	g;		//to be filled with G
		data_t*	df;		//to be filled with dF/dx
		data_t*	dg;		//to be filled with dG/dx

		data_t	lmin;		//F,G,dF,dG with order lmin,lmin+1,...,lmin+kmax will be evaluated
		size_t 	kmax;
		size_t	step;		//the number of points on rho axis
		rsrc_t*	rsrc;

		//access
		inline	data_t	_fc	(const size_t il,const size_t ix)const noexcept{return f [il*step+ix];}
		inline	data_t	_gc	(const size_t il,const size_t ix)const noexcept{return g [il*step+ix];}
		inline	data_t	_fp	(const size_t il,const size_t ix)const noexcept{return df[il*step+ix];}
		inline	data_t	_gp	(const size_t il,const size_t ix)const noexcept{return dg[il*step+ix];}
	
		inline	data_t*	_fc	(const size_t il)const noexcept{return f +il*step;}
		inline	data_t*	_gc	(const size_t il)const noexcept{return g +il*step;}
		inline	data_t*	_fp	(const size_t il)const noexcept{return df+il*step;}
		inline	data_t*	_gp	(const size_t il)const noexcept{return dg+il*step;}

		inline	size_t	leng	()const noexcept{return step*(kmax+1ul);}

		coulomb(const data_t _lmin,const size_t _kmax,const size_t _step,rsrc_t* _rsrc=global_default_arena_ptr)
		{
			lmin	=	_lmin;
			kmax	=	_kmax;
			step	=	_step;
			rsrc	=	_rsrc;
			if(rsrc!=0)
			{
				rsrc	->	acquire((void**)&f,sizeof(data_t)*_step*(_kmax+1),align_val_avx3);
				rsrc	->	acquire((void**)&df,sizeof(data_t)*_step*(_kmax+1),align_val_avx3);
				rsrc	->	acquire((void**)&g,sizeof(data_t)*_step*(_kmax+1),align_val_avx3);
				rsrc	->	acquire((void**)&dg,sizeof(data_t)*_step*(_kmax+1),align_val_avx3);
			}else
			{
				throw	runtime_error_t<std::string,void>{"invalid arena parsed in when constructing coulomb."};
			}
		}//end of constructor
		
		~coulomb()noexcept
		{
			if(rsrc!=0)
			{
				rsrc	->	release(f);
				rsrc	->	release(df);
				rsrc	->	release(g);
				rsrc	->	release(dg);
			}	
		}//end of destructor

		//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		//for z>0 and k small, non-zero error codes returned by the routine coulombfg are usually from the
		//part evaluating the absolute normalization of G,G', not F,F'. they are often non-fatal ones, 
		//for example, G,G' are not accurate enough to reach the default tolerence 1e-15, but they are still
		//very accurate. thus this function will not stop immediately when a non-zero error code occurs.
		//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

		//prepare value
		template<class para_t=double>
		int	evaluate	(para_t eta,para_t* rho) //rho should be an array, length=step
		{
			int info=0;
			for(size_t i=0;i<step;++i)
			{
				data_t*	__fc	=	f+i*(kmax+1);
				data_t*	__gc	=	g+i*(kmax+1);
				data_t*	__fp	=	df+i*(kmax+1);
				data_t*	__gp	=	dg+i*(kmax+1);
				if(rho[i]>0.)
				{
					info	+=	qpc::coulombfg<data_t>(lmin,eta,rho[i],__fc,__fp,__gc,__gp,kmax);//better
				}else//fill by zero at singular point
				{
					for(size_t k=0;k<=kmax;++k)
					{
						__fc[k]=data_t(0);
						__fp[k]=data_t(0);
						__gc[k]=data_t(0);
						__gp[k]=data_t(0);
					}
				}
			}
			if(step>1)transpose();
			return	info;
		}//end of evaluate

		template<class para_t=double>
		int	evaluate	(para_t z,para_t k,para_t* x) //x should be an array storing values of r, length=step
		{
			int 	info=0;
			data_t 	eta=-z/k;
			for(size_t i=0;i<step;++i)
			{
				data_t*	__fc	=	f+i*(kmax+1);
				data_t*	__gc	=	g+i*(kmax+1);
				data_t*	__fp	=	df+i*(kmax+1);
				data_t*	__gp	=	dg+i*(kmax+1);
				data_t	rho	=	k*x[i];
				if(rho>data_t(0))
				{
					info	+=	qpc::coulombfg<data_t>(lmin,eta,rho,__fc,__fp,__gc,__gp,kmax);//better
				}else//fill by zero at singular point (their values are not used)
				{
					for(size_t j=0;j<=kmax;++j)
					{
						__fc[j]=data_t(0);
						__fp[j]=data_t(0);
						__gc[j]=data_t(0);
						__gp[j]=data_t(0);
					}
				}
			}
			if(step>1)transpose();
			return info;
		}//end of evaluate
		
		//transpose the data, such that rho is the neighbouring variable rather than l.
		void	transpose	()noexcept
		{
			data_t*	tmp;
			rsrc->acquire((void**)&tmp,sizeof(data_t)*step*(kmax+1),align_val_avx3);
			for(size_t i=0;i< step;++i)
			for(size_t k=0;k<=kmax;++k)
			tmp[i+k*step]=f[i*(kmax+1)+k];
			for(size_t k=0;k<=kmax;++k)
			for(size_t i=0;i< step;++i)
			f[i+k*step]=tmp[i+k*step];

			for(size_t i=0;i< step;++i)
			for(size_t k=0;k<=kmax;++k)
			tmp[i+k*step]=g[i*(kmax+1)+k];
			for(size_t k=0;k<=kmax;++k)
			for(size_t i=0;i< step;++i)
			g[i+k*step]=tmp[i+k*step];

			for(size_t i=0;i< step;++i)
			for(size_t k=0;k<=kmax;++k)
			tmp[i+k*step]=df[i*(kmax+1)+k];
			for(size_t k=0;k<=kmax;++k)
			for(size_t i=0;i< step;++i)
			df[i+k*step]=tmp[i+k*step];

			for(size_t i=0;i< step;++i)
			for(size_t k=0;k<=kmax;++k)
			tmp[i+k*step]=dg[i*(kmax+1)+k];
			for(size_t k=0;k<=kmax;++k)
			for(size_t i=0;i< step;++i)
			dg[i+k*step]=tmp[i+k*step];
	
			rsrc->release(tmp);
		}//end of transpose

	};//end of coulomb
}//end of bss
}//end of qpc
