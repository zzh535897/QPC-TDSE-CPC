#pragma once
#include <libraries/support_std.h>

//====================================================================================================
//
//                      This file is part of QPC Header Only Library
//
//====================================================================================================

//====================================================================================================
//
//				Richardson Extrapolation
//
//	References:
//	[1] L. F. Richardson, Philosophical Transactions of the Royal Society A. 210 (459–470): 307–357
//
//	Description:
//	(1) A detailed principle can be found on https://en.wikipedia.org/wiki/Richardson_extrapolation	
//
//Updated by: Zhao-Han Zhang(张兆涵)  Dec. 21th, 2022
//Copyright © 2022 Zhao-Han Zhang
//====================================================================================================

namespace qpc
{
	//f xi,xf,yi,yf
	template<class odef_t,size_t n_imax=16>
	struct	richardson
	{
		using	data_t	=	typename odef_t::data_t;
		using	cord_t	=	typename odef_t::cord_t;
	
		static constexpr size_t imax	=	n_imax;

		odef_t	fun;
		cord_t	tol;

		data_t	hstep;
		size_t	nstep;

		cord_t	yi;
		cord_t	yf;
		data_t	xi;
		data_t	xf;

		cord_t*	work;

		//memory allocation
		template<class...args_t>
		explicit richardson(args_t&&...args):fun(std::forward<args_t>(args)...)
		{
			work=new cord_t[imax*3ul];
		}
		virtual ~richardson()noexcept
		{
			delete[] work;
		}
		//end of memory allocation

		inline	void	init	()noexcept
		{
			hstep=xf-xi;
			nstep=1;
		}

		inline	void	half	()noexcept
		{
			nstep*=2;
			hstep/=2;
		}

		inline	void	iter	(const size_t j,const cord_t& _next,const cord_t& _this,cord_t& _retn)const noexcept
		{
			data_t 	pow4i	=	std::pow(4.0,j+1);//4^(j+1)
			for(size_t k=0;k<_retn.size();++k)
			{
				_retn[k]=	(pow4i*_next[k]-_this[k])/(pow4i-data_t(1));//note that for j>13, 4^(j+1)>1e16, _this is lost in truncation error.
			}
		}

		inline	void	eval	(cord_t& _retn)const noexcept
		{//midpoint rule
			_retn=yi;
			cord_t	fini,fmid,ymid;
			data_t	hhalf=hstep/2.0;
			for(size_t s=0;s<nstep;++s)
			{
				data_t 	xinit	=	xi+ s*hstep;
				fun(xinit,_retn,fini);
				for(size_t k=0;k<_retn.size();++k)
				ymid[k]=_retn[k]+hhalf*fini[k];
				fun(xinit+hhalf,ymid,fmid);
				for(size_t k=0;k<_retn.size();++k)
				_retn[k]+=hstep*fmid[k];		
			}
		}

		inline	bool	crit	(const cord_t& _next,const cord_t& _this)const noexcept
		{
			for(size_t k=0;k<_next.size();++k)
			{	
				if(fabs((_next[k]-_this[k])/_this[k])>tol[k])return 0;
			}
			return 	1;
		}
	};//end of richardson

	//work size >= 3*jmax
	template<class func_t,size_t n_imax>
	int 	richardson_ex		(richardson<func_t,n_imax>& f)noexcept
	{
		using	cord_t	=	typename richardson<func_t>::cord_t;
		size_t	i;
		cord_t*	this_line	=	f.work;
		cord_t*	next_line	=	f.work+f.imax;
		f.init();
		f.eval(*this_line);
		for(i=0;i<f.imax-1;++i)
		{
			f.half();
			f.eval(*next_line);
			for(size_t j=0;j<=i;++j)
			{
				f.iter(j,next_line[j],this_line[j],next_line[j+1]);
			}
			//if within tolerence, stop
			if(f.crit(next_line[i+1],this_line[i]))
			{
				f.yf=next_line[i+1];
				break;
			}
			//move to next line
			this_line	=	next_line;
			next_line	=	f.work+((i+2)%3)*f.imax;
		}
		if(i>=f.imax-1ul)
		{
			f.yf=this_line[i];
			return -1;
		}else 	return 0;
	}//end of richardson_ex

	template<class func_t,size_t n_imax>
	int	riex	
	(
		richardson<func_t,n_imax>& f, 
		typename richardson<func_t,n_imax>::data_t const& init_x,
		typename richardson<func_t,n_imax>::cord_t const& init_y,
		typename richardson<func_t,n_imax>::data_t const& last_x,
		typename richardson<func_t,n_imax>::cord_t &	  last_y,	
		typename richardson<func_t,n_imax>::data_t const& step
	)noexcept
	{
		int info =0 ;
		f.xf = init_x;
		f.yf = init_y;
		while(f.xf + step < last_x)
		{
			f.xi = f.xf;
			f.xf +=step;
			f.yi = f.yf;
			info += richardson_ex(f);
		}	f.xi = f.xf;
			f.xf = last_x;
			f.yi = f.yf;
			info += richardson_ex(f);	
			last_y = f.yf;
		return info;
	}//end of riex
}//end of qpc
