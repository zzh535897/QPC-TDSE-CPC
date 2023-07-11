#pragma once

//=================================================================================//
//
//	description :
//	(1) EP laser in X-Y direction only couples states with the same (l-m)%2, or
//	equivalently il%2. We term il%2==0 as "even" and il%2==1 as "odd". 
//	Reordering of the eigenvectors and eigenvalues is done by converter! 
//	
//	(2) Details.
//	- If Nm is even, then two kind of transforms exist: "odd" ones and "even" ones. 
//	- There are Ne=Nm*[(Nl+1)/2] "even" and No=Nm*[Nl/2] "odd".
//	- Index 0:Ne-1 stores "odd" and Ne:Ne+No-1 stores "even".
//
// 	- If Nm is odd, then three kinds of transforms exist: "odd","even" and "zero".
// 	"zero" refers to those with zero eigenvalues and they actually does not take 
// 	part in the interaction.
// 	- There are Nl "zero". Ne=No=[(Nm-1)/2]*Nl.
//	- Index 0:Ne-1 stores "even", Ne:NeNo-1 stores "odd", the rest stores "zero".
//	
//
//=================================================================================//
namespace converter
{

	template<class dims_t>
	int	catagorize	(double* eig, comp_t* prj)
	{
		auto	eig_odd	=	std::vector<double>();
		auto	eig_eve	=	std::vector<double>();
		auto	eig_zer	=	std::vector<double>();
		auto	prj_zer	=	std::vector<comp_t>();
		auto	prj_odd	=	std::vector<comp_t>();
		auto	prj_eve	=	std::vector<comp_t>();

		for(size_t i=0;i<dims_t::m_dims*dims_t::l_dims;++i)
		{
			comp_t*	_prj = prj + i*dims_t::m_dims*dims_t::l_dims;
			if constexpr(dims_t::m_dims%2ul==1ul)//odd Nm could have "zero"
			{
				if(fabs(eig[i])<1e-8)
				{
					eig_zer.push_back(0.0);
					for(size_t j=0;j<dims_t::m_dims*dims_t::l_dims;++j)
					prj_zer.push_back(_prj[j]);
					continue;
				}
			}
			bool is_eve_zero = true;
			bool is_odd_zero = true;
			for(size_t jm=0;jm<dims_t::m_dims;++jm)
			for(size_t jl=0;jl<dims_t::l_dims;++jl)
			{
				size_t j= jm*dims_t::l_dims+jl;
				if(norm(_prj[j])>1e-16)
				{
					if(jl%2==0) is_eve_zero = false;
					if(jl%2==1) is_odd_zero = false;
					continue;
				}
			}
			if(!is_eve_zero)
			{
				if(!is_odd_zero)return -1;
				eig_eve.push_back(eig[i]);
				for(size_t j=0;j<dims_t::m_dims*dims_t::l_dims;++j)
				prj_eve.push_back(_prj[j]);
			}else 
			if(!is_odd_zero)
			{
				eig_odd.push_back(eig[i]);
				for(size_t j=0;j<dims_t::m_dims*dims_t::l_dims;++j)
				prj_odd.push_back(_prj[j]);		
			}
		}
		
		if(eig_zer.size()!=dims_t::n_zero)return -2;
		if(eig_eve.size()!=dims_t::n_even)return -3;
		if(eig_odd.size()!=dims_t::n_odds)return -3;

		for(size_t i=0;i<dims_t::n_even;++i)//eve first
		{
			eig[i] = eig_eve[i];
		}eig+= dims_t::n_even;
		for(size_t i=0;i<dims_t::n_zero;++i)//zer middle
		{
			eig[i] = eig_zer[i];
		}eig+= dims_t::n_zero;
		for(size_t i=0;i<dims_t::n_odds;++i)//odd last
		{	
			eig[i] = eig_odd[i];
		}	
		
		for(size_t i=0;i<dims_t::n_even*dims_t::m_dims*dims_t::l_dims;++i)//eve first
		{
			prj[i] = prj_eve[i];
		}prj+= dims_t::n_even*dims_t::m_dims*dims_t::l_dims;
		for(size_t i=0;i<dims_t::n_zero*dims_t::m_dims*dims_t::l_dims;++i)//zer middle
		{
			prj[i] = prj_zer[i];
		}prj+= dims_t::n_zero*dims_t::m_dims*dims_t::l_dims;	
		for(size_t i=0;i<dims_t::n_odds*dims_t::m_dims*dims_t::l_dims;++i)//odd last
		{
			prj[i] = prj_odd[i];
		}
		return 	0;
	}//end of catagorize
}//end of namespace
