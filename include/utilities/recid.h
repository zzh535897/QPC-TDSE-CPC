#pragma once

#include <utilities/chain.h>

namespace qpc
{
	template<size_t n>
	static constexpr auto	_ul	=	std::integral_constant<size_t,n>{};

	//---------------------------------------------------------------------------
	//
	//	recker<n_this,next_t> is the class implementing recursive indexing
	//
	//	(1) n_this is the product n(i+1)*n(i+2)*...n(N-1) at node i:
	//	(2) next_t is the node i+1. when i=N-1, it is void.
	//
	//---------------------------------------------------------------------------

	template<size_t n,class next_node>
	struct	recker//middle node (node 0,1,...,N-2)
	{
		using	next_t	=	next_node;
		static constexpr size_t n_this=n;

		template<class arg0_t,class...args_t>
		static constexpr auto 	in	(const arg0_t& i0,const args_t&...is)noexcept
		{
			if constexpr(sizeof...(is)>0ul)
			{
				return 	i0*n_this+next_t::in(is...);
			}else
			{
				return 	i0*n_this;
			}
		}
	};//end of recker
	
	template<size_t n>
	struct	recker<n,void>//last node (node N-1)
	{
		using	next_t	=	void;
		static constexpr size_t n_this=n;

		template<class arg0_t,class...args_t>
                static constexpr auto 	in      (const arg0_t& i0,const args_t&...is)noexcept
                {
			static_assert(sizeof...(is)==0,"too more arguments are parsed in.");
			return 	i0*n_this;
		}
	};//end of recker(last)


	//-------------------------------------------------------------------------------------------
	//
	//	recker_assembler<n...>::type gives the recker for recursive indexing with n...
	//
	//-------------------------------------------------------------------------------------------
	template<size_t...>
	struct	recker_assembler;//just declaration

	template<size_t n0,size_t...ns>
	struct	recker_assembler<n0,ns...>
	{
		using	type	=	recker<(ns*...*1ul),typename recker_assembler<ns...>::type>;
	};//end of recker_assembler

	template<>
	struct	recker_assembler<>
	{
		using	type	=	void;
	};//end of recker_assembler(last)
	//-------------------------------------------------------------------------------------------
	//
	//Usage:	
	//	recidx<n...>
	//
	//	recvec<data_t,recidx<n...>>
	//
	//-------------------------------------------------------------------------------------------
	template<size_t ndim0,size_t...ndims>
	struct	recidx
	{
		using	indx_t	=	std::array<size_t,sizeof...(ndims)+1ul>;

		using	prod_t	=	typename recker_assembler<ndim0,ndims...>::type;

		static constexpr size_t	n_rank	=	sizeof...(ndims)+1ul;

		static constexpr size_t	n_leng	=	(ndims*...*ndim0);

		static constexpr indx_t	n_dims	=	{ndim0,ndims...};

		template<class...args_t>
		static constexpr auto	in	(const args_t&...args)noexcept
		{
			if constexpr(sizeof...(args_t)>0)
			{
				return 	prod_t::in(args...);
			}else
			{
				return 	0ul;
			}
		}
		
		template<size_t n>
		using	drop_t	=	get_final<recidx,n_rank-n,ndim0,ndims...>;
	};//end of recidx

	template<class data_t,class dims_t>
	struct	recvec
	{
		data_t*	data;

		static constexpr dims_t dim{};

		constexpr inline data_t&	operator[](const size_t i)noexcept	{return data[i];}
		constexpr inline data_t const&	operator[](const size_t i)const noexcept{return data[i];}

		constexpr operator data_t* ()const noexcept{return data;}//implicit conversion

		template<class...indx_t>
		constexpr inline auto&		val	(const indx_t&...i)const noexcept//recursive structure
		{
			return	data[dims_t::in(i...)];
		}
		template<class...indx_t>
		constexpr inline auto		sub	(const indx_t&...i)const noexcept//recursive structure
		{
			if constexpr(sizeof...(i)>0ul)
			return 	recvec<data_t,typename dims_t::template drop_t<sizeof...(i)>>{data+dims_t::in(i...)};
			else//in almost all cases, calling sub with no arguments is attempting the pointer rather than myself
			return 	data;
		}
		template<class...indx_t>
		constexpr inline decltype(auto) operator()(const indx_t&...i)const noexcept//recursive structure
		{
			if constexpr(sizeof...(i)==dims_t::n_rank)
			{
				return 	data[dims_t::in(i...)];
			}else if constexpr(sizeof...(i)>0ul)
			{//in almost all cases, calling sub with incomplete arguments is attempting to access the pointer rather than value
				return 	recvec<data_t,typename dims_t::template drop_t<sizeof...(i)>>{data+dims_t::in(i...)};
			}else
			{//in almost all cases, calling sub with zero arguments is attempting access the pointer rather than a recvec object
				return 	data;
			}
		}

		template<class temp_t>
		inline	auto const&	operator*=(const temp_t& temp)const noexcept
		{
			#pragma omp for simd
			for(size_t i=0;i<dims_t::n_leng;++i)data[i]*=temp;
			return	*this;
		}
		template<class temp_t>
		inline	auto const&	operator+=(const temp_t& temp)const noexcept
		{
			#pragma omp for simd
			for(size_t i=0;i<dims_t::n_leng;++i)data[i]+=temp;
			return	*this;
		}
		
		inline	auto const&	operator*=(recvec& rhs)const noexcept
		{
			#pragma omp for simd
			for(size_t i=0;i<dims_t::n_leng;++i)data[i]*=rhs.data[i];
			return	*this;	
		}
		inline	auto const&	operator+=(recvec& rhs)const noexcept
		{
			#pragma omp for simd
			for(size_t i=0;i<dims_t::n_leng;++i)data[i]+=rhs.data[i];
			return	*this;	
		}
	};//end of recvec

}//end of qpc
