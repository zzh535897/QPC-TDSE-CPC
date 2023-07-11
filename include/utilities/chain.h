#pragma once

#include <libraries/support_std.h>

namespace qpc
{
	namespace impl_showclass
	{
		template<class...>
		struct	showclass;
		template<auto...>
		struct	showvalue;
	}//end of impl_showclass

	//node of typelist
	struct	typevoid//null case of typenode
	{
		public:
			using 	this_t	=	typevoid;
			using 	next_t	=	typevoid;

			template<size_t i>
			using	get	=	typevoid;
	};

	template<class this_type,class next_type>
	struct	typenode//general case of typenode
	{
		public:
			template<size_t i>
			static constexpr auto& get_impl()noexcept
			{
				if constexpr(i==0){
					return 	*(this_type*)(0);
				}else{
					return	next_type::template get_impl<i-1ul>();
				}
			}
			using 	this_t	=	this_type;
			using 	next_t	=	next_type;

			template<size_t i>
			using	get	=	std::remove_reference_t<decltype(get_impl<i>())>;

	};
	
	//initializer of typelist
	template<class this_type,class...rest_type>
	struct	typelist_initializer
	{
		private:
			using	next_t	=	typename typelist_initializer<rest_type...>::type_t;
		public:
			using	type_t	=	typenode<this_type,next_t>;
	};
	template<class this_type>
	struct	typelist_initializer<this_type>
	{
		private:
			using	next_t	=	typevoid;
		public:
			using	type_t	=	typenode<this_type,next_t>;	
	};

	//interface of typelist
	template<class...args_t>
	struct	typelist
	{
		public:
			using	create_linked	=	typename typelist_initializer<args_t...>::type_t;

			template<class...type_t>
			using	push_to_tail	=	typelist<args_t...,type_t...>;
			template<class...type_t>
			using	push_to_head	=	typelist<type_t...,args_t...>;	

			template<size_t i>
			using	get		=	typename create_linked::template get<i>;
	};
	template<>
	struct	typelist<>
	{
		public:
			using	create_linked	=	typenode<typevoid,typevoid>;

			template<class...type_t>
			using	push_to_tail	=	typelist<type_t...>;
			template<class...type_t>
			using	push_to_head	=	typelist<type_t...>;				
			template<size_t i>
			using	get		=	typename create_linked::template get<i>;
	};
	
	//type alias 
	template<class...args_t>
	using	linked_typelist		=	typename typelist<args_t...>::create_linked;
	template< auto...args_v>
	using	linked_autolist		=	typename typelist<std::integral_constant<decltype(args_v),args_v>...>::create_linked;
}

namespace qpc
{
	//return the i-th arg in the arglist
	namespace impl_argument
	{
		template<size_t i>
		struct	arg_helper final
		{
			template<class argo_t,class...args_t>
			static constexpr inline auto 	get	(argo_t&& argo,args_t&&...args)noexcept
			{
				static_assert(sizeof...(args_t)+1ul>i);
				if constexpr(i==0ul)
				{
					return 	std::forward<argo_t>(argo);
				}else
				{
					return 	arg_helper<i-1ul>::get(std::forward<args_t>(args)...);
				}
			}
		};//end of arg_helper
	}//end of impl_argument
	//arg interface, use arg<i>(a,b,c) to invoke
	template<size_t i,class...args_t>
	static constexpr inline auto 	arg	(args_t&&...args)noexcept{if constexpr(sizeof...(args_t)>0ul)return impl_argument::arg_helper<i>::get(std::forward<args_t>(args)...);}

	//call function with selected argument
	namespace impl_callwith
	{
		template<class invoker>
		struct	callwith;

		template<template<size_t...>class invoker,size_t...i>
		struct	callwith<invoker<i...>>//require invoker to indicate which variable to use
		{
			template<class func_t,class...args_t>
			constexpr auto operator()(func_t&& func,args_t&&...args)const noexcept
			{
				return 	func((arg<i>(std::forward<args_t>(args)...))...);
			}
		};
	}//end of impl_callwith
	template<class invoker>
	static constexpr auto	callwith	=	impl_callwith::callwith<invoker>{};
	template<size_t n>
	static constexpr auto	callwith_first	=	impl_callwith::callwith<std::make_index_sequence<n>>{};


	namespace impl_sequence
	{
		template<template<size_t...>class x,class indicator>
		struct	combine_with_sequence;
		template<template<size_t...>class x,template<size_t...>class y,size_t...i>
		struct	combine_with_sequence<x,y<i...>>//fill x with i given in y<i...>
		{
			using	type	=	x<i...>;
		};
		
		template<size_t...i>
		struct	sequence
		{
			template<size_t...j>
			static constexpr sequence<i...,j...>	push_to_tail{};
			template<size_t...j>
			static constexpr sequence<j...,i...>	push_to_head{};
		};
		template<size_t i,size_t ni,size_t nf>
		static constexpr auto impl_create_sequence_monotonic()noexcept
		{
			if constexpr(i==nf)
			{
				return	sequence<nf>{};
			}else 
			{
				return 	impl_create_sequence_monotonic<(ni<=nf?i+1ul:i-1ul),ni,nf>().template push_to_head<i>;
			}
		}
		template<size_t i,size_t nf,size_t nd>
		static constexpr auto impl_create_sequence_singleton()noexcept
		{
			static_assert(i<=nf);
			if constexpr(i==nf)
			{
				return 	sequence<(i==nd?1ul:0ul)>{};
			}else
			{
				return 	impl_create_sequence_singleton<i+1ul,nf,nd>().template push_to_head<(i==nd?1ul:0ul)>;
			}
		}

		template<size_t n,size_t i0,size_t...is>
		static constexpr auto impl_create_sequence_with_first()noexcept
		{
			static_assert(n <= 1ul+sizeof...(is));
			if constexpr(n==0)
			{
				return	sequence<>{};
			}else
			{
				return 	impl_create_sequence_with_first<n-1ul,is...>().template push_to_head<i0>;
			}
		}

		template<size_t n,size_t i0,size_t...is>
		static constexpr auto impl_create_sequence_with_final()noexcept
		{
			static_assert(n <= 1ul+sizeof...(is));
			if constexpr(n<1ul+sizeof...(is))
			{
				if constexpr(sizeof...(is)==0)
				return 	sequence<>{};
				else
				return 	impl_create_sequence_with_final<n,is...>();
			}else 
			{
				if constexpr(sizeof...(is)==0)
				return 	sequence<i0>{};
				else
				return 	impl_create_sequence_with_final<n-1,is...>().template push_to_head<i0>;
			}
		}
	}//end of impl_combine
	template<template<size_t...>class x,size_t ni,size_t nf>
	using	combine		=	typename impl_sequence::combine_with_sequence<x,std::remove_cv_t<decltype(impl_sequence::impl_create_sequence_monotonic<ni,ni,nf>())>>::type;	
	template<template<size_t...>class x,size_t ni,size_t nf,size_t nd>
	using	combine_s	=	typename impl_sequence::combine_with_sequence<x,std::remove_cv_t<decltype(impl_sequence::impl_create_sequence_singleton<ni,nf,nd>())>>::type;

	template<template<size_t...>class x,size_t n,size_t...i>
	using	get_first	=	typename impl_sequence::combine_with_sequence<x,std::remove_cv_t<decltype(impl_sequence::impl_create_sequence_with_first<n,i...>())>>::type;
	template<template<size_t...>class x,size_t n,size_t...i>
	using	get_final	=	typename impl_sequence::combine_with_sequence<x,std::remove_cv_t<decltype(impl_sequence::impl_create_sequence_with_final<n,i...>())>>::type;
}

namespace qpc
{
	struct	select_fail;//do not define, to let the compiler find the error

	template<auto condition,class succ_value,class fail_value=void>
	struct	select_cond//the class to specify a condition and its resultant value
	{
		template<auto criterion>
		using	type_t	=	std::conditional_t<criterion==condition,succ_value,fail_value>;
	};//end of select_cond

	template<class...args_t>
	struct	select_cond_return;
	template<>
	struct	select_cond_return<>
	{
		using	type_t	=	select_fail;//if all types are void, then return select_fail, which is not well-defined.
	};
	template<class anyt,class...args_t>
	struct	select_cond_return<anyt,args_t...>
	{
		using	type_t	=	anyt;//once the first type is not void, return immediately
	};
	template<class...args_t>
	struct	select_cond_return<void,args_t...>
	{
		using	type_t	=	typename select_cond_return<args_t...>::type_t;//if the first type if void, go next
	};//end of select_cond_return

	template<class type_t>
	struct	is_select_cond final:public std::integral_constant<bool,false>{};
	template<auto condition,class succ_value,class fail_value>
	struct	is_select_cond<select_cond<condition,succ_value,fail_value>> final:public std::integral_constant<bool,true>{};
	template<class type_t>
	static constexpr bool is_select_cond_v	=	is_select_cond<type_t>::value;

	template<auto criterion,class...cond_t>
	struct	select//the user class
	{
		static_assert((is_select_cond_v<cond_t>&&...&&true),"cond_t must be qpc::select_cond to specify a consecutive.");

		using	type_t	=	typename select_cond_return<typename cond_t::template type_t<criterion>...>::type_t;
	};//end of select
	template<auto criterion,class...cond_t>
	using	select_t	=	typename select<criterion,cond_t...>::type_t;
}//end of qpc

namespace qpc
{
	template<class...>
	struct	select_fail_temp;//do not define, to let the compiler find the error

	template<auto condition,template<class...>class succ_value>
	struct	select_cond_temp //the class to specify a condition and its resultant value
	{
		struct	succ_value_wrapper
		{
			template<class...args_t> 
			using type_t	=	succ_value<args_t...>;	
		};
		template<auto criterion>
		using	type_t	=	std::conditional_t
		<
			criterion==condition,
			succ_value_wrapper,void
		>;
	};//end of select_cond_temp

	template<class...args_t>
	struct	select_cond_temp_return;
	template<>
	struct	select_cond_temp_return<>
	{
		template<class...args_t>
		using	type_t	=	select_fail_temp<args_t...>;//if all types are void, then return select_fail_temp, which is not well-defined.
	};
	template<class anyt,class...args_t>
	struct	select_cond_temp_return<anyt,args_t...>
	{
		using	type_t	=	anyt;//once the first type is not void, return immediately
	};
	template<class...args_t>
	struct	select_cond_temp_return<void,args_t...>
	{
		using	type_t	=	typename select_cond_return<args_t...>::type_t;//if the first type if void, go next
	};//end of select_cond_temp_return

	template<class type_t>
	struct	is_select_cond_temp final:public std::integral_constant<bool,false>{};
	template<auto condition,template<class...>class succ_value>
	struct	is_select_cond_temp<select_cond_temp<condition,succ_value>> final:public std::integral_constant<bool,true>{};
	template<class type_t>
	static constexpr bool is_select_cond_temp_v	= is_select_cond_temp<type_t>::value;

	template<auto criterion,class...cond_t>
	struct	select_temp//the user class
	{
		static_assert((is_select_cond_temp_v<cond_t>&&...&&true),"cond_t must be qpc::select_cond_temp to specify a consecutive.");

		template<class...args_t>
		using	type_t	=	typename select_cond_temp_return<typename cond_t::template type_t<criterion>...>::type_t::template type_t<args_t...>;
	};//end of select_temp
	
}//end of qpc
