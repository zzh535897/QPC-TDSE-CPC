#pragma once
#include <libraries/support_std.h>

namespace qpc
{
	template<char...chars>
	using	tstring	=	std::integer_sequence<char,chars...>;//represent a 'string' using a template argument sequence of 'char'

	template <class T, T... chars>
	constexpr tstring<chars...> operator""_tstr()noexcept { return { }; }//add partial specialization to tstring


	//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
	//						string comparison
	//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
	template<class lhsc_t,class rhsc_t,class=void>
	struct	tstring_compare//sfinae
	{
		static constexpr bool value=false;
	};
	template<char...lhsc,char...rhsc>
	struct	tstring_compare<tstring<lhsc...>,tstring<rhsc...>,std::enable_if_t<(sizeof...(lhsc))==(sizeof...(rhsc))>>//if not, sfinae will be triggered
	{
		static constexpr bool value=((lhsc==rhsc)&&...&&true);
	};
	

	template<char...lhsc,char...rhsc>
	constexpr bool	operator== (tstring<lhsc...>,tstring<rhsc...>)noexcept//call compare to implement operator==
	{
		return 	tstring_compare<tstring<lhsc...>,tstring<rhsc...>>::value;
	}
	template<char...lhsc,char...rhsc>
	constexpr bool	operator!= (tstring<lhsc...>,tstring<rhsc...>)noexcept//call compare to implement operator!=
	{
		return 	!tstring_compare<tstring<lhsc...>,tstring<rhsc...>>::value;
	}
	//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
	//					string manipulation
	//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
	template<char...lhsc,char...rhsc>
	constexpr auto	operator+ (tstring<lhsc...>,tstring<rhsc...>)noexcept
	{
		return 	tstring<lhsc...,rhsc...>{};
	}
	

	//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
	//					string conversion
	//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
	template<class>
	struct	istring;
	template <char...chars>
	struct 	istring<tstring<chars...>> //convert to c-style string.
	{
		static_assert((sizeof...(chars))<255);
        	static constexpr char str[sizeof...(chars)+1] = {chars...,'\0'};
	};

}//end of qpc
