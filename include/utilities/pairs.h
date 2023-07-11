#pragma once

#include <utilities/string.h>//require c++17


namespace qpc
{
	template<class name,const size_t index>
	struct	nipair{};//name-index pair
	
	template<class...pair>
	struct	nilist;
	template<class...name,size_t...index>
	struct	nilist<nipair<name,index>...>
	{
		static constexpr size_t	size	=	sizeof...(name);//number of the pairs in this list

		static_assert(((index<size)&&...&&true));
		
		template<char...chars>//convert a string literal to index. if not recorded, return size, otherwise, give the recorded index.
		static constexpr size_t	indx	(tstring<chars...> x)noexcept
		{
			return 	size-(((x==name{})?(size-index):0ul)+...+0ul);
		}
	};//name-index list
}//end of qpc
