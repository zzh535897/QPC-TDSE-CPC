#pragma once
//c++ io libraries
#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>

//c++ memory management
#include <new>
#include <memory>

//c++ hardware interference
#include <ctime>
#include <chrono>
#include <sys/time.h>

//c++ meta programming helper
#include <utility>
#include <type_traits>

//c++ data structure
#include <cstring>
#include <string_view>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <array>
#include <tuple>
#include <variant>
#include <optional>
#include <bitset>
#include <queue>
#include <functional>

//c++ math libraries
#include <cmath>
#include <random>
#include <complex>
#include <limits>

//c++ algorithm libraries
#include <algorithm>

template<class T>
using	rm_cvr_t	=	std::remove_cv_t<std::remove_reference_t<T>>; //for C++17

template<class T,class R>
static constexpr bool	compare_v	=	std::is_same_v<rm_cvr_t<T>,R>;

namespace qpc
{
	using std::to_string;
	template<class type_t>
	static 	type_t	string_to	(std::string const& str)noexcept
	{	
		std::istringstream iss(str);
	        type_t num;
        	iss >> num;
	        return num;
	}
}

namespace qpc
{
	class   timer
        {
                public:
                        timeval _start;
                        timeval _final;

                        inline  void    tic()noexcept
                        {
                                gettimeofday(&_start,0);
                        }
                        inline  void    toc()noexcept
                        {
                                gettimeofday(&_final,0);
                        }
                        inline  auto    get()noexcept
                        {
                                return  (1000000.*(_final.tv_sec - _start.tv_sec) + _final.tv_usec - _start.tv_usec)/1000000.;
                        }
        };
}

namespace qpc
{
	using	long_t	=	long;
	using	bool_t	=	bool;

	using	std::size_t;
}

namespace qpc
{
	using	std::unique_ptr;
	using	std::shared_ptr;
}

namespace qpc
{
	using	std::swap;
	using	std::move;
	using	std::forward;
	using	std::get;
}

namespace qpc
{
	using	std::min;
	using	std::max;

	using 	std::sqrt;
	using	std::tanh;
	using	std::sinh;
	using 	std::cosh;
	using	std::tan;
	using 	std::cos;
	using 	std::sin;
	using 	std::log;
	using	std::exp;
	using	std::pow;
	using	std::lgamma;
	using	std::tgamma;
	using	std::atan;
	using	std::atan2;

	using	std::floor;
	using	std::ceil;
	using	std::round;
	using	std::trunc;
	using	std::lround;

	using	std::norm;
	using	std::real;
	using	std::imag;
	
	using	std::fabs;
	using	std::abs;
}

namespace qpc
{
	using	std::sort;
	using	std::iota;
	using	std::max_element;//(begin,end,cmp)
	using	std::min_element;//(begin,end,cmp)
}

namespace qpc
{

	template<class data_t>
	static constexpr data_t	get_pi	()noexcept
	{
		if constexpr(std::is_same_v<data_t,float>)
		{
			return 	3.14159265359f;
		}
		if constexpr(std::is_same_v<data_t,double>)
		{
			return 	3.1415926535897932;
		}
		if constexpr(std::is_same_v<data_t,long double>)
		{
			return 	3.14159265358979323846264L;
		}
	}	
	template<class data_t>
	static constexpr data_t	PI	=	get_pi<data_t>();

	template<class data_t>
	static constexpr auto	im	=	std::complex<data_t>{0,+1};
	template<class data_t>
	static constexpr auto	jm	=	std::complex<data_t>{0,-1};
}

namespace qpc
{
	template<class...type>
	void	print_array(const char* info,const size_t n,const type*...arr)noexcept
	{
		auto	func	=	[&](auto& obj)
		{
			std::cout<<obj<<",";
		};
		for(size_t i=0;i<n;++i)
		{
			std::cout<<"i="<<i<<","<<info<<":";
			(func(arr[i]),...);
			std::cout<<std::endl;
		}
	}
	
	
}

namespace qpc
{
	//simple arithmetic about std::array
	
	template<class data_t,size_t n>
	static constexpr inline	auto	mul	(const std::array<data_t,n>& arr)noexcept
	{
		auto result=data_t(1);
		for(size_t i=0;i<n;++i)result*=arr[i];
		return	result;
	}

	template<class data_t,size_t n>
        static constexpr inline auto    sum     (const std::array<data_t,n>& arr)noexcept
        {
		auto result=data_t(0);
		for(size_t i=0;i<n;++i)result+=arr[i];
		return	result;
        }

	template<class data_t,size_t n>
	static constexpr inline	auto	max	(const std::array<data_t,n>& arr)noexcept
	{
		return 	*std::max_element(arr.begin(),arr.end());
	}
	template<class data_t,size_t n>
	static constexpr inline	auto	min	(const std::array<data_t,n>& arr)noexcept
	{
		return 	*std::min_element(arr.begin(),arr.end());	
	}

	#define TO_DECLARE_STD_ARRAY_ARITHMETIC(oper)\
	template<class lhsv_t,class rhsv_t,size_t n>\
	static constexpr inline	auto	operator oper (const std::array<lhsv_t,n>& lhsv,const std::array<rhsv_t,n>& rhsv)noexcept\
	{\
		auto	retn	=	std::array<std::remove_reference_t<decltype(lhsv[0] oper rhsv[0])>,n>();\
		for(size_t i=0;i<n;++i)\
		{\
			retn[i]	=	lhsv[i] oper rhsv[i];\
		}\
		return 	retn;\
	}\

	TO_DECLARE_STD_ARRAY_ARITHMETIC(+)
	TO_DECLARE_STD_ARRAY_ARITHMETIC(-)
	TO_DECLARE_STD_ARRAY_ARITHMETIC(*)
	TO_DECLARE_STD_ARRAY_ARITHMETIC(/)

	TO_DECLARE_STD_ARRAY_ARITHMETIC(==)
	TO_DECLARE_STD_ARRAY_ARITHMETIC(!=)
	TO_DECLARE_STD_ARRAY_ARITHMETIC(>=)
	TO_DECLARE_STD_ARRAY_ARITHMETIC(<=)
	TO_DECLARE_STD_ARRAY_ARITHMETIC(>)
	TO_DECLARE_STD_ARRAY_ARITHMETIC(<)

	#undef TO_DECLARE_STD_ARRAY_ARITHMETIC
}

namespace qpc
{
	struct fibonacci_hash final
	{
		static inline uint64_t splitmix64(uint64_t x)noexcept
    		{
			x += 0x9e3779b97f4a7c15;
 			x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9;
			x = (x ^ (x >> 27)) * 0x94d049bb133111eb;
			return x ^ (x >> 31);
		}
		int inline operator()(uint64_t x) const noexcept
    		{
			return 	splitmix64(x);
		}	
	};
}
