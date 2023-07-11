#pragma once

#include <libraries/support_std.h>

namespace qpc
{
	template<class type_t>
	struct	zero_v
	{
		static constexpr type_t value	=	type_t(0);
	};
	template<class type_t>
	struct	idty_v
	{
		static constexpr type_t value	=	type_t(1);
	};
	template<class type_t>
	struct	unim_v;

	template<class type_t>
	static constexpr type_t	zero	=	zero_v<std::remove_reference_t<type_t>>::value;
	template<class type_t>
	static constexpr type_t idty	=	idty_v<std::remove_reference_t<type_t>>::value;
	template<class type_t>
	static constexpr type_t unim	=	unim_v<std::remove_reference_t<type_t>>::value;

	//=======================================================================================

	enum class init_type
	{
		as_array=0,
		as_value=1
	};

	//extended floating number
	template<class type_t>
	struct extnum
	{
		type_t	val;
		type_t	err;

		inline	auto	operator()()const noexcept
		{
			return val+err;
		}

		template<class args_t>
		inline	auto&	operator=(const args_t& v)noexcept
		{	
			val=v;
			err=zero<type_t>;
			return 	*this;
		}
		inline	auto	operator+()const noexcept{return *this;}
		inline	auto	operator-()const noexcept{return extnum<type_t>{-val,-err};}

		extnum()=default;
		extnum(const extnum&)=default;
		extnum(extnum&&)=default;

		extnum(const type_t& v                )noexcept:val(v),err(zero<type_t>){}
		extnum(const type_t& v,const type_t& e)noexcept:val(v),err(e){}
	};

	template<class type_t>
	static constexpr auto	two_sum	(const type_t& a,const type_t& b)noexcept
	{
		type_t	s=a+b;
		type_t	v=s-a;
		type_t	e=(a-(s-v))+(b-v);
		return	extnum<type_t>{s,e};
	}
	template<class type_t>
	static constexpr auto	operator+(const extnum<type_t>& a,const type_t& b)noexcept
	{
		auto	b1	=	two_sum(a.err,b);
		auto	x1	=	two_sum(a.val,b1.val);
		x1.err=x1.err+b1.err;
		return 	x1;
	}
	template<class type_t>
        static constexpr auto   operator+(const type_t& a,const extnum<type_t>& b)noexcept
        {
		return 	b+a;
        }
	template<class type_t>
	static constexpr auto	operator+(const extnum<type_t>& a,const extnum<type_t>& b)noexcept
	{
		auto	a1	=	two_sum(a.val,b.err);
		auto	b1	=	two_sum(a.err,b.val);
		auto	x1	=	two_sum(a1.val,b1.val);
		x1.err=x1.err+(a1.err+b1.err);
		return 	x1;
	}
}
