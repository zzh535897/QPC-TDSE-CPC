#pragma once
#include <immintrin.h>

#ifdef __AVX__			//work for GCC
	#define support_avx1 	//claim to additionally support the 128bit avx sets
#endif

#ifdef __AVX2__			//work for GCC
	#define support_avx2	//claim to additionally support the 256bit avx sets
#endif

#ifdef __AVX512__		//work for GCC
	#define support_avx3	//claim to additionally support the 512bit avx sets
#endif

#define	_0b11111111 255
#define _0b00000000 0
#define _0b01010101 85
#define _0b10101010 170
#define _0b00000011 4
#define _0b01000100 68
#define _0b11101110 238	

#include <libraries/support_omp.h>
#include <utilities/types.h>

namespace qpc
{
	enum class avx_level
	{
		origin
#ifdef support_avx1
		,avx128
#endif
#ifdef support_avx2
		,avx256
#endif
#ifdef support_avx3
		,avx512
#endif
	};

	template<avx_level level>
	struct	realnum;
	template<avx_level level>
	struct	complex;
	template<avx_level level>
	struct	integer;
	template<avx_level level>
	struct	unteger;

	template<avx_level level>struct	avx_dvecn;
	template<avx_level level>struct	avx_cvecn;

	struct	avxmask;

	template<>struct avx_dvecn<avx_level::origin>{static constexpr int n=1;};
#ifdef support_avx1
	using	avx128_complex	=	complex<avx_level::avx128>;
	using	avx128_double	=	realnum<avx_level::avx128>;
	using	avx128_sint64	=	integer<avx_level::avx128>;
	using	avx128_uint64	=	unteger<avx_level::avx128>;

	template<>struct avx_dvecn<avx_level::avx128>{static constexpr int n=2;};
	template<>struct avx_cvecn<avx_level::avx128>{static constexpr int n=1;};
#endif
#ifdef support_avx2
	using	avx256_complex	=	complex<avx_level::avx256>;
	using	avx256_double	=	realnum<avx_level::avx256>;
	using	avx256_sint64	=	integer<avx_level::avx256>;
	using	avx256_uint64	=	unteger<avx_level::avx256>;

	template<>struct avx_dvecn<avx_level::avx256>{static constexpr int n=4;};
	template<>struct avx_cvecn<avx_level::avx256>{static constexpr int n=2;};
#endif
#ifdef support_avx3
	using	avx512_complex	=	complex<avx_level::avx512>;
	using	avx512_double	=	realnum<avx_level::avx512>;
	using	avx512_sint64	=	integer<avx_level::avx512>;
	using	avx512_uint64	=	unteger<avx_level::avx512>;
	using	avx8_mask	=	avxmask;

	template<>struct avx_dvecn<avx_level::avx512>{static constexpr int n=8;};
	template<>struct avx_cvecn<avx_level::avx512>{static constexpr int n=4;};
#endif

//*==================================================================================================*//
//			To define the wrapper class for simd vectors
//*==================================================================================================*//
/*macros for general purpose*/
	#define __declare_avx_struct(prefix,alignval,classname,temparg,dataclass,indx,cast,sfop,idop,tran,sgop)\
	prefix 	struct	alignas(alignval) classname temparg\
	{\
		using	data_t	= dataclass;\
		using	this_t	= classname;\
		data_t	data;\
		indx\
		cast\
		sfop\
		idop\
		tran\
		sgop\
	};

	#define	__declare_none

	#define	__declare_indx\
	constexpr  		auto&	operator[](size_t const i)noexcept		{return data[i];}\
	constexpr const		auto&	operator[](size_t const i)const noexcept 	{return data[i];}\

	#define	__declare_cast\
	constexpr operator		auto&()noexcept		{return data;}\
	constexpr operator const	auto&()const noexcept	{return data;}\

	#define	__declare_sfop\
	constexpr		this_t&	operator += (const this_t& rhs)noexcept{data+=rhs.data;return *this;}\
	constexpr		this_t&	operator -= (const this_t& rhs)noexcept{data-=rhs.data;return *this;}\
	constexpr		this_t&	operator *= (const this_t& rhs)noexcept{data*=rhs.data;return *this;}\
	constexpr		this_t&	operator /= (const this_t& rhs)noexcept{data/=rhs.data;return *this;}

	#define	__declare_spop\
	inline	 		this_t&	operator += (const this_t& rhs)noexcept{data+=rhs.data;return *this;}\
	inline	 		this_t&	operator -= (const this_t& rhs)noexcept{data-=rhs.data;return *this;}\
	inline	 		this_t&	operator *= (const this_t& rhs)noexcept;\
	inline	 		this_t&	operator /= (const this_t& rhs)noexcept;\

	#define	__declare_idop\
	constexpr 		auto	operator +()const noexcept{return this_t{+data};}\
	constexpr		auto 	operator -()const noexcept{return this_t{-data};}

	#define	__declare_tran(trgt)\
	inline	operator		trgt()noexcept;\
	inline	operator const		trgt()const noexcept;\

	#define	__declare_eval(trgt)\
	inline			this_t& operator =(const trgt& rhs)noexcept;

	#define	__declare_sgop(oper)\
	inline	this_t operator oper()const noexcept;

/*declarations*/
	__declare_avx_struct(template<>, 8,realnum,<avx_level::origin>,double ,__declare_none,__declare_cast,__declare_sfop,__declare_idop,__declare_tran(double)        ,__declare_eval(double))

#ifdef support_avx1
	__declare_avx_struct(template<>,16,realnum,<avx_level::avx128>,__m128d,__declare_indx,__declare_cast,__declare_sfop,__declare_idop,__declare_tran(avx128_sint64),__declare_eval(double))
	__declare_avx_struct(template<>,16,complex,<avx_level::avx128>,__m128d,__declare_indx,__declare_cast,__declare_spop,__declare_idop,__declare_eval(double),__declare_none)
	__declare_avx_struct(template<>,16,integer,<avx_level::avx128>,__m128i,__declare_indx,__declare_cast,__declare_sfop,__declare_idop,__declare_tran(avx128_double),__declare_none)
	__declare_avx_struct(template<>,16,unteger,<avx_level::avx128>,__m128i,__declare_indx,__declare_cast,__declare_sfop,__declare_idop,__declare_tran(avx128_double),__declare_none)
#endif

#ifdef support_avx2	
	__declare_avx_struct(template<>,32,complex,<avx_level::avx256>,__m256d,__declare_indx,__declare_cast,__declare_spop,__declare_idop,__declare_eval(double),__declare_eval(avx128_complex))
	__declare_avx_struct(template<>,32,realnum,<avx_level::avx256>,__m256d,__declare_indx,__declare_cast,__declare_sfop,__declare_idop,__declare_tran(avx256_sint64),__declare_eval(double))
	__declare_avx_struct(template<>,32,integer,<avx_level::avx256>,__m256i,__declare_indx,__declare_cast,__declare_sfop,__declare_idop,__declare_tran(avx256_double),__declare_none)
	__declare_avx_struct(template<>,32,unteger,<avx_level::avx256>,__m256i,__declare_indx,__declare_cast,__declare_sfop,__declare_idop,__declare_tran(avx256_double),__declare_none)
#endif

#ifdef support_avx3
	__declare_avx_struct(template<>,64,complex,<avx_level::avx512>,__m512d,__declare_indx,__declare_cast,__declare_spop,__declare_idop,__declare_eval(double),__declare_eval(avx128_complex))
	__declare_avx_struct(template<>,64,realnum,<avx_level::avx512>,__m512d,__declare_indx,__declare_cast,__declare_sfop,__declare_idop,__declare_tran(avx512_sint64),__declare_eval(double))
	__declare_avx_struct(template<>,64,integer,<avx_level::avx512>,__m512i,__declare_indx,__declare_cast,__declare_sfop,__declare_idop,__declare_tran(avx512_double),__declare_none)
	__declare_avx_struct(template<>,64,unteger,<avx_level::avx512>,__m512i,__declare_indx,__declare_cast,__declare_sfop,__declare_idop,__declare_tran(avx512_double),__declare_none)
	__declare_avx_struct(__declare_none,8,avxmask,__declare_none,__mmask8,__declare_none,__declare_cast,__declare_none,__declare_none,__declare_none,__declare_sgop(!))
#endif

/*clean up macros*/	
	#undef 	__declare_avx_struct
	#undef 	__declare_none
	#undef 	__declare_indx
	#undef 	__declare_cast
	#undef 	__declare_sfop
	#undef 	__declare_tran
	#undef 	__declare_sgop

//*==================================================================================================*//
//				To cooperate with utilities/type.h
//*==================================================================================================*//
	#define	__declare_avx_value(type,name,...)\
	template<>\
	struct	name<type>\
	{\
		static constexpr type value = {__VA_ARGS__};\
	};

#ifdef support_avx1
	__declare_avx_value(realnum<avx_level::avx128>,zero_v,0.,0.)
	__declare_avx_value(complex<avx_level::avx128>,zero_v,0.,0.)
	__declare_avx_value(integer<avx_level::avx128>,zero_v,0l,0l)

	__declare_avx_value(realnum<avx_level::avx128>,idty_v,1.,1.)
	__declare_avx_value(complex<avx_level::avx128>,idty_v,1.,0.)
	__declare_avx_value(integer<avx_level::avx128>,idty_v,1l,1l)

	__declare_avx_value(complex<avx_level::avx128>,unim_v,0.,1.)
#endif

#ifdef support_avx2
	__declare_avx_value(realnum<avx_level::avx256>,zero_v,0.,0.,0.,0.)
	__declare_avx_value(complex<avx_level::avx256>,zero_v,0.,0.,0.,0.)
	__declare_avx_value(integer<avx_level::avx256>,zero_v,0l,0l,0l,0l)

	__declare_avx_value(realnum<avx_level::avx256>,idty_v,1.,1.,1.,1.)
	__declare_avx_value(complex<avx_level::avx256>,idty_v,1.,0.,1.,0.)
	__declare_avx_value(integer<avx_level::avx256>,idty_v,1l,1l,1l,1l)

	__declare_avx_value(complex<avx_level::avx256>,unim_v,0.,1.,0.,1.)
#endif

#ifdef support_avx3
	__declare_avx_value(realnum<avx_level::avx512>,zero_v,0.,0.,0.,0.,0.,0.,0.,0.)
	__declare_avx_value(complex<avx_level::avx512>,zero_v,0.,0.,0.,0.,0.,0.,0.,0.)
	__declare_avx_value(integer<avx_level::avx512>,zero_v,0l,0l,0l,0l,0l,0l,0l,0l)

	__declare_avx_value(realnum<avx_level::avx512>,idty_v,1.,1.,1.,1.,1.,1.,1.,1.)
	__declare_avx_value(complex<avx_level::avx512>,idty_v,1.,0.,1.,0.,1.,0.,1.,0.)
	__declare_avx_value(integer<avx_level::avx512>,idty_v,1l,1l,1l,1l,1l,1l,1l,1l)

	__declare_avx_value(complex<avx_level::avx512>,unim_v,0.,1.,0.,1.,0.,1.,0.,1.)

#endif
//*==================================================================================================*//
//*==================================================================================================*//
	#define __declare_oper(oper,retu,lhst,rhst,impl)\
	retu 	operator oper(lhst,rhst)noexcept{return impl;}

	#define	__declare_func(func,retu,lhst,rhst,impl)\
	retu func(lhst,rhst)noexcept{return impl;}

#ifdef support_avx1
	#include <libraries/avx/avx128_complex.h>

	#pragma omp declare reduction (+:avx128_double:omp_out+=omp_in)	
	#pragma omp declare reduction (+:avx128_complex:omp_out+=omp_in)	
	#pragma omp declare reduction (+:avx128_sint64:omp_out+=omp_in)	
#endif

#ifdef support_avx2
	#include <libraries/avx/avx256_realnum.h>

	#pragma omp declare reduction (+:avx256_double:omp_out+=omp_in)	
	#pragma omp declare reduction (+:avx256_complex:omp_out+=omp_in)	
	#pragma omp declare reduction (+:avx256_sint64:omp_out+=omp_in)	
#endif

#ifdef support_avx3
	#include <libraries/avx/avx_mask.h>

	#include <libraries/avx/avx512_complex.h>
	#include <libraries/avx/avx512_realnum.h>
	#include <libraries/avx/avx512_integer.h>

	#pragma omp declare reduction (+:avx512_double:omp_out+=omp_in)	
	#pragma omp declare reduction (+:avx512_complex:omp_out+=omp_in)	
	#pragma omp declare reduction (+:avx512_sint64:omp_out+=omp_in)
#endif
	#undef __declare_oper
	#undef __declare_func
}

#include <libraries/support_mkl.h>
