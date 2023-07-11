#pragma once
//*==================================================================================================*//
//*avxmask intrinsics
//*==================================================================================================*//
/*(1)logistic*/
	inline		avxmask avxmask::operator!()const noexcept
	{
		return 	{_knot_mask8(data)};
	}
	inline 		avxmask	operator&&	(const avxmask& x,const avxmask& y)noexcept
	{
		return 	{_kand_mask8(x,y)};
	}
	inline		avxmask	operator||	(const avxmask& x,const avxmask& y)noexcept
	{
		return 	{_kor_mask8(x,y)};
	}
	inline 		avxmask	operator^	(const avxmask& x,const avxmask& y)noexcept
	{
		return 	{_kxor_mask8(x,y)};
	}
/*(2) tri-op (false case,true case,mask value)*/
	static	inline	double		cond		(const double& x,const double& y,const bool m)noexcept
	{
		return 	m?y:x;
	}
	static	inline	avx128_complex	cond		(const avx128_complex&x,const avx128_complex&y,const bool m)noexcept
	{
		return 	m?y:x;
	}
	static	inline 	avx512_double	cond		(const avx512_double& x,const avx512_double& y,const avx8_mask& m)noexcept
	{
		return 	{_mm512_mask_mov_pd(x,m,y)};
	}
	static 	inline 	avx512_complex 	cond		(const avx512_complex&x,const avx512_complex&y,const avx8_mask& m)noexcept
	{
		return 	{_mm512_mask_mov_pd(x,m,y)};
	}
	static	inline	avx512_sint64	cond		(const avx512_sint64&x,const avx512_sint64&y,const avx8_mask& m)noexcept
	{
		return 	{_mm512_mask_mov_epi64(x,m,y)};
	}
