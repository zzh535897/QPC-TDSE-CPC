#pragma once

//some constexpr version function that cstdlib does not implemented as 'constexpr'. TODO remove me after c++2a
template<class lhs_t,class rhs_t>
static constexpr auto cmax(const lhs_t& lhs,const rhs_t& rhs)noexcept
{
	return 	lhs>rhs?lhs:rhs;	
}
template<class lhs_t,class rhs_t>
static constexpr auto cmin(const lhs_t& lhs,const rhs_t& rhs)noexcept
{	
	return 	lhs<rhs?lhs:rhs;
}
template<class lhs_t,class rhs_t>
static constexpr auto cminmax(const lhs_t& lhs,const rhs_t& rhs)noexcept
{
	return 	lhs<rhs?std::make_tuple(lhs,rhs):std::make_tuple(rhs,lhs);
}
template<class val_t>
static constexpr auto cabs(const val_t& val)noexcept
{
	return	val>=0?val:-val;
}

//====================================================================================================
//
//                    The following files are part of QPC Header Only Library
//
//====================================================================================================

//====================================================================================================
// Updated by: Zhao-Han Zhang(张兆涵)  Mar. 20th, 2023
// Updated by: Zhao-Han Zhang(张兆涵)  Dec. 18th, 2022
//
// Copyright © 2022-2023 Zhao-Han Zhang
//====================================================================================================

#include <spherical/utilities/intrinsic.hpp>
#include <spherical/utilities/converter.hpp>
#include <spherical/utilities/basisfunc.hpp>
#include <spherical/utilities/multipole.hpp>
