#pragma once

//====================================================================================================
//
//                    The following files are part of QPC Header Only Library
//
//====================================================================================================

//====================================================================================================
// Updated by: Zhao-Han Zhang(张兆涵)  Dec. 29th, 2022
//
// Copyright © 2022 Zhao-Han Zhang
//====================================================================================================

inline	void	set_sequence_lin	(double* axis,double xmin,double xmax,size_t n)noexcept
{
	double	dx	=	(xmax-xmin)/(n>1?n-1:1);
	for(size_t i=0;i<n;++i)
	{
		axis[i] =       i*dx+xmin;		//axis=X
	}
	if(n>1)
	{
		axis[n-1]=	xmax;
	}
}//end of set_sequence_lin

inline	void	set_sequence_sqr	(double* axis,double xmin,double xmax,size_t n,double c=2.0)noexcept
{
	double	dx	=	(xmax-xmin)/(n>1?n-1:1);
	for(size_t i=0;i<n;++i)
	{
		axis[i] =       sqrt(c*(i*dx+xmin));	//axis=sqrt(c*X)
	}	
	if(n>1)
	{
		axis[n-1]=	sqrt(c*xmax);
	}
}//end of set_sequence_sqr

inline	void	set_sequence_cos	(double* axis,double xmin,double xmax,size_t n,double w=1.0)noexcept
{
	double	dx	=	(xmax-xmin)/(n>1?n-1:1);
	for(size_t i=0;i<n;++i)
	{
		axis[i] =       cos(w*(i*dx+xmin));	//axis=cos(w*X)
	}	
	if(n>1)
	{
		axis[n-1]=	cos(w*xmax);
	}
}//end of set_sequence_cos

#include <spherical/spectrum/position_1e.hpp>

#include <spherical/spectrum/equation.hpp>

#include <spherical/spectrum/momentum_1e.hpp>

#include <spherical/spectrum/ionization_1e.hpp>

#include <spherical/spectrum/population_1e.hpp>

