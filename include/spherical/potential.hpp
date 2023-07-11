#pragma once

//require <libraries/support_std.h>

//--------------------------------------------------------------------------------------------
//				 centrifugal atomic potential
//--------------------------------------------------------------------------------------------

struct	hydrogenic
{	
	double 	zc;
	
	inline	double	operator()(double r)const noexcept
	{
		return	-zc/(r+1e-50);
	}
};//end of hydrogenic

struct	hydrogenic_yukawa
{
	double	zc;
	double	zs;
	double	sg;

	inline	double	operator()(double r)const noexcept
	{
		return 	-(zc+zs*exp(-sg*r))/(r+1e-50);
	}
};//end of hydrogenic_yukawa

struct	hydrogenic_xmtong
{
			//H	He	Ne	Ar	Rb	Ne+	Ar+	
	double	zc;	//1	1	1	1	1	2	2
	double	a1;	//0	1.231	8.069	16.039	24.023	8.043	14.989
	double 	a2;	//0	0.662	2.148	2.007	11.107	2.715	2.217
	double	a3;	//0	-1.325	-3.570	-25.543	115.20	0.506	-23.606
	double	a4;	//0	1.236	1.986	4.525	6.629	0.982	4.585
	double	a5;	//0	-0.231	0.931	0.961	11.977	-0.043	1.011
	double	a6;	//0	0.480	0.602	0.443	1.245	0.401	0.551

	inline	double	operator()	(double r)const noexcept
	{
		return 	-(zc+a1*exp(-a2*r)+a3*r*exp(-a4*r)+a5*exp(-a6*r))/(r+1e-50);
	}

	hydrogenic_xmtong()=default;
	explicit constexpr hydrogenic_xmtong(tstring<'H'>)noexcept:
	zc(1.),a1(0.000),a2(0.000),a3( 0.000),a4(0.000),a5( 0.000),a6( 0.000){}
	explicit constexpr hydrogenic_xmtong(tstring<'H','e'>)noexcept:
	zc(1.),a1(1.231),a2(0.662),a3(-1.325),a4(1.236),a5(-0.231),a6( 0.480){}
	explicit constexpr hydrogenic_xmtong(tstring<'N','e'>)noexcept:
	zc(1.),a1(8.069),a2(2.148),a3(-3.570),a4(1.986),a5( 0.931),a6( 0.602){}
	explicit constexpr hydrogenic_xmtong(tstring<'A','r'>)noexcept:
	zc(1.),a1(16.039),a2(2.007),a3(-25.543),a4(4.525),a5( 0.961),a6( 0.443){}
	explicit constexpr hydrogenic_xmtong(tstring<'R','b'>)noexcept:
	zc(1.),a1(24.023),a2(11.107),a3(115.20),a4(6.629),a5(11.977),a6( 1.245){}
	explicit constexpr hydrogenic_xmtong(tstring<'N','e','+'>)noexcept:
	zc(2.),a1(8.043),a2(2.715),a3( 0.506),a4(0.982),a5(-0.043),a6( 0.401){}
	explicit constexpr hydrogenic_xmtong(tstring<'A','r','+'>)noexcept:
	zc(2.),a1(14.989),a2(2.217),a3(-23.606),a4(4.585),a5( 1.011),a6( 0.551){}	
};//end of hydrogenic_xmtong


//--------------------------------------------------------------------------------------------
//				   spheroidal potential
//--------------------------------------------------------------------------------------------
struct	bicentral
{
	double	zadd;
	double	zsub;
	double	dist;

	constexpr bicentral(const double z1,const double z2,const double r)noexcept:
	zadd(z1+z2),
	zsub(z1-z2),
	dist(r){}

	constexpr auto	potn_zadd()const noexcept
	{
		return 	[&](double x){return zadd*x;};
	}
	constexpr auto	potn_zsub()const noexcept
	{
		return	[&](double x){return zsub;};
	}
};//end of bicentral
