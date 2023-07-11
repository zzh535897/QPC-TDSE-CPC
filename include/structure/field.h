#pragma once

#include <libraries/support_std.h>

#include <utilities/error.h>
//====================================================================================================
//
//                      This file is part of QPC Header Only Library
//
//====================================================================================================

//====================================================================================================
//	
//				General Laser Field Generator
//
//	An "arbitary" laser is expressed:
//
//	F_{x}(t) = F_{x,1}(t) + F_{x,2}(t) + ... F_{x,nx}(t)
//	F_{y}(t) = F_{y,1}(t) + F_{y,2}(t) + ... F_{y,ny}(t)
//	F_{z}(t) = F_{z,1}(t) + F_{z,2}(t) + ... F_{z,nz}(t)
//
//	The provided laser objects implements  F_{s,i}(t) = G_{s,i}(t) * sin(w_{s,i}*t + phi_{s,i}),
//	where G_{s,i}(t) is considered as envelope. This file provides gaussian, sine-power or 
//	trapzoidal type envelopes.
//
//Updated by: Zhao-Han Zhang(张兆涵)  Dec. 23th, 2022
//====================================================================================================

namespace qpc
{
//==========================================================================================
//	the component of Lab-Frame vector (Ex,Ey,Ez) or (Ax,Ay,Az)
//==========================================================================================

	struct	field_component_gauss;
	template<int,int>
	struct	field_component_trapz;
	template<int>
	struct	field_component_sinen;
	struct	field_component_delta;

//==========================================================================================
//				implementations	of components
//==========================================================================================
	struct	field_component_gauss final
	{
		double	para_e0;	//in a.u.
		double	para_w0; 	//in a.u.
		double	para_pd;	//in a.u.
		double	para_cep; 	//in rad
		double	para_tau;	//in a.u.
		double	para_mid;	//in a.u.

		inline	void	initialize
		(
			double e0,//in a.u.	
			double w0,//in a.u.
			double nc,//in o.c., FWHM
			double n0,//in o.c.	
			double c0 //in pi    CEP
		)noexcept
		{
			para_e0	=	e0;
			para_w0	=	w0;	
			para_pd	=	6.283185307179586/w0;
			para_cep=	3.141592653589793*c0;
			para_tau=	nc*para_pd;
			para_mid=	n0*para_pd;	
		}

		inline	double	operator()(const double t)const noexcept
		{	
			double	ts	=	(t-para_mid)/para_tau;
			double	ev	=	exp(-1.386294361119891*ts*ts);
			double	ca	=	para_e0*sin(para_w0*t+para_cep);
			return 	ca*ev;
		}

		inline	double	tmin	()const noexcept
		{
			return	para_mid-2.0*para_tau;
		}
		inline	double	tmax	()const noexcept
		{
			return 	para_mid+2.0*para_tau;
		}
	};//end of field_component_gauss

	template<int n=2>//n is suggested to be power of 2
	struct	field_component_sinen final
	{
		double	para_e0;	//in a.u.
		double	para_w0; 	//in a.u.
		double	para_pd;	//in a.u.
		double	para_cep; 	//in rad
		double	para_tau;	//in a.u.
		double	para_ini;	//in a.u.

		inline	void	initialize
		(
			double e0,//in a.u.	
			double w0,//in a.u.
			double nc,//in o.c.  #cycles
			double n0,//in o.c.   initial
			double c0 //in pi    CEP
		)noexcept
		{
			para_e0	=	e0;
			para_w0	=	w0;	
			para_pd	=	6.283185307179586/w0;
			para_cep=	3.141592653589793*c0;
			para_tau=	nc*para_pd;
			para_ini=	n0*para_pd;	
		}

		inline	double	operator()(const double t)const noexcept
		{
				
			double	ts	=	(t-para_ini)/para_tau;
			double	ev	=	t>para_ini&&t<(para_ini+para_tau)?pow(sin(3.1415926535897932*ts),n):0.0;
			double	ca	=	para_e0*sin(para_w0*t+para_cep);
			return 	ca*ev;
		}
		
		inline	double	tmin	()const noexcept
		{
			return	para_ini;
		}
		inline	double	tmax	()const noexcept
		{
			return	para_ini+para_tau;
		}
	};//end of field_component_sinen

	template<int nr=2,int nf=2>
	struct	field_component_trapz final
	{
		double	para_e0;	//in a.u.
		double	para_w0; 	//in a.u.
		double	para_pd;	//in a.u.
		double	para_cep; 	//in rad

		double	para_init;	//in a.u.
		double	para_rise;	//in a.u.
		double	para_fall;	//in a.u.
		double	para_last;	//in a.u.

		inline	void	initialize
		(
			double e0,//in a.u. 
			double w0,//in a.u.
			double nc,//in o.c.  #cycles
			double n0,//in o.c.  #starts
			double c0 //in pi
		)noexcept
		{
			para_e0	= e0;
			para_w0	= w0;
			para_pd = 6.283185307179586/w0;
			para_cep= 3.141592653589793*c0;

			para_init 	=	n0*para_pd;
			para_last	=	nc*para_pd+para_init;
			para_rise	=	nr*para_pd+para_init;
			para_fall	=	(nc-nf)*para_pd+para_init;
		}
		
		inline	double	operator()(const double t)const noexcept
		{
			if(t<para_init||t>para_last)return .0;
			double  ca;
			if(t<para_rise)
			{
				ca	=	(t-para_init)/(para_rise-para_init);
			}else if(t>para_fall)
			{
				ca	=	(para_last-t)/(para_rise-para_init);
			}else	ca	=	1.0;

			return	ca*para_e0*sin(para_w0*t+para_cep);			
		}

		inline	double	tmin	()const noexcept
		{
			return	para_init;
		}
		inline	double	tmax	()const noexcept
		{
			return 	para_last;
		}
	};//end of field_component_trapz

	struct	field_component_delta final
	{
		double	para_e0;	//in a.u.

		inline	void	initialize
		(
			double e0,//in a.u. 
			double,
			double,
			double,
			double
		)noexcept
		{
			para_e0	= e0;
		}

		inline	double	operator()(const double t)const noexcept
		{
			if(std::fabs(t)<1e-15)return para_e0;
			else return 0.0;
		}

		inline	double	tmin	()const noexcept
		{
			return	0.0;
		}
		inline	double	tmax	()const noexcept
		{
			return 	0.0;
		}
		
	};//end of field_component_delta

//==========================================================================================
//			implementations	of dipolar field
//==========================================================================================
	template<class field_type=field_component_gauss>//field_type should be a field_component class
	struct	field final
	{
		static_assert(std::is_pod_v<field_type>);	

		std::vector<field_type> field_x;	//usually Ex, Bx or Ax
		std::vector<field_type> field_y;	//usually Ey, By or Ay
		std::vector<field_type> field_z;	//usually Ez, Bz or Az

		double	para_ti;
		double	para_tf;
		double	para_dt;
		size_t	para_nt;

		inline	double	t(const size_t i)const noexcept{return para_ti+para_dt*i;}

		inline	void	initialize_t	(const double dt)noexcept
		{
			if(field_x.empty()&&field_y.empty()&&field_z.empty())
			{
				para_ti=0.0;
				para_tf=0.0;
				para_dt=dt;
				para_nt=0ul;
			}else
			{
				para_ti= std::numeric_limits<double>::infinity();
				para_tf=-std::numeric_limits<double>::infinity();
				para_dt=dt;
				for(size_t i=0;i<field_x.size();++i)
				{
					double	tmin	=	field_x[i].tmin();
					double	tmax	=	field_x[i].tmax();
					if(para_ti>tmin)para_ti=tmin;
					if(para_tf<tmax)para_tf=tmax;
				}
				for(size_t i=0;i<field_y.size();++i)
				{
					double	tmin	=	field_y[i].tmin();
					double	tmax	=	field_y[i].tmax();
					if(para_ti>tmin)para_ti=tmin;
					if(para_tf<tmax)para_tf=tmax;
				}
				for(size_t i=0;i<field_z.size();++i)
				{
					double	tmin	=	field_z[i].tmin();
					double	tmax	=	field_z[i].tmax();
					if(para_ti>tmin)para_ti=tmin;
					if(para_tf<tmax)para_tf=tmax;
				}
				para_nt=1ul+(para_tf-para_ti)/para_dt;
			}
		}//end of initialize_t

		template<char label,class...args_t>
		inline	void	initialize	(args_t&&...args)//emplace_back could throw exception from allocator but it rarely,rarely,rarely happens for this class
		{
			if constexpr(label=='x')
			{
				this->field_x.emplace_back().initialize(std::forward<args_t>(args)...);
			}else if constexpr(label=='y')
			{
				this->field_y.emplace_back().initialize(std::forward<args_t>(args)...);
			}else if constexpr(label=='z')
			{
				this->field_z.emplace_back().initialize(std::forward<args_t>(args)...);
			}else if constexpr(label=='t')
			{
				this->initialize_t(std::forward<args_t>(args)...);
			}
		}

		inline	auto operator()	(const double t)const noexcept
		{
			double	fx=0.0;
			double	fy=0.0;
			double	fz=0.0;
			for(size_t ix=0;ix<field_x.size();++ix)fx+=field_x[ix](t);
			for(size_t iy=0;iy<field_y.size();++iy)fy+=field_y[iy](t);
			for(size_t iz=0;iz<field_z.size();++iz)fz+=field_z[iz](t);	
			return 	std::make_tuple(fx,fy,fz);	
		}
	};//end of field


//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++//
//
//
//					Factory Functions
//
//
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++//

template<class envelope_t,size_t nx,size_t ny,size_t nz,class...args_t>
inline  auto    create_laser_field    
(
	const std::array<std::array<double,5>,nx>& para_field_x,
	const std::array<std::array<double,5>,ny>& para_field_y,
	const std::array<std::array<double,5>,nz>& para_field_z,
	const double para_dt,
	args_t&&...args
)
{
	try
	{
        	auto    laser   =       qpc::field<envelope_t>();

		for(size_t i=0;i<para_field_x.size();++i)
		{
			laser.template initialize<'x'>(para_field_x[i][0],para_field_x[i][1],para_field_x[i][2],para_field_x[i][3],para_field_x[i][4],std::forward<args_t>(args)...);
		}
		for(size_t i=0;i<para_field_y.size();++i)
		{
			laser.template initialize<'y'>(para_field_y[i][0],para_field_y[i][1],para_field_y[i][2],para_field_y[i][3],para_field_y[i][4],std::forward<args_t>(args)...);
		}
		for(size_t i=0;i<para_field_z.size();++i)
		{
			laser.template initialize<'z'>(para_field_z[i][0],para_field_z[i][1],para_field_z[i][2],para_field_z[i][3],para_field_z[i][4],std::forward<args_t>(args)...);
        	}       laser.template initialize<'t'>(para_dt);
        	return  laser;
	}catch(const std::bad_alloc& e)
	{
		throw qpc::runtime_error_t<std::string,void>(e.what());
	}
}//end of create_laser_field


	

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++//
//
//
//				Formated Input
//
//
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++//

template<class para_x,class para_y,class para_z>
inline	void	initialize_laser_parameter	
(
	int argc,char** argv,	//preshifted argc,argv
	para_x& para_field_x,	//has .data, .size
	para_y& para_field_y,	//has .data, .size
	para_z& para_field_z	//has .data, .size
)
{
	auto	choose_dir	=	[&](char dir_flag)
	{
		switch(dir_flag)
		{
			case 'X':return std::make_tuple(para_field_x.data(),para_field_x.size());
			case 'Y':return std::make_tuple(para_field_y.data(),para_field_y.size());
			case 'Z':return std::make_tuple(para_field_z.data(),para_field_z.size());
			default :{}
		};
		return std::make_tuple(decltype(para_field_x.data())(0),0ul);
	};

	auto	choose_pos	=	[&](char pos_flag)
	{
		return	size_t(pos_flag-'0');
	};

	while(argc>0)
	{
		auto [parptr,length]= 	choose_dir(argv[0][0]);

		if(parptr==NULL)throw qpc::runtime_error_t<std::string,void>("valid symbols if corresponding laser objects are non-empty: 'X', 'Y', 'Z'.");

		size_t	id1	=	choose_pos(argv[0][1]);
		size_t	id2	=	choose_pos(argv[0][2]);

		if(id1>=length)throw qpc::runtime_error_t<std::string,void>("index exceeds existing laser objects.");

		auto&	target	=	parptr[id1][id2];

		target = string_to<double>(argv[1]);

		argc-=2;
		argv+=2;
	};
}//end









}//end of qpc
