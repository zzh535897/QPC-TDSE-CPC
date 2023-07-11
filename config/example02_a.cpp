#pragma once

//flags to control workflow
static constexpr int   	to_save_field           =       1;	//0 = disable, 1 = enable

static constexpr int	calculate_init		=	1;	//0 = by load, 1 = by exact diagonalization
static constexpr int	calculate_prop		=	1;	//0 = disable, 1 = enable

static constexpr int	calculate_coef		=	1;	//0 = do not save, 1 = do save
static constexpr int   	calculate_disp          =       0;	//0 = disable, 1 = enable
static constexpr int	calculate_proj		=	0;	//0 = disable, 1 = only do after prop, 2= do at runtime
static constexpr int   	calculate_spec        	=       1;	//0 = disable, 1 = only PM(E)D, 2= only PWM(E)D, 3= both
static constexpr int	calculate_tsurff	=	0;	//0 = disable, 1 = enable
static constexpr int  	calculate_obsv_z        =       0;	//0 = disable, 1 = enable
static constexpr int	calculate_obsv_r	=	0;	//0 = disable, 1 = enable

static constexpr int   	use_mask_function       =       1;	//0 = disable, 1 = enable

//parameters of omp parallelization
static constexpr size_t n_core		=	8ul;	//number of logical processors to be used

//parameters of spatial configuration
static constexpr size_t n_ndim		=	500ul;
static constexpr size_t n_ldim		=	8ul;
static constexpr size_t n_mdim		=	26ul;
static constexpr long	n_mmin		=	-20;

static constexpr size_t n_rank		=	8ul;	//do not change unless you know what may happen

static constexpr double	para_rmin	=	0.0;
static constexpr double	para_rmax	=	240.;

static constexpr size_t n_pole		=	0ul;	//number of oce poles
static constexpr auto	n_knot		=	std::array<size_t,n_pole>{};

static constexpr bool	propflag_eve	=	true;
static constexpr bool	propflag_odd	=	false;
//parameters of model potential (centrifugal part)
static constexpr double para_zc		=	1.0;
static constexpr auto	potn		=	spherical::hydrogenic{para_zc};

static constexpr double para_zasy	=	1.0;	//asymptotic nuclear charge. only used for non-predefined potentials
static constexpr double para_rasy	=	200.;	//asymptotic criteria distant. only used for non-predefined potentials

//parameters of displayer
static constexpr double disp_rmin	=	para_rmin+1e-8;//avoid 0/0 error when drawing B(r)/r.
static constexpr double disp_rmax	=	para_rmax;
static constexpr double disp_thmin	=	0.5*PI<double>;
static constexpr double disp_thmax	=	1.0*PI<double>;
static constexpr double disp_phmin	=	0.0*PI<double>;
static constexpr double disp_phmax	=	2.0*PI<double>;

static constexpr size_t	disp_nr		=	size_t((disp_rmax-disp_rmin)/0.15);
static constexpr size_t disp_nth	=	1ul;
static constexpr size_t disp_nph	=	181ul;

//parameters of PMD spectrum
static constexpr double spec_krmin	=	0.01;
static constexpr double spec_krmax	=	1.2;
static constexpr double spec_thmin	=	0.5*PI<double>;
static constexpr double spec_thmax	=	1.0*PI<double>;
static constexpr double spec_phmin	=	0.0*PI<double>;
static constexpr double spec_phmax	=	2.0*PI<double>;

static constexpr size_t spec_nkr	=	400ul;
static constexpr size_t spec_nkt	=	1ul;
static constexpr size_t spec_nkp	=	181ul;

static constexpr int	spec_axis_type	=	1;	//0=linear sequence in momentum; 1=linear sequence in energy

//parameters of TSURFF
static constexpr double	tsurff_zc	=	0.0;
static constexpr double tsurff_r0	=	130.0;

//parameters of EigenState projection (only valid for @L and @E)
static constexpr size_t proj_samp_rate	=	1ul;

static constexpr auto	proj_filter	=	[](size_t l,double e,double* v)
{
	return	false;
};

//parameters of Crank-Nicolson solver
static constexpr bool   run_silently    =       0;

static constexpr double para_dt		=	0.04;
static constexpr double para_td		=	0.0;

static constexpr size_t print_rate	=	1000ul;

//parameters of observables
static constexpr auto	func_obsv_z	=	[](double r)//<f(r)*cos(th)>
{
	return	r;//position
};
static constexpr auto	func_obsv_r	=	[](double r)//<f(r)*sin(th)*exp(±iph)>
{
	return	r;//position
};

//parameters of laser facilities
using	envelope_type	=	qpc::field_component_sinen<2>;

static constexpr size_t n_field_x	=	1ul;	//for LP, this is ignored
static constexpr size_t n_field_y	=	1ul;	//for LP, this is ignored
static constexpr size_t n_field_z	=	0ul;	//for EP, this is ignored

static constexpr int	gauge		=	0;	//0 = velocity, 1 = length

//parameters of initial states
static constexpr size_t init_ns		=	1ul;

//parameters of mask function
static constexpr double mask_rmin	=	para_rmax-40.;
static constexpr double mask_rmax	=	para_rmax;
static constexpr double mask_fact	=	1./4.*para_dt;
static constexpr size_t mask_nthd	=	n_core;

//==========================================================================================================================
//
//
//							End of Config
//
//
//==========================================================================================================================
