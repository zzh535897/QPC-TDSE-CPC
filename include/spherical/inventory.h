#pragma once
#include <libraries/support_std.h>
#include <libraries/support_avx.h>
#include <libraries/support_omp.h>
#include <libraries/support_hdf.h>

#include <resources/arena.h>		//to deal with memory allocation
#include <utilities/error.h>		//to deal with exception
#include <utilities/recid.h>		//recursive indexable pointer
#include <utilities/align.h>
#include <utilities/string.h>

#include <structure/field.h>

#include <functions/basis_bsplines.h>
#include <functions/basis_legendre.h>
#include <functions/basis_coulombf.h>
#include <functions/spheroidal.h>

#include <algorithm/ed.h>		//exact diagonalization/linear system solver, by calling LAPACKE,FEAST,DSS,PARDISO etc.
#include <algorithm/rk.h>		//ordinary differential equation solvers

//====================================================================================================
//
//                      This file is part of QPC Header Only Library
//
//====================================================================================================

//====================================================================================================
//Updated by: Zhao-Han Zhang(张兆涵)  Mar. 20th, 2023
//Updated by: Zhao-Han Zhang(张兆涵)  Feb. 05th, 2023
//Updated by: Zhao-Han Zhang(张兆涵)  Dec. 18th, 2022
//
//Copyright © 2022-2023 Zhao-Han Zhang
//====================================================================================================


namespace qpc{
namespace spherical{

//-----------------------------------------------------------------------------------------------------------
using	rsrc_t	=	arena;					//all memories are allocated from an 'arena'
using 	errn_t	=	runtime_error_t<std::string,void>;	//exception class for namespace spherical
using	comp_t	=	avx128_complex;
#ifdef support_avx3
using	comp_v	=	avx512_complex;				//only explicitly used in intrinsic.hpp and propagate.hpp
#endif
//-----------------------------------------------------------------------------------------------------------
//one-electron dimension specifier
template<size_t _nr,size_t _mr,size_t _nl,size_t _nm,long _m0,bool _lf,bool _rt>
struct	dimension_1e;

//-----------------------------------------------------------------------------------------------------------
//one-electron matrix element integrator
template<class dims_t>
using	integral_result_radi	=	recvec<double,recidx<dims_t::n_dims,dims_t::n_elem>>;//the result viewer class returned by integrator_radi
template<class dims_t>
using	integral_result_angu	=	recvec<double,recidx<dims_t::m_dims,dims_t::l_dims>>;//the result viewer class returned by integrator_angu

template<class dims_t>
struct	integrator_radi;	//factory of integrals between B-splines functions
template<class dims_t>	
struct	integrator_angu;	//factory of integrals between Spherical Harmonics

//-----------------------------------------------------------------------------------------------------------
//one-electron wave function coefficients
template<class dims_t>
struct	coefficient;		//user class

template<class dims_t>
using	coefficient_view	=	recvec<comp_t,recidx<dims_t::m_dims,dims_t::l_dims,dims_t::n_dims>>;	
template<class dims_t>
struct	coefficient_data;

//-----------------------------------------------------------------------------------------------------------
//one-electron operator (general)
template<class dims_t>
struct	operator_hc;	//hamiltonian matrix (centrifugal)
template<class dims_t,size_t n_asym=1>
struct	operator_hb;	//hamiltonian matrix (prolate spheroidal), charge symmeric or non-symmetric
template<class dims_t,size_t n_pole=1>
struct	operator_hm;	//hamiltonian matrix (multipolar)

template<class dims_t>
struct	operator_sc;	//<Bi|1|Bj>, the overlap matrix (centrifugal & multipolar)
template<class dims_t>
struct	operator_sb;	//<Bi|x^2|Bj><l,m|1|l,m>-<Bi|1|Bj>*<l1,m|e^2|l2,m>, the overlap matrix (prolate spheroidal)

//-----------------------------------------------------------------------------------------------------------
//one-electron operator (radial)
template<class dims_t>
struct	operator_rsym;	//general rsubspace symmetric operator data container
template<class dims_t>
struct	operator_rasy;	//general rsubspace asymmetric operator data container

template<class dims_t>
struct	operator_rpow0;	//<Bi|r^0|Bj>
template<class dims_t>
struct	operator_rpow1;	//<Bi|r^1|Bj>
template<class dims_t>
struct	operator_rpow2;	//<Bi|r^2|Bj>
template<class dims_t>
struct	operator_rpow3;	//<Bi|r^3|Bj>
template<class dims_t>
struct	operator_rinv1;	//<Bi|1/r|Bj>
template<class dims_t>
struct	operator_rdif1;	//<Bi|1|Bj'>

template<class dims_t,size_t n_subs>
struct	operator_xsub;	//<Bi,m1|f(x)|Bj,m2>

//-----------------------------------------------------------------------------------------------------------
//one-electron operator (angular)
template<class dims_t>
struct	operator_angu;	//general angular operator data container

template<class dims_t>
struct	operator_y10;	//<l1,m1|costh|l2,m2>
template<class dims_t>
struct	operator_q10;	//<l1,m1|-sinth d/dth-costh|l2,m2>
template<class dims_t,int sign>
struct	operator_y11;	//<l1,m1|sinth*exp(+iph)|l2,m2> or <l1,m1|sinth*exp(-iph)|l2,m2>
template<class dims_t,int sign>
struct	operator_q11;	//<l1,m1|-costh*Lp-sinth|l2,m2> or <l1,m1|costh*Lm-sinth|l2,m2>

template<class dims_t>
struct	operator_eta1;	//<l1,m|eta^1|l2,m>
template<class dims_t>	
struct	operator_eta2;	//<l1,m|eta^2|l2,m>
template<class dims_t>
struct	operator_eta3;	//<l1,m|eta^3|l2,m>
template<class dims_t>
struct	operator_chi1;	//<l1,m|sqrt(1-eta^2)(d/deta)sqrt(1-eta^2)|l2,m>

//-----------------------------------------------------------------------------------------------------------
//one electron boundary absorber
template<class dims_t>
struct	absorber_mask;		//mask-type absorber (centrifugal and spheroidal are both ok)

//-----------------------------------------------------------------------------------------------------------
//one electron propgator (lanczos-arnoldi without split-operator for sphTDSE) [deprecated due to efficiency]
template<class dims_t>
struct	propagator_arnoldi;	//the base class

template<class dims_t>
struct	propagator_hc;		//P^2/2M + V(r)
template<class dims_t>
struct	propagator_hc_lgwz;	//P^2/2M + V(r) - eE*Z
template<class dims_t>
struct	propagator_hc_vgwz;	//P^2/2M + V(r) - eA*Pz
template<class dims_t>
struct	propagator_hc_lgwr;	//P^2/2M + V(r) - eE1*X-eE2*Y
template<class dims_t>
struct	propagator_hc_vgwr;	//P^2/2M + V(r) - eA1*Px-eA2*Py

//-----------------------------------------------------------------------------------------------------------
//one electron propgator (lanczos-arnoldi without split-operator for bicTDSE)
template<class dims_t,size_t n_asym>
struct	propagator_hb;		//P^2/2M + V(r1,r2) 
template<class dims_t,size_t n_asym>
struct	propagator_hb_lgwz;	//P^2/2M + V(r1,r2) - eE*Z 
template<class dims_t,size_t n_asym>
struct	propagator_hb_vgwz;	//P^2/2M + V(r1,r2) - eA*Pz

template<class dims_t,size_t n_asym>
struct	propagator_hb_lgwr;	//P^2/2M + V(r1,r2) - eEx*X  -eEy*Y  TODO
template<class dims_t,size_t n_asym>
struct	propagator_hb_vgwr;	//P^2/2M + V(r1,r2) - eAx*Px -eAy*Py TODO 

//-----------------------------------------------------------------------------------------------------------
//one electron propgator (lanczos-arnoldi without split-operator for multipolar sphTDSE) [deprecated due to efficiency]
template<class dims_t,size_t n_pole=1>
struct	propagator_hm;		//P^2/2M + Vm

//-----------------------------------------------------------------------------------------------------------
//one electron propagator (crank-nicolson with split-operator for sphTDSE and molTDSE)
template<class dims_t>
struct	cn_workspace_so;	//workspace for crank-nicolson propagator using split-operator&partial diagonalization
template<class dims_t>
struct	cn_workspace_sp;	//workspace for crank-nicolson propagator using MKL pardiso with lowrank update

template<class dims_t>
struct	cn_component_hc;	//store the data required for type exp(I @ S\H), where I is diagonal in ml-space, H is banded in r-space and only depend on l, not m
template<class dims_t>
struct	cn_component_hl;	//store the data required for type exp(Y @ S\R), where Y is banded in l-space and R is banded in r-space, R does not depend on ml.
template<class dims_t>
struct	cn_component_hm;	//store the data required for type exp(Z @ S\R), where Z is banded in ml-space and R is banded in r-space, R does not depend on ml.

template<class dims_t,class data_t>
struct	cn_transform_hl;	//store a unitary transform in l-space
template<class dims_t,class data_t>	
struct	cn_transform_hm;	//store a unitary transform in ml-space

template<class dims_t>
struct	propagator_cn_hc;	//P^2/2M + V(r)
template<class dims_t>
struct	propagator_cn_hc_lgwz;	//P^2/2M + V(r) - Ez*Z
template<class dims_t>
struct	propagator_cn_hc_lgwr;	//P^2/2M + V(r) - Ex*X-Ey*Y
template<class dims_t>
struct	propagator_cn_hc_lgws;	//P^2/2M + V(r) - Ez*Z-Ex*X
template<class dims_t>
struct	propagator_cn_hc_vgwz;	//P^2/2M + V(r) - Az*Pz/M
template<class dims_t>
struct	propagator_cn_hc_vgwr;	//P^2/2M + V(r) - Ax*Px/M-Ay*Py/M
template<class dims_t>
struct	propagator_cn_hc_vgws;	//P^2/2M + V(r) - Az*Pz/M-Ax*Px/M

template<class dims_t,size_t n_pole=1>
struct	propagator_cn_hm;	//P^2/2M + Vm(r)

template<class dims_t>
struct	propagator_cn_lgwz;	//-Ez*Z
template<class dims_t>
struct	propagator_cn_lgwr;	//-Ex*X-Ey*Y

template<class dims_t>
struct	propagator_cn_vgwz;	//-Az*Pz/M
template<class dims_t>
struct	propagator_cn_vgwr;	//-Ax*Px/M-Ay*Py/M
//-----------------------------------------------------------------------------------------------------------
//one electron observable (centrifugal)
template<class dims_t>
struct	observable_r;		//<Y11@f(r)> or <Y1-1@f(r)>
template<class dims_t>
struct	observable_z;		//<Y10@f(r)>

//one electron observable (prolate spheroidal)
template<class dims_t>
struct	observable_prolate_z;
template<class dims_t>
struct	observable_prolate_r;	//TODO

//-----------------------------------------------------------------------------------------------------------
//SAE potential models
struct	hydrogenic;
struct	hydrogenic_yukawa;
struct	hydrogenic_xmtong;

//-----------------------------------------------------------------------------------------------------------
//radial equations for ode solvers to solve
template<class data_t>
struct	equation_hydrogenic;
template<class data_t>
struct	equation_hydrogenic_yukawa;
template<class data_t>
struct	equation_hydrogenic_xmtong;
template<class data_t,class potn_t>
struct	equation_general;

template<class data_t,class func_t>
struct	equation_centrifugal;	

template<class data_t>
struct	solution;		//the container for holding a numerical solution to the radial ode function and the phase shift
//-----------------------------------------------------------------------------------------------------------
//one-electron displayer
template<class dims_t>
struct	spectrum_x;	//wave function displayer (for both centrifugal and spheroidal)

//-----------------------------------------------------------------------------------------------------------
//one-electron spectrum
template<class dims_t,size_t nkr,size_t nkt,size_t nkp>
struct	spectrum_c_pcs;	//PMD/PAD/TCS using PCS algorithm(for hydrogenic potential, it uses the analytic partial wave expansion)
template<class dims_t,size_t nkr,size_t nkt,size_t nkp>
struct	spectrum_b_pcs;	//PMD/PAD     using PCS algorithm(for bicentric potential. it uses the analytic partial wave expansion)

template<class dims_t,size_t nkr,size_t nkt,size_t nkp>
struct	spectrum_c_tsf;	//PMD/PAD using tSURFF algorithm (for centrifugal system)
template<class dims_t>
struct	spectrum_b_tsf;	//PMD/PAD using tSURFF algorithm (for centrifugal system) TODO

//-----------------------------------------------------------------------------------------------------------
//one-electron ionization yield
template<class dims_t>
struct	ionization_sph;	//for centrifugal system, using a filter of E<0 to evaluate the ionizational yield
template<class dims_t>
struct	ionization_bic;	//for spheroidal symtem, using a filter of E<0 to evaluate the ionizational yield

//-----------------------------------------------------------------------------------------------------------
//one-electron state statistics
template<class dims_t>
struct	eigenstate_sph;	//calculate the projection onto required eigenstates of spherical symmetric system
template<class dims_t>
struct	eigenstate_bic;	//calculate the projection onto required eigenstates of prolate spheroidal symmetric system

//-----------------------------------------------------------------------------------------------------------
//implementations are included in other header files.

#include <spherical/utilities/inventory.h>
#include <spherical/dimension/inventory.h>

#include <spherical/operators/inventory.h>
#include <spherical/coefficient/inventory.h>
#include <spherical/propagator/inventory.h>

#include <spherical/absorber.hpp>

#include <spherical/potential.hpp>

#include <spherical/observable/inventory.h>
#include <spherical/spectrum/inventory.h>

}}//end of qpc::spherical









