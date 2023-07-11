#include <spherical/inventory.h>

using namespace qpc;
using namespace std;
using qpc::spherical::comp_t;
using std::string;

#ifdef PARA_PATH
#include PARA_PATH
#else 
#include <../config/sphTDSE.cpp>
#endif
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
static_assert(n_core>0ul);

static_assert(mask_rmin>=para_rmin && mask_rmax<=para_rmax && mask_rmin < mask_rmax);
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
static constexpr double para_me		=	1.0; //mass of electron

using dims_t	=	spherical::dimension_1e<n_ndim,n_rank,n_ldim,n_mdim,n_mmin>;

static constexpr char default_load_path[]	=	"./sphTDSE_default_load.h5";
static constexpr char default_save_path[]	=	"./sphTDSE_default_save.h5";

namespace fpara//define the laser objects (called by @F)
{
	using	field_param	=	std::array<double,5>;//(e0,w0,nc,n0,cep), n0 is center

	static auto para_x	=	std::array<field_param,n_field_x>{};	
	static auto para_y	=	std::array<field_param,n_field_y>{};	
	static auto para_z	=	std::array<field_param,n_field_z>{};	
}//end of field

namespace pole
{
	static std::array<std::vector<double>,n_pole> para;

	static int check()noexcept
	{
		for(size_t i_pole=0ul;i_pole<n_pole;++i_pole)
		{
			if(para[i_pole].size()==0ul)
			{	
				printf("@O: uninitialized pole parameter detected.\n");return -233;
			}
		}
		return 	0;
	}
}//end of pole

namespace init//define the initial state (called by @a or called by @A)
{
	static_assert(calculate_init==0||(calculate_init==1&&init_ns>0ul)||(calculate_init==2&&n_pole>0ul&&n_field_x+n_field_y+n_field_z==0ul&&para_td>0.));

	using	comp_t		=	qpc::avx128_complex;
	static auto im		=	std::array<size_t,init_ns>{};
	static auto il		=	std::array<size_t,init_ns>{};
	static auto in		=	std::array<size_t,init_ns>{};
	static auto cf		=	std::array<comp_t,init_ns>{};

	static auto el		=	std::array<double,init_ns>{};
	static auto eh		=	std::array<double,init_ns>{};

	static double eitp = 999999.;

	static int initialize_a	(int narg, char** argv)noexcept//init by (is,im,il,in,re[,im])
	{
		if(narg!=5&&narg!=6){printf("@a: invalid number of arguments=%d.\n",narg);return -6;}
	
		size_t is	=	string_to<size_t>(argv[1]);
		if(is>=init_ns){printf("@a: invalid index %zu.\n",is);return -7;}

		if(argv[2][0]=='z')
		{
			im[is]	=	n_mmin<0?size_t(-n_mmin):0ul;
		}else 	im[is]	=	string_to<size_t>(argv[2]);

		{	
			il[is]	=	string_to<size_t>(argv[3]);
			in[is]	=	string_to<size_t>(argv[4]);
		}
		if(narg==5)
		{
			if(argv[5][0]=='o')	cf[is] = comp_t{1.0,0.0};
			else if(argv[5][0]=='i')cf[is] = comp_t{0.0,1.0};
			else 			cf[is] = comp_t{string_to<double>(argv[5]),0.0};
		}else
		{
			cf[is][0]=	string_to<double>(argv[5]);
			cf[is][1]=	string_to<double>(argv[6]);
		}
		return	0;
	}//end of initialize (@a)

	static int initialize_A	(int narg,char** argv)noexcept//init by (is,el,eh,id,re[,im])
	{
		if(narg!=5&&narg!=6){printf("@A: invalid number of arguments=%d.\n",narg);return -6;}

		size_t is	=	string_to<size_t>(argv[1]);
		if(is>=init_ns){printf("@A: invalid index %zu.\n",is);return -7;}
	
		el[is]	=	string_to<double>(argv[2]);
		eh[is]	=	string_to<double>(argv[3]);
		in[is]	=	string_to<size_t>(argv[4]);
			
		if(narg==5)
		{
			if(argv[5][0]=='o')	cf[is] = comp_t{1.0,0.0};
			else if(argv[5][0]=='i')cf[is] = comp_t{0.0,1.0};
			else 			cf[is] = comp_t{string_to<double>(argv[5]),0.0};
		}else
		{
			cf[is][0]=	string_to<double>(argv[5]);
			cf[is][1]=	string_to<double>(argv[6]);
		}
		return	0;	
	}//end of initialize (@A)

	template<class coef_t>
        static int load_initial (hdf5_file& file, coef_t& coef)
        {
                auto    temp    =       std::vector<comp_t>();
                size_t  n_dims_temp;
                size_t  n_diag_temp;
                size_t  l_dims_temp;
                size_t  m_dims_temp;
                long    m_init_temp;
                double  rmin_temp, rmax_temp;

                file.load(&n_dims_temp,1ul,"/nr");
                file.load(&n_diag_temp,1ul,"/mr");
                file.load(&l_dims_temp,1ul,"/nl");
                file.load(&m_dims_temp,1ul,"/nm");
                file.load(&m_init_temp,1ul,"/m0");

                file.load(&rmin_temp,1ul,"/rmin");
                file.load(&rmax_temp,1ul,"/rmax");

                if(n_dims_temp!=n_ndim || para_rmin!=rmin_temp || para_rmax!=rmax_temp)return -1;
                if(n_diag_temp!=n_rank)return -2;

                if(m_init_temp<n_mmin || m_init_temp+long(m_dims_temp)> n_mmin+long(n_mdim))return -3;

                if(l_dims_temp> n_ldim)return -4;

                coef.fillzero();

                temp.resize(n_dims_temp*l_dims_temp*m_dims_temp);
                file.load((double*)temp.data(),temp.size()*2ul,"/coef1");

                size_t mdiff = size_t(n_mmin - m_init_temp);
                for(size_t im=0;im<m_dims_temp;++im)
                for(size_t il=0;il<l_dims_temp;++il)
                for(size_t ir=0;ir<n_dims_temp;++ir)
                {
                        size_t jm = im +  mdiff;
                        coef(jm,il,ir) = temp[im*l_dims_temp*n_dims_temp + il*n_dims_temp + ir];
                }
                return 0;
        }//end of load_initial
}//end of init

namespace util
{
	static_assert(to_save_field==0||to_save_field==1);
	static_assert(para_rmin==0.0 && para_rmax >0.0 && n_ldim>=1ul);

	template<class file_t>	
	static void save_basic	(file_t& file)noexcept
	{
                file.save(&dims_t::n_dims,1ul,"/nr");
                file.save(&dims_t::l_dims,1ul,"/nl");
                file.save(&dims_t::m_dims,1ul,"/nm");
		file.save(&dims_t::m_init,1ul,"/m0");
                file.save(&dims_t::n_diag,1ul,"/mr");
		file.save(&para_rmin,1ul,"/rmin");
		file.save(&para_rmax,1ul,"/rmax");

		if constexpr(n_pole>0ul)
		{
			file.save(n_knot.data(),n_knot.size(),"/n_knot");
			for(size_t i=0;i<n_pole;++i)
			file.save(pole::para[i].data(),pole::para[i].size(),string("/pole_"+to_string(i)).c_str());
		}
		if constexpr(n_field_x>0ul)
		{
			for(size_t i=0;i<n_field_x;++i)
			file.save(fpara::para_x[i].data(),fpara::para_x[i].size(),string("/fieldx_"+to_string(i)).c_str());
		}
		if constexpr(n_field_y>0ul)
		{
			for(size_t i=0;i<n_field_y;++i)
			file.save(fpara::para_y[i].data(),fpara::para_y[i].size(),string("/fieldy_"+to_string(i)).c_str());
		}
		if constexpr(n_field_z>0ul)
		{
			for(size_t i=0;i<n_field_z;++i)
			file.save(fpara::para_z[i].data(),fpara::para_z[i].size(),string("/fieldz_"+to_string(i)).c_str());
		}
	}
}//end of util

namespace knot
{
        template<class radi_t>
        static void save_knots (hdf5_file& file, radi_t& radi)noexcept
        {
                file.save(radi.base.knot(),radi.base.size()+1ul,"/knot_sequence");
        }//end of save_knots
}//end of knot

namespace disp
{
	static_assert(calculate_coef==0||calculate_coef==1);
	static_assert(calculate_disp==0||calculate_disp==1);

	static_assert(disp_rmin>=para_rmin+1e-8);//avoid 0/0 error
	static_assert(disp_rmax<=para_rmax);
	static_assert(disp_thmin>=0.0&&disp_thmax<=PI<double>*1.0);
	static_assert(disp_phmin>=0.0&&disp_phmax<=PI<double>*2.0);
	static_assert(disp_nr>0ul);
	static_assert(disp_nth>0ul);
	static_assert(disp_nph>0ul);

	static spherical::spectrum_x<dims_t>* disp = 0;

	template<class radi_t,class angu_t> 	
	static void initialize(radi_t& radi,angu_t& angu)
	{
		if constexpr(calculate_disp)
		{
			disp = new spherical::spectrum_x<dims_t>();
                	disp->initialize
			(
				radi,
				disp_rmin,disp_rmax,disp_nr,
				disp_thmin,disp_thmax,disp_nth,
				disp_phmin,disp_phmax,disp_nph
			);
		}
	}

	template<class file_t,class coef_t>
	static void save_disp	(file_t& file,coef_t& coef,const string suffix="1")
	{
		if constexpr(calculate_disp)
		{
			disp->accumulate(coef,n_core);

			auto name = "/wave" + suffix;
	
               		file.save(disp->dptr(),2ul*disp->size(),name.c_str());
		}
		if constexpr(calculate_coef)
		{
			auto name = "/coef" + suffix;

			file.save(coef. dptr(),2ul*coef. size(),name.c_str());
		}
	}//end of save_disp

	template<class file_t>	
	static void save_basic	(file_t& file)noexcept
	{
		if constexpr(calculate_disp)
		{
                        file.save(&disp_rmin ,1ul,"/disp_rmin");
                        file.save(&disp_rmax ,1ul,"/disp_rmax");
                        file.save(&disp_thmin,1ul,"/disp_thmin");
                        file.save(&disp_thmax,1ul,"/disp_thmax");
                        file.save(&disp_phmin,1ul,"/disp_phmin");
                        file.save(&disp_phmax,1ul,"/disp_phmax");
                        file.save(&disp_nr   ,1ul,"/disp_nr");
                        file.save(&disp_nth  ,1ul,"/disp_nt");
                        file.save(&disp_nph  ,1ul,"/disp_np");	
		}
	}//end of save_basic	

	static void exit	()noexcept
	{
		if(disp){delete disp;disp=0;}
	}
}//end of disp

namespace spec
{
	static_assert(calculate_spec==0||calculate_spec==1||calculate_spec==2||calculate_spec==3);
	static_assert(!(gauge==1 && calculate_tsurff),"length gauge may not use t-SURFF.");

	static_assert(spec_thmin>=0.0&&spec_thmax<=PI<double>*1.0);
	static_assert(spec_phmin>=0.0&&spec_phmax<=PI<double>*2.0);
	static_assert(spec_nkr>0ul);
	static_assert(spec_nkt>0ul);
	static_assert(spec_nkp>0ul);

	static_assert(calculate_tsurff==0 || tsurff_r0<para_rmax);

	static_assert(spec_axis_type==0||spec_axis_type==1);

	static spherical::spectrum_c_pcs<dims_t,spec_nkr,spec_nkt,spec_nkp>* pcs = 0;
	static spherical::spectrum_c_tsf<dims_t,spec_nkr,spec_nkt,spec_nkp>* tsf = 0;

	//parameters of PCS choice
	static constexpr auto __pcs_advise	=	[](auto const& _potn)
	{
		if constexpr(n_pole>0ul)//General Recommended Way: analytical PCS use rb=40, zc=0 (i.e. planewave projection)
		{
			return	std::make_tuple(0.0, tsurff_r0, true);
		}else
		if constexpr(compare_v<decltype(_potn),spherical::hydrogenic>)//Recommended Way: analytical PCS, use rb=0.0, zc = para_zc
		{
			return	std::make_tuple(_potn.zc,0.0, true);
		}else
		if constexpr(compare_v<decltype(_potn),spherical::hydrogenic_yukawa>)//Recommended Way: numerical PCS, ditto
		{
			return	std::make_tuple(_potn.zc,0.0, false);
		}else
		if constexpr(compare_v<decltype(_potn),spherical::hydrogenic_xmtong>)//Recommended Way: numerical PCS, ditto
		{
			return	std::make_tuple(_potn.zc,0.0, false);
		}else
		{
			return	std::make_tuple(para_zasy , para_rasy, false);
		}		
	}(potn);

	static constexpr double pcs_zc 	=	std::get<0>(__pcs_advise);	//specify zc, if we do analytical PCS; specify zasy otherwise
	static constexpr double pcs_rb 	=	std::get<1>(__pcs_advise);	//specify rb, if we do analytical PCS; specify rasy otherwise
	static constexpr bool	pcs_fg	=	std::get<2>(__pcs_advise);	//specify whether we shall do analytical or numerical PCS

	template<class radi_t,class angu_t>
	static void initialize	(radi_t& radi, angu_t& angu)noexcept
	{
		if constexpr(calculate_spec)
		{
			pcs = new spherical::spectrum_c_pcs<dims_t,spec_nkr,spec_nkt,spec_nkp>();

			if constexpr(pcs_fg)
			{//use analytical PCS
				pcs->initialize(radi,spec_krmin,spec_krmax,spec_thmin,spec_thmax,spec_phmin,spec_phmax,pcs_zc,pcs_rb,spec_axis_type);
			}else
			{//use numerical PCS
				auto odef = spherical::generate_odefun_from_potential(potn,para_me,pcs_zc,pcs_rb);
				pcs->initialize(radi,spec_krmin,spec_krmax,spec_thmin,spec_thmax,spec_phmin,spec_phmax,odef,spec_axis_type);
			}	
		}

		if constexpr(calculate_tsurff)
		{
			tsf = new spherical::spectrum_c_tsf<dims_t,spec_nkr,spec_nkt,spec_nkp>();
			tsf->initialize(radi,angu,tsurff_r0,tsurff_zc,spec_krmin,spec_krmax,spec_thmin,spec_thmax,spec_phmin,spec_phmax,spec_axis_type);
		}
	}//end of initialize

	template<class...args_t>
	static void accumulate	(args_t&&...args)noexcept//call me at each step during propagation if you enabled tsurff
	{
		if constexpr(calculate_tsurff)
		{
			tsf->accumulate(std::forward<args_t>(args)...);		
		}
	}//end of accumulate

	template<class file_t,class coef_t>
	static void save_spec	(file_t& file,coef_t& coef)noexcept
	{
		if constexpr(calculate_spec&2)//check bit1
		{
			pcs->measure_lmd(coef,n_core);
			file.save(pcs->dptr(),pcs->size()*2ul,"/lmd1");
		}
		if constexpr(calculate_spec&1)//check bit0
		{
                	pcs->measure_prj(coef,n_core);
			file.save(pcs->dptr(),pcs->size()*2ul,"/pmd1");
		}	
		if constexpr(calculate_tsurff)
		{
			file.save(tsf->dptr(),tsf->size()*2ul,"/pmd2");
		}
	}//end of save_spec

	template<class file_t>
	static void save_basic	(file_t& file)noexcept
	{
		if constexpr(calculate_spec)
		{
                        file.save(pcs->axis_kr(),spec_nkr,"/pr");
                        file.save(pcs->axis_kt(),spec_nkt,"/pt");
                        file.save(pcs->axis_kp(),spec_nkp,"/pp");
		}
		if constexpr(calculate_tsurff)
		{
			if constexpr(!calculate_spec)
			{
				file.save(tsf->axis_kr(),spec_nkr,"/pr");
				file.save(tsf->axis_kt(),spec_nkt,"/pt");
				file.save(tsf->axis_kp(),spec_nkp,"/pp");
			}	
			file.save(&tsurff_zc,1ul,"/tsurff_zc");
			file.save(&tsurff_r0,1ul,"/tsurff_r0");
		}
	}//end of save_basic

	static void exit	()noexcept
	{
		if(pcs){delete pcs;pcs=0;}
		if(tsf){delete tsf;tsf=0;}
	}
}//end of spec

namespace proj
{
	static_assert(calculate_proj==0||calculate_proj==1||calculate_proj==2);

	static spherical::eigenstate_sph<dims_t>* proj = 0;

	static std::vector<comp_t> val;	
	static std::vector<comp_t> tmp;	
	static std::vector<size_t> nnz;

	template<class prop_t>
	static void initialize(prop_t& prop)noexcept
	{
		if constexpr(calculate_proj)
		{
			proj = new spherical::eigenstate_sph<dims_t>();
			proj->initialize(prop.s,prop.h,proj_filter);
		}
	}//end of initialize

	template<class coef_t>
	static void calc_proj	(coef_t& coef)
	{
		if constexpr(calculate_proj==2)
		{
			proj->projection(coef,tmp,nnz);
			for(size_t im=0;im<dims_t::m_dims;++im)
			for(size_t il=0;il<dims_t::l_dims;++il)
			{
				for(size_t ii=0;ii<nnz[im*dims_t::l_dims+il];++ii)
				val.push_back(tmp[im*dims_t::l_dims*dims_t::n_dims+il*dims_t::n_dims+ii]);
			}
		}
	}//end of calc_proj

	template<class file_t,class coef_t>
	static void save_proj	(file_t& file, coef_t& coef)noexcept
	{
		if constexpr(calculate_proj==1)
		{
			proj->projection(coef,val,nnz);//calculate projection onto eigen states
			file.save((double*)val.data(),val.size()*2ul,"/proj_val");
			file.save((size_t*)nnz.data(),nnz.size()    ,"/proj_nnz");
		}
		if constexpr(calculate_proj==2)
		{
			file.save((double*)val.data(),val.size()*2ul,"/proj_val");
			file.save((size_t*)nnz.data(),nnz.size()    ,"/proj_nnz");
			file.save(&proj_samp_rate,1ul,"/proj_samp_rate");
		}
	}//end of save_proj		
	
	static void exit	()noexcept
	{
		if(proj){delete proj;proj=0;}
	}
}//end of proj

namespace obsv
{
	static_assert(calculate_obsv_r==0||calculate_obsv_r==1);
	static_assert(calculate_obsv_z==0||calculate_obsv_z==1);

	static auto t		=	std::vector<double>();
	static auto x		=	std::vector<double>();
	static auto y		=	std::vector<double>();
	static auto z		=	std::vector<double>();
	static auto fx		=	std::vector<double>();
	static auto fy		=	std::vector<double>();
	static auto fz		=	std::vector<double>();

	static spherical::observable_z<dims_t>* obsv_z	= 0;
	static spherical::observable_r<dims_t>* obsv_r	= 0;

	template<class radi_t,class angu_t> 	
	static void initialize(radi_t& radi,angu_t& angu)noexcept
	{
		t.clear();
		 x.clear(); y.clear(); z.clear();
		fx.clear();fy.clear();fz.clear();
		if constexpr(calculate_obsv_z)
		{
			obsv_z	=	new spherical::observable_z<dims_t>();
			obsv_z	->	initialize(radi,angu,func_obsv_z);
		}

		if constexpr(calculate_obsv_r)
		{
			obsv_r 	=	new spherical::observable_r<dims_t>();
			obsv_r	->	initialize(radi,angu,func_obsv_r);
		}
	}

	template<class coef_t>
	static void observe	(coef_t& coef)noexcept
	{
		if constexpr(calculate_obsv_z)
		{
                	z.push_back(real(obsv_z->observe(coef,coef)));
		}
		if constexpr(calculate_obsv_r)
		{
			auto rm = obsv_r->observe_rm(coef,coef);	
			auto rp = obsv_r->observe_rp(coef,coef);
			x.push_back(real(rm+rp)/2.0);
			y.push_back(imag(rp-rm)/2.0);
		}
	}

	static void record	(double _t, double _x, double _y, double _z)noexcept
	{
		if constexpr(to_save_field||calculate_obsv_z||calculate_obsv_r)
		{
			t.push_back(_t);
		}
		if constexpr(to_save_field)
		{
			fx.push_back(_x);
			fy.push_back(_y);
			fz.push_back(_z);
		}
	}

	template<class file_t>	static void save_time	(file_t& file)noexcept
	{
		if(t.size())file.save(t.data(),t.size(),"/t_list");
	}
	template<class file_t>	static void save_field	(file_t& file)noexcept
	{
		file.save(&gauge,1ul,"gauge");
		if(fx.size())file.save(fx.data(),fx.size(),"/fx_list");
		if(fy.size())file.save(fy.data(),fy.size(),"/fy_list");
		if(fz.size())file.save(fz.data(),fz.size(),"/fz_list");
	}
	template<class file_t>	static void save_obsv	(file_t& file)noexcept
	{
		if(x.size())file.save(x.data(),x.size(),"/x_list");
		if(y.size())file.save(y.data(),y.size(),"/y_list");
		if(z.size())file.save(z.data(),z.size(),"/z_list");
	}
	static void exit()noexcept
	{
		if(obsv_z){delete obsv_z;obsv_z=0;}
		if(obsv_r){delete obsv_r;obsv_r=0;}
	}
}//end of obsv

namespace prop
{
	static_assert(calculate_prop==0||calculate_prop==1);
	static_assert(use_mask_function==0||use_mask_function==1);

	using	propagator_lp	=	qpc::select_t
	<
		gauge,
		qpc::select_cond<0, spherical::propagator_cn_hc_vgwz<dims_t> >,	//vg propagator
		qpc::select_cond<1, spherical::propagator_cn_hc_lgwz<dims_t> >	//lg propagator
	>;

	using	propagator_ep	=	qpc::select_t
	<
		gauge,
		qpc::select_cond<0, spherical::propagator_cn_hc_vgwr<dims_t> >,	//vg propagator
		qpc::select_cond<1, spherical::propagator_cn_hc_lgwr<dims_t> >	//lg propagator
	>;

	using	propagator_oce_lp	=	qpc::select_t
	<
		gauge,
		qpc::select_cond<0, spherical::propagator_cn_vgwz<dims_t> >,	//vg propagator
		qpc::select_cond<1, spherical::propagator_cn_lgwz<dims_t> >		//lg propagator
	>;

	using	propagator_oce_ep	=	qpc::select_t
	<
		gauge,
		qpc::select_cond<0, spherical::propagator_cn_vgwr<dims_t> >,	//vg propagator
		qpc::select_cond<1, spherical::propagator_cn_lgwr<dims_t> >		//lg propagator
	>;

	static_assert(propflag_odd||propflag_eve,"at least one subspace should be propagated.");

	static constexpr int isub = []() //isub only works for ep
	{
		if(propflag_odd)
		{
			if(propflag_eve) return -1; //both
			else return 3;	
		}else
		{	
			if(propflag_eve) return 2;
			return -1;	 //already avoided
		}
	}();
}//end of prop
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++//
namespace ui//define the way to print parameter
{
	static void print_param	()noexcept
	{
		printf( "Nr \tM  \tNl \tNm \tm0 \tNcore\tNpole\n"
		        "%zu\t%zu\t%zu\t%zu\t%ld\t%zu\t%zu\n"
			"min(r)\t\tmax(r)\t\tdt\t\ttd\n"
			"%.4e\t%.4e\t%.3e\t%.3e\n\n",
		n_ndim,n_rank,n_ldim,n_mdim,n_mmin,n_core,n_pole,para_rmin,para_rmax,para_dt,para_td);
		if constexpr(n_pole>0)
		{
			for(size_t i=0;i<n_pole;++i)
			{
				auto const& _para = pole::para[i];
				printf("%zu-th pole, R=%.15e:\n"
				       "Ze \t\tth(rad) \tph(rad) \n",i,_para[0]);
				for(size_t j=1;j<_para.size();j+=3)
				printf("%.6e\t%.6e\t%.6e\n",_para[j],_para[j+1],_para[j+2]);
			}printf("\n");
		}
	}//end of print_param

	static void print_laser ()noexcept
	{
		printf("NFx=%zu,NFy=%zu,NFz=%zu,gauge=%d\n",
		n_field_x,n_field_y,n_field_z,gauge);
		for(size_t ix=0;ix<n_field_x;++ix)
		{
			printf("Parameter of Fx%zu:\n"
			"para[0] \tpara[1] \tpara[2] \tpara[3] \tpara[4] \t\n"
			"%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t\n",ix,
			fpara::para_x[ix][0],
			fpara::para_x[ix][1],
			fpara::para_x[ix][2],
			fpara::para_x[ix][3],
			fpara::para_x[ix][4]);
		}
		for(size_t iy=0;iy<n_field_y;++iy)
		{
			printf("Parameter of Fy%zu:\n"
			"para[0] \tpara[1] \tpara[2] \tpara[3] \tpara[4] \t\n"
			"%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t\n",iy,
			fpara::para_y[iy][0],
			fpara::para_y[iy][1],
			fpara::para_y[iy][2],
			fpara::para_y[iy][3],
			fpara::para_y[iy][4]);
		}
		for(size_t iz=0;iz<n_field_z;++iz)
		{
			printf("Parameter of Fz%zu:\n"
			"para[0] \tpara[1] \tpara[2] \tpara[3] \tpara[4] \t\n"
			"%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t\n",iz,
			fpara::para_z[iz][0],
			fpara::para_z[iz][1],
			fpara::para_z[iz][2],
			fpara::para_z[iz][3],
			fpara::para_z[iz][4]);
		}printf("\n");
	}//end of print_laser
	
	static void print_inita(std::array<double,init_ns> const& eng)noexcept
	{
		printf("Report from @a:\nim \til \tin \tE \t\n");
		for(size_t i=0;i<init_ns;++i)
		{
			printf("%zu \t%zu \t%zu \t%.16lf\n",
			init::im[i],init::il[i],init::in[i],eng[i]);
		}
		printf("\n");
	}//end of print_inita

	static void print_initA(std::array<double,init_ns> const& eng)noexcept
	{
		printf("Report from @A:\nid \t\tEmin \t\tEmax \t\tE \n");
		for(size_t i=0;i<init_ns;++i)
		{
			printf("%zu \t\t%.7e \t%.7e \t%.16e\n",
			init::in[i],init::el[i],init::eh[i],eng[i]);
		}
		printf("\n");
	}//end of print_initA

	static void print_mannual()noexcept
	{
		printf("<executable> <job1> [arg1...] <job2> [arg2...] ...\n")	;

		printf("\t'@L' = realtime propagation with LP(z)   dipole laser and centrifugal potential.\n");
		printf("\t'@E' = realtime propagation with EP(x,y) dipole laser and centrifugal potential.\n");
		printf("\t'@O' = realtime propagation with arb-pol dipole laser and OCE expansion potential.\n");
		printf("\t\t format: @L/@E/@O [load_path] [save_path]. By default they are %s,%s.\n",default_load_path,default_save_path);

		printf("\t'@F' = set laser field parameters.\n");
		printf("\t\t format: @F {X/Y/Z}{id1}{id2} <val>. X,Y,Z specifies direction, id1 is the index of laser, id2 is the index of param, val is value.\n");
		printf("\t\t current laser parameters: 0=F0(a.u.) 1=w0(a.u.) 2=Nc(o.c) 3=Nm(o.c.) 4=Cep(pi).\n");

		printf("\t'@a' = set initial states parameters by index.\n");
		printf("\t\t format: @a <is> <im> <il> <in> <re> <im>.\n");
		printf("\t\t special character for <im>: 'z' =  m set to 0;\n");
		printf("\t\t special character for <cf>: 'o' =  c set to 1.0, 'i'= c set to 1.0i;\n");

		printf("\t'@A' = set initial states parameters by energy.\n");
		printf("\t\t format: @A <is> <El> <Eh> <id> <re> <im>.\n");
		printf("\t\t special character for <cf>: 'o' =  c set to 1.0, 'i'= c set to 1.0i;\n");

		printf("\t'@P' = set pure coulomb pole parameters. Only invokable if n_pole>0.\n");
		printf("\t\t format: @Pn <Rn> <Zc1> <th1> <ph1> [<zc2> <th2> <ph2> ...].\n");
	}//end of print_mannual
}//end of ui
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++//

void	realtime_lp	(const char* load_path,const char* save_path)//end of arg list
{
	//workhorse object
    	auto    radi 	=	spherical::integrator_radi<dims_t>();
	auto	angu	=	spherical::integrator_angu<dims_t>();

	auto	coef	=	spherical::coefficient<dims_t>();
	auto	prop	=	typename prop::propagator_lp();

	auto	laser	=	create_laser_field<envelope_type>(fpara::para_x,fpara::para_y,fpara::para_z,para_dt);

    	auto* 	mask    = 	(spherical::absorber_mask<dims_t>*)0;

	radi.initialize(para_rmin,para_rmax);
	prop.initialize(n_core,radi,angu,para_me,potn,para_dt);
	
	size_t	nt_total	=	laser.para_nt+para_td/para_dt;

	disp::initialize(radi,angu);
	obsv::initialize(radi,angu);
	spec::initialize(radi,angu);

	proj::initialize(prop);
	
	if constexpr(use_mask_function)
        {
                mask    =       new spherical::absorber_mask<dims_t>(mask_nthd);
                mask    ->      initialize(radi,mask_rmin,mask_rmax,mask_fact);
        }
	if constexpr(calculate_init==0)
	{
		hdf5_file file;
		if(file.open(load_path,'r',!run_silently))
		{
			return;
		}
		if(int err=init::load_initial(file,coef))
		{
			printf("fail to load initial state, errorcode = %d\n",err);return;	
		}
	}else
	{
		auto 	eng	=	coef.initialize<init_ns>
		(
			prop.s,prop.h,
			init::im,init::il,
			init::in,init::cf
		);//im,il,ir,cf
		if constexpr(!run_silently)ui::print_inita(eng);
	}
	//save initial state
	
	//do propagation
	if constexpr(calculate_prop)
	{
		auto    clock   =       timer();
		clock.tic();
		for(size_t it=0;it<nt_total;++it)
		{
			//calculate laser field
			auto	ti	=	laser.t(it);
			auto	tf	=	laser.t(it+1);
			auto [fx,fy,fz]	=	laser(ti);
			auto [gx,gy,gz] = 	laser(tf);
			//calculate observables
			obsv::observe(coef);
			obsv::record(ti,fx,fy,fz);
			//propagation
			if(it<laser.para_nt)
			{
				prop.propagate(coef,fz,gz);
				spec::accumulate(fz,gz,para_dt,coef);//note: only velocity gauge can use T-SURFF
			}else
			{
				prop.propagate(coef);
				spec::accumulate(para_dt,coef);//note: only velocity gauge can use T-SURFF
			}
			//apply mask function
			if constexpr(use_mask_function)
			{
				(*mask)(coef,prop.s);
			}

			//print info
			if constexpr(!run_silently)
			if(it%print_rate==0)
			{
				clock.toc();
				printf( "t/tf \t\t| it/Nt \t| Fz \t\t| elapsed time\n"
					"%.3e \t| %-9zu\t| %.6lf\t| %.3lf\n"
					"%.3e \t| %-9zu\n",
				ti,it,fz,clock.get(),laser.para_tf,nt_total);
				//clock.tic();
				fflush(stdout);
			}
			
			//compute projection
			if constexpr(calculate_proj==2)
			if(it%proj_samp_rate==0)
			{
				proj::calc_proj(coef);
			}
		}//end*/
		clock.toc();
		printf("%lf seconds in total\n",clock.get());
	}//end of propagation

	//save final results
	if constexpr(true)
        {
		hdf5_file file;
		if(file.open(save_path,'w',!run_silently))return;
		
		file.save(&laser.para_dt,1ul,"/para_dt");
		file.save(&laser.para_nt,1ul,"/para_nt");

		util::save_basic(file);		
		disp::save_basic(file);
		spec::save_basic(file);

		knot::save_knots(file,radi);

		disp::save_disp(file,coef);
		spec::save_spec(file,coef);

		proj::save_proj(file,coef);
	
		obsv::save_time(file);
		obsv::save_obsv(file);
		obsv::save_field(file);

		disp::exit();
		spec::exit();
		proj::exit();
		obsv::exit();
	}//end of save
	if constexpr(use_mask_function)delete mask;
}//end of realtime_lp

void	realtime_ep	(const char* load_path,const char* save_path)
{
	//workhorse objects
	
	auto	radi    =       spherical::integrator_radi<dims_t>();
	auto	angu    =       spherical::integrator_angu<dims_t>();

	auto	coef    =       spherical::coefficient<dims_t>();
	auto	prop	=	typename prop::propagator_ep();

	auto	laser	=	create_laser_field<envelope_type>(fpara::para_x,fpara::para_y,fpara::para_z,para_dt);

    	auto* 	mask    =       (spherical::absorber_mask<dims_t>*)0;
	
	radi.initialize(para_rmin,para_rmax);
	prop.initialize(n_core,radi,angu,para_me,potn,para_dt);

	size_t  nt_total=       laser.para_nt+para_td/para_dt;
	
	disp::initialize(radi,angu);
	obsv::initialize(radi,angu);
	spec::initialize(radi,angu);

	proj::initialize(prop);

	if constexpr(use_mask_function)
        {
                mask    =       new spherical::absorber_mask<dims_t>(mask_nthd);
                mask    ->      initialize(radi,mask_rmin,mask_rmax,mask_fact);
        }
	if constexpr(calculate_init==0)
	{
		hdf5_file file;
		if(file.open(load_path,'r',!run_silently))
		{
			return;
		}
		if(int err=init::load_initial(file,coef))
		{
			printf("fail to load initial state, errorcode = %d\n",err);return;	
		}
	}else
	{
		auto 	eng	=	coef.initialize<init_ns>
		(
			prop.s,prop.h,
			init::im,init::il,
			init::in,init::cf
		);//im,il,ir,cf
		if constexpr(!run_silently)ui::print_inita(eng);
	}

	//do propagation
	if constexpr(calculate_prop)
	{
		auto	clock	=	qpc::timer();
		clock.tic();
	
		for(size_t it=0;it<nt_total;++it)
		{
			//calculate laser field
			auto ti		=	laser.t(it);
			auto tf		=	laser.t(it+1);
			auto [fx,fy,fz]	=	laser(ti);
			auto [gx,gy,gz]	=	laser(tf);
			//calculate observables
			obsv::observe(coef);
			obsv::record(ti,fx,fy,fz);
			//propagation
			if(it<laser.para_nt)
			{
				prop.propagate<prop::isub>(coef,fx,gx,fy,gy);
				spec::accumulate(fx,gx,fy,gy,para_dt,coef);//note: only velocity gauge can use me
			}else	
			{
				prop.propagate(coef);
				spec::accumulate(para_dt,coef);//note: only velocity gauge can use me
			}
			//apply mask function
			if constexpr(use_mask_function)
			{
				(*mask)(coef,prop.s);
			}
			//print info
			if constexpr(!run_silently)
			if(it%print_rate==0)
			{
				clock.toc();
				printf( "t/tf \t\t| it/Nt \t| Fx/Fy \t| elapsed time\n"
					"%.3e \t| %-9zu\t| %.6lf\t| %.3lf\n"
					"%.3e \t| %-9zu\t| %.6lf\t\n",
				ti,it,fx,clock.get(),laser.para_tf,nt_total,fy);
				//clock.tic();
				fflush(stdout);
			}
			
			//compute projection
			if constexpr(calculate_proj==2)
			if(it%proj_samp_rate==0)
			{
				proj::calc_proj(coef);
			}
		}
		clock.toc();printf("%lf seconds in total\n",clock.get());
	}//end of propagation
	
	//save final results
	if constexpr(true)
        {
		hdf5_file file;
		if(file.open(save_path,'w',!run_silently))return;

		file.save(&laser.para_dt,1ul,"/para_dt");
		file.save(&laser.para_nt,1ul,"/para_nt");

		util::save_basic(file);		
		disp::save_basic(file);
		spec::save_basic(file);

		knot::save_knots(file,radi);

		disp::save_disp(file,coef);
		spec::save_spec(file,coef);

		proj::save_proj(file,coef);
	
		obsv::save_time(file);
		obsv::save_obsv(file);
		obsv::save_field(file);

		disp::exit();
		spec::exit();
		proj::exit();
		obsv::exit();
	}//end of save
	if constexpr(use_mask_function)delete mask;
}//end of realtime_ep

void	realtime_oce	(const char* load_path,const char* save_path)
{
	bool constexpr enable_wz	=	n_field_z>0ul;
	bool constexpr enable_wr	=	n_field_x>0ul || n_field_y>0ul;

	auto	radi	=	spherical::integrator_radi<dims_t>();
	auto	angu	=	spherical::integrator_angu<dims_t>();
	
	auto	coef	=	spherical::coefficient<dims_t>();	

	auto	prop_hm	=	spherical::propagator_cn_hm<dims_t>();	//oce field-free cn
	
	auto*	prop_wz	=	(prop::propagator_oce_lp*)0;
	auto*	prop_wr	=	(prop::propagator_oce_ep*)0;
	
	auto	laser	=	create_laser_field<envelope_type>(fpara::para_x,fpara::para_y,fpara::para_z,para_dt);

    	auto* 	mask    =       (spherical::absorber_mask<dims_t>*)0;

	{
		auto	pole_r	=	[&]()
		{
			auto	result = std::array<double,n_pole>();
			for(size_t i=0;i<n_pole;++i)
			{
				result[i] = pole::para[i][0];
			}return	result;
		}();
		radi.initialize(para_rmin,para_rmax,pole_r,n_knot);

		for(size_t i=0;i<n_pole;++i)
		{
			prop_hm.oper_h.set_pole(i,pole::para[i]);
		}
		if constexpr(calculate_init==2)
		{
			prop_hm.initialize(n_core,radi,angu,-unim<comp_t>*para_dt,para_me,potn);
		}else
		{
			prop_hm.initialize(n_core,radi,angu, idty<comp_t>*para_dt,para_me,potn);
		}
	}
	if constexpr(enable_wz)
	{
		prop_wz	=	new prop::propagator_oce_lp();
		prop_wz->	initialize(n_core,radi,angu,para_dt);
	}
	if constexpr(enable_wr)
	{
		prop_wr	=	new prop::propagator_oce_ep();
		prop_wr->	initialize(n_core,radi,angu,para_dt);
	}
	
	size_t	nt_total	=	laser.para_nt+para_td/para_dt;

	disp::initialize(radi,angu);
	obsv::initialize(radi,angu);
	spec::initialize(radi,angu);

	if constexpr(use_mask_function)
        {
                mask    =       new spherical::absorber_mask<dims_t>(mask_nthd);
                mask    ->      initialize(radi,mask_rmin,mask_rmax,mask_fact);
        }

	if constexpr(calculate_init==0)
	{
		hdf5_file file;
		if(file.open(load_path,'r',!run_silently))
		{
			return;
		}
		if(int err=init::load_initial(file,coef))
		{
			printf("fail to load initial state, errorcode = %d\n",err);return;	
		}
	}else
	if constexpr(calculate_init==1)
	{
		auto 	eng	=	coef.template initialize<init_ns,16>//
		(
			prop_hm.oper_s,prop_hm.oper_h,
			init::in,init::el,init::eh,init::cf
		);

		if constexpr(!run_silently)ui::print_initA(eng);
	}else
	if constexpr(calculate_init==2)
	{
		size_t im = dims_t::m0<0? size_t(-dims_t::m0):0ul;
		coef(im,0,0) = 1.0;
	}
	
	if constexpr(calculate_prop)
	{
		auto	clock	=	qpc::timer();
		clock.tic();
		for(size_t it=0;it<nt_total;++it)
		{
			//calculate laser field
			auto ti		=	laser.t(it);
			auto tf		=	laser.t(it+1);
			auto [fx,fy,fz]	=	laser(ti);
			auto [gx,gy,gz]	=	laser(tf);
			//calculate observables
			obsv::observe(coef);
			obsv::record(ti,fx,fy,fz);
			//propagation
			if(it<laser.para_nt)
			{
				if constexpr(enable_wr)
				prop_wr->propagate_fwd(coef,fx,fy);
				if constexpr(enable_wz)
				prop_wz->propagate_fwd(coef,fz); 
				prop_hm.propagate(coef);
				if constexpr(enable_wz)
				prop_wz->propagate_bwd(coef,gz);
				if constexpr(enable_wr)
				prop_wr->propagate_bwd(coef,gx,gy);
				//calculate spectrum
				if constexpr(enable_wr&&!enable_wz)
				{	
					spec::accumulate(fx,gx,fy,gy,para_dt,coef);//note: only velocity gauge can use me
				}else 
				if constexpr(enable_wz&&!enable_wr)
				{
					spec::accumulate(fz,gz,para_dt,coef);
				}
			}else
			{
				prop_hm.propagate(coef);
				spec::accumulate(para_dt,coef);
				if constexpr(calculate_init==2)
				{
					double _renorm = coef.normalize(prop_hm.oper_s);
					double _enow   = (_renorm-1.0)/(_renorm+1.0)/(0.5*para_dt);
					if(fabs(_enow/init::eitp -1.0)<1e-13)
					{
						printf("ITP convergence detected, E=%.13e\n",_enow);break;
					}else init::eitp = _enow;
				}
			}

			//apply mask function
			if constexpr(use_mask_function)
			{
				(*mask)(coef,prop_hm.oper_s);
			}
			//print info
			if constexpr(!run_silently)
			if(it%print_rate==0)
			{
				clock.toc();
				printf( "t/tf \t\t| it/Nt \t| Fx/Fy \t| Fz/elapsed time\n"
					"%.3e \t| %-9zu\t| %.6lf\t| %.6lf\n"
					"%.3e \t| %-9zu\t| %.6lf\t| %.3lf\n",
				ti,it,fx,fz,laser.para_tf,nt_total,fy,clock.get());
				//clock.tic();
				fflush(stdout);
			}
			
		}
	}	
	//save final results
	if constexpr(true)
        {
		hdf5_file file;
		if(file.open(save_path,'w',!run_silently))return;

		file.save(&laser.para_dt,1ul,"/para_dt");
		file.save(&laser.para_nt,1ul,"/para_nt");

		util::save_basic(file);		
		disp::save_basic(file);
		spec::save_basic(file);

		knot::save_knots(file,radi);

		disp::save_disp(file,coef);
		spec::save_spec(file,coef);

		//proj::save_proj(file,coef);
	
		obsv::save_time(file);
		obsv::save_obsv(file);
		obsv::save_field(file);

		disp::exit();
		spec::exit();
		obsv::exit();
	}//end of save
	if constexpr(enable_wz)delete prop_wz;
	if constexpr(enable_wr)delete prop_wr;
	if constexpr(use_mask_function)delete mask;
}//end of realtime_oce

int 	majors(int argc,char** argv)
{
	std::vector<char const*> load_path;
	std::vector<char const*> save_path;
	std::vector<char>	 func_flag;

	if(argc<=1||argv[1][0]!='@')
	{
		ui::print_mannual();return -1;
	}else
	{
		argc--;argv++;
		do
		{
			char	flag	=	argv[0][1];//"@?"
			size_t	narg	=	0ul;
	
			for(char** iter=argv+1;(iter<argv+argc)&&(iter[0][0]!='@');++iter)++narg;
			switch(flag)
			{
				case 'F'://laser
				{
					qpc::initialize_laser_parameter
					(
						narg,argv+1,
						fpara::para_x,
						fpara::para_y,
						fpara::para_z
					);
					narg++;//to include flag itself
					argv+=narg;
					argc-=narg;
				}break;
				case 'a'://init by index
				{
					if(int err=init::initialize_a(narg,argv))return err;
					narg++;
					argv+=narg;
					argc-=narg;
				}break;
				case 'A'://init by energy
				{
					if(int err=init::initialize_A(narg,argv))return err;
					narg++;
					argv+=narg;
					argc-=narg;
				}break;
				case 'P'://pole
				{
					if(int err=qpc::spherical::initialize_pole_parameter(narg+1,argv,pole::para))return err;
					narg++;
					argv+=narg;
					argc-=narg;
				}break;
				default:
				{
					func_flag.push_back(flag);
					argv++;argc--;
					if(narg>0){load_path.push_back(*argv);argv++;argc--;}else{load_path.push_back(default_load_path);}
					if(narg>1){save_path.push_back(*argv);argv++;argc--;}else{save_path.push_back(default_save_path);}
					if(narg>2){printf("too many arguments for flag %c!!\n",flag);return -2;}
				}
			}
		}while(argc);
	}	

	if(int err=pole::check())return err;

	if constexpr(!run_silently)
	{
		ui::print_param();
		ui::print_laser();
	}
	
	for(size_t i_task=0;i_task<func_flag.size();++i_task)
	{
		switch(func_flag[i_task])
		{
			case 'L'://LP(z) case
			{
				realtime_lp(load_path[i_task],save_path[i_task]);
			}break;
			case 'E'://EP(x,y) case
			{
				realtime_ep(load_path[i_task],save_path[i_task]);
			}break;
			case 'O'://OCE case
			{
				realtime_oce(load_path[i_task],save_path[i_task]);
			}break;
			default:
			{
				printf("invalid flag %c\n",func_flag[i_task]);
			}
		};
	}
	return	0;
}//end of majors

int main(int argc,char** argv)
{
	mkl_set_num_threads(n_core);
	omp_set_num_threads(n_core);

	int main_error_code=0;
	try
	{
		main_error_code = majors(argc,argv);
	}catch(const runtime_error_base_t& e)
	{
		e.what();
		main_error_code = -99999999;
	}
	return	main_error_code;
}//end of main
