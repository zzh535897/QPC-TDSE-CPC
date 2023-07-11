#pragma once
//==============================================================
//miscellaneous functions related to multipole expansion 
//==============================================================


template<class...args_t>
static 	inline	auto	create_poledata	(double r0,args_t&&...thph)noexcept
{
	using	pole_t		=	std::vector<double>;//[0] = rp, [1,2,3;4,5,6;...] = zc0,th0,ph0; zc1,th1,ph1;...
	return	pole_t{r0,std::forward<args_t>(thph)...};
}

//argv[0] must be "-Pn, n is an integer"
template<size_t n_pole>
inline	int	initialize_pole_parameter	
(
	int argc,char** argv,
	std::array<std::vector<double>,n_pole>& pole_para
)
{
	if(!argc)
	{
		printf("invalid number of arguments for pole initialization\n");return -1;
	}	
	if(argv[0][0]!='@'||argv[0][1]!='P')
	{
		printf("invalid flags for pole initialization:%s\n",argv[0]);return -2;
	}

	size_t i_pole = size_t(argv[0][2]-'0');
	if(i_pole>=n_pole)
	{
		printf("invalid pole index for pole initialization:%zu\n",i_pole);return -3;
	}

	auto&	target = pole_para[i_pole];
	if(target.size()>0ul)
	{
		printf("repeated initialization for %zu-th pole, size=%zu already\n",i_pole,target.size());return -4;
	}

	argc--;argv++;
	if(argc%3!=1)
	{
		printf("invalid number of arguments for pole initialization (%d). should be multiples of 3 plus 1.\n",argc);return -5;
	}

	double	rp	=	string_to<double>(*argv);
	target.push_back(rp);
	argc--;argv++;
	if(rp<=0.)
	{
		printf("invalid value of pole distance (%lf). should be a positive real number\n",rp);return -6;
	}

	while(argc)
	{
		double	zc = string_to<double>(argv[0]);
		double	th = string_to<double>(argv[1])*PI<double>;
		double	ph = string_to<double>(argv[2])*PI<double>;
		target.push_back(zc);
		target.push_back(th);
		target.push_back(ph);
		argc-=3;argv+=3;

		if(th<0.||th>PI<double>*1.00000000001)printf("warning: a non-conventional theta value (%lf) is being used.\n",th);
		if(ph<0.||ph>PI<double>*2.00000000001)printf("warning: a non-conventional phi value (%lf) is being used.",ph);
	}
	return 0;
}

template<>
inline	int	initialize_pole_parameter	
(
	int argc,char** argv,
	std::array<std::vector<double>,0>& pole_para
)
{
	printf("you may not initialize pole_para if n_pole are set to 0!\n");
	return	-1;
}

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
//(1) to count the number of non-zero terms in the 3Y-integral A(mili,λμ,mjlj)
//
// 	A(mili,λμ,mjlj) =  sqrt(4π/(2λ+1))*( Ylimi | Yλμ | Yljmj ) 
// 			= (-)^mi sqrt((2li+1)(2lj+1)) 
// 			* ({li,0}{λ,0}{lj,0})*({li,-mi}{λ,μ}{lj,+mj})
// 
// according to the selection rule			
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
template<long mmin,long mmax,size_t ldim,size_t qdim>
static constexpr size_t count_triangular1()noexcept
{
	size_t n=0;

	for(long mi=mmin;mi<mmax;++mi) 
	for(long mj=mmin;mj<mmax;++mj) 
	{
		long uabs=cabs(mi-mj); 
		long limin = cabs(mi);
		long ljmin = cabs(mj);
		long limax = limin+long(ldim);
		long ljmax = ljmin+long(ldim);
		for(long li=limin;li<limax;++li)
		for(long lj=ljmin;lj<ljmax;++lj)
		{
			auto [lmin,lmax]=cminmax(li,lj);
			long upper = cmin(long(qdim),lmax+lmin);
			long lower = cmax(     uabs ,lmax-lmin);
			if(upper>=lower)n+=size_t((upper-lower)/2+1);
		}
	}
	return n;
}//end of count_triangular1

template<long mmin,long mmax,size_t ldim,size_t qdim>
static inline size_t count_triangular2()noexcept
{
	size_t n=0;

	for(long q=0;q<long(qdim);++q)
	{
		for(long mi=mmin;mi<mmax;++mi) 
		for(long mj=mmin;mj<mmax;++mj) 
		{
			long u=mi-mj; 
			long limax = cabs(mi)+long(ldim);
			long ljmax = cabs(mj)+long(ldim);
			for(long li=cabs(mi);li<limax;++li)
			for(long lj=cabs(mj);lj<ljmax;++lj)
			{ 
				if( li+lj>=q && li+q>=lj && lj+q>=li)
				{
				double val	= gsl_sf_coupling_3j(2*li,2*q,2*lj,    0,  0,   0)
						* gsl_sf_coupling_3j(2*li,2*q,2*lj,-2*mi,2*u,2*mj);
				if(fabs(val)>1e-15)++n;}
			}
		}
	}

	return n;
	
}//end of count_triangular2
