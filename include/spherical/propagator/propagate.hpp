#pragma once

template<int symflag, class dims_t,class ifet_t>
inline	void	cn_propagate_kernel_avx1
(
	double const* smat,
	double const* rmat,
	ifet_t const  ifet,
	comp_t*	zsrc,		//zsrc := (S+aR)\(S-aR) * zsrc 
	comp_t* ytmp,
	comp_t*	utmp
)noexcept
{
	static_assert(symflag == 0 || symflag ==1, "only 0=symmetric r or 1=anti-symmetric r are acceptable.");
	static_assert(std::is_same_v<ifet_t,double> || std::is_same_v<ifet_t,comp_t>,"only double or comp_t are acceptable.");

	size_t constexpr n_dims = dims_t::n_dims;
	size_t constexpr n_diag = dims_t::n_diag;
	size_t constexpr n_elem = dims_t::n_elem;

	using a_type = comp_t;
	using b_type = comp_t;

	using oper_a = std::array<a_type,n_diag*2ul+1ul>;
	using oper_b = std::array<b_type,n_diag*2ul+1ul>;
	using ldlt_t = recvec<comp_t,recidx<n_dims,n_diag>>;

	oper_a a;//S-aR
	oper_b b;//S+aR
	ldlt_t u = {utmp};

	//solve L,U such that L*U=b, and assign y:=L\a*z simultaneously
	for(size_t i=0;i<n_dims;++i)
	{
		for(size_t j=1;j<n_elem;++j)//lower pannel
		{
			size_t k = i*n_elem - j*n_diag;
			size_t _j= n_diag-j;
			if(k>=n_dims*n_elem)//underflow of unsigned
			{
				a[_j] = zero<a_type>;
				b[_j] = zero<b_type>;	
			}else
			{
				double s = smat[k];
				double r = rmat[k];
				if constexpr(symflag==1){
				a[_j] = s + ifet * r; //anti-symmetric
				b[_j] = s - ifet * r; //anti-symmetric
				}else{
				a[_j] = s - ifet * r; //symmetric
				b[_j] = s + ifet * r;}//symmetric
			}
		}
		for(size_t j=0;j<n_elem;++j) //upper
		{
			size_t k = i*n_elem + j;
			double s = smat[k];
			double r = rmat[k];
			a[n_diag+j] = s - ifet * r;
			b[n_diag+j] = s + ifet * r;
		}
		
		size_t l = n_diag>i? n_diag-i:0;
		size_t h = n_dims-1-n_diag>i? 2*n_diag+1: n_diag+n_dims-i;	
			
		for(size_t j=l;j<n_diag;++j)
		for(size_t k=0;k<n_diag;++k)
		{
			b[j+k+1]-=b[j]*u(i+j-n_diag,k);
		}
		for(size_t j=0;j<n_diag;++j)
		{
			u(i,j)=b[j+n_diag+1]/b[n_diag];//ui
		}
		comp_t v=zero<comp_t>;
		for(size_t j=l;j<h;++j)
		{
			v+=a[j]*zsrc[i+j-n_diag];
		}
		for(size_t j=l;j<n_diag;++j)
		{
			v-=b[j]*ytmp[i+j-n_diag];
		}
		ytmp[i]=v/b[n_diag];//yi, XXX what if |b|=0
	}
	//solve x=U\y and store x into z
	for(size_t i=n_dims;i-->0ul;)
	{
		size_t jmax = std::min(n_elem,n_dims-i);
		comp_t v=zero<comp_t>;
		for(size_t j=1ul;j<jmax;++j)
		{
			v+=ytmp[i+j]*u(i,j-1ul);
		}
		zsrc[i]  = (ytmp[i] -= v);
	}
}//end of cn_propagate_kernel_avx1

#ifdef support_avx3
template<int symflag, class dims_t,class ifdt_t>
inline	void	cn_propagate_kernel_avx3
(
	double const* smat,
	double const* rmat,
	double const* zeig,
	ifdt_t const  ifdt,
	comp_t*	zsrc,		//zsrc{0:3} := (S+a{0:3}*R)\(S-a{0:3}*R) * zsrc{0:3}
	comp_v* ytmp,
	comp_v*	utmp
)noexcept
{
	static_assert(symflag == 0 || symflag ==1, "only 0=symmetric r or 1=anti-symmetric r are acceptable.");
	static_assert(std::is_same_v<ifdt_t,double> || std::is_same_v<ifdt_t,comp_t>,"only double or comp_t are acceptable.");

	size_t constexpr n_dims = dims_t::n_dims;
	size_t constexpr n_diag = dims_t::n_diag;
	size_t constexpr n_elem = dims_t::n_elem;

	__m512i constexpr vindex = {
		0ul*n_dims,0ul*n_dims+1ul,
		2ul*n_dims,2ul*n_dims+1ul,
		4ul*n_dims,4ul*n_dims+1ul,
		6ul*n_dims,6ul*n_dims+1ul};

	using oper_t = std::array<comp_v,n_diag*2ul+1ul>;
	using ldlt_t = recvec<comp_v,recidx<n_dims,n_diag>>;

	oper_t a,b;
	ldlt_t u = {utmp};

	ifdt_t ifet[4] = {ifdt*zeig[0],ifdt*zeig[1],ifdt*zeig[2],ifdt*zeig[3]};

	//solve L,U such that L*U=b, and assign y:=L\a*z simultaneously
	for(size_t i=0;i<n_dims;++i)
	{
		for(size_t j=1;j<n_elem;++j)//lower pannel
		{
			size_t k = i*n_elem - j*n_diag;
			size_t _j= n_diag-j;
			if(k>=n_dims*n_elem)//underflow of unsigned
			{
				a[_j] = zero<comp_v>;
				b[_j] = zero<comp_v>;
			}else
			{
				double s = smat[k];
				double r = rmat[k];
				if constexpr(symflag==1)//anti-symmetric
				{
					if constexpr(std::is_same_v<ifdt_t,double>)
					{
						a[_j][0] = s + ifet[0] * r;
						a[_j][1] = 0.0;
						a[_j][2] = s + ifet[1] * r;
						a[_j][3] = 0.0;
						a[_j][4] = s + ifet[2] * r;
						a[_j][5] = 0.0;
						a[_j][6] = s + ifet[3] * r;
						a[_j][7] = 0.0;
						b[_j][0] = s - ifet[0] * r;
						b[_j][1] = 0.0;
						b[_j][2] = s - ifet[1] * r;
						b[_j][3] = 0.0;
						b[_j][4] = s - ifet[2] * r;
						b[_j][5] = 0.0;
						b[_j][6] = s - ifet[3] * r;
						b[_j][7] = 0.0;
					}else
					{
						a[_j][0] = s + ifet[0][0] * r;
						a[_j][1] =     ifet[0][1] * r;
						a[_j][2] = s + ifet[1][0] * r;
						a[_j][3] =     ifet[1][1] * r;
						a[_j][4] = s + ifet[2][0] * r;
						a[_j][5] =     ifet[2][1] * r;
						a[_j][6] = s + ifet[3][0] * r;
						a[_j][7] =     ifet[3][1] * r;
						b[_j][0] = s - ifet[0][0] * r;
						b[_j][1] =   - ifet[0][1] * r;
						b[_j][2] = s - ifet[1][0] * r;
						b[_j][3] =   - ifet[1][1] * r;
						b[_j][4] = s - ifet[2][0] * r;
						b[_j][5] =   - ifet[2][1] * r;
						b[_j][6] = s - ifet[3][0] * r;
						b[_j][7] =   - ifet[3][1] * r;
					}
				}else//symmetric
				{
					if constexpr(std::is_same_v<ifdt_t,double>)
					{
						a[_j][0] = s - ifet[0] * r; 
						a[_j][1] = 0.0;
						a[_j][2] = s - ifet[1] * r; 
						a[_j][3] = 0.0;
						a[_j][4] = s - ifet[2] * r; 
						a[_j][5] = 0.0;
						a[_j][6] = s - ifet[3] * r; 
						a[_j][6] = 0.0;
						b[_j][0] = s + ifet[0] * r; 
						b[_j][1] = 0.0;
						b[_j][2] = s + ifet[1] * r; 
						b[_j][3] = 0.0;
						b[_j][4] = s + ifet[2] * r; 
						b[_j][5] = 0.0;
						b[_j][6] = s + ifet[3] * r;
						b[_j][7] = 0.0;
					}else
					{
						a[_j][0] = s - ifet[0][0] * r;
						a[_j][1] =   - ifet[0][1] * r;
						a[_j][2] = s - ifet[1][0] * r;
						a[_j][3] =   - ifet[1][1] * r;
						a[_j][4] = s - ifet[2][0] * r;
						a[_j][5] =   - ifet[2][1] * r;
						a[_j][6] = s - ifet[3][0] * r;
						a[_j][7] =   - ifet[3][1] * r;
						b[_j][0] = s + ifet[0][0] * r;
						b[_j][1] =   + ifet[0][1] * r;
						b[_j][2] = s + ifet[1][0] * r;
						b[_j][3] =   + ifet[1][1] * r;
						b[_j][4] = s + ifet[2][0] * r;
						b[_j][5] =   + ifet[2][1] * r;
						b[_j][6] = s + ifet[3][0] * r;
						b[_j][7] =   + ifet[3][1] * r;
					}
				}
			}
		}
		for(size_t j=0;j<n_elem;++j) //upper
		{
			size_t k = i*n_elem + j;
			double s = smat[k];
			double r = rmat[k];
			if constexpr(std::is_same_v<ifdt_t,double>)
			{
				a[n_diag+j][0] = s - ifet[0] * r;
				a[n_diag+j][1] = 0.0;
				a[n_diag+j][2] = s - ifet[1] * r;
				a[n_diag+j][3] = 0.0;
				a[n_diag+j][4] = s - ifet[2] * r;
				a[n_diag+j][5] = 0.0;
				a[n_diag+j][6] = s - ifet[3] * r;
				a[n_diag+j][7] = 0.0;
				b[n_diag+j][0] = s + ifet[0] * r;
				b[n_diag+j][1] = 0.0;
				b[n_diag+j][2] = s + ifet[1] * r;
				b[n_diag+j][3] = 0.0;
				b[n_diag+j][4] = s + ifet[2] * r;
				b[n_diag+j][5] = 0.0;
				b[n_diag+j][6] = s + ifet[3] * r;
				b[n_diag+j][7] = 0.0;
			}else
			{
				a[n_diag+j][0] = s - ifet[0][0] * r;
				a[n_diag+j][1] =   - ifet[0][1] * r;
				a[n_diag+j][2] = s - ifet[1][0] * r;
				a[n_diag+j][3] =   - ifet[1][1] * r;
				a[n_diag+j][4] = s - ifet[2][0] * r;
				a[n_diag+j][5] =   - ifet[2][1] * r;
				a[n_diag+j][6] = s - ifet[3][0] * r;
				a[n_diag+j][7] =   - ifet[3][1] * r;
				b[n_diag+j][0] = s + ifet[0][0] * r;
				b[n_diag+j][1] =   + ifet[0][1] * r;
				b[n_diag+j][2] = s + ifet[1][0] * r;
				b[n_diag+j][3] =   + ifet[1][1] * r;
				b[n_diag+j][4] = s + ifet[2][0] * r;
				b[n_diag+j][5] =   + ifet[2][1] * r;
				b[n_diag+j][6] = s + ifet[3][0] * r;
				b[n_diag+j][7] =   + ifet[3][1] * r;
			}
		}
		
		size_t l = n_diag>i? n_diag-i:0;
		size_t h = n_dims-1-n_diag>i? 2*n_diag+1: n_diag+n_dims-i;	
			
		for(size_t j=l;j<n_diag;++j)
		for(size_t k=0;k<n_diag;++k)
		{
			b[j+k+1]-=b[j]*u(i+j-n_diag,k);
		}
		for(size_t j=0;j<n_diag;++j)
		{
			u(i,j)=b[j+n_diag+1]/b[n_diag];//ui
		}
		comp_v v=zero<comp_v>;
		for(size_t j=l;j<h;++j)
		{
			v+=a[j]*comp_v{_mm512_i64gather_pd(vindex,zsrc+i+j-n_diag,8)};
		}
		for(size_t j=l;j<n_diag;++j)
		{
			v-=b[j]*ytmp[i+j-n_diag];
		}
		ytmp[i]=v/b[n_diag];//yi, XXX what if |b|=0
	}
	//solve x=U\y and store x into z
	for(size_t i=n_dims;i-->0ul;)
	{
		size_t jmax = std::min(n_elem,n_dims-i);
		comp_v v=zero<comp_v>;
		for(size_t j=1ul;j<jmax;++j)
		{
			v+=ytmp[i+j]*u(i,j-1ul);
		}
		_mm512_i64scatter_pd(zsrc+i,vindex,ytmp[i] -= v,8);
	}
}//end of cn_propagate_kernel_avx3
#endif
//=========================================================================================//
template<class dims_t>
inline	void	cn_hc_propagate0
(
	recvec<comp_t,recidx<dims_t::n_lmax+1ul,dims_t::n_dims,dims_t::n_elem>> upper,	//(1@S)-idt/2*(1@Hc)
	recvec<comp_t,recidx<dims_t::n_lmax+1ul,dims_t::n_dims,dims_t::n_elem>> lower,	//(1@S)+idt/2*(1@Hc)
	recvec<comp_t,recidx<dims_t::m_dims    ,dims_t::l_dims,dims_t::n_dims>> coef,	//in-place
	comp_t*	work //workspace for work=[S-idt/2*Hc]*coef (should be pre-shifted for each thread) 
)noexcept
{
	auto w=work;
	#pragma omp for simd collapse(2)
	for(size_t im=0ul;im<dims_t::m_dims;++im)
	for(size_t il=0ul;il<dims_t::l_dims;++il)
	{
		long l=dims_t::in_l(im,il);
		auto c=coef(im,il).data;
		intrinsic::symb_mul_vecd<dims_t::n_dims,dims_t::n_elem,0>(upper(l).data,c,w);
		intrinsic::ltrb_ldl_vecd<dims_t::n_dims,dims_t::n_elem  >(lower(l).data,w,w);
		intrinsic::utrb_ldl_vecd<dims_t::n_dims,dims_t::n_elem  >(lower(l).data,w,c);
	}
}//end of cn_hc_propagate0

template<class dims_t>
inline	void	cn_hl_propagate1
(
	recvec<double,recidx<dims_t::m_dims,dims_t::l_dims>> yeig,
	recvec<double,recidx<dims_t::n_dims,dims_t::n_elem>> smat,
	recvec<double,recidx<dims_t::n_dims,dims_t::n_elem>> rmat,
	cn_workspace_so<dims_t>& temp,
	double 	fzdt
)noexcept
{//Y hermitian and R real symmetric. possible usage: H=E*y10*r or H=-A*(I*q10)*(1/r). fz=E or -A.
	comp_t	ifdt	=	unim<comp_t>*(fzdt/2);
	//propagate by 	C'={(1@S)-fz*idt/2*(Yd@R)}* C'
	//		C'={(1@S)+fz*idt/2*(Yd@R)}\ C'. 
	auto	_tid	=	omp_get_thread_num();
	comp_t*	_ldl	=	temp.ldl(_tid);
	comp_t*	_dst	=	temp.tmp(_tid);//temporary value stored in temp.work

	size_t i_start,i_final;
	divide_loop_equally(dims_t::m_dims*dims_t::l_dims,omp_get_num_threads(),_tid,i_start,i_final);

#ifdef support_avx3
	size_t i_bound=i_start+(i_final-i_start)/4ul*4ul;

	for(size_t iml=i_start;iml<i_bound;iml+=4ul)
	{
		comp_t* _src = temp(iml);       //from temp.tmpc
		cn_propagate_kernel_avx3<0,dims_t>
		(
			smat(),rmat(),yeig.data+iml,ifdt,
			_src,(comp_v*)_dst,(comp_v*)_ldl
		);
	}
	for(size_t iml=i_bound;iml<i_final;++iml)
	{
		comp_t* _src = temp(iml);       //from temp.tmpc
		cn_propagate_kernel_avx1<0,dims_t>
		(
			smat(),rmat(),yeig[iml]*ifdt,
			_src,(comp_t*)_dst,(comp_t*)_ldl
		);
	}
#else
	for(size_t iml=i_start;iml<i_final;++iml)
	{
		comp_t* _src = temp(iml);       //from temp.tmpc
		cn_propagate_kernel_avx1<0,dims_t>
		(
			smat(),rmat(),yeig[iml]*ifdt,
			_src,(comp_t*)_dst,(comp_t*)_ldl
		);
	}
#endif
	#pragma omp barrier
}//end of cn_hl_propagate1

template<class dims_t>
inline	void	cn_hl_propagate2	
(
	recvec<double,recidx<dims_t::m_dims,dims_t::l_dims>> yeig,
	recvec<double,recidx<dims_t::n_dims,dims_t::n_elem>> smat,
	recvec<double,recidx<dims_t::n_dims,dims_t::n_elem>> rmat,
	cn_workspace_so<dims_t>& temp,
	double 	fzdt
)noexcept
{//Y hermitian and R imag asymmetric. possible usage: H=-A*y10*(I*d/dr). fz=E or -A
//propagate by 	C'={(1@S)-fz*idt/2*(Yd@R)}* C'
//		C'={(1@S)+fz*idt/2*(Yd@R)}\ C'. 
	double	ifdt=-fzdt/2;//I*I=-1 , extra I from I*d/dr
	auto	_tid	=	omp_get_thread_num();
	comp_t*	_dst	=	temp.tmp(_tid);//temporary values stored in temp.work
	comp_t*	_ldl	=	temp.ldl(_tid);

	size_t i_start,i_final;
	divide_loop_equally(dims_t::m_dims*dims_t::l_dims,omp_get_num_threads(),_tid,i_start,i_final);

#ifdef support_avx3
	size_t i_bound=i_start+(i_final-i_start)/4ul*4ul;

	for(size_t iml=i_start;iml<i_bound;iml+=4ul)
	{
		comp_t* _src = temp(iml);       //from temp.tmpc
		cn_propagate_kernel_avx3<1,dims_t>
		(
			smat(),rmat(),yeig.data+iml,ifdt,
			_src,(comp_v*)_dst,(comp_v*)_ldl
		);
	}
	for(size_t iml=i_bound;iml<i_final;++iml)
	{
		comp_t* _src = temp(iml);       //from temp.tmpc
		cn_propagate_kernel_avx1<1,dims_t>
		(
			smat(),rmat(),yeig[iml]*ifdt,
			_src,(comp_t*)_dst,(comp_t*)_ldl
		);
	}
#else
	for(size_t iml=i_start;iml<i_final;++iml)
	{
		comp_t* _src = temp(iml);       //from temp.tmpc
		cn_propagate_kernel_avx1<1,dims_t>
		(
			smat(),rmat(),yeig[iml]*ifdt,
			_src,(comp_t*)_dst,(comp_t*)_ldl
		);
	}
#endif
	#pragma omp barrier
}//end of cn_hl_propagate2

template<class dims_t,int isub=-1>
inline	void	cn_hm_propagate1
(
	recvec<double,recidx<dims_t::m_dims*dims_t::l_dims>> zeig,
	recvec<double,recidx<dims_t::n_dims,dims_t::n_elem>> smat,
	recvec<double,recidx<dims_t::n_dims,dims_t::n_elem>> rmat,
	cn_workspace_so<dims_t>& temp,
	double 	frdt
)noexcept
{//Z hermitian and R real symmetric. possible usage: 
//H= (Ex/2)*  (ym+yp)@r  
//H= (Ey/2)*I*(ym-yp)@r 
//H=-(Ay/2)  *(qm+qp)@(1/r)
//H= (Ax/2)*I*(qm-qp)@(1/r)
//H=-(Fx/4c)*  (pm+pp)@r^2 where Fi=dEi/dt
//H=-(Fy/4c)*I*(pm-pp)@r^2
//propagate by C'=[ 1@S-(f*idt/2)*Zd@R ]* C ;
//             C =[ 1@S+(f*idt/2)*Zd@R ]\ C';
	size_t constexpr ilow = dims_t::mpro_ilow(isub);
	size_t constexpr iupp = dims_t::mpro_iupp(isub);
	comp_t	ifdt	=	unim<comp_t>*(frdt/2);
	auto	_tid	=	omp_get_thread_num();
	comp_t*	_ldl	=	temp.ldl(_tid);
	comp_t*	_dst	=	temp.tmp(_tid);//temporary values stored in temp.work

	size_t i_start,i_final;
	divide_loop_equally(iupp-ilow,omp_get_num_threads(),_tid,i_start,i_final);

	i_start += ilow;
	i_final += ilow;
#ifdef support_avx3
	size_t i_bound=i_start+(i_final-i_start)/4ul*4ul;

	for(size_t iml=i_start;iml<i_bound;iml+=4ul)
	{
		comp_t* _src = temp(iml);       //from temp.tmpc
		cn_propagate_kernel_avx3<0,dims_t>
		(
			smat(),rmat(),zeig.data+iml,ifdt,
			_src,(comp_v*)_dst,(comp_v*)_ldl
		);
	}
	for(size_t iml=i_bound;iml<i_final;++iml)
	{
		comp_t* _src = temp(iml);       //from temp.tmpc
		cn_propagate_kernel_avx1<0,dims_t>
		(
			smat(),rmat(),zeig(iml)*ifdt,
			_src,(comp_t*)_dst,(comp_t*)_ldl
		);	
	}
#else
	for(size_t iml=i_start;iml<i_final;++iml)
	{
		comp_t* _src = temp(iml);       //from temp.tmpc
		cn_propagate_kernel_avx1<0,dims_t>
		(
			smat(),rmat(),zeig(iml)*ifdt,
			_src,(comp_t*)_dst,(comp_t*)_ldl
		);	
	}
#endif
	#pragma omp barrier
}//end of cn_hm_propagate1

template<class dims_t,int isub=-1>
inline	void	cn_hm_propagate2	
(
	recvec<double,recidx<dims_t::m_dims*dims_t::l_dims>> zeig,
	recvec<double,recidx<dims_t::n_dims,dims_t::n_elem>> smat,
	recvec<double,recidx<dims_t::n_dims,dims_t::n_elem>> rmat,
	cn_workspace_so<dims_t>& temp,
	double frdt
)noexcept	
{//Z hermitian and R pure imag anti-symmetric. possible usage: 
//H=-(Ax/2)*  (ym+yp)@I*d/dr
//H=-(Ay/2)*I*(ym-yp)@I*d/dr
//propagate by C'=[ 1@S-(f*idt/2)*Zd@R ]* C ;
//             C =[ 1@S+(f*idt/2)*Zd@R ]\ C';
	size_t constexpr ilow = dims_t::mpro_ilow(isub);
	size_t constexpr iupp = dims_t::mpro_iupp(isub);

	double	ifdt	=	-frdt/2;//I*I=-1, extra I from I*d/dr
	auto	_tid	=	omp_get_thread_num();
	comp_t*	_dst	=	temp.tmp(_tid);//temporary values stored in temp.work
	comp_t*	_ldl	=	temp.ldl(_tid);

	size_t i_start,i_final;
	divide_loop_equally(iupp-ilow,omp_get_num_threads(),_tid,i_start,i_final);

	i_start += ilow;
	i_final += ilow;
#ifdef support_avx3
	size_t i_bound=i_start+(i_final-i_start)/4ul*4ul;

	for(size_t iml=i_start;iml<i_bound;iml+=4ul)
	{
		comp_t* _src = temp(iml);       //from temp.tmpc
		cn_propagate_kernel_avx3<1,dims_t>
		(
			smat(),rmat(),zeig.data+iml,ifdt,
			_src,(comp_v*)_dst,(comp_v*)_ldl
		);
	}
	for(size_t iml=i_bound;iml<i_final;++iml)
	{
		comp_t* _src = temp(iml);       //from temp.tmpc
		cn_propagate_kernel_avx1<1,dims_t>
		(
			smat(),rmat(),zeig(iml)*ifdt,
			_src,(comp_t*)_dst,(comp_t*)_ldl
		);	
	}
#else
	for(size_t iml=i_start;iml<i_final;++iml)
	{
		comp_t* _src = temp(iml);       //from temp.tmpc
		cn_propagate_kernel_avx1<1,dims_t>
		(
			smat(),rmat(),zeig(iml)*ifdt,
			_src,(comp_t*)_dst,(comp_t*)_ldl
		);	
	}
#endif
	#pragma omp barrier
}//end of cn_hm_propagate2

template<class dims_t,int isub=-1>
inline	void	cn_hm_propagate3
(
	recvec<double,recidx<dims_t::m_dims*dims_t::l_dims>> zeig,
	cn_workspace_so<dims_t>& temp,
	double frdt
)noexcept
{//Z hermitian and R equals S. possible usage:
//H=-(Gx/4c)*I*(l0m-l0p)
//H= (Gy/4c)*  (l0m+l0p)
//propagate by C =exp[-(f*idt/2)*Zd@1]* C ;
	comp_t	ifdt	=	unim<comp_t>*(frdt/2);
	#pragma omp for
	for(size_t iml=dims_t::mpro_ilow(isub);iml<dims_t::mpro_iupp(isub);++iml)
	{
		double	_eig	=	zeig(iml);
		comp_t* _src    =       temp(iml);//read from temp, and store back to temp
		for(size_t ir=0;ir<dims_t::n_dims;++ir)
		{
			_src[ir]*=	exp(-ifdt*_eig);
		}
	}
}//end of cn_hm_propagate3