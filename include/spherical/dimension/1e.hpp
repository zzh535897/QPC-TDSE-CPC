#pragma once

//----------------------------------------------------------------------------------------
//for one electron dimension specifier, one should have 3 indices (i,l,m).
//
//_nr is the number of r index
//_mr is the rank   of r basis 
//_nl is the number of l index
//_nm is the number of m index
//_m0 is the minimal m value of m index
//
//m basis, (-m0,-m0+1,...,-m0+nm-1),  # = nm  	in total (exp(im*ph))
//l basis, (|m|,|m|+1,...,|m|+nl-1)   # = nl   	in total (legendre function Plm(costh))
//r basis, (0,1,2,...,nmax-1)         # = nr   	in total (b-spline function Bnk(r)) 
//
//the running index to loop over r,l,m subspace are often called ir,il,im
//----------------------------------------------------------------------------------------

template<size_t _nr,size_t _mr,size_t _nl,size_t _nm,long _m0,bool lf=0,bool rt=0>
struct	dimension_1e
{//designed for one electron problem

	static constexpr size_t nr	=	_nr;			
	static constexpr size_t mr	=	_mr;
	static constexpr size_t nl	=	_nl;
	static constexpr size_t nm	=	_nm;
	static constexpr long 	m0	=	_m0;	
	static constexpr long 	m1	=	_m0+long(_nm)-1;
	
	static constexpr bool	b_lfbd	=	lf;		//choose the boundary condition for left side. 0=reflective, 1=none
	static constexpr bool	b_rtbd	=	rt;		//choose the boundary condition for right side. 0=reflective, 1=none

	static constexpr int	i_type	=	(!lf&&!rt)?0:	//for spherical
						( lf&&!rt)?1:	//for spheroidal
						-1;		//invalid

	static constexpr size_t n_dims	=	_nr;		//the rank of r-subspace
	static constexpr size_t l_dims	=	_nl;		//the rank of l-subspace
	static constexpr size_t m_dims	=	_nm;		//the rank of m-subspace
	static constexpr long   m_init	=	_m0;
	static constexpr size_t n_leng	=	_nr*_nl*_nm;	//number of total elements in a wave function vector

	//-------------------- should not be called from lsub,rsub,rout,lout----------------
	static constexpr size_t n_diag	=	_mr;		//'subdiagonal' number. i.e. the number of nonzero sub-diagonal bands
	static constexpr size_t n_elem	=	_mr+1ul;	//number of stored elements in one row (symmetric or asymmtric)
	static constexpr size_t n_full	=	_mr*2ul+1ul;	//number of stored elements in one row (general)
	static constexpr size_t	n_rsub	=	n_elem*n_dims;	//number of elements in r-subspace (which is a symmetric banded matrix)

	static constexpr size_t n_mmax	=	cmax(cabs(m0),cabs(m1));//the maximum possible |m|.
	static constexpr size_t n_lmax	=	n_mmax+l_dims-1ul;	//the maximum possible l.

	//---------------- subspace division for cp&ep cases -------------------------//
	static constexpr size_t n_zero	=	m_dims%2ul==0ul?0ul:l_dims; //number of "zero" transform
	static constexpr size_t n_even	=	(m_dims/2ul*2ul)*((l_dims+1ul)/2ul);//number of "even" transform
	static constexpr size_t n_odds	=	(m_dims/2ul*2ul)*((l_dims+0ul)/2ul);//number of "odd" transform

	static constexpr size_t msub_ilow	(const int flag)noexcept
	{
		if(flag==2) return 0ul;		//2 = gathered even
		else if(flag==3) return n_even;	//3 =gathered odd
		else return 0ul;		//0,1,-1 = interleaved even/odd/all
	}
	static constexpr size_t msub_iupp	(const int flag)noexcept
	{
		if(flag==2) return n_even+n_zero;	//2 = gathered even
		else if(flag==3) return m_dims*l_dims  ;//3 = gathered odd
		else return m_dims*l_dims;	//0,1,-1 = interleaved even/odd/all
	}
	static constexpr size_t mpro_ilow	(const int flag)noexcept
	{
		if(flag==2) return 0ul;		//2 = gathered even 
		else if(flag==3) return n_even+n_zero; //3 =gathered odd
		else return 0ul;
	}
	static constexpr size_t mpro_iupp	(const int flag)noexcept
	{
		if(flag==2) return n_even;         //2 = gathered even
		else if(flag==3) return m_dims*l_dims  ;//3 = gathered odd
		else return m_dims*l_dims;
	}
	//-----------------------------------------------------------------------------------
	//convert im,il into m,l
	static constexpr void	in	(size_t im,size_t il,long& m,long& l)noexcept		
	{
		m	=	long(im)+m0;
		l	=	long(il)+cabs(m);
	}//end of in
	//convert m,l into im,il
	static constexpr void	in	(long m,long l,size_t& im,size_t& il)noexcept
	{
		im	=	size_t(m-m0);
		il	=	size_t(l-cabs(m));
	}//end of in
	
	//check whether  (m1+msh,l1+lsh) exists in the grid, where (m1,l1) given by (im1,il1). if so, (im2,il2) are set by corresponding value to (m1+msh,l1+lsh).
	template<size_t msh,size_t lsh>
	static constexpr bool	in_check(size_t im1,size_t il1,size_t& im2,size_t& il2)noexcept
	{
		return 	(im2=im1+msh)<m_dims && (il2=size_t(in_l(im1,il1))+lsh-in_mabs(im2))<l_dims;
	}//end of in_check
	
	//convert im into m
	static constexpr long	in_m	(size_t im)noexcept
	{
		return 	long(im)+m0;
	}//end of in_m

	//convert il into l
	static constexpr long 	in_l	(size_t im,size_t il)noexcept
	{
		return	long(il)+cabs(long(im)+m0);
	}//end of in_l

	static constexpr size_t in_mabs	(size_t im)noexcept
	{
		return 	size_t(cabs(long(im)+m0));
	}//end of in_mabs

	static constexpr size_t	od_m	(size_t im)noexcept
	{
		//to decide global term when spheroidal coordinate is enabled
		//return 0 if m%2==0, G=1 
		//return 1 if m%2==1, G=[(x-1)/(x+1)]^(1/2)
		if constexpr(m0%2==0)return im%2ul;
		else return (im+1ul)%2ul;
	}//end of od_m
};//end of dimension_1e
