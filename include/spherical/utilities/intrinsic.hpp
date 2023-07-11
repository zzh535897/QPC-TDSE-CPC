#pragma once

//require <libraries/support_std.h>
//require <libraries/support_avx.h>


namespace intrinsic{

#include <spherical/utilities/blas1.hpp>
#include <spherical/utilities/blas2.hpp>
	
	//---------------------------------------------------------------------------------------------------------------------------------------
	//
	//					the following codes are probably to be deprecated
	//						     
	//---------------------------------------------------------------------------------------------------------------------------------------
	//(n1,m1) =(n_dims,n_elem) of outer band
	//(n2,m2) =(n_dims,n_elem) of inner band
	//op:
	//	0 <->  =
	//	1 <-> +=
	//	2 <-> -=

	template<size_t n1,size_t m1,size_t n2,size_t m2,int op,class type_a,class type_b,class type_x,class type_y,class...mult_t>	
	static	inline	void	symb_prod_symb_mul_vecd	(const type_a* a,const type_b* b,const type_x* x,type_y* y,mult_t&&...c)noexcept
	{//y op (a@b)*(c*x) where a,b are symmetric banded matrix, c is scalar
		static_assert(op==0||op==1||op==2);
		for(size_t i1=0;i1<n1;++i1)//outer row
		{
			auto* x_this	=	x+i1*n2;
			auto* y_this	=	y+i1*n2;
			//diagonal term
			intrinsic::symb_mul_vecd<n2,m2,op>(b,x_this,y_this,(a[i1*m1]*...*c));
			//upper panel
			size_t jumax	=	std::min(m1,n1-i1);
			for(size_t j1=1;j1<jumax;++j1)
			{
				if constexpr(op<=1)intrinsic::symb_mul_vecd<n2,m2,1>(b,x_this+j1*n2,y_this,(a[i1*m1+j1]*...*c));
				if constexpr(op==2)intrinsic::symb_mul_vecd<n2,m2,2>(b,x_this+j1*n2,y_this,(a[i1*m1+j1]*...*c));
			}
			//lower panel
			size_t jlmax	=	std::min(m1,i1+1);
			for(size_t j1=1;j1<jlmax;++j1)
			{
				if constexpr(op<=1)intrinsic::symb_mul_vecd<n2,m2,1>(b,x_this-j1*n2,y_this,(a[(i1-j1)*m1+j1]*...*c));
				if constexpr(op==2)intrinsic::symb_mul_vecd<n2,m2,2>(b,x_this-j1*n2,y_this,(a[(i1-j1)*m1+j1]*...*c));
			}
		}
	}//end of nested_symb_mul_vecd

	template<size_t n1,size_t m1,size_t n2,size_t m2,int op,class type_a,class type_b,class type_x,class type_y,class...mult_t>	
	static	inline	void	symb_prod_asyb_mul_vecd	(const type_a* a,const type_b* b,const type_x* x,type_y* y,mult_t&&...c)noexcept
	{//y op (a@b)*(c*x) where a is symmetric banded matrix, b is anti-symmetric banded matrix, c is scalar
		static_assert(op==0||op==1||op==2);
		for(size_t i1=0;i1<n1;++i1)//outer row
		{
			auto* x_this	=	x+i1*n2;
			auto* y_this	=	y+i1*n2;
			//diagonal term
			intrinsic::asyb_mul_vecd<n2,m2,op>(b,x_this,y_this,(a[i1*m1]*...*c));
			//upper panel
			size_t jumax	=	std::min(m1,n1-i1);
			for(size_t j1=1;j1<jumax;++j1)
			{
				if constexpr(op<=1)intrinsic::asyb_mul_vecd<n2,m2,1>(b,x_this+j1*n2,y_this,(a[i1*m1+j1]*...*c));
				if constexpr(op==2)intrinsic::asyb_mul_vecd<n2,m2,2>(b,x_this+j1*n2,y_this,(a[i1*m1+j1]*...*c));
			}
			//lower panel
			size_t jlmax	=	std::min(m1,i1+1);
			for(size_t j1=1;j1<jlmax;++j1)
			{
				if constexpr(op<=1)intrinsic::asyb_mul_vecd<n2,m2,1>(b,x_this-j1*n2,y_this,(a[(i1-j1)*m1+j1]*...*c));
				if constexpr(op==2)intrinsic::asyb_mul_vecd<n2,m2,2>(b,x_this-j1*n2,y_this,(a[(i1-j1)*m1+j1]*...*c));
			}
		}
	}//end of symb_prod_asyb_mul_vecd

	template<size_t n1,size_t m1,size_t n2,size_t m2,int op,class type_a,class type_b,class type_x,class type_y,class...mult_t>	
	static	inline	void	asyb_prod_symb_mul_vecd	(const type_a* a,const type_b* b,const type_x* x,type_y* y,mult_t&&...c)noexcept
	{//y op (a@b)*(c*x) where a is anti-symmetric banded matrix, b is symmetric banded matrix, c is scalair
		static_assert(op==0||op==1||op==2);
		for(size_t i1=0;i1<n1;++i1)//outer row
		{
			auto* x_this	=	x+i1*n2;
			auto* y_this	=	y+i1*n2;
			//diagonal term is always zero!!
			if constexpr(op==0)
			for(size_t i2=0;i2<n2;++i2)y_this[i2]=zero<type_y>;
			//upper panel
			size_t jumax	=	std::min(m1,n1-i1);
			for(size_t j1=1;j1<jumax;++j1)
			{
				if constexpr(op<=1)intrinsic::symb_mul_vecd<n2,m2,1>(b,x_this+j1*n2,y_this,(a[i1*m1+j1]*...*c));
				if constexpr(op==2)intrinsic::symb_mul_vecd<n2,m2,2>(b,x_this+j1*n2,y_this,(a[i1*m1+j1]*...*c));
			}
			//lower panel (swap += and -=)
			size_t jlmax	=	std::min(m1,i1+1);
			for(size_t j1=1;j1<jlmax;++j1)
			{
				if constexpr(op<=1)intrinsic::symb_mul_vecd<n2,m2,2>(b,x_this-j1*n2,y_this,(a[(i1-j1)*m1+j1]*...*c));
				if constexpr(op==2)intrinsic::symb_mul_vecd<n2,m2,1>(b,x_this-j1*n2,y_this,(a[(i1-j1)*m1+j1]*...*c));
			}
		}
	}//end of asyb_prod_symb_mul_vecd

	template<size_t n1,size_t m1,size_t n2,size_t m2,int op,class type_a,class type_b,class type_x,class type_y,class...mult_t>	
	static	inline	void	asyb_prod_asyb_mul_vecd	(const type_a* a,const type_b* b,const type_x* x,type_y* y,mult_t&&...c)noexcept
	{//y op (a@b)*(c*x) where a,b are anti-symmetric banded matrix, c is scalair
		static_assert(op==0||op==1||op==2);
		for(size_t i1=0;i1<n1;++i1)//outer row
		{
			auto* x_this	=	x+i1*n2;
			auto* y_this	=	y+i1*n2;
			//diagonal term is always zero!!
			if constexpr(op==0)
			for(size_t i2=0;i2<n2;++i2)y_this[i2]=zero<type_y>;
			//upper panel
			size_t jumax	=	std::min(m1,n1-i1);
			for(size_t j1=1;j1<jumax;++j1)
			{
				if constexpr(op<=1)intrinsic::symb_mul_vecd<n2,m2,1>(b,x_this+j1*n2,y_this,(a[i1*m1+j1]*...*c));
				if constexpr(op==2)intrinsic::symb_mul_vecd<n2,m2,2>(b,x_this+j1*n2,y_this,(a[i1*m1+j1]*...*c));
			}
			//lower panel (swap += and -=)
			size_t jlmax	=	std::min(m1,i1+1);
			for(size_t j1=1;j1<jlmax;++j1)
			{
				if constexpr(op<=1)intrinsic::symb_mul_vecd<n2,m2,2>(b,x_this-j1*n2,y_this,(a[(i1-j1)*m1+j1]*...*c));
				if constexpr(op==2)intrinsic::symb_mul_vecd<n2,m2,1>(b,x_this-j1*n2,y_this,(a[(i1-j1)*m1+j1]*...*c));
			}
		}
	}//end of asyb_prod_symb_mul_vecd
	

	template<size_t n1,size_t m1,size_t n2,size_t m2,int op,class type_a,class type_x,class type_y,class...mult_t>	
	static	inline	void	double_symb_mul_vecd	(const type_a* a,const type_x* x,type_y* y,const mult_t&...c)noexcept
	{//y op a*c*x where a is double-symmetric-banded-matrix, c is scalar
		static_assert(op==0||op==1||op==2);
		if constexpr(op==0)
		{
			for(size_t i=0;i<n1*n2;++i)y[i]=zero<type_y>;//set y to 0
		}
		for(size_t i1=0;i1<n1;++i1)//outer row
		{
			//find the vector
			auto* x_this	=	x+i1*n2;
			auto* y_this	=	y+i1*n2;
			auto* a_this	=	a+i1*n2*m2*m1;
			size_t j1max	=	std::min(m1,n1-i1);
			//diagonal
			if constexpr(sizeof...(c)>0ul)
			{
				if constexpr(op<=1)intrinsic::symb_mul_vecd<n2,m2,1>(a_this,x_this,y_this,(c*...));
				if constexpr(op==2)intrinsic::symb_mul_vecd<n2,m2,2>(a_this,x_this,y_this,(c*...));
			}else
			{
				if constexpr(op<=1)intrinsic::symb_mul_vecd<n2,m2,1>(a_this,x_this,y_this);
				if constexpr(op==2)intrinsic::symb_mul_vecd<n2,m2,2>(a_this,x_this,y_this);
			}
			//off-diagonal
			for(size_t j1=1;j1<j1max;++j1)
			{
				a_this	+=	n2*m2;
				if constexpr(sizeof...(c)>0ul){
				//upper
				if constexpr(op<=1)intrinsic::symb_mul_vecd<n2,m2,1>(a_this,x_this+j1*n2,y_this,(c*...));
				if constexpr(op==2)intrinsic::symb_mul_vecd<n2,m2,2>(a_this,x_this+j1*n2,y_this,(c*...));
				//lower
				if constexpr(op<=1)intrinsic::symb_mul_vecd<n2,m2,1>(a_this,x_this,y_this+j1*n2,(c*...));
				if constexpr(op==2)intrinsic::symb_mul_vecd<n2,m2,2>(a_this,x_this,y_this+j1*n2,(c*...));
				}else{
				//upper
				if constexpr(op<=1)intrinsic::symb_mul_vecd<n2,m2,1>(a_this,x_this+j1*n2,y_this);
				if constexpr(op==2)intrinsic::symb_mul_vecd<n2,m2,2>(a_this,x_this+j1*n2,y_this);
				//lower
				if constexpr(op<=1)intrinsic::symb_mul_vecd<n2,m2,1>(a_this,x_this,y_this+j1*n2);
				if constexpr(op==2)intrinsic::symb_mul_vecd<n2,m2,2>(a_this,x_this,y_this+j1*n2);}//end
			}
		}
	}//end of double_symb_mul_vecd
	//---------------------------------------------------------------------------------------------------------------------------------------
	//
	//			SECTION. IV  to trace off A(n1,n2,:) with lhs(n1) and rhs(n2), with (n1,n2) symb structure
	//
	//---------------------------------------------------------------------------------------------------------------------------------------
	template<size_t n,size_t m,size_t n_rest,class type_a,class type_x,class type_y,class type_b>	
	static inline void	symb_trc_vecd	(const type_a* a,const type_x* x,const type_y* y,type_b* b)noexcept
	{
		static_assert(m>0ul);
		for(size_t id=0ul;id<n_rest;++id)
		{
			b[id]=zero<type_b>;
		}
		for(size_t ir=0ul;ir<n;++ir)
		{
			size_t icmax = std::min(n-ir,m);	
			for(size_t ic=0ul;ic<icmax;++ic)
			{
				auto 	coef	=	conj(x[ir])*y[ir+ic];
				auto*	atmp	=	a+(ir*m+ic)*n_rest;
				for(size_t id=0ul;id<n_rest;++id)b[id]+=atmp[id]*coef;
			}
			size_t icmin = std::min(ir,m-1ul);
			for(size_t ic=1ul;ic<=icmin;++ic)
			{
				auto 	coef	=	conj(x[ir])*y[ir-ic];
				auto*	atmp	=	a+((ir-ic)*m+ic)*n_rest;
				for(size_t id=0ul;id<n_rest;++id)b[id]+=atmp[id]*coef;
			}
		}
	}//end of symb_trc_vecd



}//end of intrinsic

