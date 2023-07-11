#pragma once
#include <omp.h>
#include <cstddef>

// omp_get_num_procs
// omp_get_num_threads
// omp_set_num_threads
// omp_get_thread_num

namespace qpc
{
	inline	void	divide_loop_equally	(const size_t n_loop,const size_t n_thread,const size_t i_thread,size_t& i_start,size_t& i_final)noexcept
	{
		//n0*(ne+1)+n1*ne	==	n_loop		==	nthread*ne+n0
		//n0       +n1		==	n_thread
		size_t 	n0	=	n_loop%n_thread;
		size_t	ne	=	n_loop/n_thread;;
		if(i_thread<n0)
		{
			i_start	=	i_thread*(ne+1ul);
			i_final	=	i_start+(ne+1ul);
		}else
		{
			i_start	=	n_loop-(n_thread-i_thread)*ne;
			i_final	=	i_start+ne;
		}
	}
}
