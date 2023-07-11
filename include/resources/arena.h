#pragma once
#include <utilities/types.h>
#include <utilities/error.h>

#include <libraries/support_std.h>

namespace qpc
{
	static constexpr std::align_val_t align_val_none        =       std::align_val_t(8);	// 8 Byte alignment
        static constexpr std::align_val_t align_val_avx1        =       std::align_val_t(16);	//16 Byte alignment
	static constexpr std::align_val_t align_val_avx2        =       std::align_val_t(32);	//32 Byte alignment
        static constexpr std::align_val_t align_val_avx3        =       std::align_val_t(64);	//64 Byte alignment

	class	arena//the base class
	{
		private:
			virtual	void	do_acquire	(void** p_trgt,size_t n_byte,std::align_val_t n_align)=0;	
			virtual	void	do_release	(void*)=0;

		public:
			inline	void	acquire		(void** p_trgt,size_t n_byte,std::align_val_t n_align=align_val_none)
			{
				do_acquire(p_trgt,n_byte,n_align);
			}
			inline	void	release		(void*	p_trgt)
			{
				do_release(p_trgt);
			}

			static 	void 	show_info	(const std::string& info)noexcept
			{
				printf("arena error:%s\n",info.c_str());
			};
			
			virtual	~arena()=default;
	};//end of arena

	class	arena_default final:public arena//assign (host) memory by C++ default new
	{
		private:
			void	do_acquire	(void** p_trgt,size_t n_byte,std::align_val_t n_align)override
			{
				*p_trgt =       new (n_align,std::nothrow) char[n_byte];
				if(p_trgt==0)   throw 	runtime_error_t<std::string,arena>{"fail to acquire memory!"};
			}
			void	do_release	(void*	p_trgt)override
			{
				if(p_trgt!=0)	delete[] (char*)p_trgt;
			}
		public:
			arena_default()=default;
                       	arena_default(arena_default& )=delete;
                        arena_default(arena_default&&)=delete;
                        ~arena_default()=default;
	};//end of arena_default

	class	arena_forward final:public arena//a linear (host) memory assignment by C++ placement new
	{
		private:
			void*	head;
			void*	tail;
			void*	capa;
			arena*	rsrc;
		
			void    do_acquire      (void** p_trgt,size_t n_byte,std::align_val_t n_align)override
                       	{
                        	if(n_byte>size_t((char*)capa-(char*)tail))
				{
                                	throw 	runtime_error_t<std::string,arena>{"fail to acquire"+std::to_string(n_byte)+" byte memory!"};
				}else{
					if(size_t(tail)%size_t(n_align)!=0)
					tail=(void*)(size_t(tail)+size_t(n_align)-size_t(tail)%size_t(n_align));
                              		*p_trgt=tail;
                                	tail=(char*)tail+n_byte;
				}
                        }
			void	do_release	(void*	p_trgt)override
			{
				//do nothing
			}
		public:
			void    refresh         ()noexcept
			{
				tail    =       head;
			}
			void	realloc		(size_t _capa)
			{
				rsrc	->	release(head);
				rsrc	->	acquire((void**)&head,_capa);
				tail    =       (char*)head;
				capa    =       (char*)head+_capa;
			}
			size_t	size		()noexcept
			{
				return 	(char*)capa-(char*)tail;
			}
			template<class type_t,class...args_t>
			void*	create		(args_t&&...args)
			{
				void*	ptr;
				acquire(&ptr,sizeof(type_t),align_val_avx3);
				new (ptr) type_t(std::forward<args_t>(args)...);
				return 	ptr;
			}
			template<class type_t>
			void*	create_vector	(size_t _size)
			{
				void*	ptr;
				acquire(&ptr,sizeof(type_t)*_size,align_val_avx3);
				return 	ptr;
			}

			arena_forward(size_t _capa,arena* _rsrc)
			{
				_rsrc   ->      acquire((void**)&head,_capa);
				tail    =       (char*)head;
				capa    =       (char*)head+_capa;
				rsrc    =       _rsrc;
			}
			arena_forward(arena_forward& )=delete;
			arena_forward(arena_forward&&)=delete;
			~arena_forward()noexcept override
			{
				rsrc    ->      release(head);
                        }

	};//end of arena_forward
	
	arena_default 	global_default_arena_obj;
	arena* 	 	global_default_arena_ptr	=	&global_default_arena_obj;
}//end of qpc
