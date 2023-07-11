#pragma once

#include <libraries/support_std.h>

namespace qpc
{
	template<class...>
	class	compile_error_t;
	template<class info_t,class oper_t>
	class	runtime_error_t;
	class	runtime_error_base_t;
	//=======================================================
	//			compile_error
	//=======================================================


	//=======================================================
	//			runtime error
	//=======================================================
	class	runtime_error_base_t
	{
		public:
			virtual void what()const noexcept=0;
	};

	template<class info_t,class oper_t>
	class	runtime_error_t:public runtime_error_base_t
	{
		private:
			info_t	info;		

		public:
			
			void 	what()const noexcept override
			{
				if constexpr(!std::is_same_v<oper_t,void>)oper_t::show_info(info);
				else std::cout<<info<<std::endl;
			}
			
			runtime_error_t(info_t&& _info):info(std::move(_info)){}
			runtime_error_t(const info_t& _info):info(_info){}
	};
}
