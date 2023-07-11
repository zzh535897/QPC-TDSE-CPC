#pragma once

namespace qpc
{
	class trash
	{
		public:
			template<class...args_t>
			explicit trash(args_t&&...)noexcept{}

			template<class...args_t>
			void operator()(args_t&&...)noexcept{}
	};
}
