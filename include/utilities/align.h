#pragma once

namespace qpc
{

	static constexpr size_t alignval	(const size_t n_byte,const size_t alignment)noexcept
	{
		return	(n_byte+alignment-1ul)/alignment*alignment;
	}//end of alignval
}//end of namespace
