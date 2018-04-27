#ifndef MTHREADLIB_MTTASK_H_INCLUDED
#define MTHREADLIB_MTTASK_H_INCLUDED

namespace Mod
{
	struct MTTask
	{
		virtual void Process( void* subject ) = 0;
	};

	struct MTTaskIndexed
	{
		virtual void Process( void* subject, size_t idx ) = 0;
	};
}

#endif