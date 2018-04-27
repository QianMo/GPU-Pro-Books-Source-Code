#ifndef MTHREADLIB_ATOMIC_H_INCLUDED
#define MTHREADLIB_ATOMIC_H_INCLUDED

namespace Mod
{
	inline void AtomicIncrement( volatile INT32& target )
	{
#ifdef MD_WIN_PLATFORM
		_InterlockedIncrement( reinterpret_cast<volatile long*>(&target) );

#else
#error Unsupported platform
#endif
	}

	//------------------------------------------------------------------------

	inline void AtomicDecrement( volatile INT32& target )
	{
#ifdef MD_WIN_PLATFORM
		_InterlockedDecrement( reinterpret_cast<volatile long*>(&target) );
#else
#error Unsupported platform
#endif
	}

	//------------------------------------------------------------------------

	inline void AtomicWrite( volatile INT32& target, INT32 val )
	{
#ifdef MD_WIN_PLATFORM
		_InterlockedExchange( reinterpret_cast<volatile long*>(&target), val );
#else
#error Unsupported platform
#endif
	}
}

#endif