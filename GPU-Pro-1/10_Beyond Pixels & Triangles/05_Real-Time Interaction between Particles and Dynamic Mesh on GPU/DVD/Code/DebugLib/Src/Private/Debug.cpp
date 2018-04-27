#include "Precompiled.h"

namespace Mod
{
#ifdef MD_WIN_PLATFORM
	void FatalError( const String& message )
	{
		FatalAppExit( 0, message.c_str() );
	}

	void DebugMessage( const String& message )
	{
		if( IsDebuggerPresent() )
		{
			OutputDebugString( message.c_str() );
		}
	}

#else
#error Unsupported platform
#endif
}