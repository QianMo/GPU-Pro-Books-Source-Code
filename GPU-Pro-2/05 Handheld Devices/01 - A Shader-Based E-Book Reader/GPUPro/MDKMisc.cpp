/******************************************************************************

 @File         MDKMisc.cpp

 @Title        MDKTools

 @Copyright    Copyright (C) 2010 by Imagination Technologies Limited.

 @Platform     Independent

 @Description  Miscellaneous functions
 
******************************************************************************/

#include <stdarg.h>
#include <stdio.h>

#include "MDKMisc.h"

// This now calls Log::Write which knows about the shell.
void DebugPrint(char const * const format, ...)
{
	va_list arg;
	va_start( arg, format );

	char	buf[1024];
#if defined(WINCE) 
	_vsnprintf(buf, 1024, format, arg);
#elif defined(__SYMBIAN32__)
	vsprintf(buf, format, arg);
#else
	vsnprintf(buf, 1024, format, arg);
#endif

	va_end( arg );
}

