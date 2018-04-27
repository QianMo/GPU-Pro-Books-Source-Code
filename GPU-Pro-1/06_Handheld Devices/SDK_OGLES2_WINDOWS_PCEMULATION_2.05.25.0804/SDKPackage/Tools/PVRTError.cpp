/******************************************************************************

 @File         PVRTError.cpp

 @Title        PVRTError

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     ANSI compatible

 @Description  

******************************************************************************/
#include "PVRTError.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

#ifdef _WIN32
#define vsnprintf _vsnprintf
#endif

/*!***************************************************************************
 @Function			PVRTErrorOutputDebug
 @Input				format		printf style format followed by arguments it requires
 @Description		Outputs a string to the standard error.
*****************************************************************************/
void PVRTErrorOutputDebug(char const * const format, ...)
{
	va_list arg;
	char	pszString[1024];

	va_start(arg, format);
#if defined(__SYMBIAN32__) || defined(UITRON) || defined(_UITRON_)
	vsprintf(pszString, format, arg);
#else
	vsnprintf(pszString, 1024, format, arg);
#endif
	va_end(arg);


#if defined(UNICODE) && !defined(__SYMBIAN32__) && !defined(UNDER_CE)
	wchar_t *pswzString = (wchar_t *)malloc((strlen(pszString) + 1) * sizeof(wchar_t));

	int i;
	for(i = 0; pszString[i] != '\0'; i++)
	{
		pswzString[i] = (wchar_t)(pszString[i]);
	}
	pswzString[i] = '\0';

	#if defined(_WIN32)
		OutputDebugString(pswzString);
	#else
		fprintf(stderr, pswzString);
	#endif

	free(pswzString);
#else
	#if defined(__SYMBIAN32__)
		RDebug::Printf(pszString);
	#elif defined(_WIN32) && !defined(UNDER_CE)
		OutputDebugString(pszString);
	#else
		fprintf(stderr, "%s", pszString);
	#endif
#endif
}

/*****************************************************************************
End of file (PVRTError.cpp)
*****************************************************************************/

