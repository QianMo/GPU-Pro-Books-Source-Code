/******************************************************************************

 @File         Globals.h

 @Title        API independent class declaration for PVREngine

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     Independent

 @Description  global definitions for the PVREngine

******************************************************************************/
#ifndef _GLOBALS_H_
#define _GLOBALS_H_

/******************************************************************************
Includes
******************************************************************************/

namespace pvrengine
{
	/******************************************************************************
	Defines
	******************************************************************************/

#ifndef PVRDELETE
#define PVRDELETE(thing)		{if(thing) delete thing; thing=NULL; }
#endif
#ifndef PVRDELETEARRAY
#define PVRDELETEARRAY(thing)	{if(thing) delete[] thing; thing=NULL; }
#endif

#ifndef PVREDEBUGOUT
#ifdef _DEBUG
#if defined(_WIN32) && !defined(UNDER_CE)
#define PVREDEBUGOUT(A) OutputDebugStringA(A);
#else
#define PVREDEBUGOUT(A) fprintf(stderr,A);
#endif
#else
#define PVREDEBUGOUT(A)
#endif
#endif

	const unsigned int PVR_INVALID = 0xffffffff;

}
#endif // _GLOBALS_H_

/******************************************************************************
End of file (PVREngine.h)
******************************************************************************/
