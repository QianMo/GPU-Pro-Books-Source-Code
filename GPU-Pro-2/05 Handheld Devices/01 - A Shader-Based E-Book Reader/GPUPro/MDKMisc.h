/******************************************************************************

 @File         MDKMisc.h

 @Title        MDKTools

 @Copyright    Copyright (C) 2010 by Imagination Technologies Limited.

 @Platform     Independent

 @Description  Miscellaneous functions
 
******************************************************************************/

#ifndef __MDKMISC_H__
#define __MDKMISC_H__

#include <assert.h>

#define SAFE_DELETE(x) { delete x; x = 0; }
#define SAFE_DELETE_ARRAY(x) { delete [] x; x = 0; }


#define brt_printf DebugPrint;
void DebugPrint(char const * const format, ...);


#endif //__MDKMISC_H__
