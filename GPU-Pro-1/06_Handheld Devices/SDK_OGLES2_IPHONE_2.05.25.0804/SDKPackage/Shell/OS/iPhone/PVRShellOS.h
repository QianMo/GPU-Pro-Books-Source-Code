/******************************************************************************

 @File         PVRShellOS.h

 @Title        iPhone/PVRShellOS

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     iPhone

 @Description  Makes programming for 3D APIs easier by wrapping surface
               initialization, Texture allocation and other functions for use by a demo.

******************************************************************************/
#ifndef _PVRSHELLOS_
#define _PVRSHELLOS_


#define PVRSHELL_DIR_SYM	'/'
#define _stricmp strcasecmp

/*!***************************************************************************
 PVRShellInitOS
 @Brief Class. Interface with specific Operative System.
*****************************************************************************/
class PVRShellInitOS
{
public:
	char* szTitle;

	float m_vec3Accel[3];
	float m_vec2PointerLocation[2];
	bool m_bPointer, m_bNormalized;

};

#endif /* _PVRSHELLOS_ */
/*****************************************************************************
 End of file (PVRShellOS.h)
*****************************************************************************/
