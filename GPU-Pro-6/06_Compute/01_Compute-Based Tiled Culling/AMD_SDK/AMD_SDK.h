//
// Copyright 2014 ADVANCED MICRO DEVICES, INC.  All Rights Reserved.
//
// AMD is granting you permission to use this software and documentation (if
// any) (collectively, the “Materials”) pursuant to the terms and conditions
// of the Software License Agreement included with the Materials.  If you do
// not have a copy of the Software License Agreement, contact your AMD
// representative for a copy.
// You agree that you will not reverse engineer or decompile the Materials,
// in whole or in part, except as allowed by applicable law.
//
// WARRANTY DISCLAIMER: THE SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF
// ANY KIND.  AMD DISCLAIMS ALL WARRANTIES, EXPRESS, IMPLIED, OR STATUTORY,
// INCLUDING BUT NOT LIMITED TO THE IMPLIED WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE, TITLE, NON-INFRINGEMENT, THAT THE SOFTWARE
// WILL RUN UNINTERRUPTED OR ERROR-FREE OR WARRANTIES ARISING FROM CUSTOM OF
// TRADE OR COURSE OF USAGE.  THE ENTIRE RISK ASSOCIATED WITH THE USE OF THE
// SOFTWARE IS ASSUMED BY YOU.
// Some jurisdictions do not allow the exclusion of implied warranties, so
// the above exclusion may not apply to You. 
// 
// LIMITATION OF LIABILITY AND INDEMNIFICATION:  AMD AND ITS LICENSORS WILL
// NOT, UNDER ANY CIRCUMSTANCES BE LIABLE TO YOU FOR ANY PUNITIVE, DIRECT,
// INCIDENTAL, INDIRECT, SPECIAL OR CONSEQUENTIAL DAMAGES ARISING FROM USE OF
// THE SOFTWARE OR THIS AGREEMENT EVEN IF AMD AND ITS LICENSORS HAVE BEEN
// ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.  
// In no event shall AMD's total liability to You for all damages, losses,
// and causes of action (whether in contract, tort (including negligence) or
// otherwise) exceed the amount of $100 USD.  You agree to defend, indemnify
// and hold harmless AMD and its licensors, and any of their directors,
// officers, employees, affiliates or agents from and against any and all
// loss, damage, liability and other expenses (including reasonable attorneys'
// fees), resulting from Your use of the Software or violation of the terms and
// conditions of this Agreement.  
//
// U.S. GOVERNMENT RESTRICTED RIGHTS: The Materials are provided with "RESTRICTED
// RIGHTS." Use, duplication, or disclosure by the Government is subject to the
// restrictions as set forth in FAR 52.227-14 and DFAR252.227-7013, et seq., or
// its successor.  Use of the Materials by the Government constitutes
// acknowledgement of AMD's proprietary rights in them.
// 
// EXPORT RESTRICTIONS: The Materials may be subject to export restrictions as
// stated in the Software License Agreement.
//

//--------------------------------------------------------------------------------------
// File: AMD_SDK.h
//
// Library include file, to drag in all AMD SDK helper classes and functions.
//--------------------------------------------------------------------------------------
#ifndef __AMD_SDK_H__
#define __AMD_SDK_H__


#define VENDOR_ID_AMD		(0x1002)
#define VENDOR_ID_NVIDIA	(0x10DE)
#define VENDOR_ID_INTEL		(0x8086)


#include "..\\DXUT\\Core\\DXUT.h"
#include "..\\DXUT\\Core\\DXUTmisc.h"
#include "..\\DXUT\\Optional\\DXUTgui.h"
#include "..\\DXUT\\Optional\\SDKmisc.h"
#include "..\\DXUT\\Optional\\SDKMesh.h"


// AMD helper classes and functions
#include "Timer.h"
#include "ShaderCache.h"
#include "HelperFunctions.h"
#include "Sprite.h" 
#include "Magnify.h"
#include "MagnifyTool.h"
#include "HUD.h"
#include "Geometry.h"
#include "LineRender.h"


// Profile helpers for timing and marking up as D3D perf blocks
#define AMD_PROFILE_RED		D3DCOLOR_XRGB( 255, 0, 0 )
#define AMD_PROFILE_GREEN	D3DCOLOR_XRGB( 0, 255, 0 )
#define AMD_PROFILE_BLUE	D3DCOLOR_XRGB( 0, 0, 255 )


#define AMDProfileBegin( col, name ) DXUT_BeginPerfEvent( col, name ); TIMER_Begin( col, name )
#define AMDProfileEnd() TIMER_End() DXUT_EndPerfEvent();

struct AMDProfileEventClass
{
	AMDProfileEventClass( unsigned int col, LPCWSTR name ) { AMDProfileBegin( col, name ); }
	~AMDProfileEventClass() { AMDProfileEnd() }
};

#define AMDProfileEvent( col, name ) AMDProfileEventClass __amd_profile_event( col, name )



#endif

//--------------------------------------------------------------------------------------
// EOF
//--------------------------------------------------------------------------------------

