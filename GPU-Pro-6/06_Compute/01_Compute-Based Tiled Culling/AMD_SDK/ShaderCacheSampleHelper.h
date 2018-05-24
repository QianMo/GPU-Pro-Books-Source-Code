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
// File: ShaderCacheSampleHelper.h
//
// Helpers to implement the DXUT related ShaderCache interface in samples.
//--------------------------------------------------------------------------------------


#pragma once

class CDXUTControl;
class CDXUTTextHelper;
typedef long HRESULT;
typedef unsigned int UINT;

namespace AMD
{

class HUD;
class ShaderCache;

//--------------------------------------------------------------------------------------
// UI control IDs
//--------------------------------------------------------------------------------------
extern const int	g_MaxApplicationControlID;
	
enum 
{
												AMD_IDC_START = 0,
	AMD_IDC_BUTTON_SHOW_SHADERCACHE_UI		 =	AMD_IDC_START,
	AMD_IDC_BUTTON_RECOMPILESHADERS_CHANGED,
	AMD_IDC_BUTTON_RECOMPILESHADERS_GLOBAL,
	AMD_IDC_BUTTON_RECREATE_SHADERS,
	AMD_IDC_CHECKBOX_AUTORECOMPILE_SHADERS,
	AMD_IDC_CHECKBOX_SHOW_SHADER_ERRORS,
	AMD_IDC_CHECKBOX_SHOW_ISA_GPR_PRESSURE,
	AMD_IDC_STATIC_TARGET_ISA,
	AMD_IDC_STATIC_TARGET_ISA_INFO,
	AMD_IDC_COMBOBOX_TARGET_ISA,
												AMD_IDC_END
};

template< typename T >
T GetEnum( T i_AMDEnum )
{
	return static_cast< T > ( g_MaxApplicationControlID + i_AMDEnum );
}

void InitApp( ShaderCache& r_ShaderCache, HUD& r_HUD, int& iY, const bool i_bAdvancedShaderCacheGUI_VisibleByDefault = false );
void ProcessUIChanges();
void RenderHUDUpdates( CDXUTTextHelper* i_pTxtHelper );
void __stdcall OnGUIEvent( UINT nEvent, int nControlID, CDXUTControl* pControl, void* pUserContext );

}; // namespace AMD

//--------------------------------------------------------------------------------------
// EOF
//--------------------------------------------------------------------------------------
