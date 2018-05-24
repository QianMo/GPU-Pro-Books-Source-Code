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
// File: ShaderCacheSampleHelper.cpp
//
// Helpers to implement the DXUT related ShaderCache interface in samples.
//--------------------------------------------------------------------------------------

#include "ShaderCacheSampleHelper.h"
#include "..\\DXUT\\Core\\DXUT.h"
#include "..\\DXUT\\Core\\DXUTmisc.h"
#include "..\\DXUT\\Optional\\DXUTgui.h"
#include "..\\DXUT\\Optional\\DXUTCamera.h"
#include "..\\DXUT\\Optional\\DXUTSettingsDlg.h"
#include "..\\DXUT\\Optional\\SDKmisc.h"
#include "..\\DXUT\\Optional\\SDKmesh.h"
#include "ShaderCache.h"
#include "Sprite.h"
#include "HUD.h"

#pragma warning( disable : 4100 ) // disable unreference formal parameter warnings for /W4 builds

static void SetHUDVisibility( AMD::HUD& r_HUD, const bool i_bHUDIsVisible )
{
    using namespace AMD;

    assert( AMD_IDC_BUTTON_SHOW_SHADERCACHE_UI == AMD_IDC_START );
    for( int i = AMD_IDC_BUTTON_SHOW_SHADERCACHE_UI + 1; i < AMD_IDC_END; ++i )
    {
        CDXUTControl* pControl = r_HUD.m_GUI.GetControl( GetEnum(i) );
        if( pControl )
        {
            pControl->SetVisible( i_bHUDIsVisible );
        }
    }
}

namespace AMD
{
	static bool		g_bAdvancedShaderCacheGUI_IsVisible = false;
	HUD				*g_pHUD = NULL;
	ShaderCache		*g_pShaderCache = NULL;

void AMD::InitApp( ShaderCache& r_ShaderCache, HUD& r_HUD, int& iY, const bool i_bAdvancedShaderCacheGUI_VisibleByDefault )
{
#if !AMD_SDK_PREBUILT_RELEASE_EXE
	g_bAdvancedShaderCacheGUI_IsVisible = i_bAdvancedShaderCacheGUI_VisibleByDefault;

	const int i_old_iY = iY;

	g_pHUD = &r_HUD;
	g_pShaderCache = &r_ShaderCache;

	{
		r_HUD.m_GUI.AddButton( GetEnum(AMD_IDC_BUTTON_SHOW_SHADERCACHE_UI),			L"ShaderCache HUD (F9)", AMD::HUD::iElementOffset, iY += AMD::HUD::iElementDelta, AMD::HUD::iElementWidth, AMD::HUD::iElementHeight, VK_F9 );

		iY = 0;
		static const int iSCElementOffset =  AMD::HUD::iElementOffset - 256;

		r_HUD.m_GUI.AddButton( GetEnum(AMD_IDC_BUTTON_RECOMPILESHADERS_CHANGED),	L"Recompile shaders (F5)", iSCElementOffset, iY += AMD::HUD::iElementDelta, AMD::HUD::iElementWidth, AMD::HUD::iElementHeight, VK_F5 );

#if AMD_SDK_INTERNAL_BUILD
		if( r_ShaderCache.GenerateISAGPRPressure() )
		{
			r_HUD.m_GUI.AddButton( GetEnum(AMD_IDC_BUTTON_RECREATE_SHADERS),		L"Gen |ISA| shaders (F6)", iSCElementOffset, iY += AMD::HUD::iElementDelta, AMD::HUD::iElementWidth, AMD::HUD::iElementHeight, VK_F6 );
		}
#endif

		r_HUD.m_GUI.AddButton( GetEnum(AMD_IDC_BUTTON_RECOMPILESHADERS_GLOBAL),		L"Build ALL shaders (F7)", iSCElementOffset, iY += AMD::HUD::iElementDelta, AMD::HUD::iElementWidth, AMD::HUD::iElementHeight, VK_F7 );

		r_HUD.m_GUI.AddCheckBox( GetEnum(AMD_IDC_CHECKBOX_AUTORECOMPILE_SHADERS),	L"Auto Recompile Shaders", iSCElementOffset, iY += AMD::HUD::iElementDelta, AMD::HUD::iElementWidth, AMD::HUD::iElementHeight, r_ShaderCache.RecompileTouchedShaders() );
		if( r_ShaderCache.ShaderErrorDisplayType() == ShaderCache::ERROR_DISPLAY_ON_SCREEN )
		{
			r_HUD.m_GUI.AddCheckBox( GetEnum(AMD_IDC_CHECKBOX_SHOW_SHADER_ERRORS),		L"Show Compiler Errors", iSCElementOffset, iY += AMD::HUD::iElementDelta, AMD::HUD::iElementWidth, AMD::HUD::iElementHeight, r_ShaderCache.ShowShaderErrors() );
		}

#if AMD_SDK_INTERNAL_BUILD
		if( r_ShaderCache.GenerateISAGPRPressure() )
		{
			r_HUD.m_GUI.AddCheckBox( GetEnum(AMD_IDC_CHECKBOX_SHOW_ISA_GPR_PRESSURE), L"Show ISA GPR Pressure", iSCElementOffset, iY += AMD::HUD::iElementDelta, AMD::HUD::iElementWidth, AMD::HUD::iElementHeight, r_ShaderCache.ShowISAGPRPressure() );
			r_HUD.m_GUI.AddStatic( GetEnum(AMD_IDC_STATIC_TARGET_ISA), L"Target ISA:", iSCElementOffset, iY += AMD::HUD::iElementDelta, AMD::HUD::iElementWidth, AMD::HUD::iElementHeight );

			CDXUTComboBox *pCombo;
			r_HUD.m_GUI.AddComboBox( GetEnum(AMD_IDC_COMBOBOX_TARGET_ISA), iSCElementOffset, iY += AMD::HUD::iElementDelta, AMD::HUD::iElementWidth, AMD::HUD::iElementHeight, 0, true, &pCombo );

			if( pCombo )
			{
				pCombo->SetDropHeight( 300 );
				for( int i = AMD::FIRST_ISA_TARGET; i < AMD::NUM_ISA_TARGETS; ++i )
				{
					pCombo->AddItem( AMD::AmdTargetInfo[ i ].m_Name, NULL );
				}
				pCombo->SetSelectedByIndex( AMD::DEFAULT_ISA_TARGET );
			}
			r_HUD.m_GUI.AddStatic( GetEnum(AMD_IDC_STATIC_TARGET_ISA_INFO), L"Press (F6) to add New ISA", iSCElementOffset, iY += AMD::HUD::iElementDelta, AMD::HUD::iElementWidth, AMD::HUD::iElementHeight );
		}
#endif
	}

	SetHUDVisibility( r_HUD, i_bAdvancedShaderCacheGUI_VisibleByDefault );

	iY = i_old_iY + AMD::HUD::iElementDelta;
#endif
}

void AMD::ProcessUIChanges()
{
#if !AMD_SDK_PREBUILT_RELEASE_EXE
	if( g_pHUD == NULL || g_pShaderCache == NULL )
		return;

	ShaderCache& r_ShaderCache = *g_pShaderCache;
	HUD& r_HUD = *g_pHUD;

	r_ShaderCache.SetRecompileTouchedShadersFlag( r_HUD.m_GUI.GetCheckBox( GetEnum(AMD_IDC_CHECKBOX_AUTORECOMPILE_SHADERS) )->GetChecked() );
	if( r_ShaderCache.ShaderErrorDisplayType() == ShaderCache::ERROR_DISPLAY_ON_SCREEN )
	{
		r_ShaderCache.SetShowShaderErrorsFlag( r_HUD.m_GUI.GetCheckBox( GetEnum(AMD_IDC_CHECKBOX_SHOW_SHADER_ERRORS) )->GetChecked() );
	}
#if AMD_SDK_INTERNAL_BUILD
	if( r_ShaderCache.GenerateISAGPRPressure() )
	{
		r_ShaderCache.SetShowShaderISAFlag( r_HUD.m_GUI.GetCheckBox( GetEnum(AMD_IDC_CHECKBOX_SHOW_ISA_GPR_PRESSURE) )->GetChecked() );
	}
#endif
#endif
}

void AMD::RenderHUDUpdates( CDXUTTextHelper* i_pTxtHelper )
{
#if !AMD_SDK_PREBUILT_RELEASE_EXE
	if( g_pHUD == NULL || g_pShaderCache == NULL )
		return;

	ShaderCache& r_ShaderCache = *g_pShaderCache;

	if( r_ShaderCache.ShadersReady() || (r_ShaderCache.ShowShaderErrors() && r_ShaderCache.HasErrorsToDisplay()) )
	{
		assert( i_pTxtHelper );
		r_ShaderCache.RenderShaderErrors( i_pTxtHelper, 15, DirectX::XMVectorSet( 1.0f, 1.0f, 0.0f, 1.0f ) );
#if AMD_SDK_INTERNAL_BUILD
		if( r_ShaderCache.GenerateISAGPRPressure() )
			r_ShaderCache.RenderISAInfo( i_pTxtHelper, 15, DirectX::XMVectorSet( 1.0f, 1.0f, 0.0f, 1.0f ) );
#endif
	}
#endif
}

//--------------------------------------------------------------------------------------
// Handles the GUI events
//--------------------------------------------------------------------------------------
void __stdcall AMD::OnGUIEvent( UINT nEvent, int nControlID, CDXUTControl* pControl, void* pUserContext )
{
#if !AMD_SDK_PREBUILT_RELEASE_EXE
	if( g_pHUD == NULL || g_pShaderCache == NULL )
		return;

	ShaderCache& r_ShaderCache = *g_pShaderCache;
	HUD& r_HUD = *g_pHUD;

	if( nControlID == GetEnum(AMD_IDC_BUTTON_RECOMPILESHADERS_CHANGED) )
	{
		// Errors will post during compile, don't need them right now
		const bool bOldShaderErrorsFlag = r_ShaderCache.ShowShaderErrors();
		r_ShaderCache.SetShowShaderErrorsFlag( false );
		r_ShaderCache.GenerateShaders( AMD::ShaderCache::CREATE_TYPE_COMPILE_CHANGES, true );
		r_ShaderCache.SetShowShaderErrorsFlag( bOldShaderErrorsFlag );
	}
	else if( nControlID == GetEnum(AMD_IDC_BUTTON_RECOMPILESHADERS_GLOBAL) )
	{
		// Errors will post during compile, don't need them right now
		const bool bOldShaderErrorsFlag = r_ShaderCache.ShowShaderErrors();
		r_ShaderCache.SetShowShaderErrorsFlag( false );
		r_ShaderCache.GenerateShaders( AMD::ShaderCache::CREATE_TYPE_FORCE_COMPILE, true );
		r_ShaderCache.SetShowShaderErrorsFlag( bOldShaderErrorsFlag );
	}
#if AMD_SDK_INTERNAL_BUILD
	else if( nControlID == GetEnum(AMD_IDC_BUTTON_RECREATE_SHADERS) )
	{
		assert( r_ShaderCache.GenerateISAGPRPressure() ); // Shouldn't call this if we aren't using ISA
		const bool k_bOK = r_ShaderCache.CloneShaders();
		assert( k_bOK );
		if( k_bOK )
		{
			// Errors won't post using this method... Should find a better one!
			// Need to somehow toggle error rendering off and then on again next frame...
			const bool bOldShaderErrorsFlag = r_ShaderCache.ShowShaderErrors();
			r_ShaderCache.SetShowShaderErrorsFlag( false );
			r_ShaderCache.GenerateShaders( AMD::ShaderCache::CREATE_TYPE_USE_CACHED, true );
			r_ShaderCache.SetShowShaderErrorsFlag( bOldShaderErrorsFlag );
		}
	}
	else if( nControlID == GetEnum(AMD_IDC_COMBOBOX_TARGET_ISA) )
	{
		assert( r_ShaderCache.GenerateISAGPRPressure() ); // Shouldn't call this if we aren't using ISA
		r_ShaderCache.SetTargetISA( (AMD::ISA_TARGET)((CDXUTComboBox*)pControl)->GetSelectedIndex() );
	}
#endif
	else if( nControlID == GetEnum(AMD_IDC_BUTTON_SHOW_SHADERCACHE_UI) )
	{
		// Toggle Render of ShaderCache GUI
		g_bAdvancedShaderCacheGUI_IsVisible = !g_bAdvancedShaderCacheGUI_IsVisible;
		SetHUDVisibility( r_HUD, g_bAdvancedShaderCacheGUI_IsVisible );
	}
#endif
}

}; // namespace AMD

//--------------------------------------------------------------------------------------
// EOF
//--------------------------------------------------------------------------------------
