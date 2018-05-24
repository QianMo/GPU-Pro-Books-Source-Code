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
// File: HUD.cpp
//
// Class definition for the AMD standard HUD interface.
//--------------------------------------------------------------------------------------


#include "..\\DXUT\\Core\\DXUT.h"
#include "..\\DXUT\\Core\\DXUTmisc.h"
#include "..\\DXUT\\Optional\\DXUTgui.h"
#include "..\\DXUT\\Optional\\SDKmisc.h"
#include "..\\DXUT\\Core\\DDSTextureLoader.h"
#include "Sprite.h"
#include "HUD.h"

using namespace AMD;

//--------------------------------------------------------------------------------------
// Constructor
//--------------------------------------------------------------------------------------
HUD::HUD()
{
    m_pLogoSRV = NULL;
}


//--------------------------------------------------------------------------------------
// Destructor
//--------------------------------------------------------------------------------------
HUD::~HUD()
{
    SAFE_RELEASE( m_pLogoSRV );
}


//--------------------------------------------------------------------------------------
// Device creation hook function, that loads the AMD logo texture, and creates a sprite 
// object
//--------------------------------------------------------------------------------------
HRESULT HUD::OnCreateDevice( ID3D11Device* pd3dDevice )
{
    HRESULT hr;
    wchar_t str[MAX_PATH];

    m_Sprite.OnCreateDevice( pd3dDevice );
    
    V_RETURN( DXUTFindDXSDKMediaFileCch( str, MAX_PATH, L"..\\AMD_SDK\\Media\\AMD.dds" ) );
    DirectX::CreateDDSTextureFromFile( pd3dDevice, str, nullptr, &m_pLogoSRV );
    DXUT_SetDebugName( m_pLogoSRV, "AMD.dds" );

    return hr;
}


//--------------------------------------------------------------------------------------
// Device destruction hook function, that releases the sprite object and 
//--------------------------------------------------------------------------------------
void HUD::OnDestroyDevice()
{
    m_Sprite.OnDestroyDevice();

    SAFE_RELEASE( m_pLogoSRV );
}


//--------------------------------------------------------------------------------------
// Resize swap chain hook function, that passes through to the sprite object 
//--------------------------------------------------------------------------------------
void HUD::OnResizedSwapChain( const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc )
{
    m_Sprite.OnResizedSwapChain( pBackBufferSurfaceDesc );
}


//--------------------------------------------------------------------------------------
// Render hook function, that calls the CDXUTDialog::OnRender method, and additionally 
// renders the AMD sprite
//--------------------------------------------------------------------------------------
void HUD::OnRender( float fElapsedTime )
{
    m_GUI.OnRender( fElapsedTime );
    m_Sprite.RenderSprite( m_pLogoSRV, DXUTGetDXGIBackBufferSurfaceDesc()->Width - 250, DXUTGetDXGIBackBufferSurfaceDesc()->Height, 250, 70, true, false );
}


Slider::Slider( CDXUTDialog& dialog, int id, int& y, const wchar_t* label, int min, int max, int& value ) :
	m_Value( value ),
	m_szLabel( label )
{
	dialog.AddStatic( id + 1000000, L"", AMD::HUD::iElementOffset, y += AMD::HUD::iElementDelta, AMD::HUD::iElementWidth, AMD::HUD::iElementHeight, false, &m_pLabel );
	dialog.AddSlider( id, AMD::HUD::iElementOffset, y += AMD::HUD::iElementDelta, AMD::HUD::iElementWidth, AMD::HUD::iElementHeight, min, max, m_Value, false, &m_pSlider );

	dialog.AddControl( this );

	OnGuiEvent();
}


void Slider::OnGuiEvent()
{
	m_Value = m_pSlider->GetValue();

	wchar_t buff[ 1024 ];
	swprintf_s( buff, 1024, L"%s: %d", m_szLabel, m_Value );

	m_pLabel->SetText( buff );
}


void Slider::SetEnabled( bool enable )
{
	m_pLabel->SetEnabled( enable );
	m_pSlider->SetEnabled( enable );
}


void Slider::SetVisible( bool visible )
{
	m_pLabel->SetVisible( visible );
	m_pSlider->SetVisible( visible );
}


void Slider::SetValue( int value )
{
	m_pSlider->SetValue( value );
	OnGuiEvent();
}

//--------------------------------------------------------------------------------------
// EOF
//--------------------------------------------------------------------------------------