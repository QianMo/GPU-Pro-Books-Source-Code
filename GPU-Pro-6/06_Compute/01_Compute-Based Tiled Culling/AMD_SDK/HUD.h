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
// File: HUD.h
//
// Class definition for the AMD standard HUD interface.
//--------------------------------------------------------------------------------------

#pragma once

namespace AMD
{

class HUD
{
public:

    // AMD standard HUD defines for GUI spacing
    static const int iElementDelta = 25;
    static const int iGroupDelta = ( iElementDelta * 2 );
    static const int iDialogWidth = 250;
    static const int iElementHeight = 24;
    static const int iElementWidth = 170;
    static const int iElementOffset = ( iDialogWidth - iElementWidth ) / 2;
    static const int iElementDropHeight = 35;

    // Public access to the CDXUTDialog is allowed for ease of use in the sample
    CDXUTDialog m_GUI;

    // Constructor / destructor
    HUD();
    ~HUD();

    // Various hook functions
    HRESULT OnCreateDevice( ID3D11Device* pd3dDevice );
    void OnDestroyDevice();
    void OnResizedSwapChain( const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc );
    void OnRender( float fElapsedTime );

private:

    // The private AMD logo texture, and sprite object
    Sprite                      m_Sprite;
    ID3D11ShaderResourceView*   m_pLogoSRV;
};


class Slider : public CDXUTControl
{
public:

	Slider( CDXUTDialog& dialog, int id, int& y, const wchar_t* label, int min, int max, int& value );
	virtual ~Slider() {}

	void OnGuiEvent();
	void SetEnabled( bool enable );
	void SetVisible( bool visible );
	void SetValue( int value );

private:

	Slider& operator=( const Slider& );

	int&			m_Value;
	const wchar_t*	m_szLabel;
	CDXUTSlider*	m_pSlider;
	CDXUTStatic*	m_pLabel;
};



} // namespace AMD

//--------------------------------------------------------------------------------------
// EOF
//--------------------------------------------------------------------------------------
