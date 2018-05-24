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
// File: MagnifyTool.h
//
// MagnifyTool class definition. This class implements a user interface based upon the DXUT framework,
// for the Magnify class. 
//--------------------------------------------------------------------------------------


#ifndef _MAGNIFYTOOL_H_
#define _MAGNIFYTOOL_H_

namespace AMD
{

// GUI defines
enum 
{
	IDC_MAGNIFY_STATIC_CAPTION   =  19 + 1024,
	IDC_MAGNIFY_CHECKBOX_ENABLE,
	IDC_MAGNIFY_CHECKBOX_STICKY,
	IDC_MAGNIFY_STATIC_PIXEL_REGION,
	IDC_MAGNIFY_SLIDER_PIXEL_REGION,
	IDC_MAGNIFY_STATIC_SCALE,
	IDC_MAGNIFY_SLIDER_SCALE
};


class MagnifyTool
{
public:

    // Constructor / destructor
    MagnifyTool();
    ~MagnifyTool();

    // Set methods
    void SetSourceResources( ID3D11Resource* pSourceRTResource, DXGI_FORMAT RTFormat, 
        int nWidth, int nHeight, int nSamples );
    void SetPixelRegion( int nPixelRegion ) { m_Magnify.SetPixelRegion( nPixelRegion ); }
    void SetScale( int nScale ) { m_Magnify.SetScale( nScale ); }
        
    // Hooks for the DX SDK Framework
    void InitApp( CDXUTDialog* pUI, int& iStartHeight, bool bSupportStickyMode = false );
    HRESULT OnCreateDevice( ID3D11Device* pd3dDevice );
    void OnResizedSwapChain( ID3D11Device* pd3dDevice, IDXGISwapChain *pSwapChain, 
        const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc, void* pUserContext, 
        int nPositionX, int nPositionY );
    void OnDestroyDevice();
    void OnGUIEvent( UINT nEvent, int nControlID, CDXUTControl* pControl, void* pUserContext );
	bool IsEnabled();
	
    // Render
    void Render();

private:

    // UI helper methods
    void EnableTool( bool bEnable );
    void EnableUI( bool bEnable );

private:

    // The DXUT dialog
    CDXUTDialog* m_pMagnifyUI;

    // Pointer to the Magnify class
    AMD::Magnify m_Magnify;

    // The source resources
    ID3D11Resource* m_pSourceRTResource;
    DXGI_FORMAT     m_RTFormat;
    int             m_nWidth;
    int             m_nHeight;
    int             m_nSamples;
    bool            m_bReleaseRTOnResize;
	bool			m_bMouseDownLastFrame;
	bool			m_bStickyShowing;
	POINT			m_StickyPoint;
};

} // namespace AMD

#endif // _MAGNIFYTOOL_H_


//--------------------------------------------------------------------------------------
// EOF
//--------------------------------------------------------------------------------------
