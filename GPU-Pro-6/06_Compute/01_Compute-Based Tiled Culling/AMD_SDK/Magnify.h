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
// File: Magnify.h
//
// Magnify class definition. This class magnifies a region of a given surface, and renders a scaled 
// sprite at the given position on the screen.
//--------------------------------------------------------------------------------------


#ifndef _MAGNIFY_H_
#define _MAGNIFY_H_

namespace AMD
{

class Magnify
{
public:

    // Constructor / destructor
    Magnify();
    ~Magnify();

    // Hooks for the DX SDK Framework
    HRESULT OnCreateDevice( ID3D11Device* pd3dDevice );
    void OnDestroyDevice();
    void OnResizedSwapChain( const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc );

    // Set methods
    void SetPixelRegion( int nPixelRegion );
    void SetScale( int nScale );
    void SetDepthRangeMin( float fDepthRangeMin );
    void SetDepthRangeMax( float fDepthRangeMax );
    void SetSourceResource( ID3D11Resource* pSourceResource, DXGI_FORMAT Format, 
        int nWidth, int nHeight, int nSamples );
    void SetSubSampleIndex( int nSubSampleIndex );

    // Captures a region, at the current cursor position, for magnification
    void Capture( POINT& Point );

    // Render the magnified region, at the capture location
    void RenderBackground();
    void RenderMagnifiedRegion();

private:

    // Private methods
    void SetPosition( int nPositionX, int nPositionY );
    void CreateInternalResources();

private:

    // Magnification settings
    int     m_nPositionX;
    int     m_nPositionY;
    int     m_nPixelRegion;
    int     m_nHalfPixelRegion;
    int     m_nScale;
    float   m_fDepthRangeMin;
    float   m_fDepthRangeMax;
    int     m_nBackBufferWidth;
    int     m_nBackBufferHeight;
    int     m_nSubSampleIndex;

    // Helper class for plotting the magnified region
    Sprite  m_Sprite;

    // Source resource data
    ID3D11Resource*             m_pSourceResource;
    ID3D11Texture2D*            m_pResolvedSourceResource;
    ID3D11Texture2D*            m_pCopySourceResource;
    ID3D11ShaderResourceView*   m_pResolvedSourceResourceSRV;
    ID3D11ShaderResourceView*   m_pCopySourceResourceSRV;
    ID3D11ShaderResourceView*   m_pSourceResourceSRV1;
    DXGI_FORMAT                 m_SourceResourceFormat;
    int                         m_nSourceResourceWidth;
    int                         m_nSourceResourceHeight;
    int                         m_nSourceResourceSamples;
    DXGI_FORMAT                 m_DepthFormat; 
    DXGI_FORMAT                 m_DepthSRVFormat;
    bool                        m_bDepthFormat;
};

}; // namespace AMD

#endif // _MAGNIFY_H_


//--------------------------------------------------------------------------------------
// EOF
//--------------------------------------------------------------------------------------
