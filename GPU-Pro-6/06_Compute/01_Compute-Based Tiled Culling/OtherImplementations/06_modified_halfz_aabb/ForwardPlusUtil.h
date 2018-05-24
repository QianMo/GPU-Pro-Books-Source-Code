//
// Copyright 2014 ADVANCED MICRO DEVICES, INC.  All Rights Reserved.
//
// AMD is granting you permission to use this software and documentation (if
// any) (collectively, the "Materials") pursuant to the terms and conditions
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
// File: ForwardPlusUtil.h
//
// Helper functions for the ComputeBasedTiledCulling sample.
//--------------------------------------------------------------------------------------

#pragma once

#include "..\\DXUT\\Core\\DXUT.h"
#include "CommonConstants.h"

// Forward declarations
namespace AMD
{
    class ShaderCache;
}
namespace ComputeBasedTiledCulling
{
    struct GuiState;
    struct DepthStencilBuffer;
    struct Scene;
    class CommonUtil;
    class LightUtil;
}

namespace ComputeBasedTiledCulling
{
    class ForwardPlusUtil
    {
    public:
        // Constructor / destructor
        ForwardPlusUtil();
        ~ForwardPlusUtil();

        void AddShadersToCache( AMD::ShaderCache *pShaderCache );

        // Various hook functions
        HRESULT OnCreateDevice( ID3D11Device* pd3dDevice );
        void OnDestroyDevice();
        HRESULT OnResizedSwapChain( ID3D11Device* pd3dDevice, const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc );
        void OnReleasingSwapChain();
        void OnRender( float fElapsedTime, const GuiState& CurrentGuiState, const DepthStencilBuffer& DepthStencilBuffer, const Scene& Scene, const CommonUtil& CommonUtil, const LightUtil& LightUtil );

    private:
        ID3D11PixelShader * GetScenePS( bool bAlphaTestEnabled ) const;

    private:
        // shaders for Forward+
        ID3D11VertexShader*         m_pScenePositionOnlyVS;
        ID3D11VertexShader*         m_pScenePositionAndTexVS;
        ID3D11VertexShader*         m_pSceneForwardVS;
        ID3D11PixelShader*          m_pSceneAlphaTestOnlyPS;
        ID3D11InputLayout*          m_pLayoutPositionOnly11;
        ID3D11InputLayout*          m_pLayoutPositionAndTex11;
        ID3D11InputLayout*          m_pLayoutForward11;

        static const int NUM_FORWARD_PIXEL_SHADERS = 2;  // alpha test on/off
        ID3D11PixelShader*          m_pSceneForwardPS[NUM_FORWARD_PIXEL_SHADERS];

        // state for Forward+
        ID3D11BlendState*           m_pBlendStateOpaque;
        ID3D11BlendState*           m_pBlendStateOpaqueDepthOnly;
        ID3D11BlendState*           m_pBlendStateAlphaToCoverageDepthOnly;
        ID3D11BlendState*           m_pBlendStateAlpha;
    };

} // namespace ComputeBasedTiledCulling

//--------------------------------------------------------------------------------------
// EOF
//--------------------------------------------------------------------------------------
