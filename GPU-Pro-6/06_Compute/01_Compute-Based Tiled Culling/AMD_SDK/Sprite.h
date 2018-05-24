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
// File: Sprite.h
//
// Sprite class definition. This class provides functionality to render sprites, at a 
// given position and scale. 
//--------------------------------------------------------------------------------------


#ifndef _SPRITE_H_
#define _SPRITE_H_

namespace AMD
{

class Sprite
{
public:

    Sprite();
    ~Sprite();

    HRESULT OnCreateDevice( ID3D11Device* pd3dDevice );
    void OnDestroyDevice();
    void OnResizedSwapChain( const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc );

    HRESULT RenderSprite( ID3D11ShaderResourceView* pTextureView, int nStartPosX,
        int nStartPosY, int nWidth, int nHeight, bool bAlpha, bool bBordered );

    HRESULT RenderSpriteMS( ID3D11ShaderResourceView* pTextureView, int nStartPosX,
        int nStartPosY, int nWidth, int nHeight, int nTextureWidth, int nTextureHeight, 
        bool bAlpha, bool bBordered, int nSampleIndex );

    HRESULT RenderSpriteAsDepth( ID3D11ShaderResourceView* pTextureView, int nStartPosX,
        int nStartPosY, int nWidth, int nHeight, bool bBordered, float fDepthRangeMin, 
        float fDepthRangeMax );
	
    HRESULT RenderSpriteAsDepthMS( ID3D11ShaderResourceView* pTextureView, int nStartPosX,
        int nStartPosY, int nWidth, int nHeight, int nTextureWidth, int nTextureHeight, 
        bool bBordered, float fDepthRangeMin, float fDepthRangeMax, int nSampleIndex );

	HRESULT RenderSpriteVolume( ID3D11ShaderResourceView* pTextureView, int nStartPosX, int nStartPosY, int nMaxWidth, int nSliceSize, bool bBordered );

	void SetSpriteColor( DirectX::XMVECTOR Color );
    void SetBorderColor( DirectX::XMVECTOR Color );
    void SetUVs( float fU1, float fV1, float fU2, float fV2 );
	void EnableScissorTest( bool enable ) { m_EnableScissorTest = enable; }
	void SetPointSample( bool pointSample ) { m_PointSampleMode = pointSample; }

private:

	void RenderBorder();
	void Render();

	// VBs
	ID3D11InputLayout*  m_pVertexLayout;
	ID3D11Buffer*       m_pVertexBuffer;
	ID3D11InputLayout*  m_pBorderVertexLayout;
	ID3D11Buffer*       m_pBorderVertexBuffer;

    // CB
    ID3D11Buffer*		m_pcbSprite;

    // Shaders
    ID3D11VertexShader* m_pSpriteVS;
    ID3D11VertexShader* m_pSpriteBorderVS;
    ID3D11PixelShader*  m_pSpritePS;
    ID3D11PixelShader*  m_pSpriteMSPS;
    ID3D11PixelShader*  m_pSpriteAsDepthPS;
    ID3D11PixelShader*  m_pSpriteAsDepthMSPS;
    ID3D11PixelShader*  m_pSpriteBorderPS;
	ID3D11PixelShader*  m_pSpriteUntexturedPS;
	ID3D11PixelShader*	m_pSpriteVolumePS;

    // States
	bool						m_EnableScissorTest;
	bool						m_PointSampleMode;
    ID3D11SamplerState*         m_pSamplePoint;
    ID3D11SamplerState*         m_pSampleLinear;
    ID3D11RasterizerState*      m_pRasterState;
	ID3D11RasterizerState*      m_pRasterStateWithScissor;
    ID3D11RasterizerState*      m_pEnableCulling;
    ID3D11BlendState*           m_pNoBlending;
    ID3D11BlendState*           m_pSrcAlphaBlending;
    ID3D11DepthStencilState*    m_pDisableDepthTestWrite;
    ID3D11DepthStencilState*    m_pEnableDepthTestWrite;
};

}; // namespace AMD

#endif // _SPRITE_H_


//--------------------------------------------------------------------------------------
// EOF
//--------------------------------------------------------------------------------------
