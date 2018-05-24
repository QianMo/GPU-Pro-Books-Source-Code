//--------------------------------------------------------------------------------------
// Copyright 2013 Intel Corporation
// All Rights Reserved
//
// Permission is granted to use, copy, distribute and prepare derivative works of this
// software for any purpose and without fee, provided, that the above copyright notice
// and this statement appear in all copies.  Intel makes no representations about the
// suitability of this software for any purpose.  THIS SOFTWARE IS PROVIDED "AS IS."
// INTEL SPECIFICALLY DISCLAIMS ALL WARRANTIES, EXPRESS OR IMPLIED, AND ALL LIABILITY,
// INCLUDING CONSEQUENTIAL AND OTHER INDIRECT DAMAGES, FOR THE USE OF THIS SOFTWARE,
// INCLUDING LIABILITY FOR INFRINGEMENT OF ANY PROPRIETARY RIGHTS, AND INCLUDING THE
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  Intel does not
// assume any responsibility for any errors which may appear in this software nor any
// responsibility to update it.
//--------------------------------------------------------------------------------------

#include "CPUT_DX11.h"
#include "CPUTSprite.h"
#include "CPUTAssetLibrary.h"
#include "CPUTMaterialDX11.h"
#include "CPUTTextureDX11.h"

class SpriteVertex
{
public:
    float mpPos[3];
    float mpUV[2];
};

//-----------------------------------------------
CPUTSprite::~CPUTSprite()
{
    SAFE_RELEASE( mpVertexBuffer );
    SAFE_RELEASE( mpMaterial );
    SAFE_RELEASE( mpInputLayout );
}

//-----------------------------------------------
HRESULT CPUTSprite::CreateSprite(
    float          spriteX,
    float          spriteY,
    float          spriteWidth,
    float          spriteHeight,
    const cString &spriteMaterialName
)
{
    HRESULT result;

    // Create resources so we can draw a sprite using the render target as a texture
    mpMaterial = CPUTAssetLibrary::GetAssetLibrary()->GetMaterial( spriteMaterialName, false );

    // Define the input layout
    D3D11_INPUT_ELEMENT_DESC pLayout[] =
    {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0,  0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
        { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT,    0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0 },
        {0}
    };
    CPUTVertexShaderDX11 *pVertexShader = ((CPUTMaterialDX11*)mpMaterial)->GetVertexShader();
    ID3D11Device *pD3dDevice = CPUT_DX11::GetDevice();
    CPUTInputLayoutCacheDX11::GetInputLayoutCache()->GetLayout( pD3dDevice, pLayout, pVertexShader, &mpInputLayout);

    // ***************************************************
    // Create Vertex Buffers
    // ***************************************************
    D3D11_BUFFER_DESC bd;
    bd.Usage = D3D11_USAGE_DYNAMIC;
    bd.ByteWidth = sizeof(SpriteVertex) * 6; // 2 tris, 3 verts each vertices
    bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    bd.MiscFlags = 0;

    const float top    = -spriteY; //-1.0f;
    const float bottom = -spriteY - spriteHeight; // 1.0f;
    const float left   =  spriteX; //-1.0f;
    const float right  =  spriteX + spriteWidth; // 1.0f;
    SpriteVertex pVertices[] = {
        {  left,    top, 1.0f,   0.0f, 0.0f },
        { right,    top, 1.0f,   1.0f, 0.0f },
        {  left, bottom, 1.0f,   0.0f, 1.0f },

        { right,    top, 1.0f,   1.0f, 0.0f },
        { right, bottom, 1.0f,   1.0f, 1.0f },
        {  left, bottom, 1.0f,   0.0f, 1.0f }
    };
    D3D11_SUBRESOURCE_DATA initialData;
    initialData.pSysMem = pVertices;
    initialData.SysMemPitch = sizeof( SpriteVertex );
    initialData.SysMemSlicePitch = 0;

    result = pD3dDevice->CreateBuffer( &bd, &initialData, &mpVertexBuffer );
    ASSERT( SUCCEEDED(result), _L("Failed creating render target debug-sprite vertex buffer") );
    CPUTSetDebugName( mpVertexBuffer, _L("CPUTSprite vertex buffer") );

    return S_OK;
} // CPUTSprite::CreateSprite()

//-----------------------------------------
void CPUTSprite::DrawSprite(
    CPUTRenderParameters &renderParams,
    CPUTMaterial         &material
)
{
    // TODO: Should we warn here?
    // If it doesn't draw, make sure you created it with createDebugSprite == true
    if( mpVertexBuffer )
    {
        ID3D11DeviceContext *pContext = ((CPUTRenderParametersDX*)&renderParams)->mpContext;

        material.SetRenderStates(renderParams);

        UINT stride = sizeof( SpriteVertex );
        UINT offset = 0;
        pContext->IASetVertexBuffers( 0, 1, &mpVertexBuffer, &stride, &offset );

        // Set the input layout
        pContext->IASetInputLayout( mpInputLayout );

        // Set primitive topology
        pContext->IASetPrimitiveTopology( D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST );

        pContext->Draw( 6, 0 );
    }
} // CPUTSprite::DrawSprite()



