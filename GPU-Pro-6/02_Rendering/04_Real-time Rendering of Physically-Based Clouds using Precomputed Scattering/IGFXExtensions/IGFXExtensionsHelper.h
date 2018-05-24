//--------------------------------------------------------------------------------------
// Copyright 2011,2012,2013 Intel Corporation
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
#pragma once

#include <D3D11.h>
#include <D3DCompiler.h>

#include "ID3D10Extensions.h"

using namespace ID3D10;

namespace IGFX
{
	struct Extensions
	{
		bool PixelShaderOrdering; 
		bool DirectResourceAccess;
	};

	HRESULT Init( ID3D11Device *pDevice );

	Extensions getAvailableExtensions( ID3D11Device *pd3dDevice );

	HRESULT CreateCPUSharedTexture2D( ID3D11Device *pd3dDevice, const D3D11_TEXTURE2D_DESC *tex2d,   ID3D11Texture2D **pCPUSharedTexture2D );
	
	HRESULT CreateGPUSharedTexture2D( ID3D11Device *pd3dDevice, const D3D11_TEXTURE2D_DESC *tex2d,   ID3D11Texture2D **pGPUSharedTexture2D, D3D11_SUBRESOURCE_DATA *initData = NULL );

	HRESULT CreateSharedTexture2D( ID3D11Device *pd3dDevice, const D3D11_TEXTURE2D_DESC *tex2d_cpu, ID3D11Texture2D **pCPUSharedTexture2D, const D3D11_TEXTURE2D_DESC *tex2d_gpu, ID3D11Texture2D **pGPUSharedTexture2D, D3D11_SUBRESOURCE_DATA *initData = NULL );

}