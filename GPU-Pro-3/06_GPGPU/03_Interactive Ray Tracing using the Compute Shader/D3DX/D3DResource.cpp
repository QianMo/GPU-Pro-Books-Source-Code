// ================================================================================ //
// Copyright (c) 2011 Arturo Garcia, Francisco Avila, Sergio Murguia and Leo Reyes	//
//																					//
// Permission is hereby granted, free of charge, to any person obtaining a copy of	//
// this software and associated documentation files (the "Software"), to deal in	//
// the Software without restriction, including without limitation the rights to		//
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies	//
// of the Software, and to permit persons to whom the Software is furnished to do	//
// so, subject to the following conditions:											//
//																					//
// The above copyright notice and this permission notice shall be included in all	//
// copies or substantial portions of the Software.									//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR		//
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,			//
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE		//
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER			//
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,	//
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE	//
// SOFTWARE.																		//
// ================================================================================ //

#include "D3DResource.h"

D3DResource::D3DResource(void)
{
	m_pResource = NULL;
	m_pSRV = NULL;
	m_pUAV = NULL;
}

D3DResource::~D3DResource(void)
{
	m_pResource->Release();
	m_pSRV->Release();
	m_pUAV->Release();
}

//----------------------------------------------------------------------------------------
// Init resource
//----------------------------------------------------------------------------------------
HRESULT D3DResource::Init( BufferBind a_iBind, BufferType a_iType, VOID* a_InitData, 
						  size_t a_SizeInBytes, UINT a_iNumElements, D3D11Object* m_d3dUtil )
{
	HRESULT hr = S_OK;

	// The implementations is an ugly if/else approach. It is clear that there are
	// better ways to do this, but for now, it works.
	if(a_iType == TEXTURE2D)
	{
		D3D11_TEXTURE2D_DESC dstex;
		ZeroMemory( &dstex, sizeof(dstex) );
		dstex.Width = 1024;
		dstex.Height = 1024;
		dstex.MipLevels = 1;
		dstex.ArraySize = a_iNumElements;
		dstex.SampleDesc.Count = 1;
		dstex.SampleDesc.Quality = 0;
		dstex.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
		dstex.Usage = D3D11_USAGE_DEFAULT;
		dstex.BindFlags = D3D11_BIND_SHADER_RESOURCE;
		dstex.CPUAccessFlags = 0;
		dstex.MiscFlags = 0;
		m_d3dUtil->GetDevice()->CreateTexture2D( &dstex, NULL, (ID3D11Texture2D**)&m_pResource );

		D3D11_SHADER_RESOURCE_VIEW_DESC desc;
		ZeroMemory( &desc, sizeof(desc) );
		desc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2DARRAY;
		desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
		desc.Texture2DArray.MostDetailedMip = 0;
		desc.Texture2DArray.MipLevels = 1;
		desc.Texture2DArray.FirstArraySlice = 0;
		desc.Texture2DArray.ArraySize = a_iNumElements;
		m_d3dUtil->GetDevice()->CreateShaderResourceView( (ID3D11Texture2D*)m_pResource, &desc, &m_pSRV );
	}
	else
	{
		if( a_iBind == CONSTANT)
		{
			HRESULT hr;
			D3D11_BUFFER_DESC bd;
			ZeroMemory( &bd, sizeof(D3D11_BUFFER_DESC) );
			bd.Usage = D3D11_USAGE_DYNAMIC;
			bd.ByteWidth = a_SizeInBytes;
			bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
			bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
			hr = m_d3dUtil->GetDevice()->CreateBuffer( &bd, NULL, (ID3D11Buffer**)&m_pResource );
			if(FAILED(hr))
			{
				return hr;
			}
		}
		else
		{
			D3D11_BUFFER_DESC buffer_desc;
			ZeroMemory( &buffer_desc, sizeof(buffer_desc) );
			buffer_desc.ByteWidth = a_SizeInBytes * a_iNumElements;
			buffer_desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
			buffer_desc.StructureByteStride = a_SizeInBytes;

			if(a_iBind == SRV_AND_UAV)
			{
				buffer_desc.Usage = D3D11_USAGE_DEFAULT;
				buffer_desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
			}
			else if(a_iBind == UAV)
			{
				buffer_desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
			}
			else if(a_iBind == SRV)
			{
				buffer_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
			}
			else
			{
				return E_INVALIDARG;
			}

			if ( a_InitData )
			{
				D3D11_SUBRESOURCE_DATA InitData;
				InitData.pSysMem = a_InitData;
				hr = m_d3dUtil->GetDevice()->CreateBuffer( &buffer_desc, &InitData, (ID3D11Buffer**)&m_pResource );
			} 
			else
			{
				 hr = m_d3dUtil->GetDevice()->CreateBuffer( &buffer_desc, NULL, (ID3D11Buffer**)&m_pResource );
			}

			if ( FAILED( hr ) )
			{
				return hr;
			}
			
			if( a_iBind == UAV || a_iBind ==  SRV_AND_UAV)
			{
				hr = CreateUAV( m_d3dUtil );
				if (FAILED( hr ) )
					return hr;
			}
			if( a_iBind == SRV || a_iBind ==  SRV_AND_UAV )
			{
				hr = CreateSRV( m_d3dUtil );
				if (FAILED( hr ) )
					return hr;
			}
		}
	}

	return hr;
}

//----------------------------------------------------------------------------------------
// Create UAV
//----------------------------------------------------------------------------------------
HRESULT D3DResource::CreateUAV( D3D11Object* m_d3dUtil )
{
	HRESULT hr = S_OK;

	D3D11_BUFFER_DESC descBuf;
	ZeroMemory( &descBuf, sizeof(descBuf) );
	((ID3D11Buffer*)m_pResource)->GetDesc( &descBuf );

	D3D11_UNORDERED_ACCESS_VIEW_DESC desc;
	ZeroMemory( &desc, sizeof(desc) );
	desc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
	desc.Buffer.FirstElement = 0;

	if ( descBuf.MiscFlags & D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS )
	{
		// This is a Raw Buffer
		desc.Format = DXGI_FORMAT_R32_TYPELESS;
		desc.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
		desc.Buffer.NumElements = descBuf.ByteWidth / 4;
	} 
	else if ( descBuf.MiscFlags & D3D11_RESOURCE_MISC_BUFFER_STRUCTURED )
	{
		// This is a Structured Buffer
		desc.Format = DXGI_FORMAT_UNKNOWN;      // Format must be must be DXGI_FORMAT_UNKNOWN, when creating a View of a Structured Buffer
		desc.Buffer.NumElements = descBuf.ByteWidth / descBuf.StructureByteStride; 
	} 
	else
	{
		return E_INVALIDARG;
	}
    
	hr = m_d3dUtil->GetDevice()->CreateUnorderedAccessView( m_pResource, &desc, &m_pUAV );

	return hr;
}

//----------------------------------------------------------------------------------------
// Create SRV
//----------------------------------------------------------------------------------------
HRESULT D3DResource::CreateSRV( D3D11Object* m_d3dUtil )
{
	HRESULT hr = S_OK;

	D3D11_BUFFER_DESC descBuf;
	ZeroMemory( &descBuf, sizeof(descBuf) );
	((ID3D11Buffer*)m_pResource)->GetDesc( &descBuf );

	D3D11_SHADER_RESOURCE_VIEW_DESC desc;
	ZeroMemory( &desc, sizeof(desc) );

	if ( descBuf.MiscFlags & D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS )
	{
		// This is a Raw Buffer
		desc.Format = DXGI_FORMAT_R32_TYPELESS;
		desc.ViewDimension = D3D11_SRV_DIMENSION_BUFFEREX;
		desc.BufferEx.FirstElement = 0;
		desc.BufferEx.Flags = D3D11_BUFFEREX_SRV_FLAG_RAW;
		desc.BufferEx.NumElements = descBuf.ByteWidth / 4;
	} 
	else if ( descBuf.MiscFlags & D3D11_RESOURCE_MISC_BUFFER_STRUCTURED )
	{
		// This is a Structured Buffer
		desc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
		desc.Format = DXGI_FORMAT_UNKNOWN;
		desc.Buffer.NumElements = descBuf.ByteWidth / descBuf.StructureByteStride;
		desc.Buffer.FirstElement = 0;
	} 
	else
	{
		return E_INVALIDARG;
	}

	hr = m_d3dUtil->GetDevice()->CreateShaderResourceView( m_pResource, &desc, &m_pSRV );

	return hr;
}