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

#ifndef __RESOURCE_H__
#define __RESOURCE_H__

#include "D3D11Object.h"

enum BufferType { RAW, STRUCTURED, TEXTURE2D };	// different types of buffer
enum BufferBind { SRV, UAV, SRV_AND_UAV, CONSTANT }; // different types of binding for a resource

class D3DResource
{
public:
	D3DResource();
	~D3DResource();

	HRESULT Init( BufferBind a_iBind, BufferType a_iType, VOID* a_InitData, size_t a_SizeInBytes, UINT a_iNumElements, D3D11Object* m_d3dUtil );
	HRESULT CreateUAV( D3D11Object* m_d3dUtil );
	HRESULT CreateSRV( D3D11Object* m_d3dUtil );

	ID3D11Resource*				GetResource() { return m_pResource; }
	ID3D11Resource**			GetPtrResource() { return &m_pResource; }
	ID3D11UnorderedAccessView*	GetUAV() { return m_pUAV; }
	ID3D11UnorderedAccessView**	GetPtrUAV() { return &m_pUAV; }
	ID3D11ShaderResourceView*	GetSRV() { return m_pSRV; }
	ID3D11ShaderResourceView**	GetPtrSRV() { return &m_pSRV; }
private:
	ID3D11Resource*				m_pResource;		// It acts as a void*
	ID3D11UnorderedAccessView*	m_pUAV;				// Not every resource needs both UAV and SRVs
	ID3D11ShaderResourceView*	m_pSRV;				// but it is a practical and fast way to do this
};

#endif