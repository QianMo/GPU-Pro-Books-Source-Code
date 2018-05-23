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
#include "CPUTInputLayoutCacheDX11.h"
#include "CPUTVertexShaderDX11.h"

extern const cString *gpDXGIFormatNames;

CPUTInputLayoutCacheDX11* CPUTInputLayoutCacheDX11::mpInputLayoutCache = NULL;

//-----------------------------------------------------------------------------
void CPUTInputLayoutCacheDX11::ClearLayoutCache()
{
	// iterate over the entire map - and release each layout object
    std::map<cString, ID3D11InputLayout*>::iterator mapIterator;

    for(mapIterator = mLayoutList.begin(); mapIterator != mLayoutList.end(); mapIterator++)
    {
        mapIterator->second->Release();  // release the ID3D11InputLayout*
    }
    mLayoutList.clear();
}

// singleton retriever
//-----------------------------------------------------------------------------
CPUTInputLayoutCacheDX11* CPUTInputLayoutCacheDX11::GetInputLayoutCache()
{
    if(NULL == mpInputLayoutCache)
    {
        mpInputLayoutCache = new CPUTInputLayoutCacheDX11();
    }
    return mpInputLayoutCache;
}

// singleton destroy routine
//-----------------------------------------------------------------------------
CPUTResult CPUTInputLayoutCacheDX11::DeleteInputLayoutCache()
{
    if(mpInputLayoutCache)
    {
        delete mpInputLayoutCache;
        mpInputLayoutCache = NULL;
    }
    return CPUT_SUCCESS;
}

// find existing, or create new, ID3D11InputLayout layout
//-----------------------------------------------------------------------------
CPUTResult CPUTInputLayoutCacheDX11::GetLayout(
    ID3D11Device *pDevice,
    D3D11_INPUT_ELEMENT_DESC *pDXLayout,
    CPUTVertexShaderDX11 *pVertexShader,
    ID3D11InputLayout **ppInputLayout
){
    // Generate the vertex layout pattern portion of the key
    cString layoutKey = GenerateLayoutKey(pDXLayout);

    // Append the vertex shader pointer to the key for layout<->vertex shader relationship
    cString address = ptoc(pVertexShader);
    layoutKey += address;

    // Do we already have one like this?
    if( mLayoutList[layoutKey] )
    {
        *ppInputLayout = mLayoutList[layoutKey];
		(*ppInputLayout)->AddRef();
        return CPUT_SUCCESS;
    }
    // Not found, create a new ID3D11InputLayout object

    // How many elements are in the input layout?
    int numInputLayoutElements=0;
    while(NULL != pDXLayout[numInputLayoutElements].SemanticName)
    {
        numInputLayoutElements++;
    }
    // Create the input layout
    HRESULT hr;
    ID3DBlob *pBlob = pVertexShader->GetBlob();
    hr = pDevice->CreateInputLayout( pDXLayout, numInputLayoutElements, pBlob->GetBufferPointer(), pBlob->GetBufferSize(), ppInputLayout );
    ASSERT( SUCCEEDED(hr), _L("Error creating input layout.") );
	CPUTSetDebugName( *ppInputLayout, _L("CPUTInputLayoutCacheDX11::GetLayout()") );

    // Store this layout object in our map
    mLayoutList[layoutKey] = *ppInputLayout;

    // Addref for storing it in our map as well as returning it (count should be = 2 at this point)
    (*ppInputLayout)->AddRef();

    return CPUT_SUCCESS;
}

// Generate a string version of the vertex-buffer's layout.  Allows us to search, compare, etc...
//-----------------------------------------------------------------------------
cString CPUTInputLayoutCacheDX11::GenerateLayoutKey(D3D11_INPUT_ELEMENT_DESC *pDXLayout)
{
    // TODO:  Duh!  We can simply memcmp the DX layouts == use the layout input description directly as the key.
    //        We just need to know how many elements, or NULL terminate it.
    //        Uses less memory, faster, etc...
    //        Duh!

    if( !pDXLayout[0].SemanticName )
    {
        return _L("");
    }
    // TODO: Use shorter names, etc...
    ASSERT( (pDXLayout[0].Format>=0) && (pDXLayout[0].Format<=DXGI_FORMAT_BC7_UNORM_SRGB), _L("Invalid DXGI Format.") );
    // Start first layout entry and no comma.
    cString layoutKey = s2ws(pDXLayout[0].SemanticName) + _L(":") + gpDXGIFormatNames[pDXLayout[0].Format];
    for( int index=1; NULL != pDXLayout[index].SemanticName; index++ )
    {
        ASSERT( (pDXLayout[index].Format>=0) && (pDXLayout[index].Format<=DXGI_FORMAT_BC7_UNORM_SRGB), _L("Invalid DXGI Format.") );
        // Add a comma and the next layout entry
        layoutKey = layoutKey + _L(",") + s2ws(pDXLayout[index].SemanticName) + _L(":") + gpDXGIFormatNames[pDXLayout[index].Format];
    }
    return layoutKey;
}
