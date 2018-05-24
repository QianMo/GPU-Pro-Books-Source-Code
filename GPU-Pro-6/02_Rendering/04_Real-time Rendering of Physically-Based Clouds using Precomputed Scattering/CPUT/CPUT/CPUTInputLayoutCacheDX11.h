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
#ifndef __CPUTINPUTLAYOUTCACHERDX11_H__
#define __CPUTINPUTLAYOUTCACHERDX11_H__

#include "CPUTInputLayoutCache.h"
#include "CPUTOSServicesWin.h"
#include "CPUTVertexShaderDX11.h"
#include <D3D11.h> // D3D11_INPUT_ELEMENT_DESC
#include <map>

class CPUTInputLayoutCacheDX11:public CPUTInputLayoutCache
{
public:
    ~CPUTInputLayoutCacheDX11()
    {
        ClearLayoutCache();
    }
    static CPUTInputLayoutCacheDX11 *GetInputLayoutCache();
    static CPUTResult DeleteInputLayoutCache();
    CPUTResult GetLayout(ID3D11Device *pDevice, D3D11_INPUT_ELEMENT_DESC *pDXLayout, CPUTVertexShaderDX11 *pVertexShader, ID3D11InputLayout **ppInputLayout);
	void ClearLayoutCache();

private:
    // singleton
    CPUTInputLayoutCacheDX11() { mLayoutList.clear(); }

    // convert the D3D11_INPUT_ELEMENT_DESC to string key
    cString GenerateLayoutKey(D3D11_INPUT_ELEMENT_DESC *pDXLayout);

    CPUTResult VerifyLayoutCompatibility(D3D11_INPUT_ELEMENT_DESC *pDXLayout, ID3DBlob *pVertexShaderBlob);

    static CPUTInputLayoutCacheDX11 *mpInputLayoutCache;
    std::map<cString, ID3D11InputLayout*> mLayoutList;
};

#endif //#define __CPUTINPUTLAYOUTCACHERDX11_H__