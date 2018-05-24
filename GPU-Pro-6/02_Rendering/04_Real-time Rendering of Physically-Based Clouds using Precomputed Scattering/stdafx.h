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

#define WIN32_LEAN_AND_MEAN

#pragma warning (disable: 4100 4127) // warning C4100:  unreferenced formal parameter;  warning C4127:  conditional expression is constant

#pragma warning (push)
#pragma warning (disable: 4201) // nonstandard extension used : nameless struct/union

//
// windows headers
//
#include <sdkddkver.h>
#include <Windows.h>
#include <tchar.h>
#include <atlcomcli.h> // for CComPtr support

//
// C++ headers
//

#include <cassert>
#include <stdexcept>
#include <algorithm>
#include <sstream>
#include <fstream>
#include <map>
#include <set>
#include <vector>
#include <list>
#include <ctime>

//
// DirectX headers
//
#include <D3D11.h>
#include <D3DX11.h>
#include <D3DX10math.h>
#include <xnamath.h>

#pragma warning (pop)

#include "Errors.h"

#include "CPUT.h"

#if defined(DEBUG) || defined(_DEBUG)
#ifndef V
#define V(x)           { hr = (x); assert(SUCCEEDED(hr)); }
#endif
#ifndef V_RETURN
#define V_RETURN(x)    { hr = (x); assert(SUCCEEDED(hr)); if( FAILED(hr) ) { return hr; } }
#endif
#else
#ifndef V
#define V(x)           { hr = (x); }
#endif
#ifndef V_RETURN
#define V_RETURN(x)    { hr = (x); if( FAILED(hr) ) { return hr; } }
#endif
#endif

//#define assert(x) if(!(x)) _CrtDbgBreak(); else {}

inline void UnbindPSResources(ID3D11DeviceContext *pCtx)
{
	ID3D11ShaderResourceView *pSRVs[20] = {NULL};
	pCtx->PSSetShaderResources(0, _countof(pSRVs), pSRVs);
}

inline void UnbindVSResources(ID3D11DeviceContext *pCtx)
{
	ID3D11ShaderResourceView *pSRVs[8] = {NULL};
	pCtx->VSSetShaderResources(0, _countof(pSRVs), pSRVs);
}

inline void UnbindGSResources(ID3D11DeviceContext *pCtx)
{
	ID3D11ShaderResourceView *pSRVs[8] = {NULL};
	pCtx->GSSetShaderResources(0, _countof(pSRVs), pSRVs);
}

inline void UpdateConstantBuffer(ID3D11DeviceContext *pDeviceCtx, ID3D11Buffer *pCB, const void *pData, size_t DataSize)
{
    D3D11_MAPPED_SUBRESOURCE MappedData;
    pDeviceCtx->Map(pCB, 0, D3D11_MAP_WRITE_DISCARD, 0, &MappedData);
    memcpy(MappedData.pData, pData, DataSize);
    pDeviceCtx->Unmap(pCB, 0);
}

// end of file
