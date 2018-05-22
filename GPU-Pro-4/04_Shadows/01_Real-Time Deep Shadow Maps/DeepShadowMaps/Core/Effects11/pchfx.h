//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (C) Microsoft Corporation.  All Rights Reserved.
//
//  File:       pchfx.h
//  Content:    D3D shader effects precompiled header
//
//////////////////////////////////////////////////////////////////////////////

#ifndef __D3DX11_PCHFX_H__
#define __D3DX11_PCHFX_H__

#include "d3d11.h"
#include "d3dx11.h"
#undef DEFINE_GUID
#include "INITGUID.h"
#include "Inc\d3dx11effect.h"

#define UNUSED -1

//////////////////////////////////////////////////////////////////////////

#define offsetof_fx( a, b ) (UINT)offsetof( a, b )

#include "Inc\d3dxGlobal.h"

#include <stddef.h>
#include <strsafe.h>

#include "Effect.h"
#include "Binary\EffectStateBase11.h"
#include "EffectLoad.h"

#include "D3DCompiler.h"

//////////////////////////////////////////////////////////////////////////

namespace D3DX11Effects
{
} // end namespace D3DX11Effects

#endif // __D3DX11_PCHFX_H__
