//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (C) Tobias Zirr.  All Rights Reserved.
//
//////////////////////////////////////////////////////////////////////////////

#pragma once

#include "D3DEffectsLiteCommon.h"

/// @addtogroup D3DEffectsLite D3D Effects Lite
/// @{

/// Resolves hooks in the given uncompiled effect.
D3DEFFECTSLITE_API D3DEffectsLiteBlob* D3DEFFECTSLITE_STDCALL D3DELHookEffect(const void *bytes, UINT byteCount,
																			  D3DEffectsLiteInclude *include, const char *const *hooked, UINT hookedCount,
																			  const char *srcName, const char *pCustomPreamble,
																			  D3DEffectsLiteAllocator *pScratchAllocator);

#ifndef D3DEFFECTSLITE_NO_CPP

namespace D3DEffectsLite
{
	/// Resolves hooks in the given uncompiled effect.
	D3DEFFECTSLITE_API Blob* D3DEFFECTSLITE_STDCALL HookEffect(
		const void *bytes, UINT byteCount,
		Include *include, const char *const *hooked, UINT hookedCount,
		const char *srcName = nullptr, const char *pCustomPreamble = nullptr,
		Allocator *pScratchAllocator = nullptr);

} // namespace

#endif

/// @}