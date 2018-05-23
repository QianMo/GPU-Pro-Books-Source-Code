/******************************************************/
/* breeze Engine Graphics Module (c) Tobias Zirr 2011 */
/******************************************************/

#pragma once
#ifndef BE_GRAPHICS_QUERY_DX11
#define BE_GRAPHICS_QUERY_DX11

#include "beGraphics.h"
#include <lean/smart/com_ptr.h>
#include <D3D11.h>

namespace beGraphics
{

namespace DX11
{

/// Creates a timing query.
BE_GRAPHICS_DX11_API lean::com_ptr<ID3D11Query, true> CreateTimingQuery(ID3D11Device *device);
/// Creates a timestamp query.
BE_GRAPHICS_DX11_API lean::com_ptr<ID3D11Query, true> CreateTimestampQuery(ID3D11Device *device);

/// Gets the frequency from the given timing query.
BE_GRAPHICS_DX11_API uint8 GetTimingFrequency(ID3D11DeviceContext *context, ID3D11Query *timingQuery);
/// Gets the time stamp from the given timer query.
BE_GRAPHICS_DX11_API uint8 GetTimestamp(ID3D11DeviceContext *context, ID3D11Query *timerQuery);

} // namespace

} // namespace

#endif