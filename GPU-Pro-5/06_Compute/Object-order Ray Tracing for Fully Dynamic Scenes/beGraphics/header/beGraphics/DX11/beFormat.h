/******************************************************/
/* breeze Engine Graphics Module (c) Tobias Zirr 2011 */
/******************************************************/

#pragma once
#ifndef BE_GRAPHICS_FORMAT_DX11
#define BE_GRAPHICS_FORMAT_DX11

#include "beGraphics.h"
#include "../beFormat.h"
#include "../beEnum.h"
#include <DXGI.h>

namespace beGraphics
{

namespace DX11
{

/// Converts the given format to a corresponding DX11 format, if available.
template <Format::T From>
struct ToFormatDX11;

template <> struct ToFormatDX11<Format::Unknown> { static const DXGI_FORMAT Value = DXGI_FORMAT_UNKNOWN; static const bool Exact = true; };


template <> struct ToFormatDX11<Format::R8U> { static const DXGI_FORMAT Value = DXGI_FORMAT_R8_UNORM; static const bool Exact = true; };
template <> struct ToFormatDX11<Format::R8S> { static const DXGI_FORMAT Value = DXGI_FORMAT_R8_SNORM; static const bool Exact = true; };

template <> struct ToFormatDX11<Format::R8G8U> { static const DXGI_FORMAT Value = DXGI_FORMAT_R8G8_UNORM; static const bool Exact = true; };
template <> struct ToFormatDX11<Format::R8G8S> { static const DXGI_FORMAT Value = DXGI_FORMAT_R8G8_SNORM; static const bool Exact = true; };

template <> struct ToFormatDX11<Format::R8G8B8U> { static const DXGI_FORMAT Value = DXGI_FORMAT_R8G8B8A8_UNORM; static const bool Exact = false; };
template <> struct ToFormatDX11<Format::R8G8B8S> { static const DXGI_FORMAT Value = DXGI_FORMAT_R8G8B8A8_SNORM; static const bool Exact = false; };

template <> struct ToFormatDX11<Format::R8G8B8A8U> { static const DXGI_FORMAT Value = DXGI_FORMAT_R8G8B8A8_UNORM; static const bool Exact = true; };
template <> struct ToFormatDX11<Format::R8G8B8A8S> { static const DXGI_FORMAT Value = DXGI_FORMAT_R8G8B8A8_SNORM; static const bool Exact = true; };
template <> struct ToFormatDX11<Format::R8G8B8X8U> { static const DXGI_FORMAT Value = DXGI_FORMAT_R8G8B8A8_UNORM; static const bool Exact = false; };

template <> struct ToFormatDX11<Format::R8G8B8A8U_SRGB> { static const DXGI_FORMAT Value = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB; static const bool Exact = true; };
template <> struct ToFormatDX11<Format::R8G8B8X8U_SRGB> { static const DXGI_FORMAT Value = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB; static const bool Exact = false; };


template <> struct ToFormatDX11<Format::R16F> { static const DXGI_FORMAT Value = DXGI_FORMAT_R16_FLOAT; static const bool Exact = true; };
template <> struct ToFormatDX11<Format::R16U> { static const DXGI_FORMAT Value = DXGI_FORMAT_R16_UINT; static const bool Exact = true; };
template <> struct ToFormatDX11<Format::R16S> { static const DXGI_FORMAT Value = DXGI_FORMAT_R16_SINT; static const bool Exact = true; };

template <> struct ToFormatDX11<Format::R16G16F> { static const DXGI_FORMAT Value = DXGI_FORMAT_R16G16_FLOAT; static const bool Exact = true; };
template <> struct ToFormatDX11<Format::R16G16U> { static const DXGI_FORMAT Value = DXGI_FORMAT_R16G16_UINT; static const bool Exact = true; };
template <> struct ToFormatDX11<Format::R16G16S> { static const DXGI_FORMAT Value = DXGI_FORMAT_R16G16_SINT; static const bool Exact = true; };

template <> struct ToFormatDX11<Format::R16G16B16F> { static const DXGI_FORMAT Value = DXGI_FORMAT_R16G16B16A16_FLOAT; static const bool Exact = false; };
template <> struct ToFormatDX11<Format::R16G16B16U> { static const DXGI_FORMAT Value = DXGI_FORMAT_R16G16B16A16_UINT; static const bool Exact = false; };
template <> struct ToFormatDX11<Format::R16G16B16S> { static const DXGI_FORMAT Value = DXGI_FORMAT_R16G16B16A16_SINT; static const bool Exact = false; };

template <> struct ToFormatDX11<Format::R16G16B16A16F> { static const DXGI_FORMAT Value = DXGI_FORMAT_R16G16B16A16_FLOAT; static const bool Exact = true; };
template <> struct ToFormatDX11<Format::R16G16B16A16U> { static const DXGI_FORMAT Value = DXGI_FORMAT_R16G16B16A16_UINT; static const bool Exact = true; };
template <> struct ToFormatDX11<Format::R16G16B16A16S> { static const DXGI_FORMAT Value = DXGI_FORMAT_R16G16B16A16_SINT; static const bool Exact = true; };


template <> struct ToFormatDX11<Format::R32F> { static const DXGI_FORMAT Value = DXGI_FORMAT_R32_FLOAT; static const bool Exact = true; };
template <> struct ToFormatDX11<Format::R32U> { static const DXGI_FORMAT Value = DXGI_FORMAT_R32_UINT; static const bool Exact = true; };
template <> struct ToFormatDX11<Format::R32S> { static const DXGI_FORMAT Value = DXGI_FORMAT_R32_SINT; static const bool Exact = true; };

template <> struct ToFormatDX11<Format::R32G32F> { static const DXGI_FORMAT Value = DXGI_FORMAT_R32G32_FLOAT; static const bool Exact = true; };
template <> struct ToFormatDX11<Format::R32G32U> { static const DXGI_FORMAT Value = DXGI_FORMAT_R32G32_UINT; static const bool Exact = true; };
template <> struct ToFormatDX11<Format::R32G32S> { static const DXGI_FORMAT Value = DXGI_FORMAT_R32G32_SINT; static const bool Exact = true; };

template <> struct ToFormatDX11<Format::R32G32B32F> { static const DXGI_FORMAT Value = DXGI_FORMAT_R32G32B32_FLOAT; static const bool Exact = true; };
template <> struct ToFormatDX11<Format::R32G32B32U> { static const DXGI_FORMAT Value = DXGI_FORMAT_R32G32B32_UINT; static const bool Exact = true; };
template <> struct ToFormatDX11<Format::R32G32B32S> { static const DXGI_FORMAT Value = DXGI_FORMAT_R32G32B32_SINT; static const bool Exact = true; };

template <> struct ToFormatDX11<Format::R32G32B32A32F> { static const DXGI_FORMAT Value = DXGI_FORMAT_R32G32B32A32_FLOAT; static const bool Exact = true; };
template <> struct ToFormatDX11<Format::R32G32B32A32U> { static const DXGI_FORMAT Value = DXGI_FORMAT_R32G32B32A32_UINT; static const bool Exact = true; };
template <> struct ToFormatDX11<Format::R32G32B32A32S> { static const DXGI_FORMAT Value = DXGI_FORMAT_R32G32B32A32_SINT; static const bool Exact = true; };


template <> struct ToFormatDX11<Format::D16> { static const DXGI_FORMAT Value = DXGI_FORMAT_D16_UNORM; static const bool Exact = true; };
template <> struct ToFormatDX11<Format::D24S8> { static const DXGI_FORMAT Value = DXGI_FORMAT_D24_UNORM_S8_UINT; static const bool Exact = true; };
template <> struct ToFormatDX11<Format::D32> { static const DXGI_FORMAT Value = DXGI_FORMAT_D32_FLOAT; static const bool Exact = true; };

BE_GRAPHICS_WRAP_ENUM_TEMPLATE(ToFormatDX11, Format::T);

/// Converts the given format to a corresponding DX11 format, if available, returns closest match or unknown otherwise.
LEAN_INLINE DXGI_FORMAT ToAPI(Format::T format)
{
	return EnumTo<DXGI_FORMAT, Format::T, BE_GRAPHICS_WRAPPED_ENUM_TEMPLATE(ToFormatDX11),
		Format::Unknown, Format::End, DXGI_FORMAT_UNKNOWN>(format);
}
/// Converts the given DX11 to a corresponding format, if available, returns unknown otherwise.
LEAN_INLINE Format::T FromAPI(DXGI_FORMAT format)
{
	return EnumFrom<DXGI_FORMAT, Format::T, BE_GRAPHICS_WRAPPED_ENUM_TEMPLATE(ToFormatDX11),
		Format::Unknown, Format::End, Format::Unknown>(format);
}

/// Constructs a DirectX 11 mode description from the given description.
inline DXGI_MODE_DESC ToAPI(const DisplayMode &mode)
{
	DXGI_MODE_DESC modeDX;
	modeDX.Format = ToAPI(mode.Format);
	modeDX.Width = mode.Width;
	modeDX.Height = mode.Height;
	modeDX.RefreshRate.Numerator = mode.Refresh.N;
	modeDX.RefreshRate.Denominator = mode.Refresh.D;
	modeDX.ScanlineOrdering = DXGI_MODE_SCANLINE_ORDER_UNSPECIFIED;
	modeDX.Scaling = DXGI_MODE_SCALING_UNSPECIFIED;
	return modeDX;
}
/// Constructs a display mode description from the given DirectX 11 mode description.
inline DisplayMode FromAPI(const DXGI_MODE_DESC &modeDX)
{
	return DisplayMode(
		modeDX.Width,
		modeDX.Height,
		FromAPI(modeDX.Format),
		RefreshRate(
			modeDX.RefreshRate.Numerator,
			modeDX.RefreshRate.Denominator) );
}

/// Constructs a DirectX 11 mode description from the given description.
inline DXGI_SAMPLE_DESC ToAPI(const SampleDesc &desc)
{
	DXGI_SAMPLE_DESC descDX;
	descDX.Count = desc.Count;
	descDX.Quality = desc.Quality;
	return descDX;
}

/// Constructs a DirectX 11 mode description from the given description.
inline SampleDesc FromAPI(const DXGI_SAMPLE_DESC &desc)
{
	return SampleDesc(
			desc.Count,
			desc.Quality
		);
}

/// Gets the size of the given DirectX GI format.
inline uint4 SizeofFormat(DXGI_FORMAT format)
{
	// Taken from SlimDX
	switch(format)
	{
	case DXGI_FORMAT_R32G32B32A32_TYPELESS:
	case DXGI_FORMAT_R32G32B32A32_FLOAT:
	case DXGI_FORMAT_R32G32B32A32_UINT:
	case DXGI_FORMAT_R32G32B32A32_SINT:
		return 16;

	case DXGI_FORMAT_R32G32B32_TYPELESS:
	case DXGI_FORMAT_R32G32B32_FLOAT:
	case DXGI_FORMAT_R32G32B32_UINT:
	case DXGI_FORMAT_R32G32B32_SINT:
		return 12;

	case DXGI_FORMAT_R16G16B16A16_TYPELESS:
	case DXGI_FORMAT_R16G16B16A16_FLOAT:
	case DXGI_FORMAT_R16G16B16A16_UNORM:
	case DXGI_FORMAT_R16G16B16A16_UINT:
	case DXGI_FORMAT_R16G16B16A16_SNORM:
	case DXGI_FORMAT_R16G16B16A16_SINT:

	case DXGI_FORMAT_R32G32_TYPELESS:
	case DXGI_FORMAT_R32G32_FLOAT:
	case DXGI_FORMAT_R32G32_UINT:
	case DXGI_FORMAT_R32G32_SINT:

	case DXGI_FORMAT_R32G8X24_TYPELESS:
	case DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS:
	case DXGI_FORMAT_X32_TYPELESS_G8X24_UINT:
	case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
		return 8;

	case DXGI_FORMAT_R10G10B10A2_TYPELESS:
	case DXGI_FORMAT_R10G10B10A2_UNORM:
	case DXGI_FORMAT_R10G10B10A2_UINT:

	case DXGI_FORMAT_R11G11B10_FLOAT:

	case DXGI_FORMAT_R8G8B8A8_TYPELESS:
	case DXGI_FORMAT_R8G8B8A8_UNORM:
	case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
	case DXGI_FORMAT_R8G8B8A8_UINT:
	case DXGI_FORMAT_R8G8B8A8_SNORM:
	case DXGI_FORMAT_R8G8B8A8_SINT:

	case DXGI_FORMAT_R16G16_TYPELESS:
	case DXGI_FORMAT_R16G16_FLOAT:
	case DXGI_FORMAT_R16G16_UNORM:
	case DXGI_FORMAT_R16G16_UINT:
	case DXGI_FORMAT_R16G16_SNORM:
	case DXGI_FORMAT_R16G16_SINT:

	case DXGI_FORMAT_R32_TYPELESS:
	case DXGI_FORMAT_R32_FLOAT:
	case DXGI_FORMAT_R32_UINT:
	case DXGI_FORMAT_R32_SINT:

	case DXGI_FORMAT_D32_FLOAT:

	case DXGI_FORMAT_R24G8_TYPELESS:
	case DXGI_FORMAT_R24_UNORM_X8_TYPELESS:
	case DXGI_FORMAT_X24_TYPELESS_G8_UINT:
	case DXGI_FORMAT_D24_UNORM_S8_UINT:

	case DXGI_FORMAT_B8G8R8A8_UNORM:
	case DXGI_FORMAT_B8G8R8X8_UNORM:
		return 4;

	case DXGI_FORMAT_R8G8_TYPELESS:
	case DXGI_FORMAT_R8G8_UNORM:
	case DXGI_FORMAT_R8G8_UINT:
	case DXGI_FORMAT_R8G8_SNORM:
	case DXGI_FORMAT_R8G8_SINT:

	case DXGI_FORMAT_R16_TYPELESS:
	case DXGI_FORMAT_R16_FLOAT:
	case DXGI_FORMAT_R16_UNORM:
	case DXGI_FORMAT_R16_UINT:
	case DXGI_FORMAT_R16_SNORM:
	case DXGI_FORMAT_R16_SINT:

	case DXGI_FORMAT_D16_UNORM:

	case DXGI_FORMAT_B5G6R5_UNORM:
	case DXGI_FORMAT_B5G5R5A1_UNORM:
		return 2;

	case DXGI_FORMAT_R8_TYPELESS:
	case DXGI_FORMAT_R8_UNORM:
	case DXGI_FORMAT_R8_UINT:
	case DXGI_FORMAT_R8_SNORM:
	case DXGI_FORMAT_R8_SINT:
	case DXGI_FORMAT_A8_UNORM:
		return 1;

	case DXGI_FORMAT_BC2_TYPELESS:
	case DXGI_FORMAT_BC2_UNORM:
	case DXGI_FORMAT_BC2_UNORM_SRGB:

	case DXGI_FORMAT_BC3_TYPELESS:
	case DXGI_FORMAT_BC3_UNORM:
	case DXGI_FORMAT_BC3_UNORM_SRGB:

	case DXGI_FORMAT_BC5_TYPELESS:
	case DXGI_FORMAT_BC5_UNORM:
	case DXGI_FORMAT_BC5_SNORM:
		return 16;

	case DXGI_FORMAT_R1_UNORM:

	case DXGI_FORMAT_BC1_TYPELESS:
	case DXGI_FORMAT_BC1_UNORM:
	case DXGI_FORMAT_BC1_UNORM_SRGB:

	case DXGI_FORMAT_BC4_TYPELESS:
	case DXGI_FORMAT_BC4_UNORM:
	case DXGI_FORMAT_BC4_SNORM:
		return 8;

	case DXGI_FORMAT_R9G9B9E5_SHAREDEXP:
		return 4;

	case DXGI_FORMAT_R8G8_B8G8_UNORM:
	case DXGI_FORMAT_G8R8_G8B8_UNORM:
		return 4;

	default:
		return 0;
	}
}

} // namespace

using DX11::ToAPI;

} // namespace

#endif