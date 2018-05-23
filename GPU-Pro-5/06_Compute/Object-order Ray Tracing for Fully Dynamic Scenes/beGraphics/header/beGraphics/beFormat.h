/******************************************************/
/* breeze Engine Graphics Module (c) Tobias Zirr 2011 */
/******************************************************/

#pragma once
#ifndef BE_GRAPHICS_FORMAT
#define BE_GRAPHICS_FORMAT

#include "beGraphics.h"

namespace beGraphics
{

/// Enumeration of data / pixel formats.
struct Format
{
	/// Enumeration.
	enum T
	{
		Unknown = 0,	///< Unknown format / none


		R8U,			///< 8-bit unsigned integer, 1 component (n/a as independent data format)
		R8S,			///< 8-bit signed integer, 1 component (n/a as independent data format)

		R8G8U,			///< 8-bit unsigned integer, 2 components (n/a as independent data format)
		R8G8S,			///< 8-bit signed integer, 2 components (n/a as independent data format)

		R8G8B8U,		///< 8-bit unsigned integer, 3 components (n/a as independent data format)
		R8G8B8S,		///< 8-bit signed integer, 3 components (n/a as independent data format)

		R8G8B8A8U,		///< 8-bit unsigned integer, 4 components
		R8G8B8A8S,		///< 8-bit signed integer, 4 components (n/a as independent data format)
		R8G8B8X8U,		///< 8-bit unsigned integer, 3 components (monitor format)


		R8G8B8A8U_SRGB,	///< 8-bit unsigned integer, 4 components, gamma-corrected
		R8G8B8X8U_SRGB,	///< 8-bit unsigned integer, 3 components (monitor format), gamma-corrected


		R16F,			///< 16-bit floating point, 1 component (n/a as independent data format)
		R16U,			///< 16-bit unsigned integer, 1 component (n/a as independent data format)
		R16S,			///< 16-bit singed integer, 1 component (n/a as independent pixel / data format)

		R16G16F,		///< 16-bit floating point, 2 components
		R16G16U,		///< 16-bit unsigned integer, 2 components (n/a as independent data format)
		R16G16S,		///< 16-bit singed integer, 2 components

		R16G16B16F,		///< 16-bit floating point, 3 components (n/a as independent pixel / data format)
		R16G16B16U,		///< 16-bit unsigned integer, 3 components (n/a as independent pixel / data format)
		R16G16B16S,		///< 16-bit singed integer, 3 components (n/a as independent pixel / data format)

		R16G16B16A16F,	///< 16-bit floating point, 4 components
		R16G16B16A16U,	///< 16-bit unsigned integer, 4 components (n/a as independent data format)
		R16G16B16A16S,	///< 16-bit singed integer, 4 components


		R32F,			///< 32-bit floating point, 1 component
		R32U,			///< 32-bit unsigned integer, 1 component (n/a as independent pixel / data format)
		R32S,			///< 32-bit singed integer, 1 component (n/a as independent pixel / data format)

		R32G32F,		///< 32-bit floating point, 2 components
		R32G32U,		///< 32-bit unsigned integer, 2 components (n/a as independent pixel / data format)
		R32G32S,		///< 32-bit singed integer, 2 components (n/a as independent pixel / data format)

		R32G32B32F,		///< 32-bit floating point, 3 components (n/a as independent pixel format)
		R32G32B32U,		///< 32-bit unsigned integer, 3 components (n/a as independent pixel / data format)
		R32G32B32S,		///< 32-bit singed integer, 3 components (n/a as independent pixel / data format)

		R32G32B32A32F,	///< 32-bit floating point, 4 components
		R32G32B32A32U,	///< 32-bit unsigned integer, 4 components (n/a as independent pixel / data format)
		R32G32B32A32S,	///< 32-bit singed integer, 4 components (n/a as independent pixel / data format)


		D16,			///< 16-bit depth
		D24S8,			///< 24-bit depth, 8-bit stencil
		D32,			///< 32-bit depth

		End
	};
	LEAN_MAKE_ENUM_STRUCT(Format)
};

/// Multisampling type description.
struct SampleDesc
{
	uint4 Count;	///< Number of samples per pixel.
	uint4 Quality;	///< Anti-aliasing quality level.

	/// Constructor.
	explicit SampleDesc(uint4 count = 1, uint4 quality = 0)
		: Count(count), Quality(quality) { }
};

/// Refresh rate.
struct RefreshRate
{
	uint4 N;	///< Refresh rate numerator.
	uint4 D;	///< Refresh rate denominator.

	/// Constructor.
	explicit RefreshRate(uint4 numerator = 0, uint4 denominator = 1)
		: N(numerator), D(denominator) { }

	/// Casts the refresh rate to integer, truncating any decimals.
	uint4 ToInt() const { return N / max<uint4>(D, 1); }
	/// Casts the refresh rate to floating-point.
	float ToFloat() const { return N / max(static_cast<float>(D), 1.0f); }
};

/// Display mode description.
struct DisplayMode
{
	uint4 Width;			///< Resolution.
	uint4 Height;			///< Resolution.
	RefreshRate Refresh;	///< Refresh rate.
	Format::T Format;		///< Back buffer format.

	/// Constructor.
	explicit DisplayMode(uint4 width = 0,
		uint4 height = 0,
		Format::T format = Format::Unknown,
		RefreshRate refresh = RefreshRate())
			: Width(width),
			Height(height),
			Format(format),
			Refresh(refresh) { }
};

} // namespace

#endif