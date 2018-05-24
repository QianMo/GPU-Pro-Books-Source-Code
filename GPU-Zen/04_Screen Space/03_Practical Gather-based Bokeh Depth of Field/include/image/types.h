#pragma once


#include <essentials/types.h>


using namespace NEssentials;


namespace NImage
{
	enum class Format { Unknown, R8, RG8, RGB8, RGBA8, R32, RG32, DXT1, DXT5 };

	enum class Filter { Box, BiCubic, Bilinear, BSpline, CatmullRom, Lanczos3 };

	struct Mipmap
	{
		uint8* data;
		uint16 width, height;
	};
}
