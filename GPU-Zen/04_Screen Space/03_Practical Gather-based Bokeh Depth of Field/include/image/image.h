#pragma once


#include "types.h"
#include <essentials/main.h>


using namespace NEssentials;


namespace NImage
{
	bool Load(const string& path, uint8*& data, uint16& width, uint16& height, Format& format);
	bool Save(const string& path, uint8* data, uint16 width, uint16 height, Format format);
	bool IsCompressed(Format format);
	bool Is32(Format format);
	uint8 BPP(Format format);
	uint8 PixelSize(Format format);
	uint32 Size(uint16 width, uint16 height, Format format);
	uint8 MipmapsCount(uint16 width, uint16 height);
	uint8* Scale(uint8* data, uint16 width, uint16 height, Format& format, uint16 scaledWidth, uint16 scaledHeight, Filter filter);
	vector<Mipmap> GenerateMipmaps(uint8* data, uint16 width, uint16 height, Format format, Filter filter);
	void SwapChannels(uint8* data, uint16 width, uint16 height, Format format, uint8 firstChannelIndex, uint8 secondChannelIndex);
	uint8* Shift(uint8* data, uint16 width, uint16 height, Format format, uint16 xOffset, uint16 yOffset);
	uint8* Convert(uint8* data, uint16 width, uint16 height, Format format, Format desiredFormat);
	uint8* Compress(uint8* data, uint16 width, uint16 height, Format desiredFormat); // data must be RGBA8
	uint8* Decompress(uint8* data, uint16 width, uint16 height, Format format);
}
