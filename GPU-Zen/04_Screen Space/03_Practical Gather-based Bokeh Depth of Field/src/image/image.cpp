#include <image/image.h>
#include <image/main.h>
#include <essentials/macros.h>
#include <math/common.h>

#include <FreeImage.h>
#include <squish.h>


using namespace NMath;


bool NImage::Load(const string& path, uint8*& data, uint16& width, uint16& height, Format& format)
{
	FREE_IMAGE_FORMAT fif;
	FIBITMAP* fib;

	fif = FreeImage_GetFileType(path.c_str());
	if (fif == FIF_UNKNOWN)
		fif = FreeImage_GetFIFFromFilename(path.c_str());

	if (fif != FIF_BMP && fif != FIF_JPEG && fif != FIF_PNG && fif != FIF_TARGA)
	{
		SAFE_CALL(freeImageCustomOutputMessageFunction)("ERROR: Can't recognize image format: " + path);
		return false;
	}

	if (FreeImage_FIFSupportsReading(fif))
		fib = FreeImage_Load(fif, path.c_str());

	if (!fib)
	{
		SAFE_CALL(freeImageCustomOutputMessageFunction)("ERROR: Can't load image data from file: " + path);
		return false;
	}

	uint8 channelsCount = 0;

	width = FreeImage_GetWidth(fib);
	height = FreeImage_GetHeight(fib);
	if (FreeImage_GetBPP(fib) == 8)
	{
		channelsCount = 1;
		format = Format::R8;
	}
	else if (FreeImage_GetBPP(fib) == 16)
	{
		channelsCount = 2;
		format = Format::RG8;
	}
	else if (FreeImage_GetBPP(fib) == 24)
	{
		channelsCount = 3;
		format = Format::RGB8;
	}
	else if (FreeImage_GetBPP(fib) == 32)
	{
		channelsCount = 4;
		format = Format::RGBA8;
	}
	else
	{
		SAFE_CALL(freeImageCustomOutputMessageFunction)("ERROR: Can't recognize image BPP: " + path);
		return false;
	}

	uint32 dataSize = width * height * channelsCount;
	data = new uint8[dataSize];
	memcpy(data, FreeImage_GetBits(fib), dataSize);

	FreeImage_Unload(fib);

	SAFE_CALL(freeImageCustomOutputMessageFunction)("OK: Image file loaded: " + path);

	return true;
}


bool NImage::Save(const string& path, uint8* data, uint16 width, uint16 height, Format format)
{
	FREE_IMAGE_FORMAT fif;
	FIBITMAP* fib;

	if (IsCompressed(format) || Is32(format))
	{
		SAFE_CALL(freeImageCustomOutputMessageFunction)("ERROR: Can't export data in given format: " + path);
		return false;
	}

	fif = FreeImage_GetFIFFromFilename(path.c_str());

	if (fif != FIF_BMP && fif != FIF_JPEG && fif != FIF_PNG && fif != FIF_TARGA)
	{
		SAFE_CALL(freeImageCustomOutputMessageFunction)("ERROR: Can't recognize file's image format: " + path);
		return false;
	}

	fib = FreeImage_Allocate(width, height, BPP(format));

	uint8 *fiData = FreeImage_GetBits(fib);
	memcpy(fiData, data, Size(width, height, format));

	FreeImage_Save((FREE_IMAGE_FORMAT)fif, fib, path.c_str());

	FreeImage_Unload(fib);

	SAFE_CALL(freeImageCustomOutputMessageFunction)("OK: Image file saved: " + path);

	return true;
}


bool NImage::IsCompressed(Format format)
{
	if (
		format == Format::DXT1 ||
		format == Format::DXT5)
	{
		return true;
	}
	else
	{
		return false;
	}
}


bool NImage::Is32(Format format)
{
	if (
		format == Format::R32 ||
		format == Format::RG32)
	{
		return true;
	}
	else
	{
		return false;
	}
}


uint8 NImage::BPP(Format format)
{
	if (format == Format::R8)
	{
		return 8;
	}
	else if (format == Format::RG8)
	{
		return 16;
	}
	else if (format == Format::RGB8)
	{
		return 24;
	}
	else if (format == Format::RGBA8)
	{
		return 32;
	}
	else if (format == Format::R32)
	{
		return 32;
	}
	else if (format == Format::RG32)
	{
		return 64;
	}
	else
	{
		return 0;
	}
}


uint8 NImage::PixelSize(Format format)
{
	if (format == Format::R8)
	{
		return 1;
	}
	else if (format == Format::RG8)
	{
		return 2;
	}
	else if (format == Format::RGB8)
	{
		return 3;
	}
	else if (format == Format::RGBA8)
	{
		return 4;
	}
	else if (format == Format::R32)
	{
		return 4;
	}
	else if (format == Format::RG32)
	{
		return 8;
	}
	else
	{
		return 0;
	}
}


uint32 NImage::Size(uint16 width, uint16 height, Format format)
{
	if (format == Format::DXT1)
	{
		int blockSize = 8;
		int blocksCount = ( (width + 3)/4 ) * ( (height + 3)/4 );
		return blockSize * blocksCount;
	}
	else if (format == Format::DXT5)
	{
		int blockSize = 16;
		int blocksCount = ( (width + 3)/4 ) * ( (height + 3)/4 );
		return blockSize * blocksCount;
	}
	else
	{
		return PixelSize(format) * width * height;
	}
}


uint8 NImage::MipmapsCount(uint16 width, uint16 height)
{
	uint8 mipmapsCount = 1;
	uint16 size = Max(width, height);

	if (size == 1)
		return 1;

	do
	{
		size >>= 1;
		mipmapsCount++;
	}
	while (size != 1);

	return mipmapsCount;
}


uint8* NImage::Scale(uint8* data, uint16 width, uint16 height, Format& format, uint16 scaledWidth, uint16 scaledHeight, Filter filter)
{
	if (IsCompressed(format) || Is32(format))
	{
		SAFE_CALL(freeImageCustomOutputMessageFunction)("ERROR: Can't scale data in given format");
		return nullptr;
	}

	uint32 scaledDataSize = Size(scaledWidth, scaledHeight, format);
	uint8* scaledData = new uint8[scaledDataSize];

	FIBITMAP* fibInput = FreeImage_Allocate(width, height, BPP(format));
	memcpy(FreeImage_GetBits(fibInput), data, Size(width, height, format));

	FIBITMAP* fibOutput = FreeImage_Rescale(fibInput, scaledWidth, scaledHeight, (FREE_IMAGE_FILTER)filter);
	memcpy(scaledData, FreeImage_GetBits(fibOutput), scaledDataSize);

	FreeImage_Unload(fibInput);
	FreeImage_Unload(fibOutput);

	return scaledData;
}


vector<NImage::Mipmap> NImage::GenerateMipmaps(uint8* data, uint16 width, uint16 height, Format format, Filter filter)
{
	vector<Mipmap> mipmaps;

	if (IsCompressed(format) || Is32(format))
	{
		SAFE_CALL(freeImageCustomOutputMessageFunction)("ERROR: Can't generate mipmaps in given format");
		return mipmaps;
	}
	if (!IsPowerOfTwo(width) || !IsPowerOfTwo(height))
	{
		SAFE_CALL(freeImageCustomOutputMessageFunction)("ERROR: Can't generate mipmaps for non-power-of-two image");
		return mipmaps;
	}

	uint8 mipmapsCount = MipmapsCount(width, height);
	mipmaps.resize(mipmapsCount);
	mipmaps[0].data = new uint8[Size(width, height, format)];

	for (uint8 i = 0; i < mipmapsCount; i++)
	{
		mipmaps[i].width = width;
		mipmaps[i].height = height;

		if (width > 1)
			width >>= 1;
		if (height > 1)
			height >>= 1;
	}

	memcpy(mipmaps[0].data, data, Size(mipmaps[0].width, mipmaps[0].height, format));

	for (uint8 i = 1; i < mipmaps.size(); i++)
		mipmaps[i].data = Scale(mipmaps[i-1].data, mipmaps[i-1].width, mipmaps[i-1].height, format, mipmaps[i].width, mipmaps[i].height, filter);

	return mipmaps;
}


void NImage::SwapChannels(uint8* data, uint16 width, uint16 height, Format format, uint8 firstChannelIndex, uint8 secondChannelIndex)
{
	if (IsCompressed(format))
	{
		SAFE_CALL(freeImageCustomOutputMessageFunction)("ERROR: Can't swap channels in given format");
		return;
	}

	uint8 pixelSize = PixelSize(format);
	uint32 pixelsCount = Size(width, height, format) / pixelSize;

	for (uint32 i = 0; i < pixelsCount; i++)
	{
		uint8 temp = data[i*pixelSize + firstChannelIndex];
		data[i*pixelSize + firstChannelIndex] = data[i*pixelSize + secondChannelIndex];
		data[i*pixelSize + secondChannelIndex] = temp;
	}
}


uint8* NImage::Shift(uint8* data, uint16 width, uint16 height, Format format, uint16 xOffset, uint16 yOffset)
{
	if (IsCompressed(format))
	{
		SAFE_CALL(freeImageCustomOutputMessageFunction)("ERROR: Can't shift data in given format");
		return nullptr;
	}

	// normalize offsets

	if (xOffset >= 0)
	{
		xOffset = xOffset % width;
	}
	else
	{
		while (xOffset < 0)
			xOffset += width;
	}

	if (yOffset >= 0)
	{
		yOffset = yOffset % height;
	}
	else
	{
		while (yOffset < 0)
			yOffset += height;
	}

	// shift

	uint8 pixelSize = PixelSize(format);
	uint32 dataSize = Size(width, height, format);
	uint8* shiftedData = new uint8[dataSize];

	for (uint16 j = 0; j < height; j++)
	{
		for (uint16 i = 0; i < width; i++)
		{
			int x = (i + xOffset) % width;
			int y = (j + yOffset) % height;

			uint8* srcData = &data[pixelSize * Idx(i, j, width)];
			uint8* dstData = &shiftedData[pixelSize * Idx(x, y, width)];

			memcpy(dstData, srcData, pixelSize);
		}
	}

	//

	return shiftedData;
}


uint8* NImage::Convert(uint8* data, uint16 width, uint16 height, Format format, Format desiredFormat)
{
	if (format != Format::RGB8)
	{
		SAFE_CALL(freeImageCustomOutputMessageFunction)("ERROR: Can't convert from this format");
		return nullptr;
	}
	if (desiredFormat != Format::RGBA8)
	{
		SAFE_CALL(freeImageCustomOutputMessageFunction)("ERROR: Can't convert to desired format");
		return nullptr;
	}

	uint32 convertedDataSize = Size(width, height, desiredFormat);
	uint8* convertedData = new uint8[convertedDataSize];

	for (uint16 j = 0; j < height; j++)
	{
		for (uint16 i = 0; i < width; i++)
		{
			int pixelIndex = Idx(i, j, width);

			convertedData[4*pixelIndex + 0] = data[3*pixelIndex + 0];
			convertedData[4*pixelIndex + 1] = data[3*pixelIndex + 1];
			convertedData[4*pixelIndex + 2] = data[3*pixelIndex + 2];
			convertedData[4*pixelIndex + 3] = 255;
		}
	}

	return convertedData;
}


uint8* NImage::Compress(uint8* data, uint16 width, uint16 height, Format desiredFormat)
{
	if (!IsCompressed(desiredFormat))
	{
		SAFE_CALL(freeImageCustomOutputMessageFunction)("ERROR: Can't compress to desired format");
		return nullptr;
	}

	int flags = 0;

	if (desiredFormat == Format::DXT1)
		flags = squish::kDxt1;
	else if (desiredFormat == Format::DXT5)
		flags = squish::kDxt5;

	uint32 dataSize = Size(width, height, Format::RGBA8);
	uint32 compressedDataSize = squish::GetStorageRequirements(width, height, flags);
	uint8* compressedData = new uint8[compressedDataSize];

	squish::CompressImage(data, width, height, compressedData, flags);

	return compressedData;
}


uint8* NImage::Decompress(uint8* data, uint16 width, uint16 height, Format format)
{
	if (!IsCompressed(format))
	{
		SAFE_CALL(freeImageCustomOutputMessageFunction)("ERROR: Can't decompress data in given format");
		return nullptr;
	}

	int flags = 0;

	if (format == Format::DXT1)
		flags = squish::kDxt1;
	else if (format == Format::DXT5)
		flags = squish::kDxt5;

	uint32 dataSize = Size(width, height, Format::RGBA8);
	uint32 compressedDataSize = squish::GetStorageRequirements(width, height, flags);
	uint8* decompressedData = new uint8[dataSize];

	squish::DecompressImage(decompressedData, width, height, data, flags);

	return decompressedData;
}
