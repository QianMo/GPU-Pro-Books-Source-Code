#include <stdafx.h>
#include <Demo.h>
#include <Image.h>

// only not compressed formats
static const unsigned int bytesPerPixel[] = 
{
  0,
  1, 1, 2, 3, 3, 4, 4, 3, 12, // Unsigned formats
  4, 4, 16, 2, 4, 4, 6, 8,
  6, 8, 12, 16,               // Float formats
  2, 4, 4, 8,
  2, 4, 4, 8,                 // Signed formats
  2, 4,                       // Depth formats
  1, 2, 2, 4, 2, 4            // Packed formats
};

// only plain formats 
static const unsigned int bytesPerChannel[] = 
{
  0,
  1, 1, 1, 1, 1, 1, 1, 1, 4,  // Unsigned formats
  1, 1, 4, 2, 2, 4, 2, 2,
  2, 2, 4, 4,                 // Float formats
  2, 2, 4, 4,
  1, 1, 2, 2                  // Signed formats
};

static const unsigned int channelCount[] = 
{
  0,
  1, 1, 2, 3, 3, 4, 4, 3, 3,  // Unsigned formats
  4, 4, 4, 1, 2, 1, 3, 4,
  3, 4, 3, 4,                 // Float formats
  1, 2, 1, 2,
  2, 4, 2, 4,                 // Signed formats
  1, 1,                       // Depth formats
  3, 3, 4, 4, 3, 4,           // Packed formats
  3, 3, 4, 4, 4, 4, 1, 2      // Compressed formats
};

void Image::Release()
{
  SAFE_DELETE_ARRAY(data);
}

bool Image::Load(const char *fileName)
{
  strcpy(name, fileName);
  char filePath[DEMO_MAX_FILEPATH];
  if(!Demo::fileManager->GetFilePath(fileName, filePath))
    return false;

  FILE *file = NULL;
  fopen_s(&file, filePath, "rb");
  if(!file)
    return false;

  DdsHeader header;
  fread(&header, sizeof(header), 1, file);
  if(header.dwMagic != FOURCC('D', 'D', 'S', ' '))
  {
    fclose(file);
    return false;
  }

  width = header.dwWidth;
  height = header.dwHeight;
  depth = header.dwDepth;
  numMipMaps = header.dwMipMapCount;
  if(header.ddsCaps.dwCaps2 & 0x00000200)
    depth = 0;
  else if(depth == 0) 
    depth = 1;
  if(numMipMaps == 0) 
    numMipMaps = 1;

  switch(header.ddpfPixelFormat.dwFourCC)
  {
  case FOURCC('D', 'X', 'T', '1'):
    format = TEX_FORMAT_DXT1;
    break;
  case FOURCC('D', 'X', 'T', '3'):
    format = TEX_FORMAT_DXT3;
    break;
  case FOURCC('D', 'X', 'T', '5'):
    format = TEX_FORMAT_DXT5;
    break;
  case FOURCC('A', 'T', 'I', '1'):
    format = TEX_FORMAT_ATI1N;
    break;
  case FOURCC('A', 'T', 'I', '2'):
    format = TEX_FORMAT_ATI2N;
    break;
  case 34:
    format = TEX_FORMAT_RG16F;
    break;
  case 36:
    format = TEX_FORMAT_RGBA16;
    break;
  case 113:
    format = TEX_FORMAT_RGBA16F;
    break;
  case 116:
    format = TEX_FORMAT_RGBA32F;
    break;
  default:
    switch(header.ddpfPixelFormat.dwRGBBitCount)
    {
    case 8:
      if(header.ddpfPixelFormat.dwRBitMask == 0xE0)
        format = TEX_FORMAT_RGB332;
      else 
        format = TEX_FORMAT_I8;
      break;
    case 16:
      if(header.ddpfPixelFormat.dwRGBAlphaBitMask)
      {
        if(header.ddpfPixelFormat.dwRGBAlphaBitMask == 0x8000)
          format = TEX_FORMAT_RGB5A1;
        else 
          format = TEX_FORMAT_IA8;
      } 
      else
      {
        if(header.ddpfPixelFormat.dwBBitMask == 0x1F)
          format = TEX_FORMAT_RGB565;
        else 
          format = TEX_FORMAT_I16;
      }
      break;
    case 24:
      format = TEX_FORMAT_BGR8;
      break;
    case 32:
      if(header.ddpfPixelFormat.dwRBitMask == 0x3FF00000)
        format = TEX_FORMAT_RGB10A2;
      else
        format = TEX_FORMAT_BGRA8;
      break;
    default:
      fclose(file);
      return false;
    }
  }

  unsigned int size = GetSize(0, numMipMaps);
  data = new unsigned char[size];
  if(!data)
  {
    fclose(file);
    return false;
  }

  if(IsCube())
  {
    for(unsigned int face=0; face<6; face++)
    {
      for(unsigned int mipMapLevel=0; mipMapLevel<numMipMaps; mipMapLevel++)
      {
        unsigned int faceSize = GetSize(mipMapLevel)/6;
        unsigned char *src = GetData(mipMapLevel)+(face*faceSize);
        fread(src, 1, faceSize, file);
      }
    }
  } 
  else 
  {
    fread(data, 1, size, file);
  }

  return true;
}

unsigned int Image::GetWidth(unsigned int mipMapLevel) const
{
  unsigned int a = width >> mipMapLevel;
  return (a==0)? 1 : a;
}

unsigned int Image::GetHeight(unsigned int mipMapLevel) const
{
  unsigned int a = height >> mipMapLevel;
  return (a==0)? 1 : a;
}

unsigned int Image::GetDepth(unsigned int mipMapLevel) const
{
  unsigned int a = depth >> mipMapLevel;
  return (a==0)? 1 : a;
}

unsigned int Image::GetPitch(unsigned int mipMapLevel) const
{
  if(!IsCompressed())
  {
    unsigned int w = GetWidth(mipMapLevel);
    return (w*bytesPerPixel[format]);
  }
  else
  {
    unsigned int bytesPerBlock;
    if((format == TEX_FORMAT_DXT1) || (format == TEX_FORMAT_ATI1N)) 
      bytesPerBlock = 8;
    else
      bytesPerBlock = 16;
    unsigned int w = GetWidth(mipMapLevel);
    w += 3;
    w >>= 2;
    return (w*bytesPerBlock);
  }
}

unsigned int Image::GetSize(unsigned int firstMipMap, unsigned int numMipMapLevels) const
{
  unsigned int w = GetWidth(firstMipMap);
  unsigned int h = GetHeight(firstMipMap);
  unsigned int d = IsCube() ? 1 : GetDepth(firstMipMap);
  unsigned int size = 0;
  while(numMipMapLevels)
  {
    if(IsCompressed())
      size += ((w + 3) >> 2) * ((h + 3) >> 2) * d;
    else 
      size += w * h * d;
    if((w == 1) && (h == 1) && (d == 1)) 
      break;
    if(w > 1) 
      w >>= 1;
    if(h > 1)
      h >>= 1;
    if(d > 1) 
      d >>= 1;
    numMipMapLevels--;
  }
  if(IsCompressed())
  {
    unsigned int bytesPerBlock;
    if((format == TEX_FORMAT_DXT1) || (format == TEX_FORMAT_ATI1N)) 
      bytesPerBlock = 8;
    else
      bytesPerBlock = 16;
    size *= bytesPerBlock;
  }
  else
  {
    size *= bytesPerPixel[format];
  }
  if(IsCube()) 
    size *= 6;
  return size;
}

unsigned char *Image::GetData(unsigned int mipMapLevel) const
{
  return (mipMapLevel<numMipMaps)? (data+GetSize(0, mipMapLevel)) : NULL;
}

unsigned int Image::GetBytesPerPixel(texFormats texFormat)
{
  assert(texFormat < TEX_FORMAT_DXT1);
  return bytesPerPixel[texFormat];
}

unsigned int Image::GetBytesPerChannel(texFormats texFormat)
{
  assert(texFormat <= TEX_FORMAT_UVWQ16);
  return bytesPerChannel[texFormat];
}

unsigned int Image::GetChannelCount(texFormats texFormat)
{
  return channelCount[texFormat]; 
}

bool Image::IsSrgbFormat(texFormats texFormat)
{
  return((texFormat == TEX_FORMAT_BGR8_SRGB) || (texFormat == TEX_FORMAT_BGRA8_SRGB) || (texFormat == TEX_FORMAT_RGBA8_SRGB) ||
         (texFormat == TEX_FORMAT_DXT1_SRGB) || (texFormat == TEX_FORMAT_DXT3_SRGB) || (texFormat == TEX_FORMAT_DXT5_SRGB));
}

texFormats Image::ConvertToSrgbFormat(texFormats texFormat)
{
  if(IsSrgbFormat(texFormat))
  {
    return texFormat;
  }
  else if((texFormat == TEX_FORMAT_BGR8) || (texFormat == TEX_FORMAT_BGRA8) || (texFormat == TEX_FORMAT_RGBA8) ||
          (texFormat == TEX_FORMAT_DXT1) || (texFormat == TEX_FORMAT_DXT3) || (texFormat == TEX_FORMAT_DXT5))
  {
    return ((texFormats)(texFormat+1));
  }
  else
  {
    return TEX_FORMAT_NONE;
  }
}

texFormats Image::ConvertFromSrgbFormat(texFormats texFormat)
{
  if((texFormat == TEX_FORMAT_BGR8) || (texFormat == TEX_FORMAT_BGRA8) || (texFormat == TEX_FORMAT_RGBA8) ||
     (texFormat == TEX_FORMAT_DXT1) || (texFormat == TEX_FORMAT_DXT3) || (texFormat == TEX_FORMAT_DXT5))
  {
    return texFormat;
  }
  else if(IsSrgbFormat(texFormat))
  {
    return ((texFormats)(texFormat-1));
  }
  else
  {
    return TEX_FORMAT_NONE;
  }
}


