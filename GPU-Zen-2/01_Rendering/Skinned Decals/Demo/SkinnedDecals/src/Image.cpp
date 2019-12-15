#include <stdafx.h>
#include <Demo.h>
#include <Image.h>

void Image::Release()
{
  if(!externalData)
    SAFE_DELETE_ARRAY(data);
}

bool Image::Load(const char *fileName)
{
  strcpy(name, fileName);
  char filePath[DEMO_MAX_FILEPATH];
  if(!Demo::fileManager->GetFilePath(fileName, filePath))
    return false;

  FILE *file = nullptr;
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
  bool isCubeMap = false;

  if((header.ddpfPixelFormat.dwFlags & DDS_FOURCC) && (header.ddpfPixelFormat.dwFourCC == FOURCC('D', 'X', '1', '0')))
  {
    DdsExtHeader extHeader;
    fread(&extHeader, sizeof(extHeader), 1, file);
    isCubeMap = (extHeader.resourceDimension == 3) && (extHeader.miscFlag & 0x4);
    switch(extHeader.dxgiFormat)
    {
    case 71:
      format = BC1_RENDER_FORMAT;
      break;
    case 74:
      format = BC2_RENDER_FORMAT;
      break;
    case 77:
      format = BC3_RENDER_FORMAT;
      break;
    case 80:
      format = BC4_RENDER_FORMAT;
      break;
    case 83:
      format = BC5_RENDER_FORMAT;
      break;
    case 98:
      format = BC7_RENDER_FORMAT;
      break;
    case 34:
      format = RG16F_RENDER_FORMAT;
      break;
    case 11:
      format = RGBA16_RENDER_FORMAT;
      break;
    case 10:
      format = RGBA16F_RENDER_FORMAT;
      break;
    case 2:
      format = RGBA32F_RENDER_FORMAT;
      break;
    case 61:
      format = R8_RENDER_FORMAT;
      break;
    case 86:
      format = RGB5A1_RENDER_FORMAT;
      break;
    case 49:
      format = RG8_RENDER_FORMAT;
      break;
    case 85:
      format = RGB565_RENDER_FORMAT;
      break;
    case 36:
      format = R16UI_RENDER_FORMAT;
      break;
    case 88:
      format = BGR8_RENDER_FORMAT;
      break;
    case 25:
      format = RGB10A2_RENDER_FORMAT;
      break;
    case 87:
      format = BGRA8_RENDER_FORMAT;
      break;
    default:
      fclose(file);
      return false;
    }
  }
  else
  {
    isCubeMap = (header.ddsCaps.dwCaps2 & 0x00000200);
    switch(header.ddpfPixelFormat.dwFourCC)
    {
    case FOURCC('D', 'X', 'T', '1'):
      format = BC1_RENDER_FORMAT;
      break;
    case FOURCC('D', 'X', 'T', '3'):
      format = BC2_RENDER_FORMAT;
      break;
    case FOURCC('D', 'X', 'T', '5'):
      format = BC3_RENDER_FORMAT;
      break;
    case FOURCC('A', 'T', 'I', '1'):
      format = BC4_RENDER_FORMAT;
      break;
    case FOURCC('A', 'T', 'I', '2'):
      format = BC5_RENDER_FORMAT;
      break;
    case 34:
      format = RG16F_RENDER_FORMAT;
      break;
    case 36:
      format = RGBA16_RENDER_FORMAT;
      break;
    case 113:
      format = RGBA16F_RENDER_FORMAT;
      break;
    case 116:
      format = RGBA32F_RENDER_FORMAT;
      break;
    default:
      switch(header.ddpfPixelFormat.dwRGBBitCount)
      {
      case 8:
        assert(header.ddpfPixelFormat.dwRBitMask != 0xE0);
        format = R8_RENDER_FORMAT;
        break;
      case 16:
        if(header.ddpfPixelFormat.dwRGBAlphaBitMask)
        {
          if(header.ddpfPixelFormat.dwRGBAlphaBitMask == 0x8000)
            format = RGB5A1_RENDER_FORMAT;
          else
            format = RG8_RENDER_FORMAT;
        }
        else
        {
          if(header.ddpfPixelFormat.dwBBitMask == 0x1F)
            format = RGB565_RENDER_FORMAT;
          else
            format = R16UI_RENDER_FORMAT;
        }
        break;
      case 24:
        format = BGR8_RENDER_FORMAT;
        break;
      case 32:
        if(header.ddpfPixelFormat.dwRBitMask == 0x3FF00000)
          format = RGB10A2_RENDER_FORMAT;
        else
          format = BGRA8_RENDER_FORMAT;
        break;
      default:
        fclose(file);
        return false;
      }
    }
  }

  if(isCubeMap)
    depth = 0;
  else if(depth == 0)
    depth = 1;
  if(numMipMaps == 0)
    numMipMaps = 1;

  UINT size = GetSize(0, numMipMaps);
  data = new unsigned char[size];
  if(!data)
  {
    fclose(file);
    return false;
  }

  if(IsCube())
  {
    for(UINT face=0; face<6; face++)
    {
      for(UINT mipMapLevel=0; mipMapLevel<numMipMaps; mipMapLevel++)
      {
        UINT faceSize = GetSize(mipMapLevel) / 6;
        unsigned char *src = GetData(mipMapLevel) + (face * faceSize);
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

void Image::Create(const ImageDesc &desc)
{
  strcpy(name, desc.name);
  width = desc.width;
  height = desc.height;
  depth = desc.depth;
  numMipMaps = desc.numMipMaps;
  format = desc.format;
  data = desc.data;
  externalData = true;
}

UINT Image::GetWidth(UINT mipMapLevel) const
{
  UINT a = width >> mipMapLevel;
  return (a == 0) ? 1 : a;
}

UINT Image::GetHeight(UINT mipMapLevel) const
{
  UINT a = height >> mipMapLevel;
  return (a == 0) ? 1 : a;
}

UINT Image::GetDepth(UINT mipMapLevel) const
{
  UINT a = depth >> mipMapLevel;
  return (a == 0) ? 1 : a;
}

UINT Image::GetPitch(UINT mipMapLevel) const
{
  if(!RenderFormat::IsCompressed(format))
  {
    UINT w = GetWidth(mipMapLevel);
    return (w * RenderFormat::GetBytesPerPixel(format));
  }
  else
  {
    UINT bytesPerBlock;
    if((format == BC1_RENDER_FORMAT) || (format == BC4_RENDER_FORMAT))
      bytesPerBlock = 8;
    else
      bytesPerBlock = 16;
    UINT w = GetWidth(mipMapLevel);
    w += 3;
    w >>= 2;
    return (w * bytesPerBlock);
  }
}

UINT Image::GetSize(UINT firstMipMap, UINT numMipMapLevels) const
{
  UINT w = GetWidth(firstMipMap);
  UINT h = GetHeight(firstMipMap);
  UINT d = IsCube() ? 1 : GetDepth(firstMipMap);
  UINT size = 0;
  while(numMipMapLevels)
  {
    if(RenderFormat::IsCompressed(format))
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
  if(RenderFormat::IsCompressed(format))
  {
    UINT bytesPerBlock;
    if((format == BC1_RENDER_FORMAT) || (format == BC4_RENDER_FORMAT))
      bytesPerBlock = 8;
    else
      bytesPerBlock = 16;
    size *= bytesPerBlock;
  }
  else
  {
    size *= RenderFormat::GetBytesPerPixel(format);
  }
  if(IsCube())
    size *= 6;
  return size;
}

unsigned char *Image::GetData(UINT mipMapLevel) const
{
  return (mipMapLevel < numMipMaps) ? (data + GetSize(0, mipMapLevel)) : nullptr;
}
