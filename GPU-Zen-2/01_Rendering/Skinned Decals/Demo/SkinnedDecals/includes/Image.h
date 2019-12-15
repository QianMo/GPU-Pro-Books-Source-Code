#ifndef IMAGE_H
#define IMAGE_H

#include <RenderFormat.h>

#define DDS_FOURCC 0x00000004
#define FOURCC(c0, c1, c2, c3) (c0 | (c1 << 8) | (c2 << 16) | (c3 << 24))

#pragma pack (push, 1)

struct DdsPixelFormat
{
  UINT dwSize;
  UINT dwFlags;
  UINT dwFourCC;
  UINT dwRGBBitCount;
  UINT dwRBitMask;
  UINT dwGBitMask;
  UINT dwBBitMask;
  UINT dwRGBAlphaBitMask;
};

struct DdsCaps
{
  UINT dwCaps1;
  UINT dwCaps2;
  UINT Reserved[2];
};

struct DdsHeader
{
  UINT dwMagic;
  UINT dwSize;
  UINT dwFlags;
  UINT dwHeight;
  UINT dwWidth;
  UINT dwPitchOrLinearSize;
  UINT dwDepth;
  UINT dwMipMapCount;
  UINT dwReserved[11];
  DdsPixelFormat ddpfPixelFormat;
  DdsCaps ddsCaps;
  UINT dwReserved2;
};

struct DdsExtHeader
{
  UINT dxgiFormat;
  UINT resourceDimension;
  UINT miscFlag;
  UINT arraySize;
  UINT miscFlags2;
};

#pragma pack (pop)

struct ImageDesc
{
  ImageDesc()
  {
    memset(this, 0, sizeof(ImageDesc));
  }

  char *name;
  UINT width, height, depth;
  UINT numMipMaps;
  renderFormats format;
  unsigned char *data;
};

// Image
//
// Supports loading of dds textures.
class Image
{
public:
  Image():
    data(nullptr),
    width(0),
    height(0),
    depth(0),
    numMipMaps(0),
    format(NONE_RENDER_FORMAT),
    externalData(false)
  {
  }

  ~Image()
  {
    Release();
  }

  void Release();

  bool Load(const char *fileName);

  void Create(const ImageDesc &desc);

  UINT GetWidth(UINT mipMapLevel=0) const;

  UINT GetHeight(UINT mipMapLevel=0) const;

  UINT GetDepth(UINT mipMapLevel=0) const;

  UINT GetPitch(UINT mipMapLevel=0) const;

  bool Is1D() const
  {
    return ((height == 1) && (depth == 1));
  }

  bool Is2D() const
  {
    return ((height > 1) && (depth == 1));
  }

  bool Is3D() const
  {
    return (depth > 1);
  }

  bool IsCube()  const
  {
    return (depth == 0);
  }

  renderFormats GetFormat() const
  {
    return format;
  }

  UINT GetNumMipMaps() const
  {
    return numMipMaps;
  }

  UINT GetSize(UINT firstMipMap, UINT numMipMapLevels=1) const;

  unsigned char *GetData(UINT mipMapLevel = 0) const;

  const char* GetName() const
  {
    return name;
  }

private:
  char name[DEMO_MAX_FILENAME];
  unsigned char *data;
  UINT width, height, depth;
  UINT numMipMaps;
  renderFormats format;
  bool externalData;

};

#endif
