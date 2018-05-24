#ifndef IMAGE_H
#define IMAGE_H

#define FOURCC(c0, c1, c2, c3) (c0 | (c1 << 8) | (c2 << 16) | (c3 << 24))

enum texFormats 
{
  TEX_FORMAT_NONE             = 0,
  TEX_FORMAT_A8               = 1,
  TEX_FORMAT_I8               = 2,
  TEX_FORMAT_IA8              = 3,
  TEX_FORMAT_BGR8             = 4,
  TEX_FORMAT_BGR8_SRGB        = 5,
  TEX_FORMAT_BGRA8            = 6,
  TEX_FORMAT_BGRA8_SRGB       = 7,
  TEX_FORMAT_RGB8             = 8,
  TEX_FORMAT_RGB32            = 9,
  TEX_FORMAT_RGBA8            = 10,
  TEX_FORMAT_RGBA8_SRGB       = 11,
  TEX_FORMAT_RGBA32  		      = 12,
  TEX_FORMAT_I16     		      = 13,
  TEX_FORMAT_IA16    		      = 14,
  TEX_FORMAT_I32     		      = 15,
  TEX_FORMAT_RGB16   		      = 16,
  TEX_FORMAT_RGBA16  		      = 17,
  TEX_FORMAT_RGB16F  		      = 18,
  TEX_FORMAT_RGBA16F 		      = 19,
  TEX_FORMAT_RGB32F  		      = 20,
  TEX_FORMAT_RGBA32F 		      = 21,
  TEX_FORMAT_I16F    		      = 22,
  TEX_FORMAT_RG16F   		      = 23, 
  TEX_FORMAT_I32F    		      = 24,
  TEX_FORMAT_IA32F   		      = 25,
  TEX_FORMAT_UV8     		      = 26,
  TEX_FORMAT_UVWQ8   		      = 27,
  TEX_FORMAT_UV16    		      = 28,
  TEX_FORMAT_UVWQ16  		      = 29,
  TEX_FORMAT_DEPTH16 		      = 30,
  TEX_FORMAT_DEPTH24_STENCIL8 = 31,
  TEX_FORMAT_RGB332  		      = 32,
  TEX_FORMAT_RGB565  		      = 33,
  TEX_FORMAT_RGB5A1  		      = 34,
  TEX_FORMAT_RGB10A2 		      = 35,
  TEX_FORMAT_UV5L6   		      = 36,
  TEX_FORMAT_UVW10A2 		      = 37,
  TEX_FORMAT_DXT1    		      = 38,
  TEX_FORMAT_DXT1_SRGB        = 39,
  TEX_FORMAT_DXT3             = 40,
  TEX_FORMAT_DXT3_SRGB        = 41,
  TEX_FORMAT_DXT5             = 42,
  TEX_FORMAT_DXT5_SRGB        = 43,
  TEX_FORMAT_ATI1N   		      = 44,
  TEX_FORMAT_ATI2N   		      = 45
};

#pragma pack (push, 1)

struct DdsPixelFormat
{
  unsigned int dwSize;
  unsigned int dwFlags;
  unsigned int dwFourCC;
  unsigned int dwRGBBitCount;
  unsigned int dwRBitMask;
  unsigned int dwGBitMask;
  unsigned int dwBBitMask;
  unsigned int dwRGBAlphaBitMask; 
};

struct DdsCaps
{
  unsigned int dwCaps1;
  unsigned int dwCaps2;
  unsigned int Reserved[2];
};

struct DdsHeader 
{
  unsigned int dwMagic;
  unsigned int dwSize;
  unsigned int dwFlags;
  unsigned int dwHeight;
  unsigned int dwWidth;
  unsigned int dwPitchOrLinearSize;
  unsigned int dwDepth; 
  unsigned int dwMipMapCount;
  unsigned int dwReserved[11];
  DdsPixelFormat ddpfPixelFormat;
  DdsCaps ddsCaps;
  unsigned int dwReserved2;
};

#pragma pack (pop)

// Image
//
class Image
{
public:
  Image():
    data(NULL),
    width(0),
    height(0),
    depth(0),
    numMipMaps(0),
    format(TEX_FORMAT_NONE)
  {
  }

  ~Image()
  {
    Release();
  }

  void Release();

  bool Load(const char *fileName);

  unsigned int GetWidth(unsigned int mipMapLevel=0) const;

  unsigned int GetHeight(unsigned int mipMapLevel=0) const;

  unsigned int GetDepth(unsigned int mipMapLevel=0) const;

  unsigned int GetPitch(unsigned int mipMapLevel=0) const;

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

  bool IsCompressed() const
  {
    return (format >= TEX_FORMAT_DXT1);
  }

  texFormats GetFormat() const
  {
    return format;
  }

  unsigned int GetNumMipMaps() const
  {
    return numMipMaps;
  }

  unsigned int GetSize(unsigned int firstMipMap, unsigned int numMipMapLevels=1) const;

  unsigned char *GetData(unsigned int mipMapLevel=0) const;

  const char* GetName() const
  {
    return name;
  }

  static unsigned int GetBytesPerPixel(texFormats texFormat);

  static unsigned int GetBytesPerChannel(texFormats texFormat);

  static unsigned int GetChannelCount(texFormats texFormat); 

  static bool IsSrgbFormat(texFormats texFormat);

  static texFormats ConvertToSrgbFormat(texFormats texFormat);

  static texFormats ConvertFromSrgbFormat(texFormats texFormat);

private:
  char name[DEMO_MAX_FILENAME]; 
  unsigned char *data;
  unsigned int width, height, depth;
  unsigned int numMipMaps;
  texFormats format;

};

#endif
