#include <stdafx.h>
#include <RenderFormat.h>

// only not compressed formats
static const UINT bytesPerPixel[] =
{
  0,
  1, 2, 3, 3, 4, 4, 3, 12, 4,  // Unsigned formats
  4, 16, 2, 4, 4, 4, 6, 8,
  4, 4,                        // Signed formats
  6, 8, 12, 16,                // Float formats
  2, 4, 4, 8,
  2, 4, 4,                     // Depth formats
  2, 2, 4, 4                   // Packed formats
};

// only plain formats 
static const UINT bytesPerChannel[] =
{
  0,
  1, 1, 1, 1, 1, 1, 1, 4, 1,   // Unsigned formats
  1, 4, 2, 2, 2, 4, 2, 2,
  1, 2,                        // Signed formats
  2, 2, 4, 4,                  // Float formats
  2, 2, 4, 4
};

static const UINT channelCount[] =
{
  0,
  1, 2, 3, 3, 4, 4, 3, 3, 4,   // Unsigned formats
  4, 4, 1, 2, 2, 1, 3, 4,
  4, 2,                        // Signed formats
  3, 4, 3, 4,                  // Float formats
  1, 2, 1, 2,
  1, 1, 1,                     // Depth formats
  3, 4, 4, 3,                  // Packed formats
  3, 3, 4, 4, 4, 4, 1, 2, 4, 4 // Compressed formats
};

static const DX12_RenderFormat dx12RenderFormats[]
{
  DXGI_FORMAT_UNKNOWN,            DXGI_FORMAT_UNKNOWN,               DXGI_FORMAT_UNKNOWN,
  DXGI_FORMAT_R8_UNORM,           DXGI_FORMAT_R8_UNORM,              DXGI_FORMAT_R8_UNORM,
  DXGI_FORMAT_R8G8_UNORM,         DXGI_FORMAT_R8G8_UNORM,            DXGI_FORMAT_R8G8_UNORM,
  DXGI_FORMAT_B8G8R8X8_TYPELESS,  DXGI_FORMAT_B8G8R8X8_UNORM,        DXGI_FORMAT_B8G8R8X8_UNORM,
  DXGI_FORMAT_B8G8R8X8_TYPELESS,  DXGI_FORMAT_B8G8R8X8_UNORM_SRGB,   DXGI_FORMAT_B8G8R8X8_UNORM_SRGB,
  DXGI_FORMAT_B8G8R8A8_TYPELESS,  DXGI_FORMAT_B8G8R8A8_UNORM,        DXGI_FORMAT_B8G8R8A8_UNORM,
  DXGI_FORMAT_B8G8R8A8_TYPELESS,  DXGI_FORMAT_B8G8R8A8_UNORM_SRGB,   DXGI_FORMAT_B8G8R8A8_UNORM_SRGB,
  DXGI_FORMAT_R8G8B8A8_TYPELESS,  DXGI_FORMAT_R8G8B8A8_UNORM,        DXGI_FORMAT_R8G8B8A8_UNORM,
  DXGI_FORMAT_R32G32B32_UINT,     DXGI_FORMAT_R32G32B32_UINT,        DXGI_FORMAT_R32G32B32_UINT,
  DXGI_FORMAT_R8G8B8A8_TYPELESS,  DXGI_FORMAT_R8G8B8A8_UNORM,        DXGI_FORMAT_R8G8B8A8_UNORM,
  DXGI_FORMAT_R8G8B8A8_TYPELESS,  DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,   DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,
  DXGI_FORMAT_R32G32B32A32_UINT,  DXGI_FORMAT_R32G32B32A32_UINT,     DXGI_FORMAT_R32G32B32A32_UINT,
  DXGI_FORMAT_R16_UINT,           DXGI_FORMAT_R16_UINT,              DXGI_FORMAT_R16_UINT,
  DXGI_FORMAT_R16G16_UNORM,       DXGI_FORMAT_R16G16_UNORM,          DXGI_FORMAT_R16G16_UNORM,
  DXGI_FORMAT_R16G16_UINT,        DXGI_FORMAT_R16G16_UINT,           DXGI_FORMAT_R16G16_UINT,
  DXGI_FORMAT_R32_UINT,           DXGI_FORMAT_R32_UINT,              DXGI_FORMAT_R32_UINT,
  DXGI_FORMAT_R16G16B16A16_UNORM, DXGI_FORMAT_R16G16B16A16_UNORM,    DXGI_FORMAT_R16G16B16A16_UNORM,
  DXGI_FORMAT_R16G16B16A16_UNORM, DXGI_FORMAT_R16G16B16A16_UNORM,    DXGI_FORMAT_R16G16B16A16_UNORM,
  DXGI_FORMAT_R8G8B8A8_SNORM,     DXGI_FORMAT_R8G8B8A8_SNORM,        DXGI_FORMAT_R8G8B8A8_SNORM,
  DXGI_FORMAT_R16G16_SNORM,       DXGI_FORMAT_R16G16_SNORM,          DXGI_FORMAT_R16G16_SNORM,
  DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R16G16B16A16_FLOAT,    DXGI_FORMAT_R16G16B16A16_FLOAT,
  DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R16G16B16A16_FLOAT,    DXGI_FORMAT_R16G16B16A16_FLOAT,
  DXGI_FORMAT_R32G32B32_FLOAT,    DXGI_FORMAT_R32G32B32_FLOAT,       DXGI_FORMAT_R32G32B32_FLOAT,
  DXGI_FORMAT_R32G32B32A32_FLOAT, DXGI_FORMAT_R32G32B32A32_FLOAT,    DXGI_FORMAT_R32G32B32A32_FLOAT,
  DXGI_FORMAT_R16_FLOAT,          DXGI_FORMAT_R16_FLOAT,             DXGI_FORMAT_R16_FLOAT,
  DXGI_FORMAT_R16G16_FLOAT,       DXGI_FORMAT_R16G16_FLOAT,          DXGI_FORMAT_R16G16_FLOAT,
  DXGI_FORMAT_R32_FLOAT,          DXGI_FORMAT_R32_FLOAT,             DXGI_FORMAT_R32_FLOAT,
  DXGI_FORMAT_R32G32_FLOAT,       DXGI_FORMAT_R32G32_FLOAT,          DXGI_FORMAT_R32G32_FLOAT,
  DXGI_FORMAT_R16_TYPELESS,       DXGI_FORMAT_R16_UNORM,             DXGI_FORMAT_D16_UNORM,
  DXGI_FORMAT_R24G8_TYPELESS,     DXGI_FORMAT_R24_UNORM_X8_TYPELESS, DXGI_FORMAT_D24_UNORM_S8_UINT,
  DXGI_FORMAT_R32_TYPELESS,       DXGI_FORMAT_R32_FLOAT,             DXGI_FORMAT_D32_FLOAT,
  DXGI_FORMAT_B5G6R5_UNORM,       DXGI_FORMAT_B5G6R5_UNORM,          DXGI_FORMAT_B5G6R5_UNORM,
  DXGI_FORMAT_B5G5R5A1_UNORM,     DXGI_FORMAT_B5G5R5A1_UNORM,        DXGI_FORMAT_B5G5R5A1_UNORM,
  DXGI_FORMAT_R10G10B10A2_UNORM,  DXGI_FORMAT_R10G10B10A2_UNORM,     DXGI_FORMAT_R10G10B10A2_UNORM,
  DXGI_FORMAT_R11G11B10_FLOAT,    DXGI_FORMAT_R11G11B10_FLOAT,		   DXGI_FORMAT_R11G11B10_FLOAT,
  DXGI_FORMAT_BC1_TYPELESS,       DXGI_FORMAT_BC1_UNORM,             DXGI_FORMAT_UNKNOWN,
  DXGI_FORMAT_BC1_TYPELESS,       DXGI_FORMAT_BC1_UNORM_SRGB,        DXGI_FORMAT_UNKNOWN,
  DXGI_FORMAT_BC2_TYPELESS,       DXGI_FORMAT_BC2_UNORM,             DXGI_FORMAT_UNKNOWN,
  DXGI_FORMAT_BC2_TYPELESS,       DXGI_FORMAT_BC2_UNORM_SRGB,        DXGI_FORMAT_UNKNOWN,
  DXGI_FORMAT_BC3_TYPELESS,       DXGI_FORMAT_BC3_UNORM,             DXGI_FORMAT_UNKNOWN,
  DXGI_FORMAT_BC3_TYPELESS,       DXGI_FORMAT_BC3_UNORM_SRGB,        DXGI_FORMAT_UNKNOWN,
  DXGI_FORMAT_BC4_TYPELESS,       DXGI_FORMAT_BC4_UNORM,             DXGI_FORMAT_UNKNOWN,
  DXGI_FORMAT_BC5_TYPELESS,       DXGI_FORMAT_BC5_UNORM,             DXGI_FORMAT_UNKNOWN,
  DXGI_FORMAT_BC7_TYPELESS,       DXGI_FORMAT_BC7_UNORM,             DXGI_FORMAT_UNKNOWN,
  DXGI_FORMAT_BC7_TYPELESS,       DXGI_FORMAT_BC7_UNORM_SRGB,        DXGI_FORMAT_UNKNOWN
};

UINT RenderFormat::GetBytesPerPixel(renderFormats renderFormat)
{
  assert(renderFormat < BC1_RENDER_FORMAT);
  return bytesPerPixel[renderFormat];
}

UINT RenderFormat::GetBytesPerChannel(renderFormats renderFormat)
{
  assert(renderFormat <= RG32F_RENDER_FORMAT);
  return bytesPerChannel[renderFormat];
}

UINT RenderFormat::GetChannelCount(renderFormats renderFormat)
{
  assert((renderFormat != BC7_RENDER_FORMAT) && (renderFormat != BC7_SRGB_RENDER_FORMAT)); // BC7 formats can have 3 or 4 channels
  return channelCount[renderFormat];
}

bool RenderFormat::IsCompressed(renderFormats renderFormat)
{
  return (renderFormat >= BC1_RENDER_FORMAT);
}

bool RenderFormat::IsSrgbFormat(renderFormats renderFormat)
{
  return((renderFormat == BGR8_SRGB_RENDER_FORMAT) || (renderFormat == BGRA8_SRGB_RENDER_FORMAT) ||
         (renderFormat == RGBA8_SRGB_RENDER_FORMAT) || (renderFormat == BC1_SRGB_RENDER_FORMAT) ||
         (renderFormat == BC2_SRGB_RENDER_FORMAT) || (renderFormat == BC3_SRGB_RENDER_FORMAT) ||
         (renderFormat == BC7_SRGB_RENDER_FORMAT));
}

renderFormats RenderFormat::ConvertToSrgbFormat(renderFormats renderFormat)
{
  if(IsSrgbFormat(renderFormat))
  {
    return renderFormat;
  }
  else if((renderFormat == BGR8_RENDER_FORMAT) || (renderFormat == BGRA8_RENDER_FORMAT) ||
          (renderFormat == RGBA8_RENDER_FORMAT) || (renderFormat == BC1_RENDER_FORMAT) ||
          (renderFormat == BC2_RENDER_FORMAT) || (renderFormat == BC3_RENDER_FORMAT) ||
          (renderFormat == BC7_RENDER_FORMAT))
  {
    return static_cast<renderFormats>(renderFormat + 1);
  }
  else
  {
    return NONE_RENDER_FORMAT;
  }
}

renderFormats RenderFormat::ConvertFromSrgbFormat(renderFormats renderFormat)
{
  if((renderFormat == BGR8_RENDER_FORMAT) || (renderFormat == BGRA8_RENDER_FORMAT) ||
     (renderFormat == RGBA8_RENDER_FORMAT) || (renderFormat == BC1_RENDER_FORMAT) ||
     (renderFormat == BC2_RENDER_FORMAT) || (renderFormat == BC3_RENDER_FORMAT) ||
     (renderFormat == BC7_RENDER_FORMAT))
  {
    return renderFormat;
  }
  else if(IsSrgbFormat(renderFormat))
  {
    return static_cast<renderFormats>(renderFormat - 1);
  }
  else
  {
    return NONE_RENDER_FORMAT;
  }
}

const DX12_RenderFormat& RenderFormat::GetDx12RenderFormat(renderFormats renderFormat)
{
  return dx12RenderFormats[renderFormat];
}