#include "stdafx.h"
#include "dds.h"

unsigned const DDS_MAGIC = 0x20534444; // "DDS "

#define DDS_HEADER_FLAGS_TEXTURE    0x00001007  // DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PIXELFORMAT 
#define DDS_HEADER_FLAGS_MIPMAP     0x00020000  // DDSD_MIPMAPCOUNT
#define DDS_HEADER_FLAGS_VOLUME     0x00800000  // DDSD_DEPTH
#define DDS_HEADER_FLAGS_PITCH      0x00000008  // DDSD_PITCH
#define DDS_HEADER_FLAGS_LINEARSIZE 0x00080000  // DDSD_LINEARSIZE

#define DDS_SURFACE_FLAGS_TEXTURE 0x00001000 // DDSCAPS_TEXTURE
#define DDS_SURFACE_FLAGS_MIPMAP  0x00400008 // DDSCAPS_COMPLEX | DDSCAPS_MIPMAP
#define DDS_SURFACE_FLAGS_CUBEMAP 0x00000008 // DDSCAPS_COMPLEX

#define DDS_FOURCC  0x00000004  // DDPF_FOURCC
#define DDS_RGB     0x00000040  // DDPF_RGB
#define DDS_RGBA    0x00000041  // DDPF_RGB | DDPF_ALPHAPIXELS
#define DDS_ALPHA   0x00000002  // DDPF_ALPHA
#define DDS_LUM     0x00020000  // DDPF_LUM

struct DDS_PIXELFORMAT
{
    uint32_t dwSize;
    uint32_t dwFlags;
    uint32_t dwFourCC;
    uint32_t dwRGBBitCount;
    uint32_t dwRBitMask;
    uint32_t dwGBitMask;
    uint32_t dwBBitMask;
    uint32_t dwABitMask;
};

#ifndef MAKEFOURCC
#   define MAKEFOURCC(ch0, ch1, ch2, ch3) ((uint32_t)(uint8_t)(ch0) | ((uint32_t)(uint8_t)(ch1) << 8) | ((uint32_t)(uint8_t)(ch2) << 16) | ((uint32_t)(uint8_t)(ch3) << 24 ))
#endif

DDS_PIXELFORMAT const DDSPF_DX10                = { sizeof(DDS_PIXELFORMAT), DDS_FOURCC, MAKEFOURCC('D','X','1','0'), 0, 0, 0, 0, 0 };
DDS_PIXELFORMAT const DDSPF_R16G16B16A16_FLOAT  = { sizeof(DDS_PIXELFORMAT), DDS_FOURCC, 113, 0, 0, 0, 0, 0 };
DDS_PIXELFORMAT const DDSPF_BC6H                = { sizeof(DDS_PIXELFORMAT), DDS_FOURCC, 808540228, 0, 0, 0, 0, 0 };

struct DDS_HEADER
{
    uint32_t dwMagic;
    uint32_t dwSize;
    uint32_t dwFlags;
    uint32_t dwHeight;
    uint32_t dwWidth;
    uint32_t dwPitchOrLinearSize;
    uint32_t dwDepth;
    uint32_t dwMipMapCount;
    uint32_t dwReserved1[11];
    DDS_PIXELFORMAT ddspf;
    uint32_t dwSurfaceFlags;
    uint32_t dwCubemapFlags;
    uint32_t dwReserved2[3];
};

struct DDS_HEADER_DXT10
{
    uint32_t dxgiFormat;
    uint32_t resourceDimension;
    uint32_t miscFlag;
    uint32_t arraySize;
    uint32_t reserved;
};

bool DDS::LoadA16B16G16R16F( char const* filename, SImage& img )
{
    img.m_width     = 0;
    img.m_height    = 0;
    img.m_data      = nullptr;
    img.m_dataSize  = 0;

    FILE* f = nullptr;
    fopen_s( &f, filename, "rb" );
    if ( !f )
    {
        return false;
    }

    DDS_HEADER hdr;
    fread( &hdr, sizeof( hdr ), 1, f );

    if ( hdr.dwMagic == DDS_MAGIC && memcmp( &hdr.ddspf, &DDSPF_R16G16B16A16_FLOAT, sizeof( hdr.ddspf ) ) == 0 )
    {
        img.m_dataSize  = hdr.dwWidth * hdr.dwHeight * 8;
        img.m_data      = new uint8_t[ img.m_dataSize ];
        img.m_width     = hdr.dwWidth;
        img.m_height    = hdr.dwHeight;
        fread( img.m_data, img.m_dataSize, 1, f );
        fclose( f );
        return true;
    }

    fclose( f );
    return false;
}

bool DDS::LoadBC6H( char const* filename, SImage& img )
{
    img.m_width     = 0;
    img.m_height    = 0;
    img.m_data      = nullptr;
    img.m_dataSize  = 0;

    FILE* f = nullptr;
    fopen_s( &f, filename, "rb" );
    if ( !f )
    {
        return false;
    }

    DDS_HEADER hdr;
    fread( &hdr, sizeof( hdr ), 1, f );

    if ( hdr.dwMagic == DDS_MAGIC && memcmp( &hdr.ddspf, &DDSPF_DX10, sizeof( hdr.ddspf ) ) == 0 )
    {
        DDS_HEADER_DXT10 hdrDX10;
        fread( &hdrDX10, sizeof( hdrDX10 ), 1, f );

        img.m_dataSize  = hdr.dwWidth * hdr.dwHeight;
        img.m_data      = new uint8_t[ img.m_dataSize ];
        img.m_width     = hdr.dwWidth;
        img.m_height    = hdr.dwHeight;
        fread( img.m_data, img.m_dataSize, 1, f );
        fclose( f );
        return true;
    }

    fclose( f );
    return false;
}