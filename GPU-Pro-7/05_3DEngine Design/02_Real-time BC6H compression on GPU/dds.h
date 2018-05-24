#pragma once

struct SImage
{
    unsigned    m_width;
    unsigned    m_height;
    uint8_t*    m_data;
    unsigned    m_dataSize;
};

namespace DDS
{
    bool LoadA16B16G16R16F( char const* filename, SImage& img );
    bool LoadBC6H( char const* filename, SImage& img );
}