//--------------------------------------------------------------------------------------
// Copyright 2013 Intel Corporation
// All Rights Reserved
//
// Permission is granted to use, copy, distribute and prepare derivative works of this
// software for any purpose and without fee, provided, that the above copyright notice
// and this statement appear in all copies.  Intel makes no representations about the
// suitability of this software for any purpose.  THIS SOFTWARE IS PROVIDED "AS IS."
// INTEL SPECIFICALLY DISCLAIMS ALL WARRANTIES, EXPRESS OR IMPLIED, AND ALL LIABILITY,
// INCLUDING CONSEQUENTIAL AND OTHER INDIRECT DAMAGES, FOR THE USE OF THIS SOFTWARE,
// INCLUDING LIABILITY FOR INFRINGEMENT OF ANY PROPRIETARY RIGHTS, AND INCLUDING THE
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  Intel does not
// assume any responsibility for any errors which may appear in this software nor any
// responsibility to update it.
//--------------------------------------------------------------------------------------
#ifndef __CPUTFONT_H__
#define __CPUTFONT_H__

#include "CPUT.h"
#include "CPUTRefCount.h"

#define CPUT_MAX_NUMBER_OF_CHARACTERS 256

class CPUTFont : public CPUTRefCount
{
friend class CPUTFontDX11;

public:
    static CPUTFont *CreateFont( cString FontName, cString AbsolutePathAndFilename);
    
    CPUT_SIZE GetGlyphSize(const char c);
    void GetGlyphUVS(const char c, const bool bEnabledVersion, float3& UV1, float3& UV2);

protected:
    ~CPUTFont(){}  // Destructor is not public.  Must release instead of delete.

    // atlas texture info
    float mAtlasWidth;
    float mAtlasHeight;
    float mDisabledYOffset;
    UINT mNumberOfGlyphsInAtlas;

    int mpGlyphMap[CPUT_MAX_NUMBER_OF_CHARACTERS];    

    int mpGlyphStarts[CPUT_MAX_NUMBER_OF_CHARACTERS];
    CPUT_SIZE mpGlyphSizes[CPUT_MAX_NUMBER_OF_CHARACTERS];
    float mpGlyphUVCoords[4*CPUT_MAX_NUMBER_OF_CHARACTERS]; // 4 floats/glyph = upper-left:(uv1.x, uv1.y), lower-right:(uv2.x, uv2.y)
    float mpGlyphUVCoordsDisabled[4*CPUT_MAX_NUMBER_OF_CHARACTERS]; // 4 floats/glyph = upper-left:(uv1.x, uv1.y), lower-right:(uv2.x, uv2.y)

    // helper functions
    int FindGlyphIndex(const char c);
};

#endif // #ifndef __CPUTFONT_H__