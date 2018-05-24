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
#include "CPUTFont.h"
#ifdef CPUT_FOR_DX11
#include "CPUTFontDX11.h"
#else    
    #error You must supply a target graphics API (ex: #define CPUT_FOR_DX11), or implement the target API for this file.
#endif



int gFontMap_active[] =
{
    'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',            // lower
    'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',            // upper
    '1','2','3','4','5','6','7','8','9','0',                                                                            // numbers
    ',', '.','/',';','\'','[',']','\\','`','~','!','@','#','$','%','^','&','*','(',')','_','+','{','}','|',':','\"','<','>','?','-','=', // symbols
    ' ','\t',                                                                                                                // space, tab
    -1
};

//-----------------------------------------------------------------------------
CPUTFont *CPUTFont::CreateFont( cString FontName, cString AbsolutePathAndFilename )
{
    // TODO: accept DX11/OGL param to control which platform we generate.
    // TODO: be sure to support the case where we want to support only one of them
#ifdef CPUT_FOR_DX11
    return CPUTFontDX11::CreateFont( FontName, AbsolutePathAndFilename );
#else    
    #error You must supply a target graphics API (ex: #define CPUT_FOR_DX11), or implement the target API for this file.
#endif

}

// return the size in pixels of the glyph
//-----------------------------------------------------------------------------
CPUT_SIZE CPUTFont::GetGlyphSize(const char c)
{    
    CPUT_SIZE size;
    size.height=0;
    size.width=0;

    int index = FindGlyphIndex(c);
    if(-1!=index)
    {
        size.width=mpGlyphSizes[index].width;
        size.height=mpGlyphSizes[index].height;
    }

    return size;
}

// return the uv coordinates for the given glyph
// upper left/lower right
//-----------------------------------------------------------------------------
void CPUTFont::GetGlyphUVS(const char c, const bool bEnabledVersion, float3 &UV1, float3 &UV2)
{
    int index = FindGlyphIndex(c);
    if(-1!=index)
    {
        if(bEnabledVersion)
        {
            UV1.x=mpGlyphUVCoords[4*index+0];
            UV1.y=mpGlyphUVCoords[4*index+1];
            UV2.x=mpGlyphUVCoords[4*index+2];
            UV2.y=mpGlyphUVCoords[4*index+3];        
        }
        else
        {
            UV1.x=mpGlyphUVCoordsDisabled[4*index+0];
            UV1.y=mpGlyphUVCoordsDisabled[4*index+1];
            UV2.x=mpGlyphUVCoordsDisabled[4*index+2];
            UV2.y=mpGlyphUVCoordsDisabled[4*index+3];   
        }
    }    
}

// find the index of the glyph that corresponds to the char passed in
//-----------------------------------------------------------------------------
int CPUTFont::FindGlyphIndex(const char c)
{
    int index=0;
    while(-1 != gFontMap_active[index])
    {
        if(c == gFontMap_active[index])
        {
            return index;
        }
        index++;
    }

    // not found
    return -1;
}

