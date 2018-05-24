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
#include "CPUTText.h"
#include "CPUTFont.h"
#include <string.h>

// texture atlas information
float gTextAtlasWidth = 750.0f;
float gTextAtlasHeight = 12.0f;

// Constructor
//------------------------------------------------------------------------------
CPUTText::CPUTText(CPUTFont *pFont):mVertexStride(0),
    mVertexOffset(0),
    mpMirrorBuffer(NULL),
    mNumCharsInString(0),
    mZDepth(1.0f),
    mpFont(pFont)
{
    // initialize the state variables
    InitialStateSet();

    mQuadSize.height=0; mQuadSize.width=0;
    mPosition.x=0; mPosition.y=0;
    mStaticText.clear();

}

// Constructor
//-----------------------------------------------------------------------------
CPUTText::CPUTText(const cString String, CPUTControlID id, CPUTFont *pFont):mVertexStride(0),
    mVertexOffset(0),
    mpMirrorBuffer(NULL),
    mNumCharsInString(0),
    mZDepth(1.0f),
    mpFont(pFont)
{
    // initialize the state variables
    InitialStateSet();

    // save the control ID for callbacks
    mcontrolID = id;

    // set as enabled
    CPUTControl::SetEnable(true);

    // reset position/sizes
    mQuadSize.height=0; mQuadSize.width=0;
    mPosition.x=0; mPosition.y=0;

    // set the text
    SetText(String);

    // store the control id
    mcontrolID = id;

}

// Initial state of the control's member variables
//-----------------------------------------------------------------------------
void CPUTText::InitialStateSet()
{
    mcontrolType = CPUT_STATIC;
    mStaticState = CPUT_CONTROL_ACTIVE;

    mPosition.x=0; mPosition.y=0;
    mQuadSize.width=0; mQuadSize.height=0;
    mDimensions.x=0; mDimensions.y=0; mDimensions.width=0; mDimensions.height=0;

    // set default text color
    mColor.r=1.0; mColor.g=1.0; mColor.b=1.0; mColor.a=1.0; 
}

// Destructor
//------------------------------------------------------------------------------
CPUTText::~CPUTText()
{
    ReleaseInstanceData();
}



// Return the dimensions of this static text object (in pixels)
//--------------------------------------------------------------------------------
void CPUTText::GetDimensions(int &width, int &height)
{
    width = mQuadSize.width;
    height = mQuadSize.height;
}

// Return the screen position of this static text object (in pixels)
//--------------------------------------------------------------------------------
void CPUTText::GetPosition(int &x, int &y)
{
    x = mPosition.x;
    y = mPosition.y;
}


// fills user defined buffer with static string
//--------------------------------------------------------------------------------
void CPUTText::GetString(cString &ButtonText)
{

    // fill user defined buffer with the string
    ButtonText = mStaticText;
}


// Enable/disable the text
//--------------------------------------------------------------------------------
void CPUTText::SetEnable(bool in_bEnabled) 
{
    // set as enabled
    CPUTControl::SetEnable(in_bEnabled);

    // recalculate
    Recalculate();
    
    // position or size may move - force a recalculation of this control's location
    // if it is managed by the auto-arrange function
    if(this->IsAutoArranged())
    {
        mControlNeedsArrangmentResizing = true;    
    }
}




// Release all instance data
//--------------------------------------------------------------------------------
void CPUTText::ReleaseInstanceData()
{
    SAFE_DELETE_ARRAY(mpMirrorBuffer);
}

//--------------------------------------------------------------------------------
CPUTResult CPUTText::RegisterInstanceData()
{
    CPUTResult result = CPUT_SUCCESS;

    // ping the font system and tell it to load this font - or add GetFont() to the AssetLibrary(!?)

    // get the pertinent mapping data back for that font

    // calculate the uv's/etc for each character

    // hold onto a retrievable copy of the texture text atlas that can be returned with this object/or make a function that sets that map (material?)

    return result;
}


// Register all static assets (used by all CPUTText objects)
//--------------------------------------------------------------------------------
CPUTResult CPUTText::RegisterStaticResources()
{
    return CPUT_SUCCESS;
}

//
//--------------------------------------------------------------------------------
CPUTResult CPUTText::UnRegisterStaticResources()
{
    // todo: Release font?
    //CPUTFontLibraryDX11::GetFontLibrary()->DeleteFont( mFontID );
    return CPUT_SUCCESS;
}

//
//--------------------------------------------------------------------------------
void CPUTText::SetPosition(int x, int y)
{
    mPosition.x = x;
    mPosition.y = y;

    Recalculate();
}


//--------------------------------------------------------------------------------
void CPUTText::DrawIntoBuffer(CPUTGUIVertex *pVertexBufferMirror, UINT *pInsertIndex, UINT pMaxBufferSize)
{
    if(!mControlVisible)
    {
        return;
    }

    if((NULL==pVertexBufferMirror) || (NULL==pInsertIndex))
    {
        return;
    }

    if(!mpMirrorBuffer) //  || !mpMirrorBufferDisabled)
    {
        return;
    }

    // Do we have enough room to put the text vertexes into the output buffer?
    int VertexCopyCount = GetOutputVertexCount();
    ASSERT( (pMaxBufferSize >= *pInsertIndex + VertexCopyCount), _L("Too many characters to fit in allocated GUI string vertex buffer.\n\nIncrease CPUT_GUI_BUFFER_STRING_SIZE size.") );
    
    // copy the string quads into the target buffer
    memcpy(&pVertexBufferMirror[*pInsertIndex], mpMirrorBuffer, sizeof(CPUTGUIVertex)*6*mNumCharsInString);
    *pInsertIndex+=6*mNumCharsInString;

    // we'll mark the control as no longer being 'dirty'
    mControlGraphicsDirty = false;
}


// Calculate the number of verticies will be needed to display this string
// 
//--------------------------------------------------------------------------------
int CPUTText::GetOutputVertexCount()
{
    // A string is made of one quad per character (including spaces)
    //
    //   ---
    //  | 1 | 
    //   ---

    //
    // calculation: (number of characters in string) * 3 verticies/triangle * 2 triangles/quad * 1 quad/character
    return mNumCharsInString * (2*3);
}

// using the supplied font, build up a quad for each character in the string
// and assemble the quads into a vertex buffer ready for drawing/memcpy
//--------------------------------------------------------------------------------
void CPUTText::Recalculate()
{
    SAFE_DELETE_ARRAY(mpMirrorBuffer);

    mNumCharsInString = (int) mStaticText.size();
    mpMirrorBuffer = new CPUTGUIVertex[(mNumCharsInString+1)*6];

    bool Enabled = false;
    if(CPUT_CONTROL_ACTIVE == mControlState)
    {
        Enabled = true;
    }

    // walk each character and build a quad from it
    float characterPosition=0;
    for(int ii=0; ii<mNumCharsInString; ii++)
    {
        // todo: unsafe cast from wchar_t to char!  Some other way???
        char character = (char) mStaticText[ii];
        float3 UV1, UV2;
        CPUT_SIZE size = mpFont->GetGlyphSize(character);
        mpFont->GetGlyphUVS(character, Enabled, UV1, UV2);
        if('\t'!=character)
        {
            AddQuadIntoMirrorBuffer(mpMirrorBuffer, ii*6, (float)mPosition.x+characterPosition, (float)mPosition.y, (float)size.width, (float)size.height, UV1, UV2);
        }
        else
        {   
            // calculate tab width = # pixels to get to next tabstop
            // Tabs are relative from BEGINNNING of the string, not absolute based on x-position of string.  So in order for columns to line
            // up, you need the starts of the strings to line up too.
            // If you want 'absolute' x alignment behavior, use this:
            //size.width = size.width - ((mPosition.x+characterPosition) % size.width);

            // simply skip the amount of space indicated in the font's tab slot
            int CurrentPositionForNextGlyph = (int) characterPosition;
            size.width = size.width - (CurrentPositionForNextGlyph % size.width);  
            AddQuadIntoMirrorBuffer(mpMirrorBuffer, ii*6, (float)mPosition.x+characterPosition, (float)mPosition.y, (float)size.width, (float)size.height, UV1, UV2);
        }
        
        // store the max height of the string
        mQuadSize.height = max(mQuadSize.height, size.height);

        // step to next X location for next character
        characterPosition+=size.width;
    }

    // store the total width of the string 
    mQuadSize.width = (int)characterPosition;   // width of string in pixels

    // tell gui system this control image is now dirty
    // and needs to rebuild it's draw list
    mControlGraphicsDirty = true;

    // position or size may move - force a recalculation of this control's location
    // if it is managed by the auto-arrange function
    if(this->IsAutoArranged())
    {
        mControlNeedsArrangmentResizing = true;
    }
}

// Register quad for drawing string on
//--------------------------------------------------------------------------------
CPUTResult CPUTText::SetText(const cString String, float depth)
{    
    HEAPCHECK;

    mStaticText = String;
    mZDepth = depth;
    
    // call recalculate function to generate new quad
    // list to display this text
    Recalculate();
     
    HEAPCHECK;
    return CPUT_SUCCESS;
}

// This generates a quad with the supplied coordinates/uv's/etc.
//------------------------------------------------------------------------
void CPUTText::AddQuadIntoMirrorBuffer(CPUTGUIVertex *pMirrorBuffer,
    int index,
    float x, 
    float y, 
    float w, 
    float h, 
    float3 uv1, 
    float3 uv2 )
{
    pMirrorBuffer[index+0].Pos = float3( x + 0.0f, y + 0.0f, mZDepth);
    pMirrorBuffer[index+0].UV = float2(uv1.x, uv1.y);
    pMirrorBuffer[index+0].Color = mColor;

    pMirrorBuffer[index+1].Pos = float3( x + w, y + 0.0f, mZDepth);
    pMirrorBuffer[index+1].UV = float2(uv2.x, uv1.y);
    pMirrorBuffer[index+1].Color = mColor;

    pMirrorBuffer[index+2].Pos = float3( x + 0.0f, y + h, mZDepth);
    pMirrorBuffer[index+2].UV = float2(uv1.x, uv2.y);
    pMirrorBuffer[index+2].Color = mColor;

    pMirrorBuffer[index+3].Pos = float3( x + w, y + 0.0f, mZDepth);
    pMirrorBuffer[index+3].UV = float2(uv2.x, uv1.y);
    pMirrorBuffer[index+3].Color = mColor;

    pMirrorBuffer[index+4].Pos = float3( x + w, y + h, mZDepth);
    pMirrorBuffer[index+4].UV = float2(uv2.x, uv2.y);
    pMirrorBuffer[index+4].Color = mColor;

    pMirrorBuffer[index+5].Pos = float3( x + 0.0f, y +h, mZDepth);
    pMirrorBuffer[index+5].UV = float2(uv1.x, uv2.y);
    pMirrorBuffer[index+5].Color = mColor;
}

//------------------------------------------------------------------------
CPUTResult CPUTText::SetColor(float r, float g, float b, float a)
{
    CPUTColor4 color;
    color.r = r; 
    color.g = g;
    color.b = b;
    color.a = a;
    return SetColor(color);

}

//------------------------------------------------------------------------
CPUTResult CPUTText::SetColor(CPUTColor4 color)
{
    if(color != mColor)
    {
        mColor = color;

        // dirty
        Recalculate();
    }

    return CPUT_SUCCESS;
}

//------------------------------------------------------------------------
CPUTColor4 CPUTText::GetColor()
{    return mColor;
    
}