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
#include "CPUTMesh.h"

//-----------------------------------------------------------------------------
// Note: The indices of these strings must match the corresponding value in enum CPUT_VERTEX_ELEMENT_SEMANTIC
const char *CPUT_VERTEX_ELEMENT_SEMANTIC_AS_STRING[] =
{
    "UNDEFINED",
    "UNDEFINED", // Note 1 is missing (back compatibility)
    "POSITON",
    "NORMAL",
    "TEXTURECOORD",
    "COLOR",
    "TANGENT",
    "BINORMAL"
};
//-----------------------------------------------------------------------------
void CPUTVertexElementDesc::Read(std::ifstream &meshFile)
{
    meshFile.read((char*)this, sizeof(*this));
}

//-----------------------------------------------------------------------------
void CPUTRawMeshData::Allocate(__int32 numElements)
{
    mVertexCount = numElements;
    mStride += mPaddingSize; // TODO: move this to stride computation
    mTotalVerticesSizeInBytes = mVertexCount * mStride;
    mpVertices = (void*)new char[(UINT)mTotalVerticesSizeInBytes];
    ::memset( mpVertices, 0, (size_t)mTotalVerticesSizeInBytes );
}

//-----------------------------------------------------------------------------
bool CPUTRawMeshData::Read(std::ifstream &modelFile)
{
    unsigned __int32 magicCookie;
    modelFile.read((char*)&magicCookie,sizeof(magicCookie));
    if( !modelFile.good() ) return false; // TODO: Yuck!  Why do we need to get here to figure out we're done?

    ASSERT( magicCookie == 1234, _L("Invalid model file.") );

    modelFile.read((char*)&mStride,                   sizeof(mStride));
    modelFile.read((char*)&mPaddingSize,              sizeof(mPaddingSize)); // DWM TODO: What is this?
    modelFile.read((char*)&mTotalVerticesSizeInBytes, sizeof(mTotalVerticesSizeInBytes));
    modelFile.read((char*)&mVertexCount,              sizeof(mVertexCount));
    modelFile.read((char*)&mTopology,                 sizeof(mTopology));
    modelFile.read((char*)&mBboxCenter,               sizeof(mBboxCenter));
    modelFile.read((char*)&mBboxHalf,                 sizeof(mBboxHalf));

    // read  format descriptors
    modelFile.read((char*)&mFormatDescriptorCount, sizeof(mFormatDescriptorCount));
    ASSERT( modelFile.good(), _L("Model file bad" ) );

    mpElements = new CPUTVertexElementDesc[mFormatDescriptorCount];
    for( UINT ii=0; ii<mFormatDescriptorCount; ++ii )
    {
        mpElements[ii].Read(modelFile);
    }
    modelFile.read((char*)&mIndexCount, sizeof(mIndexCount));
    modelFile.read((char*)&mIndexType, sizeof(mIndexType));
    ASSERT( modelFile.good(), _L("Bad model file(1)." ) );

    mpIndices = new UINT[mIndexCount];
    if( mIndexCount != 0 )
    {
        modelFile.read((char*)mpIndices, mIndexCount * sizeof(UINT));
    }
    modelFile.read((char*)&magicCookie, sizeof(magicCookie));
    ASSERT( magicCookie == 1234, _L("Model file missing magic cookie.") );
    ASSERT( modelFile.good(),    _L("Bad model file(2).") );

    if ( 0 != mTotalVerticesSizeInBytes )
    {
        Allocate(mVertexCount);  // recalculates some things
        modelFile.read((char*)(mpVertices), mTotalVerticesSizeInBytes);
    }
    modelFile.read((char*)&magicCookie, sizeof(magicCookie));
    ASSERT( modelFile.good() && magicCookie == 1234, _L("Bad model file(3).") );

    return modelFile.good();
}
