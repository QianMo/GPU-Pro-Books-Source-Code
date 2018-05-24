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
#ifndef __CPUTMESH_H__
#define __CPUTMESH_H__

#include "CPUTRefCount.h"
#include <fstream>
#include "CPUT.h"

class CPUTRenderParameters;
class CPUTMaterial;
class CPUTModel;

//-----------------------------------------------------------------------------
enum eCPUT_VERTEX_ELEMENT_SEMANTIC : UINT
{
    CPUT_VERTEX_ELEMENT_UNDEFINED    = 0,
    // Note 1 is missing (back compatibility)
    CPUT_VERTEX_ELEMENT_POSITON      = 2,
    CPUT_VERTEX_ELEMENT_NORMAL       = 3,
    CPUT_VERTEX_ELEMENT_TEXTURECOORD = 4,
    CPUT_VERTEX_ELEMENT_VERTEXCOLOR  = 5,
    CPUT_VERTEX_ELEMENT_TANGENT      = 6,
    CPUT_VERTEX_ELEMENT_BINORMAL     = 7,
};

const char *CPUT_VERTEX_ELEMENT_SEMANTIC_AS_STRING[];

//------------------------------------------------------------------------------
// These are hard coded, so you can add or deprecate, but not reuse them
// If you change or add anything to this list, be sure to update the
// CPUT_FILE_ELEMENT_TYPE_TO_CPUT_TYPE_CONVERT
// struct below as well
enum eCPUT_VERTEX_ELEMENT_TYPE : UINT
{
    tMINTYPE = 1,
    tINT8,      // 2  int =  1 byte
    tUINT8,     // 3 UINT, __int8 =  1 byte
    tINT16,     // 4 __int16 = 2 bytes
    tUINT16,    // 5 unsigned __int16  =  2 bytes
    tINT32,     // 6 __int32 = 4 bytes
    tUINT32,    // 7 unsigned __int32  =  4 bytes
    tINT64,     // 8 __int64  = 8 bytes
    tUINT64,    // 9 unsigned __int64 =  8 bytes
    tBOOL,      // 10 bool  =  1 byte - '0' = false, '1' = true, same as stl bool i/o
    tCHAR,      // 11 signed char  = 1 byte
    tUCHAR,     // 12 unsigned char  = 1 byte
    tWCHAR,     // 13 wchar_t  = 2 bytes
    tFLOAT,     // 14 float  = 4 bytes
    tDOUBLE,    // 15 double  = 8 bytes

    // add new ones here
    tINVALID = 255
};

//------------------------------------------------------------------------------
// This map defines how the above eCPUT_VERTEX_ELEMENT_TYPE's map to internal
// CPUT types.  Be sure to update them at the same time.
const CPUT_DATA_FORMAT_TYPE CPUT_FILE_ELEMENT_TYPE_TO_CPUT_TYPE_CONVERT[] =
{
    CPUT_UNKNOWN,   // 0
    CPUT_UNKNOWN,   // 1
    CPUT_I8,        // 2  int =  1 byte
    CPUT_U8,        // 3 UINT, __int8 =  1 byte
    CPUT_I16,       // 4 __int16 = 2 bytes
    CPUT_U16,       // 5 unsigned __int16  =  2 bytes
    CPUT_I32,       // 6 __int32 = 4 bytes
    CPUT_U32,       // 7 unsigned __int32  =  4 bytes
    CPUT_I64,       // 8 __int64  = 8 bytes
    CPUT_U64,       // 9 unsigned __int64 =  8 bytes
    CPUT_BOOL,      // 10 bool  =  1 byte - '0' = false, '1' = true, same as stl bool i/o
    CPUT_CHAR,      // 11 signed char  = 1 byte
    CPUT_U8,        // 12 unsigned char  = 1 byte
    CPUT_U16,       // 13 wchar_t  = 2 bytes
    CPUT_F32,       // 14 float  = 4 bytes
    CPUT_DOUBLE,    // 15 double  = 8 bytes     
};

//-----------------------------------------------------------------------------
enum eCPUT_MESH_TOPOLOGY : UINT
{
    CPUT_TOPOLOGY_POINT_LIST = 1,
    CPUT_TOPOLOGY_INDEXED_LINE_LIST,
    CPUT_TOPOLOGY_INDEXED_LINE_STRIP,
    CPUT_TOPOLOGY_INDEXED_TRIANGLE_LIST,
    CPUT_TOPOLOGY_INDEXED_TRIANGLE_STRIP,
    CPUT_TOPOLOGY_INDEXED_TRIANGLE_FAN
    // TODO: What about non-indexed variants
};

//-----------------------------------------------------------------------------
struct CPUTVertexElementDesc
{
    eCPUT_VERTEX_ELEMENT_SEMANTIC mVertexElementSemantic;     // what is is , position, UV's, etc.
    eCPUT_VERTEX_ELEMENT_TYPE     mVertexElementType; // what is the data type, floats, ints, etc.
    UINT                          mElementSizeInBytes;   // # bytes of this element
    UINT                          mOffset;   // what is the offset within the vertex data

    void Read(std::ifstream &meshFile);
};

//-----------------------------------------------------------------------------
struct CPUTRawMeshData
{
    UINT                       mStride; // how big a "vertex" is
    UINT                       mVertexCount;
    void                      *mpVertices;
    UINT                       mIndexCount;
    UINT                      *mpIndices; // TODO: what about 16-bit indices?  Use void*?
    UINT                       mFormatDescriptorCount;
    CPUTVertexElementDesc     *mpElements;
    unsigned __int64           mTotalVerticesSizeInBytes;
    eCPUT_MESH_TOPOLOGY        mTopology;
    float3                     mBboxCenter;
    float3                     mBboxHalf;
    eCPUT_VERTEX_ELEMENT_TYPE  mIndexType;
    UINT                       mPaddingSize;

    CPUTRawMeshData():
        mStride(0),
        mVertexCount(0),
        mpVertices(NULL),
        mIndexCount(0),
        mpIndices(NULL),
        mFormatDescriptorCount(0),
        mpElements(NULL),
        mTotalVerticesSizeInBytes(0),
        mTopology(CPUT_TOPOLOGY_INDEXED_TRIANGLE_LIST),
        mBboxCenter(0.0f),
        mIndexType(tUINT32),
        mBboxHalf(0.0f),
        mPaddingSize(0)
    {
    }
    ~CPUTRawMeshData()
    {
        delete[] mpVertices;
        delete[] mpElements;
        delete[] mpIndices;
    }
    void Allocate(__int32 numElements);
    bool Read(std::ifstream &mdlfile);
};

//-----------------------------------------------------------------------------
struct CPUTBufferInfo
{
    const char           *mpSemanticName;
    int                   mSemanticIndex;
    CPUT_DATA_FORMAT_TYPE mElementType;
    int                   mElementComponentCount; // e.g., a float 3 has 3 components
    UINT                  mElementSizeInBytes;
    UINT                  mOffset;
    int                   mElementCount;
};

struct D3D11_MAPPED_SUBRESOURCE;
//-----------------------------------------------------------------------------
class CPUTMesh : public CPUTRefCount
{
protected:
    eCPUT_MESH_TOPOLOGY mMeshTopology;
	UINT mInstanceCount;

public:
    CPUTMesh() : mInstanceCount(1) {}
    virtual ~CPUTMesh(){}
    // TODO: ? Change from virtual to #ifdef-controlled redirect to platform versions?
    // TODO: Use CPUT_MAPPED_SUBRESOURCE ??
    virtual D3D11_MAPPED_SUBRESOURCE MapVertices(   CPUTRenderParameters &params, eCPUTMapType type, bool wait=true ) = 0;
    virtual D3D11_MAPPED_SUBRESOURCE MapIndices(    CPUTRenderParameters &params, eCPUTMapType type, bool wait=true ) = 0;
    virtual void                     UnmapVertices( CPUTRenderParameters &params ) = 0;
    virtual void                     UnmapIndices(  CPUTRenderParameters &params ) = 0;

    virtual void SetMeshTopology(const eCPUT_MESH_TOPOLOGY meshTopology) { mMeshTopology = meshTopology; }
    virtual void BindVertexShaderLayout(CPUTMaterial *pMaterial, CPUTMaterial *pShadowCastMaterial) = 0;
    virtual CPUTResult CreateNativeResources(
        CPUTModel      *pModel,
        UINT            meshIdx,
        int             vertexDataInfoArraySize,
        CPUTBufferInfo *pVertexInfo,
        void           *pVertexData,
        CPUTBufferInfo *pIndexInfo,
        void           *pIndex
    ) = 0;

    virtual void Draw(CPUTRenderParameters &renderParams, CPUTModel *pModel) = 0;
    virtual void DrawShadow(CPUTRenderParameters &renderParams, CPUTModel *pModel) = 0;
	void IncrementInstanceCount() { mInstanceCount++; }
	void DecrementInstanceCount() { mInstanceCount--; }
};

#endif //__CPUTMESH_H__