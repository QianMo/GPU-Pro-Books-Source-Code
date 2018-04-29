// Copyright 2010 Intel Corporation
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

#include "Hair.h"
#include <windows.h>

#include <algorithm>
#include <limits>

Hair::Hair(ID3D11Device* d3dDevice, 
           ID3D11DeviceContext* d3dDeviceContext,
           const char *hairModel,
           const D3D10_SHADER_MACRO *shaderDefines) :
    mHairInputLayoutDesc(NULL),
    mHairVertexBuffer(NULL),
    mHairIndexBuffer(NULL),
    mHairInputLayout(NULL),
    mIndices(NULL),
    mTransformedDepths(NULL),
    mHairSegments(NULL),
    mHairSorted(false),
    mHairStride(0),
    mNumHairVertices(0),
    mNumHairSegments(0)
{
    // Hair
    LoadHairModel(hairModel);
    CreateShaders(shaderDefines);
    CreateVertexBuffer(d3dDevice, d3dDeviceContext);
    CreateIndexBuffer(d3dDevice);
    FillDefaultIndices(d3dDeviceContext);
}

Hair::~Hair()
{
    SAFE_RELEASE(mHairInputLayout);
    SAFE_RELEASE(mHairVertexBuffer);
    SAFE_RELEASE(mHairIndexBuffer);
    SAFE_RELEASE(mHairVSBlob);
    SAFE_RELEASE(mCameraHairCapturePSBlob);
    delete[] mIndices;
    delete[] mTransformedDepths;
}

void 
Hair::LoadHairModel(const char *filename)
{
    // Load the hair model
    int result = mHairFile.LoadFromFile(filename);

    // Check for errors
    switch (result) {
    case CY_HAIR_FILE_ERROR_CANT_OPEN_FILE:
        printf("Error: Cannot open hair file!\n");
        return;
    case CY_HAIR_FILE_ERROR_CANT_READ_HEADER:
        printf("Error: Cannot read hair file header!\n");
        return;
    case CY_HAIR_FILE_ERROR_WRONG_SIGNATURE:
        printf("Error: File has wrong signature!\n");
        return;
    case CY_HAIR_FILE_ERROR_READING_SEGMENTS:
        printf("Error: Cannot read hair segments!\n");
        return;
    case CY_HAIR_FILE_ERROR_READING_POINTS:
        printf("Error: Cannot read hair points!\n");
        return;
    case CY_HAIR_FILE_ERROR_READING_COLORS:
        printf("Error: Cannot read hair colors!\n");
        return;
    case CY_HAIR_FILE_ERROR_READING_THICKNESS:
        printf("Error: Cannot read hair thickness!\n");
        return;
    case CY_HAIR_FILE_ERROR_READING_TRANSPARENCY:
        printf("Error: Cannot read hair transparency!\n");
        return;
    default:
        printf("Hair file \"%s\" loaded.\n", filename);
    }

    int hairCount = mHairFile.GetHeader().hair_count;
    int pointCount = mHairFile.GetHeader().point_count;
    char buffer[1024];
    printf("Number of hair strands = %d\n"
           "Number of hair points = %d\n", pointCount, hairCount);
    OutputDebugStringA(buffer);
}

void
Hair::CreateShaders(const D3D10_SHADER_MACRO *shaderDefines)
{
    HRESULT hr;
    V(D3DX11CompileFromFile(L"CameraHairCapture.hlsl", shaderDefines, 0,
                            "HairVS", "vs_5_0",
                            0, 0, 0, &mHairVSBlob, 0, 0));
    V(D3DX11CompileFromFile(L"CameraHairCapture.hlsl", shaderDefines, 0,
                            "CameraHairCapturePS", "ps_5_0",
                            0, 0, 0, &mCameraHairCapturePSBlob, 0, 0));
}

void
Hair::CreateVertexBuffer(ID3D11Device *d3dDevice, 
                         ID3D11DeviceContext *d3dDeviceContext)
{
    // Define the input layout
    static const D3D11_INPUT_ELEMENT_DESC layout[] =
    {
        { "POSITION",   0, DXGI_FORMAT_R32G32B32_FLOAT, 0,  0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
        { "TANGENT",    0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0 },
        { "COLOR",      0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 24, D3D11_INPUT_PER_VERTEX_DATA, 0 },
    };

    UINT numElements = sizeof(layout) / sizeof(layout[0]);

    HRESULT hr;
    hr = 
        d3dDevice->CreateInputLayout(layout, numElements,
                                     mHairVSBlob->GetBufferPointer(),
                                     mHairVSBlob->GetBufferSize(),
                                     &mHairInputLayout);

    mHairStride = sizeof(Vertex);

    mNumHairVertices = mHairFile.GetHeader().point_count;
    const size_t bufferSize = sizeof(Vertex) * mNumHairVertices;

    // Create vertex buffer
    D3D11_BUFFER_DESC bd;
    ZeroMemory(&bd, sizeof(bd));
    bd.Usage = D3D11_USAGE_DYNAMIC;
    bd.ByteWidth = static_cast<UINT>(bufferSize);
    bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    bd.MiscFlags = 0;
    V(d3dDevice->CreateBuffer(&bd, NULL, &mHairVertexBuffer));

    // Fill data
    UpdateHairMesh(d3dDeviceContext, 50.0f, 50.0f, 50.0f);

    // The hair mesh file has a certain number of hair strands
    // Each hair strand has a number of line segments (formed with two
    // vertices). So, the total number of line segments formed by the mesh
    // is the number of hair strands * the number of line segments in each
    // hair strand. Some mesh files have different number of segments per
    // strand and others are uniform in which case they typically simply
    // store a "default segment length" which is used across all hair
    // strands.
    const UINT numHairStrands = mHairFile.GetHeader().hair_count;
    const UINT defaultSegmentLength = mHairFile.GetHeader().d_segments;
    mNumHairSegments = 0;
    const unsigned short *segments = mHairFile.GetSegmentsArray();
    if (segments) {
        for (UINT hairIndex = 0; hairIndex < numHairStrands; hairIndex++) {
            mNumHairSegments += segments[hairIndex];
        }
    } else {
        mNumHairSegments = numHairStrands * defaultSegmentLength;
    }
    mTransformedDepths = new TransformedDepth[mNumHairVertices];
    mIndices = new UINT[mNumHairSegments * 2];
    mHairSegments.resize(mNumHairSegments);
}

void
Hair::CreateIndexBuffer(ID3D11Device *d3dDevice)
{
    // Standard index buffer
    HRESULT hr;
    D3D11_BUFFER_DESC bd;
    ZeroMemory(&bd, sizeof(bd));
    bd.Usage = D3D11_USAGE_DYNAMIC;
    bd.ByteWidth = static_cast<UINT>(2 * mNumHairSegments * sizeof(DWORD));
    bd.BindFlags = D3D11_BIND_INDEX_BUFFER;
    bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    bd.MiscFlags = 0;
    SAFE_RELEASE(mHairIndexBuffer);
    V(d3dDevice->CreateBuffer(&bd, NULL, &mHairIndexBuffer));
}

void
Hair::FillDefaultIndices(ID3D11DeviceContext *d3dDeviceContext)
{
    const UINT numHairStrands = mHairFile.GetHeader().hair_count;
    const UINT defaultSegmentLength = mHairFile.GetHeader().d_segments;
    const unsigned short *segments = mHairFile.GetSegmentsArray();

    HRESULT hr;
    D3D11_MAPPED_SUBRESOURCE mappedResource;
    hr = d3dDeviceContext->Map(mHairIndexBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, 
                               &mappedResource);
    DWORD *indexBufferData = reinterpret_cast<DWORD*>(mappedResource.pData);
    UINT p = 0;
    UINT segmentCount = 0, indexCount = 0;
    if (segments) {
        for (UINT hairIndex = 0, i = 0; hairIndex < numHairStrands; hairIndex++) {
            for (UINT s = 0; s < segments[hairIndex]; ++s, ++p) {
                ++segmentCount;
                indexCount += 2;
                mIndices[i] = indexBufferData[i] = p + 0; ++i;
                mIndices[i] = indexBufferData[i] = p + 1; ++i;
            }
            ++p;
        }
    } else {
        for (UINT hairIndex = 0, i = 0; hairIndex < numHairStrands; hairIndex++) {
            for (UINT s = 0; s < defaultSegmentLength; ++s, ++p) {
                ++segmentCount;
                indexCount += 2;
                mIndices[i] = indexBufferData[i] = p + 0; ++i;
                mIndices[i] = indexBufferData[i] = p + 1; ++i;
            }
            ++p;
        }
    }
    assert(p == mNumHairVertices);
    d3dDeviceContext->Unmap(mHairIndexBuffer, 0);
}

void
Hair::UpdateHairMesh(ID3D11DeviceContext *d3dDeviceContext, 
                     float x, float y, float z)
{
    const float *points = mHairFile.GetPointsArray();
    const float *colors = mHairFile.GetColorsArray();

    // Create our local copy
    mHairVertices.resize(mNumHairVertices);
    for (size_t i = 0; i < mNumHairVertices; ++i) {
        mHairVertices[i][0] = *points - 0          + x - 50; ++points;
        mHairVertices[i][2] = -(*points + 30) + 25 + z - 50; ++points;
        mHairVertices[i][1] = *points - 8          + y - 50; ++points;
    }

    // Init bbox
    for (size_t p = 0; p < 3; ++p) {
        mBBoxMin[p] = std::numeric_limits<float>::max();
        mBBoxMax[p] = std::numeric_limits<float>::min();
    }

    // Build vertex buffer
    HRESULT hr;
    D3D11_MAPPED_SUBRESOURCE mappedResource;
    hr = d3dDeviceContext->Map(mHairVertexBuffer, 0, 
                               D3D11_MAP_WRITE_DISCARD, 0, 
                               &mappedResource);
    Vertex *vertexBufferData =
        reinterpret_cast<Vertex*>(mappedResource.pData);
    for (size_t i = 0; i < mNumHairVertices; ++i) {
        vertexBufferData[i].position = mHairVertices[i];

        for (size_t p = 0; p < 3; ++p) {
            mBBoxMin[p] = 
                std::min(vertexBufferData[i].position[p], mBBoxMin[p]);
            mBBoxMax[p] = 
                std::max(vertexBufferData[i].position[p], mBBoxMax[p]);
        }

        // The tangents we are getting from cyHairFile do not seem right.
        if (i > 1) {
            D3DXVECTOR3 point1(vertexBufferData[i - 1].position[0],
                               vertexBufferData[i - 1].position[1],
                               vertexBufferData[i - 1].position[2]);
            D3DXVECTOR3 point2(vertexBufferData[i].position[0],
                               vertexBufferData[i].position[1],
                               vertexBufferData[i].position[2]);
            D3DXVECTOR3 tangent = point2 - point1;

            vertexBufferData[i].tangent[0] = tangent.x;
            vertexBufferData[i].tangent[1] = tangent.y;
            vertexBufferData[i].tangent[2] = tangent.z;

            if (i == 1) {
                vertexBufferData[0].tangent[0] = tangent.x;
                vertexBufferData[0].tangent[1] = tangent.y;
                vertexBufferData[0].tangent[2] = tangent.z;
            }
        }

        vertexBufferData[i].color[0] = *colors; ++colors;
        vertexBufferData[i].color[1] = *colors; ++colors;
        vertexBufferData[i].color[2] = *colors; ++colors;
    }
    d3dDeviceContext->Unmap(mHairVertexBuffer, 0);
}

static float
TransformZ(const D3DXVECTOR3 &vertex, const D3DXMATRIXA16 &m)
{
    D3DXVECTOR4 transformedVertex;
    D3DXVec3Transform(&transformedVertex, &vertex, &m);
    return transformedVertex.z;
}

int
Hair::CompareSegments(const void *segment1, const void *segment2)
{
    HairSegment *h1 = (HairSegment *) segment1;
    HairSegment *h2 = (HairSegment *) segment2;
    if (h1->midPointDepth < h2->midPointDepth) {
        return 1;
    } else if (h1->midPointDepth == h2->midPointDepth) {
        return 0;
    } else {
        return -1;
    }
}

void 
Hair::ResetSort(ID3D11DeviceContext *d3dDeviceContext)
{
    if (mHairSorted) {
        FillDefaultIndices(d3dDeviceContext);
        mHairSorted = false;
    }
}

void
Hair::SortPerLine(ID3D11DeviceContext *d3dDeviceContext, 
                  const D3DXMATRIXA16 &cameraWorldView)
{
    mHairSorted = true;

    // Transform hair vertices in camera space and save the depth
    for (UINT i = 0; i < mHairVertices.size(); ++i) {
        mTransformedDepths[i].depth = TransformZ(mHairVertices[i], cameraWorldView);
    }

    // Build sort keys by taking the midpoint of each line segment
    for (UINT i = 0, p = 0; i < mNumHairSegments; ++i, p += 2) {
        // Use midpoint of line segment as sort key
        mHairSegments[i].midPointDepth = 
            (mTransformedDepths[mIndices[p + 0]].depth +
             mTransformedDepths[mIndices[p + 1]].depth) / 2.0f;
        mHairSegments[i].i1 = mIndices[p + 0];
        mHairSegments[i].i2 = mIndices[p + 1];
    }

    // Sort
    qsort(&mHairSegments[0], mNumHairSegments, sizeof(HairSegment), CompareSegments);

    // Regenerate index buffer
    HRESULT hr;
    D3D11_MAPPED_SUBRESOURCE mappedResource;
    hr = d3dDeviceContext->Map(mHairIndexBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, 
                               &mappedResource);
    DWORD *indexBufferData =
        reinterpret_cast<DWORD*>(mappedResource.pData);
    for (UINT i = 0, p = 0; i < mNumHairSegments; ++i, p += 2) {
        indexBufferData[p + 0] = mHairSegments[i].i1;
        indexBufferData[p + 1] = mHairSegments[i].i2;
    }
    d3dDeviceContext->Unmap(mHairIndexBuffer, 0);
}

void
Hair::Draw(ID3D11DeviceContext* d3dDeviceContext)
{
    // Setup input assembler
    const UINT vbOffset = 0;
    d3dDeviceContext->IASetInputLayout(mHairInputLayout);
    d3dDeviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_LINELIST);
    d3dDeviceContext->IASetVertexBuffers(0, 1, &mHairVertexBuffer,
                                         &mHairStride, &vbOffset);
    d3dDeviceContext->IASetIndexBuffer(mHairIndexBuffer, DXGI_FORMAT_R32_UINT, 0);
    d3dDeviceContext->DrawIndexed(2 * mNumHairSegments, 0, 0);
}
