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

#pragma once

#include "cyHairFile.h"
#include "DXUT.h"

#include <vector>

class Hair
{
  public:
    Hair(ID3D11Device *d3dDevice, ID3D11DeviceContext *d3dDeviceContext,
         const char *hairModel, const D3D10_SHADER_MACRO *shaderDefines);
    ~Hair();

    void GetCameraDefaults(D3DXVECTOR3& Pos, D3DXVECTOR3& Target) const;

    void GetLightDefaults(D3DXVECTOR3& Pos, D3DXVECTOR3& Target) const;

    void Draw(ID3D11DeviceContext *d3dDeviceContext);

    void GetBBox(D3DXVECTOR3 *min, D3DXVECTOR3 *max) {
        *min = mBBoxMin;
        *max = mBBoxMax;
    }

    void UpdateHairMesh(ID3D11DeviceContext *d3dDeviceContext, 
                        float x, float y, float z);
    void ResetSort(ID3D11DeviceContext *d3dDeviceContext);
    void SortPerLine(ID3D11DeviceContext *d3dDeviceContext, 
                     const D3DXMATRIXA16 &cameraWorldView);

  private:
    struct Vertex {
        D3DXVECTOR3 position;
        D3DXVECTOR3 tangent;
        D3DXVECTOR3 color;
    };

    struct TransformedDepth {
        float depth;
    };

    struct HairSegment {
        float midPointDepth;
        UINT i1;
        UINT i2;
    };

    void LoadHairModel(const char *filename);
    void CreateShaders(const D3D10_SHADER_MACRO *shaderDefines);
    void CreateVertexBuffer(ID3D11Device *d3dDevice, 
                            ID3D11DeviceContext *d3dDeviceContext);
    void CreateIndexBuffer(ID3D11Device *d3dDevice);
    void FillDefaultIndices(ID3D11DeviceContext *d3dDeviceContext);

    static int CompareSegments(const void *segment1, const void *segment2);

    cyHairFile                      mHairFile;
    const D3D11_INPUT_ELEMENT_DESC *mHairInputLayoutDesc;
    ID3D11InputLayout              *mHairInputLayout;
    UINT                            mHairStride;
    ID3D11Buffer                   *mHairVertexBuffer;
    ID3D11Buffer                   *mHairIndexBuffer;
    UINT                            mNumHairVertices;
    UINT                            mNumHairSegments;

    ID3D10Blob                     *mHairVSBlob;
    ID3D10Blob                     *mCameraHairCapturePSBlob;

    D3DXVECTOR3                     mBBoxMin;
    D3DXVECTOR3                     mBBoxMax;

    struct TransformedDepth        *mTransformedDepths;
    std::vector<HairSegment>        mHairSegments;
    UINT                           *mIndices;

    std::vector<D3DXVECTOR3>        mHairVertices;

    bool                            mHairSorted;
};
