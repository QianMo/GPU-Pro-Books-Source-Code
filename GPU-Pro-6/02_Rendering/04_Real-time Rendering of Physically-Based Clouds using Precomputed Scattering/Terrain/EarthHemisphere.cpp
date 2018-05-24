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

#include "stdafx.h"
#include "EarthHemisphere.h"
#include "Structures.fxh"
#include "ElevationDataSource.h"
#include "ShaderMacroHelper.h"

struct SHemisphereVertex
{
    D3DXVECTOR3 f3WorldPos;
    D3DXVECTOR2 f2MaskUV0;
    SHemisphereVertex() : f3WorldPos(0,0,0), f2MaskUV0(0,0){}
};

enum QUAD_TRIANGULATION_TYPE
{
    QUAD_TRIANG_TYPE_UNDEFINED = 0,

    // 01      11
    //  *------*
    //  |   .' |
    //  | .'   |
    //  * -----*
    // 00      10
    QUAD_TRIANG_TYPE_00_TO_11,

    // 01      11
    //  *------*
    //  | '.   |
    //  |   '. |
    //  * -----*
    // 00      10
    QUAD_TRIANG_TYPE_01_TO_10
};

template<typename IndexType, class CIndexGenerator>
class CTriStrip
{
public:
    CTriStrip(std::vector<IndexType> &Indices, CIndexGenerator IndexGenerator):
        m_Indices(Indices),
        m_IndexGenerator(IndexGenerator),
        m_QuadTriangType(QUAD_TRIANG_TYPE_UNDEFINED)
    {
    }
    
    void AddStrip(int iBaseIndex,
                  int iStartCol,
                  int iStartRow,
                  int iNumCols,
                  int iNumRows,
                  QUAD_TRIANGULATION_TYPE QuadTriangType)
    {
        assert( QuadTriangType == QUAD_TRIANG_TYPE_00_TO_11 || QuadTriangType == QUAD_TRIANG_TYPE_01_TO_10 );
        int iFirstVertex = iBaseIndex + m_IndexGenerator(iStartCol, iStartRow + (QuadTriangType==QUAD_TRIANG_TYPE_00_TO_11 ? 1:0));
        if(m_QuadTriangType != QUAD_TRIANG_TYPE_UNDEFINED)
        {
            // To move from one strip to another, we have to generate two degenerate triangles
            // by duplicating the last vertex in previous strip and the first vertex in new strip
            m_Indices.push_back( m_Indices.back() );
            m_Indices.push_back( iFirstVertex );
        }

        if(m_QuadTriangType != QUAD_TRIANG_TYPE_UNDEFINED && m_QuadTriangType != QuadTriangType || 
           m_QuadTriangType == QUAD_TRIANG_TYPE_UNDEFINED && QuadTriangType == QUAD_TRIANG_TYPE_01_TO_10)
        {
            // If triangulation orientation changes, or if start strip orientation is 01 to 10, 
            // we also have to add one additional vertex to preserve winding order
            m_Indices.push_back( iFirstVertex );
        }
        m_QuadTriangType = QuadTriangType;

        for(int iRow = 0; iRow < iNumRows-1; ++iRow)
        {
            for(int iCol = 0; iCol < iNumCols; ++iCol)
            {
                int iV00 = iBaseIndex + m_IndexGenerator(iStartCol+iCol, iStartRow+iRow);
                int iV01 = iBaseIndex + m_IndexGenerator(iStartCol+iCol, iStartRow+iRow+1);
                if( m_QuadTriangType == QUAD_TRIANG_TYPE_01_TO_10 )
                {
                    if( iCol == 0 && iRow == 0)
                        assert(iFirstVertex == iV00);
                    // 01      11
                    //  *------*
                    //  | '.   |
                    //  |   '. |
                    //  * -----*
                    // 00      10
                    m_Indices.push_back(iV00);
                    m_Indices.push_back(iV01);
                }
                else if( m_QuadTriangType == QUAD_TRIANG_TYPE_00_TO_11 )
                {
                    if( iCol == 0 && iRow == 0)
                        assert(iFirstVertex == iV01);
                    // 01      11
                    //  *------*
                    //  |   .' |
                    //  | .'   |
                    //  * -----*
                    // 00      10
                    m_Indices.push_back(iV01);
                    m_Indices.push_back(iV00);
                }
                else
                {
                    assert(false);
                }
            }
        
            if(iRow < iNumRows-2)
            {
                m_Indices.push_back( m_Indices.back() );
                m_Indices.push_back( iBaseIndex + m_IndexGenerator(iStartCol, iStartRow+iRow+1 + (QuadTriangType==QUAD_TRIANG_TYPE_00_TO_11 ? 1:0)) );
            }
        }
    }

private:
    QUAD_TRIANGULATION_TYPE m_QuadTriangType;
    std::vector<IndexType> &m_Indices;
    CIndexGenerator m_IndexGenerator;
};

class CStdIndexGenerator
{
public:
    CStdIndexGenerator(int iPitch):m_iPitch(iPitch){}
    UINT operator ()(int iCol, int iRow){return iCol + iRow*m_iPitch;}

private:
    int m_iPitch;
};

typedef CTriStrip<UINT,  CStdIndexGenerator> StdTriStrip32;

void ComputeVertexHeight(SHemisphereVertex &Vertex, 
                         class CElevationDataSource *pDataSource,
                         float fSamplingStep,
                         float fSampleScale)
{
    D3DXVECTOR3 &f3PosWS = Vertex.f3WorldPos;

    float fCol = f3PosWS.x / fSamplingStep;
    float fRow = f3PosWS.z / fSamplingStep;
    float fDispl = pDataSource->GetInterpolatedHeight(fCol, fRow);
    int iColOffset, iRowOffset;
    pDataSource->GetOffsets(iColOffset, iRowOffset);
    Vertex.f2MaskUV0.x = (fCol + (float)iColOffset + 0.5f)/(float)pDataSource->GetNumCols();
    Vertex.f2MaskUV0.y = (fRow + (float)iRowOffset + 0.5f)/(float)pDataSource->GetNumRows();
    
    D3DXVECTOR3 f3SphereNormal;
    D3DXVec3Normalize(&f3SphereNormal, &f3PosWS);
    f3PosWS += f3SphereNormal * fDispl * fSampleScale;
}

class CRingMeshBuilder
{
public:
    CRingMeshBuilder(ID3D11Device *pDevice,
                     const std::vector<SHemisphereVertex> &VB,
                     int iGridDimenion,
                     std::vector<SRingSectorMesh> &RingMeshes) : m_pDevice(pDevice), m_VB(VB), m_iGridDimenion(iGridDimenion), m_RingMeshes(RingMeshes){}

    void CreateMesh(int iBaseIndex, 
                    int iStartCol, 
                    int iStartRow, 
                    int iNumCols, 
                    int iNumRows, 
                    enum QUAD_TRIANGULATION_TYPE QuadTriangType)
    {
        m_RingMeshes.push_back( SRingSectorMesh() );
        auto& CurrMesh = m_RingMeshes.back();

        std::vector<UINT> IB;
        StdTriStrip32 TriStrip( IB, CStdIndexGenerator(m_iGridDimenion) );
        TriStrip.AddStrip(iBaseIndex, iStartCol, iStartRow, iNumCols, iNumRows, QuadTriangType);

        CurrMesh.uiNumIndices = (UINT)IB.size();

        // Prepare buffer description
        D3D11_BUFFER_DESC IndexBufferDesc=
        {
            (UINT)(IB.size() * sizeof(IB[0])),
            D3D11_USAGE_IMMUTABLE,
            D3D11_BIND_INDEX_BUFFER,
            0, //UINT CPUAccessFlags
            0, //UINT MiscFlags;
            0, //UINT StructureByteStride;
        };
        D3D11_SUBRESOURCE_DATA IBInitData = {&IB[0], 0, 0};
        // Create the buffer
        HRESULT hr;
        V( m_pDevice->CreateBuffer( &IndexBufferDesc, &IBInitData, &CurrMesh.pIndBuff) );

        // Compute bounding box
        auto &BB = CurrMesh.BoundBox;
        BB.fMaxX =BB.fMaxY = BB.fMaxZ = -FLT_MAX;
        BB.fMinX =BB.fMinY = BB.fMinZ = +FLT_MAX;
        for(auto Ind = IB.begin(); Ind != IB.end(); ++Ind)
        {
            const auto &CurrVert = m_VB[*Ind].f3WorldPos;
            BB.fMinX = min(BB.fMinX, CurrVert.x);
            BB.fMinY = min(BB.fMinY, CurrVert.y);
            BB.fMinZ = min(BB.fMinZ, CurrVert.z);

            BB.fMaxX = max(BB.fMaxX, CurrVert.x);
            BB.fMaxY = max(BB.fMaxY, CurrVert.y);
            BB.fMaxZ = max(BB.fMaxZ, CurrVert.z);
        }
    }

private:
    CComPtr<ID3D11Device> m_pDevice;
    std::vector<SRingSectorMesh> &m_RingMeshes;
    const std::vector<SHemisphereVertex> &m_VB;
    const int m_iGridDimenion;
};

void GenerateSphereGeometry(ID3D11Device *pDevice,
                            const float fEarthRadius,
                            int iGridDimension, 
                            const int iNumRings,
                            class CElevationDataSource *pDataSource,
                            float fSamplingStep,
                            float fSampleScale,
                            std::vector<SHemisphereVertex> &VB,
                            std::vector<UINT> &StitchIB,
                            std::vector<SRingSectorMesh> &SphereMeshes)
{
    if( (iGridDimension - 1) % 4 != 0 )
    {
        assert(false);
        iGridDimension = SRenderingParams().m_iRingDimension;
    }
    const int iGridMidst = (iGridDimension-1)/2;
    const int iGridQuart = (iGridDimension-1)/4;

    const int iLargestGridScale = iGridDimension << (iNumRings-1);
    
    CRingMeshBuilder RingMeshBuilder(pDevice, VB, iGridDimension, SphereMeshes);

    int iStartRing = 0;
    VB.reserve( (iNumRings-iStartRing) * iGridDimension * iGridDimension );
    for(int iRing = iStartRing; iRing < iNumRings; ++iRing)
    {
        int iCurrGridStart = (int)VB.size();
        VB.resize(VB.size() + iGridDimension * iGridDimension);
        float fGridScale = 1.f / (float)(1<<(iNumRings-1 - iRing));
        // Fill vertex buffer
        for(int iRow = 0; iRow < iGridDimension; ++iRow)
            for(int iCol = 0; iCol < iGridDimension; ++iCol)
            {
                auto &CurrVert = VB[iCurrGridStart + iCol + iRow*iGridDimension];
                auto &f3Pos = CurrVert.f3WorldPos;
                f3Pos.x = static_cast<float>(iCol) / static_cast<float>(iGridDimension-1);
                f3Pos.z = static_cast<float>(iRow) / static_cast<float>(iGridDimension-1);
                f3Pos.x = f3Pos.x*2 - 1;
                f3Pos.z = f3Pos.z*2 - 1;
                f3Pos.y = 0;
                float fDirectionScale = 1;
                if( f3Pos.x != 0 || f3Pos.z != 0 )
                {
                    float fDX = abs(f3Pos.x);
                    float fDZ = abs(f3Pos.z);
                    float fMaxD = max(fDX, fDZ);
                    float fMinD = min(fDX, fDZ);
                    float fTan = fMinD/fMaxD;
                    fDirectionScale = 1 / sqrt(1 + fTan*fTan);
                }
            
                f3Pos.x *= fDirectionScale*fGridScale;
                f3Pos.z *= fDirectionScale*fGridScale;
                f3Pos.y = sqrt( max(0, 1 - (f3Pos.x*f3Pos.x + f3Pos.z*f3Pos.z)) );

                f3Pos.x *= fEarthRadius;
                f3Pos.z *= fEarthRadius;
                f3Pos.y *= fEarthRadius;

                ComputeVertexHeight(CurrVert, pDataSource, fSamplingStep, fSampleScale);
                f3Pos.y -= fEarthRadius;
            }

        // Align vertices on the outer boundary
        if( iRing < iNumRings-1 )
        {
            for(int i=1; i < iGridDimension-1; i+=2)
            {
                // Top & bottom boundaries
                for(int iRow=0; iRow < iGridDimension; iRow += iGridDimension-1)
                {
                    const auto &V0 = VB[iCurrGridStart + i - 1 + iRow*iGridDimension].f3WorldPos;
                          auto &V1 = VB[iCurrGridStart + i + 0 + iRow*iGridDimension].f3WorldPos;
                    const auto &V2 = VB[iCurrGridStart + i + 1 + iRow*iGridDimension].f3WorldPos;
                    V1 = (V0+V2)/2.f;
                }

                // Left & right boundaries
                for(int iCol=0; iCol < iGridDimension; iCol += iGridDimension-1)
                {
                    const auto &V0 = VB[iCurrGridStart + iCol + (i - 1)*iGridDimension].f3WorldPos;
                          auto &V1 = VB[iCurrGridStart + iCol + (i + 0)*iGridDimension].f3WorldPos;
                    const auto &V2 = VB[iCurrGridStart + iCol + (i + 1)*iGridDimension].f3WorldPos;
                    V1 = (V0+V2)/2.f;
                }
            }


            // Add triangles stitching this ring with the next one
            int iNextGridStart = (int)VB.size();
            assert( iNextGridStart == iCurrGridStart + iGridDimension*iGridDimension);

            // Bottom boundary
            for(int iCol=0; iCol < iGridDimension-1; iCol += 2)
            {
                StitchIB.push_back(iNextGridStart + (iGridQuart + iCol/2) + iGridQuart * iGridDimension); 
                StitchIB.push_back(iCurrGridStart + (iCol+1) + 0 * iGridDimension); 
                StitchIB.push_back(iCurrGridStart + (iCol+0) + 0 * iGridDimension); 

                StitchIB.push_back(iNextGridStart + (iGridQuart + iCol/2) + iGridQuart * iGridDimension); 
                StitchIB.push_back(iCurrGridStart + (iCol+2) + 0 * iGridDimension); 
                StitchIB.push_back(iCurrGridStart + (iCol+1) + 0 * iGridDimension); 

                StitchIB.push_back(iNextGridStart + (iGridQuart + iCol/2)   + iGridQuart * iGridDimension); 
                StitchIB.push_back(iNextGridStart + (iGridQuart + iCol/2+1) + iGridQuart * iGridDimension); 
                StitchIB.push_back(iCurrGridStart + (iCol+2) + 0 * iGridDimension); 
            }

            // Top boundary
            for(int iCol=0; iCol < iGridDimension-1; iCol += 2)
            {
                StitchIB.push_back(iCurrGridStart + (iCol+0) + (iGridDimension-1) * iGridDimension); 
                StitchIB.push_back(iCurrGridStart + (iCol+1) + (iGridDimension-1) * iGridDimension); 
                StitchIB.push_back(iNextGridStart + (iGridQuart + iCol/2) + iGridQuart* 3 * iGridDimension); 

                StitchIB.push_back(iCurrGridStart + (iCol+1) + (iGridDimension-1) * iGridDimension); 
                StitchIB.push_back(iCurrGridStart + (iCol+2) + (iGridDimension-1) * iGridDimension); 
                StitchIB.push_back(iNextGridStart + (iGridQuart + iCol/2) + iGridQuart* 3 * iGridDimension); 

                StitchIB.push_back(iCurrGridStart + (iCol+2) + (iGridDimension-1) * iGridDimension); 
                StitchIB.push_back(iNextGridStart + (iGridQuart + iCol/2 + 1) + iGridQuart* 3 * iGridDimension); 
                StitchIB.push_back(iNextGridStart + (iGridQuart + iCol/2)     + iGridQuart* 3 * iGridDimension); 
            }

            // Left boundary
            for(int iRow=0; iRow < iGridDimension-1; iRow += 2)
            {
                StitchIB.push_back(iNextGridStart + iGridQuart + (iGridQuart+ iRow/2) * iGridDimension); 
                StitchIB.push_back(iCurrGridStart + 0 + (iRow+0) * iGridDimension); 
                StitchIB.push_back(iCurrGridStart + 0 + (iRow+1) * iGridDimension); 

                StitchIB.push_back(iNextGridStart + iGridQuart + (iGridQuart+ iRow/2) * iGridDimension);  
                StitchIB.push_back(iCurrGridStart + 0 + (iRow+1) * iGridDimension); 
                StitchIB.push_back(iCurrGridStart + 0 + (iRow+2) * iGridDimension); 

                StitchIB.push_back(iNextGridStart + iGridQuart + (iGridQuart + iRow/2 + 1) * iGridDimension); 
                StitchIB.push_back(iNextGridStart + iGridQuart + (iGridQuart + iRow/2)     * iGridDimension); 
                StitchIB.push_back(iCurrGridStart + 0 + (iRow+2) * iGridDimension); 
            }

            // Right boundary
            for(int iRow=0; iRow < iGridDimension-1; iRow += 2)
            {
                StitchIB.push_back(iCurrGridStart + (iGridDimension-1) + (iRow+1) * iGridDimension); 
                StitchIB.push_back(iCurrGridStart + (iGridDimension-1) + (iRow+0) * iGridDimension); 
                StitchIB.push_back(iNextGridStart + iGridQuart*3 + (iGridQuart+ iRow/2) * iGridDimension); 

                StitchIB.push_back(iCurrGridStart + (iGridDimension-1) + (iRow+2) * iGridDimension); 
                StitchIB.push_back(iCurrGridStart + (iGridDimension-1) + (iRow+1) * iGridDimension); 
                StitchIB.push_back(iNextGridStart + iGridQuart*3 + (iGridQuart+ iRow/2) * iGridDimension); 

                StitchIB.push_back(iCurrGridStart + (iGridDimension-1) + (iRow+2) * iGridDimension); 
                StitchIB.push_back(iNextGridStart + iGridQuart*3 + (iGridQuart+ iRow/2)     * iGridDimension); 
                StitchIB.push_back(iNextGridStart + iGridQuart*3 + (iGridQuart+ iRow/2 + 1) * iGridDimension); 
            }
        }


        // Generate indices for the current ring
        if( iRing == 0 )
        {
            RingMeshBuilder.CreateMesh( iCurrGridStart, 0,                   0, iGridMidst+1, iGridMidst+1, QUAD_TRIANG_TYPE_00_TO_11);
            RingMeshBuilder.CreateMesh( iCurrGridStart, iGridMidst,          0, iGridMidst+1, iGridMidst+1, QUAD_TRIANG_TYPE_01_TO_10);
            RingMeshBuilder.CreateMesh( iCurrGridStart, 0,          iGridMidst, iGridMidst+1, iGridMidst+1, QUAD_TRIANG_TYPE_01_TO_10);
            RingMeshBuilder.CreateMesh( iCurrGridStart, iGridMidst, iGridMidst, iGridMidst+1, iGridMidst+1, QUAD_TRIANG_TYPE_00_TO_11);
        }
        else
        {
            RingMeshBuilder.CreateMesh( iCurrGridStart,            0,            0,   iGridQuart+1, iGridQuart+1, QUAD_TRIANG_TYPE_00_TO_11);
            RingMeshBuilder.CreateMesh( iCurrGridStart,   iGridQuart,            0,   iGridQuart+1, iGridQuart+1, QUAD_TRIANG_TYPE_00_TO_11);

            RingMeshBuilder.CreateMesh( iCurrGridStart,   iGridMidst,            0,   iGridQuart+1, iGridQuart+1, QUAD_TRIANG_TYPE_01_TO_10);
            RingMeshBuilder.CreateMesh( iCurrGridStart, iGridQuart*3,            0,   iGridQuart+1, iGridQuart+1, QUAD_TRIANG_TYPE_01_TO_10);
                                        
            RingMeshBuilder.CreateMesh( iCurrGridStart,            0,   iGridQuart,   iGridQuart+1, iGridQuart+1, QUAD_TRIANG_TYPE_00_TO_11);
            RingMeshBuilder.CreateMesh( iCurrGridStart,            0,   iGridMidst,   iGridQuart+1, iGridQuart+1, QUAD_TRIANG_TYPE_01_TO_10);
                                        
            RingMeshBuilder.CreateMesh( iCurrGridStart, iGridQuart*3,   iGridQuart,   iGridQuart+1, iGridQuart+1, QUAD_TRIANG_TYPE_01_TO_10);
            RingMeshBuilder.CreateMesh( iCurrGridStart, iGridQuart*3,   iGridMidst,   iGridQuart+1, iGridQuart+1, QUAD_TRIANG_TYPE_00_TO_11);

            RingMeshBuilder.CreateMesh( iCurrGridStart,            0, iGridQuart*3,   iGridQuart+1, iGridQuart+1, QUAD_TRIANG_TYPE_01_TO_10);
            RingMeshBuilder.CreateMesh( iCurrGridStart,   iGridQuart, iGridQuart*3,   iGridQuart+1, iGridQuart+1, QUAD_TRIANG_TYPE_01_TO_10);

            RingMeshBuilder.CreateMesh( iCurrGridStart,   iGridMidst, iGridQuart*3,   iGridQuart+1, iGridQuart+1, QUAD_TRIANG_TYPE_00_TO_11);
            RingMeshBuilder.CreateMesh( iCurrGridStart, iGridQuart*3, iGridQuart*3,   iGridQuart+1, iGridQuart+1, QUAD_TRIANG_TYPE_00_TO_11);
        }
    }
    
    // We do not need per-vertex normals as we use normal map to shade terrain
    // Sphere tangent vertex are computed in the shader
#if 0
    // Compute normals
    const D3DXVECTOR3 *pV0 = nullptr;
    const D3DXVECTOR3 *pV1 = &VB[ IB[0] ].f3WorldPos;
    const D3DXVECTOR3 *pV2 = &VB[ IB[1] ].f3WorldPos;
    float fSign = +1;
    for(UINT Ind=2; Ind < m_uiIndicesInIndBuff; ++Ind)
    {
        fSign = -fSign;
        pV0 = pV1;
        pV1 = pV2;
        pV2 =  &VB[ IB[Ind] ].f3WorldPos;
        D3DXVECTOR3 Rib0 = *pV0 - *pV1;
        D3DXVECTOR3 Rib1 = *pV1 - *pV2;
        D3DXVECTOR3 TriN;
        D3DXVec3Cross(&TriN, &Rib0, &Rib1);
        float fLength = D3DXVec3Length(&TriN);
        if( fLength > 0.1 )
        {
            TriN /= fLength*fSign;
            for(int i=-2; i <= 0; ++i)
                VB[ IB[Ind+i] ].f3Normal += TriN;
        }
    }
    for(auto VBIt=VB.begin(); VBIt != VB.end(); ++VBIt)
    {
        float fLength = D3DXVec3Length(&VBIt->f3Normal);
        if( fLength > 1 )
            VBIt->f3Normal /= fLength;
    }

    // Adjust normals on boundaries
    for(int iRing = iStartRing; iRing < iNumRings-1; ++iRing)
    {
        int iCurrGridStart = (iRing-iStartRing) * iGridDimension*iGridDimension;
        int iNextGridStart = (iRing-iStartRing+1) * iGridDimension*iGridDimension;
        for(int i=0; i < iGridDimension; i+=2)
        {
            for(int Bnd=0; Bnd < 2; ++Bnd)
            {
                const int CurrGridOffsets[] = {0, iGridDimension-1};
                const int NextGridPffsets[] = {iGridQuart, iGridQuart*3};
                // Left and right boundaries
                {
                    auto &CurrGridN = VB[iCurrGridStart + CurrGridOffsets[Bnd] + i*iGridDimension].f3Normal;
                    auto &NextGridN = VB[iNextGridStart + NextGridPffsets[Bnd] + (iGridQuart+i/2)*iGridDimension].f3Normal;
                    auto NewN = CurrGridN + NextGridN;
                    D3DXVec3Normalize(&NewN, &NewN);
                    CurrGridN = NextGridN = NewN;
                    if( i > 1 )
                    {
                        auto &PrevCurrGridN = VB[iCurrGridStart + CurrGridOffsets[Bnd] + (i-2)*iGridDimension].f3Normal;
                        auto MiddleN = PrevCurrGridN + NewN;
                        D3DXVec3Normalize( &VB[iCurrGridStart + CurrGridOffsets[Bnd] + (i-1)*iGridDimension].f3Normal, &MiddleN);
                    }
                }

                // Bottom and top boundaries
                {
                    auto &CurrGridN = VB[iCurrGridStart +                i + CurrGridOffsets[Bnd]*iGridDimension].f3Normal;
                    auto &NextGridN = VB[iNextGridStart + (iGridQuart+i/2) + NextGridPffsets[Bnd]*iGridDimension].f3Normal;
                    auto NewN = CurrGridN + NextGridN;
                    D3DXVec3Normalize(&NewN, &NewN);
                    CurrGridN = NextGridN = NewN;
                    if( i > 1 )
                    {
                        auto &PrevCurrGridN = VB[iCurrGridStart + (i-2) + CurrGridOffsets[Bnd]*iGridDimension].f3Normal;
                        auto MiddleN = PrevCurrGridN + NewN;
                        D3DXVec3Normalize( &VB[iCurrGridStart + (i-1) + CurrGridOffsets[Bnd]*iGridDimension].f3Normal, &MiddleN);
                    }
                }
            }
        }
    }
#endif
}

HRESULT CEarthHemsiphere::CreateRenderStates(ID3D11Device* pd3dDevice)
{
    HRESULT hr;

    // Create depth stencil state
    D3D11_DEPTH_STENCIL_DESC DisableDepthTestDSDesc;
    ZeroMemory(&DisableDepthTestDSDesc, sizeof(DisableDepthTestDSDesc));
    DisableDepthTestDSDesc.DepthEnable = FALSE;
    DisableDepthTestDSDesc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ZERO;
    V( pd3dDevice->CreateDepthStencilState(  &DisableDepthTestDSDesc, &m_pDisableDepthTestDS) );

	D3D11_DEPTH_STENCIL_DESC EnableDepthTestDSDesc;
	ZeroMemory(&EnableDepthTestDSDesc, sizeof(EnableDepthTestDSDesc));
	EnableDepthTestDSDesc.DepthEnable = TRUE;
	EnableDepthTestDSDesc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
	EnableDepthTestDSDesc.DepthFunc = D3D11_COMPARISON_GREATER;
    V( pd3dDevice->CreateDepthStencilState(  &EnableDepthTestDSDesc, &m_pEnableDepthTestDS) );

    // Create default blend state
    D3D11_BLEND_DESC DefaultBlendStateDesc;
    ZeroMemory(&DefaultBlendStateDesc, sizeof(DefaultBlendStateDesc));
    DefaultBlendStateDesc.IndependentBlendEnable = FALSE;
    for(int i=0; i< _countof(DefaultBlendStateDesc.RenderTarget); i++)
        DefaultBlendStateDesc.RenderTarget[i].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
    V( pd3dDevice->CreateBlendState( &DefaultBlendStateDesc, &m_pDefaultBS) );


    // Create rasterizer state for solid fill mode
    D3D11_RASTERIZER_DESC RSSolidFill = 
    {
        D3D11_FILL_SOLID,
        D3D11_CULL_BACK,
        TRUE, //BOOL FrontCounterClockwise;
        0,// INT DepthBias;
        0,// FLOAT DepthBiasClamp;
        0,// FLOAT SlopeScaledDepthBias;
        TRUE,//BOOL DepthClipEnable;
        FALSE,//BOOL ScissorEnable;
        FALSE,//BOOL MultisampleEnable;
        FALSE,//BOOL AntialiasedLineEnable;
    };
    hr = pd3dDevice->CreateRasterizerState( &RSSolidFill, &m_pRSSolidFill );

    D3D11_RASTERIZER_DESC RSWireframeFill = RSSolidFill;
    RSWireframeFill.FillMode = D3D11_FILL_WIREFRAME;
    hr = pd3dDevice->CreateRasterizerState( &RSWireframeFill, &m_pRSWireframeFill );

    D3D11_RASTERIZER_DESC SolidFillCullBackRSDesc;
    ZeroMemory(&SolidFillCullBackRSDesc, sizeof(SolidFillCullBackRSDesc));
    SolidFillCullBackRSDesc.FillMode = D3D11_FILL_SOLID;
    SolidFillCullBackRSDesc.CullMode = D3D11_CULL_NONE;
    V( pd3dDevice->CreateRasterizerState( &SolidFillCullBackRSDesc, &m_pRSSolidFillNoCull) );

    D3D11_RASTERIZER_DESC ZOnlyPassRSDesc = RSSolidFill;
    // Disable depth clipping
    ZOnlyPassRSDesc.DepthClipEnable = FALSE;
    // Do not use slope-scaled depth bias because this results in light leaking
    // through terrain!
    //ZOnlyPassRSDesc.DepthBias = -1;
    //ZOnlyPassRSDesc.SlopeScaledDepthBias = -4;
    //ZOnlyPassRSDesc.DepthBiasClamp = -1e-6;
    ZOnlyPassRSDesc.FrontCounterClockwise = FALSE;
    V( pd3dDevice->CreateRasterizerState( &ZOnlyPassRSDesc, &m_pRSZOnlyPass) );

    D3D11_SAMPLER_DESC SamLinearMirrorDesc = 
    {
        D3D11_FILTER_MIN_MAG_MIP_LINEAR,
        D3D11_TEXTURE_ADDRESS_MIRROR,
        D3D11_TEXTURE_ADDRESS_MIRROR,
        D3D11_TEXTURE_ADDRESS_MIRROR,
        0, //FLOAT MipLODBias;
        0, //UINT MaxAnisotropy;
        D3D11_COMPARISON_NEVER, // D3D11_COMPARISON_FUNC ComparisonFunc;
        {0.f, 0.f, 0.f, 0.f}, //FLOAT BorderColor[ 4 ];
        -FLT_MAX, //FLOAT MinLOD;
        +FLT_MAX //FLOAT MaxLOD;
    };
    V( pd3dDevice->CreateSamplerState( &SamLinearMirrorDesc, &m_psamLinearMirror) );

    D3D11_SAMPLER_DESC SamPointClamp = 
    {
        D3D11_FILTER_MIN_MAG_MIP_POINT,
        D3D11_TEXTURE_ADDRESS_CLAMP,
        D3D11_TEXTURE_ADDRESS_CLAMP,
        D3D11_TEXTURE_ADDRESS_CLAMP,
        0, //FLOAT MipLODBias;
        0, //UINT MaxAnisotropy;
        D3D11_COMPARISON_NEVER, // D3D11_COMPARISON_FUNC ComparisonFunc;
        {0.f, 0.f, 0.f, 0.f}, //FLOAT BorderColor[ 4 ];
        -FLT_MAX, //FLOAT MinLOD;
        +FLT_MAX //FLOAT MaxLOD;
    };
    V( pd3dDevice->CreateSamplerState( &SamPointClamp, &m_psamPointClamp) );
    
    D3D11_SAMPLER_DESC SamLinearClamp = 
    {
        D3D11_FILTER_MIN_MAG_MIP_LINEAR,
        D3D11_TEXTURE_ADDRESS_CLAMP,
        D3D11_TEXTURE_ADDRESS_CLAMP,
        D3D11_TEXTURE_ADDRESS_CLAMP,
        0, //FLOAT MipLODBias;
        0, //UINT MaxAnisotropy;
        D3D11_COMPARISON_NEVER, // D3D11_COMPARISON_FUNC ComparisonFunc;
        {0.f, 0.f, 0.f, 0.f}, //FLOAT BorderColor[ 4 ];
        -FLT_MAX, //FLOAT MinLOD;
        +FLT_MAX //FLOAT MaxLOD;
    };
    V( pd3dDevice->CreateSamplerState( &SamLinearClamp, &m_psamLinearClamp) );


    D3D11_SAMPLER_DESC SamComparison = 
    {
        D3D11_FILTER_COMPARISON_MIN_MAG_MIP_LINEAR,
        D3D11_TEXTURE_ADDRESS_CLAMP,
        D3D11_TEXTURE_ADDRESS_CLAMP,
        D3D11_TEXTURE_ADDRESS_CLAMP,
        0, //FLOAT MipLODBias;
        0, //UINT MaxAnisotropy;
        D3D11_COMPARISON_GREATER, // D3D11_COMPARISON_FUNC ComparisonFunc;
        {0.f, 0.f, 0.f, 0.f}, //FLOAT BorderColor[ 4 ];
        -FLT_MAX, //FLOAT MinLOD;
        +FLT_MAX //FLOAT MaxLOD;
    };
    V( pd3dDevice->CreateSamplerState( &SamComparison, &m_psamComaprison) );

    D3D11_SAMPLER_DESC SamLinearWrapDesc = 
    {
        D3D11_FILTER_MIN_MAG_MIP_LINEAR,
        D3D11_TEXTURE_ADDRESS_WRAP,
        D3D11_TEXTURE_ADDRESS_WRAP,
        D3D11_TEXTURE_ADDRESS_WRAP,
        0, //FLOAT MipLODBias;
        0, //UINT MaxAnisotropy;
        D3D11_COMPARISON_NEVER, // D3D11_COMPARISON_FUNC ComparisonFunc;
        {0.f, 0.f, 0.f, 0.f}, //FLOAT BorderColor[ 4 ];
        -FLT_MAX, //FLOAT MinLOD;
        +FLT_MAX //FLOAT MaxLOD;
    };
    V( pd3dDevice->CreateSamplerState( &SamLinearWrapDesc, &m_psamLinearWrap) );
    
    return S_OK;
}

void CEarthHemsiphere::RenderNormalMap(ID3D11Device* pd3dDevice,
                                       ID3D11DeviceContext* pd3dImmediateContext,
                                       const UINT16 *pHeightMap,
                                       size_t HeightMapPitch,
                                       int iHeightMapDim)
{
    HRESULT hr;

	D3D11_TEXTURE2D_DESC HeightMapDesc =
	{
        iHeightMapDim,
        iHeightMapDim,
		1,
		1,
        DXGI_FORMAT_R16_UNORM,
        {1,0},
		D3D11_USAGE_IMMUTABLE,
		D3D11_BIND_SHADER_RESOURCE,
		0,
		0
	};
    while( (iHeightMapDim >> HeightMapDesc.MipLevels) > 1 )
        ++HeightMapDesc.MipLevels;
    std::vector<UINT16> CoarseMipLevels;
    CoarseMipLevels.resize( iHeightMapDim/2 * iHeightMapDim );

    std::vector<D3D11_SUBRESOURCE_DATA> InitData(HeightMapDesc.MipLevels);
    InitData[0].pSysMem = pHeightMap;
    InitData[0].SysMemPitch = (UINT)HeightMapPitch*sizeof(pHeightMap[0]);
    const UINT16 *pFinerMipLevel = pHeightMap;
    UINT16 *pCurrMipLevel = &CoarseMipLevels[0];
    size_t FinerMipPitch = HeightMapPitch;
    size_t CurrMipPitch = iHeightMapDim/2;
    for(UINT uiMipLevel = 1; uiMipLevel < HeightMapDesc.MipLevels; ++uiMipLevel)
    {
        auto MipWidth  = HeightMapDesc.Width >> uiMipLevel;
        auto MipHeight = HeightMapDesc.Height >> uiMipLevel;
        for(UINT uiRow=0; uiRow < MipHeight; ++uiRow)
            for(UINT uiCol=0; uiCol < MipWidth; ++uiCol)
            {
                int iAverageHeight = 0;
                for(int i=0; i<2; ++i)
                    for(int j=0; j<2; ++j)
                        iAverageHeight += pFinerMipLevel[ (uiCol*2+i) + (uiRow*2+j)*FinerMipPitch];
                pCurrMipLevel[uiCol + uiRow*CurrMipPitch] = (UINT16)(iAverageHeight>>2);
            }

        InitData[uiMipLevel].pSysMem = pCurrMipLevel;
        InitData[uiMipLevel].SysMemPitch = (UINT)CurrMipPitch*sizeof(*pCurrMipLevel);
        pFinerMipLevel = pCurrMipLevel;
        FinerMipPitch = CurrMipPitch;
        pCurrMipLevel += MipHeight*CurrMipPitch;
        CurrMipPitch = iHeightMapDim/2;
    }

    CComPtr<ID3D11Texture2D> ptex2DHeightMap;
    pd3dDevice->CreateTexture2D(&HeightMapDesc, &InitData[0], &ptex2DHeightMap);
    CComPtr<ID3D11ShaderResourceView> ptex2DHeightMapSRV;
    pd3dDevice->CreateShaderResourceView(ptex2DHeightMap, nullptr, &ptex2DHeightMapSRV);

    D3D11_TEXTURE2D_DESC NormalMapDesc = HeightMapDesc;
    NormalMapDesc.Format = DXGI_FORMAT_R8G8_SNORM;
    NormalMapDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
    NormalMapDesc.Usage = D3D11_USAGE_DEFAULT;
    
    CComPtr<ID3D11Texture2D> ptex2DNormalMap;
    pd3dDevice->CreateTexture2D(&NormalMapDesc, &InitData[0], &ptex2DNormalMap);
    pd3dDevice->CreateShaderResourceView(ptex2DNormalMap, nullptr, &m_ptex2DNormalMapSRV);

    CComPtr<ID3D11RenderTargetView> pOrigRTV;
    CComPtr<ID3D11DepthStencilView> pOrigDSV;
    pd3dImmediateContext->OMGetRenderTargets( 1, &pOrigRTV, &pOrigDSV );

    D3D11_VIEWPORT OrigViewPort;
    UINT iNumOldViewports = 1;
    pd3dImmediateContext->RSGetViewports(&iNumOldViewports, &OrigViewPort);

    CRenderTechnique RenderNormalMapTech;
    RenderNormalMapTech.SetDeviceAndContext(pd3dDevice, pd3dImmediateContext);
	V( RenderNormalMapTech.CreateVGPShadersFromFile( L"fx\\Terrain.fx", "GenerateScreenSizeQuadVS", NULL, "GenerateNormalMapPS", NULL ) );
	RenderNormalMapTech.SetDS( m_pDisableDepthTestDS );
    RenderNormalMapTech.SetRS( m_pRSSolidFillNoCull );
	RenderNormalMapTech.SetBS( m_pDefaultBS );
    
    D3D11_VIEWPORT NewViewPort;
    NewViewPort.TopLeftX = 0;
    NewViewPort.TopLeftY = 0;
    NewViewPort.MinDepth = 0;
    NewViewPort.MaxDepth = 1;


    D3D11_BUFFER_DESC CBDesc = 
    {
        sizeof(SNMGenerationAttribs),
        D3D11_USAGE_DYNAMIC,
        D3D11_BIND_CONSTANT_BUFFER,
        D3D11_CPU_ACCESS_WRITE, //UINT CPUAccessFlags
        0, //UINT MiscFlags;
        0, //UINT StructureByteStride;
    };
    CComPtr<ID3D11Buffer> pcbNMGenerationAttribs;
    V( pd3dDevice->CreateBuffer( &CBDesc, NULL, &pcbNMGenerationAttribs) );

    pd3dImmediateContext->PSSetConstantBuffers(0, 1, &pcbNMGenerationAttribs.p);

    pd3dImmediateContext->PSSetShaderResources(0, 1, &ptex2DHeightMapSRV.p);
    pd3dImmediateContext->PSSetSamplers(0, 1, &m_psamPointClamp.p);
    for(UINT uiMipLevel = 0; uiMipLevel < NormalMapDesc.MipLevels; ++uiMipLevel)
    {
        D3D11_RENDER_TARGET_VIEW_DESC RTVDesc;
        RTVDesc.Format = NormalMapDesc.Format;
        RTVDesc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
        RTVDesc.Texture2D.MipSlice = uiMipLevel;
        CComPtr<ID3D11RenderTargetView> ptex2DNormalMapRTV;
        pd3dDevice->CreateRenderTargetView(ptex2DNormalMap, &RTVDesc, &ptex2DNormalMapRTV);

        NewViewPort.Width  = (float)(NormalMapDesc.Width  >> uiMipLevel);
        NewViewPort.Height = (float)(NormalMapDesc.Height >> uiMipLevel);
        pd3dImmediateContext->RSSetViewports(1, &NewViewPort);  

        pd3dImmediateContext->OMSetRenderTargets(1, &ptex2DNormalMapRTV.p, NULL);  

        SNMGenerationAttribs NMGenerationAttribs;
        NMGenerationAttribs.m_fElevationScale = m_Params.m_TerrainAttribs.m_fElevationScale;
        NMGenerationAttribs.m_fSampleSpacingInterval = m_Params.m_TerrainAttribs.m_fElevationSamplingInterval;
        NMGenerationAttribs.m_fMIPLevel = (float)uiMipLevel;
        UpdateConstantBuffer(pd3dImmediateContext, pcbNMGenerationAttribs, &NMGenerationAttribs, sizeof(NMGenerationAttribs));

        RenderNormalMapTech.Apply();

        pd3dImmediateContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
        pd3dImmediateContext->Draw(4,0);
    }

    pd3dImmediateContext->RSSetViewports(iNumOldViewports, &OrigViewPort);
    pd3dImmediateContext->OMSetRenderTargets( 1, &pOrigRTV.p, pOrigDSV );
}


HRESULT CEarthHemsiphere::OnD3D11CreateDevice( class CElevationDataSource *pDataSource,
                                               const SRenderingParams &Params,
                                               ID3D11Device* pd3dDevice,
                                               ID3D11DeviceContext* pd3dImmediateContext,
                                               LPCTSTR HeightMapPath,
                                               LPCTSTR MaterialMaskPath,
											   LPCTSTR *TileTexturePath,
                                               LPCTSTR *TileNormalMapPath)
{
    HRESULT hr;
    m_Params = Params;

    const UINT16 *pHeightMap;
    size_t HeightMapPitch;
    pDataSource->GetDataPtr(pHeightMap, HeightMapPitch);
    int iHeightMapDim = pDataSource->GetNumCols();
    assert(iHeightMapDim == pDataSource->GetNumRows() );

    std::vector<SHemisphereVertex> VB;
    std::vector<UINT> StitchIB;
    GenerateSphereGeometry(pd3dDevice, SAirScatteringAttribs().fEarthRadius, m_Params.m_iRingDimension, m_Params.m_iNumRings, pDataSource, m_Params.m_TerrainAttribs.m_fElevationSamplingInterval, m_Params.m_TerrainAttribs.m_fElevationScale, VB, StitchIB, m_SphereMeshes);

    D3D11_BUFFER_DESC VBDesc = 
    {
        (UINT)(VB.size() * sizeof(VB[0])),
        D3D11_USAGE_IMMUTABLE,
        D3D11_BIND_VERTEX_BUFFER,
        0, //UINT CPUAccessFlags
        0, //UINT MiscFlags;
        0, //UINT StructureByteStride;
    };
    D3D11_SUBRESOURCE_DATA VBInitData = {&VB[0], 0, 0};
    hr = pd3dDevice->CreateBuffer(&VBDesc, &VBInitData , &m_pVertBuff);
    CHECK_HR_RET(hr, _T("Failed to create the Earth heimsphere vertex buffer") )


    m_uiNumStitchIndices = (UINT)StitchIB.size();
    D3D11_BUFFER_DESC StitchIndexBufferDesc=
    {
        (UINT)(m_uiNumStitchIndices * sizeof(StitchIB[0])),
        D3D11_USAGE_IMMUTABLE,
        D3D11_BIND_INDEX_BUFFER,
        0, //UINT CPUAccessFlags
        0, //UINT MiscFlags;
        0, //UINT StructureByteStride;
    };
    D3D11_SUBRESOURCE_DATA IBInitData = {&StitchIB[0], 0, 0};
    // Create the buffer
    V( pd3dDevice->CreateBuffer( &StitchIndexBufferDesc, &IBInitData, &m_pStitchIndBuff) );

    V( CreateRenderStates(pd3dDevice) );

    m_RenderEarthHemisphereZOnlyTech.SetDeviceAndContext(pd3dDevice, pd3dImmediateContext);
	V( m_RenderEarthHemisphereZOnlyTech.CreateVGPShadersFromFile( L"fx\\Terrain.fx", "HemisphereZOnlyVS", NULL, NULL, NULL ) );
	m_RenderEarthHemisphereZOnlyTech.SetDS( m_pEnableDepthTestDS );
	m_RenderEarthHemisphereZOnlyTech.SetRS( m_pRSZOnlyPass );
	m_RenderEarthHemisphereZOnlyTech.SetBS( m_pDefaultBS );
    
    RenderNormalMap(pd3dDevice, pd3dImmediateContext, pHeightMap, HeightMapPitch, iHeightMapDim);

    D3DX11CreateShaderResourceViewFromFile(pd3dDevice, MaterialMaskPath, nullptr, nullptr, &m_ptex2DMtrlMaskSRV, nullptr);

    // Load tiles
	for(int iTileTex = 0; iTileTex < (int)NUM_TILE_TEXTURES; iTileTex++)
    {
		V( D3DX11CreateShaderResourceViewFromFile(pd3dDevice, TileTexturePath[iTileTex], NULL, NULL, &m_ptex2DTilesSRV[iTileTex], NULL) );

        D3DX11_IMAGE_LOAD_INFO LoadInfo;
        memset( &LoadInfo, 0, sizeof(LoadInfo));
        D3DX11_IMAGE_INFO NMFileInfo;
        D3DX11GetImageInfoFromFile(TileNormalMapPath[iTileTex],NULL,&NMFileInfo,NULL);
        LoadInfo.Width = NMFileInfo.Width;
        LoadInfo.Height = NMFileInfo.Height;
        LoadInfo.Depth = NMFileInfo.Depth;
        LoadInfo.MipLevels = (int)( log( (double)max(NMFileInfo.Width, NMFileInfo.Height)) / log(2.0) );
        LoadInfo.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        LoadInfo.Usage = D3D11_USAGE_IMMUTABLE;
        LoadInfo.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        LoadInfo.MipFilter = D3DX11_DEFAULT;
        LoadInfo.Filter = D3DX11_DEFAULT;
        V( D3DX11CreateShaderResourceViewFromFile(pd3dDevice, TileNormalMapPath[iTileTex], &LoadInfo, NULL, &m_ptex2DTilNormalMapsSRV[iTileTex], NULL) );
	}

    D3D11_BUFFER_DESC CBDesc = 
    {
        0,
        D3D11_USAGE_DYNAMIC,
        D3D11_BIND_CONSTANT_BUFFER,
        D3D11_CPU_ACCESS_WRITE, //UINT CPUAccessFlags
        0, //UINT MiscFlags;
        0, //UINT StructureByteStride;
    };

    CBDesc.ByteWidth = sizeof(STerrainAttribs);
    V( pd3dDevice->CreateBuffer( &CBDesc, NULL, &m_pcbTerrainAttribs) );

    return S_OK;
}

void CEarthHemsiphere::OnD3D11DestroyDevice()
{
    m_pStitchIndBuff.Release();
    m_SphereMeshes.clear();
    m_pVertBuff.Release();
    m_pInputLayout.Release();
    m_RenderEarthHemisphereTech.Release();
    m_RenderEarthHemisphereZOnlyTech.Release();
    m_psamPointClamp.Release();
    m_psamLinearMirror.Release();
    m_psamLinearWrap.Release();
    m_psamLinearClamp.Release();
    m_psamComaprison.Release();
	m_pEnableDepthTestDS.Release();
    m_pDisableDepthTestDS.Release();
	m_pDefaultBS.Release();
    m_pRSSolidFill.Release();
    m_pRSSolidFillNoCull.Release();
    m_pRSZOnlyPass.Release();
    m_pRSWireframeFill.Release();
    m_ptex2DNormalMapSRV.Release();

    m_pcbTerrainAttribs.Release();

    for(int iTileTex = 0; iTileTex < NUM_TILE_TEXTURES; iTileTex++)
    {
        m_ptex2DTilesSRV[iTileTex].Release();
        m_ptex2DTilNormalMapsSRV[iTileTex].Release();
    }
}

void CEarthHemsiphere::Render(ID3D11DeviceContext* pd3dImmediateContext,
                              const D3DXMATRIX &CameraViewProjMatrix,
                              ID3D11Buffer *pcbCameraAttribs,
                              ID3D11Buffer *pcbLightAttribs,
                              ID3D11Buffer *pcMediaScatteringParams,
                              ID3D11ShaderResourceView *pShadowMapSRV,
                              ID3D11ShaderResourceView *pLiSpCloudTransparencySRV,
                              ID3D11ShaderResourceView *pPrecomputedNetDensitySRV,
                              ID3D11ShaderResourceView *pAmbientSkylightSRV,
                              bool bZOnlyPass)
{
    if( GetAsyncKeyState(VK_F9) )
    {
        m_RenderEarthHemisphereTech.Release();
    }

    if( !m_RenderEarthHemisphereTech.IsValid() )
    {
        HRESULT hr;
        CD3DShaderMacroHelper Macros;
        Macros.AddShaderMacro("TEXTURING_MODE", m_Params.m_TexturingMode);
        Macros.AddShaderMacro("NUM_TILE_TEXTURES", NUM_TILE_TEXTURES);
        Macros.AddShaderMacro("NUM_SHADOW_CASCADES", m_Params.m_iNumShadowCascades);
        Macros.AddShaderMacro("BEST_CASCADE_SEARCH", m_Params.m_bBestCascadeSearch ? true : false);
        Macros.AddShaderMacro("SMOOTH_SHADOWS", m_Params.m_bSmoothShadows ? true : false);
        Macros.AddShaderMacro("ENABLE_CLOUDS", m_Params.m_bEnableClouds ? true : false);
        

        Macros.Finalize();
        CComPtr<ID3D11Device> pDevice;
        pd3dImmediateContext->GetDevice(&pDevice);
        m_RenderEarthHemisphereTech.SetDeviceAndContext(pDevice, pd3dImmediateContext);
	    V( m_RenderEarthHemisphereTech.CreateVGPShadersFromFile( L"fx\\Terrain.fx", "HemisphereVS", NULL, "HemispherePS", Macros ) );
	    m_RenderEarthHemisphereTech.SetDS( m_pEnableDepthTestDS );
	    m_RenderEarthHemisphereTech.SetRS( m_pRSSolidFill );
	    m_RenderEarthHemisphereTech.SetBS( m_pDefaultBS );
        if( !m_pInputLayout )
        {
	        // Create vertex input layout for bounding box buffer
            const D3D11_INPUT_ELEMENT_DESC layout[] =
            {
                { "WORLD_POS",  0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0*4, D3D11_INPUT_PER_VERTEX_DATA, 0 },
                { "MASK0_UV",   0, DXGI_FORMAT_R32G32_FLOAT,    0, 3*4, D3D11_INPUT_PER_VERTEX_DATA, 0 },
            };

	        auto pVSByteCode = m_RenderEarthHemisphereTech.GetVSByteCode();
            V( pDevice->CreateInputLayout( layout, ARRAYSIZE( layout ),
                                           pVSByteCode->GetBufferPointer(),
										   pVSByteCode->GetBufferSize(),
                                           &m_pInputLayout ) );
        }
    }

    SViewFrustum ViewFrustum;
    ExtractViewFrustumPlanesFromMatrix(CameraViewProjMatrix, ViewFrustum);

    UpdateConstantBuffer(pd3dImmediateContext, m_pcbTerrainAttribs, &m_Params.m_TerrainAttribs, sizeof(m_Params.m_TerrainAttribs));

    ID3D11Buffer *pCBs[] = 
    {
        m_pcbTerrainAttribs,
        pcMediaScatteringParams,
        pcbCameraAttribs,
        pcbLightAttribs
    };
    pd3dImmediateContext->VSSetConstantBuffers(0, _countof(pCBs), pCBs);
    pd3dImmediateContext->PSSetConstantBuffers(0, _countof(pCBs), pCBs);

    ID3D11ShaderResourceView *pSRVs[4 + 2*NUM_TILE_TEXTURES] = 
    {
        m_ptex2DNormalMapSRV,
        m_ptex2DMtrlMaskSRV,
        pShadowMapSRV,
        pLiSpCloudTransparencySRV
    };
    for(int iTileTex = 0; iTileTex < NUM_TILE_TEXTURES; iTileTex++)
    {
        pSRVs[4+iTileTex] = m_ptex2DTilesSRV[iTileTex];
        pSRVs[4+NUM_TILE_TEXTURES+iTileTex] = m_ptex2DTilNormalMapsSRV[iTileTex];
    }
    pd3dImmediateContext->PSSetShaderResources(1, _countof(pSRVs), pSRVs);
    pSRVs[0] = nullptr;
    pSRVs[1] = pAmbientSkylightSRV;
    pSRVs[2] = nullptr;
    pSRVs[3] = nullptr;
    pSRVs[4] = nullptr;
    pSRVs[5] = pPrecomputedNetDensitySRV;
    pd3dImmediateContext->VSSetShaderResources(0, 6, pSRVs);

    ID3D11SamplerState *pSamplers[] = {m_psamLinearClamp, m_psamLinearMirror, m_psamLinearWrap, m_psamComaprison};
	pd3dImmediateContext->VSSetSamplers(0, _countof(pSamplers), pSamplers);
	pd3dImmediateContext->PSSetSamplers(0, _countof(pSamplers), pSamplers);

    UINT offset[1] = { 0 };
    UINT stride[1] = { sizeof(SHemisphereVertex) };
    ID3D11Buffer* const ppBuffers[1] = { m_pVertBuff };
    pd3dImmediateContext->IASetVertexBuffers( 0, 1, ppBuffers, stride, offset );

    pd3dImmediateContext->IASetInputLayout( m_pInputLayout );
    pd3dImmediateContext->IASetPrimitiveTopology( D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP );
    
    if( bZOnlyPass )
        m_RenderEarthHemisphereZOnlyTech.Apply();
    else
        m_RenderEarthHemisphereTech.Apply();
    
    for(auto MeshIt = m_SphereMeshes.begin();  MeshIt != m_SphereMeshes.end(); ++MeshIt)
    {
        UINT uiPlaneFlags = TEST_ALL_PLANES;
        if( bZOnlyPass )
        {
            // It is necessary to ensure that shadow-casting patches, which are not visible 
            // in the frustum, are still rendered into the shadow map. 
            // For z-only pass, do not clip against far clipping plane (complimentary depth is used, 
            // hence far, not near plane)
            uiPlaneFlags &= ~TEST_FAR_PLANE;
        }
        if(IsBoxVisible(ViewFrustum, MeshIt->BoundBox, uiPlaneFlags))
        {
            pd3dImmediateContext->IASetIndexBuffer( MeshIt->pIndBuff, DXGI_FORMAT_R32_UINT, 0);
            pd3dImmediateContext->DrawIndexed(MeshIt->uiNumIndices, 0, 0);
        }
    }
    
    pd3dImmediateContext->IASetPrimitiveTopology( D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST );
    pd3dImmediateContext->IASetIndexBuffer( m_pStitchIndBuff, DXGI_FORMAT_R32_UINT, 0);
    pd3dImmediateContext->DrawIndexed(m_uiNumStitchIndices, 0, 0);

    UnbindPSResources(pd3dImmediateContext);
}

void CEarthHemsiphere::UpdateParams(const SRenderingParams &NewParams)
{
    if( m_Params.m_iNumShadowCascades    != NewParams.m_iNumShadowCascades    ||
        m_Params.m_bBestCascadeSearch    != NewParams.m_bBestCascadeSearch    || 
        m_Params.m_bSmoothShadows        != NewParams.m_bSmoothShadows ||
        m_Params.m_bEnableClouds         != NewParams.m_bEnableClouds )
    {
        m_RenderEarthHemisphereTech.Release();
    }

    m_Params = NewParams;
}
