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

#include "ElevationDataSource.h"
#include "DynamicQuadTreeNode.h"
#include <exception>

#include <wincodec.h>
#include <wincodecsdk.h>
#pragma comment(lib, "WindowsCodecs.lib")

// Creates data source from the specified raw data file
CElevationDataSource::CElevationDataSource(LPCTSTR strSrcDemFile) : 
    m_iPatchSize(128),
    m_iNumLevels(0),
    m_iColOffset(0), 
    m_iRowOffset(0)
{
    HRESULT hr;
    V( CoInitialize(NULL) );

    // Create components to read 16-bit png data
    CComPtr<IWICImagingFactory> pFactory;
    hr = pFactory.CoCreateInstance(CLSID_WICImagingFactory);
    if( FAILED(hr) )
        throw std::exception("Failed to create WICImagingFactory");

    CComPtr<IWICStream> pInputStream;
    hr = pFactory->CreateStream(&pInputStream);
    if( FAILED(hr) )
        throw std::exception("Failed to create WICStream");

    hr = pInputStream->InitializeFromFilename(strSrcDemFile, GENERIC_READ);
    if( FAILED(hr) )
        throw std::exception("Failed to initialize WICStream from file");

    CComPtr<IWICBitmapDecoder> pDecoder;
    hr = pFactory->CreateDecoderFromStream(
                  pInputStream,
                  0, // vendor
                  WICDecodeMetadataCacheOnDemand,
                  &pDecoder);
    if( FAILED(hr) )
        throw std::exception("Failed to Create decoder from stream");

    //GUID ContainerFormat;
    //pDecoder->GetContainerFormat(&ContainerFormat);
    //if( ContainerFormat == GUID_ContainerFormatPng )
    //    printf("Container Format: PNG\n");
    //else if( ContainerFormat == GUID_ContainerFormatTiff )
    //    printf("Container Format: TIFF\n");

    UINT frameCount = 0;
    hr = pDecoder->GetFrameCount(&frameCount);
    //printf("Frame count %d\n", frameCount);
    assert( frameCount == 1 );

    CComPtr<IWICBitmapFrameDecode> pTheFrame;
    pDecoder->GetFrame(0, &pTheFrame);

    UINT width = 0;
    UINT height = 0;
    pTheFrame->GetSize(&width, &height);

    // Calculate minimal number of columns and rows
    // in the form 2^n+1 that encompass the data
    m_iNumCols = 1;
    m_iNumRows = 1;
    while( m_iNumCols+1 < width || m_iNumRows+1 < height)
    {
        m_iNumCols *= 2;
        m_iNumRows *= 2;
    }

    m_iNumLevels = 1;
    while( (m_iPatchSize << (m_iNumLevels-1)) < (int)m_iNumCols ||
           (m_iPatchSize << (m_iNumLevels-1)) < (int)m_iNumRows )
        m_iNumLevels++;

    m_iNumCols++;
    m_iNumRows++;

    GUID pixelFormat = { 0 };
    pTheFrame->GetPixelFormat(&pixelFormat);
    if( pixelFormat != GUID_WICPixelFormat16bppGray )
    {
        assert(false);
        throw std::exception("expected 16 bit format");
    }

    // Load the data
    m_TheHeightMap.resize( m_iNumCols * m_iNumRows );
    WICRect SrcRect;
    SrcRect.X = 0;
    SrcRect.Y = 0;
    SrcRect.Height = height;
    SrcRect.Width = width;
    pTheFrame->CopyPixels(
      &SrcRect,
      (UINT)m_iNumCols*2, //UINT stride
      (UINT)m_TheHeightMap.size()*2, //UINT bufferSize
      (BYTE*)&m_TheHeightMap[0]);

    // Duplicate the last row and column
    for(UINT iRow = 0; iRow < height; iRow++)
        for(UINT iCol = width; iCol < m_iNumCols; iCol++)
            m_TheHeightMap[iCol + iRow * m_iNumCols] = m_TheHeightMap[(width-1) + iRow * m_iNumCols];
    for(UINT iCol = 0; iCol < m_iNumCols; iCol++)
        for(UINT iRow = height; iRow < m_iNumRows; iRow++)
            m_TheHeightMap[iCol + iRow * m_iNumCols] = m_TheHeightMap[iCol + (height-1) * m_iNumCols];

    pTheFrame.Release();
    pFactory.Release();
    pDecoder.Release();
    pInputStream.Release();

    CoUninitialize();

    m_MinMaxElevation.Resize(m_iNumLevels);
    
    // Calcualte min/max elevations
    CalculateMinMaxElevations();
}

CElevationDataSource::~CElevationDataSource(void)
{
}

UINT16 CElevationDataSource :: GetGlobalMinElevation()const
{
    return m_MinMaxElevation[SQuadTreeNodeLocation()].first;
}

UINT16 CElevationDataSource :: GetGlobalMaxElevation()const
{
    return m_MinMaxElevation[SQuadTreeNodeLocation()].second;
}

int MirrorCoord(int iCoord, int iDim)
{
    iCoord = abs(iCoord);
    int iPeriod = iCoord / iDim;
    iCoord = iCoord % iDim;
    if( iPeriod & 0x01 )
    {
        iCoord = (iDim-1) - iCoord;
    }
    return iCoord;
}

float CElevationDataSource::GetInterpolatedHeight(float fCol, float fRow, int iStep)const
{
    float fCol0 = floor(fCol);
    float fRow0 = floor(fRow);
    int iCol0 = static_cast<int>(fCol0);
    int iRow0 = static_cast<int>(fRow0);
    iCol0 = (iCol0/iStep)*iStep;
    iRow0 = (iRow0/iStep)*iStep;
    float fHWeight = (fCol - (float)iCol0) / (float)iStep;
    float fVWeight = (fRow - (float)iRow0) / (float)iStep;
    iCol0 += m_iColOffset;
    iRow0 += m_iRowOffset;
    //if( iCol0 < 0 || iCol0 >= (int)m_iNumCols || iRow0 < 0 || iRow0 >= (int)m_iNumRows )
    //    return -FLT_MAX;

    int iCol1 = iCol0+iStep;//min(iCol0+iStep, (int)m_iNumCols-1);
    int iRow1 = iRow0+iStep;//min(iRow0+iStep, (int)m_iNumRows-1);

    iCol0 = MirrorCoord(iCol0, m_iNumCols);
    iCol1 = MirrorCoord(iCol1, m_iNumCols);
    iRow0 = MirrorCoord(iRow0, m_iNumRows);
    iRow1 = MirrorCoord(iRow1, m_iNumRows);

    UINT16 H00 = m_TheHeightMap[iCol0 + iRow0 * m_iNumCols];
    UINT16 H10 = m_TheHeightMap[iCol1 + iRow0 * m_iNumCols];
    UINT16 H01 = m_TheHeightMap[iCol0 + iRow1 * m_iNumCols];
    UINT16 H11 = m_TheHeightMap[iCol1 + iRow1 * m_iNumCols];
    float fInterpolatedHeight = (H00 * (1 - fHWeight) + H10 * fHWeight) * (1-fVWeight) + 
                                (H01 * (1 - fHWeight) + H11 * fHWeight) * fVWeight;
    return fInterpolatedHeight;
}

D3DXVECTOR3 CElevationDataSource::ComputeSurfaceNormal(float fCol, float fRow,
                                                       float fSampleSpacing,
                                                       float fHeightScale, 
                                                       int iStep)const
{
    float Height1 = GetInterpolatedHeight(fCol + (float)iStep, fRow, iStep);
    float Height2 = GetInterpolatedHeight(fCol - (float)iStep, fRow, iStep);
    float Height3 = GetInterpolatedHeight(fCol, fRow + (float)iStep, iStep);
    float Height4 = GetInterpolatedHeight(fCol, fRow - (float)iStep, iStep);
       
    D3DXVECTOR3 Grad;
    Grad.x = Height2 - Height1;
    Grad.y = Height4 - Height3;
    Grad.z = (float)iStep * fSampleSpacing * 2.f;

    Grad.x *= fHeightScale;
    Grad.y *= fHeightScale;
    D3DXVECTOR3 Normal;
    D3DXVec3Normalize(&Normal, &Grad);

    return Normal;
}

void CElevationDataSource::RecomputePatchMinMaxElevations(const SQuadTreeNodeLocation &pos)
{
	if( pos.level == m_iNumLevels-1 )
	{
		std::pair<UINT16, UINT16> &CurrPatchMinMaxElev = m_MinMaxElevation[SQuadTreeNodeLocation(pos.horzOrder, pos.vertOrder, pos.level)];
        int iStartCol = pos.horzOrder*m_iPatchSize;
        int iStartRow = pos.vertOrder*m_iPatchSize;
        CurrPatchMinMaxElev.first = CurrPatchMinMaxElev.second = m_TheHeightMap[iStartCol + iStartRow*m_iNumCols];
        for(int iRow = iStartRow; iRow <= iStartRow + m_iPatchSize; iRow++)
            for(int iCol = iStartCol; iCol <= iStartCol + m_iPatchSize; iCol++)
            {
                UINT16 CurrElev = m_TheHeightMap[iCol + iRow*m_iNumCols];
                CurrPatchMinMaxElev.first = min(CurrPatchMinMaxElev.first, CurrElev);
                CurrPatchMinMaxElev.second = max(CurrPatchMinMaxElev.second, CurrElev);
            }
	}
	else
	{
        std::pair<UINT16, UINT16> &CurrPatchMinMaxElev = m_MinMaxElevation[pos];
        std::pair<UINT16, UINT16> &LBChildMinMaxElev = m_MinMaxElevation[GetChildLocation(pos, 0)];
        std::pair<UINT16, UINT16> &RBChildMinMaxElev = m_MinMaxElevation[GetChildLocation(pos, 1)];
        std::pair<UINT16, UINT16> &LTChildMinMaxElev = m_MinMaxElevation[GetChildLocation(pos, 2)];
        std::pair<UINT16, UINT16> &RTChildMinMaxElev = m_MinMaxElevation[GetChildLocation(pos, 3)];

        CurrPatchMinMaxElev.first = min( LBChildMinMaxElev.first, RBChildMinMaxElev.first );
        CurrPatchMinMaxElev.first = min( CurrPatchMinMaxElev.first, LTChildMinMaxElev.first );
        CurrPatchMinMaxElev.first = min( CurrPatchMinMaxElev.first, RTChildMinMaxElev.first );

        CurrPatchMinMaxElev.second = max( LBChildMinMaxElev.second, RBChildMinMaxElev.second);
        CurrPatchMinMaxElev.second = max( CurrPatchMinMaxElev.second, LTChildMinMaxElev.second );
        CurrPatchMinMaxElev.second = max( CurrPatchMinMaxElev.second, RTChildMinMaxElev.second );
	}
}

// Calculates min/max elevations for the hierarchy
void CElevationDataSource :: CalculateMinMaxElevations()
{
    // Calculate min/max elevations starting from the finest level
    for( HierarchyReverseIterator it(m_iNumLevels); it.IsValid(); it.Next() )
    {
		RecomputePatchMinMaxElevations(it);
    }
}

void CElevationDataSource::GetDataPtr(const UINT16* &pDataPtr, size_t &Pitch)
{
	pDataPtr = &m_TheHeightMap[0];
	Pitch = m_iNumCols;
}
