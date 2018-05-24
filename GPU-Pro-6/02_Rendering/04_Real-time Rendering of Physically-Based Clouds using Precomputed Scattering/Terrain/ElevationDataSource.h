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

#pragma once

#include <vector>
#include "HierarchyArray.h"
#include "DynamicQuadTreeNode.h"

// Class implementing elevation data source
class CElevationDataSource
{
public:
    // Creates data source from the specified raw data file
    CElevationDataSource(LPCTSTR strSrcDemFile);
    virtual ~CElevationDataSource(void);

	void GetDataPtr(const UINT16* &pDataPtr, size_t &Pitch);
    
    // Returns minimal height of the whole terrain
    UINT16 GetGlobalMinElevation()const;

    // Returns maximal height of the whole terrain
    UINT16 GetGlobalMaxElevation()const;

    void RecomputePatchMinMaxElevations(const SQuadTreeNodeLocation &pos);
    
    void SetOffsets(int iColOffset, int iRowOffset){m_iColOffset = iColOffset; m_iRowOffset = iRowOffset;}
    void GetOffsets(int &iColOffset, int &iRowOffset)const{iColOffset = m_iColOffset; iRowOffset = m_iRowOffset;}

    float GetInterpolatedHeight(float fCol, float fRow, int iStep = 1)const;
    
    D3DXVECTOR3 ComputeSurfaceNormal(float fCol, float fRow,
                                     float fSampleSpacing,
                                     float fHeightScale, 
                                     int iStep = 1)const;

    unsigned int GetNumCols()const{return m_iNumCols;}
    unsigned int GetNumRows()const{return m_iNumRows;}
private:
    CElevationDataSource();

    // Calculates min/max elevations for all patches in the tree
    void CalculateMinMaxElevations();
    
    // Hierarchy array storing minimal and maximal heights for quad tree nodes
    HierarchyArray< std::pair<UINT16, UINT16> > m_MinMaxElevation;
    
    int m_iNumLevels;
    int m_iPatchSize;
    int m_iColOffset, m_iRowOffset;
    
    // The whole terrain height map
    std::vector<UINT16> m_TheHeightMap;
    unsigned int m_iNumCols, m_iNumRows;
};
