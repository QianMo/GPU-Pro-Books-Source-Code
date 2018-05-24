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
#include "Common.fxh"
#include "CloudsCommon.fxh"

cbuffer cbPostProcessingAttribs : register( b0 )
{
    SGlobalCloudAttribs g_GlobalCloudAttribs;
};

Buffer<uint>                            g_ValidCellsCounter             : register( t0 );
StructuredBuffer<SParticleIdAndDist>    g_VisibleParticlesUnorderedList : register( t1 );
StructuredBuffer<SParticleIdAndDist>    g_PartiallySortedList           : register( t1 );
RWStructuredBuffer<SParticleIdAndDist>  g_rwMergedList                  : register( u0 );
RWStructuredBuffer<SParticleIdAndDist>  g_rwPartiallySortedBuf          : register( u0 );

groupshared SParticleIdAndDist g_LocalParticleIdAndDist[ THREAD_GROUP_SIZE ];

[numthreads(THREAD_GROUP_SIZE, 1, 1)]
void SortSubsequenceBitonicCS( uint3 Gid  : SV_GroupID, 
                               uint3 GTid : SV_GroupThreadID )
{
	// See http://en.wikipedia.org/wiki/Bitonic_sorter
    uint uiParticleSerialNum = (Gid.x * THREAD_GROUP_SIZE + GTid.x);
    uint uiNumVisibleParticles = g_ValidCellsCounter.Load(0);
    if( uiParticleSerialNum < uiNumVisibleParticles )
    {
        g_LocalParticleIdAndDist[GTid.x] = g_VisibleParticlesUnorderedList[uiParticleSerialNum];
    }
    else
    {
        g_LocalParticleIdAndDist[GTid.x].uiID = 0;
        g_LocalParticleIdAndDist[GTid.x].fDistToCamera = +FLT_MAX;
    }

    GroupMemoryBarrierWithGroupSync();

    const int NumPasses = log2(THREAD_GROUP_SIZE);
    for(int iPass = 0; iPass < NumPasses; ++iPass)
    {
        bool bIsIncreasing = ((GTid.x >> (iPass+1)) & 0x01) == 0;
        for(int iSubPass = 0; iSubPass <= iPass; ++iSubPass)
        {
            int Step = 1 << (iPass-iSubPass);
            if( ( ((int)(GTid.x)) & (2*Step-1)) < Step )
            {
                int LocalInd0 = GTid.x;
                int LocalInd1 = LocalInd0 + Step;
                SParticleIdAndDist P0 = g_LocalParticleIdAndDist[LocalInd0];
                SParticleIdAndDist P1 = g_LocalParticleIdAndDist[LocalInd1];
                if(  bIsIncreasing && P0.fDistToCamera > P1.fDistToCamera ||
                    !bIsIncreasing && P0.fDistToCamera < P1.fDistToCamera )
                {
                    g_LocalParticleIdAndDist[LocalInd0] = P1;
                    g_LocalParticleIdAndDist[LocalInd1] = P0;
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }

    if( uiParticleSerialNum < uiNumVisibleParticles )
    {
        g_rwPartiallySortedBuf[uiParticleSerialNum] = g_LocalParticleIdAndDist[GTid.x];
    }
}

RWBuffer<uint>  g_SortedParticleIndices : register( u0 );

uint GetElementRank(int iSubseqStart, int iSubseqLen, int IsRightHalf, SParticleIdAndDist SearchElem)
{
	if( iSubseqLen <= 0 )
		return 0;

    float fSearchDist = SearchElem.fDistToCamera;
	int Left = iSubseqStart;
	int Right = iSubseqStart + iSubseqLen-1;

#if 0
    // For debug purposes only: compute element rank using linear search
	int rank=0;
	for(int i=Left; i <= Right; ++i)
		if(  IsRightHalf && fSearchDist >= g_PartiallySortedList[i].fDistToCamera || 
            !IsRightHalf && fSearchDist >  g_PartiallySortedList[i].fDistToCamera )
			++rank;
	return rank;
#endif
    
    // Start binary search
	while(Right > Left+1)
	{
		int Middle = (Left+Right)>>1;
		float fMiddleDist = g_PartiallySortedList[Middle].fDistToCamera;
        // IMPORTANT NOTE: if we are searching for an element from the RIGHT subsequence in
        // the LEFT subsequence, we must compare using "<=", and we should use "<" otherwise
        // Consider the following two subsequences:
        //
        //      0 1 1 2    1 2 2 3
        // 
        // If always compare using "<", we will get the following ranks:
        //
        //
        //own rank     0 1 2 3    0 1 2 3
        //second rank  0 0 0 1    1 3 3 4
        //final pos    0 1 2 4    1 4 5 7
        //             ------------------
        //             0 1 1 2    1 2 2 3
        //
        // The resulting positions are not unique and the merged sequence will be incomplete 
        // and incorrect

        // If we use "<=" for the right subsequence, we will get the correct ranks:
        //
        //
        //own rank     0 1 2 3    0 1 2 3
        //second rank  0 0 0 1    3 4 4 4
        //final pos    0 1 2 4    3 5 6 7
        //             ------------------
        //             0 1 1 2    1 2 2 3
        //
        // This method guarantees stable merge as all equal elements from the left subsequence always precede
        // elements from the right subsequence
		if(  IsRightHalf && fMiddleDist <= fSearchDist ||
			!IsRightHalf && fMiddleDist <  fSearchDist )
			Left = Middle;
		else
			Right = Middle;
		// Suppose we are looking for x in the following sequence:
		//      0    1   2   3   4    5 
		//     x-1   x   x   x   x   x+1
		// For the right subsequence, the algorithm will work like this:
		//      l        m            r
		//               l   m        r
		//                   l   m    r
		//                       l    r
		// For the left subsequence, the algorithm will work like this:
		//      l        m            r
		//      l    m   r
		//      l    r
	}
    // After we exit from the loop, we need to precisely determine which interval we fall into:
	float fLeftDist  = g_PartiallySortedList[Left ].fDistToCamera;
	float fRightDist = g_PartiallySortedList[Right].fDistToCamera;
	if(  IsRightHalf && fRightDist <= fSearchDist || 
		!IsRightHalf && fRightDist <  fSearchDist )
		return Right+1 - iSubseqStart; 
	else if( IsRightHalf && fSearchDist <  fLeftDist || 
		    !IsRightHalf && fSearchDist <= fLeftDist )
		return Left - iSubseqStart;
	else
		return Left+1 - iSubseqStart; 
};

[numthreads(THREAD_GROUP_SIZE, 1, 1)]
void MergeSubsequencesCS( uint3 Gid  : SV_GroupID, 
                          uint3 GTid : SV_GroupThreadID )
{
    int iParticleSerialNum = (Gid.x * THREAD_GROUP_SIZE + GTid.x);
    int iNumVisibleParticles = g_ValidCellsCounter.Load(0);
	if( iParticleSerialNum >= iNumVisibleParticles )
		return;

	int SubseqLen = (int)g_GlobalCloudAttribs.uiParameter;
	if(SubseqLen >= iNumVisibleParticles*2 && 
       SubseqLen > THREAD_GROUP_SIZE // Take care of the situation when there are too few particles
                                     // We still need to copy them to the other buffer
       )
    {
        // The entire sequence is sorted and is stored in both buffers. No more work to do
       	return;
    }

	if(SubseqLen>=iNumVisibleParticles)
	{
        // The entire sequence is sorted, but we need to copy it to the other buffer
		g_rwMergedList[iParticleSerialNum] = g_PartiallySortedList[iParticleSerialNum];
		return;
	}

	SParticleIdAndDist CurrParticle = g_PartiallySortedList[iParticleSerialNum];

	int IsRightHalf = ((uint)iParticleSerialNum / (uint)SubseqLen) & 0x01;
	uint iElemRankInThisSubseq = iParticleSerialNum & (SubseqLen-1);
	uint iMergedSubseqStart = iParticleSerialNum & (-SubseqLen*2);
	int iSecondSubseqStart = iMergedSubseqStart + (1-IsRightHalf) * SubseqLen;
	int iSecondSubseqLen = min( SubseqLen, iNumVisibleParticles-iSecondSubseqStart );
	uint iElemRankInSecondSubseq = GetElementRank(iSecondSubseqStart, iSecondSubseqLen, IsRightHalf, CurrParticle);
	g_rwMergedList[iMergedSubseqStart + iElemRankInThisSubseq + iElemRankInSecondSubseq] = CurrParticle;
}


[numthreads(THREAD_GROUP_SIZE, 1, 1)]
void WriteSortedPariclesToVBCS( uint3 Gid  : SV_GroupID, 
                                uint3 GTid : SV_GroupThreadID )
{
    uint uiParticleSerialNum = (Gid.x * THREAD_GROUP_SIZE + GTid.x);
    uint uiNumVisibleParticles = g_ValidCellsCounter.Load(0);
    if( uiParticleSerialNum < uiNumVisibleParticles )
	{
		g_SortedParticleIndices[uiParticleSerialNum] = g_PartiallySortedList[uiParticleSerialNum].uiID;
	}
}
