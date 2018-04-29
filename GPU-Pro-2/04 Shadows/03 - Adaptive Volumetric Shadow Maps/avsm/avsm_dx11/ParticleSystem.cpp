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

#include "DXUT.h"
#include "App.H"
#include "ParticleSystem.h"
#include "Camera.h"

#include <limits>

D3DXMATRIX gLightViewProjection;

//--------------------------------------------------------------------------------------
ParticleEmitter::ParticleEmitter() :
    mDrag(4.0f),
    mGravity(-5.0f),
    mRandScaleX(1.0f),
    mRandScaleY(5.0f),
    mRandScaleZ(1.0f),
    mLifetime(0.2f),
    mStartSize(0.03f),
    mSizeRate(4.0f)
{
    mpPos[0] = 0.0f;
    mpPos[1] = 0.0f;
    mpPos[2] = 0.1f;

    mpVelocity[0] = 0.0f;
    mpVelocity[1] = 2.0f;
    mpVelocity[2] = 0.0f;

} // ParticleEmitter()

//--------------------------------------------------------------------------------------
ParticleSystem::ParticleSystem(UINT maxNumParticles)
   : mMaxNumParticles(maxNumParticles)
{
    mEmitterCount = 0;
    mpEmitters = NULL;
    mNumSlices = MAX_SLICES;
    mActiveParticleCount = 0;
    mEvenOdd = 0;  // We ping-pong the particle source buffers.  This allows us to discard dead particles.
    mNewParticleCount = 5; // Num particles to spawn each frame
    mLookDistance  = 10.0f;
    mLightDistance = 20.0f;
    mLightWidth    = 1.0f;
    mLightHeight   = 1.0f;
    mLightNearClipDistance = 0.1f;
    mLightFarClipDistance  = 100.0f;
    mSign;
    mUnderBlend = false;
    mEnableOpacityUpdate = true;
    mEnableSizeUpdate    = true;

    // Used to track min and max particle distance values for partitioning sort bins (i.e., slices)
    mMinDist = FLT_MAX;
    mMaxDist = -mMinDist;
    mMinDistPrev = FLT_MAX;
    mMaxDistPrev = -mMinDistPrev;


    // Each particle has two triangles.  Each triangle has three vertices.  Total = six vertices per particle
    const UINT maxVertices = mMaxNumParticles * 6;
    mNumVertices = maxVertices;

    mpFirstParticleSource = NULL;
    mpFirstParticleDest = NULL;
    mpCurParticleDest = NULL;
    mCurParticleIndex = 0;
    mpParticleVertexLayout = NULL;

    mpParticleBuffer[0] = NULL;
    mpParticleBuffer[1] = NULL;

    mpCamera = NULL;

    UINT ii;
    mpParticleVertexBuffer = NULL;
    for(ii = 0; ii < MAX_SLICES; ii++)
    {
        mpParticleCount[ii] = 0;
        mpParticleSortIndex[ii] = NULL;
        mpSortBinHead[ii] = NULL;
    }

};

//--------------------------------------------------------------------------------------
ParticleSystem::~ParticleSystem()
{
    CleanupParticles();
};

//--------------------------------------------------------------------------------------
// Create Direct3D device and swap chain
//--------------------------------------------------------------------------------------
void ParticleSystem::InitializeParticles(const ParticleEmitter** pEmitter,  int emitterCount, ID3D11Device *pD3d,
                                         UINT width,  UINT height, 
                                         const D3D10_SHADER_MACRO *shaderDefines)
{
    HRESULT hr = S_OK;
    
    V(InitializeParticlesLayoutAndEffect(pD3d, hr, shaderDefines));
    V(CreateParticlesBuffersAndVertexBuffer(mNumVertices, hr, pD3d));


    // ***************************************************
    // Intialize the emitters
    // ***************************************************
    mpEmitters = new ParticleEmitter[emitterCount];
    assert(mpEmitters);
    for (int i = 0; i < emitterCount; ++i) 
    {
        if (NULL != pEmitter[i]) {
            memcpy(&mpEmitters[i], pEmitter[i], sizeof(ParticleEmitter));
        } else {
            memset(&mpEmitters[i], 0, sizeof(ParticleEmitter));
        }
    }

    mEmitterCount = emitterCount;

    int seed = 1234;  // choose a seed value 
    srand(seed);  //initialize random number generator 

    D3DXVECTOR3 i0(0.0f, 0.0f, 0.0f);
    D3DXVECTOR3 i1(1.0f, 0.0f, 0.0f);
    D3DXVECTOR3 i2(0.0f, 0.0f, 1.0f);

    mpCamera = new Camera(1.0f,
                          1.0f,
                          1.0f,
                          1000.0f,
                          i0,
                          i1,
                          i2 );

} // InitializeParticles()


//--------------------------------------------------------------------------------------
void ParticleSystem::InitializeParticles(const int numVerts, const SimpleVertex * sv, ID3D11Device *pD3d,
                                         UINT width,  UINT height,
                                         const D3D10_SHADER_MACRO *shaderDefines)
{
    HRESULT hr = S_OK;

    V(InitializeParticlesLayoutAndEffect(pD3d, hr, shaderDefines));
    V(CreateParticlesBuffersAndVertexBuffer(numVerts, hr, pD3d));

    //
    // UpdateLightViewProjection  relies on the fact that 
    // Emitter is in front of the camera.
    // So, set the Emitter to one of the particles position.
    //
    
    mpEmitters = new ParticleEmitter[1];
    mpEmitters[0].mpPos[0] = sv[0].mpPos[0];
    mpEmitters[0].mpPos[1] = sv[0].mpPos[1];
    mpEmitters[0].mpPos[2] = sv[0].mpPos[2];
    mEmitterCount = 1;

    D3DXVECTOR3 i0(0.0f, 0.0f, 0.0f);
    D3DXVECTOR3 i1(1.0f, 0.0f, 0.0f);
    D3DXVECTOR3 i2(0.0f, 0.0f, 1.0f);

    mpCamera = new Camera(1.0f,
                          1.0f,
                          1.0f,
                          1000.0f,
                          i0,
                          i1,
                          i2 );

    //////////////////////////////////////////////////////////////////////////


    //Update the particles' system state.
    mEvenOdd = mEvenOdd ? 0 : 1;
    mpFirstParticleSource = mpParticleBuffer[mEvenOdd ? 0 : 1];
    mpFirstParticleDest   = mpParticleBuffer[mEvenOdd ? 1 : 0];

    mpCurParticleDest = mpFirstParticleDest;
    mCurParticleIndex = 0;

    // Use last frame's min and max as a starting point for this frame.
    mMinDist = mMinDistPrev;
    mMaxDist = mMaxDistPrev;

    // Reset last frame's min and max so we can determine them for this frame.
    mMinDistPrev = FLT_MAX;
    mMaxDistPrev = -mMinDistPrev;

    // Reset the sort bins
    UINT ii;
    for(ii = 0; ii < mNumSlices; ii++) {
        mpSortBinHead[ii] = mpParticleSortIndex[ii];
        mpParticleCount[ii] = 0;
    }

    // Update existing particles
    mActiveParticleCount = 0;

    //Initialize rest of the things and let the particles' system going.
    UINT newParticleCount = numVerts/6;
    Particle *pDest = mpCurParticleDest;
    UINT oldActiveParticleCount = mActiveParticleCount;
    mActiveParticleCount = std::min(mActiveParticleCount + newParticleCount, mMaxNumParticles);
    newParticleCount = mActiveParticleCount - oldActiveParticleCount;
    
    for( UINT ii=0; ii < newParticleCount; ii++ ) {
        UINT jj = 6*ii;
        pDest->mpPos[0] = sv[jj].mpPos[0];
        pDest->mpPos[1] = sv[jj].mpPos[1];
        pDest->mpPos[2] = sv[jj].mpPos[2];

        D3DXVECTOR3 pos(pDest->mpPos[0], pDest->mpPos[1], pDest->mpPos[2]);
        float sliceDistance  = D3DXVec3Dot( &pos, &mLightLook );
        pDest->mSortDistance = mSign * D3DXVec3Dot( &pos, &mEyeDirection );

        // Add this particle's index to the slice's bin
        float range = mMaxDist - mMinDist;
        float sliceWidth = range / mNumSlices;
        float minDist = mMaxDist - mNumSlices * sliceWidth;

        UINT sliceIndex = (UINT)((sliceDistance-minDist)/sliceWidth);
        sliceIndex = std::min(mNumSlices-1, sliceIndex);
        sliceIndex = std::max((UINT)0, sliceIndex);

        *mpSortBinHead[sliceIndex] = mCurParticleIndex;
        mpSortBinHead[sliceIndex]++;
        mpParticleCount[sliceIndex]++;
    
        pDest->mOpacity = sv[jj].mOpacity;
        pDest->mSize = sv[jj].mSize;
        pDest->mpVelocity[0] = pDest->mpVelocity[0] = pDest->mpVelocity[0] = 0.0f;

        pDest->mRemainingLife = mpEmitters[0].mLifetime;
        pDest++;
        mCurParticleIndex++;
    }
	
	// Copy all the particles into the second buffer too.
	memcpy(mpFirstParticleSource, mpCurParticleDest, newParticleCount*sizeof(Particle));

}

//--------------------------------------------------------------------------------------
// Clean up the objects we've created
//--------------------------------------------------------------------------------------
void ParticleSystem::CleanupParticles()
{
    UINT ii;
    for(ii = 0; ii < MAX_SLICES; ii++) {
        SAFE_DELETE_ARRAY( mpParticleSortIndex[ii] );
    }
    SAFE_RELEASE(mpParticleVertexBuffer);
    SAFE_RELEASE(mpParticleVertexBufferSRV);
    SAFE_RELEASE(mpParticleVertexLayout );

    SAFE_DELETE_ARRAY( mpParticleBuffer[0] );
    SAFE_DELETE_ARRAY( mpParticleBuffer[1] );
    SAFE_DELETE_ARRAY( mpEmitters );
    SAFE_DELETE( mpCamera );

} // CleanupParticles()

//--------------------------------------------------------------------------------------
void ParticleSystem::UpdateLightViewProjection()
{
    // temp hack - create a Camera from the DXUT Camera
    const D3DXVECTOR3 &cameraLook     = mpCamera->GetLook();

    mUnderBlend = D3DXVec3Dot( &cameraLook, &mLightLook ) > 0.0f;

    mSign = mUnderBlend ? 1.0f : -1.0f;
    mHalfAngleLook = mSign * cameraLook + mLightLook;
    D3DXVec3Normalize( &mHalfAngleLook, &mHalfAngleLook );

    mEyeDirection = cameraLook;
    D3DXVec3Normalize( &mEyeDirection, &mEyeDirection );

    D3DXVec3Cross( &mLightUp, &mLightRight, &mLightLook );
    D3DXVec3Normalize( &mLightUp, &mLightUp );
    D3DXVec3Cross( &mLightRight, &mLightLook, &mLightUp );

    // Arbitrarily choose a look point in front of the camera
    // D3DXVECTOR3 lookPoint = cameraPosition + mLookDistance * cameraLook;
    D3DXVECTOR3 lookPoint(
        mpEmitters[0].mpPos[0],
        mpEmitters[0].mpPos[1],
        mpEmitters[0].mpPos[2]
    );

    // Fake a view position
    mLightPosition = lookPoint - mLightLook * mLightDistance;

    D3DXMATRIX viewMatrix, projectionMatrix;
	D3DXMatrixLookAtLH
    (
        &viewMatrix,
        &mLightPosition,
        &lookPoint,
        &mLightUp
    );

 	D3DXMatrixOrthoLH
    (
        &projectionMatrix,
        mLightWidth,
        mLightHeight,
        mLightNearClipDistance,
        mLightFarClipDistance
    );

    gLightViewProjection = viewMatrix * projectionMatrix;
} // UpdateLightViewProjection()

unsigned int ParticleSystem::GetParticleCount() const
{
    unsigned int particleCount = 0;
    for(unsigned int i = 0; i < mNumSlices; ++i) {
        particleCount += mpParticleCount[i];
    }

    return  particleCount;
}

//-----------------------------------------------------
void ParticleSystem::SpawnNewParticles()
{
    // UINT newParticleCount = mpEmitter->mSpawnRate * mDeltaSeconds;
    UINT newParticleCount = mNewParticleCount;

    Particle *pDest = mpCurParticleDest;
    for (UINT k = 0; k < mEmitterCount; ++k) {
        UINT oldActiveParticleCount = mActiveParticleCount;
        mActiveParticleCount = std::min(mActiveParticleCount + newParticleCount, mMaxNumParticles);
        newParticleCount = mActiveParticleCount - oldActiveParticleCount;
        UINT ii;
        for( ii=0; ii < newParticleCount; ii++ ) {
            pDest->mEmitterIdx = k;
            pDest->mpPos[0] = mpEmitters[k].mpPos[0];
            pDest->mpPos[1] = mpEmitters[k].mpPos[1];
            pDest->mpPos[2] = mpEmitters[k].mpPos[2];

            D3DXVECTOR3 pos(pDest->mpPos[0], pDest->mpPos[1], pDest->mpPos[2]);
            float sliceDistance  = D3DXVec3Dot( &pos, &mLightLook );    
            pDest->mSortDistance = mSign * D3DXVec3Dot( &pos, &mEyeDirection );

            // Add this particle's index to the slice's bin
            float range = mMaxDist - mMinDist;
            float sliceWidth = range / mNumSlices;
            float minDist = mMaxDist - mNumSlices * sliceWidth;

            UINT sliceIndex = (UINT)((sliceDistance-minDist)/sliceWidth);
            sliceIndex = std::min(mNumSlices-1, sliceIndex);
            sliceIndex = std::max((UINT)0, sliceIndex);

            *mpSortBinHead[sliceIndex] = mCurParticleIndex;
            mpSortBinHead[sliceIndex]++;
            mpParticleCount[sliceIndex]++;

            // Radomize the angle and radius. 
            float  angle  = (float)(2.0 * D3DX_PI * ((double)rand()/(double)(RAND_MAX)));
            float  radius = (float)(mpEmitters[k].mRandScaleX *(double)rand()/(double)(RAND_MAX));
            float  randX = cos(angle) * radius;
            float  randY = (float)(mpEmitters[k].mRandScaleY * ((double)rand()/(double)(RAND_MAX) - 0.5));
            float  randZ = sin(angle) * radius;

            pDest->mpVelocity[0] = mpEmitters[k].mpVelocity[0] + randX;
            pDest->mpVelocity[1] = mpEmitters[k].mpVelocity[1] + randY;
            pDest->mpVelocity[2] = mpEmitters[k].mpVelocity[2] + randZ;

            pDest->mSize    = mpEmitters[k].mStartSize;
            pDest->mOpacity = 1.0f;

            pDest->mRemainingLife = mpEmitters[k].mLifetime;
            UpdateBBox(pDest);

            pDest++;
            mCurParticleIndex++;
        }
    }
} // SpawnNewParticles()

Particle *ParticleSystem::mpSortedParticleBuffer = NULL;
int ParticleSystem::CompareZ(const void* a, const void* b)
{
	const UINT* sa = static_cast<const UINT*>(a);
	const UINT* sb = static_cast<const UINT*>(b);
	
    float fa = mpSortedParticleBuffer[*sa].mSortDistance;
    float fb = mpSortedParticleBuffer[*sb].mSortDistance;
	if (fa > fb) {
		return 1;
	}
	if (fa < fb) {
		return  -1;
	}
	return 0;
}


//-----------------------------------------------------
void ParticleSystem::SortParticles(float depthBounds[2], D3DXMATRIX* SortSpaceMat, bool SortBackToFront, UINT SliceCount, bool EnableUnderBlend)
{
    float depthMin =   1e20f;
    float depthMax =  -1e20f;

    unsigned int i;
    unsigned int particleCount = GetParticleCount();
    // reset particles count and indices
    for(i = 0; i < mNumSlices; i++) {
        mpSortBinHead[i] = mpParticleSortIndex[i];
        mpParticleCount[i] = 0;
    }

    // Store slice count
    mNumSlices = SliceCount;

    // Compute sort key
    D3DXVECTOR4 transformedPos;
    mCurParticleIndex = 0;
    Particle *pDest = mpFirstParticleDest;    
    for(i = 0; i < particleCount; i++) { 
        D3DXVECTOR3 pos(pDest->mpPos[0], pDest->mpPos[1], pDest->mpPos[2]);

        D3DXVec3Transform(&transformedPos, &pos, SortSpaceMat);
        float sliceDistance = transformedPos.z;

        depthMin = std::min(depthMin, sliceDistance);
        depthMax = std::max(depthMax, sliceDistance);

        pDest->mSortDistance = sliceDistance;

        *mpSortBinHead[0] = mCurParticleIndex;
        mpSortBinHead[0]++; 
        mpParticleCount[0]++;

        pDest++;
        mCurParticleIndex++;
    }

    if (NULL != depthBounds) {
        depthBounds[0] = depthMin;
        depthBounds[1] = depthMax;
    }

    const float sliceWidth  = (depthMax - depthMin) / mNumSlices;
    if (mNumSlices > 1) {
        mCurParticleIndex = 0;
        pDest = mpFirstParticleDest; 

        for(i = 0; i < mNumSlices; i++) {
            mpSortBinHead[i] = mpParticleSortIndex[i];
            mpParticleCount[i] = 0;
        }

        // Slice!
        for(i = 0; i < particleCount; i++) { 
            UINT sliceIndex = (UINT)((pDest->mSortDistance - depthMin) / sliceWidth);

            sliceIndex = std::min(mNumSlices - 1, sliceIndex);
            sliceIndex = std::max((UINT)0, sliceIndex);

            *mpSortBinHead[sliceIndex] = mCurParticleIndex;
            mpSortBinHead[sliceIndex]++; 
            mpParticleCount[sliceIndex]++;

            pDest++;
            mCurParticleIndex++;
        }
    }

    // Compute sort distance
    UINT slice;    
    if (mNumSlices == 1) {
        UINT ii;
        UINT *pSortIndex = mpParticleSortIndex[0];
        UINT particleCount = mpParticleCount[0];
        const float sortSign = SortBackToFront ? -1.0f : 1.0f;
        for(ii =0; ii < particleCount; ii++) 
        {
            D3DXVECTOR4 transfPos;
            D3DXVECTOR4 pos(mpFirstParticleDest[pSortIndex[ii]].mpPos[0], 
                            mpFirstParticleDest[pSortIndex[ii]].mpPos[1], 
                            mpFirstParticleDest[pSortIndex[ii]].mpPos[2], 
                            1.0f);
        
            D3DXVec4Transform(&transfPos, &pos, SortSpaceMat);
            mpFirstParticleDest[pSortIndex[ii]].mSortDistance = sortSign * transfPos.z;
        }
    }
    
    for(slice = 0; slice < mNumSlices; slice++) {
        UINT *pSortIndex = mpParticleSortIndex[slice];
        UINT particleCount = mpParticleCount[slice];
        if( particleCount > 1) {
            // Pass destination particle buffer to the qsort callback
            mpSortedParticleBuffer = mpFirstParticleDest;        
            qsort(pSortIndex, particleCount, sizeof(pSortIndex[0]), CompareZ);
        }
    } // foreach slice
} // SortParticles()

//-----------------------------------------------------
void ParticleSystem::PopulateVertexBuffers(ID3D11DeviceContext *pD3dCtx)
{
    HRESULT hr;
    D3D11_MAPPED_SUBRESOURCE mappedResource;
    V(pD3dCtx->Map(mpParticleVertexBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource));

    SimpleVertex *pVertex = (SimpleVertex *)mappedResource.pData;
    UINT sliceIndex;
    for(sliceIndex = 0; sliceIndex < mNumSlices; sliceIndex++) {
        UINT *pParticleSortIndex = mpParticleSortIndex[sliceIndex];
        UINT particleCountThisSlice = mpParticleCount[sliceIndex];
        UINT ii;
        for( ii=0; ii < particleCountThisSlice; ii++ ) {
            Particle *pCurParticle = &mpFirstParticleDest[pParticleSortIndex[ii]];

            // For now, hack a world-view-projection transformation.
            float xx = pCurParticle->mpPos[0];
            float yy = pCurParticle->mpPos[1];
            float zz = pCurParticle->mpPos[2];
            float opacity = pCurParticle->mOpacity;

            // Fade with square of age.
            // opacity *= opacity;

            // Note: xx, yy, zz, size, UVs, and opacity are the same for all six vertices
            // Also, the UVs are always follow the same pattern (i.e., (0,1), (1,0), (0,0), etc..).
            pVertex[0].mpPos[0] = xx;
            pVertex[0].mpPos[1] = yy;
            pVertex[0].mpPos[2] = zz;
            pVertex[0].mpUV[0]  = 0.0f;
            pVertex[0].mpUV[1]  = 1.0f;
            pVertex[0].mSize    = pCurParticle->mSize;
            pVertex[0].mOpacity = opacity;

            pVertex[1].mpPos[0] = xx;
            pVertex[1].mpPos[1] = yy;
            pVertex[1].mpPos[2] = zz;

            pVertex[1].mpUV[0]  = 1.0f;
            pVertex[1].mpUV[1]  = 0.0f;
            pVertex[1].mSize    = pCurParticle->mSize;
            pVertex[1].mOpacity = opacity;

            pVertex[2].mpPos[0] = xx;
            pVertex[2].mpPos[1] = yy;
            pVertex[2].mpPos[2] = zz;
            pVertex[2].mpUV[0]  = 0.0f;
            pVertex[2].mpUV[1]  = 0.0f;
            pVertex[2].mSize    = pCurParticle->mSize;
            pVertex[2].mOpacity = opacity;

            pVertex[3].mpPos[0] = xx;
            pVertex[3].mpPos[1] = yy;
            pVertex[3].mpPos[2] = zz;
            pVertex[3].mpUV[0]  = 0.0f;
            pVertex[3].mpUV[1]  = 1.0f;
            pVertex[3].mSize    = pCurParticle->mSize;
            pVertex[3].mOpacity = opacity;

            pVertex[4].mpPos[0] = xx;
            pVertex[4].mpPos[1] = yy;
            pVertex[4].mpPos[2] = zz;
            pVertex[4].mpUV[0]  = 1.0f;
            pVertex[4].mpUV[1]  = 1.0f;
            pVertex[4].mSize    = pCurParticle->mSize;
            pVertex[4].mOpacity = opacity;

            pVertex[5].mpPos[0] = xx;
            pVertex[5].mpPos[1] = yy;
            pVertex[5].mpPos[2] = zz;
            pVertex[5].mpUV[0]  = 1.0f;
            pVertex[5].mpUV[1]  = 0.0f;
            pVertex[5].mSize    = pCurParticle->mSize;
            pVertex[5].mOpacity = opacity;

            pVertex += 6;
        }
    } // foreach slice

    pD3dCtx->Unmap(mpParticleVertexBuffer, 0);

} // PopulateVertexBuffers()

void ParticleSystem::ResetBBox()
{
    for (size_t p = 0; p < 3; ++p) {
        mBBoxMin[p] = std::numeric_limits<float>::max();
        mBBoxMax[p] = std::numeric_limits<float>::min();
    }
}

void ParticleSystem::UpdateBBox(Particle *particle)
{
    float radius = particle->mSize * 0.5f;
    for (size_t p = 0; p < 3; ++p) {
        mBBoxMin[p] = 
            std::min(particle->mpPos[p] - radius, mBBoxMin[p]);
        mBBoxMax[p] = 
            std::max(particle->mpPos[p] + radius, mBBoxMax[p]);
    }
}


//-----------------------------------------------------
void ParticleSystem::UpdateParticles(CFirstPersonCamera *pViewCamera,
                                     CFirstPersonCamera *pLightCamera,
                                     float deltaSeconds)
{
    deltaSeconds *= 2.0f;
    
    // Update the camera
    float fov, aspect, nearClip, farClip;
    pViewCamera->GetProjParams( &fov, &aspect, &nearClip, &farClip );

    mpCamera->SetPositionAndOrientation(
        fov,
        aspect,
        nearClip,
        farClip,
        *pViewCamera->GetEyePt(),
        *pViewCamera->GetWorldAhead(),
        *pViewCamera->GetWorldUp()
    );

    ResetBBox();

    // Ping pong between buffers
    mEvenOdd = mEvenOdd ? 0 : 1;
    mpFirstParticleSource = mpParticleBuffer[mEvenOdd ? 0 : 1];
    mpFirstParticleDest   = mpParticleBuffer[mEvenOdd ? 1 : 0];

    mpCurParticleDest = mpFirstParticleDest;
    mCurParticleIndex = 0;

    mLightRight = *pLightCamera->GetWorldRight();
    mLightUp    = *pLightCamera->GetWorldUp();
    mLightLook  = *pLightCamera->GetWorldAhead();

    UpdateLightViewProjection();

    // Use last frame's min and max as a starting point for this frame.
    mMinDist = mMinDistPrev;
    mMaxDist = mMaxDistPrev;

    // Reset last frame's min and max so we can determine them for this frame.
    mMinDistPrev = FLT_MAX;
    mMaxDistPrev = -mMinDistPrev;

    // Reset the sort bins
    UINT ii;
    for(ii = 0; ii < mNumSlices; ii++) {
        mpSortBinHead[ii] = mpParticleSortIndex[ii];
        mpParticleCount[ii] = 0;
    }

    // Update existing particles
    UINT endIndex   = std::min( mActiveParticleCount, mMaxNumParticles );
    mActiveParticleCount = 0;

    Particle *pCurParticleSource = &mpFirstParticleSource[0];
    for( ii=0; ii<endIndex; ii++ ) {
        // Subtract time before processing particle to avoid negative values.
        pCurParticleSource->mRemainingLife -= deltaSeconds;

        // Ignore "dead" particles.
        if( pCurParticleSource->mRemainingLife > 0.0f ) {
            // Get pointer to next particle and increment it.
            // Do it within a mutex to be sure we're the only one updating this particle.
            Particle *pDest = mpCurParticleDest++;
            UINT particleIndex = mCurParticleIndex++;
            mActiveParticleCount++;

            pDest->mRemainingLife = pCurParticleSource->mRemainingLife;

            pDest->mpPos[0] = pCurParticleSource->mpPos[0] + pCurParticleSource->mpVelocity[0] * deltaSeconds;
            pDest->mpPos[1] = pCurParticleSource->mpPos[1] + pCurParticleSource->mpVelocity[1] * deltaSeconds;
            pDest->mpPos[2] = pCurParticleSource->mpPos[2] + pCurParticleSource->mpVelocity[2] * deltaSeconds;

            D3DXVECTOR3 pos(pDest->mpPos[0], pDest->mpPos[1], pDest->mpPos[2]);
            float sliceDistance = D3DXVec3Dot( &pos, &mHalfAngleLook );

            mMinDist = std::min( mMinDist, sliceDistance );
            mMaxDist = std::max( mMaxDist, sliceDistance );

            mMinDistPrev = std::min( mMinDistPrev, sliceDistance );
            mMaxDistPrev = std::max( mMaxDistPrev, sliceDistance );

            // *************************
            // Add this particle's index to the slice's bin
            // *************************
            float range = mMaxDist - mMinDist;
            float sliceWidth = range / mNumSlices;
            float minDist = mMaxDist - mNumSlices * sliceWidth;

            UINT sliceIndex = (UINT)((sliceDistance-minDist)/sliceWidth);
            sliceIndex = std::min(mNumSlices-1, sliceIndex);
            sliceIndex = std::max((UINT)0, sliceIndex);

            *mpSortBinHead[sliceIndex] = particleIndex;
            mpSortBinHead[sliceIndex]++;
            mpParticleCount[sliceIndex]++;

            // Update velocity
            UINT idx = 0;//pCurParticleSource->mEmitterIdx;
            float velocityScale = 1.0f - (mpEmitters[idx].mDrag * deltaSeconds);
            pDest->mpVelocity[0] = pCurParticleSource->mpVelocity[0] * velocityScale;
            pDest->mpVelocity[1] = pCurParticleSource->mpVelocity[1] * velocityScale
                                 + mpEmitters[idx].mGravity * deltaSeconds; // Y also gets gravity
            pDest->mpVelocity[2] = pCurParticleSource->mpVelocity[2] * velocityScale;

            if (mEnableSizeUpdate) {
                pDest->mSize = pCurParticleSource->mSize * (1.0f + mpEmitters[idx].mSizeRate * deltaSeconds);
            }

            static float opacityScale = 0.5f;
            if (mEnableOpacityUpdate) {
		        pDest->mOpacity = opacityScale * pCurParticleSource->mRemainingLife / mpEmitters[idx].mLifetime;
            }

            UpdateBBox(pDest);
        }
        pCurParticleSource++;
    } // foreach particle

    
    if (deltaSeconds) {
        SpawnNewParticles();
    }
} // UpdateParticles()
void ParticleSystem::Draw(ID3D11DeviceContext *pD3dCtx, ID3D10EffectTechnique* RenderTechnique, UINT Start, UINT Count, bool DrawSlice)
{
    // Setup input assembler
    UINT vbOffset = 0;
    UINT Stride = sizeof(SimpleVertex);
    pD3dCtx->IASetInputLayout(mpParticleVertexLayout);
    pD3dCtx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    pD3dCtx->IASetVertexBuffers(0, 1, &mpParticleVertexBuffer, &Stride, &vbOffset);

    if (DrawSlice) 
    {
        UINT sliceIndex;
        UINT particleCount = 0;
        
        for(sliceIndex = 0; sliceIndex < Start; sliceIndex++) {
            particleCount += mpParticleCount[sliceIndex];
        }

        Count = mpParticleCount[Start];
        Start = particleCount;
    }

    pD3dCtx->Draw(6 * Count, 6 * Start);
}

HRESULT ParticleSystem::InitializeParticlesLayoutAndEffect(ID3D11Device* pD3d, HRESULT hr, 
                                                           const D3D10_SHADER_MACRO *shaderDefines)
{
    ID3D10Blob *vertexShaderBlob = NULL;
    hr = D3DX11CompileFromFile(L"Particle.hlsl", shaderDefines, 0,
                               "DynamicParticlesShading_VS", "vs_4_0",
                               0, 0, 0, &vertexShaderBlob, 0, 0);
    assert(SUCCEEDED(hr));

    // ***************************************************
    // Define the input layout
    // ***************************************************
    D3D11_INPUT_ELEMENT_DESC layout[] =
    {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0,  0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
        { "TEXCOORD", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0 },
        { "TEXCOORD", 1, DXGI_FORMAT_R32_FLOAT,       0, 24, D3D11_INPUT_PER_VERTEX_DATA, 0 },
    };

    UINT numElements = sizeof( layout ) / sizeof( layout[0] );
    hr = pD3d->CreateInputLayout( 
        layout,
        numElements,
        vertexShaderBlob->GetBufferPointer(),
        vertexShaderBlob->GetBufferSize(),
        &mpParticleVertexLayout
        );

    return hr;
}

HRESULT ParticleSystem::CreateParticlesBuffersAndVertexBuffer( UINT numVerts, HRESULT hr, ID3D11Device * pD3d )
{
    // ***************************************************
    // Create particle simulation buffers
    // ***************************************************
    mpParticleBuffer[0] = new Particle[mMaxNumParticles]; 
    mpParticleBuffer[1] = new Particle[mMaxNumParticles];

    UINT ii;
    for(ii = 0; ii < MAX_SLICES; ii++) {
        mpParticleSortIndex[ii] = new UINT[mMaxNumParticles];
    }

    // ***************************************************
    // Create Vertex Buffers
    // ***************************************************
    D3D11_BUFFER_DESC bd;
    bd.Usage = D3D11_USAGE_DYNAMIC;
    bd.ByteWidth = sizeof(SimpleVertex) * numVerts;
    bd.BindFlags = D3D11_BIND_VERTEX_BUFFER | D3D10_BIND_SHADER_RESOURCE;
    bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    bd.MiscFlags = 0;

    hr = pD3d->CreateBuffer(&bd, NULL, &mpParticleVertexBuffer);

    CD3D11_SHADER_RESOURCE_VIEW_DESC shaderResourceDesc(
        D3D11_SRV_DIMENSION_BUFFER,
        DXGI_FORMAT_R32_FLOAT,
        0, (sizeof(SimpleVertex) * numVerts) / sizeof(float), 1);
    hr = pD3d->CreateShaderResourceView(mpParticleVertexBuffer, 
                                        &shaderResourceDesc, 
                                        &mpParticleVertexBufferSRV);

    return hr;
}
