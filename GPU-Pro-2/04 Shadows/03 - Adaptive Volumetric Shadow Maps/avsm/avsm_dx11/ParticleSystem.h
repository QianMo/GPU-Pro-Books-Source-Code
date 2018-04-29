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

#ifndef HEADER_PARTICLE_SYSTEM
#define HEADER_PARTICLE_SYSTEM

class Camera;
class CFirstPersonCamera;

typedef unsigned int UINT;
typedef unsigned char UCHAR;
typedef unsigned short USHORT;

// Version of the binary particles format
#define NUM_VERTS_PER_PARTICLE 4

extern D3DXMATRIX gLightViewProjection;

//--------------------------------------------------------------------------------------
const UINT MAX_SLICES    = 32;

//--------------------------------------------------------------------------------------
class ParticleEmitter 
{ 
public:
    float mpPos[3];
    float mpVelocity[3]; // Velocity of emitted particles, not the change in emitter position/time
    float mDrag;
    float mGravity;
    float mLifetime;
    float mStartSize;
    float mSizeRate;    
    float mRandScaleX;
    float mRandScaleY;
    float mRandScaleZ;

    ParticleEmitter();
    ~ParticleEmitter(){}
};

//--------------------------------------------------------------------------------------
// We need two structures for the particles: one for simulation, and another for rendering.
// The simulation structure contains state for each particle.
// The drawing strcuture contains data for each of the particle's vertices.
// We currently store six vertices for each particle - 2 triangles * 3 vertices per.
// We could reduce this to four with a "mini" index buffer - each particle follows the 
// same vertex-reference pattern: 0,1,2, 1,2,3.  There could also be a lrb-specific benefit
// to directly rasterizing the particles, maybe even rasterizing circles instead of quads.
//--------------------------------------------------------------------------------------
class SimpleVertex
{
public:
    float mpPos[3];
    float mpUV[2];
    float mSize;
    float mOpacity;
};


//--------------------------------------------------------------------------------------
// The Particle structure contains per-particle state required to simulate the behavior
class Particle
{
public:
    float mpPos[3];
    float mpVelocity[3];
    float mRemainingLife;
    float mSize;
    float mOpacity;
    float mSortDistance;
    UINT  mEmitterIdx;
};

//--------------------------------------------------------------------------------------
class ParticleSystem
{
    
public:
    ParticleSystem(UINT maxNumParticles);
    ~ParticleSystem();

    void SetNewParticleCount(UINT val) { mNewParticleCount = val; }
    void EnableOpacityUpdate(bool b) { mEnableOpacityUpdate = b; }
    void EnableSizeUpdate(bool b)    { mEnableSizeUpdate = b; }
    unsigned int GetParticleCount() const;
    ID3D11ShaderResourceView* GetParticleBufferSRV() {return mpParticleVertexBufferSRV;} const
    bool GetUnderBlendState() const { return mUnderBlend; }
    void GetBBox(D3DXVECTOR3 *min, D3DXVECTOR3 *max) {*min = mBBoxMin; *max = mBBoxMax;}
    
    void InitializeParticles(const ParticleEmitter** pEmitter, int emitterCount, ID3D11Device *pD3d, 
                             UINT width, UINT height,
                             const D3D10_SHADER_MACRO *shaderDefines);

    void UpdateParticles(CFirstPersonCamera *pViewCamera, CFirstPersonCamera *pLightCamera, float deltaSeconds);

    static int CompareZ(const void* a, const void* b);
    void SortParticles(float depthBounds[2], D3DXMATRIX* SortSpaceMat, bool SortBackToFront = true, UINT SliceCount = 1, bool EnableUnderBlend = false);
    void PopulateVertexBuffers(ID3D11DeviceContext *pD3dCtx);

    void Draw(ID3D11DeviceContext *pD3dCtx, ID3D10EffectTechnique* RenderTechnique, UINT Start, UINT Count, bool DrawSlice = false);

    void InitializeParticles(const int numVerts, const SimpleVertex * sv, ID3D11Device *pD3d, 
                             UINT width,  UINT height,
                             const D3D10_SHADER_MACRO *shaderDefines);

    HRESULT CreateParticlesBuffersAndVertexBuffer(UINT numVerts, HRESULT hr, ID3D11Device *pD3d);
    HRESULT InitializeParticlesLayoutAndEffect(ID3D11Device* pD3d, HRESULT hr, 
                                               const D3D10_SHADER_MACRO *shaderDefines);

private:
    void CleanupParticles();
    void UpdateLightViewProjection();
    void SpawnNewParticles();
    void ResetBBox();
    void UpdateBBox(Particle *particle);

    UINT mMaxNumParticles;
    UINT mMaxNumEmitters;
    ParticleEmitter *mpEmitters;

    UINT  mEmitterCount;
    UINT  mNumSlices;
    UINT  mActiveParticleCount;
    UINT  mEvenOdd;
    UINT  mNewParticleCount;
    float mLookDistance;
    float mLightDistance;
    float mLightWidth;
    float mLightHeight;
    float mLightNearClipDistance;
    float mLightFarClipDistance;
    float mSign;

    // Used to track min and max particle distance values for partitioning sort bins (i.e., slices)
    float mMinDist;
    float mMaxDist;
    float mMinDistPrev;
    float mMaxDistPrev;
    bool  mUnderBlend;

    // Used to track BB for all particles
    D3DXVECTOR3 mBBoxMin;
    D3DXVECTOR3 mBBoxMax;

    // Each particle has two triangles.  Each triangle has three vertices.  Total;
    UINT  mNumVertices;
    UINT  mpParticleCount[MAX_SLICES];

    Particle     *mpParticleBuffer[2];
    Particle     *mpFirstParticleSource;
    Particle     *mpFirstParticleDest;
    Particle     *mpCurParticleDest;
    UINT          mCurParticleIndex;
    UINT         *mpParticleSortIndex[MAX_SLICES];
    UINT         *mpSortBinHead[MAX_SLICES];
    D3DXVECTOR3   mLightPosition;
    D3DXVECTOR3   mLightLook;
    D3DXVECTOR3   mLightUp;
    D3DXVECTOR3   mLightRight;
    D3DXVECTOR3   mHalfAngleLook;
    D3DXVECTOR3   mEyeDirection;

    Camera        *mpCamera;
    bool          mEnableSizeUpdate;
    bool          mEnableOpacityUpdate;

    ID3D11InputLayout *mpParticleVertexLayout;
    ID3D11Buffer      *mpParticleVertexBuffer;
    ID3D11ShaderResourceView* mpParticleVertexBufferSRV;

    // sort dest buffer 
    static Particle *mpSortedParticleBuffer;

};

#endif HEADER_PARTICLE_SYSTEM
