//--------------------------------------------------------------------------------------
// File: SceneManger.cpp
//
// This is where the shadows and light linked list are calculated and rendered.
// This sample is based off Microsoft DirectX SDK sample CascadedShadowMap11
//
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------


#include "dxut.h"

#include <DirectXColors.h>
#include "SceneManager.h"
#include "DXUTcamera.h"
#include "SDKMesh.h"
#include "DirectXCollision.h"
#include "SDKmisc.h" 

using namespace DirectX;

static const XMVECTORF32 g_vHalfVector = { 0.5f, 0.5f, 0.5f, 0.5f };
static const XMVECTORF32 g_vMultiplySetzwToZero = { 1.0f, 1.0f, 0.0f, 0.0f };
static const XMVECTORF32 g_vZero = { 0.0f, 0.0f, 0.0f, 0.0f };
static const float       g_PixelOffset      = 0.0f;
static const float       g_BlendFactors[4]  = { 1.f, 1.f, 1.f, 1.f };
static DXGI_FORMAT       g_GBufferFormats[] =
{      
  DXGI_FORMAT_R16G16B16A16_FLOAT,    
  DXGI_FORMAT_R11G11B10_FLOAT,  
};

//--------------------------------------------------------------------------------------
inline void GetScaleOffset(float start, float end, float start_map_to, float end_map_to, float* result_scale, float* result_offset)
{
  float scale = (end_map_to - start_map_to) / (end - start);
  *result_scale = scale;
  *result_offset = start_map_to - (start * scale);
}

//--------------------------------------------------------------------------------------
// Call into deallocator.  
//--------------------------------------------------------------------------------------
SceneManager::~SceneManager() 
{
    ReleaseResources();
};

//--------------------------------------------------------------------------------------
HRESULT SceneManager::OnResize(uint32_t width, uint32_t height)
{
  // Validate
  if((width == m_MainVP.Width) && (height == m_MainVP.Height))
  {
    return S_OK;
  }

  m_MainVP.Width    = (float)width;
  m_MainVP.Height   = (float)height;
  m_MainVP.MinDepth = 0;
  m_MainVP.MaxDepth = 1;
  m_MainVP.TopLeftX = 0;
  m_MainVP.TopLeftY = 0;

  // Allocate the GBuffer target
  m_GBufferRT.Init( width, height, 2, g_GBufferFormats, DXGI_FORMAT_D32_FLOAT_S8X24_UINT );

  // Allocate the LLL target
  m_LLLTarget.Init(width, height);

  // Done
  return S_OK;
}

//--------------------------------------------------------------------------------------
// This function is where the real work is done. We determine the matrices and constants used in 
// shadow generation and scene generation.
//--------------------------------------------------------------------------------------
HRESULT SceneManager::InitFrame ( CFirstPersonCamera*     pViewerCamera,
                                  CFirstPersonCamera*     pLightCamera  ) 
{   
  m_pViewerCamera                  = pViewerCamera;
  m_pLightCamera                   = pLightCamera ;
  
  XMMATRIX matViewCameraProjection = pViewerCamera->GetProjMatrix();
  XMMATRIX matViewCameraView       = pViewerCamera->GetViewMatrix();
  XMMATRIX matLightCameraView      = pLightCamera->GetViewMatrix();
  
  XMMATRIX matInverseViewCamera    = XMMatrixInverse(nullptr,  matViewCameraView );
  
  m_matShadowView = matLightCameraView;
     
  m_GPULightEnvAlloc.BeginFrame( m_pd3dDeviceContext );
  m_DynamicVB.BeginFrame(m_pd3dDeviceContext); 

  // Reset the scratch memory
  m_ScratchOffset                  = 0;
      
  // Convert from min max representation to center extents representation.
  // This will make it easier to pull the points out of the transformation.
  XMVECTOR vSceneCenter   = m_vSceneAABBMin + m_vSceneAABBMax;
           vSceneCenter  *= g_vHalfVector;
  XMVECTOR vSceneExtents  = m_vSceneAABBMax - m_vSceneAABBMin;
           vSceneExtents *= g_vHalfVector;    
  
  XMVECTOR vSceneAABBPointsLightSpace[8];

  // Convert the center and extents of an AABB into 8 points
  {
    const XMVECTORF32 vExtentsMap[] = 
    { 
      {1.0f, 1.0f, -1.0f, 1.0f}, 
      {-1.0f, 1.0f, -1.0f, 1.0f}, 
      {1.0f, -1.0f, -1.0f, 1.0f}, 
      {-1.0f, -1.0f, -1.0f, 1.0f}, 
      {1.0f, 1.0f, 1.0f, 1.0f}, 
      {-1.0f, 1.0f, 1.0f, 1.0f}, 
      {1.0f, -1.0f, 1.0f, 1.0f}, 
      {-1.0f, -1.0f, 1.0f, 1.0f} 
    };

    for( INT index = 0; index < 8; ++index ) 
    {
      vSceneAABBPointsLightSpace[index] = XMVectorMultiplyAdd(vExtentsMap[index], vSceneExtents, vSceneCenter ); 
    }
  }
   
  // Transform the scene AABB to Light space.
  for( int index =0; index < 8; ++index ) 
  {
      vSceneAABBPointsLightSpace[index] = XMVector4Transform( vSceneAABBPointsLightSpace[index], matLightCameraView ); 
  }
  
  FLOAT fFrustumIntervalBegin, fFrustumIntervalEnd;
  XMVECTOR vLightCameraOrthographicMin;  // light space frustum aabb 
  XMVECTOR vLightCameraOrthographicMax;
  FLOAT fCameraNearFarRange = m_pViewerCamera->GetFarClip() - m_pViewerCamera->GetNearClip();
     
  XMVECTOR vWorldUnitsPerTexel = g_vZero; 
  
  int m_iCascadePartitionsZeroToOne[8];
  int m_iCascadePartitionsMax = 100;
  
  m_iCascadePartitionsZeroToOne[0] = 5;
  m_iCascadePartitionsZeroToOne[1] = 15;
  m_iCascadePartitionsZeroToOne[2] = 60;
  m_iCascadePartitionsZeroToOne[3] = 100;
  m_iCascadePartitionsZeroToOne[4] = 100;
  m_iCascadePartitionsZeroToOne[5] = 100;
  m_iCascadePartitionsZeroToOne[6] = 100;
  m_iCascadePartitionsZeroToOne[7] = 100;
  
  fFrustumIntervalBegin            = 0.0f;

  // We loop over the cascades to calculate the orthographic projection for each cascade.
  for( INT iCascadeIndex=0; iCascadeIndex < CASCADE_COUNT_FLAG; ++iCascadeIndex ) 
  { 
    // Scale the intervals between 0 and 1. They are now percentages that we can scale with.
    fFrustumIntervalEnd    = (FLOAT)m_iCascadePartitionsZeroToOne[ iCascadeIndex ];        
    fFrustumIntervalBegin /= (FLOAT)m_iCascadePartitionsMax;
    fFrustumIntervalEnd   /= (FLOAT)m_iCascadePartitionsMax;
    fFrustumIntervalBegin  = fFrustumIntervalBegin * fCameraNearFarRange;
    fFrustumIntervalEnd    = fFrustumIntervalEnd * fCameraNearFarRange;
    XMVECTOR vFrustumPoints[8];
    
    // This function takes the began and end intervals along with the projection matrix and returns the 8
    // points that represent the cascade Interval
    CreateFrustumPointsFromCascadeInterval( fFrustumIntervalBegin, fFrustumIntervalEnd, 
                                            matViewCameraProjection, vFrustumPoints );
    
    vLightCameraOrthographicMin = g_vFLTMAX;
    vLightCameraOrthographicMax = g_vFLTMIN;
    
    XMVECTOR vTempTranslatedCornerPoint;
    // This next section of code calculates the min and max values for the orthographic projection.
    for( int icpIndex=0; icpIndex < 8; ++icpIndex ) 
    {
      // Transform the frustum from camera view space to world space.
      vFrustumPoints[icpIndex]    = XMVector4Transform ( vFrustumPoints[icpIndex], matInverseViewCamera );
      // Transform the point from world space to Light Camera Space.
      vTempTranslatedCornerPoint  = XMVector4Transform ( vFrustumPoints[icpIndex], matLightCameraView );
      // Find the closest point.
      vLightCameraOrthographicMin = XMVectorMin ( vTempTranslatedCornerPoint, vLightCameraOrthographicMin );
      vLightCameraOrthographicMax = XMVectorMax ( vTempTranslatedCornerPoint, vLightCameraOrthographicMax );
    }
    
    // This code removes the shimmering effect along the edges of shadows due to
    // the light changing to fit the camera.
    {
      // Fit the ortho projection to the cascades far plane and a near plane of zero. 
      // Pad the projection to be the size of the diagonal of the Frustum partition. 
      // 
      // To do this, we pad the ortho transform so that it is always big enough to cover 
      // the entire camera view frustum.
      XMVECTOR vDiagonal = vFrustumPoints[0] - vFrustumPoints[6];
      vDiagonal = XMVector3Length( vDiagonal );
      
      // The bound is the length of the diagonal of the frustum interval.
      FLOAT fCascadeBound = XMVectorGetX( vDiagonal );
      
      // The offset calculated will pad the ortho projection so that it is always the same size 
      // and big enough to cover the entire cascade interval.
      XMVECTOR vBoarderOffset = ( vDiagonal - 
                                  ( vLightCameraOrthographicMax - vLightCameraOrthographicMin ) ) 
                                  * g_vHalfVector;
      // Set the Z and W components to zero.
      vBoarderOffset *= g_vMultiplySetzwToZero;
      
      // Add the offsets to the projection.
      vLightCameraOrthographicMax += vBoarderOffset;
      vLightCameraOrthographicMin -= vBoarderOffset;
      
      // The world units per texel are used to snap the shadow the orthographic projection
      // to texel sized increments.  This keeps the edges of the shadows from shimmering.
      FLOAT fWorldUnitsPerTexel = fCascadeBound / (float)CASCADE_BUFFER_SIZE;
      vWorldUnitsPerTexel       = XMVectorSet( fWorldUnitsPerTexel, fWorldUnitsPerTexel, 0.0f, 0.0f ); 
    } 
    
    float fLightCameraOrthographicMinZ = XMVectorGetZ( vLightCameraOrthographicMin );
    
    {    
      // We snape the camera to 1 pixel increments so that moving the camera does not cause the shadows to jitter.
      // This is a matter of integer dividing by the world space size of a texel
      vLightCameraOrthographicMin /= vWorldUnitsPerTexel;
      vLightCameraOrthographicMin = XMVectorFloor( vLightCameraOrthographicMin );
      vLightCameraOrthographicMin *= vWorldUnitsPerTexel;

      vLightCameraOrthographicMax /= vWorldUnitsPerTexel;
      vLightCameraOrthographicMax = XMVectorFloor( vLightCameraOrthographicMax );
      vLightCameraOrthographicMax *= vWorldUnitsPerTexel;   
    }
    
    FLOAT fNearPlane = 0.0f;
    FLOAT fFarPlane  = 10000.0f;
    
    // By intersecting the light frustum with the scene AABB we can get a tighter bound on the near and far plane.
    ComputeNearAndFar( fNearPlane, fFarPlane, vLightCameraOrthographicMin, 
                       vLightCameraOrthographicMax, vSceneAABBPointsLightSpace );
    
    // Create the orthographic projection for this cascade. 
    m_matShadowProj[ iCascadeIndex ] = XMMatrixOrthographicOffCenterLH(XMVectorGetX( vLightCameraOrthographicMin ), 
                                                                       XMVectorGetX( vLightCameraOrthographicMax ), 
                                                                       XMVectorGetY( vLightCameraOrthographicMin ), 
                                                                       XMVectorGetY( vLightCameraOrthographicMax ), 
                                                                       fNearPlane, fFarPlane );
    
    m_fCascadePartitionsFrustum[ iCascadeIndex ] = fFrustumIntervalEnd;
  }
  
  // Set the viewport
  SetViewport(pViewerCamera, &m_MainVP);
  
  // Done
  return S_OK;
}


//--------------------------------------------------------------------------------------
// Render the cascades into a texture atlas.
//--------------------------------------------------------------------------------------
HRESULT SceneManager::RenderShadowCascades( CDXUTSDKMesh* pMesh)
{
  HRESULT                 hr = S_OK;
  ID3D11RenderTargetView* pnullView = nullptr; 
  
  m_pd3dDeviceContext->ClearDepthStencilView( m_CascadedShadowMapRT.GetDSView(), D3D11_CLEAR_DEPTH, 1.0, 0 );
  
  // Set a null render target so as not to render color.
  m_pd3dDeviceContext->OMSetRenderTargets( 1, &pnullView , m_CascadedShadowMapRT.GetDSView() );
  
  m_pd3dDeviceContext->RSSetState( m_prsShadow );
  
  // Set the depth stencil state
  m_pd3dDeviceContext->OMSetDepthStencilState(m_pdsDefault, 0xFF);
  
  // Iterate over cascades and render shadows.
  for( INT currentCascade=0; currentCascade < CASCADE_COUNT_FLAG; ++currentCascade ) 
  {
    // Each cascade has its own viewport because we're storing all the cascades in one large texture.
    m_pd3dDeviceContext->RSSetViewports( 1, &m_RenderVP[currentCascade] );
    
    // We calculate the matrices in the Init function.
    XMMATRIX matWorldViewProjection = m_matShadowView * m_matShadowProj[currentCascade];
    
    D3D11_MAPPED_SUBRESOURCE MappedResource;
    V( m_pd3dDeviceContext->Map( m_pcbSimpleCB, 0, D3D11_MAP_WRITE_DISCARD, 0, &MappedResource ) );
    auto simple_cb = reinterpret_cast<SimpleCB*>( MappedResource.pData );
    
    XMStoreFloat4x4( &simple_cb->g_SimpleWorldViewProj, matWorldViewProjection );
    m_pd3dDeviceContext->Unmap( m_pcbSimpleCB, 0 );
    
    m_pd3dDeviceContext->IASetInputLayout( m_pVertexLayoutMesh );
    
    // No pixel shader is bound as we're only writing out depth.
    m_pd3dDeviceContext->VSSetShader( m_pvsRenderSimple.m_Shader, nullptr, 0 );
    m_pd3dDeviceContext->PSSetShader( nullptr, nullptr, 0 ); 
    
    m_pd3dDeviceContext->VSSetConstantBuffers( CB_SIMPLE, 1, &m_pcbSimpleCB );
    
    pMesh->Render( m_pd3dDeviceContext, 0, 1 );
  }
  
  m_pd3dDeviceContext->RSSetState( nullptr );
  m_pd3dDeviceContext->OMSetRenderTargets( 1, &pnullView, nullptr );
  
  return hr;
}

//--------------------------------------------------------------------------------------
HRESULT SceneManager::EndFrame(ID3D11RenderTargetView* prtvBackBuffer) 
{ 
  // Make sure the RT is set to the back buffer
  m_pd3dDeviceContext->OMSetRenderTargets( 1, &prtvBackBuffer, nullptr );

  // Check if we are not debugging anything
  if(m_DebugRendering == DEBUG_RENDERING_NONE)
  {
    return S_OK;
  } 
  
  float width    = float(m_GBufferRT.GetWidth() );
  float height   = float(m_GBufferRT.GetHeight());

  m_pd3dDeviceContext->RSSetState( m_prsCullNone );
  m_pd3dDeviceContext->OMSetDepthStencilState( m_pdsNoWrite, 0xFF);
  m_pd3dDeviceContext->OMSetBlendState( m_pbsNone, g_BlendFactors, 0xFFFFFFFF);
  m_pd3dDeviceContext->VSSetShader( m_pvsRender2D.m_Shader, nullptr, 0 );
  
  if(m_DebugRendering == DEBUG_RENDERING_LLL)
  {
    m_pd3dDeviceContext->PSSetShader( m_ppsDebugLight.m_Shader,  nullptr, 0 ); DrawQuad(0, 0, width, height);
    return S_OK;
  }

  m_pd3dDeviceContext->PSSetShader( m_ppsTexture.m_Shader,     nullptr, 0 );
  if(m_DebugRendering == DEBUG_RENDERING_NORMALS){ SetTexture(TEX_DIFFUSE, m_GBufferRT.GetColorTexture(0)); DrawQuad(0, 0, width, height); }
  if(m_DebugRendering == DEBUG_RENDERING_COLORS ){ SetTexture(TEX_DIFFUSE, m_GBufferRT.GetColorTexture(1)); DrawQuad(0, 0, width, height); }

  ClearTexture(0);
 
  // Done
  return S_OK;
}

//--------------------------------------------------------------------------------------
// Render the scene to the GBuffer.
//--------------------------------------------------------------------------------------
HRESULT SceneManager::RenderGBuffer( CDXUTSDKMesh* pMesh) 
{
  ID3D11DepthStencilView*  pdsv           = m_GBufferRT.GetDSView(); 
  ID3D11RenderTargetView*  pnrmv          = m_GBufferRT.GetRTView(0); 
  ID3D11RenderTargetView*  pcolv          = m_GBufferRT.GetRTView(1); 

  float                    clr_color[]    = { 0, 0, 0, 0};

  // Clear color
  m_pd3dDeviceContext->ClearRenderTargetView( pnrmv, clr_color );
  m_pd3dDeviceContext->ClearRenderTargetView( pcolv, clr_color );

  // Clear depth stencil
  m_pd3dDeviceContext->ClearDepthStencilView( pdsv, D3D11_CLEAR_DEPTH, 1.0, 0 );
    
  m_pd3dDeviceContext->OMSetRenderTargets( 2, m_GBufferRT.GetRTViews(), pdsv );
  m_pd3dDeviceContext->OMSetBlendState( m_pbsNone, g_BlendFactors, 0xFFFFFFFF);
  m_pd3dDeviceContext->OMSetDepthStencilState(m_pdsDefault, 0xFF);
   
  // Transforms
  {
    D3D11_MAPPED_SUBRESOURCE MappedResource;
    XMMATRIX                 matCameraProj          = m_pViewerCamera->GetProjMatrix();
    XMMATRIX                 matCameraView          = m_pViewerCamera->GetViewMatrix();
    XMMATRIX                 matWorldViewProjection = matCameraView * matCameraProj;

    m_pd3dDeviceContext->Map( m_pcbSimpleCB, 0, D3D11_MAP_WRITE_DISCARD, 0, &MappedResource );
    auto simple_cb = reinterpret_cast<SimpleCB*>( MappedResource.pData );
  
    XMStoreFloat4x4( &simple_cb->g_SimpleWorldViewProj, matWorldViewProjection );
    XMStoreFloat4x4( &simple_cb->g_SimpleWorld,         XMMatrixIdentity()     ); 

    m_pd3dDeviceContext->Unmap( m_pcbSimpleCB, 0 );
    m_pd3dDeviceContext->VSSetConstantBuffers( CB_SIMPLE, 1, &m_pcbSimpleCB );
  }

  SetViewport(m_pViewerCamera, &m_MainVP);
  m_pd3dDeviceContext->RSSetState( m_prsCullBackFaces );

  m_pd3dDeviceContext->IASetInputLayout( m_pVertexLayoutMesh );
 
  m_pd3dDeviceContext->VSSetShader( m_pvsRenderScene.m_Shader, nullptr, 0 );
  m_pd3dDeviceContext->PSSetShader( m_ppsGBuffer.m_Shader,     nullptr, 0 );

  SetSampler( SAM_LINEAR, m_pSamLinear );
  SetSampler( SAM_POINT,  m_pSamPoint  );

  pMesh->Render( m_pd3dDeviceContext, 0, 1 );

  return S_OK;
}

//--------------------------------------------------------------------------------------------------
inline float Lerpf(float a, float b, float i)
{
  return a * (1.f - i) + b * i; 
}

//--------------------------------------------------------------------------------------
inline float RandFloatNormalized()
{
  return static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
}

//--------------------------------------------------------------------------------------
// Clear and fill the Light Linked List
//--------------------------------------------------------------------------------------
HRESULT SceneManager::ProcessLinkedList()
{
  // Remove resources
  ClearTextures(6, 3);

  // Unbind the previously bound UAVs
  ClearResourceView( SRV_LIGHT_LINKED );
  ClearResourceView( SRV_LIGHT_OFFSET );

  // Declare the UAVs
  const void*     pUAVs[]      = { m_LLLTarget.GetFragmentLinkUAV(), m_LLLTarget.GetStartOffsetUAV(), m_LLLTarget.GetBoundsUAV() };

  D3D11_VIEWPORT  lll_vp       = { 0, 0, (float)m_LLLTarget.GetWidth(), (float)m_LLLTarget.GetHeight(), 0, 1};

  // Set the viewport
  SetViewport(m_pViewerCamera, &lll_vp);
   
  m_pd3dDeviceContext->RSSetState( m_prsCullNone );

  // Set the depth stencil state
  m_pd3dDeviceContext->OMSetDepthStencilState(m_pdsDefault, 0xFF);

  // Clear the LLL and down-sample the depth
  {
    // Start over
    uint32_t init_indices[] = { 0, 0, 0, 0 };

    m_pd3dDeviceContext->OMSetRenderTargetsAndUnorderedAccessViews( 1,
                                                                    m_LLLTarget.GetLinearDepthTarget()->GetRTViews(),
                                                                    nullptr,
                                                                    UAV_LIGHT_LINKED, 
                                                                    3, 
                                                                    (ID3D11UnorderedAccessView**)pUAVs,
                                                                    init_indices ); 

    // Disable Blending
    m_pd3dDeviceContext->OMSetBlendState( m_pbsNone, g_BlendFactors, 0xFFFFFFFF);

    // Set point sampler                                                                
    SetSampler( SAM_POINT, m_pSamPoint  );

    // Set the full resolution hyper depth
    SetTexture(TEX_DEPTH, m_GBufferRT.GetDepthTexture() );

    m_pd3dDeviceContext->VSSetShader( m_pvsRender2D.m_Shader,  nullptr, 0 );
    m_pd3dDeviceContext->PSSetShader( m_ppsClearLLL.m_Shader,  nullptr, 0 );

    DrawQuad(0, 0, lll_vp.Width, lll_vp.Height);

    // Clear the texture
    ClearTexture( TEX_DEPTH );
  }

  // Fill the light linked list
  if(m_DynamicLights)
  {
    XMMATRIX        matCameraProj     = m_pViewerCamera->GetProjMatrix();
    XMMATRIX        matCameraView     = m_pViewerCamera->GetViewMatrix();
    XMMATRIX        matViewProjection = matCameraView * matCameraProj;
  
    BoundingFrustum fproj( matCameraProj );
    BoundingFrustum frustum;
    XMVECTOR        xm_nplane;
    XMFLOAT4A       near_plane;

    // Transform the frustum into world space
    fproj.Transform(frustum, m_pViewerCamera->GetWorldMatrix() );

    // Fetch the near plane
    frustum.GetPlanes(&xm_nplane, nullptr, nullptr, nullptr, nullptr, nullptr);
    XMStoreFloat4A(&near_plane, xm_nplane);

    // Unique seed to guarantee same values over and over again
    srand(0xF7328EAC);

    float fade_interval = 8;
    float fade_distance = m_pViewerCamera->GetFarClip() - 48;
    float spacing       = 12.5f;
    float base_radius   = 5.0f;
    int   light_count   = 0; 

    float x_bound                = XMVectorGetX(m_vSceneAABBMin);
    float z_bound                = XMVectorGetZ(m_vSceneAABBMin);
     
    GPULightEnv* gpu_lights      = m_GPULightEnvAlloc.Allocate( MAX_LLL_LIGHTS );
    GPULightEnv* scratch_lights  = (GPULightEnv*)ScratchAlloc(sizeof(GPULightEnv) * MAX_LLL_LIGHTS); 
                                 
    GPULightEnv* clipped_light   = scratch_lights; 
    GPULightEnv* unclipped_light = gpu_lights; 
    
    // Create the lights environments
    for(float x = XMVectorGetX(m_vSceneAABBMax) - base_radius; x > x_bound; x -= spacing)
    {
      for(float z = XMVectorGetZ(m_vSceneAABBMax) - base_radius; (z > z_bound) && (light_count < MAX_LLL_LIGHTS); z -= spacing)
      {
        float     id      = RandFloatNormalized();
        float     radius  = Lerpf(base_radius, base_radius* 2, id);
        float3    center; 
                          
        center.x          = x      + Lerpf(-base_radius, base_radius, id);
        center.y          = radius * Lerpf(0.25f, 0.5f, id);
        center.z          = z      + Lerpf(-radius, radius, id);
                          
        float     r       = Lerpf(0.1, 2, RandFloatNormalized());
        float     g       = Lerpf(0.1, 2, RandFloatNormalized());
        float     b       = Lerpf(0.1, 2, RandFloatNormalized());

        // Do the frustum test
        if( frustum.Intersects(BoundingSphere(center, radius)) == true)
        {          
          XMVECTOR     center_to_cam = XMVectorSubtract(XMLoadFloat3(&center), m_pViewerCamera->GetEyePt());
          XMVECTOR     distance_vec  = XMVector3Length(center_to_cam);
          float        distance      = XMVectorGetX(distance_vec) + radius;
          float        fade          = std::min((fade_distance - distance)/fade_interval, 1.0f); 
 
          // Validate he fade amount
          if(fade > 0.0f)
          {
            // Slight radius bump to account for over-sized shell
            bool          near_clip_intersect   = (near_plane.x * center.x) + (near_plane.y * center.y) + (near_plane.z * center.z) + near_plane.w >= (-radius * 1.05f);
            GPULightEnv*  light_env;
             
            // Check if we are intersecting the near clip
            if (near_clip_intersect)
            {
              light_env = clipped_light++; 
            } 
            else
            {
              light_env = unclipped_light++; 
            }

            // Fill the light attributes
            light_env->m_WorldPos      = center;
            light_env->m_Radius        = radius;

            light_env->m_LinearColor   = float3(r * fade, g * fade, b * fade);
            light_env->m_SpecIntensity = 1.0f; 

            ++light_count;
          }
        }
      }
    }

    //Light count
    int   unclipped_count = int(unclipped_light - gpu_lights    );
    int   clipped_count   = int(clipped_light   - scratch_lights);
    float light_idx   = 0;

    // Copy the clipped lights into the GPU memory
    memcpy(unclipped_light,  scratch_lights, clipped_count * sizeof(GPULightEnv) );

    // Unbind the linear depth from the output
    m_pd3dDeviceContext->OMSetRenderTargetsAndUnorderedAccessViews( 0,
                                                                    nullptr,
                                                                    nullptr,
                                                                    UAV_LIGHT_LINKED, 
                                                                    3, 
                                                                    (ID3D11UnorderedAccessView**)pUAVs,
                                                                    nullptr); 
    // Disable RGBA
    m_pd3dDeviceContext->OMSetBlendState( m_pbsDisableRGBA, g_BlendFactors, 0xFFFFFFFF);

    // Set the depth stencil state
    m_pd3dDeviceContext->OMSetDepthStencilState(m_pdsDefault, 0xFF);

    m_pd3dDeviceContext->IASetInputLayout( m_pVertexLayoutLight );
    m_pd3dDeviceContext->IASetPrimitiveTopology( D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST );
     
    m_pd3dDeviceContext->VSSetShader( m_pvsRenderLight.m_Shader,  nullptr, 0 );

    // Set the linear depth texture
    SetTexture(TEX_DEPTH, m_LLLTarget.GetLinearDepthTexture() );

    // Disable culling
    m_pd3dDeviceContext->RSSetState( m_prsCullNone ); 
    m_pd3dDeviceContext->PSSetShader( m_ppsInsertLightNoCulling.m_Shader,  nullptr, 0 ); 
 
    // Render the unclipped lights via one instanced drawcall
    if(unclipped_count != 0)
    {
      D3D11_MAPPED_SUBRESOURCE MappedResource;
      m_pd3dDeviceContext->Map( m_pcbLightInstancesCB, 0, D3D11_MAP_WRITE_DISCARD, 0, &MappedResource );
      auto inst_cb = reinterpret_cast<LightInstancesCB*>( MappedResource.pData );

      for(int l_idx = 0; l_idx < unclipped_count; ++l_idx, ++light_idx)
      {
        int             int_idx    = int(light_idx);
        GPULightEnv*    light_env  = gpu_lights                + int_idx;
        LightInstance*  light_inst = inst_cb->m_LightInstances + int_idx;

        // Fill the instance information
        { 
          XMMATRIX light_trans = XMMatrixTranslation(light_env->m_WorldPos.x, light_env->m_WorldPos.y, light_env->m_WorldPos.z);
          XMStoreFloat4x4( &light_inst->m_WorldViewProj, light_trans * matViewProjection );
          light_inst->m_LightIndex  = light_idx;
          light_inst->m_Radius      = light_env->m_Radius;
        } 
      }        
      
      m_pd3dDeviceContext->Unmap( m_pcbLightInstancesCB, 0 );
      m_pd3dDeviceContext->VSSetConstantBuffers( CB_LIGHT_INSTANCES, 1, &m_pcbLightInstancesCB );

      // Draw all the instances in one call
      DrawEllipsoidLightShells(unclipped_count);
    } 
     
    // Check if we have any clipped geometry
    if(clipped_count)
    {
      D3D11_MAPPED_SUBRESOURCE MappedResource;

      // Switch pixel shaders
      m_pd3dDeviceContext->PSSetShader( m_ppsInsertLightBackFace.m_Shader,  nullptr, 0 ); 

      // Render the clipped lights each with two drawcalls
      for(int l_idx = 0; l_idx < clipped_count; ++l_idx, ++light_idx)
      {
        GPULightEnv* light_env = gpu_lights + int(light_idx);
               
        m_pd3dDeviceContext->Map( m_pcbSimpleCB, 0, D3D11_MAP_WRITE_DISCARD, 0, &MappedResource );
        auto simple_cb = reinterpret_cast<SimpleCB*>( MappedResource.pData );
  
        {  
          XMMATRIX light_trans = XMMatrixTranslation(light_env->m_WorldPos.x, light_env->m_WorldPos.y, light_env->m_WorldPos.z);
          XMStoreFloat4x4( &simple_cb->g_SimpleWorldViewProj, light_trans * matViewProjection );
          simple_cb->g_SimpleLightIndex  = light_idx; 
          simple_cb->g_SimpleRadius      = light_env->m_Radius;
        }

        m_pd3dDeviceContext->Unmap( m_pcbSimpleCB, 0 );
        m_pd3dDeviceContext->VSSetConstantBuffers( CB_LIGHT_INSTANCES, 1, &m_pcbSimpleCB );

        // Render the front faces first
        m_pd3dDeviceContext->RSSetState( m_prsCullBackFaces );
        DrawEllipsoidLightShells(1); 

        // Render the back faces last
        m_pd3dDeviceContext->RSSetState( m_prsCullFrontFaces ); 
        ReDrawEllipsoidLightShell();
      }
    } 
  }

  // Clear the UAVs and bind them as SRVs
  {
    const void*     pUAVs[]      = { nullptr, nullptr, nullptr};
    m_pd3dDeviceContext->OMSetRenderTargetsAndUnorderedAccessViews( 0,
                                                                    nullptr,
                                                                    nullptr,
                                                                    UAV_LIGHT_LINKED, 
                                                                    3, 
                                                                    (ID3D11UnorderedAccessView**)pUAVs,
                                                                    nullptr ); 

    SetResourceView( SRV_LIGHT_LINKED, m_LLLTarget.GetFragmentLinkSRV() );
    SetResourceView( SRV_LIGHT_OFFSET, m_LLLTarget.GetStartOffsetSRV()  );
  }

  // Bind the global light environments
  SetResourceView( SRV_LIGHT_ENV, m_GPULightEnvAlloc.GetSRV()  );

  // Clear the linear depth texture
  ClearTexture( TEX_DEPTH );

  // Done
  return S_OK;
}

//--------------------------------------------------------------------------------------
// Composite the scene.
//--------------------------------------------------------------------------------------
HRESULT SceneManager::CompositeScene(ID3D11RenderTargetView* prtvBackBuffer)
{
  HRESULT                  hr             = S_OK;
  
  // Fill the constant buffer
  {
    D3D11_MAPPED_SUBRESOURCE MappedResource;
  
    V( m_pd3dDeviceContext->Map( m_pcbShadowCB, 0, D3D11_MAP_WRITE_DISCARD, 0, &MappedResource ) );
    auto shadow_data = reinterpret_cast<ShadowDataCB*>( MappedResource.pData );
  
    // These are the for loop begin end values. 
    shadow_data->m_iPCFBlurForLoopEnd   = m_iPCFBlurSize / 2 +1;
    shadow_data->m_iPCFBlurForLoopStart = m_iPCFBlurSize / -2;
  
    // This is a floating point number that is used as the percentage to blur between maps.    
    shadow_data->m_fCascadeBlendArea    = m_fBlurBetweenCascadesAmount;
    shadow_data->m_fTexelSize           = 1.0f / CASCADE_BUFFER_SIZE; 
    shadow_data->m_fNativeTexelSizeInX  = shadow_data->m_fTexelSize / CASCADE_COUNT_FLAG;
  
    XMMATRIX matTextureScale            = XMMatrixScaling(  0.5f, -0.5f, 1.0f );
                                      
    XMMATRIX matTextureTranslation      = XMMatrixTranslation( .5f, .5f, 0.f );
    XMMATRIX scaleToTile                = XMMatrixScaling( 1.0f / CASCADE_COUNT_FLAG, 1.0, 1.0 );
  
    shadow_data->m_fShadowBiasFromGUI   = m_fPCFOffset;
    shadow_data->m_fShadowPartitionSize = 1.0f / CASCADE_COUNT_FLAG;
  
    XMStoreFloat4x4( &shadow_data->m_mShadow, m_matShadowView );
    for(int index=0; index < CASCADE_COUNT_FLAG; ++index ) 
    {
      XMMATRIX mShadowTexture = m_matShadowProj[index] * matTextureScale * matTextureTranslation;
      shadow_data->m_vCascadeScale[index].x = XMVectorGetX( mShadowTexture.r[0] );
      shadow_data->m_vCascadeScale[index].y = XMVectorGetY( mShadowTexture.r[1] );
      shadow_data->m_vCascadeScale[index].z = XMVectorGetZ( mShadowTexture.r[2] );
      shadow_data->m_vCascadeScale[index].w = 1;
      
      XMStoreFloat3( reinterpret_cast<XMFLOAT3*>( &shadow_data->m_vCascadeOffset[index] ), mShadowTexture.r[3] );
      shadow_data->m_vCascadeOffset[index].w = 0;
    }
  
    // The border padding values keep the pixel shader from reading the borders during PCF filtering.
    shadow_data->m_fMaxBorderPadding = (float)( CASCADE_BUFFER_SIZE  - 1.0f ) / CASCADE_BUFFER_SIZE;
    shadow_data->m_fMinBorderPadding = (float)( 1.0f ) / CASCADE_BUFFER_SIZE;
  
    XMVECTOR ep = m_pLightCamera->GetEyePt();
    XMVECTOR lp = m_pLightCamera->GetLookAtPt();
    ep         -= lp;
    ep          = XMVector3Normalize( ep );
  
    XMStoreFloat3( reinterpret_cast<XMFLOAT3*>( &shadow_data->m_vLightDir ), ep );
    shadow_data->m_nCascadeLevels     = CASCADE_COUNT_FLAG;
    shadow_data->m_iVisualizeCascades = false;
    m_pd3dDeviceContext->Unmap( m_pcbShadowCB, 0 );
  }
  
  // Set the constant buffer 
  m_pd3dDeviceContext->PSSetConstantBuffers( CB_SHADOW_DATA, 1, &m_pcbShadowCB);
  
  // Set the viewport
  SetViewport(m_pViewerCamera, &m_MainVP);
  m_pd3dDeviceContext->RSSetState( m_prsCullNone ); 
  
  m_pd3dDeviceContext->OMSetDepthStencilState(m_pdsDefault, 0xFF);
  m_pd3dDeviceContext->OMSetRenderTargets( 1, &prtvBackBuffer, NULL );
  m_pd3dDeviceContext->OMSetBlendState( m_pbsNone, g_BlendFactors, 0xFFFFFFFF);
  
  //Set the shaders
  m_pd3dDeviceContext->VSSetShader( m_pvsRender2D.m_Shader,  nullptr, 0 );
  m_pd3dDeviceContext->PSSetShader( m_ppsComposite.m_Shader, nullptr, 0 );
  
  //Set the samplers
  SetSampler( SAM_LINEAR, m_pSamLinear   );
  SetSampler( SAM_POINT,  m_pSamPoint    );
  SetSampler( SAM_SHADOW, m_pSamShadowPCF);
  
  // Set the textures
  SetTexture(TEX_DEPTH,  m_GBufferRT.GetDepthTexture() );
  SetTexture(TEX_NRM,    m_GBufferRT.GetColorTexture(0));
  SetTexture(TEX_COL,    m_GBufferRT.GetColorTexture(1));
  SetTexture(TEX_SHADOW, m_CascadedShadowMapRT.GetDepthTexture());
  
  // Draw
  { 
    BoundingFrustum bounding_frustum;
    BoundingFrustum::CreateFromMatrix(bounding_frustum, m_pViewerCamera->GetProjMatrix());
  
    XMFLOAT3        corners[8];
    bounding_frustum.GetCorners( corners );
  
    uint32_t kFrsutumCornerLeftTop     = 4;
    uint32_t kFrsutumCornerLeftBottom  = 7;
    uint32_t kFrsutumCornerRightTop    = 5;
    uint32_t kFrsutumCornerRightBottom = 6;
  
    XMMATRIX matViewToWorld   = m_pViewerCamera->GetWorldMatrix();
    
    XMVECTOR worldVecTopLeft  = XMVector3TransformNormal(XMLoadFloat3(&corners[kFrsutumCornerLeftTop    ] ), matViewToWorld);
    XMVECTOR worldVecBotLeft  = XMVector3TransformNormal(XMLoadFloat3(&corners[kFrsutumCornerLeftBottom ] ), matViewToWorld);
    XMVECTOR worldVecTopRight = XMVector3TransformNormal(XMLoadFloat3(&corners[kFrsutumCornerRightTop   ] ), matViewToWorld);
    XMVECTOR worldVecBotRight = XMVector3TransformNormal(XMLoadFloat3(&corners[kFrsutumCornerRightBottom] ), matViewToWorld);
  
    m_pd3dDeviceContext->IASetInputLayout( m_pVertexLayoutMesh );
    m_pd3dDeviceContext->IASetPrimitiveTopology( D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP );
  
    SimpleVertex* verts = DynamicVertexAlloc<SimpleVertex>(4);
  
    float x0    = 0;
    float y0    = 0;
  
    float x1    = x0 + m_GBufferRT.GetWidth();
    float y1    = y0 + m_GBufferRT.GetHeight();
  
    float depth = 0.0;
  
    // Store the world vectors
    XMStoreFloat3((XMFLOAT3*)&verts[0].nx, worldVecTopLeft );
    XMStoreFloat3((XMFLOAT3*)&verts[1].nx, worldVecBotLeft );
    XMStoreFloat3((XMFLOAT3*)&verts[2].nx, worldVecTopRight);
    XMStoreFloat3((XMFLOAT3*)&verts[3].nx, worldVecBotRight);
  
    //tl
    verts[0].x = x0;
    verts[0].y = y0;
    verts[0].z = depth;
    verts[0].u = 0;
    verts[0].v = 0;
  
    //bl
    verts[1].x = x0;
    verts[1].y = y1;
    verts[1].z = depth;
    verts[1].u = 0;
    verts[1].v = 1;
  
  
    //tr
    verts[2].x = x1;
    verts[2].y = y0;
    verts[2].z = depth;
    verts[2].u = 1;
    verts[2].v = 0;
  
    //br
    verts[3].x = x1;
    verts[3].y = y1;
    verts[3].z = depth;
    verts[3].u = 1;
    verts[3].v = 1;
  
    // Draw
    DynamicVertexDrawEnd(4);
  }
  
  // Clear the textures
  ClearTextures(0, 6);
  
  m_pd3dDeviceContext->OMSetRenderTargets( 1, &prtvBackBuffer, m_GBufferRT.GetDSView() );

  // Done
  return hr;
}

//--------------------------------------------------------------------------------------
HRESULT SceneManager::DrawAlpha(CDXUTSDKMesh* pMesh)
{
  XMMATRIX        matCameraProj     = m_pViewerCamera->GetProjMatrix();
  XMMATRIX        matCameraView     = m_pViewerCamera->GetViewMatrix();
  XMMATRIX        matViewProjection = matCameraView * matCameraProj;

  BoundingFrustum fproj( matCameraProj );
  BoundingFrustum frustum;

  // Transform the frustum into world space
  fproj.Transform(frustum, m_pViewerCamera->GetWorldMatrix() );

  // Enable alpha blending but no depth write (testing is still on)
  m_pd3dDeviceContext->OMSetBlendState( m_pbsAlpha, g_BlendFactors, 0xFFFFFFFF);
  m_pd3dDeviceContext->OMSetDepthStencilState(m_pdsNoWrite, 0xFF);

  m_pd3dDeviceContext->RSSetState( m_prsCullBackFaces );

  m_pd3dDeviceContext->IASetInputLayout( m_pVertexLayoutMesh );

  m_pd3dDeviceContext->VSSetShader( m_pvsRenderScene.m_Shader,  nullptr, 0 );
  m_pd3dDeviceContext->PSSetShader( m_ppsLit3D.m_Shader,        nullptr, 0 );

  SetSampler( SAM_LINEAR, m_pSamLinear );
  SetSampler( SAM_POINT,  m_pSamPoint  );
 
  SetTexture(TEX_SHADOW, m_CascadedShadowMapRT.GetDepthTexture());

  // Unique seed to guarantee same values over and over again
  srand(0xF328E7AC);
   
  float spacing       = 20.0f;
  float base_scale    = 2.5f;
  float base_height   = base_scale * 1.5f;
   
  float x_bound       = XMVectorGetX(m_vSceneAABBMin);
  float z_bound       = XMVectorGetZ(m_vSceneAABBMin);

  D3D11_MAPPED_SUBRESOURCE MappedResource;
    
  // Create the lights environments
  for(float x = XMVectorGetX(m_vSceneAABBMax) - spacing; x > x_bound; x -= spacing)
  {
    for(float z = XMVectorGetZ(m_vSceneAABBMax) - spacing; (z > z_bound); z -= spacing)
    { 
      float     scale   = Lerpf(base_scale, base_scale* 2, RandFloatNormalized());
      float3    center; 
                        
      center.x               = x           + Lerpf(-base_scale, base_scale, RandFloatNormalized());
      center.y               = base_height + scale * Lerpf(-0.15f, 0.15f,   RandFloatNormalized());
      center.z               = z           + Lerpf( base_scale,-base_scale, RandFloatNormalized());
 
      XMMATRIX   scaling     = XMMatrixScaling(scale, scale, scale);
      XMMATRIX   rotation    = XMMatrixRotationRollPitchYaw(0, XM_2PI * RandFloatNormalized(), 0);
      XMMATRIX   translation = XMMatrixTranslation(center.x, center.y, center.z);
      XMMATRIX   matWorld    = scaling * rotation * translation;

      // Do the frustum test
      if( frustum.Intersects(BoundingSphere(center, scale *1.5f)) == true)
      {          
        m_pd3dDeviceContext->Map( m_pcbSimpleCB, 0, D3D11_MAP_WRITE_DISCARD, 0, &MappedResource );
        auto simple_cb = reinterpret_cast<SimpleCB*>( MappedResource.pData );

        XMStoreFloat4x4( &simple_cb->g_SimpleWorldViewProj, matWorld * matViewProjection );
        XMStoreFloat4x4( &simple_cb->g_SimpleWorld,         matWorld                     ); 

        m_pd3dDeviceContext->Unmap( m_pcbSimpleCB, 0 );
        m_pd3dDeviceContext->VSSetConstantBuffers( CB_SIMPLE, 1, &m_pcbSimpleCB );
        pMesh->Render( m_pd3dDeviceContext, 0, 1 );
      }
    }
  }

  // Disable blending
  m_pd3dDeviceContext->OMSetBlendState( m_pbsNone, g_BlendFactors, 0xFFFFFFFF);

  // Reset depth stencil state
  m_pd3dDeviceContext->OMSetDepthStencilState(m_pdsDefault, 0xFF);

  ClearTexture(TEX_SHADOW);

  // Done
  return S_OK;
}

//--------------------------------------------------------------------------------------
void SceneManager::SetViewport(CFirstPersonCamera* pViewerCamera, D3D11_VIEWPORT* vp)
{
  BoundingFrustum bounding_frustum;
  BoundingFrustum::CreateFromMatrix(bounding_frustum, m_pViewerCamera->GetProjMatrix());

  XMFLOAT3        corners[8];
  bounding_frustum.GetCorners( corners );

  uint32_t kFrsutumCornerLeftTop     = 4;
  uint32_t kFrsutumCornerRightBottom = 6;

  D3D11_MAPPED_SUBRESOURCE MappedResource;
  HRESULT                  hr = S_OK;

  V( m_pd3dDeviceContext->Map( m_pcbFrameCB, 0, D3D11_MAP_WRITE_DISCARD, 0, &MappedResource ) );
  auto pcbFrame = reinterpret_cast<FrameCB*>( MappedResource.pData );
   
  XMStoreFloat4x4(&pcbFrame->m_mViewToWorld, pViewerCamera->GetWorldMatrix());

  GetScaleOffset( vp->TopLeftX - g_PixelOffset, vp->Width  - g_PixelOffset,  0.f,  1.f, &pcbFrame->m_vViewportToScreenUVs.x, &pcbFrame->m_vViewportToScreenUVs.z );
  GetScaleOffset( vp->TopLeftY - g_PixelOffset, vp->Height - g_PixelOffset,  0.f,  1.f, &pcbFrame->m_vViewportToScreenUVs.y, &pcbFrame->m_vViewportToScreenUVs.w );

  GetScaleOffset( vp->TopLeftX - g_PixelOffset, vp->Width  - g_PixelOffset, -1.f,  1.f, &pcbFrame->m_vViewportToClip.x, &pcbFrame->m_vViewportToClip.z );
  GetScaleOffset( vp->TopLeftY - g_PixelOffset, vp->Height - g_PixelOffset,  1.f, -1.f, &pcbFrame->m_vViewportToClip.y, &pcbFrame->m_vViewportToClip.w );

  float         ss_corner_x       = corners[kFrsutumCornerLeftTop    ].x/corners[kFrsutumCornerLeftTop    ].z;
  float         ss_corner_y       = corners[kFrsutumCornerLeftTop    ].y/corners[kFrsutumCornerLeftTop    ].z;

  pcbFrame->m_vScreenToView.x     = ss_corner_x;
  pcbFrame->m_vScreenToView.y     = ss_corner_y;
                                  
  pcbFrame->m_vScreenToView.z     = corners[kFrsutumCornerRightBottom].x/corners[kFrsutumCornerRightBottom    ].z - ss_corner_x;
  pcbFrame->m_vScreenToView.w     = corners[kFrsutumCornerRightBottom].y/corners[kFrsutumCornerRightBottom    ].z - ss_corner_y;
                                  
  double                f_d       = pViewerCamera->GetFarClip();
  double                n_d       = pViewerCamera->GetNearClip();
  double                fac_a     = f_d/(f_d-n_d);
  double                fac_b     = n_d/(n_d-f_d);
  double                fac_c     = f_d * fac_b;
  float                 fac_d     = 0.0f;

  pcbFrame->m_vLinearDepthConsts  = XMFLOAT4((float)fac_a, -(float)fac_b, -(float)fac_c, fac_d);

  pcbFrame->m_iLLLWidth           = m_LLLTarget.GetWidth();
  pcbFrame->m_iLLLHeight          = m_LLLTarget.GetHeight();
  pcbFrame->m_iLLLMaxAlloc        = m_LLLTarget.GetMaxLinkedElements();
  pcbFrame->m_fLightPushScale     = m_pViewerCamera->GetAspectRatio() / (2.0f * m_LLLTarget.GetWidth());
  
  XMStoreFloat4(   &pcbFrame->m_vCameraPos,     pViewerCamera->GetEyePt());

  m_pd3dDeviceContext->Unmap( m_pcbFrameCB, 0 );

  m_pd3dDeviceContext->RSSetViewports(   1, vp);

  m_pd3dDeviceContext->VSSetConstantBuffers( CB_FRAME, 1, &m_pcbFrameCB );
  m_pd3dDeviceContext->PSSetConstantBuffers( CB_FRAME, 1, &m_pcbFrameCB );

  // Done
}

//--------------------------------------------------------------------------------------
void SceneManager::DrawQuad(float x, float y, float w, float h)
{
  m_pd3dDeviceContext->IASetInputLayout( m_pVertexLayoutMesh );
  m_pd3dDeviceContext->IASetPrimitiveTopology( D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP );

  SimpleVertex* verts = DynamicVertexAlloc<SimpleVertex>(4);

  float x0    = x;
  float y0    = y;
              
  float x1    = x0 + w;
  float y1    = y0 + h;

  float depth = 0.0;

  //tl
  verts[0].x = x0;
  verts[0].y = y0;
  verts[0].z = depth;
  verts[0].u = 0;
  verts[0].v = 0;

  //bl
  verts[1].x = x0;
  verts[1].y = y1;
  verts[1].z = depth;
  verts[1].u = 0;
  verts[1].v = 1;


  //tr
  verts[2].x = x1;
  verts[2].y = y0;
  verts[2].z = depth;
  verts[2].u = 1;
  verts[2].v = 0;
  
  //br
  verts[3].x = x1;
  verts[3].y = y1;
  verts[3].z = depth;
  verts[3].u = 1;
  verts[3].v = 1;

  // Draw
  DynamicVertexDrawEnd(4);
}

//--------------------------------------------------------------------------------------------------
void* SceneManager::DynamicVertexAllocVerts( uint32_t max_vertex_count, uint32_t vertex_stride )
{
  //DynamicVertexAllocVerts called twice without DynamicVertexDraw being called
  assert(m_DynamicVertexAllocCount == 0);

  m_DynamicVertexAllocCount = max_vertex_count;
  m_DynamicVertexStride     = vertex_stride;
  if (max_vertex_count)
  {
    m_DynamicVertexAlloc    = m_DynamicVB.Alloc( max_vertex_count * vertex_stride, m_DynamicVertexOffset );
    //"DynamicVertexAllocVerts failed"
    assert(m_DynamicVertexAlloc);
  }
  else
  {
    m_DynamicVertexAlloc  = NULL;
  }

  return m_DynamicVertexAlloc;
}

//--------------------------------------------------------------------------------------------------
void SceneManager::DynamicVertexRedraw()
{
  m_pd3dDeviceContext->Draw(m_DynamicVertexDrawnCount, 0);
}

//--------------------------------------------------------------------------------------------------
void SceneManager::DynamicVertexDrawEnd( uint32_t vertex_count )
{
  //DynamicVertexDrawEnd exceeded allocated vertex count
  assert(m_DynamicVertexAllocCount >= vertex_count);

  // Bind the dynamic vertex buffer
  m_pd3dDeviceContext->IASetVertexBuffers( 0, 1, m_DynamicVB.GetBuffer(), &m_DynamicVertexStride, &m_DynamicVertexOffset );

  // Draw the verts
  m_pd3dDeviceContext->Draw(vertex_count, 0);
  
  m_DynamicVertexDrawnCount = vertex_count;
  m_DynamicVertexAllocCount = 0;
  m_DynamicVertexAlloc      = NULL;
}

//--------------------------------------------------------------------------------------
// Used to compute an intersection of the orthographic projection and the Scene AABB
//--------------------------------------------------------------------------------------
struct Triangle 
{
    XMVECTOR pt[3];
    bool culled;
};


//--------------------------------------------------------------------------------------
// Computing an accurate near and far plane will decrease surface acne and Peter-panning.
// Surface acne is the term for erroneous self shadowing.  Peter-panning is the effect where
// shadows disappear near the base of an object.
// As offsets are generally used with PCF filtering due self shadowing issues, computing the
// correct near and far planes becomes even more important.
// This concept is not complicated, but the intersection code is.
//--------------------------------------------------------------------------------------
void SceneManager::ComputeNearAndFar( FLOAT& fNearPlane, 
                                      FLOAT& fFarPlane, 
                                      FXMVECTOR vLightCameraOrthographicMin, 
                                      FXMVECTOR vLightCameraOrthographicMax, 
                                      XMVECTOR* pvPointsInCameraView ) 
{
  // Initialize the near and far planes
  fNearPlane = FLT_MAX;
  fFarPlane = -FLT_MAX;

  Triangle triangleList[16];
  INT iTriangleCnt = 1;

  triangleList[0].pt[0] = pvPointsInCameraView[0];
  triangleList[0].pt[1] = pvPointsInCameraView[1];
  triangleList[0].pt[2] = pvPointsInCameraView[2];
  triangleList[0].culled = false;

  // These are the indices used to tesselate an AABB into a list of triangles.
  static const INT iAABBTriIndexes[] = 
  {
    0,1,2,  1,2,3,
    4,5,6,  5,6,7,
    0,2,4,  2,4,6,
    1,3,5,  3,5,7,
    0,1,4,  1,4,5,
    2,3,6,  3,6,7 
  };

  INT iPointPassesCollision[3];

  // At a high level: 
  // 1. Iterate over all 12 triangles of the AABB.  
  // 2. Clip the triangles against each plane. Create new triangles as needed.
  // 3. Find the min and max z values as the near and far plane.

  //This is easier because the triangles are in camera spacing making the collisions tests simple comparisions.

  float fLightCameraOrthographicMinX = XMVectorGetX( vLightCameraOrthographicMin );
  float fLightCameraOrthographicMaxX = XMVectorGetX( vLightCameraOrthographicMax ); 
  float fLightCameraOrthographicMinY = XMVectorGetY( vLightCameraOrthographicMin );
  float fLightCameraOrthographicMaxY = XMVectorGetY( vLightCameraOrthographicMax );

  for( INT AABBTriIter = 0; AABBTriIter < 12; ++AABBTriIter ) 
  {

    triangleList[0].pt[0] = pvPointsInCameraView[ iAABBTriIndexes[ AABBTriIter*3 + 0 ] ];
    triangleList[0].pt[1] = pvPointsInCameraView[ iAABBTriIndexes[ AABBTriIter*3 + 1 ] ];
    triangleList[0].pt[2] = pvPointsInCameraView[ iAABBTriIndexes[ AABBTriIter*3 + 2 ] ];
    iTriangleCnt = 1;
    triangleList[0].culled = FALSE;

    // Clip each invidual triangle against the 4 frustums.  When ever a triangle is clipped into new triangles, 
    //add them to the list.
    for( INT frustumPlaneIter = 0; frustumPlaneIter < 4; ++frustumPlaneIter ) 
    {

      FLOAT fEdge;
      INT iComponent;

      if( frustumPlaneIter == 0 ) 
      {
        fEdge = fLightCameraOrthographicMinX; // todo make float temp
        iComponent = 0;
      } 
      else if( frustumPlaneIter == 1 ) 
      {
        fEdge = fLightCameraOrthographicMaxX;
        iComponent = 0;
      } 
      else if( frustumPlaneIter == 2 ) 
      {
        fEdge = fLightCameraOrthographicMinY;
        iComponent = 1;
      } 
      else 
      {
        fEdge = fLightCameraOrthographicMaxY;
        iComponent = 1;
      }

      for( INT triIter=0; triIter < iTriangleCnt; ++triIter ) 
      {
        // We don't delete triangles, so we skip those that have been culled.
        if( !triangleList[triIter].culled ) 
        {
          INT iInsideVertCount = 0;
          XMVECTOR tempOrder;
          // Test against the correct frustum plane.
          // This could be written more compactly, but it would be harder to understand.

          if( frustumPlaneIter == 0 ) 
          {
            for( INT triPtIter=0; triPtIter < 3; ++triPtIter ) 
            {
              if( XMVectorGetX( triangleList[triIter].pt[triPtIter] ) >
                XMVectorGetX( vLightCameraOrthographicMin ) ) 
              { 
                iPointPassesCollision[triPtIter] = 1;
              }
              else 
              {
                iPointPassesCollision[triPtIter] = 0;
              }
              iInsideVertCount += iPointPassesCollision[triPtIter];
            }
          }
          else if( frustumPlaneIter == 1 ) 
          {
            for( INT triPtIter=0; triPtIter < 3; ++triPtIter ) 
            {
              if( XMVectorGetX( triangleList[triIter].pt[triPtIter] ) < 
                XMVectorGetX( vLightCameraOrthographicMax ) )
              {
                iPointPassesCollision[triPtIter] = 1;
              }
              else
              { 
                iPointPassesCollision[triPtIter] = 0;
              }
              iInsideVertCount += iPointPassesCollision[triPtIter];
            }
          }
          else if( frustumPlaneIter == 2 ) 
          {
            for( INT triPtIter=0; triPtIter < 3; ++triPtIter ) 
            {
              if( XMVectorGetY( triangleList[triIter].pt[triPtIter] ) > 
                XMVectorGetY( vLightCameraOrthographicMin ) ) 
              {
                iPointPassesCollision[triPtIter] = 1;
              }
              else 
              {
                iPointPassesCollision[triPtIter] = 0;
              }
              iInsideVertCount += iPointPassesCollision[triPtIter];
            }
          }
          else 
          {
            for( INT triPtIter=0; triPtIter < 3; ++triPtIter ) 
            {
              if( XMVectorGetY( triangleList[triIter].pt[triPtIter] ) < 
                XMVectorGetY( vLightCameraOrthographicMax ) ) 
              {
                iPointPassesCollision[triPtIter] = 1;
              }
              else 
              {
                iPointPassesCollision[triPtIter] = 0;
              }
              iInsideVertCount += iPointPassesCollision[triPtIter];
            }
          }

          // Move the points that pass the frustum test to the begining of the array.
          if( iPointPassesCollision[1] && !iPointPassesCollision[0] ) 
          {
            tempOrder =  triangleList[triIter].pt[0];   
            triangleList[triIter].pt[0] = triangleList[triIter].pt[1];
            triangleList[triIter].pt[1] = tempOrder;
            iPointPassesCollision[0] = TRUE;            
            iPointPassesCollision[1] = FALSE;            
          }
          if( iPointPassesCollision[2] && !iPointPassesCollision[1] ) 
          {
            tempOrder =  triangleList[triIter].pt[1];   
            triangleList[triIter].pt[1] = triangleList[triIter].pt[2];
            triangleList[triIter].pt[2] = tempOrder;
            iPointPassesCollision[1] = TRUE;            
            iPointPassesCollision[2] = FALSE;                        
          }
          if( iPointPassesCollision[1] && !iPointPassesCollision[0] ) 
          {
            tempOrder =  triangleList[triIter].pt[0];   
            triangleList[triIter].pt[0] = triangleList[triIter].pt[1];
            triangleList[triIter].pt[1] = tempOrder;
            iPointPassesCollision[0] = TRUE;            
            iPointPassesCollision[1] = FALSE;            
          }

          if( iInsideVertCount == 0 ) 
          { // All points failed. We're done,  
            triangleList[triIter].culled = true;
          }
          else if( iInsideVertCount == 1 ) 
          {// One point passed. Clip the triangle against the Frustum plane
            triangleList[triIter].culled = false;

            // 
            XMVECTOR vVert0ToVert1 = triangleList[triIter].pt[1] - triangleList[triIter].pt[0];
            XMVECTOR vVert0ToVert2 = triangleList[triIter].pt[2] - triangleList[triIter].pt[0];

            // Find the collision ratio.
            FLOAT fHitPointTimeRatio = fEdge - XMVectorGetByIndex( triangleList[triIter].pt[0], iComponent ) ;
            // Calculate the distance along the vector as ratio of the hit ratio to the component.
            FLOAT fDistanceAlongVector01 = fHitPointTimeRatio / XMVectorGetByIndex( vVert0ToVert1, iComponent );
            FLOAT fDistanceAlongVector02 = fHitPointTimeRatio / XMVectorGetByIndex( vVert0ToVert2, iComponent );
            // Add the point plus a percentage of the vector.
            vVert0ToVert1 *= fDistanceAlongVector01;
            vVert0ToVert1 += triangleList[triIter].pt[0];
            vVert0ToVert2 *= fDistanceAlongVector02;
            vVert0ToVert2 += triangleList[triIter].pt[0];

            triangleList[triIter].pt[1] = vVert0ToVert2;
            triangleList[triIter].pt[2] = vVert0ToVert1;

          }
          else if( iInsideVertCount == 2 ) 
          { // 2 in  // tesselate into 2 triangles


            // Copy the triangle\(if it exists) after the current triangle out of
            // the way so we can override it with the new triangle we're inserting.
            triangleList[iTriangleCnt] = triangleList[triIter+1];

            triangleList[triIter].culled = false;
            triangleList[triIter+1].culled = false;

            // Get the vector from the outside point into the 2 inside points.
            XMVECTOR vVert2ToVert0 = triangleList[triIter].pt[0] - triangleList[triIter].pt[2];
            XMVECTOR vVert2ToVert1 = triangleList[triIter].pt[1] - triangleList[triIter].pt[2];

            // Get the hit point ratio.
            FLOAT fHitPointTime_2_0 =  fEdge - XMVectorGetByIndex( triangleList[triIter].pt[2], iComponent );
            FLOAT fDistanceAlongVector_2_0 = fHitPointTime_2_0 / XMVectorGetByIndex( vVert2ToVert0, iComponent );
            // Calcaulte the new vert by adding the percentage of the vector plus point 2.
            vVert2ToVert0 *= fDistanceAlongVector_2_0;
            vVert2ToVert0 += triangleList[triIter].pt[2];

            // Add a new triangle.
            triangleList[triIter+1].pt[0] = triangleList[triIter].pt[0];
            triangleList[triIter+1].pt[1] = triangleList[triIter].pt[1];
            triangleList[triIter+1].pt[2] = vVert2ToVert0;

            //Get the hit point ratio.
            FLOAT fHitPointTime_2_1 =  fEdge - XMVectorGetByIndex( triangleList[triIter].pt[2], iComponent ) ;
            FLOAT fDistanceAlongVector_2_1 = fHitPointTime_2_1 / XMVectorGetByIndex( vVert2ToVert1, iComponent );
            vVert2ToVert1 *= fDistanceAlongVector_2_1;
            vVert2ToVert1 += triangleList[triIter].pt[2];
            triangleList[triIter].pt[0] = triangleList[triIter+1].pt[1];
            triangleList[triIter].pt[1] = triangleList[triIter+1].pt[2];
            triangleList[triIter].pt[2] = vVert2ToVert1;
            // Cncrement triangle count and skip the triangle we just inserted.
            ++iTriangleCnt;
            ++triIter;


          }
          else 
          { // all in
            triangleList[triIter].culled = false;

          }
        }// end if !culled loop            
      }
    }
    for( INT index=0; index < iTriangleCnt; ++index ) 
    {
      if( !triangleList[index].culled ) 
      {
        // Set the near and far plan and the min and max z values respectively.
        for( int vertind = 0; vertind < 3; ++ vertind ) 
        {
          float fTriangleCoordZ = XMVectorGetZ( triangleList[index].pt[vertind] );
          if( fNearPlane > fTriangleCoordZ ) 
          {
            fNearPlane = fTriangleCoordZ;
          }
          if( fFarPlane  <fTriangleCoordZ ) 
          {
            fFarPlane = fTriangleCoordZ;
          }
        }
      }
    }
  }    

}

//--------------------------------------------------------------------------------------
// This function takes the camera's projection matrix and returns the 8
// points that make up a view frustum.
// The frustum is scaled to fit within the Begin and End interval paramaters.
//--------------------------------------------------------------------------------------
void SceneManager::CreateFrustumPointsFromCascadeInterval( float fCascadeIntervalBegin, 
                                                        FLOAT fCascadeIntervalEnd, 
                                                        CXMMATRIX vProjection,
                                                        XMVECTOR* pvCornerPointsWorld ) 
{
    BoundingFrustum vViewFrust( vProjection );
    vViewFrust.Near = fCascadeIntervalBegin;
    vViewFrust.Far = fCascadeIntervalEnd;

    static const XMVECTORU32 vGrabY = {0x00000000,0xFFFFFFFF,0x00000000,0x00000000};
    static const XMVECTORU32 vGrabX = {0xFFFFFFFF,0x00000000,0x00000000,0x00000000};

    XMVECTORF32 vRightTop    = {vViewFrust.RightSlope,vViewFrust.TopSlope,1.0f,1.0f};
    XMVECTORF32 vLeftBottom  = {vViewFrust.LeftSlope,vViewFrust.BottomSlope,1.0f,1.0f};
    XMVECTORF32 vNear        = {vViewFrust.Near,vViewFrust.Near,vViewFrust.Near,1.0f};
    XMVECTORF32 vFar         = {vViewFrust.Far,vViewFrust.Far,vViewFrust.Far,1.0f};
    XMVECTOR vRightTopNear   = XMVectorMultiply( vRightTop, vNear );
    XMVECTOR vRightTopFar    = XMVectorMultiply( vRightTop, vFar );
    XMVECTOR vLeftBottomNear = XMVectorMultiply( vLeftBottom, vNear );
    XMVECTOR vLeftBottomFar  = XMVectorMultiply( vLeftBottom, vFar );

    pvCornerPointsWorld[0] = vRightTopNear;
    pvCornerPointsWorld[1] = XMVectorSelect( vRightTopNear, vLeftBottomNear, vGrabX );
    pvCornerPointsWorld[2] = vLeftBottomNear;
    pvCornerPointsWorld[3] = XMVectorSelect( vRightTopNear, vLeftBottomNear,vGrabY );

    pvCornerPointsWorld[4] = vRightTopFar;
    pvCornerPointsWorld[5] = XMVectorSelect( vRightTopFar, vLeftBottomFar, vGrabX );
    pvCornerPointsWorld[6] = vLeftBottomFar;
    pvCornerPointsWorld[7] = XMVectorSelect( vRightTopFar ,vLeftBottomFar, vGrabY );
}
