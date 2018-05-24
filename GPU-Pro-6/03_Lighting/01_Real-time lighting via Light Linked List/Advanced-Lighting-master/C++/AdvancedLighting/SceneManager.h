//--------------------------------------------------------------------------------------
// File: SceneManager.h
//
// This is where the shadows and light linked list are calculated and rendered.
// This sample is based off Microsoft DirectX SDK sample CascadedShadowMap11
//
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------
#pragma once

#include "RenderTarget.h"
#include "ShaderShared.h" 

#define ALIGN_16(n)     ((((size_t)(n)) + 15 ) & ~15 )

class CFirstPersonCamera;
class CDXUTSDKMesh;

#pragma warning(push)
#pragma warning(disable: 4324)

static const DirectX::XMVECTORF32 g_vFLTMAX = {  FLT_MAX,  FLT_MAX,  FLT_MAX , FLT_MAX };
static const DirectX::XMVECTORF32 g_vFLTMIN = { -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX };


struct SimpleVertex
{
  float  x,  y,  z;
  float nx, ny, nz;
  float  u,  v;
};

struct VertexShader
{
  VertexShader() : m_Shader(nullptr),
    m_ShaderBlob(nullptr)
                   {}

  ID3D11VertexShader* m_Shader;
  ID3DBlob*           m_ShaderBlob;

  inline void         Release()
  {
    SAFE_RELEASE(m_Shader);
    SAFE_RELEASE(m_ShaderBlob);
  }
};

//------------------------------------------------------------------------------------------------
struct PixelShader
{
  PixelShader()  : m_Shader(nullptr),
                   m_ShaderBlob(nullptr)
                   {}

  ID3D11PixelShader* m_Shader;
  ID3DBlob*          m_ShaderBlob;

  inline void        Release()
  {
    SAFE_RELEASE(m_Shader);
    SAFE_RELEASE(m_ShaderBlob);
  }
};
 
//--------------------------------------------------------------------------------------------------
class DynamicD3DBuffer
{
public:

  DynamicD3DBuffer(){ Clear(); }

  HRESULT   Init(ID3D11Device* pd3dDevice, uint32_t max_size);
  void      Destroy() { SAFE_RELEASE(m_D3DBuffer); }

  void      BeginFrame(ID3D11DeviceContext* pd3dDeviceContext);
  void*     Alloc( size_t size, uint32_t& offset );

  inline ID3D11Buffer** GetBuffer()  { return &m_D3DBuffer; }

private:

  inline void Clear()
  {
    m_Size       = 0;
    m_CurrentPos = 0;
    m_D3DBuffer  = NULL;
    m_BaseAddress= NULL;
  }

  uint32_t        m_Size;           // Size of the buffer
  uint32_t        m_CurrentPos;     // Current pos in the buffer
  void*           m_BaseAddress;
  ID3D11Buffer*   m_D3DBuffer; 
};
 
//------------------------------------------------------------------------------------------------------------------
__declspec(align(16)) class GPULightEnvAlloc
{

public:
  //--------------------------------------------------------------------------------------------------
  bool  Init(ID3D11Device* pd3dDevice);

  //--------------------------------------------------------------------------------------------------
  void  Destroy();

  //--------------------------------------------------------------------------------------------------
  void  BeginFrame(ID3D11DeviceContext*    pd3dDeviceContext);

  //--------------------------------------------------------------------------------------------------
  GPULightEnv* AllocateReflectionVolumes(uint32_t count);

  //--------------------------------------------------------------------------------------------------
  GPULightEnv* Allocate(uint32_t count = 1);

  //--------------------------------------------------------------------------------------------------                                       
  inline ID3D11ShaderResourceView*    GetSRV()                              { return m_StructuredBufferViewer;                 }
                                                                                                                               
  //-------------------------------------------------------------------------------------------------=                          
  inline ID3D11Buffer*                GetBuffer()                           { return m_StructuredBuffer;                       }
                                                                                                                               
  //--------------------------------------------------------------------------------------------------                         
  inline uint32_t                     GetAllocCount() const                 { return m_FrameMemOffset/sizeof(GPULightEnv);     }

  //--------------------------------------------------------------------------------------------------
  inline uint32_t                     GetAllocIndex(const GPULightEnv* env) { return uint32_t(env - (GPULightEnv*)m_FrameMem); }

  //--------------------------------------------------------------------------------------------------
  inline uint32_t                     GetReflectionIndices() const          { return (m_ReflEndIndex << 16) | m_ReflStartIndex;}

  //--------------------------------------------------------------------------------------------------
  inline uint32_t                     GetReflStartIndex()    const          { return m_ReflStartIndex; }

  //--------------------------------------------------------------------------------------------------
  inline uint32_t                     GetReflEndIndex()      const          { return m_ReflEndIndex;   }

private: 
  ID3D11ShaderResourceView*    m_StructuredBufferViewer;
  ID3D11Buffer*                m_StructuredBuffer;
  void*                       m_Placement;

  uint32_t                    m_FrameMemOffset;
  uint32_t                    m_FrameMemMax;  
  uint8_t*                    m_FrameMem; 

  uint32_t                    m_ReflStartIndex;
  uint32_t                    m_ReflEndIndex;
};

//-------------------------------------------------------------------------------------------------------------------------------
enum DebugRendering
{
  DEBUG_RENDERING_NONE   ,
  DEBUG_RENDERING_NORMALS,
  DEBUG_RENDERING_COLORS ,
  DEBUG_RENDERING_LLL    ,
};

//-------------------------------------------------------------------------------------------------------------------------------
__declspec(align(16)) class SceneManager 
{
public:
    SceneManager();
    ~SceneManager();
    
    // This runs when the application is initialized.
    HRESULT Init( ID3D11Device*           pd3dDevice,
                  ID3D11DeviceContext*    pd3dDeviceContext,
                  CDXUTSDKMesh*           pMesh,
                  uint32_t                width,
                  uint32_t                height);
    
    HRESULT ReleaseResources();

    // This runs first thing per frame 
    HRESULT InitFrame( CFirstPersonCamera* pViewerCamera,
                       CFirstPersonCamera* pLightCamera );
                        
    HRESULT RenderShadowCascades(CDXUTSDKMesh* pMesh);

    HRESULT RenderGBuffer(CDXUTSDKMesh* pMesh);
     
    HRESULT ProcessLinkedList();

    HRESULT CompositeScene(ID3D11RenderTargetView* prtvBackBuffer);

    HRESULT DrawAlpha(CDXUTSDKMesh* pMesh);

    // This runs last thing per frame
    HRESULT EndFrame(ID3D11RenderTargetView*   prtvBackBuffer);
 
    HRESULT OnResize(uint32_t width, uint32_t height);

    DirectX::XMVECTOR GetSceneAABBMin() const { return m_vSceneAABBMin; };
    DirectX::XMVECTOR GetSceneAABBMax() const { return m_vSceneAABBMax; };

    // Dynamic vertex allocation functions
    void*     DynamicVertexAllocVerts(uint32_t vertex_count, uint32_t vertex_stride);
    template <class T>
    inline T* DynamicVertexAlloc(uint32_t vertex_count)       { return (T*)DynamicVertexAllocVerts(vertex_count, sizeof(T)); }

    // Dynamic vertex draw functions
    void      DynamicVertexDrawEnd(uint32_t vertex_count);
    void      DynamicVertexRedraw();

    HRESULT   ReloadShaders();

    FLOAT                               m_fCascadePartitionsFrustum[MAX_CASCADES]; // Values are  between near and far
    FLOAT                               m_fPCFOffset; 
    FLOAT                               m_fBlurBetweenCascadesAmount;
    INT                                 m_iPCFBlurSize;

    DebugRendering                      m_DebugRendering;
    bool                                m_DynamicLights;

private:
    void*       ScratchAlloc(uint32_t size);

    void        SetViewport(CFirstPersonCamera* pViewerCamera, D3D11_VIEWPORT* vp);

    void        SetSampler(uint32_t slot,  ID3D11SamplerState* sam);
    inline void ClearSampler(uint32_t slot){ ClearSamplers(slot, 1); }
    void        ClearSamplers(uint32_t slot, uint32_t count);
    
    void        SetResourceView(uint32_t slot,  ID3D11ShaderResourceView*  srv);
    inline void ClearResourceView(uint32_t slot){ ClearResourceViews(slot, 1); }
    void        ClearResourceViews(uint32_t slot, uint32_t count);

    inline void SetTexture(uint32_t slot,  const Texture* tex) { SetResourceView(slot, tex->m_View);  }
    inline void ClearTexture(uint32_t slot)                    { ClearResourceViews(slot, 1);         }
    inline void ClearTextures(uint32_t slot, uint32_t count)   { ClearResourceViews(slot, count);     }

    void        DrawQuad(float x, float y, float w, float h);

    // Draw an ellipsoid light shell
    void        DrawEllipsoidLightShells(int inst_count);
    void        ReDrawEllipsoidLightShell();

    // Compute the near and far plane by intersecting an Ortho Projection with the Scenes AABB.
    void        ComputeNearAndFar( FLOAT& fNearPlane, 
                                   FLOAT& fFarPlane, 
                                   DirectX::FXMVECTOR vLightCameraOrthographicMin, 
                                   DirectX::FXMVECTOR vLightCameraOrthographicMax, 
                                   DirectX::XMVECTOR* pvPointsInCameraView 
                                 );
   
    void        CreateFrustumPointsFromCascadeInterval ( FLOAT fCascadeIntervalBegin, 
                                                         FLOAT fCascadeIntervalEnd, 
                                                         DirectX::CXMMATRIX vProjection,
                                                         DirectX::XMVECTOR* pvCornerPointsWorld
                                                       );

    DirectX::XMVECTOR                   m_vSceneAABBMin;
    DirectX::XMVECTOR                   m_vSceneAABBMax;
                                                                               // For example: when the shadow buffer size changes.
    DirectX::XMMATRIX                   m_matShadowProj[MAX_CASCADES]; 
    DirectX::XMMATRIX                   m_matShadowView;

    CFirstPersonCamera*                 m_pViewerCamera;
    CFirstPersonCamera*                 m_pLightCamera ;

    // D3D11 variables
    ID3D11InputLayout*                  m_pVertexLayoutLight;
    ID3D11InputLayout*                  m_pVertexLayoutMesh;

    VertexShader                        m_pvsRenderSimple;
    VertexShader                        m_pvsRenderLight; 
    VertexShader                        m_pvsRenderScene; 
    VertexShader                        m_pvsRender2D; 

    PixelShader                         m_ppsInsertLightNoCulling; 
    PixelShader                         m_ppsInsertLightBackFace; 
    PixelShader                         m_ppsDebugLight; 
    PixelShader                         m_ppsComposite; 
    PixelShader                         m_ppsClearLLL; 
    PixelShader                         m_ppsGBuffer; 
    PixelShader                         m_ppsTexture; 
    PixelShader                         m_ppsLit3D;    

    LightLinkedListTarget               m_LLLTarget;

    RenderTarget                        m_CascadedShadowMapRT; 
    RenderTarget                        m_GBufferRT;

    ID3D11Buffer*                       m_pcbLightInstancesCB;  
    ID3D11Buffer*                       m_pcbShadowCB;  
    ID3D11Buffer*                       m_pcbSimpleCB;  
    ID3D11Buffer*                       m_pcbFrameCB;  

    ID3D11Buffer*                       m_pLightVB;  

    ID3D11DepthStencilState*            m_pdsDefault;
    ID3D11DepthStencilState*            m_pdsNoWrite;
     
    ID3D11RasterizerState*              m_prsCullFrontFaces;
    ID3D11RasterizerState*              m_prsCullBackFaces;
    ID3D11RasterizerState*              m_prsCullNone; 
    ID3D11RasterizerState*              m_prsShadow; 

    ID3D11BlendState*                   m_pbsDisableRGBA;
    ID3D11BlendState*                   m_pbsAlpha;
    ID3D11BlendState*                   m_pbsNone;

    D3D11_VIEWPORT                      m_RenderVP[MAX_CASCADES];
    D3D11_VIEWPORT                      m_RenderOneTileVP;
    D3D11_VIEWPORT                      m_MainVP;

    ID3D11SamplerState*                 m_pSamLinear;
    ID3D11SamplerState*                 m_pSamPoint;
    ID3D11SamplerState*                 m_pSamShadowPCF;

    ID3D11DeviceContext*                m_pd3dDeviceContext;
    ID3D11Device*                       m_pd3dDevice;
    
    GPULightEnvAlloc                    m_GPULightEnvAlloc;
    DynamicD3DBuffer                    m_DynamicVB;

    uint32_t                            m_DynamicVertexAllocCount;
    uint32_t                            m_DynamicVertexDrawnCount;
    uint32_t                            m_DynamicVertexOffset;
    uint32_t                            m_DynamicVertexStride;
    void*                               m_DynamicVertexAlloc;

    void*                               m_ScratchBase; 
    uint32_t                            m_ScratchSize; 
    uint32_t                            m_ScratchOffset;
};

#pragma warning(pop)