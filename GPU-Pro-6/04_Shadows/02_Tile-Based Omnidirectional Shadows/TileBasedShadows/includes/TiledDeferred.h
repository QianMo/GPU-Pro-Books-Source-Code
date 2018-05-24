#ifndef TILED_DEFERRED_LIGHTING_H
#define TILED_DEFERRED_LIGHTING_H

#include <IPostProcessor.h>
#include <PointLight.h>
#include <TileMap.h>

enum shadowModes
{
  NO_SHADOW_SM=0,
  TILED_SHADOW_SM,
  CUBE_SHADOW_SM,
  NUM_SHADOW_MODES
};

class OGL_Shader;
class OGL_UniformBuffer;
class OGL_StructuredBuffer;
class RenderTargetConfig;
class OGL_RasterizerState;
class OGL_DepthStencilState;
class OGL_BlendState;
class OGL_TimerQueryObject;

// TiledDeferred
//
// Performs Tiled Deferred Lighting. For the point light shadows Tiled-based Omnidirectional Shadows
// is used, respectively a cube map array-based approach as reference.
class TiledDeferred: public IPostProcessor
{
public: 
  friend class PointLight;

  // contains general information required for Tiled Deferred Lighting 
  struct TiledDeferredInfo
  {
    Vector2 invShadowMapSize;
    unsigned int numLights;
    unsigned int tileSize;
    unsigned int numTilesX; 
  };

  // contains information for each tile used for Tiled Deferred Lighting
  struct TileInfo
  {
    unsigned int startLightIndex;
    unsigned int endLightIndex;
  };

  // contains information for each mesh
  struct MeshInfo
  {
    Vector3 mins;
    unsigned int firstIndex;
    Vector3 maxes;
    unsigned int numIndices;
  };

  // contains information for each indirect draw command
  struct DrawIndirectCmd
  {
    unsigned int count;
    unsigned int instanceCount;
    unsigned int firstIndex;
    int baseVertex;
    unsigned int baseInstance;
  };

  TiledDeferred():
    computeRT(NULL),
    tileInfoGenRTC(NULL),
    lightIndexSB(NULL),
    tileInfoSB(NULL),
    drawListGenRTC(NULL),
    meshInfoSB(NULL),
    indirectDrawSB(NULL),
    shadowLightIndexSB(NULL),
    drawListGenShader(NULL),
    tiledShadowMapRT(NULL),
    cubeShadowMapRT(NULL),
    tiledShadowMapShader(NULL),
    shadowRS(NULL),
    defaultDSS(NULL),
    noColorWriteBS(NULL),
    lightRTC(NULL),
    noDepthTestDSS(NULL),
    tiledShadowMapVisShader(NULL),
    tiledDeferredInfoUB(NULL),
    shadowTimerQueryObject(NULL),
    illumTimerQueryObject(NULL),
    numMeshes(0),
    shadowMode(TILED_SHADOW_SM),
    useFrustumCulling(true),
    visTiledShadowMap(false)
  {
    strcpy(name, "TiledDeferred");
    for(unsigned int i=0; i<NUM_SHADOW_MODES; i++)
    {
      tileInfoGenShaders[i] = NULL;
      illumShaders[i] = NULL;
      lightSBs[i] = NULL;
    }
    for(unsigned int i=0; i<6; i++)
      faceIndexUBs[i] = NULL;
  }

  virtual ~TiledDeferred();

  virtual bool Create() override;

  virtual OGL_RenderTarget* GetOutputRT() const override
  {
    return NULL;
  }

  virtual void Execute() override;

  PointLight* CreatePointLight(const Vector3 &position, float radius, const Color &color);

  bool CreateIndirectDrawData();

  void SetShadowMode(shadowModes shadowMode)
  {
    this->shadowMode = shadowMode;
  }

  shadowModes GetShadowMode() const
  {
    return shadowMode;
  }

  void EnableFrustumCulling(bool enable)
  {
    useFrustumCulling = enable;
  }

  bool IsFrustumCullingEnabled() const
  {
    return useFrustumCulling;
  }

  void EnableTiledShadowMapVis(bool enable)
  {
    visTiledShadowMap = enable;
  }

  bool IsTiledShadowMapVisEnabled() const
  {
    return visTiledShadowMap;
  }

  unsigned int GetNumVisibleLights() const
  {
    return tiledDeferredInfo.numLights;
  }

  // gets time taken by the CPU for shadow map rendering
  double GetShadowCpuElapsedTime() const
  {
    if((shadowMode == TILED_SHADOW_SM) || (shadowMode == CUBE_SHADOW_SM))
      return shadowTimerQueryObject->GetCpuElapsedTime();
    else
      return 0.0;
  }

  // gets time taken by the GPU for shadow map rendering
  double GetShadowGpuElapsedTime() const
  {
    if((shadowMode == TILED_SHADOW_SM) || (shadowMode == CUBE_SHADOW_SM))
      return shadowTimerQueryObject->GetGpuElapsedTime();
    else
      return 0.0;
  }

  // gets time taken by the GPU for illumination
  double GetIllumGpuElapsedTime() const
  {
    return illumTimerQueryObject->GetGpuElapsedTime();
  }

private:
  void UpdateLights();

  void UpdateLightBuffer();

  void ComputeTileInfo();

  void ComputeDrawList();

  void AddShadowMapSurfaces();

  void AddLitSurface();

  void AddShadowMapVisSurface();
  
  // list of all dynamic point lights
  List<PointLight*> pointLights;

  TiledDeferredInfo tiledDeferredInfo;

  // objects used to generate light-indices and tile-info list for tiled deferred lighting
  OGL_RenderTarget *computeRT;
  RenderTargetConfig *tileInfoGenRTC;
  OGL_StructuredBuffer *lightIndexSB;
  OGL_StructuredBuffer *tileInfoSB;
  OGL_Shader *tileInfoGenShaders[NUM_SHADOW_MODES];

  // objects used to generate draw-list indirectly for tiled shadow map rendering
  RenderTargetConfig *drawListGenRTC;  
  OGL_StructuredBuffer *meshInfoSB;
  OGL_StructuredBuffer *indirectDrawSB;
  OGL_StructuredBuffer *shadowLightIndexSB;
  OGL_Shader *drawListGenShader; 

  // objects used for tiled and cube shadow map rendering
  OGL_RenderTarget *tiledShadowMapRT;
  OGL_RenderTarget *cubeShadowMapRT;
  OGL_Shader *tiledShadowMapShader;
  OGL_UniformBuffer *faceIndexUBs[6];
  OGL_RasterizerState *shadowRS;
  OGL_DepthStencilState *defaultDSS;
  OGL_BlendState *noColorWriteBS;

  // objects used for tiled deferred lighting
  RenderTargetConfig *lightRTC; 
  OGL_Shader *illumShaders[NUM_SHADOW_MODES];
  OGL_DepthStencilState *noDepthTestDSS;
 
  // objects used for visualizing tiled shadow map
  OGL_Shader *tiledShadowMapVisShader;

  // structured buffers for storing light information
  OGL_StructuredBuffer *lightSBs[NUM_SHADOW_MODES];
  
  // uniform buffer for storing tiled deferred lighting related information
  OGL_UniformBuffer *tiledDeferredInfoUB;

  // timer-query object for measuring CPU/ GPU times for shadow map rendering
  OGL_TimerQueryObject *shadowTimerQueryObject;

  // timer-query object for measuring CPU/ GPU times for illumination
  OGL_TimerQueryObject *illumTimerQueryObject;
 
  // quad-tree for managing tiles within tiled shadow map
  TileMap tileMap;

  // helper variables
  unsigned int numMeshes;
  shadowModes shadowMode;  
  bool useFrustumCulling;
  bool visTiledShadowMap;

};

#endif