#include <stdafx.h>
#include <Demo.h>
#include <TiledDeferred.h>

// Size of tile should not be changed since tileInfoGen.csh is assuming this value.
#define TILE_SIZE 16 

// Resolution of tiled shadow map. Resolution must be power of two and square, since quad-tree  
// for managing tiles will not work correctly otherwise.
#define TILED_SM_RES 8192

// Maximum resolution of one tile within tiled shadow map. Resolution must be power of two and 
// square, since quad-tree for managing tiles will not work correctly otherwise. Furthermore
// resolution must be at least 16.
#define MAX_TILE_RES 512

// Specifies how many levels the quad-tree for managing tiles within tiled shadow map should 
// have. The higher the value, the smaller the resolution of the smallest used tile will be.
// In the current configuration of 8192 resolution and 8 levels, the smallest tile will have 
// a resolution of 64. 16 is the smallest allowed value for the min tile resolution.
#define NUM_QUAD_TREE_LEVELS 8 

// Resolution of each face of the cube shadow map array.
#define CUBE_SM_RES 256

TiledDeferred::~TiledDeferred()
{
  SAFE_DELETE_PLIST(pointLights);
}

bool TiledDeferred::Create()
{
  {
    // create compute render-target 
    RenderTargetDesc rtDesc;
    rtDesc.depth = 0;
    computeRT = Demo::renderer->CreateRenderTarget(rtDesc);
    if(!computeRT)
      return false;
  }

  {
    const unsigned int numTiles = (SCREEN_WIDTH/TILE_SIZE)*(SCREEN_HEIGHT/TILE_SIZE);

    // create structured buffer for light indices
    lightIndexSB = Demo::renderer->CreateStructuredBuffer(numTiles*MAX_NUM_POINT_LIGHTS, sizeof(unsigned int), DYNAMIC_SBF | APPEND_SBF);
    if(!lightIndexSB)
      return false;

    // create structured buffer for tile-info list
    tileInfoSB = Demo::renderer->CreateStructuredBuffer(numTiles, sizeof(TileInfo), DYNAMIC_SBF);
    if(!tileInfoSB)
      return false;

    // create render-target config for computing light indices and tile-info list
    RtConfigDesc rtcDesc;
    rtcDesc.numColorBuffers = 0;
    rtcDesc.numStructuredBuffers = 2;
    rtcDesc.structuredBuffers[0] = lightIndexSB;
    rtcDesc.structuredBuffers[1] = tileInfoSB;
    rtcDesc.flags = COMPUTE_RTCF;
    tileInfoGenRTC = Demo::renderer->CreateRenderTargetConfig(rtcDesc);
    if(!tileInfoGenRTC)
      return false;

    // create shaders to generate light indices and tile-info list
    tileInfoGenShaders[NO_SHADOW_SM] = Demo::resourceManager->LoadShader("shaders/tileInfoGen.sdr");
    if(!tileInfoGenShaders[NO_SHADOW_SM])
      return false;
    tileInfoGenShaders[TILED_SHADOW_SM] = Demo::resourceManager->LoadShader("shaders/tileInfoGen.sdr", 1); // (Permutation 1 = TILED_SHADOW)
    if(!tileInfoGenShaders[TILED_SHADOW_SM])
      return false;
    tileInfoGenShaders[CUBE_SHADOW_SM] = Demo::resourceManager->LoadShader("shaders/tileInfoGen.sdr", 2); // (Permutation 2 = CUBE_SHADOW)
    if(!tileInfoGenShaders[CUBE_SHADOW_SM])
      return false;
  }
  
  {
    {
      // create tiled shadow map
      RenderTargetDesc rtDesc;
      rtDesc.width = TILED_SM_RES;
      rtDesc.height = TILED_SM_RES;
      rtDesc.depthStencilBufferDesc.format = TEX_FORMAT_DEPTH16;
      tiledShadowMapRT = Demo::renderer->CreateRenderTarget(rtDesc);
      if(!tiledShadowMapRT)
        return false;
    }

    {
      // create cube shadow map array
      RenderTargetDesc rtDesc;
      rtDesc.width = CUBE_SM_RES;
      rtDesc.height = CUBE_SM_RES;
      rtDesc.depth = MAX_NUM_POINT_LIGHTS;
      rtDesc.depthStencilBufferDesc.format = TEX_FORMAT_DEPTH16;
      rtDesc.depthStencilBufferDesc.rtFlags = CUBEMAP_RTF; 
      cubeShadowMapRT = Demo::renderer->CreateRenderTarget(rtDesc);
      if(!cubeShadowMapRT)
        return false;
    }

    // create shader for drawing tile-based shadows indirectly
    tiledShadowMapShader = Demo::resourceManager->LoadShader("shaders/tiledShadowMapPoint.sdr");
    if(!tiledShadowMapShader)
      return false;

    // create uniform buffers with cube map face indices for cube shadow map rendering
    for(unsigned int i=0; i<6; i++)
    {
      faceIndexUBs[i] = Demo::renderer->CreateUniformBuffer(sizeof(int));
      if(!faceIndexUBs[i])
        return false;
      const int faceIndex = i;
      faceIndexUBs[i]->Update(&faceIndex);
    }

    // create state objects for tiled shadow map rendering 
    RasterizerDesc rasterDesc;
    rasterDesc.cullMode = BACK_CULL;
    rasterDesc.slopeScaledDepthBias = 3.4f;
    rasterDesc.numClipPlanes = 3;
    shadowRS = Demo::renderer->CreateRasterizerState(rasterDesc);
    if(!shadowRS)
      return false;
    DepthStencilDesc depthStencilDesc; 
    defaultDSS = Demo::renderer->CreateDepthStencilState(depthStencilDesc);
    if(!defaultDSS) 
      return false;
    BlendDesc blendDesc;
    blendDesc.colorMask = 0;
    noColorWriteBS = Demo::renderer->CreateBlendState(blendDesc);
    if(!noColorWriteBS)
      return false;
  }

  {
    // render direct illumination only into accumulation render-target of GBuffers
    RtConfigDesc rtcDesc;
    rtcDesc.numColorBuffers = 1;
    lightRTC = Demo::renderer->CreateRenderTargetConfig(rtcDesc);
    if(!lightRTC)
      return false;

    // create shaders for illumination
    illumShaders[NO_SHADOW_SM] = Demo::resourceManager->LoadShader("shaders/illum.sdr");
    if(!illumShaders[NO_SHADOW_SM])
      return false;
    illumShaders[TILED_SHADOW_SM] = Demo::resourceManager->LoadShader("shaders/illum.sdr", 1); // (Permutation 1 = TILED_SHADOW)
    if(!illumShaders[TILED_SHADOW_SM])
      return false;
    illumShaders[CUBE_SHADOW_SM] = Demo::resourceManager->LoadShader("shaders/illum.sdr", 2); // (Permutation 2 = CUBE_SHADOW)
    if(!illumShaders[CUBE_SHADOW_SM])
      return false;

    // only illuminate actual geometry, not sky
    DepthStencilDesc depthStencilDesc;
    depthStencilDesc.depthTest = false;
    depthStencilDesc.depthMask = false;
    depthStencilDesc.stencilTest = true; 
    depthStencilDesc.stencilRef = 1;
    depthStencilDesc.stencilPassOp = KEEP_STENCIL_OP;
    noDepthTestDSS = Demo::renderer->CreateDepthStencilState(depthStencilDesc);
    if(!noDepthTestDSS)
      return false;
  }
 
  {
    // create shader for visualizing tiled shadow map
    tiledShadowMapVisShader = Demo::resourceManager->LoadShader("shaders/shadowMapVis.sdr");
    if(!tiledShadowMapVisShader)
      return false;
  }
  
  {
    // create structured buffers for storing light properties
    lightSBs[NO_SHADOW_SM] = Demo::renderer->CreateStructuredBuffer(MAX_NUM_POINT_LIGHTS, sizeof(PointLight::LightBufferData), DYNAMIC_SBF);
    if(!lightSBs[NO_SHADOW_SM])  
      return false;
    lightSBs[TILED_SHADOW_SM] = Demo::renderer->CreateStructuredBuffer(MAX_NUM_POINT_LIGHTS, sizeof(PointLight::LightBufferData)+sizeof(PointLight::TiledShadowBufferData), DYNAMIC_SBF);
    if(!lightSBs[TILED_SHADOW_SM])  
      return false;
    lightSBs[CUBE_SHADOW_SM] = Demo::renderer->CreateStructuredBuffer(MAX_NUM_POINT_LIGHTS, sizeof(PointLight::LightBufferData)+sizeof(PointLight::CubeShadowBufferData), DYNAMIC_SBF);
    if(!lightSBs[CUBE_SHADOW_SM])  
      return false;
  }

  // create uniform buffer for tiled deferred rendering related info
  tiledDeferredInfoUB = Demo::renderer->CreateUniformBuffer(sizeof(TiledDeferredInfo));
  if(!tiledDeferredInfoUB)
    return false;
  tiledDeferredInfo.tileSize = TILE_SIZE;
  tiledDeferredInfo.numTilesX = SCREEN_WIDTH/TILE_SIZE;

  // create timer-query object for measuring CPU/ GPU times of shadow map rendering 
  shadowTimerQueryObject = Demo::renderer->CreateTimerQueryObject();
  if(!shadowTimerQueryObject)
    return false;

  // create timer-query object for measuring CPU/ GPU times of illumination
  illumTimerQueryObject = Demo::renderer->CreateTimerQueryObject();
  if(!illumTimerQueryObject)
    return false;
  
  // initialize tile-map that will manage tiles within tiled shadow map
  if(!tileMap.Init(TILED_SM_RES, MAX_TILE_RES, NUM_QUAD_TREE_LEVELS))
    return false;

  return true;
}

PointLight* TiledDeferred::CreatePointLight(const Vector3 &position, float radius, const Color &color)
{
  if(pointLights.GetSize() >= MAX_NUM_POINT_LIGHTS)
    return NULL;

  PointLight *pointLight = new PointLight;
  if(!pointLight)
    return NULL;
  if(!pointLight->Create(position, radius, color))
  {
    SAFE_DELETE(pointLight);
    return NULL;
  }
  pointLights.AddElement(&pointLight);
  return pointLight;
}

bool TiledDeferred::CreateIndirectDrawData()
{
  // calculate number of meshes
  numMeshes = 0;
  for(unsigned int i=0; i<Demo::resourceManager->GetNumDemoMeshes(); i++) 
  {
    numMeshes += Demo::resourceManager->GetDemoMesh(i)->GetNumSubMeshes();
  }

  // create structured buffer for storing information for meshes
  meshInfoSB = Demo::renderer->CreateStructuredBuffer(numMeshes, sizeof(MeshInfo));
  if(!meshInfoSB)  
    return false;
  unsigned int meshIndex = 0;
  meshInfoSB->BeginUpdate();
  for(unsigned int i=0; i<Demo::resourceManager->GetNumDemoMeshes(); i++) 
  {
    const DemoMesh *mesh = Demo::resourceManager->GetDemoMesh(i);
    for(unsigned int j=0; j<mesh->GetNumSubMeshes(); j++)
    {
      const DemoSubmesh *subMesh = mesh->GetSubMesh(j);
      MeshInfo meshInfo;
      meshInfo.mins = subMesh->boundingBox.mins;
      meshInfo.firstIndex = subMesh->firstIndex;
      meshInfo.maxes = subMesh->boundingBox.maxes;
      meshInfo.numIndices = subMesh->numIndices;
      meshInfoSB->Update(meshIndex++, 0, sizeof(MeshInfo), &meshInfo);  
    }
  }
  meshInfoSB->EndUpdate(); 

  // create structured buffer for indirect draw-list
  indirectDrawSB = Demo::renderer->CreateStructuredBuffer(numMeshes*MAX_NUM_POINT_LIGHTS, sizeof(DrawIndirectCmd), DYNAMIC_SBF | APPEND_SBF | INDIRECT_DRAW_SBF);
  if(!indirectDrawSB)
    return false;
  
  // create structured buffer for shadow light-indices 
  shadowLightIndexSB = Demo::renderer->CreateStructuredBuffer(numMeshes*MAX_NUM_POINT_LIGHTS, sizeof(unsigned int), DYNAMIC_SBF | APPEND_SBF);
  if(!shadowLightIndexSB)
    return false;
  
  // create render-target config for computing indirect draw-list
  RtConfigDesc rtcDesc;
  rtcDesc.numColorBuffers = 0;
  rtcDesc.numStructuredBuffers = 2;
  rtcDesc.structuredBuffers[0] = indirectDrawSB;
  rtcDesc.structuredBuffers[1] = shadowLightIndexSB;
  rtcDesc.flags = COMPUTE_RTCF;
  drawListGenRTC = Demo::renderer->CreateRenderTargetConfig(rtcDesc);
  if(!drawListGenRTC)
    return false;

  // create shader to generate indirect draw-list
  drawListGenShader = Demo::resourceManager->LoadShader("shaders/drawListGen.sdr");
  if(!drawListGenShader)
    return false;

  return true;
}

static int CompareLights(const void *a, const void *b)
{
  const PointLight *sA = (*(PointLight**)a);
  const PointLight *sB = (*(PointLight**)b);

  if(sA->GetLightArea() > sB->GetLightArea())
    return -1;
  else if(sA->GetLightArea() < sB->GetLightArea())
    return 1;
  if(sA->GetIndex()< sB->GetIndex())
    return -1;
  else if(sA->GetIndex() > sB->GetIndex())
    return 1;

  return 0;
} 

void TiledDeferred::UpdateLights()
{
  // First calculate for all active lights, if they are in the view-frustum, their screen-space AABB
  // and an approximate light-area, that determines how high the resolution of the corresponding tile
  // within the tiled shadow map will be.
  for(unsigned int i=0; i<pointLights.GetSize(); i++) 
  {
    if(pointLights[i]->IsActive())
      pointLights[i]->CalculateVisibility();
  } 

  // Sort all lights according to their light-area. In this way lights with larger light-area will be
  // at the beginning of the list, which is required to optimally use the space in the tiled shadow
  // map. Invisible lights will be the end of the list since their light-area is 0.0f, followed by all
  // inactive lights with a light-area of -1.0f.
  pointLights.Sort(CompareLights);

  // Clear quad-tree that manages the tiles and update all active/ visible lights.
  tileMap.Clear();
  tiledDeferredInfo.numLights = 0;
  for(unsigned int i=0; i<pointLights.GetSize(); i++) 
  {
    if(pointLights[i]->IsActive() && pointLights[i]->IsVisible())
    {
      pointLights[i]->Update(tiledDeferredInfo.numLights++);
    }
  }

  if(shadowMode == TILED_SHADOW_SM)
  {
    tiledDeferredInfo.invShadowMapSize.x = 1.0f/(float)tiledShadowMapRT->GetWidth();
    tiledDeferredInfo.invShadowMapSize.y = 1.0f/(float)tiledShadowMapRT->GetHeight();
  }
  else if(shadowMode == CUBE_SHADOW_SM)
  {
    tiledDeferredInfo.invShadowMapSize.x = 1.0f/(float)cubeShadowMapRT->GetWidth();
    tiledDeferredInfo.invShadowMapSize.y = 1.0f/(float)cubeShadowMapRT->GetHeight();
  }
 
  tiledDeferredInfoUB->Update(&tiledDeferredInfo);
}

void TiledDeferred::UpdateLightBuffer()
{
  // Update structured buffers, that store the properties of all active/ visible lights, corresponding to the currently
  // set shadow mode.
  switch(shadowMode)
  {
  case NO_SHADOW_SM:
    {
      lightSBs[NO_SHADOW_SM]->BeginUpdate();
      unsigned int lightIndex = 0;
      for(unsigned int i=0; i<pointLights.GetSize(); i++) 
      {
        if(pointLights[i]->IsActive() && pointLights[i]->IsVisible())
        {
          lightSBs[NO_SHADOW_SM]->Update(lightIndex, 0, sizeof(PointLight::LightBufferData), pointLights[i]->GetLightBufferData());
          lightIndex++;
        }
      }
      lightSBs[NO_SHADOW_SM]->EndUpdate();
      break;
    }

  case TILED_SHADOW_SM:
    {
      lightSBs[TILED_SHADOW_SM]->BeginUpdate();
      unsigned int lightIndex = 0;
      for(unsigned int i=0; i<pointLights.GetSize(); i++) 
      {
        if(pointLights[i]->IsActive() && pointLights[i]->IsVisible())
        {
          lightSBs[TILED_SHADOW_SM]->Update(lightIndex, 0, sizeof(PointLight::LightBufferData), pointLights[i]->GetLightBufferData());
          lightSBs[TILED_SHADOW_SM]->Update(lightIndex, sizeof(PointLight::LightBufferData), sizeof(PointLight::TiledShadowBufferData), 
                                            pointLights[i]->GetTiledShadowBufferData());
          lightIndex++;
        }
      }
      lightSBs[TILED_SHADOW_SM]->EndUpdate();
      break;
    }

  case CUBE_SHADOW_SM:
    {
      lightSBs[CUBE_SHADOW_SM]->BeginUpdate();
      unsigned int lightIndex = 0;
      for(unsigned int i=0; i<pointLights.GetSize(); i++) 
      {
        if(pointLights[i]->IsActive() && pointLights[i]->IsVisible())
        {
          lightSBs[CUBE_SHADOW_SM]->Update(lightIndex, 0, sizeof(PointLight::LightBufferData), pointLights[i]->GetLightBufferData());
          lightSBs[CUBE_SHADOW_SM]->Update(lightIndex, sizeof(PointLight::LightBufferData), sizeof(PointLight::CubeShadowBufferData), pointLights[i]->GetCubeShadowBufferData());
          lightIndex++;
        }
      }
      lightSBs[CUBE_SHADOW_SM]->EndUpdate();
      break;
    }
  }
}

void TiledDeferred::ComputeTileInfo()
{
  // Determine inside a compute shader which lights are intersecting the tiles of the screen. This is done by 
  // checking the screen-space AABB of the light against the screen-space AABB of each tile. The indices of all
  // intersecting lights will be written into one linear list in lightIndicesSB, the start-/end-index of each 
  // tile into this list will be written into tileInfoSB.
  GpuCmd gpuCmd(COMPUTE_CM);
  gpuCmd.order = PRE_SHADOW_CO;
  gpuCmd.compute.renderTarget = computeRT;
  gpuCmd.compute.renderTargetConfig = tileInfoGenRTC;
  gpuCmd.compute.textures[COLOR_TEX_ID] = Demo::renderer->GetRenderTarget(GBUFFERS_RT_ID)->GetDepthStencilTexture(); // depth
  gpuCmd.compute.customUBs[0] = tiledDeferredInfoUB;
  gpuCmd.compute.customSBs[0] = lightSBs[shadowMode];
  gpuCmd.compute.numThreadGroupsX = SCREEN_WIDTH/TILE_SIZE;
  gpuCmd.compute.numThreadGroupsY = SCREEN_HEIGHT/TILE_SIZE;
  gpuCmd.compute.numThreadGroupsZ = 1;
  gpuCmd.compute.shader = tileInfoGenShaders[shadowMode]; 
  Demo::renderer->AddGpuCmd(gpuCmd);
}

void TiledDeferred::ComputeDrawList()
{
  // Generate inside a compute shader all draw-commands, necessary to render the shadow maps of all lights in 
  // the scene. This is done by spawning one thread group for each mesh. Within one thread-group for each
  // relevant light a thread is spawned. Each thread checks the sphere of the corresponding point light against 
  // the AABB of the current mesh. If at least one light intersects the AABB of the mesh, then a indirect draw-
  // command will be generated with the instance count set to the number of intersecting lights. The corresponding 
  // light indices are written into one linear list in shadowLightIndicesSB. The start index of the current draw-
  // command into this list is written into its BaseInstance property, which is otherwise unused, since we are
  // using non-instanced vertex attributes. Later on when the draw-commands are executed indirectly the offset in 
  // the light list can be determined by reading gl_BaseInstanceARB and gl_InstanceID in the corresponding vertex 
  // shader, which avoids a lookup in a further buffer if we would use gl_DrawIDARB instead. With the help of this 
  // offset and the shadowLightIndicesSB the current light index can be generated for rendering the shadow map of 
  // the corresponding point light. In this way a massive amount of shadow maps can be rendered with only submitting
  // one indirect draw-call on the client side.
  GpuCmd gpuCmd(COMPUTE_CM);
  gpuCmd.order = PRE_SHADOW_CO;
  gpuCmd.compute.renderTarget = computeRT;
  gpuCmd.compute.renderTargetConfig = drawListGenRTC;
  gpuCmd.compute.customUBs[0] = tiledDeferredInfoUB;
  gpuCmd.compute.customSBs[0] = meshInfoSB;
  gpuCmd.compute.customSBs[1] = lightSBs[TILED_SHADOW_SM];
  gpuCmd.compute.numThreadGroupsX = numMeshes;
  gpuCmd.compute.numThreadGroupsY = 1;
  gpuCmd.compute.numThreadGroupsZ = 1;
  gpuCmd.compute.shader = drawListGenShader;
  Demo::renderer->AddGpuCmd(gpuCmd);
}

void TiledDeferred::AddShadowMapSurfaces()
{
  {
    // begin timer query for shadow map rendering
    GpuCmd gpuCmd(TIMER_QUERY_CM);
    gpuCmd.order = SHADOW_BEGIN_TIMER_QUERY_CO;
    gpuCmd.timerQuery.object = shadowTimerQueryObject;
    gpuCmd.timerQuery.mode = BEGIN_TIMER_QUERY;
    Demo::renderer->AddGpuCmd(gpuCmd);
  }

  if(shadowMode == TILED_SHADOW_SM)
  {
    // Render the shadow maps of all lights in the scene by submitting one indirect draw-call with the previously generated
    // indirect draw-buffer. Since we don't know on software side how many draw-commands had been spawned by the GPU, we
    // use the GL_ARB_indirect_parameters extension, which clamps the in software specified max draw-count to a value 
    // specified in a GPU buffer.
    // All shadow maps are rendered as tiles into one large texture atlas. The resolution of each tile is dynamically adapted 
    // to the screen-space light-area of the corresponding light. In this way we not only can reduce the bandwidth for writing 
    // and reading the shadow map texels, but with the help of a quad-tree we can efficiently pack all tiles in a way, that a 
    // very high amount of shadow maps can be kept within a limited texture area.
    // The shadow map for each point light is generated in the spirit of the Tetrahedron Omnidirectional Shadow Mapping technique,
    // but in a heavily modified form. For each triangle generated by the indirect draw-call a geometry-shader is spawned. In the
    // geometry shader for each of the four tetrahedron faces the clip-distances are determined to the corresponding three planes,
    // that separate the volume of the regular tetrahedron. These clip-distances are passed to the programmable clipping unit of the
    // GPU via gl_ClipDistance. In this way the shadow map for each tetrahedron face and each light can be rapidly written into the 
    // texture atlas. The angle between the clipping planes of the regular tetrahedron have been slightly enlarged, in order to
    // enable later on artifact-free filtering of the shadow-map (in our case 16x PCF times 4x Hardware-PCF). Prior to emitting the
    // primitive the anyway calculated clip-distances are used to reject the primitive, which further accelerates the algorithm.
    // Furthermore prior to doing any operations in the geometry shader, manual back-face culling is performed to again increase
    // the performance. Against all expectations it has shown to be far more faster to do a loop over all four faces in one geometry
    // shader invocation, rather than invoking the geometry shader four times. The reasons for this can be that in a high percentage 
    // of cases less than four primitives have to be emitted and that the light buffer data has to be fetched only once for each 
    // incoming primitive. Furthermore the manual back-face culling is shared among the four faces.
    GpuCmd gpuCmd(INDIRECT_DRAW_CM);
    gpuCmd.order = SHADOW_CO;
    gpuCmd.indirectDraw.renderTarget = tiledShadowMapRT;
    gpuCmd.indirectDraw.primitiveType = TRIANGLES_PRIMITIVE;	
    gpuCmd.indirectDraw.camera = Demo::renderer->GetCamera(MAIN_CAMERA_ID);
    gpuCmd.indirectDraw.vertexLayout = Demo::renderer->GetVertexLayout(SHADOW_VL_ID);
    gpuCmd.indirectDraw.vertexBuffer = Demo::renderer->GetVertexBuffer(GEOMETRY_VB_ID);
    gpuCmd.indirectDraw.indexBuffer = Demo::renderer->GetIndexBuffer(GEOMETRY_IB_ID);
    gpuCmd.indirectDraw.stride = sizeof(DrawIndirectCmd); 
    gpuCmd.indirectDraw.maxDrawCount = numMeshes; 
    gpuCmd.indirectDraw.rasterizerState = shadowRS;
    gpuCmd.indirectDraw.depthStencilState = defaultDSS;
    gpuCmd.indirectDraw.blendState = noColorWriteBS;
    gpuCmd.indirectDraw.customSBs[0] = shadowLightIndexSB;
    gpuCmd.indirectDraw.customSBs[1] = lightSBs[TILED_SHADOW_SM];
    gpuCmd.indirectDraw.customSBs[2] = indirectDrawSB;
    gpuCmd.indirectDraw.shader = tiledShadowMapShader;
    Demo::renderer->AddGpuCmd(gpuCmd); 	
  }
  else if(shadowMode == CUBE_SHADOW_SM)
  {
    // As reference a classic cube map-based approach has been chosen. On software side for each light and each cube map
    // face it is determined, whether each mesh is intersecting the corresponding frustum. Then for each relevant cube map
    // face and each light one draw-command is submitted. To speed up rendering one large vertex/ index buffer is used
    // just like for the tiled variant to avoid expensive resource switching. Since we are using Tiled Deferred Lighting 
    // the shadow maps of all lights must be contained in one single texture. Therefore a cube map array texture is used.
    // In contrast to the tiled variant the amount of CPU submitted draw-calls is extremely high.
    for(unsigned int i=0; i<pointLights.GetSize(); i++) 
    {
      if(pointLights[i]->IsActive() && pointLights[i]->IsVisible())
      {
        for(unsigned int j=0; j<Demo::resourceManager->GetNumDemoMeshes(); j++) 
          Demo::resourceManager->GetDemoMesh(j)->AddCubeShadowMapSurfaces(pointLights[i]);
      }
    }
  }

  {
    // end timer query for shadow map rendering
    GpuCmd gpuCmd(TIMER_QUERY_CM);
    gpuCmd.order = SHADOW_END_TIMER_QUERY_CO;
    gpuCmd.timerQuery.object = shadowTimerQueryObject;
    gpuCmd.timerQuery.mode = END_TIMER_QUERY;
    Demo::renderer->AddGpuCmd(gpuCmd);
  }
}

void TiledDeferred::AddLitSurface()
{
  {
    // begin timer query for illumination rendering
    GpuCmd gpuCmd(TIMER_QUERY_CM);
    gpuCmd.order = ILLUM_BEGIN_TIMER_QUERY_CO;
    gpuCmd.timerQuery.object = illumTimerQueryObject;
    gpuCmd.timerQuery.mode = BEGIN_TIMER_QUERY;
    Demo::renderer->AddGpuCmd(gpuCmd);
  }

  {
    // The illumination is done according to the Tiled Deferred Lighting technique. For this a full-screen quad
    // is rendered with the fragment shader. For each pixel the corresponding screen tile is determined and with
    // the help of the tileInfoSB and lightIndicesSB each pixel is illuminated by all lights intersecting the tile
    // to which the pixel belongs. During the iteration for each light the corresponding shadow map is fetched 
    // either from the tiled shadow map in TILED_SHADOW shadow mode or from the cube map array in CUBE_SHADOW
    // shadow mode.
    OGL_RenderTarget *gBuffersRT = Demo::renderer->GetRenderTarget(GBUFFERS_RT_ID);
    GpuCmd gpuCmd(DRAW_CM);
    Demo::renderer->SetupPostProcessSurface(gpuCmd.draw);
    gpuCmd.order = ILLUM_CO;
    gpuCmd.draw.renderTarget = gBuffersRT;
    gpuCmd.draw.renderTargetConfig = lightRTC;
    gpuCmd.draw.camera = Demo::renderer->GetCamera(MAIN_CAMERA_ID);
    gpuCmd.draw.textures[COLOR_TEX_ID] = gBuffersRT->GetTexture(1); // albedoGloss
    gpuCmd.draw.textures[NORMAL_TEX_ID] = gBuffersRT->GetTexture(2); // normal
    gpuCmd.draw.textures[SPECULAR_TEX_ID] = gBuffersRT->GetDepthStencilTexture(); // depth
    if(shadowMode == TILED_SHADOW_SM)
      gpuCmd.draw.textures[CUSTOM0_TEX_ID] = tiledShadowMapRT->GetDepthStencilTexture(); // shadow map
    else if(shadowMode == CUBE_SHADOW_SM)
      gpuCmd.draw.textures[CUSTOM0_TEX_ID] = cubeShadowMapRT->GetDepthStencilTexture(); // shadow map
    gpuCmd.draw.samplers[CUSTOM0_TEX_ID] = Demo::renderer->GetSampler(SHADOW_MAP_SAMPLER_ID);
    gpuCmd.draw.customUBs[0] = tiledDeferredInfoUB;
    gpuCmd.draw.customSBs[0] = lightSBs[shadowMode];
    gpuCmd.draw.customSBs[1] = lightIndexSB;
    gpuCmd.draw.customSBs[2] = tileInfoSB;
    gpuCmd.draw.depthStencilState = noDepthTestDSS;
    gpuCmd.draw.shader = illumShaders[shadowMode];
    Demo::renderer->AddGpuCmd(gpuCmd); 
  }
  
  {
    // end timer query for illumination rendering
    GpuCmd gpuCmd(TIMER_QUERY_CM);
    gpuCmd.order = ILLUM_END_TIMER_QUERY_CO;
    gpuCmd.timerQuery.object = illumTimerQueryObject;
    gpuCmd.timerQuery.mode = END_TIMER_QUERY;
    Demo::renderer->AddGpuCmd(gpuCmd);
  }
}

void TiledDeferred::AddShadowMapVisSurface()
{
  // Visualize the tiled shadow map.
  GpuCmd gpuCmd(DRAW_CM);
  gpuCmd.order = GUI_CO;
  gpuCmd.draw.renderTarget = Demo::renderer->GetRenderTarget(BACK_BUFFER_RT_ID);
  gpuCmd.draw.camera = Demo::renderer->GetCamera(MAIN_CAMERA_ID);
  gpuCmd.draw.textures[COLOR_TEX_ID] = tiledShadowMapRT->GetDepthStencilTexture(); // shadow map
  gpuCmd.draw.samplers[COLOR_TEX_ID] = Demo::renderer->GetSampler(POINT_SAMPLER_ID);
  gpuCmd.draw.shader = tiledShadowMapVisShader;
  Demo::renderer->SetupPostProcessSurface(gpuCmd.draw);
  gpuCmd.draw.primitiveType = LINES_PRIMITIVE;
  gpuCmd.draw.numElements = 2;
  Demo::renderer->AddGpuCmd(gpuCmd); 
}

void TiledDeferred::Execute()
{
  if(!active)
    return;

  UpdateLights();
  UpdateLightBuffer();

  ComputeTileInfo();

  if(shadowMode == TILED_SHADOW_SM)
    ComputeDrawList();

  if((shadowMode == TILED_SHADOW_SM) || (shadowMode == CUBE_SHADOW_SM))
    AddShadowMapSurfaces();

  AddLitSurface();

  if((shadowMode == TILED_SHADOW_SM) && visTiledShadowMap)
    AddShadowMapVisSurface();

  if((shadowMode == TILED_SHADOW_SM) || (shadowMode == CUBE_SHADOW_SM))
    shadowTimerQueryObject->QueryResult();

  illumTimerQueryObject->QueryResult();
}
