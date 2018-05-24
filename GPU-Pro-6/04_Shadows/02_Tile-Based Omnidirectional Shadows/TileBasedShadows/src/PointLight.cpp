#include <stdafx.h>
#include <Demo.h>
#include <TiledDeferred.h>
#include <Aabb.h>
#include <PointLight.h>

bool PointLight::Create(const Vector3 &position, float radius, const Color &color)
{
  // cache pointer to TiledDeferred post-processor
  tiledDeferredPP = (TiledDeferred*)Demo::renderer->GetPostProcessor("TiledDeferred");
  if(!tiledDeferredPP)
    return false;

  lightIndex = tiledDeferredPP->pointLights.GetSize();

  lightBD.position = position;
  lightBD.radius = radius;
  lightBD.color = color;

  // tiled shadow map
  {
    const float fov0 = 143.98570868f+1.99273682f;
    const float fov1 = 125.26438968f+2.78596497f;
    tiledShadowProjMatrices[0].SetPerspective(Vector2(fov0, fov1), 0.2f, radius);
    tiledShadowProjMatrices[1].SetPerspective(Vector2(fov1, fov0), 0.2f, radius);

    Matrix4 xRotMatrix, yRotMatrix, zRotMatrix;
    xRotMatrix.SetRotationY(180.0f);
    yRotMatrix.SetRotationX(27.36780516f);
    tiledShadowRotMatrices[0] = yRotMatrix*xRotMatrix;
    xRotMatrix.SetRotationY(0.0f);
    yRotMatrix.SetRotationX(27.36780516f);
    zRotMatrix.SetRotationZ(90.0f);
    tiledShadowRotMatrices[1] = zRotMatrix*yRotMatrix*xRotMatrix; 
    xRotMatrix.SetRotationY(270.0f);
    yRotMatrix.SetRotationX(-27.36780516f);
    tiledShadowRotMatrices[2] = yRotMatrix*xRotMatrix; 
    xRotMatrix.SetRotationY(90.0f);
    yRotMatrix.SetRotationX(-27.36780516f);
    zRotMatrix.SetRotationZ(90.0f);
    tiledShadowRotMatrices[3] = zRotMatrix*yRotMatrix*xRotMatrix; 
  }

  // cube shadow map
  {
    cubeShadowProjMatrix.SetPerspective(Vector2(90.0f, 90.0f), 0.2f, radius);

    // TEXTURE_CUBE_MAP_POSITIVE_X
    cubeShadowRotMatrices[0].Set(0.0f, 0.0f, -1.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f);

    // TEXTURE_CUBE_MAP_NEGATIVE_X 
    cubeShadowRotMatrices[1].Set(0.0f, 0.0f, 1.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f);

    // TEXTURE_CUBE_MAP_POSITIVE_Y
    cubeShadowRotMatrices[2].Set(1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f);

    // TEXTURE_CUBE_MAP_NEGATIVE_Y
    cubeShadowRotMatrices[3].Set(1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f);

    // TEXTURE_CUBE_MAP_POSITIVE_Z
    cubeShadowRotMatrices[4].Set(1.0f, 0.0f, 0.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f);

    // TEXTURE_CUBE_MAP_NEGATIVE_Z
    cubeShadowRotMatrices[5].Set(-1.0f, 0.0f, 0.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f);
  }

  // shader for cube shadow map generation
  cubeShadowMapShader = Demo::resourceManager->LoadShader("shaders/cubeShadowMapPoint.sdr");
  if(!cubeShadowMapShader)
    return false;

  lightIndexUB = Demo::renderer->CreateUniformBuffer(sizeof(int));  
  if(!lightIndexUB)
    return false; 

  RasterizerDesc rasterDesc;
  rasterDesc.cullMode = BACK_CULL;
  backCullRS = Demo::renderer->CreateRasterizerState(rasterDesc);
  if(!backCullRS)
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

  return true;
}

void PointLight::CalculateVisibility()
{
  lightArea = 0.0f;
  lightBD.mins.Set(-1.0f, -1.0f, -1.0f, 0.0f);
  lightBD.maxes.Set(-1.0f, -1.0f, -1.0f, 0.0f);
  
  visible = !tiledDeferredPP->useFrustumCulling;

  const Camera *camera = Demo::renderer->GetCamera(MAIN_CAMERA_ID);

  // check, if light volume is inside camera frustum
  if(!camera->GetFrustum().IsSphereInside(lightBD.position, lightBD.radius))
    return;

  visible = true;

  // calculate bounding box of light sphere in view-space
  Vector3 centerVS = camera->GetViewMatrix()*lightBD.position;
  Vector3 corners[8];
  corners[0] = centerVS+Vector3(-lightBD.radius, lightBD.radius, lightBD.radius);
  corners[1] = centerVS+Vector3(-lightBD.radius, -lightBD.radius, lightBD.radius);
  corners[2] = centerVS+Vector3(lightBD.radius, -lightBD.radius, lightBD.radius);
  corners[3] = centerVS+Vector3(lightBD.radius, lightBD.radius, lightBD.radius);
  corners[4] = centerVS+Vector3(-lightBD.radius, lightBD.radius, -lightBD.radius);
  corners[5] = centerVS+Vector3(-lightBD.radius, -lightBD.radius, -lightBD.radius);
  corners[6] = centerVS+Vector3(lightBD.radius, -lightBD.radius, -lightBD.radius);
  corners[7] = centerVS+Vector3(lightBD.radius, lightBD.radius, -lightBD.radius);

  // calculate bounding box in clip-space
  Aabb boundingBox;
  boundingBox.Clear();
  for(unsigned int i=0; i<8; i++)
  {
    if(corners[i].z > 0.0f)
      corners[i].z = 0.0f;
    corners[i] = camera->GetProjMatrix()*corners[i];
    corners[i] = (corners[i]*0.5f)+Vector3(0.5f, 0.5f, 0.5f);
    boundingBox.Inflate(corners[i]);
  }
 
  // calculate light area in screen-space
  float width = (boundingBox.maxes.x-boundingBox.mins.x)*SCREEN_WIDTH;
  float height = (boundingBox.maxes.y-boundingBox.mins.y)*SCREEN_HEIGHT;
  lightArea = std::max<float>(width, height);

  // clip bounding box 
  boundingBox.mins.Clamp(Vector3(0.0f, 0.0f, 0.0f), Vector3(1.0f, 1.0f, 1.0f));
  boundingBox.maxes.Clamp(Vector3(0.0f, 0.0f, 0.0f), Vector3(1.0f, 1.0f, 1.0f));

  lightBD.mins.Set(boundingBox.mins);
  lightBD.maxes.Set(boundingBox.maxes);
}

void PointLight::CalculateTiledShadowMatrices()
{
  // get corresponding tile in tiled shadow map
  Tile tile;
  bool result = tiledDeferredPP->tileMap.GetTile(lightArea, tile);
  assert(result);

  Matrix4 shadowTexMatrices[4];
  shadowTexMatrices[0].Set(tile.size, 0.0f, 0.0f, 0.0f, 0.0f, tile.size*0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 
                           tile.position.x, tile.position.y-(tile.size*0.5f), 0.0f, 1.0f);
  shadowTexMatrices[1].Set(tile.size*0.5f, 0.0f, 0.0f, 0.0f, 0.0f, tile.size, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 
                           tile.position.x+(tile.size*0.5f), tile.position.y, 0.0f, 1.0f);
  shadowTexMatrices[2].Set(tile.size, 0.0f, 0.0f, 0.0f, 0.0f, tile.size*0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 
                           tile.position.x, tile.position.y+(tile.size*0.5f), 0.0f, 1.0f);
  shadowTexMatrices[3].Set(tile.size*0.5f, 0.0f, 0.0f, 0.0f, 0.0f, tile.size, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 
                           tile.position.x-(tile.size*0.5f), tile.position.y, 0.0f, 1.0f);

  Matrix4 shadowTransMatrix, shadowViewMatrix;
  shadowTransMatrix.SetTranslation(-lightBD.position);
  for(unsigned int i=0; i<4; i++)
  {
    shadowViewMatrix = tiledShadowRotMatrices[i]*shadowTransMatrix;
    unsigned int index = i & 1;
    tiledShadowBD.shadowViewProjTexMatrices[i] = shadowTexMatrices[i]*tiledShadowProjMatrices[index]*shadowViewMatrix;
  }
}

void PointLight::CalculateCubeShadowMatrices()
{
  Matrix4 shadowTransMatrix, shadowViewMatrix;
  shadowTransMatrix.SetTranslation(-lightBD.position);
  for(unsigned int i=0; i<6; i++)
  {
    shadowViewMatrix = cubeShadowRotMatrices[i]*shadowTransMatrix;
    cubeShadowBD.shadowViewProjMatrices[i] = cubeShadowProjMatrix*shadowViewMatrix;
    cubeFrustums[i].Update(cubeShadowBD.shadowViewProjMatrices[i]);
  }
}

void PointLight::Update(unsigned int lightIndex)
{	
  this->lightIndex = lightIndex;

  if(tiledDeferredPP->shadowMode == TILED_SHADOW_SM)
    CalculateTiledShadowMatrices();
  else if(tiledDeferredPP->shadowMode == CUBE_SHADOW_SM)
  {
    lightIndexUB->Update(&this->lightIndex);
    CalculateCubeShadowMatrices();
  }
}

void PointLight::SetupCubeShadowMapSurface(DrawCmd &drawCmd, unsigned int faceIndex)
{
  assert(faceIndex < 6); 
  drawCmd.renderTarget = tiledDeferredPP->cubeShadowMapRT; 
  drawCmd.rasterizerState = backCullRS;
  drawCmd.depthStencilState = defaultDSS;
  drawCmd.blendState = noColorWriteBS;
  drawCmd.customUBs[0] = lightIndexUB;
  drawCmd.customUBs[1] = tiledDeferredPP->faceIndexUBs[faceIndex];
  drawCmd.customSBs[0] = tiledDeferredPP->lightSBs[CUBE_SHADOW_SM];
  drawCmd.shader = cubeShadowMapShader;
}
