#ifndef POINT_LIGHT_H
#define POINT_LIGHT_H

#include <Frustum.h>

struct DrawCmd;
class OGL_UniformBuffer;
class OGL_Shader;
class OGL_RasterizerState;
class OGL_DepthStencilState;
class OGL_BlendState;
class TiledDeferred;

// PointLight
//
class PointLight
{
public:
  friend class TiledDeferred;

  struct LightBufferData
  {
    LightBufferData():
      radius(0.0f)
    {
    }

    Vector3 position;
    float radius;
    Color color;
    Vector4 mins;
    Vector4 maxes;
  };

  struct TiledShadowBufferData
  {
    Matrix4 shadowViewProjTexMatrices[4];
  };

  struct CubeShadowBufferData
  {
    Matrix4 shadowViewProjMatrices[6];
  };

  PointLight():
    tiledDeferredPP(NULL),
    cubeShadowMapShader(NULL),
    lightIndexUB(NULL),
    backCullRS(NULL),
    defaultDSS(NULL),
    noColorWriteBS(NULL),
    lightArea(0.0f), 
    lightIndex(0),
    active(true),  
    visible(false)  
  {
  }

  bool Create(const Vector3 &position, float radius, const Color &color);

  void CalculateVisibility();

  void Update(unsigned int lightIndex);

  void SetupCubeShadowMapSurface(DrawCmd &drawCmd, unsigned int faceIndex);

  void SetPosition(const Vector3 &position)
  {
    lightBD.position = position;
  }

  Vector3 GetPosition() const
  {
    return lightBD.position;
  }

  void SetRadius(float radius)
  {
    lightBD.radius = radius;
  }

  float GetRadius() const
  {
    return lightBD.radius;
  }

  void SetColor(const Color &color)
  {
    lightBD.color = color;
  }

  Color GetColor() const
  {
    return lightBD.color;
  }

  const Frustum& GetFrustum(unsigned int faceIndex) const
  {
    assert(faceIndex < 6);
    return cubeFrustums[faceIndex];
  }

  const void* GetLightBufferData() const
  {
    return &lightBD;
  }

  const void* GetTiledShadowBufferData() const
  {
    return &tiledShadowBD;
  }

  const void* GetCubeShadowBufferData() const
  {
    return &cubeShadowBD;
  }

  float GetLightArea() const
  {
    return (active ? lightArea : -1.0f);
  }

  unsigned int GetIndex() const
  {
    return lightIndex;
  }

  void SetActive(bool active) 
  {
    this->active = active;
  }

  bool IsActive() const
  {
    return active;
  }

  bool IsVisible() const
  {
    return visible; 
  }
  
private:  
  void CalculateTiledShadowMatrices();

  void CalculateCubeShadowMatrices();

  // data for light structured-buffers
  LightBufferData lightBD;
  TiledShadowBufferData tiledShadowBD;
  CubeShadowBufferData cubeShadowBD;

  TiledDeferred *tiledDeferredPP;

  OGL_Shader *cubeShadowMapShader;
  OGL_UniformBuffer *lightIndexUB;
  OGL_RasterizerState *backCullRS;
  OGL_DepthStencilState *defaultDSS;
  OGL_BlendState *noColorWriteBS;
  Matrix4 tiledShadowProjMatrices[2], tiledShadowRotMatrices[4];
  Matrix4 cubeShadowProjMatrix, cubeShadowRotMatrices[6];
  Frustum cubeFrustums[6];
  float lightArea;
  unsigned int lightIndex;
  bool active; 
  bool visible;
 
};

#endif
