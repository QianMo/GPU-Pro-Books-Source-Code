#ifndef OGL_RASTERIZER_STATE_H
#define OGL_RASTERIZER_STATE_H

#include <render_states.h>

// descriptor for setting up OGL_RasterizerState
struct RasterizerDesc
{
  RasterizerDesc():
    fillMode(SOLID_FILL),
    cullMode(NONE_CULL),
    depthBias(0.0f),
    slopeScaledDepthBias(0.0f),
    numClipPlanes(0),
    scissorTest(false),
    multisampleEnable(false)
  {
  }

  bool operator== (const RasterizerDesc &desc) const
  {
    if(fillMode != desc.fillMode)
      return false;
    if(cullMode != desc.cullMode)
      return false;
    if(!IS_EQUAL(depthBias, desc.depthBias))
      return false;
    if(!IS_EQUAL(slopeScaledDepthBias, desc.slopeScaledDepthBias))
      return false;
    if(numClipPlanes != desc.numClipPlanes)
      return false;
    if(scissorTest != desc.scissorTest)
      return false;
    if(multisampleEnable != desc.multisampleEnable)
      return false;
    return true;
  }

  fillModes fillMode;
  cullModes cullMode;
  float depthBias;
  float slopeScaledDepthBias;
  unsigned int numClipPlanes;
  bool scissorTest;
  bool multisampleEnable;
};

// OGL_RasterizerState
//
class OGL_RasterizerState
{
public:
  bool Create(const RasterizerDesc &desc);

  void Set() const;

  const RasterizerDesc& GetDesc() const
  {
    return desc;
  }

private:
  RasterizerDesc desc;

};

#endif