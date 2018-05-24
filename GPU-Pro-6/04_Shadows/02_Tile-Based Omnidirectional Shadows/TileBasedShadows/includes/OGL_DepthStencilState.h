#ifndef OGL_DEPTH_STENCIL_STATE_H
#define OGL_DEPTH_STENCIL_STATE_H

#include <render_states.h>

// descriptor for setting up OGL_DepthStencilState
struct DepthStencilDesc
{
  DepthStencilDesc():
    depthFunc(LEQUAL_COMP_FUNC),
    stencilRef(0),
    stencilMask(~0),
    stencilFailOp(KEEP_STENCIL_OP),
    stencilDepthFailOp(INCR_SAT_STENCIL_OP),
    stencilPassOp(INCR_SAT_STENCIL_OP),
    stencilFunc(ALWAYS_COMP_FUNC),
    depthTest(true),
    depthMask(true),
    stencilTest(false)
  {
  }

  bool operator== (const DepthStencilDesc &desc) const
  {
    if(depthFunc != desc.depthFunc)
      return false;
    if(stencilRef != desc.stencilRef)
      return false;
    if(stencilMask != desc.stencilMask)
      return false;
    if(stencilFailOp != desc.stencilFailOp)
      return false;
    if(stencilDepthFailOp != desc.stencilDepthFailOp)
      return false;
    if(stencilPassOp != desc.stencilPassOp)
      return false;
    if(stencilFunc != desc.stencilFunc)
      return false;
    if(depthTest != desc.depthTest)
      return false;
    if(depthMask != desc.depthMask)
      return false;
    if(stencilTest != desc.stencilTest)
      return false;
    return true;
  }

  comparisonFuncs depthFunc;
  unsigned int stencilRef;
  unsigned int stencilMask;
  stencilOps stencilFailOp;
  stencilOps stencilDepthFailOp;
  stencilOps stencilPassOp;
  comparisonFuncs stencilFunc;
  bool depthTest;
  bool depthMask;
  bool stencilTest;
};

// OGL_DepthStencilState
//
class OGL_DepthStencilState
{
public:
  bool Create(const DepthStencilDesc &desc);

  void Set() const;

  const DepthStencilDesc& GetDesc() const
  {
    return desc;
  }

private:
  DepthStencilDesc desc;

};

#endif