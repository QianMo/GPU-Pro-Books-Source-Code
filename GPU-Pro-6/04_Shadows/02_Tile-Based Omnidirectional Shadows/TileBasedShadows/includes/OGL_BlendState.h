#ifndef OGL_BLEND_STATE_H
#define OGL_BLEND_STATE_H

#include <render_states.h>

// descriptor for setting up OGL_BlendState
struct BlendDesc
{
  BlendDesc():
    srcColorBlend(ONE_BLEND),
    dstColorBlend(ONE_BLEND),
    blendColorOp(ADD_BLEND_OP),
    srcAlphaBlend(ONE_BLEND),
    dstAlphaBlend(ONE_BLEND),
    blendAlphaOp(ADD_BLEND_OP),
    colorMask(ALL_COLOR_MASK),
    blend(false)
  {
  }

  bool operator== (const BlendDesc &desc) const
  {
    if(srcColorBlend != desc.srcColorBlend)
      return false;
    if(dstColorBlend != desc.dstColorBlend)
      return false;
    if(blendColorOp != desc.blendColorOp)
      return false;
    if(srcAlphaBlend != desc.srcAlphaBlend)
      return false;
    if(dstAlphaBlend != desc.dstAlphaBlend)
      return false;
    if(blendAlphaOp != desc.blendAlphaOp)
      return false;	
    if(constBlendColor != desc.constBlendColor)
      return false;
    if(colorMask != desc.colorMask)
      return false;
    if(blend != desc.blend)
      return false;
    return true;
  }

  blendOptions srcColorBlend;
  blendOptions dstColorBlend;
  blendOps blendColorOp;
  blendOptions srcAlphaBlend;
  blendOptions dstAlphaBlend;
  blendOps blendAlphaOp;	
  Vector4 constBlendColor;
  unsigned char colorMask;
  bool blend;
};

// OGL_BlendState
//
class OGL_BlendState
{
public:
  bool Create(const BlendDesc &desc);

  void Set() const;

  const BlendDesc& GetDesc() const
  {
    return desc;
  }

private:
  BlendDesc desc;

};

#endif