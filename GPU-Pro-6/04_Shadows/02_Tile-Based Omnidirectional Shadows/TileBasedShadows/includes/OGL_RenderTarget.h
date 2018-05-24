#ifndef OGL_RENDER_TARGET_H
#define OGL_RENDER_TARGET_H

#include <render_states.h>

// max number of color-buffers, which can be attached to 1 render-target
#define MAX_NUM_COLOR_BUFFERS 8

enum renderTargetFlags
{
  CUBEMAP_RTF=1, // cubemap render-target
  SRGB_READ_RTF=2 // convert color from SRGB to linear space
};

class OGL_Texture;
class OGL_Sampler;
class RenderTargetConfig;

// descriptor for a render-target buffer
struct RtBufferDesc
{
  RtBufferDesc():
    format(TEX_FORMAT_NONE),
    rtFlags(0)
  {
  }

  texFormats format;
  unsigned int rtFlags;
};

// descriptor for setting up OGL_RenderTarget
struct RenderTargetDesc
{
  RenderTargetDesc():
    width(0),
    height(0),
    depth(1)
  {
  }

  unsigned int CalcNumColorBuffers() const
  {
    unsigned int numColorBuffers = 0;
    for(unsigned int i=0; i<MAX_NUM_COLOR_BUFFERS; i++)
    {
      if(colorBufferDescs[i].format != TEX_FORMAT_NONE)
        numColorBuffers++;
      else
        break;
    }
    return numColorBuffers;
  }

  unsigned int width, height, depth; 
  RtBufferDesc colorBufferDescs[MAX_NUM_COLOR_BUFFERS];
  RtBufferDesc depthStencilBufferDesc;
}; 

// OGL_RenderTarget
//
// Render-target to render/ write into. Can be configured via RenderTargetConfig for each draw-call/ dispatch.
class OGL_RenderTarget
{
public:
  OGL_RenderTarget():
    width(0),
    height(0),
    depth(0),	
    numColorBuffers(0),
    clearMask(0),
    frameBufferName(0),
    frameBufferTextures(NULL),
    depthStencilTexture(NULL),
    clearTarget(true),
    isBackBuffer(false)
  {
  }

  ~OGL_RenderTarget()
  {
    Release();
  }

  void Release();

  bool Create(const RenderTargetDesc &desc);

  bool CreateBackBuffer();

  void Bind(const RenderTargetConfig *rtConfig=NULL);

  // indicate, that render-target should be cleared
  void Reset()
  {
    clearTarget = true;
  }

  void Clear(unsigned int newClearMask) const;

  OGL_Texture* GetTexture(unsigned int index=0) const
  {
    assert((index < numColorBuffers) && (!isBackBuffer));
    return &frameBufferTextures[index];
  }

  OGL_Texture* GetDepthStencilTexture() const
  {
    return depthStencilTexture;
  }

  unsigned int GetWidth() const
  {
    return width;
  }

  unsigned int GetHeight() const
  {
    return height;
  }

  unsigned int GetDepth() const
  {
    return depth;
  }

private:	
  unsigned int width, height, depth;
  unsigned int numColorBuffers;
  unsigned int clearMask;
  GLuint frameBufferName; 
  OGL_Texture *frameBufferTextures;
  OGL_Texture *depthStencilTexture;
  bool clearTarget;	
  bool isBackBuffer;

};

#endif