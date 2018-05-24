#ifndef RENDER_TARGET_CONFIG_H
#define RENDER_TARGET_CONFIG_H

class OGL_RenderTarget;
class OGL_StructuredBuffer;

enum rtConfigFlags
{
  COMPUTE_RTCF=1, // compute shader used
  SRGB_WRITE_RTCF=2 // convert color from linear into SRGB space
};

// descriptor for setting up RenderTargetConfig
struct RtConfigDesc
{
  RtConfigDesc():
    firstColorBufferIndex(0),
    numColorBuffers(1),
    numStructuredBuffers(0),
    flags(0)
  {
    memset(structuredBuffers, 0, sizeof(OGL_StructuredBuffer*)*NUM_STRUCTURED_BUFFER_BP);
  }

  bool operator== (const RtConfigDesc &desc) const
  {
    if(firstColorBufferIndex != desc.firstColorBufferIndex)
      return false;
    if(numColorBuffers != desc.numColorBuffers)
      return false;
    if(numStructuredBuffers != desc.numStructuredBuffers)
      return false;
    if(flags != desc.flags)
      return false;
    for(unsigned int i=0; i<NUM_STRUCTURED_BUFFER_BP; i++)
    {
      if(structuredBuffers[i] != desc.structuredBuffers[i])
        return false;
    }
    return true;
  }

  unsigned int firstColorBufferIndex; // index of first render-target to render into 
  unsigned int numColorBuffers; // number of render-targets to render into
  unsigned int numStructuredBuffers; // number of structured-buffers to write into
  OGL_StructuredBuffer *structuredBuffers[NUM_STRUCTURED_BUFFER_BP]; // structured-buffers to write into
  unsigned int flags;
};

// RenderTargetConfig
//
// Offers possibility to configure a render-target.
class RenderTargetConfig
{
public:
  bool Create(const RtConfigDesc &desc)
  {
    this->desc = desc;
    return true;
  }

  const RtConfigDesc& GetDesc() const
  {
    return desc;
  }

private:
  RtConfigDesc desc;

};

#endif