#ifndef GPU_CMD_H
#define GPU_CMD_H

#include <render_states.h>

class OGL_VertexLayout;
class OGL_VertexBuffer;
class OGL_IndexBuffer;
class OGL_UniformBuffer;
class OGL_StructuredBuffer;
class RenderTargetConfig;
class OGL_RenderTarget;
class OGL_RasterizerState;
class OGL_DepthStencilState;
class OGL_BlendState;
class OGL_Texture;
class OGL_Sampler;
class OGL_Shader;
class OGL_TimerQueryObject;
class Camera;

enum gpuCmdOrders
{
  BASE_CO=0, // fill GBuffers
  PRE_SHADOW_CO, // generate tile-info and indirect-draw list
  SHADOW_BEGIN_TIMER_QUERY_CO, // begin querying shadow-map generation times
  SHADOW_CO, // generate shadow maps
  SHADOW_END_TIMER_QUERY_CO, // end querying shadow-map generation times
  ILLUM_BEGIN_TIMER_QUERY_CO, // begin querying illumination times
  ILLUM_CO, // illumination
  ILLUM_END_TIMER_QUERY_CO, // end querying illumination times
  SKY_CO, // render sky
  POST_PROCESS_CO, // perform post-processing
  GUI_CO // render GUI
};

enum gpuCmdModes
{
  DRAW_CM=0, // draw-call
  INDIRECT_DRAW_CM, // indirect draw-call
  COMPUTE_CM, // dispatch of compute-shader
  TIMER_QUERY_CM, // timer query
};

struct ShaderCmd
{
  OGL_RenderTarget *renderTarget;
  RenderTargetConfig *renderTargetConfig;
  Camera *camera;
  OGL_Texture *textures[NUM_TEXTURE_BP];
  OGL_Sampler *samplers[NUM_TEXTURE_BP];    
  OGL_UniformBuffer *customUBs[NUM_CUSTOM_UNIFORM_BUFFER_BP];
  OGL_StructuredBuffer *customSBs[NUM_STRUCTURED_BUFFER_BP];
  OGL_Shader *shader;
};

struct BaseDrawCmd: public ShaderCmd
{
  OGL_VertexLayout *vertexLayout;
  OGL_VertexBuffer *vertexBuffer;
  OGL_IndexBuffer *indexBuffer;
  OGL_RasterizerState *rasterizerState;
  OGL_DepthStencilState *depthStencilState;
  OGL_BlendState *blendState;
  primitiveTypes primitiveType; 
};

struct DrawCmd: public BaseDrawCmd
{
  unsigned int firstIndex;
  unsigned int numElements; 
  unsigned int numInstances;
};

struct IndirectDrawCmd: public BaseDrawCmd
{
  unsigned int stride;
  unsigned int maxDrawCount;
};

struct ComputeCmd: public ShaderCmd
{
  unsigned int numThreadGroupsX;
  unsigned int numThreadGroupsY;
  unsigned int numThreadGroupsZ;
};

struct TimerQueryCmd
{
  OGL_TimerQueryObject *object;
  timerQueryModes mode;
};

// GpuCmd
//
class GpuCmd
{
public:
  friend class OGL_Renderer;

  explicit GpuCmd(gpuCmdModes mode)
  {
    Reset(mode);
  }

  void Reset(gpuCmdModes mode)
  {
    memset(this, 0, sizeof(GpuCmd));
    this->mode = mode;
    if((mode == DRAW_CM) || (mode == INDIRECT_DRAW_CM))
      draw.primitiveType = TRIANGLES_PRIMITIVE;
    if(mode == DRAW_CM)
      draw.numInstances = 1;
  }

  gpuCmdModes GetMode() const
  {
    return mode;
  }

  unsigned int GetID() const
  {
    return ID;
  }

  gpuCmdOrders order;
  union
  {
    DrawCmd draw;
    IndirectDrawCmd indirectDraw;
    ComputeCmd compute;
    TimerQueryCmd timerQuery;
  };

private:
  gpuCmdModes mode;
  unsigned int ID; 

};

#endif


