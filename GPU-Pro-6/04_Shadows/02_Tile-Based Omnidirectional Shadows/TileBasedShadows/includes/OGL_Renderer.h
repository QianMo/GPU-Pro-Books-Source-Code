#ifndef OGL_RENDERER_H
#define OGL_RENDERER_H

#include <vertex_types.h>
#include <List.h>
#include <GpuCmd.h>
#include <OGL_Sampler.h>
#include <OGL_RasterizerState.h>
#include <OGL_DepthStencilState.h>
#include <OGL_BlendState.h>
#include <RenderTargetConfig.h>
#include <OGL_RenderTarget.h>
#include <OGL_VertexLayout.h>
#include <OGL_VertexBuffer.h>
#include <OGL_IndexBuffer.h>
#include <OGL_UniformBuffer.h>
#include <OGL_StructuredBuffer.h>
#include <OGL_TimerQueryObject.h>
#include <OGL_Shader.h>
#include <Camera.h>
#include <IPostProcessor.h>

#define CLEAR_COLOR Vector4(0.0f, 0.0f, 0.0f, 0.0f) // render-target clear color
#define CLEAR_DEPTH 1.0f // render-target clear depth
#define CLEAR_STENCIL 0 // render-target clear stencil 

// macro for passing int offset-values for OpenGL buffers
#define INT_BUFFER_OFFSET(i) ((int*)NULL+(i)) 

// renderer related pool sizes
enum rendererPools
{
  GEOMETRY_VERTEX_POOL=200000,
  GEOMETRY_INDEX_POOL=850000,
  GPU_CMD_POOL=60000
};

// predefined IDs for frequently used samplers
enum samplerID
{
  POINT_SAMPLER_ID=0,
  LINEAR_SAMPLER_ID,
  TRILINEAR_SAMPLER_ID,
  SHADOW_MAP_SAMPLER_ID
};

// predefined IDs for frequently used render-targets
enum renderTargetID
{
  BACK_BUFFER_RT_ID=0, // back buffer
  GBUFFERS_RT_ID // geometry buffers
};

// predefined IDs for frequently used vertex-layouts
enum vertexLayoutID
{
  GEOMETRY_VL_ID=0,
  SHADOW_VL_ID
};

// predefined IDs for frequently used vertex-buffers
enum vertexBufferID
{
  GEOMETRY_VB_ID=0
};

// predefined IDs for frequently used index-buffers
enum indexBufferID
{
  GEOMETRY_IB_ID=0
};

// predefined IDs for frequently used cameras 
enum cameraID
{
  MAIN_CAMERA_ID
};

// OGL_Renderer
//
// Manages OpenGL 4.4 rendering.
class OGL_Renderer
{
public:
  OGL_Renderer():
    hDC(NULL),
    hRC(NULL),
    noneCullRS(NULL),
    noDepthTestDSS(NULL),
    defaultBS(NULL), 
    gpuAMD(false),
    lastGpuCmd(DRAW_CM),
    numShadowDrawCalls(0)
  {
  }

  ~OGL_Renderer()
  {
    Destroy();
  }

  void Destroy();	

  bool Create();

  OGL_Sampler* CreateSampler(const SamplerDesc &desc);

  OGL_Sampler* GetSampler(unsigned int ID) const
  {
    assert(ID < samplers.GetSize());
    return samplers[ID];
  }

  OGL_RasterizerState* CreateRasterizerState(const RasterizerDesc &desc);

  OGL_DepthStencilState* CreateDepthStencilState(const DepthStencilDesc &desc);

  OGL_BlendState* CreateBlendState(const BlendDesc &desc);

  RenderTargetConfig* CreateRenderTargetConfig(const RtConfigDesc &desc);

  OGL_RenderTarget* CreateRenderTarget(const RenderTargetDesc &desc);

  OGL_RenderTarget* CreateBackBufferRt();

  OGL_RenderTarget* GetRenderTarget(unsigned int ID) const
  {
    assert(ID < renderTargets.GetSize());
    return renderTargets[ID];
  }

  OGL_VertexLayout* CreateVertexLayout(const VertexElementDesc *vertexElementDescs, unsigned int numVertexElementDescs);

  OGL_VertexLayout* GetVertexLayout(unsigned int ID) const
  {
    assert(ID < vertexLayouts.GetSize());
    return vertexLayouts[ID];
  }

  OGL_VertexBuffer* CreateVertexBuffer(unsigned int vertexSize, unsigned int maxVertexCount, bool dynamic);

  OGL_VertexBuffer* GetVertexBuffer(unsigned int ID) const
  {
    assert(ID < vertexBuffers.GetSize());
    return vertexBuffers[ID];
  }

  OGL_IndexBuffer* CreateIndexBuffer(unsigned int maxIndexCount, bool dynamic);

  OGL_IndexBuffer* GetIndexBuffer(unsigned int ID) const
  {
    assert(ID < indexBuffers.GetSize());
    return indexBuffers[ID];
  }

  OGL_UniformBuffer* CreateUniformBuffer(unsigned int bufferSize);

  OGL_StructuredBuffer* CreateStructuredBuffer(unsigned int elementCount, unsigned int elementSize, unsigned int flags=0);

  OGL_TimerQueryObject* CreateTimerQueryObject();

  Camera* CreateCamera(float fovy, float aspectRatio, float nearClipDistance, float farClipDistance);

  Camera* GetCamera(unsigned int ID) const
  {
    assert(ID < cameras.GetSize());
    return cameras[ID];
  }

  template<class T> T* CreatePostProcessor()
  {
    T *postProcessor = new T;
    if(!postProcessor)
      return NULL;
    if(!postProcessor->Create())
    {
      SAFE_DELETE(postProcessor);
      return NULL;
    }
    postProcessors.AddElement((IPostProcessor**)(&postProcessor));
    return postProcessor;
  }

  IPostProcessor* GetPostProcessor(const char *name) const;

  bool UpdateBuffers(bool dynamic);

  void SetupPostProcessSurface(DrawCmd &drawCmd);

  // add new GPU command per frame 
  void AddGpuCmd(const GpuCmd &cmd);

  void ClearFrame();

  // execute all GPU commands, which have been passed per frame to renderer
  void ExecuteGpuCmds();

  // save a BMP screen-shot
  void SaveScreenshot() const;

  bool IsGpuAMD() const
  {
    return gpuAMD;
  }

  unsigned int GetNumShadowDrawCalls() const
  {
    return numShadowDrawCalls;
  }

private:  
  // init OpenGL extensions
  bool InitGLExtensions(); 

  // create frequently used objects
  bool CreateDefaultObjects();

  void ExecutePostProcessors();

  // set draw states for passed draw command
  void SetDrawStates(const BaseDrawCmd &cmd);

  // set shader states for passed shader command
  void SetShaderStates(const ShaderCmd &cmd);

  // set shader params for passed shader command
  void SetShaderParams(const ShaderCmd &cmd);
  
  void Draw(const DrawCmd &cmd);

  void DrawIndirect(const IndirectDrawCmd &cmd);

  void Dispatch(const ComputeCmd &cmd); 

  void TimerQuery(const TimerQueryCmd &cmd);

  // list of all samplers
  List<OGL_Sampler*> samplers;

  // list of all rasterizer states
  List<OGL_RasterizerState*> rasterizerStates;

  // list of all depth-stencil states
  List<OGL_DepthStencilState*> depthStencilStates;

  // list of all blend states
  List<OGL_BlendState*> blendStates;

  // list of all render-target configs
  List<RenderTargetConfig*> renderTargetConfigs;

  // list of all render-targets
  List<OGL_RenderTarget*> renderTargets;

  // list of all vertex layouts
  List<OGL_VertexLayout*> vertexLayouts;

  // list of all vertex buffers
  List<OGL_VertexBuffer*> vertexBuffers;

  // list of all index buffers
  List<OGL_IndexBuffer*> indexBuffers;

  // list of all uniform buffers
  List<OGL_UniformBuffer*> uniformBuffers;

  // list of all structured buffers
  List<OGL_StructuredBuffer*> structuredBuffers;

  // list of all timer-query objects
  List<OGL_TimerQueryObject*> timerQueryObjects;

  // list of all cameras
  List<Camera*> cameras;	

  // list of all post-processors
  List<IPostProcessor*> postProcessors;

  // list of all per frame passed GPU commands
  List<GpuCmd> gpuCmds;

  HDC hDC; // device context
  HGLRC hRC; // GL rendering context

  // render-states, frequently used by post-processors
  OGL_RasterizerState *noneCullRS;
  OGL_DepthStencilState *noDepthTestDSS;
  OGL_BlendState *defaultBS;

  // helper variables
  bool gpuAMD;
  GpuCmd lastGpuCmd;
  unsigned int numShadowDrawCalls;

};

#endif 