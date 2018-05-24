#include <stdafx.h>
#include <Demo.h>
#include <OGL_Renderer.h>

void OGL_Renderer::Destroy()
{
  SAFE_DELETE_PLIST(samplers);
  SAFE_DELETE_PLIST(rasterizerStates);
  SAFE_DELETE_PLIST(depthStencilStates);
  SAFE_DELETE_PLIST(blendStates);
  SAFE_DELETE_PLIST(renderTargetConfigs);
  SAFE_DELETE_PLIST(renderTargets);
  SAFE_DELETE_PLIST(vertexLayouts);
  SAFE_DELETE_PLIST(vertexBuffers);
  SAFE_DELETE_PLIST(indexBuffers);
  SAFE_DELETE_PLIST(uniformBuffers);
  SAFE_DELETE_PLIST(structuredBuffers);
  SAFE_DELETE_PLIST(timerQueryObjects);
  SAFE_DELETE_PLIST(cameras);
  SAFE_DELETE_PLIST(postProcessors);

#ifndef UNIX_PORT  

  // release GL rendering context
  if(hRC)
  {
    wglMakeCurrent(NULL, NULL);
    wglDeleteContext(hRC);
    hRC = NULL;
  }

  // release device context
  if(hDC)
  {
    ReleaseDC(Demo::window->GetHWnd(), hDC);
    hDC = NULL;
  }

#else

  SDL_GL_DeleteContext(hRC);
  SDL_DestroyRenderer(hDC);

#endif /* UNIX_PORT */
}

bool OGL_Renderer::InitGLExtensions()
{
  // init GLEW OpenGL Extension Library
  glewExperimental = GL_TRUE;
  GLenum info = glewInit();
  if(info != GLEW_OK)
  {
    MessageBox(NULL, (const char*)glewGetErrorString(info), "Error", MB_OK | MB_ICONEXCLAMATION);
    return false;
  }

  // check, if OpenGL 4.4 is supported
  GLint majorVersion = 0;
  GLint minorVersion = 0;
  glGetIntegerv(GL_MAJOR_VERSION, &majorVersion);
  glGetIntegerv(GL_MINOR_VERSION, &minorVersion);
  int version = (majorVersion*100)+(minorVersion*10);
  if(version < 440)
  {
    MessageBox(NULL, "OpenGL 4.4 not supported!", "Error", MB_OK | MB_ICONEXCLAMATION);
    return false;
  }

  // check, if GL_ARB_shader_draw_parameters is supported
  if(!glewIsSupported("GL_ARB_shader_draw_parameters "))
  {
    MessageBox(NULL, "GL_ARB_shader_draw_parameters not supported!", "Error", MB_OK | MB_ICONEXCLAMATION);
    return false;
  }

  // check, if GL_ARB_indirect_parameters is supported
  if(!glewIsSupported("GL_ARB_indirect_parameters "))
  {
    MessageBox(NULL, "GL_ARB_indirect_parameters not supported!", "Error", MB_OK | MB_ICONEXCLAMATION);
    return false;
  }

  return true;
}

#ifdef _DEBUG
static void GLAPIENTRY DebugOutput(GLenum source, GLenum type, GLuint id, GLenum severity,
                                   GLsizei length, const GLchar* message, GLvoid* userParam)
{
  char debugSource[32] = {0};
  switch(source)
  {
  case GL_DEBUG_SOURCE_API:
    strcpy(debugSource, "OpenGL");
    break;
  case GL_DEBUG_SOURCE_WINDOW_SYSTEM:
    strcpy(debugSource, "Windows");
    break;
  case GL_DEBUG_SOURCE_SHADER_COMPILER:
    strcpy(debugSource, "Shader Compiler");
    break;
  case GL_DEBUG_SOURCE_THIRD_PARTY:
    strcpy(debugSource, "Third Party");
    break;
  case GL_DEBUG_SOURCE_APPLICATION:
    strcpy(debugSource, "Application");
    break;
  case GL_DEBUG_SOURCE_OTHER:
    strcpy(debugSource, "Other");
    break;
  }

  char debugType[32] = {0};
  switch(type)
  {
  case GL_DEBUG_TYPE_ERROR:
    strcpy(debugType, "Error");
    break;
  case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR:
    strcpy(debugType, "Deprecated behavior");
    break;
  case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:
    strcpy(debugType, "Undefined behavior");
    break;
  case GL_DEBUG_TYPE_PORTABILITY:
    strcpy(debugType, "Portability");
    break;
  case GL_DEBUG_TYPE_PERFORMANCE:
    strcpy(debugType, "Performance");
    break;
  case GL_DEBUG_TYPE_OTHER:
    strcpy(debugType, "Message");
    break;
  case GL_DEBUG_TYPE_MARKER:
    strcpy(debugType, "Marker");
    break;
  case GL_DEBUG_TYPE_PUSH_GROUP:
    strcpy(debugType, "Push group");
    break;
  case GL_DEBUG_TYPE_POP_GROUP:
    strcpy(debugType, "Pop group");
    break;
  }

  char severityType[32] = {0};
  switch(severity)
  {
  case GL_DEBUG_SEVERITY_HIGH:
    strcpy(severityType, "[OGL-Error]");
    break;
  case GL_DEBUG_SEVERITY_MEDIUM:
    strcpy(severityType, "[OGL-Warning]");
    break;
  case GL_DEBUG_SEVERITY_LOW:
    strcpy(severityType, "[OGL-Info]");
    break;
  }

  if((id != 131076) && (id != 131184) && (id != 131185) && (id != 131186) && (id != 131188) && (id != 131204) && (id != 131218))
  {
    char buffer[4096];
    sprintf(buffer, "%s %s: %s %d: %s\n", severityType, debugSource, debugType, id, message);
    OutputDebugString(buffer); 
  }
}
#endif

bool OGL_Renderer::Create()
{
#ifndef UNIX_PORT  

  // get device context
  if(!(hDC = GetDC(Demo::window->GetHWnd())))
  {
    MessageBox(NULL, "Failed to create a GL device context!", "ERROR", MB_OK | MB_ICONEXCLAMATION);  
    Destroy();
    return false;
  }

  // choose standard GL pixel-format
  PIXELFORMATDESCRIPTOR pfd;
  memset(&pfd, 0, sizeof(PIXELFORMATDESCRIPTOR));
  pfd.nSize = sizeof(PIXELFORMATDESCRIPTOR);
  pfd.nVersion = 1;
  pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
  pfd.iPixelType = PFD_TYPE_RGBA;
  pfd.cColorBits = 24;
  pfd.cRedBits = 8;
  pfd.cGreenBits = 8;
  pfd.cBlueBits = 8;
  pfd.cAlphaBits = 8;
  pfd.cDepthBits = 24;
  pfd.cStencilBits = 8;
  pfd.iLayerType = PFD_MAIN_PLANE;
  GLuint chosenFormat;
  if(!(chosenFormat = ChoosePixelFormat(hDC,&pfd)))
  {
    MessageBox(NULL, "Failed to find a suitable pixel-format!", "ERROR", MB_OK | MB_ICONEXCLAMATION);  
    Destroy();
    return false; 
  }

  // set standard GL pixel-format
  if(!SetPixelFormat(hDC, chosenFormat, &pfd))
  {
    MessageBox(NULL, "Failed to set the pixel-format!", "ERROR", MB_OK | MB_ICONEXCLAMATION);  
    Destroy();
    return false;
  }

  // create GL rendering context
  if(!(hRC = wglCreateContext(hDC)))
  {
    MessageBox(NULL, "Failed to create a GL rendering context!", "ERROR", MB_OK | MB_ICONEXCLAMATION);  
    Destroy();
    return false;
  }

  // make GL rendering context current
  if(!wglMakeCurrent(hDC, hRC))
  {
    MessageBox(NULL, "Failed to make GL rendering context current!", "ERROR", MB_OK | MB_ICONEXCLAMATION); 
    Destroy();
    return false;
  }

  // init wgl-Extensions
  if(!InitWGLExtensions(hDC))
  {
    MessageBox(NULL, "Failed to init wgl-Extensions!", "ERROR", MB_OK | MB_ICONEXCLAMATION); 
    Destroy();
    return false;
  }

  // toggle vertical synchronization
  wglSwapIntervalEXT((VSYNC_ENABLED > 0) ? 1 : 0); 

#else
 
  hDC = SDL_CreateRenderer(Demo::window->GetHWnd(), -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_TARGETTEXTURE);
  if(!hDC)
  {
    MessageBox(NULL, "Failed to create a GL device context!", "ERROR", MB_OK | MB_ICONEXCLAMATION);
    Destroy();
    return false;
  }

  hRC = SDL_GL_CreateContext(Demo::window->GetHWnd());
  if(!hRC)
  {
    MessageBox(NULL, "Failed to create a GL rendering context!", "ERROR", MB_OK | MB_ICONEXCLAMATION);
    Destroy();
    return false;
  }
  
  SDL_GL_SetSwapInterval((VSYNC_ENABLED > 0) ? 1 : 0); 
  
#endif /* UNIX_PORT */

  // check whether current GPU is AMD
  const char *vendorString = (const char*)glGetString(GL_VENDOR);
  gpuAMD = (strstr(vendorString, "ATI") != NULL);
  
  // init gl-Extensions
  if(!InitGLExtensions())
  {
    MessageBox(NULL, "Failed to init gl-Extensions!", "ERROR", MB_OK | MB_ICONEXCLAMATION); 
    Destroy();
    return false;
  }

  // setup debug output
#ifdef _DEBUG
  glEnable(GL_DEBUG_OUTPUT);
  glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
  glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, NULL, GL_TRUE);
  glDebugMessageCallback(DebugOutput, NULL);
#else
  glDisable(GL_DEBUG_OUTPUT);
  glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS); // necessary, since otherwise NV OpenGL driver (340.52) crashes
#endif

  // set OpenGL states
  glClearColor(CLEAR_COLOR.x, CLEAR_COLOR.y, CLEAR_COLOR.z, CLEAR_COLOR.w);
  glClearDepth(CLEAR_DEPTH);
  glClearStencil(CLEAR_STENCIL);
  glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);

  if(!CreateDefaultObjects())
    return false;

  // pre-allocate some GPU commands, to prevent initial stutters
  gpuCmds.Resize(GPU_CMD_POOL);

  return true;
}

bool OGL_Renderer::CreateDefaultObjects()
{
  // create frequently used samplers

  // POINT_SAMPLER
  SamplerDesc samplerDesc;
  samplerDesc.filter = MIN_MAG_POINT_FILTER;
  if(!CreateSampler(samplerDesc))
    return false;

  // LINEAR_SAMPLER
  if(!CreateSampler(SamplerDesc()))
    return false;

  // TRILINEAR_SAMPLER
  samplerDesc.filter = MIN_MAG_MIP_LINEAR_FILTER;
  samplerDesc.adressU = REPEAT_TEX_ADDRESS;
  samplerDesc.adressV = REPEAT_TEX_ADDRESS;
  samplerDesc.adressW = REPEAT_TEX_ADDRESS;
  if(!CreateSampler(samplerDesc))
    return false;

  // SHADOW_MAP_SAMPLER
  samplerDesc.filter = COMP_MIN_MAG_LINEAR_FILTER;
  samplerDesc.adressU = CLAMP_TEX_ADRESS;
  samplerDesc.adressV = CLAMP_TEX_ADRESS;
  samplerDesc.adressW = CLAMP_TEX_ADRESS;
  samplerDesc.compareFunc = LEQUAL_COMP_FUNC;
  if(!CreateSampler(samplerDesc))
    return false;


  // create frequently used render-targets

  // BACK_BUFFER_RT 
  if(!CreateBackBufferRt())
    return false;

  {
    // GBUFFERS_RT	
    // 1. frameBufferTextures[0]:
    //    accumulation buffer 
    // 2. frameBufferTextures[1]:
    //    RGB-channel: albedo, Alpha-channel: specular intensity
    // 3. frameBufferTextures[2]:
    //    RGB-channel: normal, Alpha-channel: unused 
    RenderTargetDesc rtDesc;
    rtDesc.width = SCREEN_WIDTH;
    rtDesc.height = SCREEN_HEIGHT;
    rtDesc.colorBufferDescs[0].format = TEX_FORMAT_RGBA16F;
    rtDesc.colorBufferDescs[1].format = TEX_FORMAT_RGBA8;
    rtDesc.colorBufferDescs[1].rtFlags = SRGB_READ_RTF;
    rtDesc.colorBufferDescs[2].format = TEX_FORMAT_RGB10A2;
    rtDesc.depthStencilBufferDesc.format = TEX_FORMAT_DEPTH24_STENCIL8; 
    if(!CreateRenderTarget(rtDesc))
      return false;
  }


  // create frequently used vertex-layouts
  {
    // GEOMETRY_VL
    VertexElementDesc vertexElementDescs[4] = { POSITION_ATTRIB, R32G32B32_FLOAT_EF, 0,
                                                TEXCOORDS_ATTRIB, R32G32_FLOAT_EF, 12,
                                                NORMAL_ATTRIB, R32G32B32_FLOAT_EF, 20,
                                                TANGENT_ATTRIB, R32G32B32A32_FLOAT_EF, 32 };
    if(!Demo::renderer->CreateVertexLayout(vertexElementDescs, 4))
      return false;
  }

  {
    // SHADOW_VL
    VertexElementDesc vertexElementDescs[1] = { POSITION_ATTRIB, R32G32B32_FLOAT_EF, 0 };
    if(!Demo::renderer->CreateVertexLayout(vertexElementDescs, 1))
      return false;
  }


  // create frequently used vertex-buffers

  // GEOEMTRY_VB
  if(!Demo::renderer->CreateVertexBuffer(sizeof(GeometryVertex), GEOMETRY_VERTEX_POOL, false))
    return false;


  // create frequently used index-buffers

  // GEOMETRY_IB
  if(!Demo::renderer->CreateIndexBuffer(GEOMETRY_INDEX_POOL, false))
    return false;


  // create render-states, frequently used by post-processors
  RasterizerDesc rasterDesc;
  noneCullRS = CreateRasterizerState(rasterDesc);
  if(!noneCullRS)
    return false;

  DepthStencilDesc depthStencilDesc;
  depthStencilDesc.depthTest = false;
  depthStencilDesc.depthMask = false;
  noDepthTestDSS = CreateDepthStencilState(depthStencilDesc);
  if(!noDepthTestDSS)
    return false;

  BlendDesc blendDesc;
  defaultBS = CreateBlendState(blendDesc);
  if(!defaultBS)
    return false;


  // create frequently used cameras

  // MAIN_CAMERA
  if(!CreateCamera(80.0f, (float)SCREEN_WIDTH/(float)SCREEN_HEIGHT, 2.0f, 10000.0f))
    return false;

  return true;
}

OGL_Sampler* OGL_Renderer::CreateSampler(const SamplerDesc &desc)
{
  for(unsigned int i=0; i<samplers.GetSize(); i++)
  {
    if(samplers[i]->GetDesc() == desc)
      return samplers[i];
  }
  OGL_Sampler *sampler = new OGL_Sampler;
  if(!sampler)
    return NULL;
  if(!sampler->Create(desc))
  {
    SAFE_DELETE(sampler);
    return NULL;
  }
  samplers.AddElement(&sampler);
  return sampler;
}

OGL_RasterizerState* OGL_Renderer::CreateRasterizerState(const RasterizerDesc &desc)
{
  for(unsigned int i=0; i<rasterizerStates.GetSize(); i++)
  {
    if(rasterizerStates[i]->GetDesc() == desc)
      return rasterizerStates[i];
  }
  OGL_RasterizerState *rasterizerState = new OGL_RasterizerState;
  if(!rasterizerState)
    return NULL;
  if(!rasterizerState->Create(desc))
  {
    SAFE_DELETE(rasterizerState);
    return NULL;
  }
  rasterizerStates.AddElement(&rasterizerState);
  return rasterizerState;
}

OGL_DepthStencilState* OGL_Renderer::CreateDepthStencilState(const DepthStencilDesc &desc)
{
  for(unsigned int i=0; i<depthStencilStates.GetSize(); i++)
  {
    if(depthStencilStates[i]->GetDesc() == desc)
      return depthStencilStates[i];
  }
  OGL_DepthStencilState *depthStencilState = new OGL_DepthStencilState;
  if(!depthStencilState)
    return NULL;
  if(!depthStencilState->Create(desc))
  {
    SAFE_DELETE(depthStencilState);
    return NULL;
  }
  depthStencilStates.AddElement(&depthStencilState);
  return depthStencilState;
}

OGL_BlendState* OGL_Renderer::CreateBlendState(const BlendDesc &desc)
{
  for(unsigned int i=0; i<blendStates.GetSize(); i++)
  {
    if(blendStates[i]->GetDesc() == desc)
      return blendStates[i];
  }
  OGL_BlendState *blendState = new OGL_BlendState;
  if(!blendState)
    return NULL;
  if(!blendState->Create(desc))
  {
    SAFE_DELETE(blendState);
    return NULL;
  }
  blendStates.AddElement(&blendState);
  return blendState;
}

RenderTargetConfig* OGL_Renderer::CreateRenderTargetConfig(const RtConfigDesc &desc)
{
  for(unsigned int i=0; i<renderTargetConfigs.GetSize(); i++)
  {
    if(renderTargetConfigs[i]->GetDesc() == desc)
      return renderTargetConfigs[i];
  }
  RenderTargetConfig *renderTargetConfig = new RenderTargetConfig;
  if(!renderTargetConfig)
    return NULL;
  if(!renderTargetConfig->Create(desc))
  {
    SAFE_DELETE(renderTargetConfig);
    return NULL;
  }
  renderTargetConfigs.AddElement(&renderTargetConfig);
  return renderTargetConfig;
}

OGL_RenderTarget* OGL_Renderer::CreateRenderTarget(const RenderTargetDesc &desc)
{
  OGL_RenderTarget *renderTarget = new OGL_RenderTarget;
  if(!renderTarget)
    return NULL;
  if(!renderTarget->Create(desc))
  {
    SAFE_DELETE(renderTarget);
    return NULL;
  }
  renderTargets.AddElement(&renderTarget);
  return renderTarget;
}

OGL_RenderTarget* OGL_Renderer::CreateBackBufferRt()
{
  OGL_RenderTarget *backBuffer = new OGL_RenderTarget;
  if(!backBuffer)
    return NULL;
  if(!backBuffer->CreateBackBuffer())
  {
    SAFE_DELETE(backBuffer);
    return NULL;
  }
  renderTargets.AddElement(&backBuffer);
  return backBuffer;
}

OGL_VertexLayout* OGL_Renderer::CreateVertexLayout(const VertexElementDesc *vertexElementDescs, unsigned int numVertexElementDescs)
{
  for(unsigned int i=0; i<vertexLayouts.GetSize(); i++)
  {
    if(vertexLayouts[i]->IsEqual(vertexElementDescs, numVertexElementDescs))
      return vertexLayouts[i];
  }
  OGL_VertexLayout *vertexLayout = new OGL_VertexLayout;
  if(!vertexLayout)
    return NULL;
  if(!vertexLayout->Create(vertexElementDescs, numVertexElementDescs))
  {
    SAFE_DELETE(vertexLayout);
    return NULL;
  }
  vertexLayouts.AddElement(&vertexLayout);
  return vertexLayout;
}

OGL_VertexBuffer* OGL_Renderer::CreateVertexBuffer(unsigned int vertexSize, unsigned int maxVertexCount, bool dynamic)
{
  OGL_VertexBuffer *vertexBuffer = new OGL_VertexBuffer;
  if(!vertexBuffer)
    return NULL;
  if(!vertexBuffer->Create(vertexSize, maxVertexCount, dynamic))
  {
    SAFE_DELETE(vertexBuffer);
    return NULL;
  }
  vertexBuffers.AddElement(&vertexBuffer);
  return vertexBuffer;
}

OGL_IndexBuffer* OGL_Renderer::CreateIndexBuffer(unsigned int maxIndexCount, bool dynamic)
{
  OGL_IndexBuffer *indexBuffer = new OGL_IndexBuffer;
  if(!indexBuffer)
    return NULL;
  if(!indexBuffer->Create(maxIndexCount, dynamic))
  {
    SAFE_DELETE(indexBuffer);
    return NULL;
  }
  indexBuffers.AddElement(&indexBuffer);
  return indexBuffer;
}

OGL_UniformBuffer* OGL_Renderer::CreateUniformBuffer(unsigned int bufferSize)
{
  OGL_UniformBuffer *uniformBuffer = new OGL_UniformBuffer;
  if(!uniformBuffer)
    return NULL;
  if(!uniformBuffer->Create(bufferSize))
  {
    SAFE_DELETE(uniformBuffer);
    return NULL;
  }
  uniformBuffers.AddElement(&uniformBuffer);
  return uniformBuffer;
}

OGL_StructuredBuffer* OGL_Renderer::CreateStructuredBuffer(unsigned int elementCount, unsigned int elementSize, unsigned int flags)
{
  OGL_StructuredBuffer *structuredBuffer = new OGL_StructuredBuffer;
  if(!structuredBuffer)
    return NULL;
  if(!structuredBuffer->Create(elementCount, elementSize, flags))
  {
    SAFE_DELETE(structuredBuffer);
    return NULL;
  }
  structuredBuffers.AddElement(&structuredBuffer);
  return structuredBuffer;
}

OGL_TimerQueryObject* OGL_Renderer::CreateTimerQueryObject()
{
  OGL_TimerQueryObject *timerQueryObject = new OGL_TimerQueryObject;
  if(!timerQueryObject)
    return NULL;
  if(!timerQueryObject->Create())
  {
    SAFE_DELETE(timerQueryObject);
    return NULL;
  }
  timerQueryObjects.AddElement(&timerQueryObject);
  return timerQueryObject;
}

Camera* OGL_Renderer::CreateCamera(float fovy, float aspectRatio, float nearClipDistance, float farClipDistance)
{
  Camera *camera = new Camera;
  if(!camera)
    return NULL;
  if(!camera->Init(fovy, aspectRatio, nearClipDistance, farClipDistance))
  {
    SAFE_DELETE(camera);
    return NULL;
  }
  cameras.AddElement(&camera);
  return camera;
}

IPostProcessor* OGL_Renderer::GetPostProcessor(const char *name) const
{
  if(!name)
    return NULL;
  for(unsigned int i=0; i<postProcessors.GetSize(); i++)
  {
    if(strcmp(name, postProcessors[i]->GetName()) == 0)
      return postProcessors[i];
  }
  return NULL;
}

bool OGL_Renderer::UpdateBuffers(bool dynamic)
{
  for(unsigned int i=0; i<vertexBuffers.GetSize(); i++)
  {
    OGL_VertexBuffer *vertexBuffer = GetVertexBuffer(i);
    if(vertexBuffer->IsDynamic() == dynamic)
    {
      if(!vertexBuffer->Update())
        return false;
    }
  }
  for(unsigned int i=0; i<indexBuffers.GetSize(); i++)
  {
    OGL_IndexBuffer *indexBuffer = indexBuffers[i];
    if(indexBuffer->IsDynamic() == dynamic)
    {
      if(!indexBuffer->Update())
        return false;
    }
  }
  return true;
}

void OGL_Renderer::SetupPostProcessSurface(DrawCmd &drawCmd)
{
  drawCmd.primitiveType = TRIANGLES_PRIMITIVE;
  drawCmd.numElements = 3;
  drawCmd.rasterizerState = noneCullRS;
  drawCmd.depthStencilState = noDepthTestDSS;
  drawCmd.blendState = defaultBS;
}

void OGL_Renderer::AddGpuCmd(const GpuCmd &gpuCmd)
{
  int index = gpuCmds.AddElements(1, &gpuCmd);
  gpuCmds[index].ID = index;
  if(gpuCmd.order == SHADOW_CO)
    numShadowDrawCalls++;
}

void OGL_Renderer::ClearFrame()
{
  gpuCmds.Clear(); 
  lastGpuCmd.Reset(DRAW_CM);
  for(unsigned int i=0; i<renderTargets.GetSize(); i++)
    renderTargets[i]->Reset();
}

void OGL_Renderer::ExecutePostProcessors()
{
  for(unsigned int i=0; i<postProcessors.GetSize(); i++)
    postProcessors[i]->Execute();
}

void OGL_Renderer::SetDrawStates(const BaseDrawCmd &cmd)
{ 
  if(cmd.rasterizerState != lastGpuCmd.draw.rasterizerState)
  {
    assert(cmd.rasterizerState != NULL);
    cmd.rasterizerState->Set();
    lastGpuCmd.draw.rasterizerState = cmd.rasterizerState;
  } 

  if(cmd.depthStencilState != lastGpuCmd.draw.depthStencilState)
  {   
    assert(cmd.depthStencilState != NULL);
    cmd.depthStencilState->Set();
    lastGpuCmd.draw.depthStencilState = cmd.depthStencilState;
  }

  if(cmd.blendState != lastGpuCmd.draw.blendState)
  {   
    assert(cmd.blendState != NULL);
    cmd.blendState->Set();
    lastGpuCmd.draw.blendState = cmd.blendState;
  }

  if((cmd.vertexLayout != lastGpuCmd.draw.vertexLayout) || (cmd.vertexBuffer != lastGpuCmd.draw.vertexBuffer) ||
     (cmd.indexBuffer != lastGpuCmd.draw.indexBuffer))
  {
    if(cmd.vertexLayout)
      cmd.vertexLayout->Bind();
    if(cmd.vertexBuffer)
      cmd.vertexBuffer->Bind();
    if(cmd.indexBuffer)
      cmd.indexBuffer->Bind();
    lastGpuCmd.draw.vertexLayout = cmd.vertexLayout;
    lastGpuCmd.draw.vertexBuffer = cmd.vertexBuffer;
    lastGpuCmd.draw.indexBuffer = cmd.indexBuffer;
  }
}

void OGL_Renderer::SetShaderStates(const ShaderCmd &cmd)
{ 
  if((cmd.renderTarget != lastGpuCmd.draw.renderTarget) || (cmd.renderTargetConfig != lastGpuCmd.draw.renderTargetConfig))
  {
    if(cmd.renderTarget)
      cmd.renderTarget->Bind(cmd.renderTargetConfig);
    lastGpuCmd.draw.renderTarget = cmd.renderTarget;	  
    lastGpuCmd.draw.renderTargetConfig = cmd.renderTargetConfig;
  }

  if(cmd.shader != lastGpuCmd.draw.shader)
  {
    assert(cmd.shader != NULL);  
    cmd.shader->Bind();
    lastGpuCmd.draw.shader = cmd.shader;
  }
}

void OGL_Renderer::SetShaderParams(const ShaderCmd &cmd)
{
  // set camera uniform buffer
  if(cmd.camera)
    cmd.shader->SetUniformBuffer(CAMERA_UB_BP, cmd.camera->GetUniformBuffer());

  // set custom uniform buffers
  for(unsigned int i=0; i<NUM_CUSTOM_UNIFORM_BUFFER_BP; i++)
  {
    if(cmd.customUBs[i])
      cmd.shader->SetUniformBuffer((uniformBufferBP)(CUSTOM0_UB_BP+i), cmd.customUBs[i]);
  }

  // set custom structured buffers
  for(unsigned int i=0; i<NUM_STRUCTURED_BUFFER_BP; i++)
  {
    if(cmd.customSBs[i])
      cmd.shader->SetStructuredBuffer((structuredBufferBP)(CUSTOM0_SB_BP+i), cmd.customSBs[i]);
  }

  // set textures
  for(unsigned int i=0; i<NUM_TEXTURE_BP; i++)
  {
    if(cmd.textures[i])
      cmd.shader->SetTexture((textureBP)i, cmd.textures[i], cmd.samplers[i]);
  }
}

// compare-function passed to qsort
static int CompareGpuCmds(const void *a, const void *b)
{
  const GpuCmd *cA = (GpuCmd*)a;
  const GpuCmd *cB = (GpuCmd*)b;

  if(cA->order < cB->order)
    return -1;
  else if(cA->order > cB->order)
    return 1;
  if(cA->GetID( )< cB->GetID())
    return -1;
  else if(cA->GetID() > cB->GetID())
    return 1;
  return 0;
} 

void OGL_Renderer::ExecuteGpuCmds()
{
  numShadowDrawCalls = 0;
  ExecutePostProcessors();
  UpdateBuffers(true);
  gpuCmds.Sort(CompareGpuCmds);
  for(unsigned int i=0; i<gpuCmds.GetSize(); i++)
  { 
    switch(gpuCmds[i].mode)
    {	
    case DRAW_CM:
      {
        SetDrawStates(gpuCmds[i].draw);
        SetShaderStates(gpuCmds[i].draw);
        SetShaderParams(gpuCmds[i].draw);
        Draw(gpuCmds[i].draw);
        break;
      }
    
    case INDIRECT_DRAW_CM:
      {
        SetDrawStates(gpuCmds[i].indirectDraw);
        SetShaderStates(gpuCmds[i].indirectDraw);
        SetShaderParams(gpuCmds[i].indirectDraw);
        DrawIndirect(gpuCmds[i].indirectDraw);
        break;
      }
 
    case COMPUTE_CM:
      {
        SetShaderStates(gpuCmds[i].compute);
        SetShaderParams(gpuCmds[i].compute);
        Dispatch(gpuCmds[i].compute);
        break;
      }

    case TIMER_QUERY_CM:
      {
        TimerQuery(gpuCmds[i].timerQuery);
        break;
      }
    }
  } 

  // draw tweak bars
  Demo::renderer->GetSampler(POINT_SAMPLER_ID)->Bind(COLOR_TEX_BP);
  TwDraw();  

  SwapBuffers(hDC);
} 

void OGL_Renderer::Draw(const DrawCmd &cmd)
{
  bool indexed = (cmd.indexBuffer != NULL);
  if(indexed)
  {
    if(cmd.numInstances < 2)
      glDrawElements(cmd.primitiveType, cmd.numElements, GL_UNSIGNED_INT, INT_BUFFER_OFFSET(cmd.firstIndex));
    else 
      glDrawElementsInstanced(cmd.primitiveType, cmd.numElements, GL_UNSIGNED_INT, INT_BUFFER_OFFSET(cmd.firstIndex), cmd.numInstances);
  }
  else
  {
    if(cmd.numInstances < 2)
      glDrawArrays(cmd.primitiveType, cmd.firstIndex, cmd.numElements); 
    else 
      glDrawArraysInstanced(cmd.primitiveType, cmd.firstIndex, cmd.numElements, cmd.numInstances);
  }
}

void OGL_Renderer::DrawIndirect(const IndirectDrawCmd &cmd)
{
  glMultiDrawElementsIndirectCountARB(cmd.primitiveType, GL_UNSIGNED_INT, INT_BUFFER_OFFSET(1), 0, cmd.maxDrawCount, cmd.stride);
}

void OGL_Renderer::Dispatch(const ComputeCmd &cmd)
{
  glDispatchCompute(cmd.numThreadGroupsX, cmd.numThreadGroupsY, cmd.numThreadGroupsZ);
}

void OGL_Renderer::TimerQuery(const TimerQueryCmd &cmd)
{
  assert(cmd.object != NULL);
  if(cmd.mode == BEGIN_TIMER_QUERY)
    cmd.object->BeginQuery();
  else
    cmd.object->EndQuery();
}

void OGL_Renderer::SaveScreenshot() const
{
  // try to find a not existing path for screen-shot	
  char filePath[DEMO_MAX_FILEPATH];
  for(unsigned int i=0; i<1000; i++)
  {
    _snprintf(filePath, sizeof(filePath), "../Data/screenshots/screen%d.bmp", i);
    if(!Demo::fileManager->FilePathExists(filePath))
      break;
    if(i == 999)
      return;
  }

  // fill up BMP info-header
  BITMAPINFOHEADER infoHeader;
  memset(&infoHeader,0,sizeof(BITMAPINFOHEADER));
  infoHeader.biSize = sizeof(BITMAPINFOHEADER);
  infoHeader.biWidth = SCREEN_WIDTH;
  infoHeader.biHeight = SCREEN_HEIGHT;
  infoHeader.biPlanes = 1; 
  infoHeader.biBitCount = 24;
  infoHeader.biCompression = BI_RGB;    
  infoHeader.biSizeImage =  infoHeader.biWidth*infoHeader.biHeight*3; 

  // fill up BMP file-header
  BITMAPFILEHEADER fileHeader;
  memset(&fileHeader,0,sizeof(BITMAPFILEHEADER));  
  fileHeader.bfOffBits = sizeof(BITMAPFILEHEADER)+sizeof(BITMAPINFOHEADER);
  fileHeader.bfSize = fileHeader.bfOffBits+infoHeader.biSizeImage;  
  fileHeader.bfType = 0x4D42;

  // create/ fill up data with back-buffer pixels
  unsigned char *data = new unsigned char[SCREEN_WIDTH*SCREEN_HEIGHT*3];
  if(!data)
    return;
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  glViewport(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT);
  glReadPixels(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT, GL_BGR, GL_UNSIGNED_BYTE, data);

  // write file-header/ info-header/ data to bitmap-file
  FILE *file = NULL;
  fopen_s(&file, filePath, "wb");
  if(file != NULL)
  {
    fwrite(&fileHeader, sizeof(BITMAPFILEHEADER), 1, file);
    fwrite(&infoHeader, sizeof(BITMAPINFOHEADER), 1, file);
    fwrite(data, 1, SCREEN_WIDTH*SCREEN_HEIGHT*3, file);
    fclose(file);
  }

  SAFE_DELETE_ARRAY(data);
}
