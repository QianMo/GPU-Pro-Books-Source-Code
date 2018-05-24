#include <stdafx.h>
#include <Demo.h>
#include <TiledDeferred.h>
#include <Sky.h>
#include <FinalProcessor.h>
#include <Application.h>

#define INIT_CAMERA_POSITION Vector3(835.0f, 770.0f, -32.0f)
#define INIT_CAMERA_ROTATION Vector3(94.0f, 27.0f, 0.0f)
#define MOUSE_SPEED 0.5f
#define CAMERA_MOVE_SPEED 0.2f

// callbacks for AntTweakbar
static void TW_CALL SetShadowMode(const void *value, void *clientData)
{
  ((TiledDeferred*)clientData)->SetShadowMode(*((shadowModes*)value));
}

static void TW_CALL GetShadowMode(void *value, void *clientData)
{
  *((shadowModes*)value) = ((TiledDeferred*)clientData)->GetShadowMode();
}

static void TW_CALL SetNumLights(const void *value, void *clientData)
{
  ((Application*)clientData)->SetNumActiveLights(*((unsigned int*)value));
}

static void TW_CALL GetNumLights(void *value, void *clientData)
{
  *((unsigned int*)value) = ((Application*)clientData)->GetNumActiveLights();
}

static void TW_CALL EnableFrustumCulling(const void *value, void *clientData)
{
  ((TiledDeferred*)clientData)->EnableFrustumCulling(*((const bool*)value));
}

static void TW_CALL IsFrustumCullingEnabled(void *value, void *clientData)
{
  *((bool*)value) = ((TiledDeferred*)clientData)->IsFrustumCullingEnabled();
}

static void TW_CALL EnableLightAnimation(const void *value, void *clientData)
{
  ((Application*)clientData)->EnablePathLightAnimation(*((const bool*)value));
}

static void TW_CALL IsLightAnimationEnabled(void *value, void *clientData)
{
  *((bool*)value) = ((Application*)clientData)->IsPathLightAnimationEnabled();
}

static void TW_CALL ShowTiledShadowMap(const void *value, void *clientData)
{
  ((TiledDeferred*)clientData)->EnableTiledShadowMapVis(*((const bool*)value));
}

static void TW_CALL IsTiledShadowMapShown(void *value, void *clientData)
{
  *((bool*)value) = ((TiledDeferred*)clientData)->IsTiledShadowMapVisEnabled();
}

static void TW_CALL ShowProfiling(const void *value, void *clientData)
{
  ((Application*)clientData)->ShowProfilingInfo(*((const bool*)value));
}

static void TW_CALL IsProfilingShown(void *value, void *clientData)
{
  *((bool*)value) = ((Application*)clientData)->IsProfilingInfoShown();
}


bool Application::Init()
{
  if(!Demo::Create())
    return false;

  char exeDirectory[DEMO_MAX_FILEPATH];
  if((!Demo::fileManager->GetExeDirectory(exeDirectory)) ||
     (!Demo::fileManager->SetWorkDirectory(exeDirectory)))
  {
    return false;
  }
  Demo::fileManager->AddDirectory("../Data/");

  if(!Demo::window->Create())
    return false;

  if(!Demo::renderer->Create())
    return false;

  Demo::timeManager->Init();

  // init AntTweakBar
  if(!TwInit(TW_OPENGL_CORE, NULL))
    return false;
  TwWindowSize(SCREEN_WIDTH, SCREEN_HEIGHT);

  tiledDeferredPP = Demo::renderer->CreatePostProcessor<TiledDeferred>();
  if(!tiledDeferredPP)
    return false;

  if(!Demo::renderer->CreatePostProcessor<Sky>())
    return false;

  if(!Demo::renderer->CreatePostProcessor<FinalProcessor>())
    return false;

  if(!OnInit())
    return false;
  
  if(!Demo::renderer->UpdateBuffers(false))
    return false;

  if(!tiledDeferredPP->CreateIndirectDrawData())
    return false;

  // set fixed random seed to ensure reproducible results
  srand(0);

  return true;
}

void Application::Run()
{
  // main loop of application
  while(!quit)
  {
    // handle window messages
    if(!Demo::window->HandleMessages())
      quit = true;

    Demo::timeManager->Update();

    Demo::renderer->ClearFrame();

    OnRun();

    for(unsigned int i=0; i<Demo::resourceManager->GetNumDemoMeshes(); i++) 
      Demo::resourceManager->GetDemoMesh(i)->AddBaseSurfaces();

    for(unsigned int i=0; i<Demo::resourceManager->GetNumFonts(); i++) 
      Demo::resourceManager->GetFont(i)->AddSurfaces();

    // execute all GPU commands
    Demo::renderer->ExecuteGpuCmds();

    Demo::inputManager->Update();
  }
}

void Application::Shutdown()
{
  // terminate AntTweakBar
  TwTerminate();

  Demo::Release();
}

void Application::Quit()
{
  quit = true;
}

void Application::SetNumActiveLights(unsigned int numActiveLights)
{
  assert(numActiveLights <= MAX_NUM_POINT_LIGHTS);
  this->numActiveLights = numActiveLights;
  for(unsigned int i=0; i<MAX_NUM_POINT_LIGHTS; i++)
    pathPointLights[i].SetActive(i < numActiveLights);
}

void Application::EnablePathLightAnimation(bool enable)
{
  pathLightsAnimated = enable;  
  for(unsigned int i=0; i<MAX_NUM_POINT_LIGHTS; i++)
    pathPointLights[i].SetPaused(!enable);
}

bool Application::OnInit()
{
  // cache pointer to main camera
  mainCamera = Demo::renderer->GetCamera(MAIN_CAMERA_ID);
  if(!mainCamera)
    return false;

  // set initial camera position/ rotation 
  mainCamera->Update(INIT_CAMERA_POSITION, INIT_CAMERA_ROTATION);

  // init previous mouse position
  prevMousePos = Demo::inputManager->GetMousePos();

  // create default font for displaying info
  defaultFont = Demo::resourceManager->LoadFont("fonts/arial.font");
  if(!defaultFont)
    return false;
  
  // load sponza mesh
  if(!Demo::resourceManager->LoadDemoMesh("meshes/sponza.mesh"))
    return false;

  // set bounds for moving point lights  
  PathPointLight::SetBounds(Vector3(-1330.0f, 50.0, -620.0), Vector3(1200.0f, 950.0f, 550.0f));

  // create point lights that follow a random path
  for(unsigned int i=0; i<MAX_NUM_POINT_LIGHTS; i++)
  {
    if(!pathPointLights[i].Init())
      return false;
    pathPointLights[i].SetActive(i < numActiveLights);
  }

  // create a tweak bar 
  TwBar *tweakBar = TwNewBar("Settings");
  if(!tweakBar)
    return false;

  // configure tweak bars
  TwDefine("GLOBAL help='WASD - Move camera\nRMB - Enable mouse look\nESC - Quit application\nR - Reset\nP - Screenshot'"); 
  TwDefine("GLOBAL fontsize=3");  
  TwDefine("GLOBAL fontresizable=false"); 
  TwDefine("TW_HELP position='30 120'");
  TwDefine("TW_HELP size='450 200'");
  TwDefine("TW_HELP visible=false"); 
  TwDefine("TW_HELP iconified=false"); 
  TwDefine("TW_HELP iconifiable=false"); 
  TwDefine("Settings visible=false"); 
  TwDefine("Settings iconified=false"); 
  TwDefine("Settings iconifiable=false"); 
  const int barSize[2] = {350, 148};
  TwSetParam(tweakBar, NULL, "size", TW_PARAM_INT32, 2, barSize);
  const int barPosition[2] = {SCREEN_WIDTH - 380, 28};
  TwSetParam(tweakBar, nullptr, "position", TW_PARAM_INT32, 2, barPosition);
  const int valueWidth = 150;
  TwSetParam(tweakBar, nullptr, "valueswidth", TW_PARAM_INT32, 1, &valueWidth);

  // add variables to tweak bar 
  const unsigned int numModes = 3;
  TwEnumVal modeEnumValues[numModes] =
  {
    {NO_SHADOW_SM, "No Shadows"}, 
    {TILED_SHADOW_SM, "Tile-based Shadows"}, 
    {CUBE_SHADOW_SM, "Cube Map Shadows"}
  }; 
  TwType modeType = TwDefineEnum("ModeType", modeEnumValues, numModes);
  const char *modeDefinition =
  {
    "help='No Shadows = Shadows disabled\nTile-based Shadows = Tile-based Omnidirectional Shadows\nCube Map Shadows = Shadows using a cube map array'"
  };
  TwAddVarCB(tweakBar, "Shadow mode", modeType, SetShadowMode, GetShadowMode, tiledDeferredPP, modeDefinition);
  char numLightsDefinition[16];
  _snprintf(numLightsDefinition, sizeof(numLightsDefinition), "min=1 max=%i", MAX_NUM_POINT_LIGHTS);
  TwAddVarCB(tweakBar, "Number of lights", TW_TYPE_UINT32, SetNumLights, GetNumLights, this, numLightsDefinition);
  TwAddVarCB(tweakBar, "Frustum culling of lights", TW_TYPE_BOOLCPP, EnableFrustumCulling, IsFrustumCullingEnabled, tiledDeferredPP, NULL);
  TwAddVarCB(tweakBar, "Animation of lights", TW_TYPE_BOOLCPP, EnableLightAnimation, IsLightAnimationEnabled, this, NULL);
  TwAddVarCB(tweakBar, "Show Tiled Shadow Map", TW_TYPE_BOOLCPP, ShowTiledShadowMap, IsTiledShadowMapShown, tiledDeferredPP, NULL);
  TwAddVarCB(tweakBar, "Show profiling", TW_TYPE_BOOLCPP, ShowProfiling, IsProfilingShown, this, NULL);

  return true;
}

void Application::HandleInput()
{
  // update camera	
  Vector3 cameraRotation = mainCamera->GetRotation();
  POINT currentMousePos = Demo::inputManager->GetMousePos();
  if(Demo::inputManager->GetTriggerState(VK_RBUTTON))
  { 
    Demo::inputManager->ShowMouseCursor(false);
    cameraRotation.x -= (currentMousePos.x-prevMousePos.x)*MOUSE_SPEED;
    cameraRotation.y += (currentMousePos.y-prevMousePos.y)*MOUSE_SPEED;
    CLAMP(cameraRotation.y, -80.0f, 80.0f);
    Demo::inputManager->SetMousePos(prevMousePos);
  }
  else
  {
    Demo::inputManager->ShowMouseCursor(true);
    prevMousePos = currentMousePos;
  }
  
  Vector3 velocity;
  if(Demo::inputManager->GetTriggerState(char('W'))) // move forward
  {
    velocity.x = -sin(DEG2RAD(cameraRotation.x));
    velocity.y = -tan(DEG2RAD(cameraRotation.y));
    velocity.z = -cos(DEG2RAD(cameraRotation.x));
  }
  if(Demo::inputManager->GetTriggerState(char('S'))) // move backward
  {
    velocity.x = sin(DEG2RAD(cameraRotation.x));
    velocity.y = tan(DEG2RAD(cameraRotation.y));
    velocity.z = cos(DEG2RAD(cameraRotation.x));
  }
  if(Demo::inputManager->GetTriggerState(char('A'))) // move left
  {
    velocity.x = sin(DEG2RAD(cameraRotation.x-90.0f));
    velocity.z = cos(DEG2RAD(cameraRotation.x-90.0f));
  }
  if(Demo::inputManager->GetTriggerState(char('D'))) // move right
  {
    velocity.x = -sin(DEG2RAD(cameraRotation.x-90.0f));
    velocity.z = -cos(DEG2RAD(cameraRotation.x-90.0f));
  }
  Vector3 cameraPosition = mainCamera->GetPosition();
  cameraPosition += velocity.GetNormalized()*CAMERA_MOVE_SPEED*(float)Demo::timeManager->GetFrameInterval();

  mainCamera->Update(cameraPosition, cameraRotation);

  // toggle help 
  if(Demo::inputManager->GetSingleTriggerState(VK_F1))
  {
    showHelp = !showHelp;
    if(showHelp)
      TwDefine("TW_HELP visible=true");  
    else
      TwDefine("TW_HELP visible=false"); 
  }

  // toggle settings
  if(Demo::inputManager->GetSingleTriggerState(VK_F2))
  {
    showSettings = !showSettings;
    if(showSettings)
      TwDefine("Settings visible=true");  
    else
      TwDefine("Settings visible=false"); 
  }

  // reset settings
  if(Demo::inputManager->GetSingleTriggerState(char('R')))
  {
    mainCamera->Update(INIT_CAMERA_POSITION, INIT_CAMERA_ROTATION);
    numActiveLights = MAX_NUM_POINT_LIGHTS/2;
    pathLightsAnimated = true;
    for(unsigned int i=0; i<MAX_NUM_POINT_LIGHTS; i++)
    {
      pathPointLights[i].SetPaused(false);
      pathPointLights[i].SetActive(i < numActiveLights);
    }
    tiledDeferredPP->SetShadowMode(TILED_SHADOW_SM);
    tiledDeferredPP->EnableFrustumCulling(true);
    tiledDeferredPP->EnableTiledShadowMapVis(false);
    showProfilingInfo = false;
  }

  // save screen-shot 
  if(Demo::inputManager->GetSingleTriggerState(char('P')))
    Demo::renderer->SaveScreenshot();

  // quit application by pressing ESCAPE
  if(Demo::inputManager->GetSingleTriggerState(VK_ESCAPE))
    Quit();
}

void Application::DisplayInfo()
{
  defaultFont->Print(Vector2(-0.95f, 0.86f), 0.04f, Color(0.0f, 1.0f, 0.0f), "FPS: %.2f", Demo::timeManager->GetFPS());

  if(showHelp)
    defaultFont->Print(Vector2(-0.95f, 0.78f), 0.03f, Color(1.0f, 1.0f, 0.0f), "F1 - Hide help");
  else
    defaultFont->Print(Vector2(-0.95f, 0.78f), 0.03f, Color(1.0f, 1.0f, 0.0f), "F1 - Show help");
  
  if(showSettings)
    defaultFont->Print(Vector2(-0.95f, 0.72f), 0.03f, Color(1.0f, 1.0f, 0.0f), "F2 - Hide settings");
  else
    defaultFont->Print(Vector2(-0.95f, 0.72f), 0.03f, Color(1.0f, 1.0f, 0.0f), "F2 - Show settings");

  if(showProfilingInfo)
  {
    defaultFont->Print(Vector2(0.42f, -0.58f), 0.03f, Color(1.0f, 1.0f, 1.0f), "Number of active lights: %i", numActiveLights);
    defaultFont->Print(Vector2(0.42f, -0.64f), 0.03f, Color(1.0f, 1.0f, 1.0f), "Number of visible lights: %i", tiledDeferredPP->GetNumVisibleLights());
    defaultFont->Print(Vector2(0.42f, -0.7f), 0.03f, Color(1.0f, 1.0f, 1.0f), "Number of shadow draw-calls: %i", Demo::renderer->GetNumShadowDrawCalls());
    defaultFont->Print(Vector2(0.42f, -0.76f), 0.03f, Color(1.0f, 1.0f, 1.0f), "Frame time: %.2f", Demo::timeManager->GetFrameInterval());
    defaultFont->Print(Vector2(0.42f, -0.82f), 0.03f, Color(1.0f, 1.0f, 1.0f), "CPU time - shadow: %.2f", tiledDeferredPP->GetShadowCpuElapsedTime());
    defaultFont->Print(Vector2(0.42f, -0.88f), 0.03f, Color(1.0f, 1.0f, 1.0f), "GPU time - shadow %.2f", tiledDeferredPP->GetShadowGpuElapsedTime());
    defaultFont->Print(Vector2(0.42f, -0.94f), 0.03f, Color(1.0f, 1.0f, 1.0f), "GPU time - lighting: %.2f", tiledDeferredPP->GetIllumGpuElapsedTime());
  }
}

void Application::OnRun()
{
  HandleInput();	
    
  DisplayInfo();
  
  if(pathLightsAnimated)
  {
    for(unsigned int i=0; i<MAX_NUM_POINT_LIGHTS; i++)
      pathPointLights[i].Update();
  }
}

