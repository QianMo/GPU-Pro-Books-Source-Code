#include <stdafx.h>
#include <Demo.h>
#include <Sky.h>
#include <Shading.h>
#include <FinalProcessor.h>
#include <Application.h>

#define CAMERA_POSITION Vector3(0.0f, 58.0f, 64.0f)
#define CAMERA_ROTATION Vector3(2.5f, 10.5f, 0.0f)
#define CAMERA_MOVE_SPEED 0.1f
#define MOUSE_SPEED 0.5f
#define DIR_LIGHT_DIR_X 0.0f
#define DIR_LIGHT_DIR_Y -1.0f
#define DIR_LIGHT_DIR_Z -0.65f

enum decalMaterialIds
{
  SQUARE_DECAL_MATERIAL=0,
  CIRCLE_DECAL_MATERIAL,
  STAR_DECAL_MATERIAL,
  NUM_DECAL_MATERIALS
};

const char *decalMaterialNames[NUM_DECAL_MATERIALS] = 
{
  "materials/decals/square.mtl",
  "materials/decals/circle.mtl",
  "materials/decals/star.mtl"
};

struct HitResult
{
  DemoModel *hitModel;
  float minHitDistance;
  float maxHitDistance;
};

static void GetClosetIntersection(const Vector3 &rayOrigin, const Vector3 &rayDir, HitResult &hitResult)
{
  DemoModel *hitModel = nullptr;
  float minHitDistance = FLT_MAX;
  float maxHitDistance = -FLT_MAX;

  for(UINT i=0; i<Demo::resourceManager->GetNumDemoModels(); i++)
  {
    float tMin, tMax;
    DemoModel *model = Demo::resourceManager->GetDemoModel(i);
    if(model->GetBounds().IntersectRay(rayOrigin, rayDir, tMin, tMax))
    {
      if(tMin < minHitDistance)
      {
        hitModel = model;
        minHitDistance = tMin;
        maxHitDistance = tMax;
      }
    }
  }

  hitResult.hitModel = hitModel;
  hitResult.minHitDistance = minHitDistance;
  hitResult.maxHitDistance = maxHitDistance;
}

void Application::ResetSettings()
{
  decalMaterialIndex = 0;
  decalWidth = 5.0f;
  decalHeight = 5.0f;
  decalAngle = 0.0f;
  dirLightDirX = DIR_LIGHT_DIR_X;
  dirLightDirY = DIR_LIGHT_DIR_Y;
  dirLightDirZ = DIR_LIGHT_DIR_Z;
  showProfiling = false;
  pauseAnim = false;
  useDecals = true;
  debugDecalMask = false;
}

bool Application::Init()
{
  if(!Demo::Create())
    return false;

  if(!Demo::threadManager->Init())
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

  if(!Demo::inputManager->Init())
    return false;

  if(!Demo::guiManager->Init())
    return false;

  shadingPP = Demo::renderer->CreatePostProcessor<Shading>();
  if(!shadingPP)
    return false;

  if(!Demo::renderer->CreatePostProcessor<Sky>())
    return false;

  if(!Demo::renderer->CreatePostProcessor<FinalProcessor>())
    return false;

  if(!OnInit())
    return false;

  if(!Demo::renderer->FinalizeInit())
    return false;

  return true;
}

void Application::Run()
{
  // main loop of application
  while(!quit)
  {
    if(!Demo::window->HandleMessages())
      quit = true;

    Demo::timeManager->Update();

    Demo::renderer->BeginFrame();

    Demo::guiManager->BeginFrame();

    OnRun();

    for(UINT i=0; i<Demo::resourceManager->GetNumDemoModels(); i++)
      Demo::resourceManager->GetDemoModel(i)->Render();

    for(UINT i=0; i<Demo::resourceManager->GetNumFonts(); i++)
      Demo::resourceManager->GetFont(i)->Render();

    Demo::guiManager->EndFrame();

    Demo::renderer->EndFrame();

    Demo::inputManager->Update();
  }
}

void Application::Shutdown()
{
  Demo::Release();
}

void Application::Quit()
{
  quit = true;
}

bool Application::OnInit()
{
  // create default font for displaying info
  defaultFont = Demo::resourceManager->LoadFont("fonts/arial.font");
  if(!defaultFont)
    return false;

  // cache pointer to main camera
  mainCamera = Demo::renderer->GetCamera(MAIN_CAMERA_ID);
  if (!mainCamera)
    return false;

  // set initial camera position/ rotation 
  mainCamera->Update(CAMERA_POSITION, CAMERA_ROTATION);

  // init previous mouse position
  prevMousePos = Demo::inputManager->GetMousePos();

  // load model
  model = Demo::resourceManager->LoadDemoModel("models/bob.model");
  if(!model)
    return false;

  // init skinned decals
  if(!model->InitSkinnedDecals(decalMaterialNames, NUM_DECAL_MATERIALS))
    return false;

  // create directional light
  dirLight = shadingPP->CreateDirectionalLight(Vector3(0.0f, -1.0f, -0.65f), Color(1.0f, 1.0f, 1.0f), 2.0f);
  if(!dirLight)
	  return false;

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
    cameraRotation.x -= (currentMousePos.x - prevMousePos.x)*MOUSE_SPEED;
    cameraRotation.y += (currentMousePos.y - prevMousePos.y)*MOUSE_SPEED;
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
    velocity.x = sin(DEG2RAD(cameraRotation.x - 90.0f));
    velocity.z = cos(DEG2RAD(cameraRotation.x - 90.0f));
  }
  if(Demo::inputManager->GetTriggerState(char('D'))) // move right
  {
    velocity.x = -sin(DEG2RAD(cameraRotation.x - 90.0f));
    velocity.z = -cos(DEG2RAD(cameraRotation.x - 90.0f));
  }
  Vector3 cameraPosition = mainCamera->GetPosition();
  cameraPosition += velocity.GetNormalized() * CAMERA_MOVE_SPEED * (float)Demo::timeManager->GetFrameInterval();

  mainCamera->Update(cameraPosition, cameraRotation);

  // add decal
  if(Demo::inputManager->GetSingleTriggerState(VK_SPACE))
  {
    HitResult hitResult;
    GetClosetIntersection(cameraPosition, mainCamera->GetDirection(), hitResult);

    if(hitResult.hitModel == model)
    {
      Matrix4 rotMatrix;
      rotMatrix.SetRotation(mainCamera->GetDirection(), decalAngle);

      AddDecalInfo decalInfo;
      decalInfo.rayOrigin = cameraPosition;
      decalInfo.rayDir = mainCamera->GetDirection();
      decalInfo.decalTangent = rotMatrix * mainCamera->GetRightVector();
      decalInfo.decalSize.Set(decalWidth, decalHeight, 5.0f);
      decalInfo.minHitDistance = hitResult.minHitDistance;
      decalInfo.maxHitDistance = hitResult.maxHitDistance;
      decalInfo.decalMaterialIndex = decalMaterialIndex;
      hitResult.hitModel->GetSkinnedDecals()->AddSkinnedDecal(decalInfo);
    }
  }

  // remove decal
  if(Demo::inputManager->GetSingleTriggerState(VK_BACK))
  {
    HitResult hitResult;
    GetClosetIntersection(cameraPosition, mainCamera->GetDirection(), hitResult);

    if(hitResult.hitModel == model)
    {
      RemoveDecalInfo decalInfo;
      decalInfo.rayOrigin = cameraPosition;
      decalInfo.rayDir = mainCamera->GetDirection();
      decalInfo.minHitDistance = hitResult.minHitDistance;
      decalInfo.maxHitDistance = hitResult.maxHitDistance;
      hitResult.hitModel->GetSkinnedDecals()->RemoveSkinnedDecal(decalInfo);
    }
  }

  // clear decals
  if(Demo::inputManager->GetSingleTriggerState(char('C')))
  {
    model->GetSkinnedDecals()->ClearSkinnedDecals();
  }

  // toggle help 
	if(Demo::inputManager->GetSingleTriggerState(VK_F1))
		showHelp = !showHelp;

  // toggle settings
	if(Demo::inputManager->GetSingleTriggerState(VK_F2))
		showSettings = !showSettings;

  // reset settings
  if(Demo::inputManager->GetSingleTriggerState(char('R')))
  {
    ResetSettings();
    mainCamera->Update(CAMERA_POSITION, CAMERA_ROTATION);
    dirLight->SetDirection(Vector3(dirLightDirX, dirLightDirY, dirLightDirZ));
    model->PauseAnim(pauseAnim);
    model->UseDecals(useDecals);
    model->EnableDebugDecalMask(debugDecalMask); 
  }

  // save screen-shot 
  if(Demo::inputManager->GetSingleTriggerState(char('P')))
    Demo::renderer->CaptureScreen();

  // quit application
  if(Demo::inputManager->GetSingleTriggerState(VK_ESCAPE))
    Quit();
}

void Application::DisplayInfo()
{
  defaultFont->Print(Vector2(-0.95f, 0.86f), 0.04f, Color(0.0f, 1.0f, 0.0f), "FPS: %.2f", Demo::timeManager->GetFPS());
	defaultFont->Print(Vector2(-0.95f, 0.78f), 0.03f, Color(1.0f, 1.0f, 0.0f), showHelp ? "F1 - Hide help" : "F1 - Show help");
  defaultFont->Print(Vector2(-0.95f, 0.72f), 0.03f, Color(1.0f, 1.0f, 0.0f), showSettings ? "F2 - Hide settings" : "F2 - Show settings");

  if(showHelp)
  {
    if(Demo::guiManager->BeginWindow("Help", Vector2(42.0, 160.0f), Vector2(200.0f, 180.0f)))
    {
      Demo::guiManager->ShowText("WASD - Move camera");
      Demo::guiManager->ShowText("RMB - Enable mouse look"); 
      Demo::guiManager->ShowText("Space - Add decal"); 
      Demo::guiManager->ShowText("Backspace - Remove decal"); 
      Demo::guiManager->ShowText("C - Clear decals"); 
      Demo::guiManager->ShowText("R - Reset settings");
      Demo::guiManager->ShowText("P - Screenshot");
      Demo::guiManager->ShowText("ESC - Quit application");
    }
    Demo::guiManager->EndWindow(); 
  }

  if(showSettings)
  {
    if(Demo::guiManager->BeginWindow("Settings", Vector2(SCREEN_WIDTH - 340, 20.0f), Vector2(310.0f, 400.0f)))
    {
      if(ImGui::CollapsingHeader("New Decal", nullptr, true, true))
      {
        const char* decalMaterialItems[] = { "Square", "Circle", "Star" };
        Demo::guiManager->ShowComboBox("Material", 155.0f, decalMaterialItems, _countof(decalMaterialItems), decalMaterialIndex);

        Demo::guiManager->ShowSliderFloat("Width", 155.0f, 3.0f, 10.0f, decalWidth);
  
        Demo::guiManager->ShowSliderFloat("Height", 155.0f, 3.0f, 10.0f, decalHeight);

        Demo::guiManager->ShowSliderFloat("Angle", 155.0f, 0.0f, 360.0f, decalAngle);
      }

      if(ImGui::CollapsingHeader("Animation", nullptr, true, true))
      {
        if(Demo::guiManager->ShowCheckBox("Pause animation", pauseAnim))
        {
          model->PauseAnim(pauseAnim);
        }
      }

      if(ImGui::CollapsingHeader("Lighting", nullptr, true, true))
      {
        if(Demo::guiManager->ShowSliderFloat("DirLightX", 155.0f, -1.0f, 1.0f, dirLightDirX))
        {
          dirLight->SetDirection(Vector3(dirLightDirX, dirLightDirY, dirLightDirZ));
        }

        if(Demo::guiManager->ShowSliderFloat("DirLightY", 155.0f, -1.0f, 1.0f, dirLightDirY))
        {
          dirLight->SetDirection(Vector3(dirLightDirX, dirLightDirY, dirLightDirZ));
        }

        if(Demo::guiManager->ShowSliderFloat("DirLightZ", 155.0f, -1.0f, 1.0f, dirLightDirZ))
        {
          dirLight->SetDirection(Vector3(dirLightDirX, dirLightDirY, dirLightDirZ));
        }
      }

      if(ImGui::CollapsingHeader("Profiling", nullptr, true, true))
      {
         Demo::guiManager->ShowCheckBox("Show profiling", showProfiling);

         if(Demo::guiManager->ShowCheckBox("Use decals", useDecals))
         {
           model->UseDecals(useDecals);
         }
      }

      if(ImGui::CollapsingHeader("Debug", nullptr, true, true))
      {
        if(Demo::guiManager->ShowCheckBox("Show decal mask", debugDecalMask))
        {
          model->EnableDebugDecalMask(debugDecalMask);
        }
      }
    }
    Demo::guiManager->EndWindow();
  }

  if(showProfiling)
  {
    defaultFont->Print(Vector2(-0.95f, -0.82f), 0.03f, Color(1.0f, 1.0f, 1.0f), "Skinning: %.4f ms", model->GetSkinningGpuTime());
    defaultFont->Print(Vector2(-0.95f, -0.88f), 0.03f, Color(1.0f, 1.0f, 1.0f), "Shading:  %.4f ms", model->GetBasePassGpuTime());
  }

  // draw crosshair
  defaultFont->Print(Vector2(-0.0058f, -0.0422f), 0.04f, Color(1.0f, 1.0f, 1.0f), "+");
}

void Application::OnRun()
{
  HandleInput();	
    
  DisplayInfo();
}

