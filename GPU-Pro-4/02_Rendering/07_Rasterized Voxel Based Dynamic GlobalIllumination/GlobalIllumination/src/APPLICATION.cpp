#include <stdafx.h>
#include <DEMO.h>
#include <DEFERRED_LIGHTING.h>
#include <GLOBAL_ILLUM.h>
#include <SKY.h>
#include <FINAL_PROCESSOR.h>
#include <APPLICATION.h>

bool APPLICATION::Init()
{
	if(!DEMO::Create())
		return false;

	char exeDirectory[DEMO_MAX_FILEPATH];
	if((!DEMO::fileManager->GetExeDirectory(exeDirectory))||
		(!DEMO::fileManager->SetWorkDirectory(exeDirectory)))
		return false;
	DEMO::fileManager->AddDirectory("../Data/");

	if(!DEMO::window->Create())
		return false;

	if(!DEMO::renderer->Create())
		return false;

	DEMO::timeManager->Init();

	if(!DEMO::renderer->CreatePostProcessor<DEFERRED_LIGHTING>())
		return false;

	// the GLOBAL_ILLUM post-processor is responsible for generating dynamic global illumination
	globalIllum = DEMO::renderer->CreatePostProcessor<GLOBAL_ILLUM>();
	if(!globalIllum)
		return false;

	if(!DEMO::renderer->CreatePostProcessor<SKY>())
		return false;

	if(!DEMO::renderer->CreatePostProcessor<FINAL_PROCESSOR>())
		return false;

	if(!OnInit())
		return false;

	DEMO::inputManager->CenterMousePos(); 

	return true;
}

void APPLICATION::Run()
{
	// main loop of application
	while(!quit)
	{
		// handle window messages
		if(!DEMO::window->HandleMessages())
			quit = true;

		DEMO::timeManager->Update();

		DEMO::renderer->ClearFrame();

		OnRun();

		DEMO::renderer->UpdateLights();

		int numDemoMeshes = DEMO::resourceManager->GetNumDemoMeshes();
		for(int i=0;i<numDemoMeshes;i++) 
		{
			DEMO_MESH *demoMesh = DEMO::resourceManager->GetDemoMesh(i);
			if(demoMesh->IsActive()) 
				demoMesh->AddSurfaces();
		}

		int numFonts = DEMO::resourceManager->GetNumFonts();
		for(int i=0;i<numFonts;i++) 
		{
			FONT *font = DEMO::resourceManager->GetFont(i);
			if(font->IsActive()) 
				font->AddSurfaces();
		}

		// draw all surfaces
		DEMO::renderer->DrawSurfaces();

		DEMO::inputManager->Update();
	}
}

void APPLICATION::Shutdown()
{
	OnShutdown();
	DEMO::Release();
}

void APPLICATION::Quit()
{
	quit = true;
}

bool APPLICATION::OnInit()
{
	// cache pointer to main camera
	mainCamera = DEMO::renderer->GetCamera(MAIN_CAMERA_ID);
	if(!mainCamera)
		return false;

	// set initial camera position/ rotation 
	mainCamera->Update(VECTOR3D(632.0f,150.0f,-142.0f),VECTOR3D(158.0f,0.0f,0.0f));

	// create default font for displaying info
	defaultFont = DEMO::resourceManager->LoadFont("fonts/arial.font");
	if(!defaultFont)
		return false;
	
	// load sponza mesh
	if(!DEMO::resourceManager->LoadDemoMesh("meshes/sponza.mesh"))
		return false;

	// create directional light
	dirLight = DEMO::renderer->CreateDirectionalLight(VECTOR3D(0.2403f,-0.9268f,0.2886f),COLOR(1.0f,1.0f,1.0f),1.5f);
	if(!dirLight)
		return false;

	// set control-points of path for all moving point-lights  
	PATH_POINT_LIGHT::SetControlPoints(-1350.0f,1250.0f,-600.0f,500.0f);

  // create point lights that follow a simple path
	if(!pathPointLights[0].Init(VECTOR3D(-550.0f,10.0f,500.0f),260.0f,COLOR(1.0f,1.0f,0.9f),1.5f,VECTOR3D(1.0f,0.0f,0.0f)))
		return false;
  if(!pathPointLights[1].Init(VECTOR3D(550.0f,10.0f,500.0f),260.0f,COLOR(1.0f,1.0f,0.9f),1.5f,VECTOR3D(1.0f,0.0f,0.0f)))
		return false;
	if(!pathPointLights[2].Init(VECTOR3D(-550.0f,10.0f,-600.0f),260.0f,COLOR(1.0f,1.0f,0.9f),1.5f,VECTOR3D(-1.0f,0.0f,0.0f)))
		return false;
	if(!pathPointLights[3].Init(VECTOR3D(550.0f,10.0f,-600.0f),260.0f,COLOR(1.0f,1.0f,0.9f),1.5f,VECTOR3D(-1.0f,0.0f,0.0f)))
		return false;
	if(!pathPointLights[4].Init(VECTOR3D(-1350.0f,10.0f,-30.0f),260.0f,COLOR(1.0f,1.0f,0.9f),1.5f,VECTOR3D(0.0f,0.0f,1.0f)))
		return false;
	if(!pathPointLights[5].Init(VECTOR3D(1250.0f,10.0f,-30.0f),260.0f,COLOR(1.0f,1.0f,0.9f),1.5f,VECTOR3D(0.0f,0.0f,-1.0f)))
		return false;
	if(!pathPointLights[6].Init(VECTOR3D(1250.0f,720.0f,450.0f),260.0f,COLOR(1.0f,1.0f,0.9f),1.5f,VECTOR3D(1.0f,0.0f,0.0f)))
		return false;
	if(!pathPointLights[7].Init(VECTOR3D(1200.0f,720.0f,-600.0f),260.0f,COLOR(1.0f,1.0f,0.9f),1.5f,VECTOR3D(1.0f,0.0f,0.0f)))
		return false;
	if(!pathPointLights[8].Init(VECTOR3D(-1350.0f,720.0f,460.0f),260.0f,COLOR(1.0f,1.0f,0.9f),1.5f,VECTOR3D(-1.0f,0.0f,0.0f)))
		return false;
	if(!pathPointLights[9].Init(VECTOR3D(-1320.0f,720.0f,-600.0f),260.0f,COLOR(1.0f,1.0f,0.9f),1.5f,VECTOR3D(-1.0f,0.0f,0.0f)))
		return false;
	if(!pathPointLights[10].Init(VECTOR3D(-40.0f,720.0f,500.0f),260.0f,COLOR(1.0f,1.0f,0.9f),1.5f,VECTOR3D(0.0f,0.0f,1.0f)))
		return false;
	if(!pathPointLights[11].Init(VECTOR3D(-40.0f,720.0f,-600.0f),260.0f,COLOR(1.0f,1.0f,0.9f),1.5f,VECTOR3D(0.0f,0.0f,-1.0f)))
		return false;

	return true;
}

void APPLICATION::HandleInput()
{
	// update camera	
	VECTOR3D cameraRotation = mainCamera->GetRotation();
	POINT mPos = DEMO::inputManager->GetMousePos();
	cameraRotation.x -= (mPos.x-(SCREEN_WIDTH>>1))*MOUSE_SPEED;
	cameraRotation.y += (mPos.y-(SCREEN_HEIGHT>>1))*MOUSE_SPEED;
	CLAMP(cameraRotation.y,-80.0f,80.0f);
	DEMO::inputManager->CenterMousePos();
	
	VECTOR3D velocity;
	if(DEMO::inputManager->GetTriggerState(char('W'))) // move forward
	{
		velocity.x = -sin(DEG2RAD(cameraRotation.x));
		velocity.y = -tan(DEG2RAD(cameraRotation.y));
		velocity.z = -cos(DEG2RAD(cameraRotation.x));
	}
	if(DEMO::inputManager->GetTriggerState(char('S'))) // move backward
	{
		velocity.x = sin(DEG2RAD(cameraRotation.x));
		velocity.y = tan(DEG2RAD(cameraRotation.y));
		velocity.z = cos(DEG2RAD(cameraRotation.x));
	}
	if(DEMO::inputManager->GetTriggerState(char('A'))) // move left
	{
		velocity.x = sin(DEG2RAD(cameraRotation.x-90));
		velocity.z = cos(DEG2RAD(cameraRotation.x-90));
	}
	if(DEMO::inputManager->GetTriggerState(char('D'))) // move right
	{
		velocity.x = -sin(DEG2RAD(cameraRotation.x-90));
		velocity.z = -cos(DEG2RAD(cameraRotation.x-90));
	}
	VECTOR3D cameraPosition = mainCamera->GetPosition();
	cameraPosition += velocity.GetNormalized()*CAMERA_MOVE_SPEED*(float)DEMO::timeManager->GetFrameInterval();

	mainCamera->Update(cameraPosition,cameraRotation);

  // change direction of directional light
	VECTOR3D direction = dirLight->GetDirection();
  if(DEMO::inputManager->GetTriggerState(VK_UP))
		direction.z -= (float)DEMO::timeManager->GetFrameInterval()*DIRLIGHT_MOVE_SPEED;
	if(DEMO::inputManager->GetTriggerState(VK_DOWN))
		direction.z += (float)DEMO::timeManager->GetFrameInterval()*DIRLIGHT_MOVE_SPEED;
	if(DEMO::inputManager->GetTriggerState(VK_LEFT))
		direction.x -= (float)DEMO::timeManager->GetFrameInterval()*DIRLIGHT_MOVE_SPEED;
	if(DEMO::inputManager->GetTriggerState(VK_RIGHT))
		direction.x += (float)DEMO::timeManager->GetFrameInterval()*DIRLIGHT_MOVE_SPEED;
	dirLight->SetDirection(direction);

	// toggle help 
	if(DEMO::inputManager->GetSingleTriggerState(VK_F1))
		showHelp = !showHelp;

	// change global illum mode
	if(DEMO::inputManager->GetSingleTriggerState(VK_F2))
		globalIllum->SetGlobalIllumMode(DEFAULT_GIM);
	if(DEMO::inputManager->GetSingleTriggerState(VK_F3))
		globalIllum->SetGlobalIllumMode(DIRECT_ILLUM_ONLY_GIM);
	if(DEMO::inputManager->GetSingleTriggerState(VK_F4))
		globalIllum->SetGlobalIllumMode(INDIRECT_ILLUM_ONLY_GIM);
	if(DEMO::inputManager->GetSingleTriggerState(VK_F5))
		globalIllum->SetGlobalIllumMode(VISUALIZE_GIM);
	if(DEMO::inputManager->GetSingleTriggerState(VK_F6))
		globalIllum->EnableOcclusion(!globalIllum->IsOcclusionEnabled());
	
	// toggle animation of point lights
	if(DEMO::inputManager->GetSingleTriggerState(VK_F7))
	{
		pathLightsAnimated = !pathLightsAnimated;
		for(int i=0;i<NUM_PATH_POINT_LIGHTS;i++)
			pathPointLights[i].SetPaused(!pathLightsAnimated);
	}

	// toggle point lights
	if(DEMO::inputManager->GetSingleTriggerState(VK_F8))
	{
		pathLigthsEnabled = !pathLigthsEnabled;
		for(int i=0;i<NUM_PATH_POINT_LIGHTS;i++)
			pathPointLights[i].SetActive(pathLigthsEnabled);
	}

	// save screen-shot 
	if(DEMO::inputManager->GetSingleTriggerState(VK_F9))
		DEMO::renderer->SaveScreenshot();
}

void APPLICATION::DisplayInfo()
{
	defaultFont->Print(VECTOR2D(-0.95f,0.86f),0.04f,COLOR(0.0f,1.0f,0.0f),"FPS: %.2f",DEMO::timeManager->GetFPS());
	if(showHelp)
	  defaultFont->Print(VECTOR2D(-0.95f,0.78f),0.03f,COLOR(1.0f,1.0f,0.0f),"F1 - Hide help");
	else
    defaultFont->Print(VECTOR2D(-0.95f,0.78f),0.03f,COLOR(1.0f,1.0f,0.0f),"F1 - Show help");

	if(showHelp)
	{
		defaultFont->Print(VECTOR2D(-0.95f,0.72f),0.03f,COLOR(1.0f,1.0f,0.0f),"WASD - Move camera");
		defaultFont->Print(VECTOR2D(-0.95f,0.66f),0.03f,COLOR(1.0f,1.0f,0.0f),"Arrows - Change direction of directional light");
		defaultFont->Print(VECTOR2D(-0.95f,0.6f),0.03f,COLOR(1.0f,1.0f,0.0f),"ESC - Quit application");
    defaultFont->Print(VECTOR2D(-0.95f,0.54f),0.03f,COLOR(1.0f,1.0f,0.0f),"F2 - Default combined output");
		defaultFont->Print(VECTOR2D(-0.95f,0.48f),0.03f,COLOR(1.0f,1.0f,0.0f),"F3 - Direct illumination only");
		defaultFont->Print(VECTOR2D(-0.95f,0.42f),0.03f,COLOR(1.0f,1.0f,0.0f),"F4 - Indirect illumination only");
		defaultFont->Print(VECTOR2D(-0.95f,0.36f),0.03f,COLOR(1.0f,1.0f,0.0f),"F5 - Visualize voxel grid");
		if(globalIllum->IsOcclusionEnabled())
		  defaultFont->Print(VECTOR2D(-0.95f,0.3f),0.03f,COLOR(1.0f,1.0f,0.0f),"F6 - Disable occlusion");
		else
      defaultFont->Print(VECTOR2D(-0.95f,0.3f),0.03f,COLOR(1.0f,1.0f,0.0f),"F6 - Enable occlusion");
		if(pathLightsAnimated)
			defaultFont->Print(VECTOR2D(-0.95f,0.24f),0.03f,COLOR(1.0f,1.0f,0.0f),"F7 - Pause animation of point lights");
		else
			defaultFont->Print(VECTOR2D(-0.95f,0.24f),0.03f,COLOR(1.0f,1.0f,0.0f),"F7 - Resume animation of point lights");
		if(pathLigthsEnabled)
			defaultFont->Print(VECTOR2D(-0.95f,0.18f),0.03f,COLOR(1.0f,1.0f,0.0f),"F8 - Disable point lights");
		else
			defaultFont->Print(VECTOR2D(-0.95f,0.18f),0.03f,COLOR(1.0f,1.0f,0.0f),"F8 - Enable point lights");
    defaultFont->Print(VECTOR2D(-0.95f,0.12f),0.03f,COLOR(1.0f,1.0f,0.0f),"F9 - Screenshot");
	}
}

void APPLICATION::OnRun()
{
	HandleInput();	
		
	DisplayInfo();
	
	if(pathLigthsEnabled && pathLightsAnimated)
	{
		for(int i=0;i<NUM_PATH_POINT_LIGHTS;i++)
			pathPointLights[i].Update();
	}
	
	// quit application by pressing ESCAPE
	if(DEMO::inputManager->GetSingleTriggerState(VK_ESCAPE))
	{
	  Quit();
	}
}

void APPLICATION::OnShutdown()
{
	delete this;
}