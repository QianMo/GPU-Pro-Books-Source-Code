
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <windows.h>
#include "GL/glew.h"
#include <GL/gl.h>

#include <stdio.h>
#include <assert.h>

#include <io.h>
#include <sys/types.h>
#include <sys/stat.h>

#include <vector>
#include <algorithm>

#include "../Main/DemoManager.h"

#include "../Input/InputManager.h"

#include "../Render/TextRenderManager.h"
#include "../Render/RenderManager.h"
#include "../Render/MaterialManager.h"
#include "../Render/RenderObject.h"
#include "../Render/Sphere.h"
#include "../Render/RenderNode.h"
#include "../Render/ShaderManager.h"
#include "../Render/ShadowManager.h"

#include "../Render/ScreenSpaceCurvature.h"

#include "../Level/Light.h"
#include "../Level/Camera.h"

#include "../Util/Matrix4.h"
#include "../Util/Vector3.h"
#include "../Util/Ray.h"
#include "../Util/Math.h"
#include "../Util/ConfigLoader.h"

#include "../Level/LevelLoader.h"

#include "../Graphics/TextureManager.h"

#include "../Physic/Physic.h"
#include "../Physic/Fluid.h"
#include "../Physic/FluidMetaDataManager.h"

#include <iostream>
#include <fstream>
#include <cstring>
#include <sstream>

#include <GL/glut.h>

#include <IL/ilut.h>

float DemoManager::FOV = 40.0f;

#define MSG_TIMEOUT 2.0f


// -----------------------------------------------------------------------------
// ------------------------------- SaveSettingsCB ------------------------------
// -----------------------------------------------------------------------------
void TW_CALL SaveSettingsCB(void* clientData)
{
	DemoManager::Instance()->SaveRenderSettings();
}


// -----------------------------------------------------------------------------
// ---------------------------- SaveFluidSettingsCB ----------------------------
// -----------------------------------------------------------------------------
void TW_CALL SaveFluidSettingsCB(void* clientData)
{
	DemoManager::Instance()->SaveFluidSettings();
}


// -----------------------------------------------------------------------------
// ----------------------------- ReinizializeFluid -----------------------------
// -----------------------------------------------------------------------------
void TW_CALL ReinitializeFluidCB(void* clientData)
{
	DemoManager::Instance()->ReinitializeFluid();
}


// -----------------------------------------------------------------------------
// ----------------------- DemoManager::DemoManager ----------------------------
// -----------------------------------------------------------------------------
DemoManager::DemoManager(void)
{
	isInitialized = false;

	loadScene = false;

	camera = NULL;
	light = NULL;

	screenSpaceCurvature = NULL;

	currentWidth = ConfigLoader::Instance()->GetScreenWidth();
	currentHeight = ConfigLoader::Instance()->GetScreenHeight();

	currentScene = ConfigLoader::Instance()->GetSceneIndex();

	renderUI = true;
}


// -----------------------------------------------------------------------------
// ----------------------- DemoManager::Init -----------------------------------
// -----------------------------------------------------------------------------
void DemoManager::Init(void)
{
	time = 0.0;
	msgTimer = MSG_TIMEOUT;

	MaterialManager::Instance()->LoadMaterialsFromFile("data/util/materials.xml");

	// set render states
	glEnable(GL_CULL_FACE);
	glShadeModel(GL_SMOOTH);
	glEnable(GL_DEPTH_TEST);
	glDepthMask(GL_TRUE);

	glDepthFunc(GL_LEQUAL);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);	
	glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);	

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glDisable(GL_LIGHTING);

	glPixelStorei(GL_PACK_ALIGNMENT, 1);

	timeSum = 0.0f;
	frameRate = 0.0f;
	frameCount = 1;
	sprintf_s(fpsString, "FPS:");

	renderStatistics = false;
	useViewFrustumCulling = false;

	RenderObject::SetFrustumCulling(useViewFrustumCulling);

	currentRenderMode = RENDER_NONE;

	camera = new Camera();
	camera->Init();

	light = new Light();
	light->Init(Vector3(0.0f, 250.0f, 10.0f));

	ShaderManager::Instance()->Init(light);

	// load scene
	LoadScene(true);

	glEnable(GL_TEXTURE_2D);

	// Initiate Shadow Mapping
	ShadowManager::Instance()->Init(1024.0f, 4.0f);

	const char* filename[6];
	filename[0] = "data/textures/west.dds";
	filename[1] = "data/textures/east.dds";
	filename[2] = "data/textures/up.dds";
	filename[3] = "data/textures/down.dds";
	filename[4] = "data/textures/south.dds";
	filename[5] = "data/textures/north.dds";

	cubeMap = TextureManager::Instance()->LoadCubeMap(filename, false);
	ShaderManager::Instance()->SetParameterTexture(ShaderManager::SP_FLUID_CUBE_MAP, cubeMap);

	//////////////////////////////////////////////////////////////////////////
	// create gui

	char guiInit[128];
	sprintf_s(guiInit, " Settings color='150 50 50 50' size='340 440' position='10 %d' iconified='true' ", ConfigLoader::Instance()->GetScreenHeight()-470);

	bar = TwNewBar("Settings");
	TwDefine(" GLOBAL fontSize=3 iconAlign=horizontal ");
	TwDefine(guiInit);

	renderMode = (unsigned int)ScreenSpaceCurvature::RENDER_MODE_ALL;

	TwEnumVal renderModeEV[] =
	{
		{ ScreenSpaceCurvature::RENDER_MODE_BACKGROUND_LAYER, "background layer"},
		{ ScreenSpaceCurvature::RENDER_MODE_BACK_WATER_LAYER, "+ back water layer"},
		{ ScreenSpaceCurvature::RENDER_MODE_FOAM_LAYER, "+ foam Layer"},
		{ ScreenSpaceCurvature::RENDER_MODE_FRONT_WATER_LAYER, "+ front water layer"},
		{ ScreenSpaceCurvature::RENDER_MODE_ALL, "+ highlights"},
	};
	TwType renderModeType = TwDefineEnum("RenderMode", renderModeEV, 5);
	TwAddVarRW(bar, "RenderMode", renderModeType, &renderMode, " help='Shows different stages of the compositing algorithm.' ");

	TwAddVarRW(bar, "BaseColor", TW_TYPE_COLOR3F, &renderDescription.baseColor, "help='BaseColor'");
	TwAddVarRW(bar, "ColorFalloff", TW_TYPE_COLOR4F, &renderDescription.colorFalloff, "help='ColorFalloff'");
	TwAddVarRW(bar, "FalloffScale", TW_TYPE_FLOAT, &renderDescription.falloffScale, "min=0 max=1 step=0.0001 help='FalloffScale'");

	TwAddSeparator(bar, "sep0", NULL);

	TwAddVarRW(bar, "SpecularColor", TW_TYPE_COLOR3F, &renderDescription.specularColor, "help='SpecularColor'");
	TwAddVarRW(bar, "SpecularShininess", TW_TYPE_FLOAT, &renderDescription.specularShininess, "min=10 max=5000 step=1.0 help='SpecularShininess'");

	TwAddSeparator(bar, "sep1", NULL);

	TwAddVarRW(bar, "SprayColor", TW_TYPE_COLOR4F, &renderDescription.sprayColor, "help='SprayColor'");
	TwAddVarRW(bar, "DensityThreshold", TW_TYPE_FLOAT, &renderDescription.densityThreshold, "min=0 max=10000 step=1.0 help='DensityThreshold'");

	TwAddSeparator(bar, "sep2", NULL);

	TwAddVarRW(bar, "FresnelBias", TW_TYPE_FLOAT, &renderDescription.fresnelBias, "min=0 max=10 step=0.01 help='FresnelBias'");
	TwAddVarRW(bar, "FresnelScale", TW_TYPE_FLOAT, &renderDescription.fresnelScale, "min=0 max=10 step=0.01 help='FresnelScale'");
	TwAddVarRW(bar, "FresnelPower", TW_TYPE_FLOAT, &renderDescription.fresnelPower, "min=0 max=10 step=0.01 help='FresnelPower'");

	TwAddSeparator(bar, "sep3", NULL);

	TwAddVarRW(bar, "FoamBackColor", TW_TYPE_COLOR3F, &renderDescription.foamBackColor, "help='FoamBackColor'");
	TwAddVarRW(bar, "FoamFrontColor", TW_TYPE_COLOR3F, &renderDescription.foamFrontColor, "help='FoamFrontColor'");
	TwAddVarRW(bar, "FoamFalloffScale", TW_TYPE_FLOAT, &renderDescription.foamFalloffScale, "min=0 max=1 step=0.0001 help='FoamFalloffScale'");
	TwAddVarRW(bar, "FoamDepthThreshold", TW_TYPE_FLOAT, &renderDescription.foamDepthThreshold, "min=0 max=50 step=0.1 help='FoamDepthThreshold'");
	TwAddVarRW(bar, "FoamFrontFalloffScale", TW_TYPE_FLOAT, &renderDescription.foamFrontFalloffScale, "min=0 max=2 step=0.001 help='FoamFrontFalloffScale'");
	TwAddVarRW(bar, "FoamLifetime", TW_TYPE_FLOAT, &renderDescription.foamLifetime, "min=0 max=50 step=0.01 help='FoamLifetime'");

	TwAddSeparator(bar, "sep4", NULL);

	TwAddVarRW(bar, "ParticleSize", TW_TYPE_FLOAT, &renderDescription.particleSize, "min=0 max=5 step=0.0001 help='ParticleSize'");
	TwAddVarRW(bar, "ThicknessRefraction", TW_TYPE_FLOAT, &renderDescription.thicknessRefraction, "min=0 max=10 step=0.01 help='ThicknessRefraction'");
	TwAddVarRW(bar, "FluidThicknessScale", TW_TYPE_FLOAT, &renderDescription.fluidThicknessScale, "min=0 max=10 step=0.0001 help='FluidThicknessScale'");
	TwAddVarRW(bar, "FoamThicknessScale", TW_TYPE_FLOAT, &renderDescription.foamThicknessScale, "min=0 max=10 step=0.0001 help='foamThicknessScale'");
	TwAddVarRW(bar, "WorldSpaceKernelRadius", TW_TYPE_FLOAT, &renderDescription.worldSpaceKernelRadius, "min=0 max=10 step=0.01 help='WorldSpaceKernelRadius'");
	TwAddVarRW(bar, "WeberNumberThreshold", TW_TYPE_FLOAT, &renderDescription.foamThreshold, "min=0 max=1000 step=0.1 help='WeberNumberThreshold'");

	TwAddSeparator(bar, "sep5", NULL);

	TwAddVarRW(bar, "UseNoise", TW_TYPE_BOOLCPP, &renderDescription.useNoise, " label='UseNoise' false='OFF' true='ON'");
	TwAddVarRW(bar, "NoiseDepthFalloff", TW_TYPE_FLOAT, &renderDescription.noiseDepthFalloff, "min=0 max=15 step=0.01 help='NoiseDepthFalloff'");
	TwAddVarRW(bar, "NormalNoiseWeight", TW_TYPE_FLOAT, &renderDescription.normalNoiseWeight, "min=0 max=1 step=0.001 help='NormalNoiseWeight'");

	TwAddSeparator(bar, "sep6", NULL);

	TwAddVarRW(bar, "Render AABB", TW_TYPE_BOOLCPP, &renderDescription.renderAABB, " label='Render AABB' false='OFF' true='ON'");
	TwAddButton(bar, "Reset Settings", SaveSettingsCB, this, " label='Save Settings' ");

	//////////////////////////////////////////////////////////////////////////

	sprintf_s(guiInit, " Fluid color='150 50 50 50' size='340 440' position='360 %d' iconified='true' ", ConfigLoader::Instance()->GetScreenHeight()-470);

	fluidBar = TwNewBar("Fluid");
	TwDefine(" GLOBAL fontSize=3 iconAlign=horizontal ");
	TwDefine(guiInit);

	TwAddButton(fluidBar, "Reinitialize", ReinitializeFluidCB, this, 
		" label='Update fluid' help='Restart the fluid simulation with new parameters.' ");

	TwEnumVal simulationMethodEV[] =
	{
		{ NX_F_NO_PARTICLE_INTERACTION,	"NO_PARTICLE_INTERACTION" },
		{ NX_F_MIXED_MODE,				"MIXED_MODE"			  },
		{ NX_F_SPH,						"SPH"					  },
	};
	TwType simulationMethodType = TwDefineEnum("SimulationMethod", simulationMethodEV, 3);

	TwAddVarRW(fluidBar, "MaxParticles", TW_TYPE_UINT32, &fluidDescription.maxParticles, " min=0 max=65535 step=1000 help='Maximal number of particles for the fluid used in the simulation.' ");
	TwAddVarRW(fluidBar, "NumReserveParticles", TW_TYPE_UINT32, &fluidDescription.numReserveParticles, " min=0 max=65535 step=100 help='Defines the number of particles which are reserved for creation at runtime.' ");
	TwAddVarRW(fluidBar, "RestParticlesPerMeter", TW_TYPE_FLOAT, &fluidDescription.restParticlesPerMeter, " min=0 max=10 step=0.01 help='The particle resolution given as particles per linear meter measured when the fluid is in its rest state (relaxed).' ");
	TwAddVarRW(fluidBar, "RestDensity", TW_TYPE_FLOAT, &fluidDescription.restDensity, " min=0 max=100000 step=10 help='Target density for the fluid (water is about 1000).' ");
	TwAddVarRW(fluidBar, "KernelRadiusMultiplier", TW_TYPE_FLOAT, &fluidDescription.kernelRadiusMultiplier, " min=0 max=10 step=0.01 help='Radius of sphere of influence for particle interaction.' ");
	TwAddVarRW(fluidBar, "MotionLimitMultiplier", TW_TYPE_FLOAT, &fluidDescription.motionLimitMultiplier, " min=0 max=20 step=0.01 help='Maximal distance a particle is allowed to travel within one timestep.' ");
	TwAddVarRW(fluidBar, "CollisionDistanceMultiplier", TW_TYPE_FLOAT, &fluidDescription.collisionDistanceMultiplier, " min=0 max=10 step=0.01 help='Defines the distance between particles and collision geometry, which is maintained during simulation.' ");
	TwAddVarRW(fluidBar, "Stiffness", TW_TYPE_FLOAT, &fluidDescription.stiffness, " min=0 max=100 step=0.01 help='The stiffness of the particle interaction related to the pressure.' ");
	TwAddVarRW(fluidBar, "Viscosity", TW_TYPE_FLOAT, &fluidDescription.viscosity, " min=0 max=100 step=0.01 help='The viscosity of the fluid defines its viscous behavior.' ");
	TwAddVarRW(fluidBar, "SurfaceTension", TW_TYPE_FLOAT, &fluidDescription.surfaceTension, " min=0 max=10 step=0.01 help='The surfaceTension of the fluid defines an attractive force between particles.' ");
	TwAddVarRW(fluidBar, "Damping", TW_TYPE_FLOAT, &fluidDescription.damping, " min=0 max=10 step=0.01 help='Velocity damping constant, which is globally applied to each particle.' ");
	TwAddVarRW(fluidBar, "FadeInTime", TW_TYPE_FLOAT, &fluidDescription.fadeInTime, " min=0 max=10 step=0.01 help='Defines a timespan for the particle fade-in.' ");
	TwAddVarRW(fluidBar, "RestitutionForStaticShapes", TW_TYPE_FLOAT, &fluidDescription.restitutionForStaticShapes, " min=0 max=1 step=0.01 help='Defines the restitution coefficient used for collisions of the fluid particles with static shapes' ");
	TwAddVarRW(fluidBar, "DynamicFrictionForStaticShapes", TW_TYPE_FLOAT, &fluidDescription.dynamicFrictionForStaticShapes, " min=0 max=1 step=0.01 help='Defines the dynamic friction of the fluid regarding the surface of a static shape.' ");
	TwAddVarRW(fluidBar, "RestitutionForDynamicShapes", TW_TYPE_FLOAT, &fluidDescription.restitutionForDynamicShapes, " min=0 max=1 step=0.01 help='Defines the restitution coefficient used for collisions of the fluid particles with dynamic shapes.' ");
	TwAddVarRW(fluidBar, "DynamicFrictionForDynamicShapes", TW_TYPE_FLOAT, &fluidDescription.dynamicFrictionForDynamicShapes, " min=0 max=1 step=0.01 help='Defines the dynamic friction of the fluid regarding the surface of a dynamic shape.' ");
	TwAddVarRW(fluidBar, "CollisionResponseCoefficient", TW_TYPE_FLOAT, &fluidDescription.collisionResponseCoefficient, " min=0 max=10 step=0.001 help='Defines a factor for the impulse transfer from fluid to colliding rigid bodies. ' ");
	TwAddVarRW(fluidBar, "SimulationMethod", simulationMethodType, &fluidDescription.simulationMethod, " help='Defines whether or not particle interactions are considered in the simulation.' ");

	TwAddButton(fluidBar, "Save settings", SaveFluidSettingsCB, this, 
		" label='Save fluid parameters' help='Save fluid parameters.' ");

	isInitialized = true;
}


// -----------------------------------------------------------------------------
// ----------------------- DemoManager::Reshape --------------------------------
// -----------------------------------------------------------------------------
void DemoManager::Reshape(int& w, int& h)
{
	// prevent a division by zero
	if (h==0)
		h=1;

	currentWidth = w;
	currentHeight = h;

	glViewport(0,0,w,h);

	TextRenderManager::Instance()->ResetProjection(w, h);

	if (screenSpaceCurvature)
		screenSpaceCurvature->SetWindowSize(w, h);
}


// -----------------------------------------------------------------------------
// ----------------------- DemoManager::Update ---------------------------------
// -----------------------------------------------------------------------------
void DemoManager::Update(void)
{
	if (!isInitialized || loadScene)
		return;

	// delta time
	deltaTime = Math::Clamp(timer.GetDeltaTime(), 0.0f, 0.1f);
	timer.Start();

	frameRate += deltaTime;

	time += deltaTime;
	timeSum += deltaTime;
	frameCount++;

	if (resetFluid)
	{
		Physic::Instance()->ExitFluid();

		unsigned int i;
		for (i=0; i<dynamicActorIDs.size(); i++)
		{
			char nodeName[32];
			sprintf_s(nodeName, "DynamicElement%.4d", dynamicActorIDs[i]);

			RenderManager::Instance()->RemoveNodeFromRoot(nodeName);
			Physic::Instance()->ReleaseActor(dynamicActorIDs[i]);
		}

		SetupDynamicActors();

		Physic::Instance()->InitFluid(currentScene, &fluidDescription);

		screenSpaceCurvature->SetFluid(Physic::Instance()->GetFluid());

		resetFluid = false;
	}

	ShadowManager::Instance()->Update(deltaTime);

	// set projection here, to be able to do the frustum culling in the update method
	SetProjectionMatrix();

	// store projection matrix for later usage
	glGetFloatv(GL_PROJECTION_MATRIX, projectionMatrix.entry);

	// Do simulation step in the physics world
	Physic::Instance()->Simulate(deltaTime);

	// Process the userinput
	UpdateInput(deltaTime);

	light->Update(Vector3(0.0f, 250.0f, 10.0f), Vector3(0.0f, 0.0f, 0.0f), 90.0f, 1.0f, 100.0f, 300.0f);

	// calculate view frustum
	CalculateViewFrustum();

	// set the current ball ray options and then update the rendering system
	RenderObject::SetViewFrustum(viewFrustumPlanes);

	// update the rendering stuff
	RenderManager::Instance()->Update(deltaTime);

	// update physics stuff
	Physic::Instance()->Update(deltaTime);

	// update fluid renderer
	screenSpaceCurvature->SetViewFrustum(viewFrustumPlanes);
	screenSpaceCurvature->SetRenderDescription(renderDescription);
	screenSpaceCurvature->SetRenderMode((ScreenSpaceCurvature::RenderMode)renderMode);
	screenSpaceCurvature->Update(deltaTime);

	// tell glut to render the scene
	glutPostRedisplay();
}


// -----------------------------------------------------------------------------
// --------------------- DemoManager::PerformCpuCaluations ---------------------
// -----------------------------------------------------------------------------
void DemoManager::PerformCpuCaluations(float deltaTime)
{
	screenSpaceCurvature->UpdateMetaData(deltaTime);
}


// -----------------------------------------------------------------------------
// ----------------------- DemoManager::Render ---------------------------------
// -----------------------------------------------------------------------------
void DemoManager::Render(void)
{
	if (!isInitialized || loadScene)
	{
		glClearColor (0.0, 0.0, 0.0, 1.0);
		glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

		TextRenderManager::Instance()->Begin();
		{
			TextRenderManager::Instance()->RenderText("loading...", TextRenderManager::FONT_9_BY_15, currentWidth - 140, currentHeight - 40, 0.0f, 1.0f, 0.0f);
		}
		TextRenderManager::Instance()->End();
	}
	else
	{
		RenderPreProcessing();
		RenderFluid();
		RenderToScreen();

		ShaderManager::Instance()->DisableShader();

		RenderDebugStuff();

		// render gui
		if (renderUI)
			TwDraw();
	}

	glutSwapBuffers();
}


// -----------------------------------------------------------------------------
// ----------------------- DemoManager::PostRenderUpdate -----------------------
// -----------------------------------------------------------------------------
void DemoManager::PostRenderUpdate(void)
{
	if (!isInitialized)
		Init();

	if (loadScene)
	{
		LoadScene();
		loadScene = false;
	}
}


// -----------------------------------------------------------------------------
// ----------------------- DemoManager::Exit -----------------------------------
// -----------------------------------------------------------------------------
void DemoManager::Exit(void)
{
	ShadowManager::Instance()->Exit();
	ShaderManager::Instance()->Exit();

	Physic::Instance()->ExitFluid();
	Physic::Instance()->Exit();

	Physic::Instance()->ReleaseLevel();
	RenderManager::Instance()->Exit();

	FluidMetaDataManager::Instance()->Exit();

	if (screenSpaceCurvature)
		delete screenSpaceCurvature;

	if (camera)
	{
		camera->Exit();

		delete camera;
		camera = NULL;
	}
	if (light)
	{
		delete light;
		light = NULL;
	}

	glDeleteTextures(1, &cubeMap);
	TextureManager::Instance()->DeleteAllTextures();
}


// -----------------------------------------------------------------------------
// --------------------------- DemoManager::LoadScene --------------------------
// -----------------------------------------------------------------------------
void DemoManager::LoadScene(bool initFluidMetaData)
{
	char fileName[64];

	// Clean up
	Physic::Instance()->ExitFluid();
	Physic::Instance()->Exit();

	Physic::Instance()->ReleaseLevel();
	RenderManager::Instance()->Exit();

	// Init physis engine
	Physic::Instance()->Init();

	// Init fluid meta data manager
	if (initFluidMetaData)
		FluidMetaDataManager::Instance()->Init();

	// Init render manager
	RenderManager::Instance()->Init();

	printf("loading scene\n\t");

	// Load scene
	sprintf_s(fileName, "data/levels/scene%.2d.xml", currentScene+1);
	LevelLoader::Instance()->LoadLevel(fileName, (currentScene == 1));

	camera->SetCameraPosition(LevelLoader::Instance()->GetCameraPosition());
	camera->SetCameraDirection(LevelLoader::Instance()->GetCameraDirection());

	printf("\nloading render and fluid description\n");

	sprintf_s(fileName, "data/levels/renderDescription%.2d.xml", currentScene+1);
	renderDescription.LoadFromFile(fileName);

	sprintf_s(fileName, "data/levels/fluidDescription%.2d.xml", currentScene+1);
	fluidDescription.LoadFromFile(fileName);

	// add dynamic objects to the scene
	SetupDynamicActors();

	printf("initializing physic\n");

	// after loading the level we can init the fluid (because all static actors which are in the level are loaded)
	Physic::Instance()->InitFluid(currentScene, &fluidDescription);
	resetFluid = false;

	// Init fluid renderer
	if (!screenSpaceCurvature)
	{
		screenSpaceCurvature = new ScreenSpaceCurvature();
		screenSpaceCurvature->SetFluid(Physic::Instance()->GetFluid());
		screenSpaceCurvature->Init();

		screenSpaceCurvature->SetFieldOfView(FOV);
		screenSpaceCurvature->SetWindowSize(currentWidth, currentHeight);
	}
	else
		screenSpaceCurvature->SetFluid(Physic::Instance()->GetFluid());

	printf("done\n");
}


// -----------------------------------------------------------------------------
// ---------------------- DemoManager::SetupDynamicActors ----------------------
// -----------------------------------------------------------------------------
void DemoManager::SetupDynamicActors(void)
{
	dynamicActorIDs.clear();

	const std::vector<FluidMetaDataManager::DynamicObject*> dynamicObjects = FluidMetaDataManager::Instance()->GetDynamicObjects(currentScene);

	unsigned int i;
	for (i=0; i<dynamicObjects.size(); i++)
	{
		unsigned int uniqueTmp;
		unsigned int keyTmp;

		switch (dynamicObjects[i]->type)
		{
		case FluidMetaDataManager::DOT_SPHERE:
			{
				RenderManager::Instance()->AddSphere(uniqueTmp, keyTmp, dynamicObjects[i]->size, 50, dynamicObjects[i]->position, 1);
				dynamicActorIDs.push_back(keyTmp);
			}
			break;

		case FluidMetaDataManager::DOT_BOX:
			{
				RenderManager::Instance()->AddBox(uniqueTmp, keyTmp, dynamicObjects[i]->position,
					dynamicObjects[i]->size*2.0f, dynamicObjects[i]->size*2.0f, dynamicObjects[i]->size*2.0f,
					Vector3(1.0f, 1.0f, 1.0f), Vector3(1.0f, 1.0f, 1.0f), 1);
				dynamicActorIDs.push_back(keyTmp);
			}
			break;

		default:
			assert(false);
		}
	}
}


// -----------------------------------------------------------------------------
// ------------------- DemoManager::SetupPerspectiveRendering ------------------
// -----------------------------------------------------------------------------
void DemoManager::SetupPerspectiveRendering(void)
{
	SetProjectionMatrix();
	camera->SetViewMatrix();
}


// -----------------------------------------------------------------------------
// ----------------- DemoManager::RenderPreProcessing --------------------------
// -----------------------------------------------------------------------------
void DemoManager::RenderPreProcessing(void)
{
	unsigned int passCount = ShadowManager::Instance()->GetPassCount();
	ShadowManager::Instance()->BeginShadow();

	unsigned int i;
	for (i=0; i<passCount; i++)
	{
		ShadowManager::Instance()->ShadowPass(i);
		RenderShadowCasters();
	}

	ShadowManager::Instance()->EndShadow();

	ShadowManager::Instance()->ComputeConvolutionMap();

	// render the scene into a texture
	screenSpaceCurvature->BeginRenderScene();
	{
		currentRenderMode = RENDER_SCENE;
		RenderScene();
	}
	screenSpaceCurvature->EndRenderScene();
}


// -----------------------------------------------------------------------------
// ----------------------- DemoManager::RenderFluid ----------------------------
// -----------------------------------------------------------------------------
void DemoManager::RenderFluid(void)
{
	ShaderManager::Instance()->DisableShader();

	Matrix4 viewMatrix = camera->GetCameraMatrix();

	Matrix4 invView = viewMatrix;
	Matrix4 invProj = projectionMatrix;
	Matrix4 invViewProj = viewMatrix*projectionMatrix;

	invView.Invert();
	invProj.Invert();
	invViewProj.Invert();

	ShaderManager::Instance()->SetMatrixParameterfc(ShaderManager::SP_FLUID_INV_VIEW, invView.entry);
	ShaderManager::Instance()->SetMatrixParameterfc(ShaderManager::SP_FLUID_INV_PROJ, invProj.entry);
	ShaderManager::Instance()->SetMatrixParameterfc(ShaderManager::SP_FLUID_INV_VIEWPROJ, invViewProj.entry);

	Vector3 lightPosition = Vector3(0.0f, 250.0f, 10.0f);
	lightPosition = viewMatrix*lightPosition;
	screenSpaceCurvature->SetLightPosition(lightPosition);

	screenSpaceCurvature->Render();

	ShaderManager::Instance()->DisableShader();
}


// -----------------------------------------------------------------------------
// ------------------ DemoManager::RenderShadowCasters ------------------------
// -----------------------------------------------------------------------------
void DemoManager::RenderShadowCasters(void)
{
	RenderManager::Instance()->Render();
	Physic::Instance()->Render(false);
}


// -----------------------------------------------------------------------------
// -------------------- DemoManager::RenderScene -------------------------------
// -----------------------------------------------------------------------------
void DemoManager::RenderScene(void)
{
	SetProjectionMatrix();

	glClearColor (0.0, 0.0, 0.0, 1.0);
	glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

	// render the skybox
	RenderManager::Instance()->RenderSkyBox(camera);

	// camera and light stuff
	camera->SetViewMatrix();

	Matrix4 viewMatrix = camera->GetCameraMatrix();

	ShadowManager::Instance()->PrepareFinalPass(viewMatrix, 0);

	ShaderManager::Instance()->SetMatrixParameterfc(ShaderManager::SP_VIEW, viewMatrix.entry);
	{
		RenderManager::Instance()->Render();
	}

	ShaderManager::Instance()->DisableShader();

	Physic::Instance()->Render(true);
}


// -----------------------------------------------------------------------------
// ------------------------ DemoManager::RenderToScreen ------------------------
// -----------------------------------------------------------------------------
void DemoManager::RenderToScreen()
{
	Vector2 bufferSize = Vector2((float)currentWidth, (float)currentHeight);

	glClear(GL_DEPTH_BUFFER_BIT|GL_COLOR_BUFFER_BIT);
	glViewport(0, 0, currentWidth, currentHeight);

	ShaderManager::Instance()->SetParameterTexture(ShaderManager::SP_POST_SCENE_MAP, screenSpaceCurvature->GetResultTexture());
	ShaderManager::Instance()->EnableShader(ShaderManager::SHADER_EFFECT_POST_PROCESS, "RenderScreenBuffer");

	glBegin(GL_QUADS);
	{
		glTexCoord2f(0.0f, 0.0f);
		glVertex3i (-1.0f, -1.0f, 0.0f);

		glTexCoord2f(bufferSize.x, 0.0f);
		glVertex3i (1.0f, -1.0f, 0.0f);

		glTexCoord2f(bufferSize.x, bufferSize.y);
		glVertex3i (1.0f, 1.0f, 0.0f);

		glTexCoord2f(0.0f, bufferSize.y);
		glVertex3i (-1.0f, 1.0f, 0.0f);
	}
	glEnd();

	ShaderManager::Instance()->DisableShader();
}


// -----------------------------------------------------------------------------
// --------------------- DemoManager::RenderDebugStuff -------------------------
// -----------------------------------------------------------------------------
void DemoManager::RenderDebugStuff(void)
{
	glBindTexture(GL_TEXTURE_2D, NULL);
	glBindTexture(GL_TEXTURE_RECTANGLE_ARB, NULL);
	ShaderManager::Instance()->DisableShader();

	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	// if one second has passed update the fpsString and reset the counter variables
	if (timeSum >= 1.0f)
	{
		sprintf_s(fpsString, "FPS: %d (%f ms)", frameCount, deltaTime*1000.0f);
		timeSum = 0.0f;
		frameRate = 0.0f;
		frameCount = 0;
	}

	TextRenderManager::Instance()->Begin();
	{
		if (msgTimer < MSG_TIMEOUT)
		{
			msgTimer += deltaTime;

			TextRenderManager::Instance()->RenderText(msgString, TextRenderManager::FONT_9_BY_15, 10, 20, 0.0f, 1.0f, 0.0f);
		}

		if (renderStatistics)
		{
			TextRenderManager::Instance()->RenderText(fpsString, TextRenderManager::FONT_9_BY_15, 10, 40, 1.0f, 1.0f, 0.0f);

			sprintf_s(particleCountString, "ParticleCount: %d", Physic::Instance()->GetFluid()->GetFluidBufferNum());
			TextRenderManager::Instance()->RenderText(particleCountString, TextRenderManager::FONT_9_BY_15, 10, 60, 1.0f, 1.0f, 0.0f);
		}
	}
	TextRenderManager::Instance()->End();
}


// -----------------------------------------------------------------------------
// -------------------------- DemoManager::SetDebugMsg -------------------------
// -----------------------------------------------------------------------------
void DemoManager::SetDebugMsg(const char* str)
{
	msgTimer = 0.0f;
	sprintf_s(msgString, str);
}


// -----------------------------------------------------------------------------
// -------------------------- DemoManager::UpdateInput -------------------------
// -----------------------------------------------------------------------------
void DemoManager::UpdateInput(float deltaTime)
{
	camera->Update(deltaTime);

	if (InputManager::Instance()->IsKeyPressedAndReset(KEY_F1))
	{
		renderStatistics = !renderStatistics;

		msgTimer = 0.0f;
		if (renderStatistics)
			SetDebugMsg("Statistics enabled");
		else
			SetDebugMsg("Statistics disabled");
	}

	if (InputManager::Instance()->IsKeyPressedAndReset(KEY_F5))
	{
		if (currentScene > 0)
		{
			currentScene--;
			loadScene = true;
		}
	}
	else if (InputManager::Instance()->IsKeyPressedAndReset(KEY_F6))
	{
		if (currentScene < (NUM_SCENES-1))
		{
			currentScene++;
			loadScene = true;
		}
	}

	if (InputManager::Instance()->IsKeyPressedAndReset('r'))
	{
		SetDebugMsg("Fluid reset");
		resetFluid = true;
	}

	// exit
	if (InputManager::Instance()->IsKeyPressedAndReset(KEY_ESCAPE))
	{
		Exit();
		exit(EXIT_SUCCESS);
	}
}


// -----------------------------------------------------------------------------
// ---------------------- DemoManager::SetProjectionMatrix ---------------------
// -----------------------------------------------------------------------------
void DemoManager::SetProjectionMatrix(void) const
{
	float nearZ = 10.0f;
	float farZ = 2000.0f;

	float fov = 40.0f;

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	gluPerspective(fov, (GLdouble) currentWidth / (GLdouble) currentHeight, nearZ, farZ);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}


// -----------------------------------------------------------------------------
// ----------------------- DemoManager::calculateViewFrustum -------------------
// -----------------------------------------------------------------------------
void DemoManager::CalculateViewFrustum(void)
{
	Matrix4 projection;
	Matrix4 modelViewProjection;
	float factor;

	glGetFloatv(GL_PROJECTION_MATRIX, projection.entry);
	modelViewProjection = camera->GetFixedCameraMatrix() * projection;

	// far clipping plane
	viewFrustumPlanes[0].plane[0] = modelViewProjection.entry[ 3] - modelViewProjection.entry[ 2];
	viewFrustumPlanes[0].plane[1] = modelViewProjection.entry[ 7] - modelViewProjection.entry[ 6];
	viewFrustumPlanes[0].plane[2] = modelViewProjection.entry[11] - modelViewProjection.entry[10];
	viewFrustumPlanes[0].plane[3] = modelViewProjection.entry[15] - modelViewProjection.entry[14];

	factor = (float) sqrt( viewFrustumPlanes[0].plane[0] * viewFrustumPlanes[0].plane[0] +  
		viewFrustumPlanes[0].plane[1] * viewFrustumPlanes[0].plane[1] + 
		viewFrustumPlanes[0].plane[2] * viewFrustumPlanes[0].plane[2] );

	viewFrustumPlanes[0].plane[0] /= factor;
	viewFrustumPlanes[0].plane[1] /= factor;
	viewFrustumPlanes[0].plane[2] /= factor;
	viewFrustumPlanes[0].plane[3] /= factor;

	// near clipping plane
	viewFrustumPlanes[1].plane[0] = modelViewProjection.entry[ 3] + modelViewProjection.entry[ 2];
	viewFrustumPlanes[1].plane[1] = modelViewProjection.entry[ 7] + modelViewProjection.entry[ 6];
	viewFrustumPlanes[1].plane[2] = modelViewProjection.entry[11] + modelViewProjection.entry[10];
	viewFrustumPlanes[1].plane[3] = modelViewProjection.entry[15] + modelViewProjection.entry[14];

	factor = (float) sqrt( viewFrustumPlanes[1].plane[0] * viewFrustumPlanes[1].plane[0] + 
		viewFrustumPlanes[1].plane[1] * viewFrustumPlanes[1].plane[1] + 
		viewFrustumPlanes[1].plane[2] * viewFrustumPlanes[1].plane[2] );

	viewFrustumPlanes[1].plane[0] /= factor;
	viewFrustumPlanes[1].plane[1] /= factor;
	viewFrustumPlanes[1].plane[2] /= factor;
	viewFrustumPlanes[1].plane[3] /= factor;

	// bottom clipping plane
	viewFrustumPlanes[2].plane[0] = modelViewProjection.entry[ 3] + modelViewProjection.entry[ 1];
	viewFrustumPlanes[2].plane[1] = modelViewProjection.entry[ 7] + modelViewProjection.entry[ 5];
	viewFrustumPlanes[2].plane[2] = modelViewProjection.entry[11] + modelViewProjection.entry[ 9];
	viewFrustumPlanes[2].plane[3] = modelViewProjection.entry[15] + modelViewProjection.entry[13];

	factor = (float) sqrt( viewFrustumPlanes[2].plane[0] * viewFrustumPlanes[2].plane[0] + 
		viewFrustumPlanes[2].plane[1] * viewFrustumPlanes[2].plane[1] + 
		viewFrustumPlanes[2].plane[2] * viewFrustumPlanes[2].plane[2] );

	viewFrustumPlanes[2].plane[0] /= factor;
	viewFrustumPlanes[2].plane[1] /= factor;
	viewFrustumPlanes[2].plane[2] /= factor;
	viewFrustumPlanes[2].plane[3] /= factor;

	// top clipping plane
	viewFrustumPlanes[3].plane[0] = modelViewProjection.entry[ 3] - modelViewProjection.entry[ 1];
	viewFrustumPlanes[3].plane[1] = modelViewProjection.entry[ 7] - modelViewProjection.entry[ 5];
	viewFrustumPlanes[3].plane[2] = modelViewProjection.entry[11] - modelViewProjection.entry[ 9];
	viewFrustumPlanes[3].plane[3] = modelViewProjection.entry[15] - modelViewProjection.entry[13];

	factor = (float) sqrt( viewFrustumPlanes[3].plane[0] * viewFrustumPlanes[3].plane[0] + 
		viewFrustumPlanes[3].plane[1] * viewFrustumPlanes[3].plane[1] + 
		viewFrustumPlanes[3].plane[2] * viewFrustumPlanes[3].plane[2] );

	viewFrustumPlanes[3].plane[0] /= factor;
	viewFrustumPlanes[3].plane[1] /= factor;
	viewFrustumPlanes[3].plane[2] /= factor;
	viewFrustumPlanes[3].plane[3] /= factor;

	// right clipping plane
	viewFrustumPlanes[4].plane[0] = modelViewProjection.entry[ 3] - modelViewProjection.entry[ 0];
	viewFrustumPlanes[4].plane[1] = modelViewProjection.entry[ 7] - modelViewProjection.entry[ 4];
	viewFrustumPlanes[4].plane[2] = modelViewProjection.entry[11] - modelViewProjection.entry[ 8];
	viewFrustumPlanes[4].plane[3] = modelViewProjection.entry[15] - modelViewProjection.entry[12];

	factor = (float) sqrt( viewFrustumPlanes[4].plane[0] * viewFrustumPlanes[4].plane[0] + 
		viewFrustumPlanes[4].plane[1] * viewFrustumPlanes[4].plane[1] + 
		viewFrustumPlanes[4].plane[2] * viewFrustumPlanes[4].plane[2] );

	viewFrustumPlanes[4].plane[0] /= factor;
	viewFrustumPlanes[4].plane[1] /= factor;
	viewFrustumPlanes[4].plane[2] /= factor;
	viewFrustumPlanes[4].plane[3] /= factor;

	// left clipping plane
	viewFrustumPlanes[5].plane[0] = modelViewProjection.entry[ 3] + modelViewProjection.entry[ 0];
	viewFrustumPlanes[5].plane[1] = modelViewProjection.entry[ 7] + modelViewProjection.entry[ 4];
	viewFrustumPlanes[5].plane[2] = modelViewProjection.entry[11] + modelViewProjection.entry[ 8];
	viewFrustumPlanes[5].plane[3] = modelViewProjection.entry[15] + modelViewProjection.entry[12];

	factor = (float) sqrt( viewFrustumPlanes[5].plane[0] * viewFrustumPlanes[5].plane[0] + 
		viewFrustumPlanes[5].plane[1] * viewFrustumPlanes[5].plane[1] + 
		viewFrustumPlanes[5].plane[2] * viewFrustumPlanes[5].plane[2] );

	viewFrustumPlanes[5].plane[0] /= factor;
	viewFrustumPlanes[5].plane[1] /= factor;
	viewFrustumPlanes[5].plane[2] /= factor;
	viewFrustumPlanes[5].plane[3] /= factor;
}


// -----------------------------------------------------------------------------
// ---------------------- DemoManager::SaveRenderSettings ----------------------
// -----------------------------------------------------------------------------
void DemoManager::SaveRenderSettings(void) const
{
	char fileName[64];
	sprintf_s(fileName, "data/levels/renderDescription%.2d.xml", currentScene+1);
	renderDescription.SaveToFile(fileName);
}


// -----------------------------------------------------------------------------
// ----------------------- DemoManager::SaveFluidSettings ----------------------
// -----------------------------------------------------------------------------
void DemoManager::SaveFluidSettings(void)
{
	if (fluidDescription.IsValid())
	{
		char fileName[64];
		sprintf_s(fileName, "data/levels/fluidDescription%.2d.xml", currentScene+1);
		fluidDescription.SaveToFile(fileName);

		resetFluid = true;

		printf("Fluid Settings saved to %s\n", fileName);
	}
	else
	{
		printf("FluidDescription not valid!\n");
		fluidDescription.LoadFromFluid(Physic::Instance()->GetFluid());
	}
}


// -----------------------------------------------------------------------------
// ----------------------- DemoManager::ReinitializeFluid ----------------------
// -----------------------------------------------------------------------------
void DemoManager::ReinitializeFluid(void)
{
	if (fluidDescription.IsValid())
	{
		resetFluid = true;
	}
	else
	{
		printf("FluidDescription not valid!\n");
		fluidDescription.LoadFromFluid(Physic::Instance()->GetFluid());
	}
}


// -----------------------------------------------------------------------------
// -------------------------- DemoManager::GetViewMatrix -----------------------
// -----------------------------------------------------------------------------
const Matrix4 DemoManager::GetViewMatrix(void) const
{
	if (currentRenderMode == RENDER_SCENE)
	{
		return camera->GetCameraMatrix();
	}
	else
	{
		assert(false);
		return Matrix4::IDENTITY;
	}
}
