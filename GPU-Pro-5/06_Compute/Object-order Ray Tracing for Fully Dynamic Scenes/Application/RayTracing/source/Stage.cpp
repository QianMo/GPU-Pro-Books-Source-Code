/******************************************************/
/* Object-order Ray Tracing Demo (c) Tobias Zirr 2013 */
/******************************************************/

#include "stdafx.h"

#include "Stage.h"
#include <Windows.h>

#include "Scene.h"
#include "RayTracing/Pipeline.h"

#include <beScene/beShaderDrivenPipeline.h>

#include "Viewer.h"
#include "FreeCamera.h"

#include <AntTweakBar.h>

#include <fstream>
#include <algorithm>

#include <sstream>
#include <lean/logging/log.h>

#ifdef LEAN_DEBUG_BUILD
	#define TRACING_DEBUG_POSTFIX(x) x "Debug"
#else
	#define TRACING_DEBUG_POSTFIX(x) x
#endif

// #define AUTO_BENCHMARK
// #define NO_RT

struct SceneDesc
{
	char const* filePath;
	char const* posFilePath;
};
#define SCENE_FILEPATH(name, filename) "Data/Maps/" name "/" filename ".scene.xml"
#define SCENE_POS_FILEPATH(name) name ".benchpos"
#define SCENE_DESC(name, filename) { SCENE_FILEPATH(name, filename), SCENE_POS_FILEPATH(name) }

SceneDesc const scenes[] = {
	  SCENE_DESC("Sponza", "SponzaAnimated")
	, SCENE_DESC("Sibenik", "SibenikT2")
	, SCENE_DESC("Simple", "Simple")
};
TwEnumVal const sceneEnumValues[] = {
	  { 0, "Sponza"}
	, { 1, "Sibenik" }
	, { 2, "Assorted"}
};

namespace app
{

// Constructor.
Stage::Stage(beg::Device *pGraphicsDevice, int sceneIdx)
	: m_welcomeBar(pGraphicsDevice)
	, m_sceneBar(pGraphicsDevice, sceneIdx)
	
	, m_pResourceManager( besc::CreateResourceManager(pGraphicsDevice,
#ifndef NO_RT
		TRACING_DEBUG_POSTFIX("RayTracingEffectCache"), "RayTracingEffects",
#else
		TRACING_DEBUG_POSTFIX("EffectCache"), "Effects",
#endif
		"Textures", "Materials", "Meshes")
	  )
	, m_pRenderer( besc::CreateEffectDrivenRenderer(pGraphicsDevice, m_pResourceManager->Monitor()) )

	, m_timeStep(1.0 / 60.0)
	, m_frame(0)
	, m_bPaused(false)
{
	// Pipeline
#ifndef NO_RT
	m_pPipeline = new tracing::Pipeline(m_pRenderer, m_pResourceManager);
	m_pRenderer = m_pPipeline->GetTracingRenderer();
#else
	besc::LoadRenderingPipeline(*m_pRenderer->Pipeline(),
			*m_pResourceManager->EffectCache()->GetEffect("Pipelines/LPR/Pipeline.fx", nullptr, 0),
			*m_pRenderer->RenderableDrivers()
		);
#endif

	// Scene
	m_pScene = new Scene(scenes[sceneIdx].filePath, m_pRenderer, m_pResourceManager);
#ifndef NO_RT
	m_pPipeline->SetBatchSize(m_pScene->GetMaxTriangleCount());
	m_pPipeline->SetBounds(m_pScene->GetMin(), m_pScene->GetMax());
#endif
	m_sceneBar.triangleCount = m_pScene->GetTriangleCount();
	m_sceneBar.maxBatchTriCount = m_pScene->GetMaxTriangleCount();
	m_sceneBar.batchCount = m_pScene->GetBatchCount();
	m_sceneBar.extents = m_pScene->GetMax() - m_pScene->GetMin();

	// Camera
	m_pCamera = m_pScene->GetWorld()->Entities()->AddEntity("Camera", bees::Entities::AnonymousPersistentID);
	m_pCameraController = new_resource besc::CameraController(m_pScene->GetScene());
	m_pViewer = new Viewer(m_pCameraController, *m_pRenderer, *m_pResourceManager);

	// NOTE: Disable normal camera perspective scheduling!
	SetUpBackBufferRendering(*m_pCameraController, *m_pRenderer);
#ifdef NO_RT
	m_pScene->GetScene()->AddPerspective(m_pCameraController->GetPerspective());
#endif

	m_pCamera->AddControllerKeep(m_pCameraController);
	m_pCamera->Attach();

	m_pFreeCamera = new FreeCamera(m_pCamera);

	{
		std::fstream benchPosFile(scenes[sceneIdx].posFilePath, std::ios::in);

		while (!benchPosFile.fail())
		{
			bem::fvec3 pos;
			bem::fmat3 orient;

			for (int i  = 0; i < 3; ++i)
				benchPosFile >> pos[i];
			for (int i  = 0; i < 9; ++i)
				benchPosFile >> orient.element(i);

			if (!benchPosFile.fail())
				m_benchPositions.push_back( BenchmarkPosition(pos, orient) );
		}
	}

	TwAddVarRW(m_sceneBar.bar, "Pause", TW_TYPE_BOOLCPP, &m_bPaused, NULL);
	TwSetTopBar(m_welcomeBar.bar);
}

// Destructor.
Stage::~Stage()
{
	{
		std::fstream benchPosFile(scenes[m_sceneBar.currentScene].posFilePath, std::ios::out | std::ios::trunc);

		if (!benchPosFile.fail())
		{
			for (bench_pos_vector::const_iterator it = m_benchPositions.begin(); it != m_benchPositions.end(); ++it)
			{
				for (int i  = 0; i < 3; ++i)
					benchPosFile << it->pos[i] << ' ';
				for (int i  = 0; i < 9; ++i)
					benchPosFile << it->orient.element(i) << ' ';

				benchPosFile << std::endl;
			}
		}
	}
}

Stage::SceneBar::SceneBar(beg::Device *device, int currentScene)
	: currentScene(currentScene)
	, nextScene(currentScene)
{
	TwBar* metricsBar = LEAN_ASSERT_NOT_NULL( TwGetBarByName("Metrics") );
	int margins[2], area[2];
	TwGetParam(metricsBar, nullptr, "position", TW_PARAM_INT32, lean::arraylen(margins), margins);
	TwGetParam(metricsBar, nullptr, "size", TW_PARAM_INT32, lean::arraylen(area), area);

	bar = TwNewBar( ("Scene" + identityString(this)).c_str() );
	TwSetParam(bar, nullptr, "label", TW_PARAM_CSTRING, 1, "Scene");
	
	TwType sceneEnumType = TwDefineEnum("scene", sceneEnumValues, lean::arraylen(sceneEnumValues));
	TwAddVarRW(bar, "scene", sceneEnumType, &nextScene, NULL);
	TwAddVarRO(bar, "# triangles", TW_TYPE_UINT32, &triangleCount, NULL);
	TwAddVarRO(bar, "max ^/batch", TW_TYPE_UINT32, &maxBatchTriCount, NULL);
	TwAddVarRO(bar, "# batches", TW_TYPE_UINT32, &batchCount, NULL);
	TwAddVarRO(bar, "X", TW_TYPE_FLOAT, &extents.x, "group=extents");
	TwAddVarRO(bar, "Y", TW_TYPE_FLOAT, &extents.y, "group=extents");
	TwAddVarRO(bar, "Z", TW_TYPE_FLOAT, &extents.z, "group=extents");

	int size[2] = { 190, 190 }; // area[0] / 6, area[1] / 5
	int pos[2] = { area[0] - margins[0] - size[0], margins[1] };
	TwSetParam(bar, nullptr, "position", TW_PARAM_INT32, 2, pos);
	TwSetParam(bar, nullptr, "size", TW_PARAM_INT32, 2, size);

	TwSetParam(bar, nullptr, "color", TW_PARAM_CSTRING, 1, "90 0 0");
}

Stage::SceneBar::~SceneBar()
{
	TwDeleteBar(bar);
}

Stage::WelcomeBar::WelcomeBar(beg::Device *device)
{
	TwBar* metricsBar = TwGetBarByName("Metrics");
	int margins[2], area[2];
	TwGetParam(metricsBar, nullptr, "position", TW_PARAM_INT32, lean::arraylen(margins), margins);
	TwGetParam(metricsBar, nullptr, "size", TW_PARAM_INT32, lean::arraylen(area), area);

	bar = TwNewBar( ("Welcome" + identityString(this)).c_str() );
	TwSetParam(bar, nullptr, "label", TW_PARAM_CSTRING, 1, "Welcome");
	TwAddButton(bar, "warmupText", nullptr, nullptr, " label='Warming up shader optimizer ...'");
	TwAddSeparator(bar, "sep", nullptr);
	TwAddButton(bar, "helpText", nullptr, nullptr, " label='Click '?' at the bottom for help.'");
	
	int size[2] = { 420, 100 }; // [0] / 3, area[1] / 7
	int pos[2] = { (area[0] - size[0]) / 2, (area[1] - size[1]) / 2 };
	TwSetParam(bar, nullptr, "position", TW_PARAM_INT32, 2, pos);
	TwSetParam(bar, nullptr, "size", TW_PARAM_INT32, 2, size);

	TwSetParam(bar, nullptr, "color", TW_PARAM_CSTRING, 1, "0 90 25");
}

Stage::WelcomeBar::~WelcomeBar()
{
	TwDeleteBar(bar);
}

// Steps the application.
void Stage::Step(const beLauncher::InputState &input)
{
	// Reload changed resources
	do
	{
		m_pResourceManager->Commit();
		m_pRenderer->Commit();
		m_pScene->GetWorld()->Commit();
		m_pPipeline->Commit();
	}
	while (m_pResourceManager->Monitor()->ChangesPending());

	// Frame time
	double correctedTimeStep;
	{
		double prevTimeStepError = 0.0;

		if (m_frame > 0)
		{
			double nextTimeStep = max( 0.95 * m_timeStep + 0.05 * min(m_timer.seconds(), 0.1), 0.002 );

			while (m_timer.seconds() < m_timeStep - 0.003)
				::Sleep(1);

			prevTimeStepError = min(m_timer.seconds() - m_timeStep, 0.05);
			m_timeStep = nextTimeStep;
		}

		// DONT correct errors - although correct, catching up with missed time would
		// only results in more stuttering?
		correctedTimeStep = m_timeStep; // + prevTimeStepError;

		m_timer.tick();
	}

	// Input & Animation
	UpdateBench(*input.KeyState);
	m_pFreeCamera->Step((float) correctedTimeStep, input);
	m_pScene->Step( (m_bPaused) ? 0.0f : (float) correctedTimeStep );

	// Benchmarks
	if ( JustPressed(*input.KeyState, 'B')
#ifdef AUTO_BENCHMARK
		|| m_frame == 12 || 
#endif
		) Benchmark();

	// Hide welcome bar after first (real) frame
	if (m_frame == 1)
		TwSetParam(m_welcomeBar.bar, nullptr, "visible", TW_PARAM_CSTRING, 1, "false");

	// Warm up driver shader optimizer on first frame: Render a few images with varying complexity
	for (int i = 0, warmupPassCount = (m_frame == 0) ? m_pPipeline->GetWarmupPassCount() : 1; i < warmupPassCount; ++i)
	{
		// Classic rendering (mostly disabled)
		m_pScene->Render();
#ifndef NO_RT
		// This is where the tracing happens
		m_pPipeline->Render( m_pScene->GetScene(), *m_pCameraController->GetPerspective() );
#endif
		// Present w/ tweak bar UI
		TwDraw();
		m_pRenderer->Device()->Present(false);
	}

	++m_frame;
}

// Do benchmarking.
void Stage::Benchmark()
{
#ifndef NO_RT
	static const size_t BenchmarkIterations = 5;
	std::stringstream benchStream;

	for (bench_pos_vector::const_iterator it = m_benchPositions.begin(); it != m_benchPositions.end(); ++it)
	{
		m_pCamera->SetPosition(it->pos);
		m_pCamera->SetOrientation(it->orient);

		std::vector<tracing::Pipeline::benchmark_vector> results(BenchmarkIterations);

		for (size_t i = 0; i < BenchmarkIterations; ++i)
		{
			m_pCameraController->SetTime(0.0f);
			m_pScene->Step(0.0001f);

			// Classic rendering (mostly disabled)
			m_pScene->Render();

			// Tracing
			m_pPipeline->Render( m_pScene->GetScene(), *m_pCameraController->GetPerspective(), &results[i] );

			m_pRenderer->Device()->Present(false);
		}

		std::vector<float> sortedQuantity(BenchmarkIterations);

		for (size_t j = 0; j < results[0].size(); ++j)
		{
			for (size_t i = 0; i < BenchmarkIterations; ++i)
				sortedQuantity[i] = results[i][j];

			std::sort(sortedQuantity.begin(), sortedQuantity.end());

			// Log median
			if (j) benchStream << '\t';
			benchStream << (sortedQuantity[(BenchmarkIterations - 1) / 2] + sortedQuantity[BenchmarkIterations / 2]) / 2.0f;
		}

		benchStream << std::endl;
	}

	std::string benchText = benchStream.str();
	lean::debug_stream() << benchText;

	// Copy results to clipboard
	{
		HGLOBAL hMem =  GlobalAlloc(GMEM_MOVEABLE, benchText.size() + 1);
		memcpy(GlobalLock(hMem), benchText.c_str(), benchText.size() + 1);
		GlobalUnlock(hMem);

		OpenClipboard(0);
		EmptyClipboard();
		SetClipboardData(CF_TEXT, hMem);
		CloseClipboard();
	}
#endif
}

// Benchmark position management.
void Stage::UpdateBench(const beLauncher::KeyboardState &input)
{
	size_t numPressed = (m_frame == 0) ? 0 : -1;

	for (size_t i = 1; i <= 10; ++i)
		if (JustPressed(input, '0' + i % 10))
			numPressed = i - 1;

	if (Pressed(input, 'R') && numPressed != -1)
	{
		// Record camera position
		if (numPressed < m_benchPositions.size())
			m_benchPositions[numPressed] = BenchmarkPosition(m_pCamera->GetPosition(), m_pCamera->GetOrientation());
		else // if (JustPressed(input, 'R'))
			m_benchPositions.push_back( BenchmarkPosition(m_pCamera->GetPosition(), m_pCamera->GetOrientation()) );
	}
	// Restore camera position
	else if (numPressed < m_benchPositions.size())
	{
		m_pCamera->SetPosition(m_benchPositions[numPressed].pos);
		m_pCamera->SetOrientation(m_benchPositions[numPressed].orient);
	}

	if (JustPressed(input, 'P'))
		m_bPaused = !m_bPaused;
}

// Updates the screen rectangle.
void Stage::UpdateScreen(const bem::ivec2 &pos, const bem::ivec2 &ext)
{
	m_pCameraController->SetAspect( (float) ext[0] / ext[1] );
}

// Gets the next scene.
Stage* Stage::GetNextScene()
{
	if (m_sceneBar.nextScene == m_sceneBar.currentScene)
		return this;

	lean::scoped_ptr<Stage> newStage( new Stage(m_pRenderer->Device(), m_sceneBar.nextScene) );
	newStage->m_pCameraController->SetAspect(m_pCameraController->GetAspect());
	return newStage.detach();
}

} // namespace