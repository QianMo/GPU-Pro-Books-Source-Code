#pragma once

#include "Tracing.h"

#include <beLauncher/beInput.h>

#include <beScene/beResourceManager.h>
#include <beScene/beEffectDrivenRenderer.h>

#include <beEntitySystem/beEntities.h>
#include <beScene/beCameraController.h>

#include <lean/smart/resource_ptr.h>
#include <lean/smart/scoped_ptr.h>
#include <lean/time/highres_timer.h>

#include <beMath/beVector.h>

struct CTwBar;

namespace app
{

class Scene;
class FreeCamera;
class Viewer;

namespace tracing { class Pipeline; }

/// Stage.
class Stage
{
private:
	struct WelcomeBar
	{
		CTwBar *bar;
		WelcomeBar(beg::Device *device);
		~WelcomeBar();
	};
	WelcomeBar m_welcomeBar;
	struct SceneBar
	{
		CTwBar *bar;
		int const currentScene;
		int nextScene;
		unsigned triangleCount;
		unsigned maxBatchTriCount;
		unsigned batchCount;
		bem::fvec3 extents;
		SceneBar(beg::Device *device, int currentScene);
		~SceneBar();
	};
	SceneBar m_sceneBar;

	lean::resource_ptr<besc::ResourceManager> m_pResourceManager;
	lean::resource_ptr<besc::EffectDrivenRenderer> m_pRenderer;
	
	lean::scoped_ptr<tracing::Pipeline> m_pPipeline;

	lean::scoped_ptr<Scene> m_pScene;

	lean::scoped_ptr<beEntitySystem::Entity> m_pCamera;
	lean::resource_ptr<besc::CameraController> m_pCameraController;
	lean::scoped_ptr<Viewer> m_pViewer;
	lean::scoped_ptr<FreeCamera> m_pFreeCamera;

	lean::highres_timer m_timer;
	double m_timeStep;
	int m_frame;
	bool m_bPaused;

	struct BenchmarkPosition
	{
		bem::fvec3 pos;
		bem::fmat3 orient;

		BenchmarkPosition(const bem::fvec3 &pos, const bem::fmat3 &orient)
			: pos(pos), orient(orient) { }
	};
	typedef std::vector<BenchmarkPosition> bench_pos_vector;
	bench_pos_vector m_benchPositions;

	/// Benchmark position management.
	void UpdateBench(const beLauncher::KeyboardState &input);
	/// Do benchmarking.
	void Benchmark();

public:
	/// Constructor.
	Stage(beg::Device *pGraphicsDevice, int sceneIdx = 0);
	/// Destructor.
	~Stage();

	/// Steps the application.
	void Step(const beLauncher::InputState &input);

	/// Updates the screen rectangle.
	void UpdateScreen(const bem::ivec2 &pos, const bem::ivec2 &ext);

	/// Gets the renderer.
	besc::EffectDrivenRenderer* GetRenderer() const { return m_pRenderer; }
	/// Gets the resource manager.
	besc::ResourceManager* GetResourceManager() const { return m_pResourceManager; }

	/// Gets the next scene.
	Stage* GetNextScene();
};

} // namespace