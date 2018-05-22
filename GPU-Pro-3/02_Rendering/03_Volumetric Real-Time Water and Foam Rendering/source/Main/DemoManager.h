#ifndef __DEMOMANAGER__H__
#define __DEMOMANAGER__H__

#include <Windows.h>
#include <vector>

#include "../Util/Ray.h"
#include "../Util/Math.h"
#include "../Util/Timer.h"
#include "../Util/Matrix4.h"
#include "../Util/Singleton.h"
#include "../Util/AntTweakBar.h"

#include "../Level/Camera.h"

#include "../Physic/FluidDescription.h"

#include "../Render/FluidRenderDescription.h"

class Light;
class RenderObject;
class Sphere;
class SpriteEngine;
class ParticleSystem;

class ScreenSpaceCurvature;


// -----------------------------------------------------------------------------
/// 
// -----------------------------------------------------------------------------
/// \ingroup 
/// 
/// 
// -----------------------------------------------------------------------------
class DemoManager : public Singleton<DemoManager>
{
	friend class Singleton<DemoManager>;

public:
	enum RenderMode 
	{
		RENDER_NONE = 0,
		RENDER_SCENE
	};

	enum { NUM_SCENES = 3 };

	DemoManager(void);

	/// Inits the demo manager
	void Init(void);

	/// Exits the demo manager
	void Exit(void);

	/// Called on reshape of the window
	void Reshape(int& w, int& h);

	/// Update the entire scene
	void Update(void);

	/// Performs calculations on the cpu while the physics simulation is running on the gpu (called by Physic::Simulate)
	void PerformCpuCaluations(float deltaTime);

	/// Render the scene
	void Render(void);

	/// Update which is called after the rendering has been done
	void PostRenderUpdate(void);

	/// Perspective rendering helper function
	void SetupPerspectiveRendering(void);

	/// Get members...
	RenderMode GetCurrentRenderMode(void) const { return currentRenderMode; }
	const Camera* GetCamera(void) const { return camera; }
	Light* GetLight(void) const { return light; }

	const Matrix4 GetViewMatrix(void) const;

	/// Gui callback functions
	void SaveRenderSettings(void) const;
	void SaveFluidSettings(void);
	void ReinitializeFluid(void);

	/// Reset msg timer and msg itself
	void SetDebugMsg(const char* str);
	
private:
	/// Loads the current scene stuff
	void LoadScene(bool initFluidMetaData=false);
	void SetupDynamicActors(void);

	/// Render scene into texture
	void RenderPreProcessing(void);

	/// Render fluid into texture
	void RenderFluid(void);

	/// Render things that can cast a shadow
	void RenderShadowCasters(void);
	
	/// Render scene
	void RenderScene(void);

	/// Render final image
	void RenderToScreen(void);

	/// Render debug text, wireframe,...
	void RenderDebugStuff(void);

	/// Input stuff
	void UpdateInput(float deltaTime);

	/// Sets the projection matrix
	void SetProjectionMatrix(void) const;

	/// Calculate view frustum
	void CalculateViewFrustum(void);

	static float FOV;

	bool isInitialized;

	/// Timer stuff
	double deltaTime;
	Timer timer;

	// elapsed time since start of the current level (not the running time of the program)
	double time;

	// planes of the view frustum
	Math::FrustumPlane viewFrustumPlanes[6];

	// stuff for showing framerate on display
	float timeSum;
	LONG frameCount;
	float frameRate;
	char fpsString[32];
	char particleCountString[32];

	// render properties
	bool renderStatistics;
	bool useViewFrustumCulling;
	bool renderUI;

	bool loadScene;
	unsigned int currentScene;

	RenderMode currentRenderMode;

	// the light
	Light* light;

	// the camera
	Camera* camera;

	// fluid renderer
	ScreenSpaceCurvature* screenSpaceCurvature;

	float msgTimer;
	char msgString[128];

	int currentWidth;
	int currentHeight;

	unsigned int cubeMap;

	Matrix4 projectionMatrix;

	bool resetFluid;

	/// Gui stuff
	TwBar* bar;
	TwBar* fluidBar;

	FluidDescription fluidDescription;
	FluidRenderDescription renderDescription;

	unsigned int renderMode;

	std::vector<unsigned int> dynamicActorIDs;
};

#endif