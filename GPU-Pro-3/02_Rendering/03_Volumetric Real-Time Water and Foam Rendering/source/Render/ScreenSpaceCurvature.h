#ifndef __SCREENSPACECURVATURE__H__
#define __SCREENSPACECURVATURE__H__

#include <string>
#include <list>
#include <vector>

#include "NxPhysics.h"

#include "../Util/Singleton.h"
#include "../Util/Vector2.h"
#include "../Util/Vector3.h"
#include "../Util/Vector4.h"
#include "../Util/Matrix4.h"
#include "../Util/Color.h"
#include "../Util/Timer.h"
#include "../Util/Math.h"

#include "../Render/FluidRenderDescription.h"

#include <IL/il.h>
#include <IL/ilu.h>
#include <IL/ilut.h>

#define USE_DOWNSAMPLE_SHIFT 1

#define DOWNSAMPLE_SHIFT (1)

using namespace std;

class Matrix4;
class Fluid;

class ScreenSpaceCurvature
{
public:

	enum RenderMode {
		RENDER_MODE_BACKGROUND_LAYER = 0,
		RENDER_MODE_BACK_WATER_LAYER,
		RENDER_MODE_FOAM_LAYER,
		RENDER_MODE_FRONT_WATER_LAYER,
		RENDER_MODE_ALL,

		RENDER_MODE_COUNT
	};

	ScreenSpaceCurvature(void);
	~ScreenSpaceCurvature(void);

	/// Init the screen space curvature
	void Init(void);

	/// Update the screen space curvature
	void Update(float deltaTime);
	void UpdateMetaData(float deltaTime);

	/// Render scene
	void BeginRenderScene(void);
	void EndRenderScene(void);

	/// Render the screen space curvature
	void Render(void);

	/// Exit the screen space curvature
	void Exit(void);

	/// Set the fluid
	void SetFluid(Fluid* _fluid) { fluid = _fluid; }

	/// Set view frustum planes
	void SetViewFrustum(const Math::FrustumPlane* _frustumPlanes) { frustumPlanes = _frustumPlanes; }

	/// Set screen parameters
	void SetFieldOfView(float fov) { fieldOfView = fov; }
	void SetWindowSize(int width, int height);

	/// Set a new render description
	void SetRenderDescription(FluidRenderDescription _renderDescription) { renderDescription = _renderDescription; }

	/// Set light position (eye space
	void SetLightPosition(Vector3 _value) { fluidLightPosition = _value; }

	/// Return the 3D noise texture
	unsigned int GetNoiseTexture(void) const { return perlinNoiseTexture; }

	/// Return the final result
	unsigned int GetResultTexture(void) const { return resultTexture; }

	/// Set active render mode
	void SetRenderMode(RenderMode mode) { renderMode = mode; }

private:

	/// Render target types
	enum RenderTarget {
		RENDER_TARGET_DISABLED = 0,

		RENDER_TARGET_SCENE,
		RENDER_TARGET_FOAM_DEPTH,
		RENDER_TARGET_DEPTH,

		RENDER_TARGET_THICKNESS,
		RENDER_TARGET_FOAM_THICKNESS,
		
		RENDER_TARGET_NOISE,
		
		RENDER_TARGET_RESULT,
		RENDER_TARGET_COUNT
	};

	enum FoamPhase
	{
		FP_NONE = 0,
		FP_WATER_TO_FOAM,
		FP_FOAM,
		FP_FOAM_TO_WATER,

		FP_COUNT
	};

	struct MetaData 
	{
		float foam;
		float lifetime;
		float timer;

		FoamPhase phase;
	};

	static float EPSILON;
	static float THRESHOLD_MIN;
	static float BLUR_DEPTH_FALLOFF;
	static float DEPTH_THRESHOLD;

	void UpdateVBO(void);

	/// Render the screen space curvature
	void RenderFoamDepth(void);
	void RenderDepth(void);
	void RenderSmooth(void);

	void RenderNoise(void);

	void RenderThickness(void);
	void RenderFoamThickness(void);

	void RenderComposition(void);
	void RenderSpray(void);

	void DisableVBO(void) const;

	/// Render point sprites
	void RenderParticles(void);

	/// Helper stuff
	void SetRenderTarget(RenderTarget target);
	void RenderQuad(float u, float v) const;

	/// Debug rendering
	void RenderBoundingBoxes(void);

	/// Filter operations
	void ScaleDownTexture(unsigned int dest, unsigned int src);
	unsigned int SmoothTexture(unsigned int tex[2]);

	/// View frustum culling test
	bool FrustumAABBIntersect(const Math::FrustumPlane* planes, const Vector3& mins, const Vector3& maxs) const;

	void CheckFrameBufferState(void) const;

	/// Create and Load data
	unsigned int CreateTexture(GLenum target, int w, int h, unsigned int internalformat, GLenum format);
	void BuildNoiseTexture(void);
	void BuildFoamNoiseTexture(void);

	/// Current render mode
	RenderMode renderMode;

	/// Physics fluid
	Fluid* fluid;
	MetaData* fluidMetaData;

	/// Render Parameters
	FluidRenderDescription renderDescription;

	/// View frustum planes
	const Math::FrustumPlane* frustumPlanes;

	Vector2 bufferSize;

	/// FBO stuff
	unsigned int frameBuffer;
	unsigned int renderBuffer;
	unsigned int depthRenderBuffer;

	/// Screen information
	int windowWidth;
	int windowHeight;

	int scaleDownWidth;
	int scaleDownHeight;

	float lowResFactor;

	/// Camera parameters
	float fieldOfView;
	float invFocalLength;
	float aspectRatio;

	/// Textures
	unsigned int sceneTexture;
	unsigned int foamDepthTexture;
	unsigned int depthTexture;

	unsigned int thicknessTexture;
	unsigned int foamThicknessTexture;

#if USE_DOWNSAMPLE_SHIFT
	unsigned int downsampleDepthTexture[DOWNSAMPLE_SHIFT];
	unsigned int downsampleFoamDepthTexture[DOWNSAMPLE_SHIFT];
#endif
	unsigned int smoothedDepthTexture;

	unsigned int noiseTexture;
	unsigned int resultTexture;

	unsigned int perlinNoiseTexture;
	unsigned int foamPerlinNoiseTexture;

	unsigned int currentDepthSource;
	unsigned int currentFoamDepthSource;

	/// Light position in eye space
	Vector3 fluidLightPosition;

	unsigned int vbo;
	unsigned int visiblePacketCount;
	int* vboStartIndices;
	int* vboIndexCount;

	Timer timer;
	Timer specialTimer;

	struct StatsData 
	{
		float overall;
		float depth;
		float smooth;
		float thickness;
	};
	
	//////////////////////////////////////////////////////////////////////////

	unsigned int occQuery;
	unsigned int currentIterationCount;
};

#endif