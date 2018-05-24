#pragma once
#include <d3d12.h>
#include <d3dcompiler.h>
#include <string>
#include <SimpleMath.h>
#include "Types.h"
#include "Constants.h"
#include "Camera.h"
#include "OBJLoader.h"
#include "Texture.h"
#include "KGraphicsDevice.h"
#include "KShader.h"
#include "GBuffer.h"
#include "Log.h"
#include <algorithm>
#include "LightAssignmentRaster.h"
#include "ClusterRenderer.h"
#include "LightShapeRenderer.h"
#include "LightManager.h"
#include "InputManager.h"
#include "SharedContext.h"

#include <AntTweakBar.h>
using namespace DirectX::SimpleMath;

struct LinkedLightID
{
	uint32 lightID;
	uint32 link;
};

class Application
{
public:
	Application(char* p_args);
	~Application();

	void ShadeClusterSnapShot();
	void ShapeClusterSnapShot();
	void ClusterSnapShotCount();
private:

	void HandleEvents();
	void Init();
	void Run();
	void Update(float dt);
	void Draw();
	void Release();
	void SetupAntTweakBar();

	KBuffer m_cbCamera;
	KBuffer m_cbLightTypePoint;
	KBuffer m_cbLightTypeSpot;

	KSampler m_wrapSampler;

	ID3D12RootSignature* g_RootSignature;
	ID3D12PipelineState* g_PSO;
	ID3D12PipelineState* m_PSOLinear;
	ID3D12PipelineState* m_PSOColor;
	ID3D12PipelineState* m_PSOLinearColor;
	ID3D12GraphicsCommandList* g_CommandList;

	std::unique_ptr<Model> g_SponzaModel;
	std::unique_ptr<LightShape> m_SphereShape;
	std::unique_ptr<LightShape> m_ConeShape;
	std::unique_ptr<LightShape> m_SphereShapeLow;
	std::unique_ptr<LightShape> m_ConeShapeLow;

	std::unique_ptr<Camera> m_camera;
	std::unique_ptr<LightAssignmentRaster> m_laRaster;
	std::unique_ptr<ClusterRenderer> m_clusterRenderer;
	std::unique_ptr<LightShapeRenderer> m_lightShapeRenderer;
	std::unique_ptr<InputManager> g_inputManager;
	std::unique_ptr<LightManager> m_lightManager;
	std::unique_ptr<GBuffer> m_gbuffer;

	bool m_running;
	bool m_moveLights;
	bool m_showShapes;
	bool m_showSpotLights;
	bool m_showPointLights;
	bool m_useLowLODSphere;
	bool m_useLowLODCone;
	bool m_useOldShellPipe;
	bool m_showLightDensity;

	//Debug hack
	bool m_firstFrame;
	bool m_completeClusterSnap;

	uint32 m_numPointLights;
	uint32 m_numSpotLights;

	float time_shell;
	float time_fill;
	float time_geometry;
	float time_shading;
	float time_spot_shell;
	float time_spot_fill;
	float time_tot_LA;
	float time_tot;

	uint32 m_nrclusts;

	enum DEPTH_DIST
	{
		LINEAR,
		EXPONENTIAL,
	} m_depthDist;

	enum LIGHT_TYPE
	{
		POINT,
		SPOT,
	} m_activeSnapType;

	uint32 m_fps;
	uint32 m_numPointShapes;
	uint32 m_numSpotShapes;
	uint32 m_activeLightSnapIndex;

	TwBar* antbar;

	std::vector<SDL_Event> m_atbSDLEvents;
};