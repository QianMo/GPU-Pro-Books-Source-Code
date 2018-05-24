#pragma once
#include "KGraphicsDevice.h"
#include "SimpleMath.h"
#include <random>
#include "Constants.h"

using namespace DirectX::SimpleMath;

class LightManager
{
public:
	LightManager();
	~LightManager();
	void Update(float dt);

	KShaderResourceView GetPointLightSRV()		{ return m_pointLightBuffer.srv; }
	KShaderResourceView GetSpotLightSRV()		{ return m_spotLightBuffer.srv; }

private:

	struct LightTravelData
	{
		Vector3 direction;
		float time;
	};

	KBuffer m_spotLightBuffer;
	KBuffer m_pointLightBuffer;

	void RandomPointLightDistribution();
	void RandomSpotLightDistribution();
	void RandomAMDDistribution();

	void UpdatePointLights(float dt);
	void UpdateSpotLights(float dt);

	PointLight m_pointLights[Constants::NR_POINTLIGHTS];
	Vector3 m_pointLightRandPoints[Constants::NR_POINTLIGHTS][10];
	LightTravelData m_pointLightTravelData[Constants::NR_POINTLIGHTS];

	SpotLight m_spotLights[Constants::NR_SPOTLIGHTS];
	Vector3 m_spotLightRandPoints[Constants::NR_SPOTLIGHTS][10];
	LightTravelData m_spotLightTravelData[Constants::NR_SPOTLIGHTS];

	std::mt19937 mt;
	std::uniform_real_distribution<float> light_time;
	std::uniform_int_distribution<uint32> waypoint;

};