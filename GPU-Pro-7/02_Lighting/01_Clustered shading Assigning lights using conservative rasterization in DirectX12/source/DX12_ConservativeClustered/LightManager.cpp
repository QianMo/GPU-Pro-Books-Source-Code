#include "LightManager.h"
#include "SharedContext.h"

static float GetRandFloat(float fRangeMin, float fRangeMax)
{
	return  (float)rand() / (RAND_MAX + 1) * (fRangeMax - fRangeMin) + fRangeMin;
}

static Vector4 GetRandColor()
{
	static unsigned uCounter = 0;
	uCounter++;

	Vector4 Color;
	if (uCounter % 2 == 0)
	{
		// since green contributes the most to perceived brightness, 
		// cap it's min value to avoid overly dim lights
		Color = Vector4(GetRandFloat(0.0f, 1.0f), GetRandFloat(0.27f, 1.0f), GetRandFloat(0.0f, 1.0f), 1.0f);
	}
	else
	{
		// else ensure the red component has a large value, again 
		// to avoid overly dim lights
		Color = Vector4(GetRandFloat(0.9f, 1.0f), GetRandFloat(0.0f, 1.0f), GetRandFloat(0.0f, 1.0f), 1.0f);
	}

	return Color;
}

static Vector3 GetRandLightDirection()
{
	static unsigned uCounter = 0;
	uCounter++;

	Vector3 vLightDir;
	vLightDir.x = GetRandFloat(-1.0f, 1.0f);
	vLightDir.y = GetRandFloat(0.1f, 1.0f);
	vLightDir.z = 1.0f - GetRandFloat(-1.0f, 1.0f);

	if (uCounter % 2 == 0)
	{
		vLightDir.y = -vLightDir.y;
	}

	vLightDir.Normalize();

	return vLightDir;
}

LightManager::LightManager()
{
	m_pointLightBuffer = shared_context.gfx_device->CreateBuffer(Constants::NR_POINTLIGHTS, sizeof(PointLight), KBufferType::STRUCTURED, D3D12_HEAP_TYPE_UPLOAD);
	m_spotLightBuffer = shared_context.gfx_device->CreateBuffer(Constants::NR_SPOTLIGHTS, sizeof(SpotLight), KBufferType::STRUCTURED, D3D12_HEAP_TYPE_UPLOAD);
	
	mt.seed(1729);
	light_time = std::uniform_real_distribution<float>(5.0f, 10.0f);
	waypoint = std::uniform_int_distribution<uint32>(0, 9);

	RandomAMDDistribution();
	//RandomPointLightDistribution();
	//RandomSpotLightDistribution();
}

LightManager::~LightManager()
{
	m_pointLightBuffer.resource->Release();
	m_spotLightBuffer.resource->Release();
}

void LightManager::RandomAMDDistribution()
{
	srand(1);
	Vector3 vBBoxMax = Vector3(1799.90796f, 1429.43311f, 1182.80701f);
	Vector3 vBBoxMin = Vector3(-1920.94580f, -126.442505f, -1105.42590f);
	float fRadius = 173.886459f;
	float fSpotRadius = 231.848618f;

	for (int i = 0; i < Constants::NR_POINTLIGHTS; ++i)
	{
		m_pointLights[i].posRange = Vector4(GetRandFloat(vBBoxMin.x, vBBoxMax.x), GetRandFloat(vBBoxMin.y, vBBoxMax.y), 1.0f - GetRandFloat(vBBoxMin.z, vBBoxMax.z), fRadius);
		m_pointLights[i].color = GetRandColor();
	}
	memcpy(m_pointLightBuffer.mem, &m_pointLights[0], Constants::NR_POINTLIGHTS * sizeof(PointLight));

	for (int i = 0; i < Constants::NR_SPOTLIGHTS; ++i)
	{
		m_spotLights[i].posRange = Vector4(GetRandFloat(vBBoxMin.x, vBBoxMax.x), GetRandFloat(vBBoxMin.y, vBBoxMax.y), 1.0f - GetRandFloat(vBBoxMin.z, vBBoxMax.z), fSpotRadius);
		m_spotLights[i].color = GetRandColor();

		Vector3 dir = GetRandLightDirection();

		//35.26438968 degrees
		m_spotLights[i].dirAngle = Vector4(dir.x, dir.y, dir.z, 0.615479709f);
	}
	memcpy(m_spotLightBuffer.mem, &m_spotLights[0], Constants::NR_SPOTLIGHTS * sizeof(SpotLight));

	ZeroMemory(&m_pointLightTravelData[0], Constants::NR_POINTLIGHTS * sizeof(LightTravelData));

	//Create random waypoints
	for (int i = 0; i < Constants::NR_POINTLIGHTS; ++i)
	{
		for (int j = 0; j < 10; ++j)
		{
			float x = GetRandFloat(-100.0f, 100.0f);
			float y = GetRandFloat(-100.0f, 100.0f);
			float z = GetRandFloat(-100.0f, 100.0f);

			Vector3 lightPos = Vector3(m_pointLights[i].posRange);
			m_pointLightRandPoints[i][j] = Vector3(x, y, z) + lightPos;
		}
	}

	ZeroMemory(&m_spotLightTravelData[0], Constants::NR_SPOTLIGHTS * sizeof(LightTravelData));

	//Create random waypoints
	for (int i = 0; i < Constants::NR_SPOTLIGHTS; ++i)
	{
		for (int j = 0; j < 10; ++j)
		{
			float x = GetRandFloat(-100.0f, 100.0f);
			float y = GetRandFloat(-100.0f, 100.0f);
			float z = GetRandFloat(-100.0f, 100.0f);

			Vector3 lightPos = Vector3(m_spotLights[i].posRange);
			m_spotLightRandPoints[i][j] = Vector3(x, y, z) + lightPos;
		}
	}
}

void LightManager::RandomPointLightDistribution()
{
	//Generate light data
	//Some "random" light placement
	std::uniform_real_distribution<float> range_dist(75.0f, 175.0f);
	std::uniform_real_distribution<float> x_dist(-1800.0f, 1800.0f);
	std::uniform_real_distribution<float> y_dist(-100.0f, 1400.0f);
	std::uniform_real_distribution<float> z_dist(-1100.0f, 1050.0f);
	std::uniform_real_distribution<float> move_dist(-100.0f, 100.0f);
	std::uniform_real_distribution<float> color_dist;

	for (int i = 0; i < Constants::NR_POINTLIGHTS; ++i)
	{
		float rangeRand = 170.0f;//range_dist(mt);
		float x = x_dist(mt);
		float y = y_dist(mt);
		float z = z_dist(mt);

		float r = color_dist(mt);
		float g = color_dist(mt);
		float b = color_dist(mt);

		m_pointLights[i].posRange = Vector4(x, y, z, rangeRand);
		m_pointLights[i].color = Vector4(r, g, b, 1.0f);
	}

	//For debug purposes
	if (Constants::NR_POINTLIGHTS == 1)
	{
		m_pointLights[0].posRange = Vector4(0, 75, 0, 173.886459f);
		m_pointLights[0].color = Vector4(0.96f, 0.65f, 0.000f, 1.0f);
	}

	ZeroMemory(&m_pointLightTravelData[0], Constants::NR_POINTLIGHTS * sizeof(LightTravelData));

	//Create random waypoints
	for (int i = 0; i < Constants::NR_POINTLIGHTS; ++i)
	{
		for (int j = 0; j < 10; ++j)
		{
			float x = move_dist(mt);
			float y = move_dist(mt);
			float z = move_dist(mt);

			Vector3 lightPos = Vector3(m_pointLights[i].posRange);
			m_pointLightRandPoints[i][j] = Vector3(x, y, z) + lightPos;
		}
	}

	memcpy(m_pointLightBuffer.mem, &m_pointLights[0], Constants::NR_POINTLIGHTS * sizeof(PointLight));
}
void LightManager::RandomSpotLightDistribution()
{
	//Generate light data
	//Some "random" light placement
	std::uniform_real_distribution<float> range_dist(400.0f, 800.0f);
	std::uniform_real_distribution<float> x_dist(-1800.0f, 1800.0f);
	std::uniform_real_distribution<float> y_dist(-100.0f, 1400.0f);
	std::uniform_real_distribution<float> z_dist(-1100.0f, 1050.0f);
	std::uniform_real_distribution<float> move_dist(-100.0f, 100.0f);
	std::uniform_real_distribution<float> dir_dist(-1.0f, 1.0f);
	std::uniform_real_distribution<float> color_dist;

	for (int i = 0; i < Constants::NR_SPOTLIGHTS; ++i)
	{
		float rangeRand = 231.848618f;//range_dist(mt);
		float x = x_dist(mt);
		float y = y_dist(mt);
		float z = z_dist(mt);

		float r = color_dist(mt);
		float g = color_dist(mt);
		float b = color_dist(mt);
		
		m_spotLights[i].posRange = Vector4(x, y, z, rangeRand);
		m_spotLights[i].color = Vector4(r, g, b, 1.0f);

		Vector3 dir = Vector3(dir_dist(mt), 0, dir_dist(mt));
		dir.Normalize();

		//35.26438968 degrees
		m_spotLights[i].dirAngle = Vector4(dir.x, 0, dir.z, 0.615479709f);
	}

	//For debug purposes
	if (Constants::NR_SPOTLIGHTS == 1)
	{
		m_spotLights[0].posRange = Vector4(0, 0, 0, 225);
		m_spotLights[0].color = Vector4(1, 0, 0, 1.0f);
	}

	ZeroMemory(&m_spotLightTravelData[0], Constants::NR_SPOTLIGHTS * sizeof(LightTravelData));

	//Create random waypoints
	for (int i = 0; i < Constants::NR_SPOTLIGHTS; ++i)
	{
		for (int j = 0; j < 10; ++j)
		{
			float x = move_dist(mt);
			float y = move_dist(mt);
			float z = move_dist(mt);

			Vector3 lightPos = Vector3(m_spotLights[i].posRange);
			m_spotLightRandPoints[i][j] = Vector3(x, y, z) + lightPos;
		}
	}

	memcpy(m_spotLightBuffer.mem, &m_spotLights[0], Constants::NR_SPOTLIGHTS * sizeof(SpotLight));
}

void LightManager::UpdatePointLights(float dt)
{
	//Update light positions
	for (int lig = 0; lig < Constants::NR_POINTLIGHTS; ++lig)
	{
		Vector3 lightPos = Vector3(m_pointLights[lig].posRange);

		if (m_pointLightTravelData[lig].time <= 0.0f)
		{
			m_pointLightTravelData[lig].time = light_time(mt);

			Vector3 direction = m_pointLightRandPoints[lig][waypoint(mt)] - lightPos;
			direction.Normalize();

			m_pointLightTravelData[lig].direction = direction;
		}

		m_pointLightTravelData[lig].time -= dt;

		//New light pos
		lightPos += m_pointLightTravelData[lig].direction * 50.0f * dt;

		memcpy(&m_pointLights[lig].posRange, &lightPos, 3 * sizeof(float));
	}

	memcpy(m_pointLightBuffer.mem, &m_pointLights[0], Constants::NR_POINTLIGHTS * sizeof(PointLight));
}

void LightManager::UpdateSpotLights(float dt)
{
	//Update light positions

	for (int lig = 0; lig < Constants::NR_SPOTLIGHTS; ++lig)
	{

		Vector3 lightPos = Vector3(m_spotLights[lig].posRange);

		if (m_spotLightTravelData[lig].time <= 0.0f)
		{
			m_spotLightTravelData[lig].time = light_time(mt);

			Vector3 direction = m_spotLightRandPoints[lig][waypoint(mt)] - lightPos;
			direction.Normalize();

			m_spotLightTravelData[lig].direction = direction;
		}

		Vector3 dir = Vector3(m_spotLights[lig].dirAngle);

		float angle = DirectX::XM_PIDIV4*dt*0.5f;

		dir.x = dir.x * cos(angle) - dir.z * sin(angle);
		dir.z = dir.x * sin(angle) + dir.z * cos(angle);
		dir.Normalize();

		m_spotLightTravelData[lig].time -= dt;

		//New light pos
		lightPos += m_spotLightTravelData[lig].direction * 50.0f * dt;

		memcpy(&m_spotLights[lig].dirAngle, &dir, 3 * sizeof(float));
		memcpy(&m_spotLights[lig].posRange, &lightPos, 3 * sizeof(float));
	}

	memcpy(m_spotLightBuffer.mem, &m_spotLights[0], Constants::NR_SPOTLIGHTS * sizeof(SpotLight));
}

void LightManager::Update(float dt)
{
	UpdatePointLights(dt);
	UpdateSpotLights(dt);
}

