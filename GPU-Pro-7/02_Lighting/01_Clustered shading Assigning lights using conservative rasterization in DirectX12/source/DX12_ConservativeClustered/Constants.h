#ifndef GLOBAL_CONSTANTS
#define GLOBAL_CONSTANTS

#include <string>
#include <d3dcommon.h>
#include <SimpleMath.h>

struct PointLight
{
	DirectX::SimpleMath::Vector4 posRange;
	DirectX::SimpleMath::Vector4 color;
};

struct SpotLight
{
	DirectX::SimpleMath::Vector4 posRange;
	DirectX::SimpleMath::Vector4 color;
	DirectX::SimpleMath::Vector4 dirAngle;
};

struct KPlane
{
	KPlane(){};
	KPlane(DirectX::SimpleMath::Vector3 ppos, DirectX::SimpleMath::Vector3 pnormal)
	{
		pos = ppos;
		normal = pnormal;
	}

	DirectX::SimpleMath::Vector3 pos;
	DirectX::SimpleMath::Vector3 normal;
};

namespace Constants
{
	const int32 WINDOW_WIDTH = 1536;
	const int32 WINDOW_HEIGHT = 768;

	const float FARZ = 3000.0f;
	const float NEARZ = 0.5f;
	const float NEAR_CLUST = 50.0f;

	//Tweak these to change cluster grid size
	const uint32 NR_X_CLUSTS = 24; //Muliplier of 12 gives 1:1 with resolution
	const uint32 NR_Y_CLUSTS = 12; //Half of NR_X_CLUSTS gives 1:1 with resolution
	const uint32 NR_Z_CLUSTS = 128;

	const uint32 LOG2_TILE = (uint32)std::log2<uint32>(WINDOW_WIDTH / NR_X_CLUSTS);

	const uint32 NR_X_PLANES = NR_X_CLUSTS + 1;
	const uint32 NR_Y_PLANES = NR_Y_CLUSTS + 1;
	const uint32 NR_Z_PLANES = NR_Z_CLUSTS + 1;

	const uint32 NR_OF_CLUSTS = NR_X_CLUSTS * NR_Y_CLUSTS * NR_Z_CLUSTS;
					 
	//Max supported lights per type is 2048 (Render target restriction, easily increased by tiling multiple lights per render target)
	const uint32 NR_SPOTLIGHTS = 2048;
	const uint32 NR_POINTLIGHTS = 2048;

	const uint32 NR_OF_LIGHT = NR_SPOTLIGHTS + NR_POINTLIGHTS;

	const uint32 MAX_LIGHTS_PER_CLUSTER = 30;

	const uint32 LIGHT_INDEX_LIST_COUNT = (NR_OF_CLUSTS * MAX_LIGHTS_PER_CLUSTER);

	const std::string definitions[] =
	{
		std::to_string(WINDOW_WIDTH),
		std::to_string(WINDOW_HEIGHT),
		std::to_string(NR_X_CLUSTS),
		std::to_string(NR_Y_CLUSTS),
		std::to_string(NR_Z_CLUSTS),
		std::to_string(NEARZ),
		std::to_string(FARZ),
		std::to_string(NEAR_CLUST),
		std::to_string(LOG2_TILE)
	};

	const D3D_SHADER_MACRO SHADER_DEFINES[] =
	{
		"WINDOW_WIDTH", definitions[0].c_str(),
		"WINDOW_HEIGHT", definitions[1].c_str(),
		"CLUSTERSX", definitions[2].c_str(),
		"CLUSTERSY", definitions[3].c_str(),
		"CLUSTERSZ", definitions[4].c_str(),
		"NEARZ", definitions[5].c_str(),
		"FARZ", definitions[6].c_str(),
		"NEAR_CLUST", definitions[7].c_str(),
		"LOG2_TILE", definitions[8].c_str(),
		0, 0
	};

	const D3D_SHADER_MACRO SHADER_DEFINES_COLOR[] =
	{
		"WINDOW_WIDTH", definitions[0].c_str(),
		"WINDOW_HEIGHT", definitions[1].c_str(),
		"CLUSTERSX", definitions[2].c_str(),
		"CLUSTERSY", definitions[3].c_str(),
		"CLUSTERSZ", definitions[4].c_str(),
		"NEARZ", definitions[5].c_str(),
		"FARZ", definitions[6].c_str(),
		"NEAR_CLUST", definitions[7].c_str(),
		"LOG2_TILE", definitions[8].c_str(),
		"SHOW_LIGHTS_PER_PIXEL", 0,
		0, 0
	};

	const D3D_SHADER_MACRO SHADER_DEFINES_LINEAR[] =
	{
		"WINDOW_WIDTH", definitions[0].c_str(),
		"WINDOW_HEIGHT", definitions[1].c_str(),
		"CLUSTERSX", definitions[2].c_str(),
		"CLUSTERSY", definitions[3].c_str(),
		"CLUSTERSZ", definitions[4].c_str(),
		"NEARZ", definitions[5].c_str(),
		"FARZ", definitions[6].c_str(),
		"NEAR_CLUST", definitions[7].c_str(),
		"LOG2_TILE", definitions[8].c_str(),
		"LINEAR_DEPTH_DIST", 0,
		0, 0
	};

	const D3D_SHADER_MACRO SHADER_DEFINES_LINEAR_COLOR[] =
	{
		"WINDOW_WIDTH", definitions[0].c_str(),
		"WINDOW_HEIGHT", definitions[1].c_str(),
		"CLUSTERSX", definitions[2].c_str(),
		"CLUSTERSY", definitions[3].c_str(),
		"CLUSTERSZ", definitions[4].c_str(),
		"NEARZ", definitions[5].c_str(),
		"FARZ", definitions[6].c_str(),
		"NEAR_CLUST", definitions[7].c_str(),
		"LOG2_TILE", definitions[8].c_str(),
		"LINEAR_DEPTH_DIST", 0,
		"SHOW_LIGHTS_PER_PIXEL", 0,
		0, 0
	};
}

namespace COLOR
{
	const float WHITE[] = { 1.0f, 1.0f, 1.0f, 1.0f };
	const float BLACK[] = { 0.0f, 0.0f, 0.0f, 1.0f };
	const float RED[] = { 1.0f, 0.0f, 0.0f, 1.0f };
	const float GREEN[] = { 0.0f, 1.0f, 0.0f, 1.0f };
	const float BLUE[] = { 0.0f, 0.0f, 1.0f, 1.0f };
}

#endif