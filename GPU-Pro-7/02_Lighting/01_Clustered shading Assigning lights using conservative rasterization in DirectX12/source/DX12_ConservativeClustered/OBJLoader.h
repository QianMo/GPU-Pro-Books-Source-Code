#pragma once
#include <vector>
#include <map>
#include <SimpleMath.h>
#include "Vert.h"
#include "Model.h"
#include <memory>
#include "LightShape.h"

using namespace DirectX::SimpleMath;

class OBJLoader
{
public:
	OBJLoader();
	~OBJLoader();
	
	std::unique_ptr<LightShape> LoadLightShape(const char* file, ID3D12GraphicsCommandList* gfx_command_list);
	std::unique_ptr<Model> LoadBIN(const char* file, ID3D12GraphicsCommandList* gfx_command_list);
};