#ifndef MATERIAL_H
#define MATERIAL_H

#include "DX11_RASTERIZER_STATE.h"
#include "DX11_DEPTH_STENCIL_STATE.h"
#include "DX11_BLEND_STATE.h"

#define NUM_RENDER_STATES 27 // number of render-states that can be specified in material file

class DX11_SHADER;
class DX11_TEXTURE;

struct MATERIAL_INFO
{
	char name[32]; // info as string
	int mode; // info as int
};

// MATERIAL
//   Loaded from a simple text-file (".mtl") with 3 blocks:
//   1."Textures"
//     - "ColorTexture"
//     - "NormalTexture" 
//		 - "SpecularTexture"
//   2."RenderStates"
//     - "cull" -> culling -> requires 1 additional parameter: cull mode    
//     - "noDepthTest" -> disable depth-testing
//     - "noDepthMask" -> disable depth-mask
//     - "colorBlend" -> color blending -> requires 3 additional parameters: srcColorBlend/ dstColorBlend/ blendColorOp
//     - "alphaBlend" -> alpha blending -> requires 3 additional parameters: srcAlphaBlend/ dstAlphaBlend/ blendAlphaOp
//   3."Shader"
//     - "permutation" -> requires 1 additional parameter: permutation mask of shader
//     - "file" -> requires 1 additional parameter: filename of shader
// - all parameters are optional 
// - order of parameters is indifferent (except in "Shader": permutation must be specified before file)   
class MATERIAL
{
public:
	MATERIAL()
	{
		colorTexture = NULL;
		normalTexture = NULL;
		specularTexture = NULL;
		rasterizerState = NULL;
		depthStencilState = NULL;
		blendState = NULL;
		shader = NULL;
	}

	bool Load(const char *fileName);

	const char* GetName() const
	{
		return name;
	}

	DX11_TEXTURE *colorTexture;
	DX11_TEXTURE *normalTexture;
	DX11_TEXTURE *specularTexture;
	DX11_RASTERIZER_STATE *rasterizerState;
	DX11_DEPTH_STENCIL_STATE *depthStencilState;
	DX11_BLEND_STATE *blendState;
	DX11_SHADER *shader;

private:
	// load "Textures"-block
	void LoadTextures(std::ifstream &file);
	
	// load "RenderStates"-block
	void LoadRenderStates(std::ifstream &file);
	
	// load "Shader"-block
	bool LoadShader(std::ifstream &file);
	
	char name[DEMO_MAX_FILENAME];
	RASTERIZER_DESC rasterDesc;	
	DEPTH_STENCIL_DESC depthStencilDesc;
	BLEND_DESC blendDesc;
	
};

#endif