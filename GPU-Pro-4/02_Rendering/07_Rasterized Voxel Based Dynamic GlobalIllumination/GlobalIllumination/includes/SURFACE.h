#ifndef SURFACE_H
#define SURFACE_H

#include <render_states.h>

#define NUM_CUSTOM_TEXURES              6 // number of custom textures that can be set in SURFACE
#define NUM_CUSTOM_STRUCTURED_BUFFERS   2 // number of custom structured-buffers that can be set in SURFACE

class DX11_VERTEX_BUFFER;
class DX11_INDEX_BUFFER;
class DX11_UNIFORM_BUFFER;
class DX11_STRUCTURED_BUFFER;
class RENDER_TARGET_CONFIG;
class DX11_RENDER_TARGET;
class DX11_RASTERIZER_STATE;
class DX11_DEPTH_STENCIL_STATE;
class DX11_BLEND_STATE;
class DX11_TEXTURE;
class DX11_SHADER;
class MATERIAL;
class ILIGHT;
class CAMERA;

enum renderOrders
{
	BASE_RO, // fill GBuffer
	GRID_FILL_RO, // fill voxel-grid
	SHADOW_RO, // generate shadow maps
	ILLUM_RO, // direct illumination
	GRID_ILLUM_RO, // illuminate voxel-grid
	GLOBAL_ILLUM_RO, // generate global illumination
	SKY_RO, // render sky
	GUI_RO, // render GUI
	POST_PROCESS_RO // perform post-processing
};

enum renderModes
{
	INDEXED_RM=0, // indexed draw-call
	NON_INDEXED_RM, //non-indexed draw-call
	COMPUTE_RM // dispatch of compute-shader
};

// SURFACE 
//   Represents a draw-/ dispatch batch.
class SURFACE
{
public:
  friend class DX11_RENDERER;

	SURFACE()
	{ 
		renderTarget = NULL;
		renderTargetConfig = NULL;
		renderOrder = BASE_RO;
		vertexBuffer = NULL;
		indexBuffer = NULL;
		camera = NULL;
		primitiveType = TRIANGLES_PRIMITIVE;
		firstIndex = 0;
		numElements = 0;
		material = NULL;
		colorTexture = NULL;
		normalTexture = NULL;
		specularTexture = NULL;
		for(int i=0;i<NUM_CUSTOM_TEXURES;i++)
			customTextures[i] = NULL;
		light = NULL;
		rasterizerState = NULL;
		depthStencilState = NULL;
		blendState = NULL;
		customUB = NULL;
	  for(int i=0;i<NUM_CUSTOM_STRUCTURED_BUFFERS;i++)
		  customSBs[i] = NULL;
		shader = NULL;
		renderMode = INDEXED_RM;
		numInstances = 1;
		numThreadGroupsX = 0;
		numThreadGroupsY = 0;
		numThreadGroupsZ = 0;
		ID = 0;
	}

	SURFACE(const SURFACE &rhs)
	{ 
		renderTarget = rhs.renderTarget;
		renderTargetConfig = rhs.renderTargetConfig;
		renderOrder = rhs.renderOrder;
		vertexBuffer = rhs.vertexBuffer;
		indexBuffer = rhs.indexBuffer;
		camera = rhs.camera;
		primitiveType = rhs.primitiveType;
		firstIndex = rhs.firstIndex;
		numElements = rhs.numElements;
		material = rhs.material;
		colorTexture = rhs.colorTexture;
		normalTexture = rhs.normalTexture;
		specularTexture = rhs.specularTexture; 
		for(int i=0;i<NUM_CUSTOM_TEXURES;i++)
			customTextures[i] = rhs.customTextures[i];
		light = rhs.light;
		rasterizerState = rhs.rasterizerState;
		depthStencilState = rhs.depthStencilState;
		blendState = rhs.blendState;
		customUB = rhs.customUB;
		for(int i=0;i<NUM_CUSTOM_STRUCTURED_BUFFERS;i++)
		  customSBs[i] = rhs.customSBs[i];
		shader = rhs.shader;
		renderMode = rhs.renderMode;
		numInstances = rhs.numInstances;
		numThreadGroupsX = rhs.numThreadGroupsX;
		numThreadGroupsY = rhs.numThreadGroupsY;
		numThreadGroupsZ = rhs.numThreadGroupsZ;
		ID = rhs.ID;
	}

	void SURFACE::operator= (const SURFACE &surface)
	{
		renderTarget = surface.renderTarget;
		renderTargetConfig = surface.renderTargetConfig;
		renderOrder = surface.renderOrder;
		vertexBuffer = surface.vertexBuffer;
		indexBuffer = surface.indexBuffer;
		camera = surface.camera;
		primitiveType = surface.primitiveType;
		firstIndex = surface.firstIndex;
		numElements = surface.numElements;
		material = surface.material;
		colorTexture = surface.colorTexture;
		normalTexture = surface.normalTexture;
		specularTexture = surface.specularTexture; 
		for(int i=0;i<NUM_CUSTOM_TEXURES;i++)
		  customTextures[i] = surface.customTextures[i];
		light = surface.light;
		rasterizerState = surface.rasterizerState;
		depthStencilState = surface.depthStencilState;
		blendState = surface.blendState;
		customUB = surface.customUB;
		for(int i=0;i<NUM_CUSTOM_STRUCTURED_BUFFERS;i++)
		  customSBs[i] = surface.customSBs[i];   
		shader = surface.shader;
		renderMode = surface.renderMode;
		numInstances = surface.numInstances;
		numThreadGroupsX = surface.numThreadGroupsX;
    numThreadGroupsY = surface.numThreadGroupsY;
		numThreadGroupsZ = surface.numThreadGroupsZ;
		ID = surface.ID;
	}

	int GetID() const
	{
		return ID;
	}

	DX11_RENDER_TARGET *renderTarget;
	RENDER_TARGET_CONFIG *renderTargetConfig;
	renderOrders renderOrder;
	DX11_VERTEX_BUFFER *vertexBuffer;
	DX11_INDEX_BUFFER *indexBuffer;
	CAMERA *camera;
	primitiveTypes primitiveType; 
	int firstIndex;
	int numElements; 
	MATERIAL *material;
	DX11_TEXTURE *colorTexture;
	DX11_TEXTURE *normalTexture;
	DX11_TEXTURE *specularTexture;
	DX11_TEXTURE *customTextures[NUM_CUSTOM_TEXURES];
	ILIGHT *light;
	DX11_RASTERIZER_STATE *rasterizerState;
  DX11_DEPTH_STENCIL_STATE *depthStencilState;
	DX11_BLEND_STATE *blendState;
	DX11_UNIFORM_BUFFER *customUB;
	DX11_STRUCTURED_BUFFER *customSBs[NUM_CUSTOM_STRUCTURED_BUFFERS];
	DX11_SHADER *shader;
	renderModes renderMode;
	int numInstances;
	int numThreadGroupsX;
	int numThreadGroupsY;
	int numThreadGroupsZ;

private:
	int ID; 
};

#endif


