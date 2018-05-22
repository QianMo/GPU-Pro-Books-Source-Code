#ifndef DEMO_MESH_H
#define DEMO_MESH_H

#include <LIST.h>
#include <GLOBAL_ILLUM.h>

#define CURRENT_DEMO_MESH_VERSION 1

class RENDER_TARGET_CONFIG;
class DX11_VERTEX_BUFFER;
class DX11_INDEX_BUFFER;
class DX11_UNIFORM_BUFFER;
class DX11_SHADER;
class MATERIAL;

struct DEMO_SUBMESH
{
	DEMO_SUBMESH()
	{
		material = NULL;
		firstIndex = 0;
		numIndices = 0;
	}

	MATERIAL *material;
	int firstIndex;
	int numIndices;
};

// DEMO_MESH
//   Simple custom mesh format (".mesh") for storing non-animated meshes. Since for simplicity this demo does 
//   not use any kind of visibility determination algorithms, all sub-meshes with the same material are already
//   pre-batched. Furthermore the normalized normals and tangents as well as the tangent-space handedness are 
//   already calculated.
class DEMO_MESH
{
public:
	DEMO_MESH()
	{
		active = true;
	  vertexBuffer = NULL;
		indexBuffer = NULL;
		multiRTC = NULL;
		globalIllumPP = NULL;
	}

	~DEMO_MESH()
	{
		Release();
	}

	void Release();

	bool Load(const char *filename);

	void AddSurfaces();

	void SetActive(bool active)
	{
		this->active = active;
	}

	bool IsActive() const
	{
		return active;
	}

	const char* GetName() const
	{
		return name;
	}

private:	 
	// adds surfaces for filling the GBuffer
	void AddBaseSurfaces();

	// adds surfaces for generating the voxel-grids
	void AddGridSurfaces();

	// adds surfaces for generating shadow map for specified light
	void AddShadowMapSurfaces(int lightIndex);
	
	LIST<DEMO_SUBMESH*> subMeshes; // list of all sub-meshes
	bool active;
	char name[DEMO_MAX_FILENAME];

  DX11_VERTEX_BUFFER *vertexBuffer;
	DX11_INDEX_BUFFER *indexBuffer;
	RENDER_TARGET_CONFIG *multiRTC; 
	GLOBAL_ILLUM *globalIllumPP;

};

#endif