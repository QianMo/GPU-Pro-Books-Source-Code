#ifndef RESOURCE_MANAGER_H
#define RESOURCE_MANAGER_H

#include <LIST.h>
#include <DX11_SHADER.h>
#include <DX11_TEXTURE.h>
#include <MATERIAL.h>
#include <FONT.h>
#include <DEMO_MESH.h>

// RESOURCE_MANAGER
//   Manages resources (shaders, textures, materials, fonts, demo-meshes).
class RESOURCE_MANAGER
{
public: 
	RESOURCE_MANAGER()
	{
	}

	~RESOURCE_MANAGER()
	{
		Release();
	}

	void Release();

	// loads ".sdr" shader-file (references the actual shader source files)
	DX11_SHADER* LoadShader(const char *fileName,int permutationMask=0);
	
	DX11_TEXTURE* LoadTexture(const char *fileName,DX11_SAMPLER *sampler=NULL); 

	// loads ".mtl" material-file
	MATERIAL* LoadMaterial(const char *fileName);

	// loads ".font" font-file 
	FONT* LoadFont(const char *fileName);

	FONT* GetFont(int index);

	// loads ".mesh" mesh-file
	DEMO_MESH* LoadDemoMesh(const char *fileName);

	DEMO_MESH* GetDemoMesh(int index) const;

	int GetNumFonts() const
	{
		return fonts.GetSize();
	}

	int GetNumDemoMeshes() const
	{
		return demoMeshes.GetSize();
	}

private:
	LIST<DX11_SHADER*> shaders;
	LIST<DX11_TEXTURE*> textures;
	LIST<MATERIAL*> materials;
	LIST<FONT*> fonts;
	LIST<DEMO_MESH*> demoMeshes;

};

#endif
