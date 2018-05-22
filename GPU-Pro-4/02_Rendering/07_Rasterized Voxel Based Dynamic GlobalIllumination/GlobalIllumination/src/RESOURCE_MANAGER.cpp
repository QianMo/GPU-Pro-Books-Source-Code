#include <stdafx.h>
#include <DEMO.h>
#include <RESOURCE_MANAGER.h>

#include <DX11_TEXTURE.h>

void RESOURCE_MANAGER::Release()
{ 
	SAFE_DELETE_PLIST(shaders);
	SAFE_DELETE_PLIST(textures);
	SAFE_DELETE_PLIST(materials);
	SAFE_DELETE_PLIST(fonts); 
	SAFE_DELETE_PLIST(demoMeshes);
}

DX11_SHADER* RESOURCE_MANAGER::LoadShader(const char *fileName,int permutationMask)
{
	for(int i=0;i<shaders.GetSize();i++)
	{
		if((strcmp(shaders[i]->GetName(),fileName)==0)&&
			 (shaders[i]->GetPermutationMask()==permutationMask))
			return shaders[i];
	}
	DX11_SHADER *shader = new DX11_SHADER;
	if(!shader)
		return NULL;
	if(!shader->Load(fileName,permutationMask))
	{
		SAFE_DELETE(shader);
		return NULL;
	}
  shaders.AddElement(&shader);
	return shader;
}

DX11_TEXTURE* RESOURCE_MANAGER::LoadTexture(const char *fileName,DX11_SAMPLER *sampler)
{
	for(int i=0;i<textures.GetSize();i++)
	{
		if((strcmp(textures[i]->GetName(),fileName)==0)&&
			(textures[i]->GetSampler()==sampler))
			return textures[i];
	}
	DX11_TEXTURE *texture = new DX11_TEXTURE;
	if(!texture)
		return NULL;
	if(!texture->LoadFromFile(fileName,sampler))
	{
		SAFE_DELETE(texture);
		return NULL;
	}
	textures.AddElement(&texture);
	return texture;
}

MATERIAL* RESOURCE_MANAGER::LoadMaterial(const char *fileName)
{
	for(int i=0;i<materials.GetSize();i++)
	{
		if(strcmp(materials[i]->GetName(),fileName)==0)
			return materials[i];
	}
	MATERIAL *material = new MATERIAL;
	if(!material)
		return NULL;
	if(!material->Load(fileName))
	{
		SAFE_DELETE(material);
		return NULL;
	}
	materials.AddElement(&material);
	return material;
}

FONT* RESOURCE_MANAGER::LoadFont(const char *fileName)
{
	for(int i=0;i<fonts.GetSize();i++)
	{
    if(strcmp(fonts[i]->GetName(),fileName)==0)
			return fonts[i];
	}
	FONT *font = new FONT;
	if(!font)
		return NULL;
	if(!font->Load(fileName))
	{
		SAFE_DELETE(font);
		return NULL;
	}
	fonts.AddElement(&font); 
	return font;
}

FONT* RESOURCE_MANAGER::GetFont(int index)
{
	if((index<0)||(index>=fonts.GetSize()))
		return NULL;
	return fonts[index];
}

DEMO_MESH* RESOURCE_MANAGER::LoadDemoMesh(const char *fileName)
{
	DEMO_MESH *demoMesh = new DEMO_MESH;
	if(!demoMesh)
		return NULL;
	if(!demoMesh->Load(fileName))
	{
		SAFE_DELETE(demoMesh);
		return NULL;
	}
	demoMeshes.AddElement(&demoMesh);
	return demoMesh;
}

DEMO_MESH* RESOURCE_MANAGER::GetDemoMesh(int index) const
{
	if((index<0)||(index>=demoMeshes.GetSize()))
		return NULL;
	return demoMeshes[index];
}



