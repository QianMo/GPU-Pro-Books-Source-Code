#include <stdafx.h>
#include <DEMO.h>
#include <DEMO_MESH.h>

void DEMO_MESH::Release()
{
	SAFE_DELETE_PLIST(subMeshes);
}

bool DEMO_MESH::Load(const char *filename)
{
	// cache pointer to GLOBAL_ILLUM post-processor
	globalIllumPP = (GLOBAL_ILLUM*)DEMO::renderer->GetPostProcessor("GLOBAL_ILLUM"); 
	if(!globalIllumPP)
		return false;

	// load ".mesh" file
	strcpy(name,filename);
	char filePath[DEMO_MAX_FILEPATH];
	if(!DEMO::fileManager->GetFilePath(filename,filePath))
		return false;
	FILE *file;
	fopen_s(&file,filePath,"rb");
	if(!file)
		return false;

	// check idString
	char idString[5];
	memset(idString,0,5);
	fread(idString,sizeof(char),4,file);
	if(strcmp(idString,"DMSH")!=0)
	{
		fclose(file);
		return false;
	}

	// check version
	int version;
	fread(&version,sizeof(int),1,file);
	if(version!=CURRENT_DEMO_MESH_VERSION)
	{
		fclose(file);
		return false;
	}

	// get number of vertices
	int numVertices;
	fread(&numVertices,sizeof(int),1,file);
	if(numVertices<3)
	{
		fclose(file);
		return false;
	}

	// load vertices
  GEOMETRY_VERTEX *vertices = new GEOMETRY_VERTEX[numVertices];
	if(!vertices)
	{
		fclose(file);
		return false;
	}
	fread(vertices,sizeof(GEOMETRY_VERTEX),numVertices,file);

	//  create vertex-buffer
	VERTEX_ELEMENT_DESC vertexLayout[4] = { POSITION_ELEMENT,R32G32B32_FLOAT_EF,0,
																					TEXCOORDS_ELEMENT,R32G32_FLOAT_EF,3,
																					NORMAL_ELEMENT,R32G32B32_FLOAT_EF,5,
																					TANGENT_ELEMENT,R32G32B32A32_FLOAT_EF,8 };
	vertexBuffer = DEMO::renderer->CreateVertexBuffer(vertexLayout,4,false,numVertices);
	if(!vertexBuffer)	
	{
    SAFE_DELETE_ARRAY(vertices);
		fclose(file);
		return false;
	}
  vertexBuffer->AddVertices(numVertices,(float*)vertices);
	vertexBuffer->Update();
	SAFE_DELETE_ARRAY(vertices);

	// get number of indices
	int numIndices;
	fread(&numIndices,sizeof(int),1,file);
	if(numIndices<3)
	{
		fclose(file);
		return false;
	}

  // load indices 
  int *indices = new int[numIndices];
	if(!indices)
	{
		fclose(file);
		return false;
	}
	fread(indices,sizeof(int),numIndices,file);

	// create index-buffer
	indexBuffer = DEMO::renderer->CreateIndexBuffer(false,numIndices);
	if(!indexBuffer)
	{
    SAFE_DELETE_ARRAY(indices);
		fclose(file);
		return false;
	}
	indexBuffer->AddIndices(numIndices,indices);
	indexBuffer->Update();
	SAFE_DELETE_ARRAY(indices);

	// get number of sub-meshes
	int numSubMeshes;
	fread(&numSubMeshes,sizeof(int),1,file);
	if(numSubMeshes<1)
	{
		fclose(file);
		return false;
	}

	// load/ create sub-meshes
	for(int i=0;i<numSubMeshes;i++)
	{
    DEMO_SUBMESH *subMesh = new DEMO_SUBMESH;
		if(!subMesh)
		{
			fclose(file);
			return false;
		}
		char materialName[256];
		fread(materialName,sizeof(char),256,file);
		subMesh->material = DEMO::resourceManager->LoadMaterial(materialName);
		if(!subMesh->material)
		{
			fclose(file);
			return false;
		}
		fread(&subMesh->firstIndex,sizeof(int),1,file);
		fread(&subMesh->numIndices,sizeof(int),1,file);
		subMeshes.AddElement(&subMesh);
	}

	fclose(file);
	
	// render into albedoGloss and normalDepth render-target of GBuffer
	RT_CONFIG_DESC rtcDesc;
	rtcDesc.numColorBuffers = 2;
	rtcDesc.firstColorBufferIndex = 1;
	multiRTC = DEMO::renderer->CreateRenderTargetConfig(rtcDesc);
	if(!multiRTC)
		return false;
	
	return true;
}

void DEMO_MESH::AddBaseSurfaces()
{
	for(int i=0;i<subMeshes.GetSize();i++) 
	{
		SURFACE surface;	
		surface.renderTarget = DEMO::renderer->GetRenderTarget(GBUFFER_RT_ID);
		surface.renderTargetConfig = multiRTC;
		surface.renderOrder = BASE_RO;
		surface.primitiveType = TRIANGLES_PRIMITIVE;
		surface.camera = DEMO::renderer->GetCamera(MAIN_CAMERA_ID);
		surface.vertexBuffer = vertexBuffer;
		surface.indexBuffer = indexBuffer;
		surface.firstIndex = subMeshes[i]->firstIndex;
		surface.numElements = subMeshes[i]->numIndices;
		surface.material = subMeshes[i]->material;
		surface.colorTexture = subMeshes[i]->material->colorTexture;
		surface.normalTexture = subMeshes[i]->material->normalTexture;
		surface.specularTexture =subMeshes[i]->material->specularTexture;
		surface.rasterizerState = subMeshes[i]->material->rasterizerState;
		surface.depthStencilState = subMeshes[i]->material->depthStencilState;
		surface.blendState = subMeshes[i]->material->blendState;
		surface.shader = subMeshes[i]->material->shader;
		DEMO::renderer->AddSurface(surface);
	}
}

void DEMO_MESH::AddGridSurfaces()
{
	// adds surfaces for generating fine and coarse resolution voxel-grid
	for(int i=0;i<2;i++)
	{
		for(int j=0;j<subMeshes.GetSize();j++) 
		{
			SURFACE surface;
			surface.renderOrder = GRID_FILL_RO;
			surface.vertexBuffer = vertexBuffer;
			surface.indexBuffer = indexBuffer;
			surface.primitiveType = TRIANGLES_PRIMITIVE;
			surface.firstIndex = subMeshes[j]->firstIndex;
			surface.numElements = subMeshes[j]->numIndices;
			surface.colorTexture = subMeshes[j]->material->colorTexture; 
			globalIllumPP->SetupGridSurface(surface,(gridTypes)i);
			DEMO::renderer->AddSurface(surface);
		}
	}
}
 
void DEMO_MESH::AddShadowMapSurfaces(int lightIndex)
{
	for(int i=0;i<subMeshes.GetSize();i++)
	{
		SURFACE surface;
		surface.vertexBuffer = vertexBuffer;
		surface.indexBuffer = indexBuffer;
		surface.primitiveType = TRIANGLES_PRIMITIVE;
		surface.firstIndex = subMeshes[i]->firstIndex;
		surface.numElements = subMeshes[i]->numIndices;
		DEMO::renderer->GetLight(lightIndex)->SetupShadowMapSurface(&surface); 
		DEMO::renderer->AddSurface(surface);
	}
}

void DEMO_MESH::AddSurfaces()
{
	AddBaseSurfaces();
	AddGridSurfaces();
	for(int i=0;i<DEMO::renderer->GetNumLights();i++) 
	{
		if((!DEMO::renderer->GetLight(i)->IsActive())||(!DEMO::renderer->GetLight(i)->HasShadow()))
			continue;
		AddShadowMapSurfaces(i);
	}
}
