#include <stdafx.h>
#include <Demo.h>
#include <PointLight.h>
#include <DemoMesh.h>

void DemoMesh::Release()
{
  SAFE_DELETE_PLIST(subMeshes);
}

bool DemoMesh::Load(const char *filename)
{
  // load ".mesh" file
  char filePath[DEMO_MAX_FILEPATH];
  if(!Demo::fileManager->GetFilePath(filename, filePath))
    return false;
  FILE *file;
  fopen_s(&file, filePath, "rb");
  if(!file)
    return false;

  // check idString
  char idString[5];
  memset(idString, 0, 5);
  fread(idString, sizeof(char), 4, file);
  if(strcmp(idString, "DMSH") != 0)
  {
    fclose(file);
    return false;
  }

  // check version
  unsigned int version;
  fread(&version, sizeof(unsigned int), 1, file);
  if(version != CURRENT_DEMO_MESH_VERSION)
  {
    fclose(file);
    return false;
  }

  // get number of vertices
  unsigned int numVertices;
  fread(&numVertices, sizeof(unsigned int), 1, file);
  if(numVertices < 3)
  {
    fclose(file);
    return false;
  }

  // load vertices
  GeometryVertex *vertices = new GeometryVertex[numVertices];
  if(!vertices)
  {
    fclose(file);
    return false;
  }
  fread(vertices, sizeof(GeometryVertex), numVertices, file);

  // add vertices to vertex buffer
  OGL_VertexBuffer *vertexBuffer = Demo::renderer->GetVertexBuffer(GEOMETRY_VB_ID);
  if(!vertexBuffer)
  {
    SAFE_DELETE_ARRAY(vertices);
    fclose(file);
    return false;
  }
  const unsigned int vertexOffset = vertexBuffer->GetVertexCount();
  vertexBuffer->AddVertices(numVertices, vertices);
  SAFE_DELETE_ARRAY(vertices);

  // get number of indices
  unsigned int numIndices;
  fread(&numIndices, sizeof(unsigned int), 1, file);
  if(numIndices < 3)
  {
    fclose(file);
    return false;
  }

  // load indices 
  unsigned int *indices = new unsigned int[numIndices];
  if(!indices)
  {
    fclose(file);
    return false;
  }
  fread(indices, sizeof(unsigned int), numIndices, file);
  if(vertexOffset > 0)
  {
    for(unsigned int i=0; i<numIndices; i++)
      indices[i] += vertexOffset;
  }

  // add indices to index buffer
  OGL_IndexBuffer *indexBuffer = Demo::renderer->GetIndexBuffer(GEOMETRY_IB_ID);
  if(!indexBuffer)
  {
    SAFE_DELETE_ARRAY(indices);
    fclose(file);
    return false;
  }
  const unsigned int indexOffset = indexBuffer->GetIndexCount();
  indexBuffer->AddIndices(numIndices, indices);
  SAFE_DELETE_ARRAY(indices);

  // get number of sub-meshes
  unsigned int numSubMeshes;
  fread(&numSubMeshes, sizeof(unsigned int), 1, file);
  if(numSubMeshes < 1)
  {
    fclose(file);
    return false;
  }

  // load/ create sub-meshes
  for(unsigned int i=0; i<numSubMeshes; i++)
  {
    DemoSubmesh *subMesh = new DemoSubmesh;
    if(!subMesh)
    {
      fclose(file);
      return false;
    }
    char materialName[256];
    fread(materialName, sizeof(char), 256, file);
    subMesh->material = Demo::resourceManager->LoadMaterial(materialName);
    if(!subMesh->material)
    {
      fclose(file);
      SAFE_DELETE(subMesh);
      return false;
    }
    fread(&subMesh->firstIndex, sizeof(unsigned int), 1, file);
    fread(&subMesh->numIndices, sizeof(unsigned int), 1, file);
    fread(&subMesh->boundingBox.mins[0], sizeof(float), 6, file);
    subMesh->firstIndex += indexOffset;
    subMeshes.AddElement(&subMesh);
  }

  fclose(file);
  
  // render into albedoGloss and normal render-target of GBuffers
  RtConfigDesc rtcDesc;
  rtcDesc.numColorBuffers = 2;
  rtcDesc.firstColorBufferIndex = 1;
  multiRTC = Demo::renderer->CreateRenderTargetConfig(rtcDesc);
  if(!multiRTC)
    return false;
  
  return true;
}

void DemoMesh::AddBaseSurfaces()
{
  for(unsigned int i=0; i<subMeshes.GetSize(); i++) 
  {
    GpuCmd gpuCmd(DRAW_CM);		
    gpuCmd.order = BASE_CO;
    gpuCmd.draw.renderTarget = Demo::renderer->GetRenderTarget(GBUFFERS_RT_ID);
    gpuCmd.draw.renderTargetConfig = multiRTC;
    gpuCmd.draw.primitiveType = TRIANGLES_PRIMITIVE;
    gpuCmd.draw.camera = Demo::renderer->GetCamera(MAIN_CAMERA_ID);
    gpuCmd.draw.vertexLayout = Demo::renderer->GetVertexLayout(GEOMETRY_VL_ID);
    gpuCmd.draw.vertexBuffer = Demo::renderer->GetVertexBuffer(GEOMETRY_VB_ID);
    gpuCmd.draw.indexBuffer = Demo::renderer->GetIndexBuffer(GEOMETRY_IB_ID);
    gpuCmd.draw.firstIndex = subMeshes[i]->firstIndex;
    gpuCmd.draw.numElements = subMeshes[i]->numIndices;
    gpuCmd.draw.textures[COLOR_TEX_ID] = subMeshes[i]->material->colorTexture;
    gpuCmd.draw.samplers[COLOR_TEX_ID] = Demo::renderer->GetSampler(TRILINEAR_SAMPLER_ID);
    gpuCmd.draw.textures[NORMAL_TEX_ID] = subMeshes[i]->material->normalTexture;
    gpuCmd.draw.samplers[NORMAL_TEX_ID] = Demo::renderer->GetSampler(TRILINEAR_SAMPLER_ID);
    gpuCmd.draw.textures[SPECULAR_TEX_ID] = subMeshes[i]->material->specularTexture;
    gpuCmd.draw.samplers[SPECULAR_TEX_ID] = Demo::renderer->GetSampler(TRILINEAR_SAMPLER_ID);
    gpuCmd.draw.rasterizerState = subMeshes[i]->material->rasterizerState;
    gpuCmd.draw.depthStencilState = subMeshes[i]->material->depthStencilState;
    gpuCmd.draw.blendState = subMeshes[i]->material->blendState;
    gpuCmd.draw.shader = subMeshes[i]->material->shader;
    Demo::renderer->AddGpuCmd(gpuCmd);
  }
}

void DemoMesh::AddCubeShadowMapSurfaces(PointLight *light)
{
  for(unsigned int i=0; i<subMeshes.GetSize(); i++) 
  {
    for(unsigned int j=0; j<6; j++)
    {
      if(!light->GetFrustum(j).IsAabbInside(subMeshes[i]->boundingBox))
        continue;
    
      GpuCmd gpuCmd(DRAW_CM);
      gpuCmd.order = SHADOW_CO;
      gpuCmd.draw.primitiveType = TRIANGLES_PRIMITIVE;
      gpuCmd.draw.vertexLayout = Demo::renderer->GetVertexLayout(SHADOW_VL_ID);
      gpuCmd.draw.vertexBuffer = Demo::renderer->GetVertexBuffer(GEOMETRY_VB_ID);
      gpuCmd.draw.indexBuffer = Demo::renderer->GetIndexBuffer(GEOMETRY_IB_ID);
      gpuCmd.draw.firstIndex = subMeshes[i]->firstIndex;
      gpuCmd.draw.numElements = subMeshes[i]->numIndices;
      light->SetupCubeShadowMapSurface(gpuCmd.draw, j);  
      Demo::renderer->AddGpuCmd(gpuCmd);
    }
  }
}

