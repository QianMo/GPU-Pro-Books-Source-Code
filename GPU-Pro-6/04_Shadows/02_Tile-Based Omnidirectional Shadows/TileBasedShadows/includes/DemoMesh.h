#ifndef DEMO_MESH_H
#define DEMO_MESH_H

#include <List.h>
#include <Aabb.h>

#define CURRENT_DEMO_MESH_VERSION 2

class RenderTargetConfig;
class OGL_VertexLayout;
class OGL_VertexBuffer;
class OGL_IndexBuffer;
class OGL_UniformBuffer;
class OGL_Shader;
class Material;
class PointLight;

struct DemoSubmesh
{
  DemoSubmesh():
    material(NULL),
    firstIndex(0),
    numIndices(0)
  {
  }

  Material *material;
  unsigned int firstIndex;
  unsigned int numIndices;
  Aabb boundingBox;
};

// DemoMesh
//
// Simple custom mesh format (".mesh") for storing non-animated meshes. The normalized normals and tangents, 
// the tangent-space handedness and the axis-aligned bounding-boxes of the sub-meshes are already calculated.
class DemoMesh
{
public:
  DemoMesh():
    multiRTC(NULL)
  {
  }

  ~DemoMesh()
  {
    Release();
  }

  void Release();

  bool Load(const char *filename);

  // adds surfaces for filling the GBuffers
  void AddBaseSurfaces();

  // adds surfaces for generating cube shadow map
  void AddCubeShadowMapSurfaces(PointLight *light);

  const DemoSubmesh* GetSubMesh(unsigned int index) const
  {
    assert(index < subMeshes.GetSize());
    return subMeshes[index];
  }

  unsigned int GetNumSubMeshes() const
  {
    return subMeshes.GetSize();   
  }

private:	 
  List<DemoSubmesh*> subMeshes; // list of all sub-meshes
  RenderTargetConfig *multiRTC; 

};

#endif