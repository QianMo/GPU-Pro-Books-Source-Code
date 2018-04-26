#pragma once
#include "SceneManager.h"
#include "Directory.h"

class Theatre;
class XMLNode;
class Scene;
class RaytracingMesh;
class RaytracingEntity;

class RaytracingScene :
	public SceneManager
{
	typedef CompositList<RaytracingMesh*> RaytracingMeshList;
//	typedef std::map<std::wstring, RaytracingMesh*> RaytracingMeshDirectory;
	typedef CompositList<RaytracingEntity*> RaytracingEntityList;

	RaytracingMeshList raytracingMeshList;
//	RaytracingMeshDirectory raytracingMeshDirectory;
	RaytracingEntityList raytracingEntityList;

	void createRaytracingResources();

	/// A texture array. Every slice contains the kd-tree nodes of a ray tracing mesh.
	ID3D10Texture1D* raytracingBIHNodeTableArray;
	/// A texture array. Every slice contains the triangles of a ray tracing mesh.
	ID3D10Texture2D* raytracingTriangleTableArray;

	/// Shader Resource View for rayTraceKDNodeTableArray;
	ID3D10ShaderResourceView* raytracingBIHNodeTableArraySRV;
	/// Shader Resource View for rayTraceTriangleTableArray;
	ID3D10ShaderResourceView* raytracingTriangleTableArraySRV;

	struct RaytracingEntityData
	{
		D3DXMATRIX modelMatrix;
		D3DXMATRIX modelMatrixInverse;
		D3DXVECTOR4	diffuse;
		D3DXVECTOR4	specular;
		unsigned int meshIndex;
		unsigned int padding1;
		unsigned int padding2;
		unsigned int padding3;
	};

	/// GPU buffer resource containing ray tracing entity structures.
	ID3D10Buffer*	raytracingEntityBuffer;

	void updateRaytracingResources();


	Entity* decorateEntity(Entity* entity, XMLNode& entityNode, bool& processed);
	void finish();

public:
	RaytracingScene(Theatre* theatre, XMLNode& xMainNode);

	void render(const RenderContext& context);
	void animate(double dt, double t);
	void control(const ControlContext& context);
	void processMessage( const MessageContext& context);

	~RaytracingScene();
};
