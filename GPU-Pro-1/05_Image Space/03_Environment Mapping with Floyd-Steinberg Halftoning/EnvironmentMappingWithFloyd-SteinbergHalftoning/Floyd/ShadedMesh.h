#pragma once

#include "Directory.h"
#include "Role.h"
#include "EffectSettings.h"

class RenderContext;
class Repertoire;
class ShadingMaterial;
class Rendition;

class ShadedMesh
{
	/// D3D mesh reference.
	ID3DX10Mesh* mesh;
	
	typedef CompositMap<const Role, ShadingMaterial*, CompareRoles>  ShadingMaterialDirectory;
	ShadingMaterialDirectory shadingMaterialDirectory;

	EffectSettings overrides;

public:
	/// Basic constructor, used when building a ShadedMesh explicitly. (in XML level file)
	ShadedMesh(ID3DX10Mesh* mesh);
	/// Destructor. Releases contained Roles.
	~ShadedMesh(void);

	// Renders the mesh using the Role indicated in context.
	void render(const RenderContext& context);

	ID3DX10Mesh* getMesh();

	void addShadingMaterial(const Role role, ShadingMaterial* shadingMaterial);

	EffectSettings* getOverrides() {return &overrides;}
};
