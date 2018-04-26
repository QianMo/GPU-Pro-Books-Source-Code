#include "DXUT.h"
#include "ShadedMesh.h"
#include "Rendition.h"
#include "Repertoire.h"
#include "RenderContext.h"
#include "ShadingMaterial.h"
#include "Theatre.h"
#include "Play.h"

ShadedMesh::ShadedMesh(ID3DX10Mesh* mesh)
{
	this->mesh = mesh;
}

ShadedMesh::~ShadedMesh(void)
{
	shadingMaterialDirectory.deleteAll();
}


void ShadedMesh::render(const RenderContext& context)
{
	ShadingMaterialDirectory::iterator iShadingMaterial = shadingMaterialDirectory.find(context.role);
	if(iShadingMaterial != shadingMaterialDirectory.end())
		iShadingMaterial->second->renderMesh(context, mesh, overrides);
	else
		EggERR("ShadedMesh does not support requested role. [This is not a problem, if it was intended. Nothing will be rendered.]" << context.theatre->getPlay()->getRoleName(context.role));
}


ID3DX10Mesh* ShadedMesh::getMesh()
{
	return mesh;
}

void ShadedMesh::addShadingMaterial(const Role role, ShadingMaterial* shadingMaterial)
{
	shadingMaterialDirectory[role] = shadingMaterial;
}