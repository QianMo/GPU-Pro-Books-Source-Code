#include "DXUT.h"
#include "ShadedEntity.h"
#include "ShadedMesh.h"

ShadedEntity::ShadedEntity(ShadedMesh* shadedMesh)
:Entity(EntityClass::ShadedEntity)
{
	this->shadedMesh = shadedMesh;
}

ShadedEntity::~ShadedEntity(void)
{
}

ShadedMesh* ShadedEntity::getShadedMesh()
{
	return shadedMesh;
}

void ShadedEntity::render(const RenderContext& context)
{
	shadedMesh->render(context);
}

