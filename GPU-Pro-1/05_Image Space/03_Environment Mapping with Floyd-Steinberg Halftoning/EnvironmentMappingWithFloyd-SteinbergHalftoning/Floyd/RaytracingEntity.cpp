#include "DXUT.h"
#include "RaytracingEntity.h"

RaytracingEntity::RaytracingEntity(Entity* decoratedEntity)
:EntityDecorator(EntityClass::RaytracingEntity, decoratedEntity)
{
}

RaytracingEntity::~RaytracingEntity(void)
{
}

void RaytracingEntity::setRaytracingMesh(RaytracingMesh* raytracingMesh)
{
	this->raytracingMesh = raytracingMesh;
}

RaytracingMesh* RaytracingEntity::getRaytracingMesh()
{
	return raytracingMesh;
}
