#pragma once
#include "EntityDecorator.h"

class RaytracingMesh;

class RaytracingEntity :
	public EntityDecorator
{
	RaytracingMesh* raytracingMesh;
public:
	RaytracingEntity(Entity* decoratedEntity);
	~RaytracingEntity(void);

	void setRaytracingMesh(RaytracingMesh* raytracingMesh);
	RaytracingMesh* getRaytracingMesh();

};
