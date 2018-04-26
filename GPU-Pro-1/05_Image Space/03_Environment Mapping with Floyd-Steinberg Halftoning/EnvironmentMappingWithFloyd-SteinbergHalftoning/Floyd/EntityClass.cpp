#include "DXUT.h"
#include "EntityClass.h"
#include "Entity.h"
#include "ShadedEntity.h"
#include "EntityDecorator.h"

EntityClass::EntityClass(unsigned int id, bool isDecorator):
id(id), isDecorator(isDecorator)
{
}

bool EntityClass::operator==(const EntityClass& o) const
{
	return id == o.id;
}

const EntityClass EntityClass::ShadedEntity(0, false);
const EntityClass EntityClass::StaticEntity(1, true);
const EntityClass EntityClass::PhysicsEntity(2, true);
const EntityClass EntityClass::PhysicsCharacter(3, true);
const EntityClass EntityClass::RaytracingEntity(4, true);
const EntityClass EntityClass::OccluderSphereSetEntity(5, true);
const EntityClass EntityClass::KdTreeEntity(6, true);
const EntityClass EntityClass::SpotlightEntity(7, true);


::ShadedEntity* EntityClass::asShadedEntity(Entity* entity)
{
	while(entity->entityClass.isDecorator)
	{
		entity = ((EntityDecorator*)entity)->getDecoratedEntity();
		if(entity == NULL)
			EggERR("An EntityDecorator decorates a NULL entity.");
	}
	if(entity->entityClass == ShadedEntity)
		return (::ShadedEntity*)entity;
	EggERR("Upcast from Entity to ShadedEntity failed.");
	return NULL;
}

::KdTreeEntity* EntityClass::asKdTreeEntity(Entity* entity)
{
	while(entity->entityClass.isDecorator )
	{
		if(entity->entityClass == KdTreeEntity)
			return (::KdTreeEntity*)entity;
		entity = ((EntityDecorator*)entity)->getDecoratedEntity();
	}
	EggERR("Upcast from Entity to KdTreeEntity failed.");
	return NULL;
}
