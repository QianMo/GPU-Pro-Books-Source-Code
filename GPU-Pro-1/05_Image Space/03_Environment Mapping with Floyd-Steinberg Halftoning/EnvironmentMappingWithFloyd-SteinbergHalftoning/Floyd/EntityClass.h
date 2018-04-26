#pragma once

class Entity;
class ShadedEntity;
class StaticEntity;
class PhysicsEntity;
class PhysicsCharacter;
class RaytracingEntity;
class OccluderSphereSetEntity;
class KdTreeEntity;

class EntityClass
{
	const unsigned int id;
	const bool isDecorator;
	EntityClass(unsigned int id, bool isDecorator);

	bool operator==(const EntityClass& o) const;
public:
	static const EntityClass ShadedEntity;
	static const EntityClass StaticEntity;
	static const EntityClass PhysicsEntity;
	static const EntityClass PhysicsCharacter;
	static const EntityClass RaytracingEntity;
	static const EntityClass OccluderSphereSetEntity;
	static const EntityClass KdTreeEntity;
	static const EntityClass SpotlightEntity;

	static ::ShadedEntity* asShadedEntity(Entity* entity);
	static ::KdTreeEntity* asKdTreeEntity(Entity* entity);


};
