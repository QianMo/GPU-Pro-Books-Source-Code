#include "DXUT.h"
#include "Entity.h"
#include "ShadedMesh.h"
#include "Theatre.h"
#include "Camera.h"

//+ Render
Entity::Entity(const EntityClass& entityClass)
:entityClass(entityClass)
{
}

Entity::~Entity()
{
}

void Entity::animate(double dt)
{
}

void Entity::control(const ControlContext& context)
{
}

void Entity::interact(Entity* target)
{
	target->affect(this);
}

void Entity::affect(Entity* affector)
{
	if(affector == this)
		return;
	//entity - entity interaction
}


void Entity::getModelMatrix(D3DXMATRIX& modelMatrix)
{
	D3DXMatrixIdentity(&modelMatrix);
}

void Entity::getModelMatrixInverse(D3DXMATRIX& modelMatrixInverse)
{
	D3DXMatrixIdentity(&modelMatrixInverse);
}

void Entity::getRotationMatrixInverse(D3DXMATRIX& rotationMatrixInverse)
{
	D3DXMatrixIdentity(&rotationMatrixInverse);
}

D3DXVECTOR3 Entity::getTargetPosition()
{
	return D3DXVECTOR3(0, 0, 0);
}
