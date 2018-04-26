#include "DXUT.h"
#include "EntityDecorator.h"

EntityDecorator::EntityDecorator(const EntityClass& entityClass, Entity* decoratedEntity)
:Entity(entityClass)
{
	this->decoratedEntity= decoratedEntity;
}

EntityDecorator::~EntityDecorator(void)
{
	delete decoratedEntity;
}

void EntityDecorator::render(const RenderContext& context)
{
	decoratedEntity->render(context);
}

void EntityDecorator::animate(double dt)
{
	decoratedEntity->animate(dt);
}

void EntityDecorator::control(const ControlContext& context)
{
	decoratedEntity->control(context);
}

void EntityDecorator::interact(Entity* target)
{
	decoratedEntity->interact(target);
}

void EntityDecorator::affect(Entity* affector)
{
	decoratedEntity->affect(affector);
}

void EntityDecorator::getModelMatrix(D3DXMATRIX& modelMatrix)
{
	decoratedEntity->getModelMatrix(modelMatrix);
}

void EntityDecorator::getModelMatrixInverse(D3DXMATRIX& modelMatrixInverse)
{
	decoratedEntity->getModelMatrixInverse(modelMatrixInverse);
}

void EntityDecorator::getRotationMatrixInverse(D3DXMATRIX& rotationMatrixInverse)
{
	decoratedEntity->getRotationMatrixInverse(rotationMatrixInverse);
}

D3DXVECTOR3 EntityDecorator::getTargetPosition()
{
	return decoratedEntity->getTargetPosition();
}