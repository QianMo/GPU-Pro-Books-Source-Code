#include "DXUT.h"
#include "StaticEntity.h"
#include "Theatre.h"
#include "Camera.h"

StaticEntity::StaticEntity(Entity* decoratedEntity, const NxVec3& position, const NxQuat& orientation)
:EntityDecorator(EntityClass::StaticEntity, decoratedEntity)
{
	D3DXMatrixTranslation(&modelMatrix, position.x, position.y, position.z);
	D3DXMatrixInverse(&modelMatrixInverse, NULL, &modelMatrix);
	NxMat33 m(orientation);
	D3DXMatrixIdentity(&rotationMatrix);
	m.getColumnMajorStride4((float*)&rotationMatrix);
}

StaticEntity::~StaticEntity()
{
}

void StaticEntity::render(const RenderContext& context)
{
	context.theatre->getEffect()->GetVariableByName("modelMatrix")->AsMatrix()->SetMatrix((float*)&modelMatrix);
	context.theatre->getEffect()->GetVariableByName("modelMatrixInverse")->AsMatrix()->SetMatrix((float*)&modelMatrixInverse);

	D3DXMATRIX modelViewProjMatrix = modelMatrix * context.camera->getViewMatrix() * context.camera->getProjMatrix();
	context.theatre->getEffect()->GetVariableByName("modelViewProjMatrix")->AsMatrix()->SetMatrix((float*)&modelViewProjMatrix);

	D3DXMATRIX modelViewMatrix = modelMatrix* context.camera->getViewMatrix();
	context.theatre->getEffect()->GetVariableByName("modelViewMatrix")->AsMatrix()->SetMatrix((float*)&modelViewMatrix);

	decoratedEntity->render(context);
}

void StaticEntity::setPosition(const D3DXVECTOR3& position)
{
	D3DXMatrixTranslation(&modelMatrix, position.x, position.y, position.z);
	D3DXMatrixInverse(&modelMatrixInverse, NULL, &modelMatrix);
}


void StaticEntity::getModelMatrix(D3DXMATRIX& modelMatrix)
{
	modelMatrix = this->modelMatrix;
}

void StaticEntity::getModelMatrixInverse(D3DXMATRIX& modelMatrixInverse)
{
	modelMatrixInverse = this->modelMatrixInverse;
}

void StaticEntity::getRotationMatrixInverse(D3DXMATRIX& rotationMatrixInverse)
{
	D3DXMatrixTranspose(&rotationMatrixInverse, &rotationMatrix);
}


D3DXVECTOR3 StaticEntity::getTargetPosition()
{
	D3DXVECTOR3 ret;
	D3DXVec3TransformCoord(&ret, &D3DXVECTOR3(0, 0, 0), &modelMatrix);
	return ret;
}