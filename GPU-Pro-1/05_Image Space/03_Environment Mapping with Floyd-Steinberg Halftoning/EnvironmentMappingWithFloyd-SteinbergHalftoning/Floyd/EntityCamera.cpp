#include "DXUT.h"
#include "EntityCamera.h"
#include "Entity.h"

const D3DXVECTOR3& EntityCamera::getEyePosition()
{
	D3DXMATRIX entityModelMatrix;
	owner->getModelMatrix(entityModelMatrix);
	
	D3DXVec3TransformCoord(&worldEyePosition, &eyePosition, &entityModelMatrix);
	return worldEyePosition;
}

const D3DXMATRIX& EntityCamera::getOrientProjMatrixInverse()
{
	D3DXMATRIX entityRotationMatrixInverse;
	owner->getRotationMatrixInverse(entityRotationMatrixInverse);
	D3DXMATRIX eyePosTranslationMatrix;
	D3DXMatrixTranslation(&eyePosTranslationMatrix, eyePosition.x, eyePosition.y, eyePosition.z);

	D3DXMatrixInverse(&worldOrientProjMatrixInverse, NULL, &(entityRotationMatrixInverse * eyePosTranslationMatrix * viewMatrix  * projMatrix));

	return worldOrientProjMatrixInverse;
}

const D3DXMATRIX& EntityCamera::getViewMatrix()
{
	D3DXMATRIX entityModelMatrixInverse;
	owner->getModelMatrixInverse(entityModelMatrixInverse);

	worldViewMatrix = entityModelMatrixInverse * viewMatrix;

	return worldViewMatrix;
}

const D3DXMATRIX& EntityCamera::getProjMatrix() 
{
	return projMatrix;
}

EntityCamera::EntityCamera(Entity* owner)
{
	this->owner = owner;

	this->eyePosition = D3DXVECTOR3(0, 0, 0);
	this->lookAt = D3DXVECTOR3(10, 0, 0);
	this->up  = D3DXVECTOR3(0, 1, 0);
	D3DXMatrixLookAtLH(&viewMatrix, &eyePosition, &lookAt, &up);

	this->fov = 3.14;
	this->aspect = 1;
	this->front = 1;
	this->back = 1000;
	D3DXMatrixPerspectiveFovLH(&projMatrix, fov, aspect, front, back);
}

EntityCamera::EntityCamera(Entity* owner, const D3DXVECTOR3& eyePosition, const D3DXVECTOR3& lookAt, const D3DXVECTOR3& up)
{
	this->owner = owner;
	this->eyePosition = eyePosition;
	this->lookAt = lookAt;
	this->up = up;
	D3DXMatrixLookAtLH(&viewMatrix, &eyePosition, &lookAt, &up);

	this->fov = 3.14;
	this->aspect = 1;
	this->front = 1;
	this->back = 1000;
	D3DXMatrixPerspectiveFovLH(&projMatrix, fov, aspect, front, back);
}

EntityCamera::EntityCamera(Entity* owner, const D3DXVECTOR3& eyePosition, const D3DXVECTOR3& lookAt, const D3DXVECTOR3& up, double fov, double aspect, double front, double back)
{
	this->owner = owner;
	this->eyePosition = eyePosition;
	this->lookAt = lookAt;
	this->up = up;
	D3DXMatrixLookAtLH(&viewMatrix, &eyePosition, &lookAt, &up);
	
	this->fov = fov;
	this->aspect = aspect;
	this->front = front;
	this->back = back;

	D3DXMatrixPerspectiveFovLH(&projMatrix, fov, aspect, front, back);
}

void EntityCamera::setAspect(double aspect)
{
	this->aspect = aspect;
	D3DXMatrixPerspectiveFovLH(&projMatrix, fov, aspect, front, back);
}
