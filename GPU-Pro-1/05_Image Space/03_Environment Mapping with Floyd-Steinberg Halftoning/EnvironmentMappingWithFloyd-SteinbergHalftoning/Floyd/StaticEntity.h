#pragma once
#include "entitydecorator.h"

class StaticEntity :
	public EntityDecorator
{
	D3DXMATRIX modelMatrix;
	D3DXMATRIX modelMatrixInverse;
	D3DXMATRIX rotationMatrix;
public:
	StaticEntity(Entity* decoratedEntity, const NxVec3& position, const NxQuat& orientation);
	~StaticEntity();

	void render(const RenderContext& context);


	void setPosition(const D3DXVECTOR3& position);

	/// Returns the model matrix. To be used for positioning of light sources and cameras attached to the entity.
	virtual void getModelMatrix(D3DXMATRIX& modelMatrix);
	/// Returns the inverse of the model matrix.
	virtual void getModelMatrixInverse(D3DXMATRIX& modelMatrixInverse);
	/// Returns the inverse of the rotation matrix. To be used for the view transformation of attached cameras.
	virtual void getRotationMatrixInverse(D3DXMATRIX& rotationMatrixInverse);

	/// Returns reference point for entity-relative AI targets.
	virtual D3DXVECTOR3 getTargetPosition();
};
