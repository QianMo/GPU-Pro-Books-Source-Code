#pragma once
#include "Node.h"
#include "EntityClass.h"

class NodeGroup;

/// Scene graph leaf node, an entity in the virtual world.
class Entity
	: public Node
{
	friend class EntityClass;
	const EntityClass& entityClass;
public:
	/// Constructor.
	Entity(const EntityClass& entityClass);
	/// Destructor. Does not release ShadedMesh.
	virtual ~Entity(void);

	/// Renders the entity as seen from a given camera in a given role.
	virtual void render(const RenderContext& context)=0;
	/// Updates time-varying entity proporties.
	virtual void animate(double dt);
	/// Evaluates effects (forces, inputs, AI, etc.) on entity.
	virtual void control(const ControlContext& context);
	/// Invoked as part of the control sequence. Invokes target->affect(this).
	virtual void interact(Entity* target);
	/// Applies effect of affector on entity. Invoked as part of the control sequence.
	virtual void affect(Entity* affector);
	/// Returns the model matrix. To be used for positioning of light sources and cameras attached to the entity.
	virtual void getModelMatrix(D3DXMATRIX& modelMatrix);
	/// Returns the inverse of the model matrix.
	virtual void getModelMatrixInverse(D3DXMATRIX& modelMatrixInverse);
	/// Returns the inverse of the rotation matrix. To be used for the view transformation of attached cameras.
	virtual void getRotationMatrixInverse(D3DXMATRIX& rotationMatrixInverse);

	/// Returns reference point for entity-relative AI targets.
	virtual D3DXVECTOR3 getTargetPosition();
};
