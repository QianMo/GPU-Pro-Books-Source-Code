#pragma once

#include "Role.h"
#include "TaskContext.h"

class Camera;

/// Structure passed as parameter to render calls.
class RenderContext : public TaskContext
{
public:

	Camera* camera;
	/// Model transform inherited from ancestor nodes in the scene graph.
	const D3DXMATRIX* nodeTransformMatrix;
	/// Identifier to pick appropriate ShadedMesh role.
	const Role role;
	unsigned int instanceCount;

	/// Constructor.
	RenderContext(
		Theatre* theatre,
		ResourceOwner* localResourceOwner,
		Camera* camera,
		const D3DXMATRIX* nodeTransformMatrix,
		const Role role,
		unsigned int instanceCount)
		:TaskContext(theatre, localResourceOwner),role(role)
	{
		this->camera = camera;
		this->nodeTransformMatrix = nodeTransformMatrix;
		this->instanceCount = instanceCount;
	}
	RenderContext*		asRenderContext(){return this;}
};