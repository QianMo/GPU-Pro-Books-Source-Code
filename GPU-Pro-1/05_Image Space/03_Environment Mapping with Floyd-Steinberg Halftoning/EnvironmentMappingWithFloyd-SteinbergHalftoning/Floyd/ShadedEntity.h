#pragma once
#include "Entity.h"

class ShadedMesh;

class ShadedEntity :
	public Entity
{
	/// The geometry and shaders the entity is using.
	ShadedMesh* shadedMesh;
public:
	ShadedEntity(ShadedMesh* shadedMesh);
	~ShadedEntity(void);

	ShadedMesh* getShadedMesh();

	void render(const RenderContext& context);
};
