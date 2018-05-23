/************************************************************/
/* breeze Engine Entity System Module  (c) Tobias Zirr 2011 */
/************************************************************/

#pragma once
#ifndef BE_ENTITYSYSTEM_RENDERABLEHOST
#define BE_ENTITYSYSTEM_RENDERABLEHOST

#include "beEntitySystem.h"
#include "beRenderable.h"
#include <vector>

namespace beEntitySystem
{

/// Animated interface.
class RenderableHost : public Renderable
{
private:
	typedef std::vector<Renderable*> renderable_vector;
	renderable_vector m_render;

public:
	/// Constructor.
	BE_ENTITYSYSTEM_API RenderableHost();
	/// Destructor.
	BE_ENTITYSYSTEM_API ~RenderableHost();

	/// Renders all renderable content.
	BE_ENTITYSYSTEM_API void Render();

	/// Adds a renderable controller.
	BE_ENTITYSYSTEM_API void AddRenderable(Renderable *renderable);
	/// Removes a renderable controller.
	BE_ENTITYSYSTEM_API void RemoveRenderable(Renderable *renderable);
};

} // namespace

#endif