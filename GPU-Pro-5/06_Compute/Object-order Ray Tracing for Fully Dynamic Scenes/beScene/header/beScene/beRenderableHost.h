/************************************************************/
/* breeze Engine Entity System Module  (c) Tobias Zirr 2011 */
/************************************************************/

#pragma once
#ifndef BE_CORE_RENDERABLEHOST
#define BE_CORE_RENDERABLEHOST

#include "beScene.h"
#include "beRenderable.h"
#include <beCore/beMany.h>
#include <vector>

namespace beScene
{

/// Animated interface.
class RenderableHost
{
private:
	typedef std::vector<Renderable*> renderable_vector;
	renderable_vector m_render;

public:
	/// Range of renderables.
	typedef beCore::Range<Renderable*const*> Renderables;

	/// Constructor.
	BE_SCENE_API RenderableHost();
	/// Destructor.
	BE_SCENE_API ~RenderableHost();

	/// Adds a renderable controller.
	BE_SCENE_API void AddRenderable(Renderable *renderable);
	/// Removes a renderable controller.
	BE_SCENE_API void RemoveRenderable(Renderable *renderable);
	/// Gets the stored renderables.
	BE_SCENE_API Renderables GetRenderables() const { return beCore::MakeRangeN(&m_render[0], m_render.size()); };
};

} // namespace

#endif