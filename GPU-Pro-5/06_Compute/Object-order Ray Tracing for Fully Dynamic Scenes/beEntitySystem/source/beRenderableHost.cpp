/************************************************************/
/* breeze Engine Entity System Module  (c) Tobias Zirr 2011 */
/************************************************************/

#include "beEntitySystemInternal/stdafx.h"
#include "beEntitySystem/beRenderableHost.h"
#include <lean/functional/algorithm.h>
#include <lean/logging/errors.h>

namespace beEntitySystem
{

// Constructor.
RenderableHost::RenderableHost()
{
}

// Destructor.
RenderableHost::~RenderableHost()
{
}

// Renders all renderable content.
void RenderableHost::Render()
{
	for (renderable_vector::const_iterator it = m_render.begin(); it != m_render.end(); ++it)
		(*it)->Render();
}

// Adds a renderable controller.
void RenderableHost::AddRenderable(Renderable *renderable)
{
	if (!renderable)
	{
		LEAN_LOG_ERROR_MSG("renderable may not be nullptr");
		return;
	}

	lean::push_unique(m_render, renderable);
}

// Removes a renderable controller.
void RenderableHost::RemoveRenderable(Renderable *renderable)
{
	lean::remove(m_render, renderable);
}

} // namespace
