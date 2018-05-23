/************************************************************/
/* breeze Engine Scene Module          (c) Tobias Zirr 2011 */
/************************************************************/

#include "beSceneInternal/stdafx.h"
#include "beScene/beRenderableHost.h"
#include "beScene/beRenderingPipeline.h"
#include <lean/functional/algorithm.h>
#include <lean/logging/errors.h>

namespace beScene
{

// Constructor.
RenderableHost::RenderableHost()
{
}

// Destructor.
RenderableHost::~RenderableHost()
{
}

// Adds a renderable controller.
void RenderableHost::AddRenderable(Renderable *renderable)
{
	if (!renderable)
	{
		LEAN_LOG_ERROR_MSG("renderable may not be nullptr");
		return;
	}

	m_render.push_back(renderable);
}

// Removes a renderable controller.
void RenderableHost::RemoveRenderable(Renderable *renderable)
{
	lean::remove(m_render, renderable);
}

} // namespace
