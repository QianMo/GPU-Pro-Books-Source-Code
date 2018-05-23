/************************************************************/
/* breeze Engine Scene Module          (c) Tobias Zirr 2011 */
/************************************************************/

#include "beSceneInternal/stdafx.h"
#include "beScene/bePerspectiveHost.h"

#include <lean/functional/algorithm.h>
#include <lean/logging/errors.h>

namespace beScene
{

// Constructor.
PerspectiveHost::PerspectiveHost()
{
}

// Destructor.
PerspectiveHost::~PerspectiveHost()
{
}

// Adds a renderable controller.
void PerspectiveHost::AddPerspective(PipelinePerspective *perspective)
{
	if (!perspective)
	{
		LEAN_LOG_ERROR_MSG("perspective may not be nullptr");
		return;
	}

	m_render.push_back(perspective);
}

// Removes a renderable controller.
void PerspectiveHost::RemovePerspective(PipelinePerspective *perspective)
{
	lean::remove(m_render, perspective);
}

// Clears all perspectives.
void PerspectiveHost::ClearPerspectives()
{
	m_render.clear();
}

} // namespace
