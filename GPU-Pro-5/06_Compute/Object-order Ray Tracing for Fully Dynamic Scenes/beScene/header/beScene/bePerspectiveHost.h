/************************************************************/
/* breeze Engine Entity System Module  (c) Tobias Zirr 2011 */
/************************************************************/

#pragma once
#ifndef BE_CORE_PERSPECTIVEHOST
#define BE_CORE_PERSPECTIVEHOST

#include "beScene.h"
#include <beCore/beMany.h>
#include <vector>

namespace beScene
{

class PipelinePerspective;

/// Collects perspectives to be rendered.
class PerspectiveHost
{
private:
	typedef std::vector<PipelinePerspective*> renderable_vector;
	renderable_vector m_render;

public:
	/// Range of perspectives.
	typedef beCore::Range<PipelinePerspective*const*> Perspectives;

	/// Constructor.
	BE_SCENE_API PerspectiveHost();
	/// Destructor.
	BE_SCENE_API ~PerspectiveHost();

	/// Adds a perspective.
	BE_SCENE_API void AddPerspective(PipelinePerspective *perspective);
	/// Removes a perspective.
	BE_SCENE_API void RemovePerspective(PipelinePerspective *perspective);
	/// Clears all perspectives.
	BE_SCENE_API void ClearPerspectives();
	/// Gets the stored perspectives.
	BE_SCENE_API Perspectives GetPerspectives() const { return beCore::MakeRangeN(&m_render[0], m_render.size()); };

};

} // namespace

#endif