/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#pragma once
#ifndef BE_SCENE_PERSPECTIVE_POOL
#define BE_SCENE_PERSPECTIVE_POOL

#include "beScene.h"
#include <beCore/beShared.h>
#include <beCore/bePool.h>
#include "beRenderingLimits.h"
#include <lean/smart/resource_ptr.h>
#include <lean/smart/scoped_ptr.h>

namespace beScene
{

struct PerspectiveDesc;
class PipelinePerspective;
class Pipe;
class PipelineProcessor;

class PipePool;

/// Pool of pipes.
class PerspectivePool : public beCore::Resource
{
public:
	typedef beCore::Pool<PipelinePerspective> pool;

	lean::resource_ptr<PipePool> m_pPipePool;

private:
	pool m_pool;

public:
	/// Constructor.
	BE_SCENE_API PerspectivePool(PipePool *pPipePool);
	/// Destructor.
	BE_SCENE_API ~PerspectivePool();

	/// Gets a perspective.
	BE_SCENE_API PipelinePerspective* GetPerspective(Pipe *pPipe,
		PipelineProcessor *pProcessor = nullptr, PipelineStageMask stageMask = NormalPipelineStages);
	/// Gets a perspective.
	BE_SCENE_API PipelinePerspective* GetPerspective(const PerspectiveDesc &desc, Pipe *pPipe,
		PipelineProcessor *pProcessor = nullptr, PipelineStageMask stageMask = NormalPipelineStages);

	/// Gets the optional pipe pool.
	LEAN_INLINE PipePool* GetPipePool() const { return m_pPipePool; }
};

} // namespace

#endif