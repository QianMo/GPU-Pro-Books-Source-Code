/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#pragma once
#ifndef BE_SCENE_RENDERABLE
#define BE_SCENE_RENDERABLE

#include "beScene.h"
#include "beRenderingLimits.h"

namespace beScene
{

class PipelinePerspective;
struct PipelineQueueID;
struct PipelineStageDesc;
struct RenderQueueDesc;
class RenderContext;

/// Renderable interface.
class LEAN_INTERFACE Renderable
{
	LEAN_INTERFACE_BEHAVIOR(Renderable)

public:
	/// Perform visiblity culling.
	virtual void Cull(PipelinePerspective &perspective) const = 0;
	/// Prepares the given render queue for the given perspective, returning true if active.
	virtual bool Prepare(PipelinePerspective &perspective, PipelineQueueID queueID,
		const PipelineStageDesc &stageDesc, const RenderQueueDesc &queueDesc) const = 0;
	/// Prepares the collected render queues for the given perspective.
	BE_SCENE_API virtual void Collect(PipelinePerspective &perspective) const { }
	/// Performs optional optimization such as sorting.
	virtual void Optimize(const PipelinePerspective &perspective, PipelineQueueID queueID) const = 0;
	/// Prepares rendering from the collected render queues for the given perspective.
	BE_SCENE_API virtual void PreRender(const PipelinePerspective &perspective, const RenderContext &context) const { }
	/// Renders the given render queue for the given perspective.
	virtual void Render(const PipelinePerspective &perspective, PipelineQueueID queueID, const RenderContext &context) const = 0;
	/// Renders the given single object for the given perspective.
	virtual void Render(uint4 objectID, const PipelinePerspective &perspective, PipelineQueueID queueID, const RenderContext &context) const = 0;
	/// Finalizes rendering from the collected render queues for the given perspective.
	BE_SCENE_API virtual void PostRender(const PipelinePerspective &perspective, const RenderContext &context) const { }
	/// Releases temporary rendering resources.
	BE_SCENE_API virtual void ReleaseIntermediate(PipelinePerspective &perspective) const { }
};

} // namespace

#endif