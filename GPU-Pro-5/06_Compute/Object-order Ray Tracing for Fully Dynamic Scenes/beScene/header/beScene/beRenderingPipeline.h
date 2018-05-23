/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#pragma once
#ifndef BE_SCENE_RENDERINGPIPELINE
#define BE_SCENE_RENDERINGPIPELINE

#include "beScene.h"
#include <beCore/beShared.h>
#include <lean/pimpl/pimpl_ptr.h>
#include "beRenderingLimits.h"
#include "bePipelinePerspective.h"
#include "beQueueSetup.h"
#include <lean/smart/resource_ptr.h>

namespace beScene
{
	
// Prototypes
class Pipe;
class RenderContext;
class PipelineProcessor;

/// Pipeline stage description.
struct PipelineStageDesc
{
	int4 Layer;										///< Layer index.
	bool Normal;									///< Part of normal rendering.
	bool Conditional;								///< Only setup if active.
	lean::resource_ptr<const QueueSetup> Setup;		///< Queue setup handler.

	/// Constructor.
	LEAN_INLINE explicit PipelineStageDesc(
		int4 layer,
		bool bNormal = true,
		const QueueSetup *pSetup = nullptr,
		bool bConditional = false)
			: Layer(layer),
			Normal(bNormal),
			Conditional(bConditional),
			Setup(pSetup)  { }
};

/// Render queue description.
struct RenderQueueDesc
{
	int4 Layer;										///< Layer index.
	bool DepthSort;									///< Depth sort flag.
	bool Backwards;									///< Backwards sorting flag.
	bool Conditional;								///< Only setup if active.
	lean::resource_ptr<const QueueSetup> Setup;		///< Queue setup handler.

	/// Constructor.
	LEAN_INLINE explicit RenderQueueDesc(
		int4 layer,
		bool bDepthSort = false,
		bool bBackwards = false,
		const QueueSetup *pSetup = nullptr,
		bool bConditional = true)
			: Layer(layer),
			DepthSort(bDepthSort),
			Backwards(bBackwards),
			Conditional(bConditional),
			Setup(pSetup) { }
};

/// Rendering Pipeline.
class RenderingPipeline : public beCore::Resource
{
private:
	utf8_string m_name;

	class Impl;
	lean::pimpl_ptr<Impl> m_impl;

public:
	/// Constructor.
	BE_SCENE_API RenderingPipeline(const utf8_ntri &name);
	/// Destructor.
	BE_SCENE_API ~RenderingPipeline();

	/// Prepares rendering of all stages and queues.
	BE_SCENE_API void Prepare(PipelinePerspective &perspective, const Renderable *const *renderables, uint4 renderableCount,
		PipelineStageMask overrideStageMask = 0, bool bNoChildren = false) const;
	/// Optimizes the given stage and queue.
	BE_SCENE_API void Optimize(PipelinePerspective &perspective, const Renderable *const *renderables, uint4 renderableCount,
		PipelineStageMask overrideStageMask = 0, bool bNoChildren = false) const;
	/// Renders child perspectives & prepares rendering for the given perspective.
	BE_SCENE_API void PreRender(const PipelinePerspective &perspective, const Renderable *const *renderables, uint4 renderableCount,
		const RenderContext &context, bool bNoChildren = false) const;
	/// Renders the given stage and queue.
	BE_SCENE_API void RenderStages(const PipelinePerspective &perspective, const Renderable *const *renderables, uint4 renderableCount,
		const RenderContext &context, PipelineStageMask overrideStageMask = 0) const;
	/// Finalizes rendering for the given perspective.
	BE_SCENE_API void PostRender(const PipelinePerspective &perspective, const Renderable *const *renderables, uint4 renderableCount,
		const RenderContext &context, bool bFinalProcessing = true) const;
	/// Renders the given stage and queue.
	BE_SCENE_API void Render(const PipelinePerspective &perspective, const Renderable *const *renderables, uint4 renderableCount,
		const RenderContext &context, PipelineStageMask overrideStageMask = 0, bool bNoChildren = false, bool bFinalProcessing = true) const;
	/// Releases temporary rendering resources.
	BE_SCENE_API void ReleaseIntermediate(PipelinePerspective &perspective, const Renderable *const *renderables, uint4 renderableCount,
		bool bReleasePerspective = true) const;

	/// Adds a pipeline stage according to the given description.
	BE_SCENE_API uint2 AddStage(const utf8_ntri &stageName, const PipelineStageDesc &desc);
	/// Adds a render queue according to the given description.
	BE_SCENE_API uint2 AddQueue(const utf8_ntri &queueName, const RenderQueueDesc &desc);

	/// Sets a new setup for the given pipeline stage.
	BE_SCENE_API void SetStageSetup(uint2 stageID, const QueueSetup *pSetup);
	/// Sets a new setup for the given render queue.
	BE_SCENE_API void SetQueueSetup(uint2 queueID, const QueueSetup *pSetup);

	/// Gets the ID of the pipeline stage identified by the given name.
	BE_SCENE_API uint2 GetStageID(const utf8_ntri &stageName) const;
	/// Gets the ID of the render queue identified by the given name.
	BE_SCENE_API uint2 GetQueueID(const utf8_ntri &queueName) const;

	/// Gets the number of pipeline stages.
	BE_SCENE_API uint2 GetStageCount() const;
	/// Gets the number of render queues.
	BE_SCENE_API uint2 GetQueueCount() const;

	/// Gets the description of the given pipeline stage.
	BE_SCENE_API const PipelineStageDesc& GetStageDesc(uint2 stageID) const;
	/// Gets the description of the given pipeline stage.
	BE_SCENE_API const RenderQueueDesc& GetQueueDesc(uint2 queueID) const;

	/// Gets an ordered sequence of pipeline stage IDs.
	BE_SCENE_API const uint2* GetOrderedStageIDs() const;
	/// Gets a pipeline stage slot lookup table.
	BE_SCENE_API const uint2* GetStageSlots() const;
	/// Gets an ordered sequence of render queue IDs.
	BE_SCENE_API const uint2* GetOrderedQueueIDs() const;
	/// Gets a render queue slot lookup table.
	BE_SCENE_API const uint2* GetQueueSlots() const;

	/// Gets the ID of the default pipeline stage.
	BE_SCENE_API void SetDefaultStageID(uint2 stageID);
	/// Gets the ID of the default pipeline stage.
	BE_SCENE_API uint2 GetDefaultStageID() const;
	/// Gets the ID of the default pipeline stage.
	BE_SCENE_API void SetDefaultQueueID(uint2 stageID, uint2 queueID);
	/// Gets the ID of the default pipeline stage.
	BE_SCENE_API uint2 GetDefaultQueueID(uint2 stageID) const;

	/// Gets the mask of normal pipeline stages.
	BE_SCENE_API PipelineStageMask GetNormalStages() const;

	/// Sets the name.
	BE_SCENE_API void SetName(const utf8_ntri &name);
	/// Gets the name.
	LEAN_INLINE const utf8_string& GetName() const { return m_name; }
};

} // nmaespace

#endif