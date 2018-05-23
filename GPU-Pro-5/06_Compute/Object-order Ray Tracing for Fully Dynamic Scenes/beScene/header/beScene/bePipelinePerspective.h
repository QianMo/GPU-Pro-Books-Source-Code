/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#pragma once
#ifndef BE_SCENE_PIPELINE_PERSPECTIVE
#define BE_SCENE_PIPELINE_PERSPECTIVE

#include "beScene.h"
#include "bePerspective.h"
#include "beRenderingLimits.h"
#include <vector>
#include <lean/containers/simple_vector.h>
#include "beRenderable.h"
#include <lean/smart/resource_ptr.h>
#include <beCore/beMany.h>
#include <lean/tags/handle.h>

namespace beScene
{
	
// Prototypes
class Pipe;
class RenderContext;
class PipelineProcessor;
class Renderable;

struct OrderedRenderJob
{
	const Renderable *Renderable;
	uint4 ID;
	uint4 SortIndex;

	OrderedRenderJob() { }
	OrderedRenderJob(const class Renderable *renderable, uint4 id, uint4 sortIndex)
		: Renderable(renderable),
		ID(id),
		SortIndex(sortIndex) { }

	friend bool operator <(OrderedRenderJob left, OrderedRenderJob right)
	{
		return (left.SortIndex < right.SortIndex);
	}
};

/// Rendering perspective.
class PipelinePerspective : public Perspective
{
public:
	/// Ordered render queue
	struct OrderedQueue
	{
		PipelineQueueID ID;

		typedef lean::simple_vector<OrderedRenderJob, lean::simple_vector_policies::pod> jobs_t;
		jobs_t jobs;

		OrderedQueue(PipelineQueueID id)
			: ID(id) { }

		friend PipelineQueueID GetID(const OrderedQueue &queue) { return queue.ID; }
	};

	typedef lean::simple_vector<lean::com_ptr<PipelinePerspective>, lean::containers::vector_policies::semipod> perspectives_t;
	typedef lean::simple_vector<OrderedQueue, lean::containers::vector_policies::semipod> queues_t;

private:
	PipelineState m_state;

	perspectives_t m_children;

	queues_t m_orderedQueues;
	uint4 m_activeOrderedQueues;

	uint4 m_stageMask;
	lean::resource_ptr<Pipe> m_pPipe;
	lean::resource_ptr<PipelineProcessor> m_pProcessor;

	/// Resets this perspective after all temporary data has been discarded.
	BE_SCENE_API virtual void ResetReleased() LEAN_OVERRIDE;

public:
	/// Constructor.
	BE_SCENE_API PipelinePerspective(Pipe *pPipe, PipelineProcessor *pProcessor, PipelineStageMask stageMask);
	/// Destructor.
	BE_SCENE_API ~PipelinePerspective();

	using Perspective::Reset;
	/// Sets the perspective description and resets all contents.
	BE_SCENE_API void Reset(const PerspectiveDesc &desc, Pipe *pPipe, PipelineProcessor *pProcessor, PipelineStageMask stageMask);
	/// Sets the perspective description and resets all contents.
	BE_SCENE_API void Reset(Pipe *pPipe, PipelineProcessor *pProcessor, PipelineStageMask stageMask);
	/// Releases shared references held.
	BE_SCENE_API void ReleaseIntermediate() LEAN_OVERRIDE;

	/// Sets the pipe.
	BE_SCENE_API void SetPipe(Pipe *pPipe);
	/// Sets the processor.
	BE_SCENE_API void SetProcessor(PipelineProcessor *pProcessor);
	/// Sets the stage mask.
	BE_SCENE_API void SetStageMask(PipelineStageMask stageMask);

	typedef beCore::Range<PipelinePerspective*const*> PerspectiveRange;

	/// Adds the given perspective.
	BE_SCENE_API void AddPerspective(PipelinePerspective *perspective);
	/// Gets the perspectives.
	LEAN_INLINE PerspectiveRange GetPerspectives() const { return beCore::MakeRangeN(&m_children[0].get(), m_children.size()); } 

	/// Queue of render jobs.
	typedef lean::handle<OrderedQueue*, PipelinePerspective> QueueHandle;
	/// Gets an ordered queue that allows for the addition of render jobs.
	BE_SCENE_API QueueHandle QueueRenderJobs(PipelineQueueID queueID);
	/// Adds a render job to an ordered queue.
	BE_SCENE_API void AddRenderJob(QueueHandle queue, const OrderedRenderJob &renderJob);
	// Clears all pipeline stages and render queues.
	BE_SCENE_API void ClearRenderJobs();

	/// Prepares the given ordered queue.
	BE_SCENE_API void Prepare(PipelineQueueID queueID);
	/// Renders the given ordered queue.
	BE_SCENE_API void Render(PipelineQueueID queueID, const RenderContext &context) const;
	/// Processes the given queue.
	BE_SCENE_API void Process(PipelineQueueID queueID, const RenderContext &context) const;

	/// Optionally gets a pipe.
	Pipe* GetPipe() const LEAN_OVERRIDE { return m_pPipe; }

	/// Gets a stage mask.
	LEAN_INLINE PipelineStageMask GetStageMask() const { return m_stageMask; }
	/// Gets the pipeline state.
	LEAN_INLINE PipelineState& GetPipelineState() { return m_state; }
	/// Gets the pipeline state.
	LEAN_INLINE const PipelineState& GetPipelineState() const { return m_state; }
};

} // namespace

#endif