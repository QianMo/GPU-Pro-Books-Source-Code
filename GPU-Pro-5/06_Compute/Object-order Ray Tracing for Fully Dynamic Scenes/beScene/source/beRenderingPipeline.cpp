/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#include "beSceneInternal/stdafx.h"
#include "beScene/beRenderingPipeline.h"
#include "beScene/bePipelinePerspective.h"
#include <lean/functional/algorithm.h>
#include <lean/logging/errors.h>

namespace beScene
{

/// Rendering Pipeline.
class RenderingPipeline::Impl
{
public:
	/// Pipeline stage.
	struct Stage
	{
		utf8_string name;		///< Name.
		PipelineStageDesc desc;	///< Description.

		/// Constructor.
		Stage(const utf8_ntri &name,
			const PipelineStageDesc &desc)
				: name( name.to<utf8_string>() ),
				desc( desc ) { }
	};
	typedef std::vector<Stage> stage_vector;
	stage_vector stages;

	typedef std::vector<uint2> id_vector;
	id_vector sortedStageIDs;
	id_vector sortedStageSlots;

	PipelineStageMask normalStageMask;

	/// Render queue.
	struct Queue
	{
		utf8_string name;		///< Name.
		RenderQueueDesc desc;	///< Description.

		/// Constructor.
		Queue(const utf8_ntri &name,
			const RenderQueueDesc &desc)
				: name( name.to<utf8_string>() ),
				desc( desc ) { }
	};
	typedef std::vector<Queue> queue_vector;
	queue_vector queues;

	id_vector sortedQueueIDs;
	id_vector sortedQueueSlots;

	uint2 defaultStageID;
	uint2 defaultQueueIDs[MaxPipelineStageCount];

	/// Constructor.
	Impl()
		: normalStageMask(0),
		defaultStageID(InvalidPipelineStage),
		defaultQueueIDs() { }
};

// Constructor.
RenderingPipeline::RenderingPipeline(const utf8_ntri &name)
	: m_name(name.to<utf8_string>()),
	m_impl(new Impl())
{
}

// Destructor.
RenderingPipeline::~RenderingPipeline()
{
}

// Prepares rendering of all stages and queues.
void RenderingPipeline::Prepare(PipelinePerspective &perspective, const Renderable *const *renderables, uint4 renderableCount,
								PipelineStageMask overrideStageMask, bool bNoChildren) const
{
	PipelineState &state = perspective.GetPipelineState();

	for (uint4 i = 0; i < renderableCount; ++i)
		renderables[i]->Cull(perspective);

	uint4 stageMask = (overrideStageMask) ? overrideStageMask : perspective.GetStageMask();
	if (stageMask & NormalPipelineStagesOnly)
		stageMask &= m_impl->normalStageMask;

	for (Impl::id_vector::const_iterator stageIt = m_impl->sortedStageIDs.begin();
		stageIt != m_impl->sortedStageIDs.end(); ++stageIt)
	{
		if (ComputeStageMask(*stageIt) & stageMask)
			for (Impl::id_vector::const_iterator queueIt = m_impl->sortedQueueIDs.begin();
				queueIt != m_impl->sortedQueueIDs.end(); ++queueIt)
			{
				PipelineQueueID queueID(*stageIt, *queueIt);
				bool bQueueActive = false;

				for (uint4 i = 0; i < renderableCount; ++i)
					bQueueActive |= renderables[i]->Prepare(perspective, queueID,
						m_impl->stages[queueID.StageID].desc, m_impl->queues[queueID.QueueID].desc);
			
				if (bQueueActive)
					state.ActiveQueues[queueID.StageID] |= ComputeQueueMask(queueID.QueueID);
			}
	}

	for (uint4 i = 0; i < renderableCount; ++i)
		renderables[i]->Collect(perspective);

	if (!bNoChildren)
		for (PipelinePerspective::PerspectiveRange perspectives = perspective.GetPerspectives(); perspectives; ++perspectives)
			Prepare(**perspectives, renderables, renderableCount);
}

// Optimizes the given stage and queue.
void RenderingPipeline::Optimize(PipelinePerspective &perspective, const Renderable *const *renderables, uint4 renderableCount,
								 PipelineStageMask overrideStageMask, bool bNoChildren) const
{
	PipelinePerspective::PerspectiveRange perspectives = perspective.GetPerspectives();
	
	if (!bNoChildren)
		for (uint4 i = Size(perspectives); i-- > 0; )
			Optimize(*perspectives.Begin[i], renderables, renderableCount);

	uint4 stageMask = (overrideStageMask) ? overrideStageMask : perspective.GetStageMask();
	if (stageMask & NormalPipelineStagesOnly)
		stageMask &= m_impl->normalStageMask;

	for (Impl::id_vector::const_iterator stageIt = m_impl->sortedStageIDs.begin();
		stageIt != m_impl->sortedStageIDs.end(); ++stageIt)
	{
		if (ComputeStageMask(*stageIt) & stageMask)
			for (Impl::id_vector::const_iterator queueIt = m_impl->sortedQueueIDs.begin();
				queueIt != m_impl->sortedQueueIDs.end(); ++queueIt)
			{
				PipelineQueueID queueID(*stageIt, *queueIt);

				for (uint4 i = 0; i < renderableCount; ++i)
					renderables[i]->Optimize(perspective, queueID);

				perspective.Prepare(queueID);
			}
	}
}

// Renders child perspectives & prepares rendering for the given perspective.
void RenderingPipeline::PreRender(const PipelinePerspective &perspective, const Renderable *const *renderables, uint4 renderableCount,
								  const RenderContext &context, bool bNoChildren) const
{
	PipelinePerspective::PerspectiveRange perspectives = perspective.GetPerspectives();

	if (!bNoChildren)
		for (uint4 i = Size(perspectives); i-- > 0; )
			Render(*perspectives.Begin[i], renderables, renderableCount, context, 0, false, false);

	for (uint4 i = 0; i < renderableCount; ++i)
		renderables[i]->PreRender(perspective, context);
}

/// Renders the given stage and queue.
void RenderingPipeline::RenderStages(const PipelinePerspective &perspective, const Renderable *const *renderables, uint4 renderableCount,
									 const RenderContext &context, PipelineStageMask overrideStageMask) const
{
	uint4 stageMask = (overrideStageMask) ? overrideStageMask : perspective.GetStageMask();
	if (stageMask & NormalPipelineStagesOnly)
		stageMask &= m_impl->normalStageMask;

	const PipelineState &state = perspective.GetPipelineState();

	for (Impl::id_vector::const_iterator stageIt = m_impl->sortedStageIDs.begin();
		stageIt != m_impl->sortedStageIDs.end(); ++stageIt)
	{
		if (ComputeStageMask(*stageIt) & stageMask)
		{
			const PipelineStageDesc &stageDesc = m_impl->stages[*stageIt].desc;

			// Set stage
			if (stageDesc.Setup && (!stageDesc.Conditional || state.ActiveQueues[*stageIt]))
				stageDesc.Setup->SetupRendering(*stageIt, InvalidRenderQueue, perspective, context);

			// Render all queues
			for (Impl::id_vector::const_iterator queueIt = m_impl->sortedQueueIDs.begin();
				queueIt != m_impl->sortedQueueIDs.end(); ++queueIt)
			{
				PipelineQueueID queueID(*stageIt, *queueIt);

				const RenderQueueDesc &queueDesc = m_impl->queues[queueID.QueueID].desc;

				// Set up individual queues
				if (queueDesc.Setup && (!stageDesc.Conditional || state.ActiveQueues[queueID.StageID] & ComputeQueueMask(queueID.QueueID)))
					queueDesc.Setup->SetupRendering(queueID.StageID, queueID.QueueID, perspective, context);

				for (uint4 i = 0; i < renderableCount; ++i)
					renderables[i]->Render(perspective, queueID, context);

				perspective.Render(queueID, context);

				// Individual queue processing
				perspective.Process(queueID, context);
			}

			// Stage processing
			perspective.Process(PipelineQueueID(*stageIt, InvalidRenderQueue), context);
		}
	}
}

// Finalizes rendering for the given perspective.
void RenderingPipeline::PostRender(const PipelinePerspective &perspective, const Renderable *const *renderables, uint4 renderableCount,
								   const RenderContext &context, bool bFinalProcessing) const
{
	// Global post-processing
	if (bFinalProcessing)
		perspective.Process(PipelineQueueID(InvalidPipelineStage, InvalidRenderQueue), context);

	for (uint4 i = 0; i < renderableCount; ++i)
		renderables[i]->PostRender(perspective, context);
}

// Renders the given stage and queue.
void RenderingPipeline::Render(const PipelinePerspective &perspective, const Renderable *const *renderables, uint4 renderableCount,
							   const RenderContext &context, PipelineStageMask overrideStageMask, bool bNoChildren, bool bFinalProcessing) const
{
	PreRender(perspective, renderables, renderableCount, context, bNoChildren);
	RenderStages(perspective, renderables, renderableCount, context, overrideStageMask);
	PostRender(perspective, renderables, renderableCount, context, bFinalProcessing);
}

// Releases temporary rendering resources.
void RenderingPipeline::ReleaseIntermediate(PipelinePerspective &perspective, const Renderable *const *renderables, uint4 renderableCount,
											bool bReleasePerspective) const
{
	PipelinePerspective::PerspectiveRange perspectives = perspective.GetPerspectives();

	for (uint4 i = Size(perspectives); i-- > 0; )
		// NOTE: Don't release child perspectives just yet
		ReleaseIntermediate(*perspectives.Begin[i], renderables, renderableCount, false);

	for (uint4 i = 0; i < renderableCount; ++i)
		renderables[i]->ReleaseIntermediate(perspective);

	if (bReleasePerspective)
		// NOTE: Releases child perspectives recursively
		perspective.ReleaseIntermediate();
}

namespace
{

/// Sorts structed elements by their names.
template <class Type>
struct NameAttributeCompare
{
	utf8_ntr name;

	NameAttributeCompare(const utf8_ntr &name)
		: name(name) { }

	LEAN_INLINE bool operator ()(const Type &elem) { return (elem.name == name); }
};

/// Sorts IDs to structed elements by their description layer attributes.
template <class Element>
struct IDByLayerCompare
{
	const Element *elements;

	IDByLayerCompare(const Element *elements)
		: elements(elements) { }

	LEAN_INLINE bool operator ()(uint2 left, uint2 right) { return (elements[left].desc.Layer < elements[right].desc.Layer); }
};

} // namespace

// Adds a pipeline stage according to the given description.
uint2 RenderingPipeline::AddStage(const utf8_ntri &stageName, const PipelineStageDesc &desc)	
{
	Impl::stage_vector &stages = m_impl->stages;
	Impl::stage_vector::iterator it = std::find_if(
		stages.begin(), stages.end(),
		NameAttributeCompare<Impl::Stage>(make_ntr(stageName)) );

	// Enforce unique names
	if (it == stages.end())
	{
		if (it == stages.end())
			it = stages.insert(stages.end(), Impl::Stage(stageName, desc));
		else
			*it = Impl::Stage(stageName, desc);

		// Insert stage into ordered rendering sequence
		uint2 stageID = static_cast<uint2>(it - stages.begin());
		Impl::id_vector &stageIDs = m_impl->sortedStageIDs;
		lean::push_sorted(stageIDs, 
			stageID,
			IDByLayerCompare<Impl::Stage>(&stages[0]) );

		if (desc.Normal)
			m_impl->normalStageMask |= ComputeStageMask(stageID);

		// Rebuild stage slot table
		Impl::id_vector &stageSlots = m_impl->sortedStageSlots;
		stageSlots.resize(stages.size(), static_cast<uint2>(-1));
		for (Impl::id_vector::const_iterator it = stageIDs.begin(); it != stageIDs.end(); ++it)
			stageSlots[*it] = static_cast<uint2>(it - stageIDs.begin());
	}
	// Warn about mismatching adds
	else if(memcmp(&it->desc, &desc, sizeof(desc)) != 0)
		LEAN_LOG_ERROR_CTX("Stage added twice, descriptions mismatching", stageName.c_str());

	return static_cast<uint2>(it - stages.begin());
}

// Adds a render queue according to the given description.
uint2 RenderingPipeline::AddQueue(const utf8_ntri &queueName, const RenderQueueDesc &desc)
{
	Impl::queue_vector &queues = m_impl->queues;
	Impl::queue_vector::iterator it = std::find_if(
		queues.begin(), queues.end(), 
		NameAttributeCompare<Impl::Queue>(make_ntr(queueName)) );

	// Enforce unique names
	if (it == queues.end())
	{
		if (it == queues.end())
			it = queues.insert(queues.end(), Impl::Queue(queueName, desc));
		else
			*it = Impl::Queue(queueName, desc);

		// Insert queue into ordered rendering sequence
		uint2 queueID = static_cast<uint2>(it - queues.begin());
		Impl::id_vector &queueIDs = m_impl->sortedQueueIDs;
		lean::push_sorted(m_impl->sortedQueueIDs, 
			queueID,
			IDByLayerCompare<Impl::Queue>(&queues[0]) );

		// Rebuild queue slot table
		Impl::id_vector &queueSlots = m_impl->sortedQueueSlots;
		queueSlots.resize(queues.size(), static_cast<uint2>(-1));
		for (Impl::id_vector::const_iterator it = queueIDs.begin(); it != queueIDs.end(); ++it)
			queueSlots[*it] = static_cast<uint2>(it - queueIDs.begin());
	}
	// Warn about mismatching adds
	else if(memcmp(&it->desc, &desc, sizeof(desc)) != 0)
		LEAN_LOG_ERROR_CTX("Queue added twice, descriptions mismatching", queueName.c_str());

	return static_cast<uint2>(it - queues.begin());
}

// Sets a new setup for the given pipeline stage.
void RenderingPipeline::SetStageSetup(uint2 stageID, const QueueSetup *pSetup)
{
	LEAN_ASSERT(stageID < m_impl->stages.size());

	m_impl->stages[stageID].desc.Setup = pSetup;
}

// Sets a new setup for the given render queue.
void RenderingPipeline::SetQueueSetup(uint2 queueID, const QueueSetup *pSetup)
{
	LEAN_ASSERT(queueID < m_impl->queues.size());

	m_impl->queues[queueID].desc.Setup = pSetup;
}

// Gets the ID of the pipeline stage identified by the given name.
uint2 RenderingPipeline::GetStageID(const utf8_ntri &stageName) const
{
	Impl::stage_vector::const_iterator it = std::find_if(
		m_impl->stages.begin(), m_impl->stages.end(),
		NameAttributeCompare<Impl::Stage>(make_ntr(stageName)) );

	return (it != m_impl->stages.end())
		? static_cast<uint2>(it - m_impl->stages.begin())
		: InvalidPipelineStage;
}

// Gets the ID of the render queue identified by the given name.
uint2 RenderingPipeline::GetQueueID(const utf8_ntri &queueName) const
{
	Impl::queue_vector::const_iterator it = std::find_if(
		m_impl->queues.begin(), m_impl->queues.end(), 
		NameAttributeCompare<Impl::Queue>(make_ntr(queueName)) );

	return (it != m_impl->queues.end())
		? static_cast<uint2>(it - m_impl->queues.begin())
		: InvalidRenderQueue;
}

// Gets the number of pipeline stages.
uint2 RenderingPipeline::GetStageCount() const
{
	return static_cast<uint2>(m_impl->stages.size());
}

// Gets the number of render queues.
uint2 RenderingPipeline::GetQueueCount() const
{
	return static_cast<uint2>(m_impl->queues.size());
}

// Gets the description of the given pipeline stage.
const PipelineStageDesc& RenderingPipeline::GetStageDesc(uint2 stageID) const
{
	LEAN_ASSERT(stageID < m_impl->stages.size());
	return m_impl->stages[stageID].desc;
}

// Gets the description of the given pipeline stage.
const RenderQueueDesc& RenderingPipeline::GetQueueDesc(uint2 queueID) const
{
	LEAN_ASSERT(queueID < m_impl->queues.size());
	return m_impl->queues[queueID].desc;
}

// Gets an ordered sequence of pipeline stage IDs.
const uint2* RenderingPipeline::GetOrderedStageIDs() const
{
	return &m_impl->sortedStageIDs[0];
}

// Gets a pipeline stage slot lookup table.
const uint2* RenderingPipeline::GetStageSlots() const
{
	return &m_impl->sortedStageSlots[0];
}

// Gets an ordered sequence of render queue IDs.
const uint2* RenderingPipeline::GetOrderedQueueIDs() const
{
	return &m_impl->sortedQueueIDs[0];
}

// Gets a render queue slot lookup table.
const uint2* RenderingPipeline::GetQueueSlots() const
{
	return &m_impl->sortedQueueSlots[0];
}

// Gets the ID of the default pipeline stage.
void RenderingPipeline::SetDefaultStageID(uint2 stageID)
{
	if (stageID < m_impl->stages.size())
		m_impl->defaultStageID = stageID;
}

// Gets the ID of the default pipeline stage.
uint2 RenderingPipeline::GetDefaultStageID() const
{
	return m_impl->defaultStageID;
}

// Gets the ID of the default pipeline stage.
void RenderingPipeline::SetDefaultQueueID(uint2 stageID, uint2 queueID)
{
	if (stageID < m_impl->stages.size() && queueID < m_impl->queues.size())
		m_impl->defaultQueueIDs[stageID] = queueID;
}

// Gets the ID of the default pipeline stage.
uint2 RenderingPipeline::GetDefaultQueueID(uint2 stageID) const
{
	return (stageID < m_impl->stages.size())
		? m_impl->defaultQueueIDs[stageID]
		: InvalidRenderQueue;
}

// Gets the mask of normal pipeline stages.
PipelineStageMask RenderingPipeline::GetNormalStages() const
{
	return m_impl->normalStageMask;
}

// Sets the name.
void RenderingPipeline::SetName(const utf8_ntri &name)
{
	m_name.assign(name.begin(), name.end());
}

} // namespace
