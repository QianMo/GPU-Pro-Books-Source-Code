/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#include "beSceneInternal/stdafx.h"
#include "beScene/bePipelinePerspective.h"
#include "beScene/beRenderable.h"
#include "beScene/bePipe.h"
#include "beScene/bePipelineProcessor.h"

#include <beMath/beVector.h>
#include <beMath/beMatrix.h>

#include <lean/functional/algorithm.h>

namespace beScene
{

// Constructor.
PipelinePerspective::PipelinePerspective(Pipe *pPipe, PipelineProcessor *pProcessor, PipelineStageMask stageMask)
	: m_activeOrderedQueues( 0 ),
	m_stageMask( stageMask ),
	m_pPipe( pPipe ),
	m_pProcessor( pProcessor )
{
}

// Destructor.
PipelinePerspective::~PipelinePerspective()
{
}

// Sets the perspective description.
void PipelinePerspective::Reset(const PerspectiveDesc &desc, Pipe *pPipe, PipelineProcessor *pProcessor, PipelineStageMask stageMask)
{
	Reset(pPipe, pProcessor, stageMask);
	SetDesc(desc);
}

// Sets the perspective description.
void PipelinePerspective::Reset(Pipe *pPipe, PipelineProcessor *pProcessor, PipelineStageMask stageMask)
{
	// Free any perspective data
	Reset();

	m_pPipe = pPipe;
	m_pProcessor = pProcessor;
	m_stageMask = stageMask;
}

// Resets this perspective after all temporary data has been discarded.
void PipelinePerspective::ResetReleased()
{
	m_pPipe = nullptr;
	m_pProcessor = nullptr;

	Perspective::ResetReleased();
}

// Releases resource references held.
void PipelinePerspective::ReleaseIntermediate()
{
	ClearRenderJobs();
	if (m_pPipe)
		m_pPipe->Release();
	Perspective::ReleaseIntermediate();

	// Transitively release all children
	for (perspectives_t::iterator it = m_children.begin(); it != m_children.end(); ++it)
		(*it)->ReleaseIntermediate();
	m_children.clear();
}

// Sets the pipe.
void PipelinePerspective::SetPipe(Pipe *pPipe)
{
	m_pPipe = pPipe;
}

// Sets the processor.
void PipelinePerspective::SetProcessor(PipelineProcessor *pProcessor)
{
	m_pProcessor = pProcessor;
}

// Sets the stage mask.
void PipelinePerspective::SetStageMask(PipelineStageMask stageMask)
{
	m_stageMask = stageMask;
}

// Adds the given perspective.
void PipelinePerspective::AddPerspective(PipelinePerspective *perspective)
{
	LEAN_ASSERT_NOT_NULL(perspective);
	m_children.push_back(perspective);
}

// Gets an ordered queue that allows for the addition of render jobs.
PipelinePerspective::QueueHandle PipelinePerspective::QueueRenderJobs(PipelineQueueID queueID)
{
	queues_t::iterator queuesEnd = m_orderedQueues.begin() + m_activeOrderedQueues;
	queues_t::iterator queue = lean::find_sorted(m_orderedQueues.begin(), queuesEnd, queueID, PipelineQueueID::Less());

	if (queue == queuesEnd)
	{
		// Reuse old queue
		if (m_activeOrderedQueues < m_orderedQueues.size())
		{
			queue->ID = queueID;
			queue->jobs.clear();
			queue = lean::insert_last(m_orderedQueues.begin(), queue, PipelineQueueID::Less(), lean::containers::trivial_construction_t());
		}
		// Insert new queue
		else
			queue = lean::push_sorted(m_orderedQueues, OrderedQueue(queueID), PipelineQueueID::Less());

		++m_activeOrderedQueues;
	}

	return QueueHandle(queue);
}

// Adds a render job to an ordered queue.
void PipelinePerspective::AddRenderJob(QueueHandle queue, const OrderedRenderJob &renderJob)
{
	LEAN_ASSERT_NOT_NULL(renderJob.Renderable);
	queue.value->jobs.push_back(renderJob);
}

// Clears all pipeline stages and render queues.
void PipelinePerspective::ClearRenderJobs()
{
	m_state.Reset();

	m_activeOrderedQueues = 0;
	for (queues_t::iterator it = m_orderedQueues.begin(); it != m_orderedQueues.end(); ++it)
		it->jobs.clear();
}

namespace
{

const PipelinePerspective::OrderedQueue* GetQueue(PipelineQueueID queueID, const PipelinePerspective::queues_t &queues, uint4 activeQueues)
{
	PipelinePerspective::queues_t::const_iterator queuesEnd = queues.begin() + activeQueues;
	PipelinePerspective::queues_t::const_iterator it = lean::find_sorted(queues.begin(), queuesEnd, queueID, PipelineQueueID::Less());

	return (it != queuesEnd)
		? &*it
		: nullptr;
}

} // namespace

// Prepares the given ordered queue.
void PipelinePerspective::Prepare(PipelineQueueID queueID)
{
	OrderedQueue *queue = const_cast<OrderedQueue*>( GetQueue(queueID, m_orderedQueues, m_activeOrderedQueues) );

	if (queue)
		std::sort(queue->jobs.begin(), queue->jobs.end());
}

// Renders the given ordered queue.
void PipelinePerspective::Render(PipelineQueueID queueID, const RenderContext &context) const
{
	const OrderedQueue *queue = GetQueue(queueID, m_orderedQueues, m_activeOrderedQueues);

	if (queue)
		for (OrderedQueue::jobs_t::const_pointer it = queue->jobs.begin(); it != queue->jobs.end(); ++it)
		{
			const OrderedRenderJob &job = *it;
			job.Renderable->Render(job.ID, *this, queueID, context);
		}
}

// Processes the given queue.
void PipelinePerspective::Process(PipelineQueueID queueID, const RenderContext &context) const
{
	if (m_pProcessor)
		m_pProcessor->Render(queueID.StageID, queueID.QueueID, this, context);
}

} // namespace
