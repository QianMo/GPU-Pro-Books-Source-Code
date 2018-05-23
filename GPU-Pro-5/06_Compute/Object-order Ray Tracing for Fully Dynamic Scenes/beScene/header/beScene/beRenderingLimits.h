/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#pragma once
#ifndef BE_SCENE_RENDERINGLIMITS
#define BE_SCENE_RENDERINGLIMITS

#include "beScene.h"
#include <bitset>

#ifndef BE_SCENE_MAX_RENDER_QUEUE_COUNT

	/// Maximum number of pipeline stages any renderable may be added to.
	/// @ingroup GlobalSwitches
	/// @ingroup SceneLibrary
	#define BE_SCENE_MAX_PIPELINE_STAGE_COUNT 32

	/// Maximum number of render queues any renderable may be added to.
	/// @ingroup GlobalSwitches
	/// @ingroup SceneLibrary
	#define BE_SCENE_MAX_RENDER_QUEUE_COUNT 32

#endif

namespace beScene
{

/// Maximum number of pipeline stages any renderable may be added to.
const uint2 MaxPipelineStageCount = BE_SCENE_MAX_PIPELINE_STAGE_COUNT;
/// Maximum number of pipeline stages any renderable may be added to.
const uint2 MaxRenderQueueCount = BE_SCENE_MAX_RENDER_QUEUE_COUNT;

/// Stage ID type.
typedef uint2 PipelineStageID;
/// Queue ID type.
typedef uint2 RenderQueueID;

struct PipelineQueueID
{
	PipelineStageID StageID;
	RenderQueueID QueueID;
	
	PipelineQueueID(PipelineStageID stageID, RenderQueueID queueID)
		: StageID(stageID),
		QueueID(queueID) { }

	friend bool operator <(PipelineQueueID left, PipelineQueueID right)
	{
		return (left.StageID < right.StageID) || (left.StageID == right.StageID && left.QueueID < right.QueueID);
	}
	friend bool operator ==(PipelineQueueID left, PipelineQueueID right)
	{
		return (left.StageID == right.StageID) && (left.QueueID == right.QueueID);
	}

	friend PipelineQueueID GetID(PipelineQueueID id) { return id; }

	struct Less
	{
		template <class L, class R>
		bool operator ()(const L &left, const R &right) { return GetID(left) < GetID(right); }
	};
	struct Equal
	{
		template <class L, class R>
		bool operator ()(const L &left, const R &right) { return GetID(left) == GetID(right); }
	};
};

/// Invalid stage ID.
const PipelineStageID InvalidPipelineStage = static_cast<PipelineStageID>(-1);
/// Invalid queue ID.
const RenderQueueID InvalidRenderQueue = static_cast<RenderQueueID>(-1);

/// Stage mask type.
typedef uint4 PipelineStageMask;
/// Queue mask type.
typedef uint4 RenderQueueMask;

LEAN_STATIC_ASSERT(lean::size_info<PipelineStageMask>::bits >= MaxPipelineStageCount);
LEAN_STATIC_ASSERT(lean::size_info<RenderQueueMask>::bits >= MaxRenderQueueCount);

/// All normal stages.
const PipelineStageMask NormalPipelineStages = -1;
/// All normal queues.
const RenderQueueMask NormalRenderQueues = -1;

/// Only normal stages.
const PipelineStageMask NormalPipelineStagesOnly = 1 << (lean::size_info<PipelineStageMask>::bits - 1);
/// Only normal queues.
const RenderQueueMask NormalRenderQueuesOnly = 1 << (lean::size_info<RenderQueueMask>::bits - 1);

/// All stages.
const PipelineStageMask AllPipelineStages = ~NormalPipelineStagesOnly;
/// All queues.
const RenderQueueMask AllRenderQueues = ~NormalRenderQueuesOnly;

/// Gets the stage mask from the given stage.
LEAN_INLINE PipelineStageMask ComputeStageMask(PipelineStageID stageID)
{
	return 1U << stageID;
}

/// Gets the queue mask from the given stage.
LEAN_INLINE RenderQueueMask ComputeQueueMask(RenderQueueID queueID)
{
	return 1U << queueID;
}

struct PipelineState
{
	RenderQueueMask ActiveQueues[MaxPipelineStageCount];

	PipelineState()
		: ActiveQueues() { }

	void Reset()
	{
		memset(ActiveQueues, 0, sizeof(ActiveQueues));
	}
};

} // namespace

#endif