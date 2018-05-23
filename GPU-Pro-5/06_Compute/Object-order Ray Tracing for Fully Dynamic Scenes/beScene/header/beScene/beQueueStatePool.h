/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#pragma once
#ifndef BE_SCENE_QUEUE_STATE_POOL
#define BE_SCENE_QUEUE_STATE_POOL

#include "beScene.h"
#include "beRenderingLimits.h"

#include <lean/containers/simple_vector.h>
#include <lean/functional/algorithm.h>

namespace beScene
{

/// Utility base class performing basic queue state management.
struct QueueStateBase
{
	typedef QueueStateBase base_type;
	PipelineQueueID ID;	///< Queue ID.

	/// Constructor.
	QueueStateBase()
		: ID(InvalidPipelineStage, InvalidRenderQueue) { }
	
	/// Resets this state.
	void Reset(PipelineQueueID newID)
	{
		this->ID = newID;
	}
	
	friend PipelineQueueID GetID(const QueueStateBase &queue) { return queue.ID; }
};

/// Pool perspective states.
template <class QueueState, class VectorPolicy>
class QueueStatePool
{
public:
	typedef lean::simple_vector<QueueState, VectorPolicy> pool_t;
	pool_t Pool;				///< Queue pool.
	uint4 ActiveCount;			///< Number of active queues in the pool.
	uint4 StructureRevision;	///< Versioning for synching.

	/// Constructor.
	QueueStatePool()
		: ActiveCount(0),
		StructureRevision(0) { }

	/// Gets a NEW queue object.
	QueueState& NewQueue(PipelineQueueID queueID)
	{
		typename pool_t::iterator itActiveQueuesEnd = this->Pool.begin() + this->ActiveCount;

		// Reuse old queue, add new if none available
		if (itActiveQueuesEnd == this->Pool.end())
		{
			this->Pool.push_back();
			itActiveQueuesEnd = this->Pool.end() - 1;
		}
		itActiveQueuesEnd->Reset(queueID);

		// Sort queues by ID
		itActiveQueuesEnd = lean::insert_last(this->Pool.begin(), itActiveQueuesEnd,
			PipelineQueueID::Less(), typename pool_t::construction_policy::move_tag());
		++this->ActiveCount;
		++this->StructureRevision;

		return *itActiveQueuesEnd;
	}

	/// Gets a queue from the given ID.
	QueueState& GetQueue(PipelineQueueID queueID)
	{
		typename pool_t::iterator itActiveQueuesEnd = this->Pool.begin() + this->ActiveCount;
		typename pool_t::iterator itQueue = lean::find_sorted(this->Pool.begin(), itActiveQueuesEnd, queueID, PipelineQueueID::Less());

		// Insert queue, if not yet existent
		return (itQueue != itActiveQueuesEnd)
			? *itQueue
			: NewQueue(queueID);
	}

	/// Gets an existing queue from the given ID.
	const QueueState* GetExistingQueue(PipelineQueueID queueID) const
	{
		typename pool_t::const_iterator itActiveQueuesEnd = this->Pool.begin() + this->ActiveCount;
		typename pool_t::const_iterator itQueue = lean::find_sorted(this->Pool.begin(), itActiveQueuesEnd, queueID, PipelineQueueID::Less());
		return (itQueue != itActiveQueuesEnd) ? &*itQueue : nullptr;
	}
	/// Gets an existing queue from the given ID.
	QueueState* GetExistingQueue(const PipelineQueueID queueID)
	{
		return const_cast<QueueState*>(
				const_cast<const QueueStatePool*>(this)->GetExistingQueue(queueID)
			);
	}

	/// Gets an existing queue from the given parallel queue.
	template <class OtherQueueState, class P>
	const QueueState& GetParallelQueue(const QueueStatePool<OtherQueueState, P> &src, const OtherQueueState *srcState) const
	{
		LEAN_ASSERT(this->StructureRevision == src.StructureRevision);
		LEAN_ASSERT(this->ActiveCount == src.ActiveCount);
		uint4 queueIdx = srcState - &src.Pool[0];
		LEAN_ASSERT(queueIdx < src.ActiveCount);
		const QueueState &state = this->Pool[queueIdx];
		LEAN_ASSERT(state.ID == srcState->ID);
		return state;
	}
	/// Gets an existing queue from the given parallel queue.
	template <class OtherQueueState, class P>
	QueueState& GetParallelQueue(const QueueStatePool<OtherQueueState, P> &src, const OtherQueueState *srcState)
	{
		return const_cast<QueueState&>(
				const_cast<const QueueStatePool*>(this)->GetParallelQueue(src, srcState)
			);
	}

	/// Synchronizes this queue with the given view.
	template <class OtherQueueState, class P>
	void CopyFrom(const QueueStatePool<OtherQueueState, P> &source)
	{
		if (this->StructureRevision != source.StructureRevision)
		{
			this->ActiveCount = 0;
			this->Pool.resize(source.ActiveCount);
			
			for (uint4 i = 0, count = source.ActiveCount; i < count; ++i)
				this->Pool[i].Reset(source.Pool[i].ID);
			this->ActiveCount = source.ActiveCount;

			this->StructureRevision = source.StructureRevision;
		}
	}

	/// Resets all queues.
	void Reset()
	{
		for (typename pool_t::iterator it = this->Pool.begin(), itEnd = this->Pool.end(); it != itEnd; ++it)
			it->Reset(it->ID);
	}

	/// Shrinks the pool to only those queues that are currently active.
	void Shrink()
	{
		this->Pool.shrink(this->ActiveCount);
	}

	/// Clears all active queues.
	void Clear()
	{
		this->ActiveCount = 0;
		++this->StructureRevision;
	}

	typedef QueueState* iterator;
	typedef const QueueState* const_iterator;
	iterator begin() { return this->Pool.data(); }
	const_iterator begin() const { return this->Pool.data(); }
	iterator end() { return this->Pool.data() + this->ActiveCount; }
	const_iterator end() const { return this->Pool.data() + this->ActiveCount; }
	QueueState& operator [](size_t pos) { return this->Pool[pos]; }
	const QueueState& operator [](size_t pos) const { return this->Pool[pos]; }
	size_t size() const { return this->ActiveCount; }
};

} // namespace

#endif