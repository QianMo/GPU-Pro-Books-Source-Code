/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#pragma once
#ifndef BE_SCENE_PERSPECTIVE_STATE_POOL
#define BE_SCENE_PERSPECTIVE_STATE_POOL

#include "beScene.h"
#include <beCore/bePool.h>
#include "beRenderingLimits.h"
#include "bePerspective.h"
#include <lean/smart/resource_ptr.h>
#include <lean/smart/scoped_ptr.h>

namespace beScene
{

/// Utility base class performing basic perspective state management.
template <class Parent, class Derived>
struct PerspectiveStateBase : public beCore::PooledToRefCounted<Derived>
{
	typedef PerspectiveStateBase base_type;	///< Type of this base class.
	typedef Parent parent_type;				///< Owner type.
	parent_type *const Parent;				///< Owner of this perspective state.
	Perspective *PerspectiveBinding;		///< Perspective that this state was last attached to.

	/// Constructor.
	explicit PerspectiveStateBase(parent_type *parent)
		: Parent(parent),
		PerspectiveBinding() { }
	/// Destructor. Makes sure perspective state is detached before destruction.
	~PerspectiveStateBase()
	{
		// IMPORTANT: Don't re-detach, check if still in use (perspective might have already been destructed)
		if (IsUsed() && this->PerspectiveBinding)
		{
			// ORDER: Null binding first, NEVER call Derived::Reset from destructor!
			Perspective *binding = this->PerspectiveBinding;
			this->PerspectiveBinding = nullptr;
			binding->SetState(this->Parent, nullptr);
		}
	}

	/// Marks this state detached & calls reset.
	void UsersReleased()
	{
		// IMPORTANT: Only reset if still bound, NEVER call Derived::Reset from destructor!
		if (this->PerspectiveBinding)
			// NOTE: Keep perspective affinity until explicitly reset (perspective might re-appear)
			static_cast<Derived*>(this)->Reset(this->PerspectiveBinding);
	}

	/// Resets this state.
	void Reset(Perspective *newPerspective)
	{
		LEAN_ASSERT(!IsUsed() || this->PerspectiveBinding == newPerspective);
		this->PerspectiveBinding = newPerspective;
	}
};

/// Pool perspective states.
template <class PerspectiveState>
class PerspectiveStatePool : public beCore::Pool<PerspectiveState>
{
public:
	/// Gets the state for the given perspective.
	PerspectiveState* FreeElement(const Perspective *perspective)
	{
		for (typename PerspectiveStatePool::ElementVector::iterator it = this->Elements.begin(); it != this->Elements.end(); ++it)
		{
			PerspectiveState *element = it->get();

			if (!element->IsUsed() && element->PerspectiveBinding == perspective)
				return element;
		}

		return this->PerspectiveStatePool::Pool::FreeElement();
	}

	/// Gets a NEW state object for the given perspective.
	template <class Parent>
	PerspectiveState& NewState(Perspective *perspective, Parent *parent)
	{
		typename PerspectiveState::parent_type *parentEquityped = parent;

		PerspectiveState *state = this->FreeElement(perspective);
		if (!state)
			state = this->AddElement( new PerspectiveState(parentEquityped) );

		state->Reset(perspective);
		perspective->SetState(parentEquityped, state);
		return *state;
	}

	/// Gets the state for the given perspective.
	template <class Parent>
	PerspectiveState& GetState(Perspective &perspective, Parent *parent)
	{
		typename PerspectiveState::parent_type *parentEquityped = parent;
		PerspectiveState *state = static_cast<PerspectiveState*>( perspective.GetState(parentEquityped) );
		if (!state) state = &NewState(&perspective, parentEquityped);
		LEAN_ASSERT(state->Parent == parent);
		return *state;
	}

	/// Gets an existing state for the given perspective.
	template <class Parent>
	PerspectiveState& GetExistingState(const Perspective &perspective, Parent *parent)
	{
		typename PerspectiveState::parent_type *parentEquityped = parent;
		return *LEAN_ASSERT_NOT_NULL( static_cast<PerspectiveState*>( perspective.GetState(parentEquityped) ) );
	}

	typedef PerspectiveState*const* iterator;
	typedef const PerspectiveState*const* const_iterator;
	iterator begin() { return &this->Elements.data()->get(); }
	const_iterator begin() const { return &this->Elements.data()->get(); }
	iterator end() { return &this->Elements.data()->get() + this->Elements.size(); }
	const_iterator end() const { return &this->Elements.data()->get() + this->Elements.size(); }
	PerspectiveState& operator [](size_t pos) { return this->Pool[pos]; }
	const PerspectiveState& operator [](size_t pos) const { return this->Pool[pos]; }
	size_t size() const { return this->Elements.size(); }
};

} // namespace

#endif