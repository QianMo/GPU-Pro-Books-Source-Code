/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#include "beSceneInternal/stdafx.h"
#include "beScene/bePerspective.h"

#include <beMath/beMatrix.h>
#include <beMath/beProjection.h>

#include <lean/functional/algorithm.h>

namespace beScene
{

// Multiplies the given two matrices.
beMath::fmat4 PerspectiveDesc::Multiply(const beMath::fmat4 &a, const beMath::fmat4 &b)
{
	return mul(a, b);
}

// Extracts a view frustum from the given view projection matrix.
void PerspectiveDesc::ExtractFrustum(beMath::fplane3 frustum[6], const beMath::fmat4 &viewProj)
{
	extract_frustum(viewProj, frustum);
}

// Constructor.
Perspective::Perspective()
	: m_dataHeap(1024),
	m_dataHeapWatermark(0)
{
}

// Constructor.
Perspective::Perspective(const PerspectiveDesc &desc)
	: m_desc(desc),
	m_dataHeap(1024),
	m_dataHeapWatermark(0)
{
}

// Destructor.
Perspective::~Perspective()
{
}

// Resets this perspective.
void Perspective::ResetReleased()
{
	m_state.clear();
}

// Discards temporary data.
void Perspective::ReleaseIntermediate()
{
	FreeData();
}

// Updates the description of this perspective.
void Perspective::SetDesc(const PerspectiveDesc &desc)
{
	m_desc = desc;
}

namespace
{

struct StateSorter
{
	static LEAN_INLINE const void* GetOwner(const void *owner) { return owner; }
	static LEAN_INLINE const void* GetOwner(const Perspective::State &s) { return s.Owner; }

	struct Less
	{
		template <class L, class R>
		LEAN_INLINE bool operator ()(const L &left, const R &right) const { return GetOwner(left) < GetOwner(right); }
	};
	struct Equal
	{
		template <class L, class R>
		LEAN_INLINE bool operator ()(const L &left, const R &right) const { return GetOwner(left) == GetOwner(right); }
	};
};

} // namespace

// Stores the given state object for the given owner.
void Perspective::SetState(const void *owner, beCore::RefCounted *state)
{
	state_t::iterator it = lean::find_sorted(m_state.begin(), m_state.end(), owner, StateSorter::Less());

	if (it != m_state.end())
	{
		if (state)
			it->Data = state;
		else
			m_state.erase(it);
	}
	else if (state)
		lean::push_sorted(m_state, State(owner, state), StateSorter::Less());
}

// Retrieves the state object stored for the given owner.
beCore::RefCounted* Perspective::GetState(const void *owner) const
{
	state_t::const_iterator it = lean::find_sorted(m_state.begin(), m_state.end(), owner, StateSorter::Less());

	return (it != m_state.end())
		? it->Data
		: nullptr;
}

// Allocates data from the perspective's internal heap.
void* Perspective::AllocateData(size_t size)
{
	m_dataHeapWatermark += size;
	return m_dataHeap.allocate(size);
}

// Frees all data allocated from this perspective's internal heap.
void Perspective::FreeData()
{
	size_t minDataCapacity = m_dataHeapWatermark;
	m_dataHeapWatermark = 0;

	m_dataHeap.clearButFirst();
	m_dataHeap.reserve(minDataCapacity);
}

} // namespace
