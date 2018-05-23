/*****************************************************/
/* breeze Engine Core Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef BE_CORE_MANY
#define BE_CORE_MANY

#include "beCore.h"
#include <lean/tags/transitive_ptr.h>

namespace beCore
{

/// Handle to an entity.
template <class Group, class Index = uint4>
struct GroupElementHandle
{
	friend Group;

	typedef Group group_type;
	lean::transitive_ptr<Group, true> Group;	///< Parent container.
	typedef Index index_type;
	const Index Index;							///< Internal element index, MAY CHANGE.

	LEAN_INLINE const GroupElementHandle& operator =(const GroupElementHandle &right) const
	{
		const_cast<group_type*&>(Group.get()) = const_cast<group_type*>(right.Group.get());
		const_cast<index_type&>(Index) = right.Index;
		return *this;
	}

protected:
	/// Internal constructor.
	LEAN_INLINE GroupElementHandle(group_type *group, index_type index)
		: Group(group),
		Index(index) { }

	/// Sets the index.
	LEAN_INLINE void SetIndex(index_type index)
	{
		const_cast<index_type&>(Index) = index;
	}
};

/// Range.
template <class Index>
struct Range
{
	typedef Index index_type;	///< Index type;
	typedef Index iterator;		///< Index type;

	Index Begin;	///< Beginning of the range.
	Index End;		///< End of the range.

	/// Empty range constructor.
	LEAN_INLINE Range()
		: Begin(), End() { }
	/// Range constructor.
	LEAN_INLINE Range(Index begin, Index end)
		: Begin(begin), End(end) { }
	/// Conversion constructor.
	template <class OtherIdx>
	LEAN_INLINE Range(const Range<OtherIdx> &r)
		: Begin(r.Begin), End(r.End) { }

	/// Element access.
	LEAN_INLINE typename lean::iterator_reflexive<iterator>::reference operator [](ptrdiff_t pos) const { return *(Begin + pos); }
	/// First element access.
	LEAN_INLINE typename lean::iterator_reflexive<iterator>::reference operator *() const { return *Begin; }
	/// First element access.
	LEAN_INLINE typename iterator operator ->() const { return Begin; }
	/// Checks if non-empty.
	LEAN_INLINE operator bool() const { return Begin != End; }
	/// Increments begin.
	LEAN_INLINE Range& operator ++() { ++Begin; return *this; }

	/// Gets the beginning (STL compatibility).
	LEAN_INLINE iterator& begin() { return Begin; }
	/// Gets the end (STL compatibility).
	LEAN_INLINE iterator& end() { return End; }
	/// Gets the beginning (STL compatibility).
	LEAN_INLINE iterator begin() const { return Begin; }
	/// Gets the end (STL compatibility).
	LEAN_INLINE iterator end() const { return End; }
	/// Checks if empty (STL compatibility).
	LEAN_INLINE bool empty() const { return (Begin == End); }

	/// Gets the size (STL compatibility).
	LEAN_INLINE size_t size() const { return End - Begin; }
};

/// Makes a range from the given pointers.
/// @relatesalso Range
template <class Pointer>
LEAN_INLINE Range<Pointer> MakeRange(Pointer begin, Pointer end)
{
	return Range<Pointer>(begin, end);
}

/// Makes a range from the given pointers.
/// @relatesalso Range
template <class Pointer, class Diff>
LEAN_INLINE Range<Pointer> MakeRangeN(Pointer begin, Diff count)
{
	return Range<Pointer>(begin, static_cast<Pointer>(begin + count));
}

/// @relates Range
/// @{

/// Gets the size.
template <class Pointer>
LEAN_INLINE size_t Size(const Range<Pointer> &range)
{
	return range.End - range.Begin;
}

/// Gets the size.
template <class SizeT, class Pointer>
LEAN_INLINE SizeT Size(const Range<Pointer> &range)
{
	return static_cast<SizeT>(range.End - range.Begin);
}

/// Gets the size.
template <class Pointer>
LEAN_INLINE uint4 Size4(const Range<Pointer> &range)
{
	return static_cast<uint4>(range.End - range.Begin);
}

/// Continuous data access.
template <class Pointer>
LEAN_INLINE typename lean::iterator_reflexive<Pointer>::pointer Data(const Range<Pointer> &range)
{
	return range.Begin;
}

/// Continuous data access.
template <class Pointer>
LEAN_INLINE typename lean::iterator_reflexive<Pointer>::pointer DataEnd(const Range<Pointer> &range)
{
	return range.End;
}

/// To standard range.
template <class Pointer>
LEAN_INLINE lean::range<Pointer> ToSTD(const Range<Pointer> &range)
{
	return lean::range<Pointer>(range.Begin, range.End);
}

/// Gets the beginning (for each compatibility).
template <class Pointer>
LEAN_INLINE Pointer begin(const Range<Pointer> &range)
{
	return range.Begin;
}

/// Gets the end (for each compatibility).
template <class Pointer>
LEAN_INLINE Pointer end(const Range<Pointer> &range)
{
	return range.End;
}

/// @}

} // namespace

/// @addtogroup GlobalMacros
/// @{

#define BE_STATIC_PIMPL_HANDLE(handle) \
	LEAN_STATIC_PIMPL_AT(*(handle).Group); \
	LEAN_ASSERT(VerifyHandle(m, handle));
#define BE_STATIC_PIMPL_HANDLE_CONST(handle) \
	LEAN_STATIC_PIMPL_AT_CONST(*(handle).Group); \
	LEAN_ASSERT(VerifyHandle(m, handle));

#define BE_FREE_STATIC_PIMPL_HANDLE(t, handle) \
	LEAN_FREE_STATIC_PIMPL_AT(t, *(handle).Group); \
	LEAN_ASSERT(VerifyHandle(m, handle));
#define BE_FREE_STATIC_PIMPL_HANDLE_CONST(t, handle) \
	LEAN_FREE_STATIC_PIMPL_AT_CONST(t, *(handle).Group); \
	LEAN_ASSERT(VerifyHandle(m, handle));

/// @}

#endif