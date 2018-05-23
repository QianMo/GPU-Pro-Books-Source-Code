/*****************************************************/
/* breeze Engine Core Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef BE_CORE_QUERY_RESULT
#define BE_CORE_QUERY_RESULT

#include "beCore.h"

namespace beCore
{
	
/// Query result base class.
template <class Result>
class QueryResult
{
protected:
	QueryResult& operator =(const QueryResult&) { return *this; }
	~QueryResult() throw() { }

public:
	/// Value type.
	typedef Result value_type;

	/// Resizes the buffer.
	virtual value_type* Grow(size_t sizeHint) = 0;

	/// Sets the actual result count.
	virtual void SetCount(size_t count) = 0;
	/// Gets the actual result count.
	virtual size_t GetCount() const = 0;

	/// Gets the result buffer.
	virtual value_type* GetBuffer() = 0;
	/// Gets the result buffer.
	virtual const value_type* GetBuffer() const = 0;
	/// Gets the result buffer stride.
	virtual size_t GetStride() const = 0;
	/// Gets the result buffer size.
	virtual size_t GetCapacity() const = 0;

	/// Called when the result accumulation has been finished.
	virtual void Finalize() { };
};

/// Advances the given pointer by n times the stride.
template <class Value>
LEAN_INLINE Value* Advance(Value *ptr, size_t stride, ptrdiff_t n = 1)
{
	// C-style cast to get rid of modifiers
	return reinterpret_cast<Value*>( (char*) ptr + n * static_cast<ptrdiff_t>(stride) );
}

/// Computes the difference of the given pointers respecting the given stride.
template <class Value>
LEAN_INLINE ptrdiff_t Difference(Value *ptr1, Value *ptr2, size_t stride)
{
	// C-style cast to get rid of modifiers
	return ((char*) ptr1 - (char*) ptr2) / static_cast<ptrdiff_t>(stride);
}

} // namespace

#endif