/*****************************************************/
/* breeze Engine Core Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef BE_CORE_VECTOR_QUERY_RESULT
#define BE_CORE_VECTOR_QUERY_RESULT

#include "beCore.h"
#include "beQueryResult.h"
#include <lean/containers/simple_vector.h>

namespace beCore
{

/// Vector query result.
template <class Vector, class Result = typename Vector::value_type, size_t ResultOffset = 0, size_t GrowthDenominator = 1>
class VectorQueryResultAdapter : public QueryResult<Result>
{
private:
	Vector &m_vector;
	size_t m_offset;
	size_t m_count;

public:
	/// Vector type.
	typedef Vector vector_type;

	/// Constructor.
	LEAN_INLINE VectorQueryResultAdapter(vector_type &vector, size_t offset = 0)
		: m_vector(vector),
		m_offset(offset),
		m_count(0)
	{
		LEAN_ASSERT(m_offset <= m_vector.size());
	}

	/// Resizes the buffer.
	value_type* Grow(size_t sizeHint)
	{
		size_t size = m_vector.size();
		size_t growth = size / GrowthDenominator;
		size_t maxSize = m_vector.max_size();

		// Mind overflow
		size = (maxSize - growth < size)
			? maxSize
			: size + growth;

		// Respect far-sighted hints
		if (size - m_offset < sizeHint)
			size = sizeHint + m_offset;

		m_vector.resize(size);

		return GetBuffer();
	}

	/// Sets the result count.
	void SetCount(size_t count)
	{
		LEAN_ASSERT(count <= m_vector.size() - m_offset);
		m_count = count;
	}
	/// Gets the actual result count.
	size_t GetCount() const { return m_count; }

	/// Expands the vector to its full capacity.
	LEAN_INLINE void Expand()
	{
		m_vector.resize(m_vector.capacity());
	}
	/// Trims the stored vector to the number of results added.
	LEAN_INLINE void Trim()
	{
		m_vector.resize(m_offset + m_count);
	}

	/// Gets the result buffer.
	value_type* GetBuffer()
	{
		return reinterpret_cast<value_type*>(
			reinterpret_cast<char*>(&m_vector[0] + m_offset) + ResultOffset );
	}
	/// Gets the result buffer.
	const value_type* GetBuffer() const
	{
		return reinterpret_cast<const value_type*>(
			reinterpret_cast<const char*>(&m_vector[0] + m_offset) + ResultOffset );
	}
	/// Gets the result buffer stride.
	size_t GetStride() const { return sizeof(typename vector_type::value_type); }
	/// Gets the result buffer size.
	size_t GetCapacity() const { return m_vector.size() - m_offset; }
};

/// Vector query result.
template <class Result, class SimpleVectorPolicy, class Allocator, size_t GrowthDenominator = 1>
class VectorQueryResult
	: public VectorQueryResultAdapter<
		lean::simple_vector<Result, SimpleVectorPolicy, Allocator>,
		Result,
		0,
		GrowthDenominator >
{
private:
	typedef VectorQueryResultAdapter<
		lean::simple_vector<Result, SimpleVectorPolicy, Allocator>,
		Result,
		0,
		GrowthDenominator > base_type;

	typename base_type::vector_type m_vector;

public:
	/// Constructor.
	LEAN_INLINE VectorQueryResult()
		: base_type(m_vector) { }

	/// Trims the result vector.
	void Finalize()
	{
		this->Trim();
	}

	/// Gets the result vector.
	LEAN_INLINE const typename base_type::vector_type& GetVector() const { return m_vector; }
};

} // namespace

#endif