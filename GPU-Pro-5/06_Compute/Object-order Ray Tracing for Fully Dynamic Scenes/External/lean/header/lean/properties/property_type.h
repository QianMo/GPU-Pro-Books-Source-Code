/*****************************************************/
/* lean Properties              (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_PROPERTIES_PROPERTY_TYPE
#define LEAN_PROPERTIES_PROPERTY_TYPE

#include "../lean.h"
#include "../smart/cloneable.h"
#include "../tags/noncopyable.h"
#include "../type_info.h"

namespace lean
{
namespace properties
{

/// Property type interface.
class LEAN_INTERFACE property_type
{
	LEAN_INTERFACE_BEHAVIOR(property_type)

public:
	/// Gets the size required by the given number of elements.
	virtual size_t size(size_t count) const = 0;
	/// Gets the element type info.
	virtual const type_info& type_info() const = 0;

	/// Allocates the given number of elements.
	virtual void* allocate(size_t count) const = 0;
	/// Constructs the given number of elements.
	virtual void construct(void *elements, size_t count) const = 0;
	/// Destructs the given number of elements.
	virtual void destruct(void *elements, size_t count) const = 0;
	/// Deallocates the given number of elements.
	virtual void deallocate(void *elements, size_t count) const = 0;
};

/// Allocates & constructs property data.
inline void* new_property_data(const property_type &type, size_t count)
{
	void *data = type.allocate(count);

	try
	{
		type.construct(data, count);
	}
	catch (...)
	{
		type.deallocate(data, count);
		throw;
	}

	return data;
}

/// Destructs & deletes property data.
inline void delete_property_data(const property_type &type, void *data, size_t count)
{
	try
	{
		type.destruct(data, count);
	}
	catch (...)
	{
		type.deallocate(data, count);
		throw;
	}

	type.deallocate(data, count);
}

/// Destructs property data using a given type object.
struct destruct_property_data_policy
{
	LEAN_INLINE static void release(const property_type &type, void *data, size_t count)
	{
		type.destruct(data, count);
	}
};
/// Deallocates property data using a given type object.
struct deallocate_property_data_policy
{
	LEAN_INLINE static void release(const property_type &type, void *data, size_t count)
	{
		type.deallocate(data, count);
	}
};
/// Deletes property data using a given type object.
struct delete_property_data_policy
{
	LEAN_INLINE static void release(const property_type &type, void *data, size_t count)
	{
		delete_property_data(type, data, count);
	}
};

/// Holds arbitrary property data.
template <class Policy = delete_property_data_policy>
class scoped_property_data : public noncopyable
{
private:
	const class property_type *m_type;
	void *m_data;
	size_t m_count;

public:
	/// Holds the given data & frees it according to the given policy on destruction.
	LEAN_INLINE scoped_property_data(const class property_type &type, void *data, size_t count)
		: m_type(&type),
		m_data(data),
		m_count(count) { }
	/// Holds the given data & frees it according to the given policy on destruction.
	LEAN_INLINE scoped_property_data(const class property_type &type, size_t count)
		: m_type( &type ),
		m_data( new_property_data(type, count) ),
		m_count( count ) { }
#ifndef LEAN0X_NO_RVALUE_REFERENCES
	/// Moves data from the given object to this object.
	LEAN_INLINE scoped_property_data(scoped_property_data &&right)
		: m_type(right.m_type),
		m_data(right.m_data),
		m_count(right.m_count)
	{
		right.m_type = nullptr;
	}
#endif
	/// Moves data from the given object to this object.
	LEAN_INLINE scoped_property_data(scoped_property_data &right, consume_t)
		: m_type(right.m_type),
		m_data(right.m_data),
		m_count(right.m_count)
	{
		right.m_type = nullptr;
	}
	/// Frees the stored data according to the given policy.
	LEAN_INLINE ~scoped_property_data()
	{
		if (m_type)
			Policy::release(*m_type, m_data, m_count);
	}

	/// Gets the data pointer.
	LEAN_INLINE void* data() { return m_data; }
	/// Gets the data pointer.
	LEAN_INLINE const void* data() const { return m_data; }

	/// Gets the number of data elements.
	LEAN_INLINE size_t count() const { return m_count; }

	/// Gets the type.
	LEAN_INLINE const class property_type& property_type() { return *m_type; }
};

/// Enhanced type info.
struct property_type_info : public type_info
{
	/// Property type.
	const property_type *property_type;

	/// Constructor.
	property_type_info(const std::type_info &type, size_t size, const class property_type &propertyType)
		: type_info(type, size),
		property_type(&propertyType) { }
	/// Constructor.
	property_type_info(const type_info &type, const class property_type &propertyType)
		: type_info(type),
		property_type(&propertyType) { }
};

} // namespace

using properties::property_type;

using properties::scoped_property_data;
using properties::delete_property_data_policy;
using properties::destruct_property_data_policy;
using properties::deallocate_property_data_policy;

using properties::property_type_info;

} // namespace

#endif