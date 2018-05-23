/*****************************************************/
/* breeze Engine Core Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef BE_CORE_OPAQUE_HANDLE
#define BE_CORE_OPAQUE_HANDLE

#include "beCore.h"
#include "beWrapper.h"

namespace beCore
{

/// Opaque handle interface.
template <class Tag>
class OpaqueHandle
{
protected:
	void *m_handle;

	/// Constructor.
	LEAN_INLINE OpaqueHandle(void *handle)
		: m_handle(handle) { }

public:
	/// Constructor.
	LEAN_INLINE OpaqueHandle()
		: m_handle() { }
	/// Copy constructor.
	LEAN_INLINE OpaqueHandle(const OpaqueHandle &right)
		: m_handle(right.m_handle) { }
	/// Copy assignment operator.
	LEAN_INLINE OpaqueHandle& operator =(const OpaqueHandle &right)
	{
		m_handle = right.m_handle;
		return *this;
	}
};


/// Qualified handle declaration.
template <class Tag>
class QualifiedHandle;

template <class Tag, class Interface, class WrapperClass>
class QualifiedHandlePrototype : public WrapperClass, public OpaqueHandle<Tag>
{
	friend WrapperClass;
	
	/// Converts the handle into a pointer of the interface type.
	LEAN_INLINE Interface*const& GetInterface() const { return reinterpret_cast<Interface*const&>(m_handle); }
	
public:
	/// Constructs a qualified handle from the given interface pointer.
	QualifiedHandlePrototype(Interface *pInterface = nullptr)
		: OpaqueHandle<Tag>(pInterface) { }

	/// Converts the given array of interface pointers into an array of qualified handles.
	static LEAN_INLINE Interface** Array(OpaqueHandle<Tag> *arr) { return reinterpret_cast<Interface**>(arr); }
	/// Converts the given array of interface pointers into an array of qualified handles.
	static LEAN_INLINE Interface*const* Array(const OpaqueHandle<Tag> *arr) { return reinterpret_cast<Interface*const*>(arr); }
	
	/// Converts the given array of interface pointers into an array of qualified handles.
	static LEAN_INLINE QualifiedHandle<Tag>* Array(Interface **arr) { return reinterpret_cast<QualifiedHandle<Tag>*>(arr); }
	/// Converts the given array of interface pointers into an array of qualified handles.
	static LEAN_INLINE const QualifiedHandle<Tag>* Array(Interface *const *arr) { return reinterpret_cast<const QualifiedHandle<Tag>*>(arr); }
};

/// Default qualified handle implementation.
#define BE_CORE_DEFINE_QUALIFIED_HANDLE(Tag, Interface, Wrapper)														\
	template <>																											\
	class beCore::QualifiedHandle<Tag>																					\
		: public beCore::QualifiedHandlePrototype< Tag, Interface, Wrapper< Interface, beCore::QualifiedHandle<Tag> > >	\
	{																													\
	public:																												\
		QualifiedHandle(Interface *pInterface = nullptr)																\
			: QualifiedHandlePrototype(pInterface) { }																	\
	};

/// Casts the given opaque handle to its qualified equivalent.
template <class Tag>
LEAN_INLINE QualifiedHandle<Tag>* ToImpl(OpaqueHandle<Tag> *handle)
{
	return static_cast< QualifiedHandle<Tag>* >(handle);
}
/// Casts the given opaque handle to its qualified equivalent.
template <class Tag>
LEAN_INLINE const QualifiedHandle<Tag>* ToImpl(const OpaqueHandle<Tag> *handle)
{
	return static_cast< const QualifiedHandle<Tag>* >(handle);
}
/// Casts the given opaque handle to its qualified equivalent.
template <class Tag>
LEAN_INLINE QualifiedHandle<Tag>& ToImpl(OpaqueHandle<Tag> &handle)
{
	return *ToImpl(&handle);
}
/// Casts the given opaque handle to its qualified equivalent.
template <class Tag>
LEAN_INLINE const QualifiedHandle<Tag>& ToImpl(const OpaqueHandle<Tag> &handle)
{
	return *ToImpl(&handle);
}

} // namespace

#endif