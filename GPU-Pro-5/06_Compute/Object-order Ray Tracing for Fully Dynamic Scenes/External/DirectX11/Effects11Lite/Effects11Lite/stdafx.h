//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (C) Tobias Zirr. All Rights Reserved.
//
//////////////////////////////////////////////////////////////////////////////

// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once

#include "targetver.h"

#define WIN32_LEAN_AND_MEAN             // Exclude rarely-used stuff from Windows headers

#define _ITERATOR_DEBUG_LEVEL 0
#define _HAS_ITERATOR_DEBUGGING 0
#define _SECURE_SCL 0
#define _CRT_DISABLE_PERFCRIT_LOCKS 1

// Disable warnings
#define _CRT_SECURE_NO_WARNINGS 1 // Using _standard_ library
#define _SCL_SECURE_NO_WARNINGS 1

#ifdef _MSC_VER

#pragma warning(push)
// Some warnings won't go, even with _CRT_SECURE_NO_WARNINGS defined
#pragma warning(disable : 4996)

#endif

#define INITGUID
#include "D3D11.h"
#include "D3DCompiler.h"

#include <cassert>

#ifndef D3DEFFECTSLITE_INTERLOCKED_INCREMENT
	#define D3DEFFECTSLITE_INTERLOCKED_INCREMENT(v) InterlockedIncrement(&v)
#endif
#ifndef D3DEFFECTSLITE_INTERLOCKED_DECREMENT
	#define D3DEFFECTSLITE_INTERLOCKED_DECREMENT(v) InterlockedDecrement(&v)
#endif

#ifndef D3DEFFECTSLITE_ASSUME
	#define D3DEFFECTSLITE_ASSUME(v) __assume(v)
#endif

#define construct_ref(...) (__VA_ARGS__&) (const __VA_ARGS__&) __VA_ARGS__

template <class T, size_t S> char (&arraylen_helper(T (&a)[S]))[S];
#define arraylen(a) sizeof(arraylen_helper(a))

#define offsetof_uint(s, m) (UINT) offsetof(s, m)
#define strideof_uint(s, m) offsetof_uint(s, m[1]) - offsetof_uint(s, m[0])

// POD() initialization
#pragma warning(disable : 4345)

template <class T>
class transitive_ptr
{
	T *p;

public:
	transitive_ptr(T *p = nullptr) : p(p) { }
	template <class S> transitive_ptr(S *p) : p(p) { }

	transitive_ptr& operator =(T *p) { this->p = p; return *this; }

	T*& get() { return p; }
	const T *const & get() const { return p; }
//	volatile T *volatile & get() volatile { return p; }
//	const volatile T *const volatile & get() const volatile { return p; }

	T* operator ->() { return p; }
	const T* operator ->() const { return p; }
//	volatile T* operator ->() volatile { return p; }
//	const volatile T* operator ->() const volatile { return p; }

	T& operator *() { return p; }
	const T& operator *() const { return p; }
//	volatile T& operator *() volatile { return p; }
//	const volatile T& operator *() const volatile { return p; }

	operator T*&() { return p; }
	operator const T *const &() const { return p; }
//	operator volatile T *volatile &() volatile { return p; }
//	operator const volatile T *const volatile &() const volatile { return p; }
};

template <>
class transitive_ptr<void>
{
	void *p;

public:
	transitive_ptr(void *p = nullptr) : p(p) { }
	template <class S> transitive_ptr(S *p) : p(p) { }

	transitive_ptr& operator =(void *p) { this->p = p; return *this; }

	void*& get() { return p; }
	const void *const & get() const { return p; }
//	volatile void *volatile & get() volatile { return const_cast<volatile void *volatile &>(p); }
//	const volatile void *const volatile & get() const volatile { return p; }

	operator void*&() { return p; }
	operator const void *const &() const { return p; }
//	operator volatile void *volatile &() volatile { return const_cast<volatile void *volatile &>(p); }
//	operator const volatile void *const volatile &() const volatile { return p; }
};

#define D3DEFFECTSLITE_CONST_PTR_ACCESS(T) transitive_ptr<T>

template <class COMType>
inline void com_acquire(COMType *p)
{
	p->AddRef();
}

template <class COMType>
inline void com_release(COMType *p)
{
	p->Release();
}

enum bind_reference_t { bind_reference };

template <class COMType>
class com_ptr
{
public:
	typedef COMType com_type;
	typedef COMType* value_type;

private:
	com_type *m_object;

	static com_type* acquire_ref(com_type *object)
	{
		if (object)
			com_acquire(object);

		return object;
	}

	static void release_ref(com_type *object)
	{
		if (object)
			com_release(object);
	}

public:
	com_ptr(com_type *object = nullptr)
		: m_object( acquire_ref(object) ) { }
	template <class OtherCOMType>
	com_ptr(OtherCOMType *object)
		: m_object( acquire_ref(object) ) { }

	com_ptr(com_type *object, bind_reference_t)
		: m_object(object) { }

	com_ptr(const com_ptr &right)
		: m_object( acquire_ref(right.m_object) ) { }
	template <class OtherCOMType>
	com_ptr(const com_ptr<OtherCOMType> &right)
		: m_object( acquire_ref(right.get()) ) { }

#ifndef D3DEFFECTSLITE_NO_RVALUE_REFERENCES
	template <class OtherCOMType>
	com_ptr(com_ptr<OtherCOMType> &&right)
		: m_object(right.unbind()) { }
#endif
	
	~com_ptr() throw()
	{
		release_ref(m_object);
	}

	void rebind(com_type *object)
	{
		com_type *prevObject = m_object;
		m_object = object;
		release_ref(prevObject);
	}
	com_type* unbind()
	{
		com_type *preObject = m_object;
		m_object = nullptr;
		return preObject;
	}
	void reset(com_type *object)
	{
		// Do not check for redundant assignment
		// -> The code handles redundant assignment just fine
		// -> Checking generates up to twice the code due to unfortunate compiler optimization application order
		rebind( acquire_ref(object) );
	}
	void release()
	{
		rebind(nullptr);
	}

	com_ptr& operator =(com_type *object)
	{
		reset(object);
		return *this;
	}
	com_ptr& operator =(const com_ptr &right)
	{
		reset(right.m_object);
		return *this;
	}
	template <class OtherCOMType>
	com_ptr& operator =(const com_ptr<OtherCOMType> &right)
	{
		reset(right.get());
		return *this;
	}

#ifndef D3DEFFECTSLITE_NO_RVALUE_REFERENCES
	template <class OtherCOMType>
	com_ptr& operator =(com_ptr<OtherCOMType> &&right)
	{
		// Self-assignment would be wrong
		if ((void*) this != (void*) &right)
			rebind(right.unbind());

		return *this;
	}
#endif

	com_type*const& get() const { return m_object; }
	com_type& operator *() const { return *m_object; }
	com_type* operator ->() const { return m_object; }

	/// Gets a double-pointer allowing for COM-style object retrieval. The pointer returned may
	/// only ever be used until the next call to one of this COM pointer's methods.
	com_type** rebind()
	{
		release();
		return &m_object;
	}
};
