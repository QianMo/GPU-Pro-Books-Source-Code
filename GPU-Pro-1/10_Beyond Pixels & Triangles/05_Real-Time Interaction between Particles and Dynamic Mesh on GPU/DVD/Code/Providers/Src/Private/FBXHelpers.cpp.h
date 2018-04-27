#ifndef PROVIDERS_FBXHELPERS_CPP_H_INCLUDED
#define PROVIDERS_FBXHELPERS_CPP_H_INCLUDED

#include "FBXHelpers.h"

namespace Mod
{
	//------------------------------------------------------------------------

	template <typename T>
	FBXWrapper<T>::FBXWrapper() :
	ptr( NULL )
	{
	}

	//------------------------------------------------------------------------

	template <typename T>
	FBXWrapper<T>::FBXWrapper( T* a_ptr ) :
	ptr( a_ptr )
	{
	}

	//------------------------------------------------------------------------

	template <typename T>
	FBXWrapper<T>::~FBXWrapper()
	{
		Release();
	}

	template <typename T>
	void
	FBXWrapper<T>::Release()
	{
		if( ptr )
			ptr->Destroy();
		ptr = NULL;
	}

	//------------------------------------------------------------------------

	template <typename T>
	void
	FBXWrapper<T>::Reset( T* a_ptr )
	{
		Release();
		ptr = a_ptr;
	}

	//------------------------------------------------------------------------

	template <typename T>
	T*
	FBXWrapper<T>::operator->() const
	{
		MD_ASSERT(ptr);
		return ptr;
	}

	//------------------------------------------------------------------------

	template <typename T>
	T&
	FBXWrapper<T>::operator* () const
	{
		MD_ASSERT(ptr);
		return *ptr;
	}

	//------------------------------------------------------------------------

	template <typename T>
	T*
	FBXWrapper<T>::Get() const
	{
		return ptr;
	}

	//------------------------------------------------------------------------

	template <typename T>
	bool
	FBXWrapper<T>::IsNull() const
	{
		return ptr ? false : true;
	}

}


#endif
