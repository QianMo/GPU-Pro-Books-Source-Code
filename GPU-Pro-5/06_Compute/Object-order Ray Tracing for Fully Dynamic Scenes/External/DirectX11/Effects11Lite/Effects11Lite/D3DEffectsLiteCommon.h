//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (C) Tobias Zirr.  All Rights Reserved.
//
//////////////////////////////////////////////////////////////////////////////

#pragma once

// TODO: Could get rid of this as well
#include <Windows.h>

/// @addtogroup D3DEffectsLite D3D Effects Lite
/// @{

#ifndef D3DEFFECTSLITE_API
	/// Redefine to _declspec(dllexport|dllimport) to build DLL.
	#define D3DEFFECTSLITE_API
#endif

#ifndef D3DEFFECTSLITE_INTERFACE
	#ifdef _MSC_VER
		#define D3DEFFECTSLITE_INTERFACE _declspec(novtable)
	#else
		/// Interface v-table optimization, e.g. _declspec(novtable). Redefinable.
		#define D3DEFFECTSLITE_INTERFACE
	#endif
#endif

#ifndef D3DEFFECTSLITE_INTERFACE_BEHAVIOR
	/// Makes the given class behave like an interface. Redefinable.
	#define D3DEFFECTSLITE_INTERFACE_BEHAVIOR(name) \
		protected: \
			name& operator =(const name&) { return *this; } \
			~name() { }
#endif

#ifndef D3DEFFECTSLITE_CONST_PTR_ACCESS
	/// Enforces transitive constness. Redefinable.
	#define D3DEFFECTSLITE_CONST_PTR_ACCESS(T) T const *
#endif

#ifndef D3DEFFECTSLITE_STDCALL
	/// API calling convention. Redefinable.
	#define D3DEFFECTSLITE_STDCALL _stdcall
#endif

/// Implement this interface to provide customized storage for effect data.
class D3DEFFECTSLITE_INTERFACE D3DEffectsLiteAllocator
{
	D3DEFFECTSLITE_INTERFACE_BEHAVIOR(D3DEffectsLiteAllocator)

public:
	/// Allocates the given number of bytes. Returns nullptr on failure.
	virtual void* D3DEFFECTSLITE_STDCALL Allocate(UINT size) = 0;
	/// Frees the given block of memory. Data may be nullptr.
	virtual void D3DEFFECTSLITE_STDCALL Free(void *data) = 0;
};

/// Provides access to arbitrary byte data.
class D3DEFFECTSLITE_INTERFACE D3DEffectsLiteBlob : public IUnknown
{
	D3DEFFECTSLITE_INTERFACE_BEHAVIOR(D3DEffectsLiteBlob)

public:
	/// Gets a pointer to the stored data.
	virtual void* D3DEFFECTSLITE_STDCALL Data() const = 0;
	/// Gets the size of the stored data.
	virtual UINT D3DEFFECTSLITE_STDCALL Size() const = 0;
};

/// Include types.
enum D3DEffectsLiteIncludeType
{
	D3DEffectsLiteIncludeLocal,
	D3DEffectsLiteIncludeSystem
};

/// Provides access to arbitrary byte data.
class D3DEFFECTSLITE_INTERFACE D3DEffectsLiteInclude
{
	D3DEFFECTSLITE_INTERFACE_BEHAVIOR(D3DEffectsLiteInclude)

public:
	/// Opens the given include file.
	virtual HRESULT D3DEFFECTSLITE_STDCALL Open(D3DEffectsLiteIncludeType type, const char *fileName,
		const void *parent, const void **child, UINT *childSize) = 0;
	/// Closes the given include file.
	virtual void D3DEFFECTSLITE_STDCALL Close(const void *child) = 0;
};

/// Gets the default allocator.
D3DEFFECTSLITE_API D3DEffectsLiteAllocator* D3DEFFECTSLITE_STDCALL D3DELGetDefaultAllocator();
/// Sets the global allocator.
D3DEFFECTSLITE_API void D3DEFFECTSLITE_STDCALL D3DELSetGlobalAllocator(D3DEffectsLiteAllocator *allocator);
/// Gets the global allocator.
D3DEFFECTSLITE_API D3DEffectsLiteAllocator* D3DEFFECTSLITE_STDCALL D3DELGetGlobalAllocator();

#ifndef D3DEFFECTSLITE_NO_CPP

#ifdef DOXYGEN_READ_THIS
	/// Define this to disable the C++-style API.
	#define D3DEFFECTSLITE_NO_CPP
#endif

/// C++-style API namespace.
namespace D3DEffectsLite
{
	typedef D3DEffectsLiteAllocator Allocator;
	typedef D3DEffectsLiteBlob Blob;

	struct IncludeType
	{
		typedef D3DEffectsLiteIncludeType T;

		static const T Local = D3DEffectsLiteIncludeLocal;
		static const T System = D3DEffectsLiteIncludeSystem;
	};
	typedef D3DEffectsLiteInclude Include;

	D3DEFFECTSLITE_API Allocator* D3DEFFECTSLITE_STDCALL GetDefaultAllocator();
	D3DEFFECTSLITE_API void D3DEFFECTSLITE_STDCALL SetGlobalAllocator(Allocator*);
	D3DEFFECTSLITE_API Allocator* D3DEFFECTSLITE_STDCALL GetGlobalAllocator();

} // namespace

#endif

/// @}