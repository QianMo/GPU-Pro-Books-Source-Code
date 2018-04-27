#ifndef D3D10DRV_D3D10IMPORTEDFUNCTIONS_H_INCLUDED
#define D3D10DRV_D3D10IMPORTEDFUNCTIONS_H_INCLUDED

#include "WrapSys/Src/Forw.h"

namespace Mod
{

	typedef
	HRESULT (WINAPI *D3D10CompileEffectFromMemoryType)(	void *pData, SIZE_T DataLength, LPCSTR pSrcFileName, CONST D3D10_SHADER_MACRO *pDefines, 
														ID3D10Include *pInclude, UINT HLSLFlags, UINT FXFlags, 
														ID3D10Blob **ppCompiledEffect, ID3D10Blob **ppErrors );

	extern
	D3D10CompileEffectFromMemoryType MD_D3D10CompileEffectFromMemory;

	typedef
	HRESULT (WINAPI *D3D10CreateEffectPoolFromMemoryType)(	void *pData, SIZE_T DataLength, UINT FXFlags, ID3D10Device *pDevice,
															ID3D10EffectPool **ppEffectPool );


	extern D3D10CreateEffectPoolFromMemoryType MD_D3D10CreateEffectPoolFromMemory;

#ifdef MD_D3D10_STATIC_LINK
	void MD_InitD3D10ImportedFunctions();
#else
	void MD_InitD3D10ImportedFunctions( DynamicLibraryPtr lib );
#endif



}

#endif