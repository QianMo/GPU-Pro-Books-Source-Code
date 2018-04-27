#include "Precompiled.h"

#include "WrapSys/Src/DynamicLibrary.h"

#include "D3D10ImportedFunctions.h"

namespace Mod
{

	D3D10CompileEffectFromMemoryType MD_D3D10CompileEffectFromMemory;
	D3D10CreateEffectPoolFromMemoryType MD_D3D10CreateEffectPoolFromMemory;

#ifdef MD_D3D10_STATIC_LINK

	void MD_InitD3D10ImportedFunctions()
	{
		MD_D3D10CompileEffectFromMemory = D3D10CompileEffectFromMemory;
		MD_D3D10CreateEffectPoolFromMemory = D3D10CreateEffectPoolFromMemory;
	}

#else

#define MD_IMPORT_D3D10_FUNCTION(name)						\
	sizeof (MD_##name = name);								\
	MD_##name = (name##Type) lib->GetProcAddress( #name );	\
	MD_FERROR_ON_FALSE( MD_##name );


	void MD_InitD3D10ImportedFunctions( DynamicLibraryPtr lib )
	{
		MD_IMPORT_D3D10_FUNCTION(D3D10CompileEffectFromMemory)
		MD_IMPORT_D3D10_FUNCTION(D3D10CreateEffectPoolFromMemory)		
	}
#endif
}