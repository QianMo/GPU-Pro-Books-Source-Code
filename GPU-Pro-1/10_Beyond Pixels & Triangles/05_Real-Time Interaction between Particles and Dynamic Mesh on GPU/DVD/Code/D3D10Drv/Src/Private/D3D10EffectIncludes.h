#ifndef D3D10DRV_D3D10EFFECTINCLUDES_H_INCLUDED
#define D3D10DRV_D3D10EFFECTINCLUDES_H_INCLUDED

#include "Forw.h"

namespace Mod
{
	class D3D10EffectIncludes : public ID3D10Include
	{
		// types
	public:
		typedef Types2< LPCVOID, String > :: Map KeyMap;

		// construction/ destruction
	public:
		D3D10EffectIncludes();
		~D3D10EffectIncludes();

		// manipulation/ access
	public:
		void Reset();

		// polymorphism
	private:
		virtual HRESULT __stdcall Close(	LPCVOID pData )	OVERRIDE;
		virtual HRESULT __stdcall Open(		D3D10_INCLUDE_TYPE IncludeType,
											LPCSTR pFileName,
											LPCVOID pParentData,
											LPCVOID *ppData,
											UINT *pBytes ) OVERRIDE;

		// data
	private:
		KeyMap mKeyMap;
	};
}

#endif
