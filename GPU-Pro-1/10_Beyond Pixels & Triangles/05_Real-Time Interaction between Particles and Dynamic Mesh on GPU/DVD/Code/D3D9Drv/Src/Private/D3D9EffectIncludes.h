#ifndef D3D10DRV_D3D9EFFECTINCLUDES_H_INCLUDED
#define D3D10DRV_D3D9EFFECTINCLUDES_H_INCLUDED

#include "Forw.h"

namespace Mod
{
	class D3D9EffectIncludes : public ID3DXInclude
	{
		// types
	public:
		typedef Types2< LPCVOID, String > :: Map KeyMap;

		// construction/ destruction
	public:
		D3D9EffectIncludes();
		~D3D9EffectIncludes();

		// manipulation/ access
	public:
		void Reset();

		// polymorphism
	private:
		virtual HRESULT __stdcall Open(		D3DXINCLUDE_TYPE IncludeType,
											LPCSTR pFileName,
											LPCVOID pParentData,
											LPCVOID *ppData,
											UINT *pBytes ) OVERRIDE;
		
		virtual HRESULT __stdcall Close(	LPCVOID pData )	OVERRIDE;
		// data
	private:
		KeyMap mKeyMap;
	};
}

#endif
