#include "Precompiled.h"

#include "Providers/Src/EffectIncludeProvider.h"
#include "Providers/Src/Providers.h"

#include "D3D10EffectIncludes.h"

namespace Mod
{
	D3D10EffectIncludes::D3D10EffectIncludes()
	{

	}

	//------------------------------------------------------------------------

	D3D10EffectIncludes::~D3D10EffectIncludes()
	{

	}

	//------------------------------------------------------------------------

	void
	D3D10EffectIncludes::Reset()
	{
		mKeyMap.clear();
	}

	//------------------------------------------------------------------------

	/*virtual*/
	HRESULT
	D3D10EffectIncludes::Close(	LPCVOID pData )
	{
		KeyMap::iterator found = mKeyMap.find( pData );
		MD_FERROR_ON_FALSE( found != mKeyMap.end() );

		Providers::Single().GetEffectIncludeProv()->RemoveItem( found->second );

		return S_OK;
	}

	//------------------------------------------------------------------------

	/*virtual*/
	HRESULT
	D3D10EffectIncludes::Open(	D3D10_INCLUDE_TYPE /*IncludeType*/,
								LPCSTR pFileName,
								LPCVOID /*pParentData*/,
								LPCVOID *ppData,
								UINT *pBytes )
	{
		String file = ToString( pFileName );

		BytesPtr bytes = Providers::Single().GetEffectIncludeProv()->GetItem( file );

		// do not support empty includes

		void *rawptr = bytes->GetRawPtr();

		MD_FERROR_ON_FALSE( rawptr );

		*ppData = rawptr;
		*pBytes = UINT( bytes->GetSize() );

		mKeyMap[ *ppData ] = file;

		return S_OK;
	}

}