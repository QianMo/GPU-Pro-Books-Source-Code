#include "Precompiled.h"

#include "D3D9Exception.h"

#include "D3D9Instance.h"

namespace Mod
{
	D3D9Instance::D3D9Instance()
	{

		// create DXGI factory
		{
			IDirect3D9* d3d = Direct3DCreate9( D3D_SDK_VERSION );
			MD_FERROR_ON_FALSE( d3d );
			mD3D9Instance.set( d3d );
		}

		// Build mode list
		{
			for( UINT i = 0, e = mD3D9Instance->GetAdapterModeCount( D3DADAPTER_DEFAULT, D3D9_DISPLAY_FORMAT ); i < e; i ++ )
			{
				D3DDISPLAYMODE mode;
				HRESULT hr = mD3D9Instance->EnumAdapterModes( D3DADAPTER_DEFAULT, D3D9_DISPLAY_FORMAT, i, &mode );

				if( hr == D3D_OK )
				{
					mModeDescs.push_back( mode );
				}
			}

			typedef Types < UINT64 > :: Set SModSet;
			SModSet simplifiedModes;

			for( size_t i = 0, e = mModeDescs.size(); i < e; i ++ )
			{
				simplifiedModes.insert( ((UINT64)mModeDescs[i].Width << 32) + mModeDescs[ i ].Height );
			}

			for( SModSet::const_iterator i = simplifiedModes.begin(), e = simplifiedModes.end(); i != e; ++i )
			{
				SimplifiedModeDesc d;
				d.width = (*i) >> 32;
				d.height = (UINT32)(*i);

				mSimplifiedModeDescs.push_back( d );
			}
		}

		Exists() = true;
	}

	//------------------------------------------------------------------------

	D3D9Instance::~D3D9Instance()
	{
		Exists() = false;
	}

	//------------------------------------------------------------------------
	/*static*/

	D3D9Instance&
	D3D9Instance::Single()
	{
		static D3D9Instance single;

		return single;
	}

	//------------------------------------------------------------------------

	/*static*/
	bool&
	D3D9Instance::Exists()
	{
		static bool exists = false;
		return exists;
	}

	//------------------------------------------------------------------------

	const D3D9Instance::Ptr&
	D3D9Instance::Get() const
	{
		return mD3D9Instance;
	}

	//------------------------------------------------------------------------

	const D3D9Instance::ModeDescs&
	D3D9Instance::GetModeDescs() const
	{
		return mModeDescs;
	}

	//------------------------------------------------------------------------

	const
	D3D9Instance::SimplifiedModeDescs&
	D3D9Instance::GetSimplifiedModeDescs() const
	{
		return mSimplifiedModeDescs;
	}

	//------------------------------------------------------------------------

	void
	D3D9Instance::Release()
	{
		mD3D9Instance.set( NULL );
	}
}