#include "Precompiled.h"

#include "D3D10Exception.h"

#include "DXGIFactory.h"

namespace Mod
{
	DXGIFactory::DXGIFactory()
	{

		// create DXGI factory
		{
			typedef HRESULT (WINAPI *CreateDXGIFactoryPtrTypes)(REFIID , void **);

			CreateDXGIFactoryPtrTypes CreateDXGIFactoryPtr;

#ifdef MD_D3D10_STATIC_LINK
			CreateDXGIFactoryPtr = CreateDXGIFactory;
#else
			DynamicLibraryConfig dlcfg = { L"dxgi.dll" };
			mDXGILibrary = System::Single().CreateDynamicLibrary( dlcfg );

			CreateDXGIFactoryPtr = (CreateDXGIFactoryPtrTypes)mDXGILibrary->GetProcAddress( "CreateDXGIFactory" );
#endif

			// check for consistency... (got an error hear - update the CreateDXGIFactoryPtr from dxgi header)
			sizeof (CreateDXGIFactoryPtr = CreateDXGIFactory);

			IDXGIFactory * factory;
			D3D10_THROW_IF( CreateDXGIFactoryPtr( __uuidof(IDXGIFactory), (void**)(&factory) ) );

			mDXGIFactory.set( factory );
		}

		// Build mode list
		{
			IDXGIAdapter * adapter; 
			Types < ComPtr< IDXGIAdapter > > :: Vec adapters; 
			{
				UINT i = 0; 
				while( mDXGIFactory->EnumAdapters( i++, &adapter ) != DXGI_ERROR_NOT_FOUND ) 
				{ 
					adapters.resize( adapters.size() + 1 ); 
					adapters.back().set( adapter );
				}
			}

			adapter = &*adapters[ 0 ];

			IDXGIOutput * output;
			Types < ComPtr< IDXGIOutput > > :: Vec outputs;
			{
				UINT i = 0;
				while( adapter->EnumOutputs( i++, &output) != DXGI_ERROR_NOT_FOUND )
				{
					outputs.resize( outputs.size() + 1 );
					outputs.back().set( output );
				}
			}

			output = &*outputs[ 0 ];

			UINT num			= 0;
			DXGI_FORMAT format	= D3D10_DISPLAY_FORMAT;
			UINT flags			= DXGI_ENUM_MODES_INTERLACED | DXGI_ENUM_MODES_SCALING;

			D3D10_THROW_IF( output->GetDisplayModeList( format, flags, &num, 0) );

			mModeDescs.resize( num );

			D3D10_THROW_IF( output->GetDisplayModeList( format, flags, &num, &mModeDescs[0] ) );

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
	}

	//------------------------------------------------------------------------

	DXGIFactory::~DXGIFactory()
	{
	}

	//------------------------------------------------------------------------
	/*static*/

	DXGIFactory&
	DXGIFactory::Single()
	{
		static DXGIFactory single;

		return single;
	}

	//------------------------------------------------------------------------

	const DXGIFactory::Ptr&
	DXGIFactory::Get() const
	{
		return mDXGIFactory;
	}

	//------------------------------------------------------------------------

	const DXGIFactory::ModeDescs&
	DXGIFactory::GetModeDescs() const
	{
		return mModeDescs;
	}

	//------------------------------------------------------------------------

	const
	DXGIFactory::SimplifiedModeDescs&
	DXGIFactory::GetSimplifiedModeDescs() const
	{
		return mSimplifiedModeDescs;
	}

	//------------------------------------------------------------------------

	void
	DXGIFactory::Release()
	{
		mDXGIFactory.set( NULL );
	}
}