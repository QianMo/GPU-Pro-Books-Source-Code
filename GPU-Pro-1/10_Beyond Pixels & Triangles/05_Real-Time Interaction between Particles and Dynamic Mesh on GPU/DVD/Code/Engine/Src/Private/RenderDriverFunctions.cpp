#include "Precompiled.h"

#include "WrapSys/Src/DynamicLibraryConfig.h"
#include "WrapSys/Src/DynamicLibrary.h"
#include "WrapSys/Src/System.h"

#include "RenderDriverFunctionsConfig.h"
#include "RenderDriverFunctions.h"

#define MD_NAMESPACE RenderDriverFunctionsNS
#include "ConfigurableImpl.cpp.h"

#ifdef _DEBUG

#ifdef MD_WIN32_PLATFORM
#define MD_RENDERLIB9 L"D3D9Drv_DEBUG_32"
#define MD_RENDERLIB10 L"D3D10Drv_DEBUG_32"
#elif defined( MD_WIN64_PLATFORM )
#define MD_RENDERLIB9 L"D3D9Drv_DEBUG"
#define MD_RENDERLIB10 L"D3D10Drv_DEBUG"
#else
#error Unsupported platform!
#endif

#else

#ifdef MD_WIN32_PLATFORM
#define MD_RENDERLIB9 L"D3D9Drv_32"
#define MD_RENDERLIB10 L"D3D10Drv_32"
#elif defined( MD_WIN64_PLATFORM )
#define MD_RENDERLIB9 L"D3D9Drv"
#define MD_RENDERLIB10 L"D3D10Drv"
#else
#error Unsupported platform!
#endif

#endif


namespace Mod
{
	RenderDriverFunctions::RenderDriverFunctions( const RenderDriverFunctionsConfig& cfg ) :
	Parent( cfg ),
	mCreateDevice( NULL ),
	mCreateEffectProvider( NULL ),
	mCreateEffectPoolProvider( NULL )
	{
		DynamicLibraryConfig dlcfg;
		if( cfg.useD3D10 )
		{
			dlcfg.name = MD_RENDERLIB10;
		}
		else
		{
			dlcfg.name = MD_RENDERLIB9;
		}

		mRenderDriver = System::Single().CreateDynamicLibrary( dlcfg );

		mCreateDevice				= (CreateDeviceFunc)				mRenderDriver->GetProcAddress( "CreateDevice" );
		mCreateEffectProvider		= (CreateEffectProviderFunc)		mRenderDriver->GetProcAddress( "CreateEffectProvider" );
		mCreateEffectPoolProvider	= (CreateEffectPoolProviderFunc)	mRenderDriver->GetProcAddress( "CreateEffectPoolProvider" );

		MD_FERROR_ON_FALSE( mCreateDevice				);
		MD_FERROR_ON_FALSE( mCreateEffectProvider		);
		MD_FERROR_ON_FALSE( mCreateEffectPoolProvider	);

	}

	//------------------------------------------------------------------------

	RenderDriverFunctions::~RenderDriverFunctions() 
	{
	}

	//------------------------------------------------------------------------

	DevicePtr
	RenderDriverFunctions::CreateDevice( const DeviceConfig& cfg ) const
	{
		return mCreateDevice( cfg );
	}

	//------------------------------------------------------------------------

	EffectProviderPtr
	RenderDriverFunctions::CreateEffectProvider( const EffectProviderConfig& cfg ) const
	{
		return mCreateEffectProvider( cfg );
	}

	//------------------------------------------------------------------------

	EffectPoolProviderPtr
	RenderDriverFunctions::CreateEffectPoolProvider( const EffectPoolProviderConfig& cfg ) const
	{
		return mCreateEffectPoolProvider( cfg );
	}

	//------------------------------------------------------------------------

	DynamicLibraryPtr
	RenderDriverFunctions::GetDriverLib() const
	{
		return mRenderDriver;
	}

}