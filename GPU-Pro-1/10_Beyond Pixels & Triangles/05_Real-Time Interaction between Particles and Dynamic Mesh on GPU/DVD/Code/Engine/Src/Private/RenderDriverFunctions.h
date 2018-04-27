#ifndef ENGINE_RENDERDRIVERFUNCTIONS_H_INCLUDED
#define ENGINE_RENDERDRIVERFUNCTIONS_H_INCLUDED

#include "WrapSys/Src/Forw.h"
#include "Providers/Src/Forw.h"
#include "Wrap3D/Src/Forw.h"

#include "Forw.h"

#include "BlankExpImp.h"

#define MD_NAMESPACE RenderDriverFunctionsNS
#include "ConfigurableImpl.h"

namespace Mod
{

	class RenderDriverFunctions : public RenderDriverFunctionsNS::ConfigurableImpl<RenderDriverFunctionsConfig>
	{
		// types
	public:
		typedef DevicePtr				(*CreateDeviceFunc)( const DeviceConfig& );
		typedef EffectProviderPtr		(*CreateEffectProviderFunc)( const EffectProviderConfig& );
		typedef EffectPoolProviderPtr	(*CreateEffectPoolProviderFunc)( const EffectPoolProviderConfig& );

		// constructors / destructors
	public:
		explicit RenderDriverFunctions( const RenderDriverFunctionsConfig& cfg );
		~RenderDriverFunctions();
	
		// manipulation/ access
	public:
		DevicePtr				CreateDevice( const DeviceConfig& cfg ) const;
		EffectProviderPtr		CreateEffectProvider( const EffectProviderConfig& cfg ) const;
		EffectPoolProviderPtr	CreateEffectPoolProvider( const EffectPoolProviderConfig& cfg ) const;

		DynamicLibraryPtr		GetDriverLib() const;

		// data
	private:
		DynamicLibraryPtr	mRenderDriver;

		CreateDeviceFunc				mCreateDevice;
		CreateEffectProviderFunc		mCreateEffectProvider;
		CreateEffectPoolProviderFunc	mCreateEffectPoolProvider;
	};
}

#endif