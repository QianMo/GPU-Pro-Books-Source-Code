#ifndef ENGINE_ENGINE_H_INCLUDED
#define ENGINE_ENGINE_H_INCLUDED

#include "Common/Src/Forw.h"
#include "WrapSys/Src/Forw.h"
#include "Wrap3D/Src/Forw.h"
#include "Providers/Src/Forw.h"
#include "SceneRender/Src/Forw.h"

#include "Forw.h"

#include "ExportDefs.h"

#define MD_NAMESPACE EngineNS
#include "Singleton.h"
#define MD_NAMESPACE EngineNS
#include "ConfigurableImpl.h"


namespace Mod
{

	class Engine :	public EngineNS::Singleton< Engine >,
					public EngineNS::ConfigurableImpl< EngineConfig >
					
	{
		// types
	public:
		struct W3DShutDowner
		{
			~W3DShutDowner();
		};

		// construction/ destruction
	public:
		EXP_IMP	explicit Engine( const EngineConfig& cfg );
		EXP_IMP	~Engine();

		// manipulation/ access
	public:
		EXP_IMP	bool				Update();
		EXP_IMP WindowPtr			GetMainWindow() const;
		EXP_IMP DevicePtr			GetDevice() const;

		EXP_IMP ConstExprParserPtr	GetConstExprParser() const;

		EXP_IMP VFontDefProviderPtr	GetVFontDefProvider() const;

		EXP_IMP XMLDocPtr			CreateSettingsDoc() const;

		// data (note : ORDER matters here, Device must go first )
	private:
		// force call graphlib shutdown last
		RenderDriverFunctionsPtr	mRenderDriverFunctions;
		W3DShutDowner				mW3DShutDowner;

		DevicePtr				mDevice;

		WindowPtr				mMainWindow;
		EffectProviderPtr		mEffectProvider;
		EffectPoolProviderPtr	mEffectPoolProvider;

		EffectDefProviderPtr				mEffectDefProvider;

		VFontProviderPtr					mVFontProvider;
		VFontDefProviderPtr					mVFontDefProvider;

		ModelProviderPtr					mModelProvider;
		ModelCmptDefProviderPtr				mModelCmptDefProvider;

		ModelCmptRawDataProviderPtr			mModelCmptRawDataProvider;

		ConstExprParserPtr					mConstExprParser;
		EngineSettingsPtr					mSettings;
	};

	struct EngineDestroyer
	{
		EXP_IMP ~EngineDestroyer();
	};

	EXP_IMP void SaveEngineConfig();

}

#endif