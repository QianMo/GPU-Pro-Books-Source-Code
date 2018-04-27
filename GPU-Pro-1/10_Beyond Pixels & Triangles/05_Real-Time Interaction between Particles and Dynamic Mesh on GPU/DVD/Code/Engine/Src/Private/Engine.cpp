#include "Precompiled.h"

#if MD_WIN_PLATFORM
#include "SysWinDrv/Src/Exports.h"
#else
#error this platform is not implemented yet
#endif

#include "Common/Src/XMLDocConfig.h"
#include "Common/Src/XMLDoc.h"
#include "Common/Src/XMLElemConfig.h"
#include "Common/Src/XMLElem.h"

#include "Common/Src/FileHelpers.h"

#include "WrapSys/Src/Exports.h"
#include "WrapSys/Src/System.h"
#include "WrapSys/Src/Window.h"
#include "WrapSys/Src/ConfigurationDialogOptions.h"

#include "Wrap3D/Src/Exports.h"

#include "Providers/Src/TextureProviderConfig.h"
#include "Providers/Src/TextureProvider.h"

#include "Providers/Src/EffectProviderConfig.h"
#include "Providers/Src/EffectProvider.h"
#include "Providers/Src/EffectPoolProviderConfig.h"

#include "Providers/Src/VertexCmptFactoryProviderConfig.h"
#include "Providers/Src/VertexCmptFactoryProvider.h"

#include "Providers/Src/VertexCmptDefProviderConfig.h"
#include "Providers/Src/VertexCmptDefProvider.h"

#include "Providers/Src/EffectDefProviderConfig.h"
#include "Providers/Src/EffectDefProvider.h"

#include "Providers/Src/VFontProviderConfig.h"
#include "Providers/Src/VFontProvider.h"

#include "Providers/Src/VFontDefProviderConfig.h"
#include "Providers/Src/VFontDefProvider.h"

#include "Providers/Src/ModelCmptDefProviderConfig.h"
#include "Providers/Src/ModelCmptDefProvider.h"

#include "Providers/Src/ModelCmptRawDataImporterProviderConfig.h"
#include "Providers/Src/ModelCmptRawDataImporterProvider.h"

#include "Providers/Src/ModelCmptRawDataProviderConfig.h"
#include "Providers/Src/ModelCmptRawDataProvider.h"

#include "Providers/Src/ModelProviderConfig.h"
#include "Providers/Src/ModelProvider.h"

#include "Providers/Src/ConstExprParserConfig.h"
#include "Providers/Src/ConstExprParser.h"

#include "Providers/Src/ConstEvalOperationGroupProviderConfig.h"
#include "Providers/Src/ConstEvalOperationGroupProvider.h"

#include "Providers/Src/EffectSubVariationMapConfig.h"

#include "Providers/Src/ProvidersConfig.h"
#include "Providers/Src/Providers.h"

#include "RenderLib/Src/DefaultSceneEffectSubVariationMap.h"

#include "RenderDriverFunctionsConfig.h"
#include "RenderDriverFunctions.h"

#include "EngineSettingsConfig.h"
#include "EngineSettings.h"

#include "EngineExternalConfig.h"
#include "Exports.h"

#include "PlatformWindowConfig.h"

#include "EngineConfig.h"

// must come last to flush EXP_IMP defines...
#include "Engine.h"

#define MD_NAMESPACE EngineNS
#include "ConfigurableImpl.cpp.h"
#define MD_NAMESPACE EngineNS
#include "Singleton.cpp.h"

namespace Mod
{

	template class EngineNS::Singleton< Engine >;
	template class EngineNS::ConfigurableImpl< EngineConfig >;

	namespace
	{
		// mutable state is modified
		void ModifyConfig( const EngineConfig& cfg );
	}

	EXP_IMP
	Engine::Engine( const EngineConfig& cfg ) try : 
	Parent( cfg )
	{
		EngineSettingsConfig scfg;
		scfg.configFile = cfg.cfgFilePath;

		mSettings.reset( new EngineSettings( scfg ) );

		mMainWindow = System::Single().CreateWindow( *cfg.mainWindowConfig );


		RenderDriverFunctionsConfig rdfcfg;
		rdfcfg.useD3D10 = mSettings->Get().useDX10;

		mRenderDriverFunctions.reset( new RenderDriverFunctions( rdfcfg ) );

#if MD_WIN_PLATFORM
		// device
		{
			DeviceConfig modifiedDevCfg		= GetConfig().deviceConfig;
			modifiedDevCfg.targetWindow		= mMainWindow.get();			

			mDevice			= mRenderDriverFunctions->CreateDevice( modifiedDevCfg );
		}

		// effect and effect pool providers
		{
			EffectProviderConfigBase baseCfg;

			baseCfg.autoCreateCache = true;
			baseCfg.dev = mDevice;

			baseCfg.path				= cfg.mediaPath + L"Effects\\";
			baseCfg.cachePath			= cfg.mediaPath + L"Cache\\Effects\\";
			baseCfg.extension			= L".fx";
			baseCfg.cachedExtension		= L".cfx";
			baseCfg.forceCache			= mSettings->Get().forceEffectCache;

			{
				EffectProviderConfig epcfg;
				static_cast<EffectProviderConfigBase&>(epcfg) = baseCfg;
				mEffectProvider = mRenderDriverFunctions->CreateEffectProvider( epcfg );
			}

			{
				EffectPoolProviderConfig eppcfg;
				static_cast<EffectProviderConfigBase&>(eppcfg) = baseCfg;
				eppcfg.extension		= L".fxh";
				eppcfg.path				+= L"Inc\\";
				eppcfg.cachePath		+= L"Inc\\";
				mEffectPoolProvider = mRenderDriverFunctions->CreateEffectPoolProvider( eppcfg );
			}
		}
#endif

		// provider couple ( gradually new providers are set )
		{
			ProvidersConfig pcfg;

			pcfg.mediaPath				= cfg.mediaPath;
			pcfg.sharedMediaPath		= cfg.sharedMediaPath;

			pcfg.dev					= mDevice;

			const EffectProviderConfig& epcfg = mEffectProvider->GetConfig();

			pcfg.effectIncludePath		= epcfg.path;
			pcfg.effectVariationsFile	= cfg.mediaPath + L"XMLDefs\\EffectVariations.xml";
			pcfg.effectCachePath		= epcfg.cachePath;

			new Providers( pcfg );

			Providers::Single().SetEffectParamsDataCreator( cfg.effectParamsDataCreator );
		}

		// update providers couple
		{
			Providers::Single().SetEffectProv( mEffectProvider );
			Providers::Single().SetEffectPoolProv( mEffectPoolProvider );
		}

		// model raw data provider
		{
			ModelCmptRawDataProviderConfig mcpcfg;
			mcpcfg.path	= cfg.mediaPath + L"Models\\";

			mModelCmptRawDataProvider.reset( new ModelCmptRawDataProvider( mcpcfg ) );

			// update providers couple
			Providers::Single().SetModelCmptRawDataProv( mModelCmptRawDataProvider );
		}

		// vertex component factory provider
		{
			Providers::Single().CreateVertexCmptFactoryProv();
		}

		// vertex component def provider
		{
			VertexCmptDefProviderConfig vcdcfg;
			vcdcfg.dev			= mDevice;
			ReadBytes( cfg.sharedMediaPath + L"XMLDefs\\VertexCmptsDefs.xml", vcdcfg.docBytes );

			VertexCmptDefProviderPtr vcdefsProv( new VertexCmptDefProvider( vcdcfg ) );

			// update providers couple
			Providers::Single().SetVertexCmptDefProv( vcdefsProv );
		}

		// effect subvar map
		{
			EffectSubVariationMapPtr& effSubVarMap = config().effSubVarMap;
			if( !effSubVarMap && cfg.effectSubVariationMapCreator )
			{
				effSubVarMap = cfg.effectSubVariationMapCreator( EffectSubVariationMapConfig() );
			}

			Providers::Single().SetEffectSubVariationMap( effSubVarMap );
		}

		// effect def provider
		{
			EffectDefProviderConfig edpcfg;
			ReadBytes( cfg.mediaPath + L"XMLDefs\\EffectDefs.xml", edpcfg.docBytes );
			mEffectDefProvider.reset( new EffectDefProvider(edpcfg) );

			// update providers couple
			Providers::Single().SetEffectDefProv( mEffectDefProvider );
		}

		// model def provider
		{
			ModelCmptDefProviderConfig mcdcfg;
			mcdcfg.dev				= mDevice;
			mcdcfg.path				= cfg.mediaPath + L"XMLDefs\\ModelCmptDefs\\";

			mModelCmptDefProvider.reset( new ModelCmptDefProvider( mcdcfg ) );

			// update providers couple
			Providers::Single().SetModelCmptDefProv( mModelCmptDefProvider );
		}

		// model provider
		{
			ModelProviderConfig mpcfg;
			mpcfg.dev				= mDevice;
			mpcfg.path				= cfg.mediaPath + L"XMLDefs\\Models\\";
			mpcfg.fontDefProvider	= mVFontDefProvider;
			mpcfg.forceCache		= mSettings->Get().forceModelCache;

			mModelProvider.reset( new ModelProvider( mpcfg ) );

			// update providers couple
			Providers::Single().SetModelProv( mModelProvider );
		}

		// vfont provider
		{
			VFontProviderConfig vfcfg;

			vfcfg.dev			= mDevice;
			vfcfg.path			= cfg.mediaPath + L"Fonts\\";

			mVFontProvider.reset( new VFontProvider(vfcfg) );
		}

		// vfontdef provider
		{
			VFontDefProviderConfig vfcfg;
			vfcfg.path			= cfg.mediaPath + L"XMLDefs\\Fonts\\";
			vfcfg.fontProv		= mVFontProvider;
			vfcfg.textureProv	= Providers::Single().GetTextureProv();

			mVFontDefProvider.reset( new VFontDefProvider( vfcfg ) );
		}

		// providers couple postinit
		{
			Providers::Single().PostInit();
		}

		// const expr parser
		{
			
			ConstEvalOperationGroupProviderConfig ogpcfg;

			ConstExprParserConfig cfg;
			cfg.operationGroupProv.reset( new ConstEvalOperationGroupProvider(ogpcfg) );

			mConstExprParser.reset( new ConstExprParser( cfg ) );
		}

		if( cfg.externalInit )
		{
			cfg.externalInit();
		}
		
		mMainWindow->SetVisible(true);
	}
	catch( Exception& e )
	{
		SysShowMessage( L"Failed to initialize the engine because of the exception.\n Exception description:\n\""
						+ e.GetReason() + L"\"");
		throw;
	}

	//------------------------------------------------------------------------

	Engine::W3DShutDowner::~W3DShutDowner()
	{
		W3DShutDown();
	}

	EXP_IMP
	Engine::~Engine()
	{
		delete &Providers::Single();
	}

	//------------------------------------------------------------------------

	EXP_IMP
	bool
	Engine::Update()
	{
		return mMainWindow->Update();
	}

	//------------------------------------------------------------------------

	EXP_IMP
	WindowPtr
	Engine::GetMainWindow() const
	{
		return mMainWindow;
	}

	//------------------------------------------------------------------------

	EXP_IMP
	DevicePtr
	Engine::GetDevice() const
	{
		return mDevice;
	}

	//------------------------------------------------------------------------

	EXP_IMP
	ConstExprParserPtr
	Engine::GetConstExprParser() const
	{
		return mConstExprParser;
	}

	//------------------------------------------------------------------------

	EXP_IMP
	VFontDefProviderPtr
	Engine::GetVFontDefProvider() const
	{
		return mVFontDefProvider;
	}
	
	//------------------------------------------------------------------------

	EXP_IMP
	XMLDocPtr
	Engine::CreateSettingsDoc() const
	{
		XMLDocConfig dcfg;
		XMLDocPtr result( new XMLDoc( dcfg ) );

		// root
		{
			XMLElemConfig elcfg;
			elcfg.name = L"root";

			result->AddElement( elcfg, 0 );
		}

		mSettings->AppendToXMLDoc( result );

		return result;
	}

	//------------------------------------------------------------------------

	EXP_IMP
	EngineDestroyer::~EngineDestroyer()
	{
		DestroyEngine();
	}

	//------------------------------------------------------------------------

	EXP_IMP void SaveEngineConfig()
	{

		XMLDocPtr doc = Engine::Single().CreateSettingsDoc();

		const EngineConfig& cfg = Engine::Single().GetConfig();
		EngineExternalConfig ecfg;

		const int MEMBER_COUNT_START = __LINE__;
		ecfg.fullScreen			= cfg.deviceConfig.isFullScreen;
		ecfg.mainWindowWidth	= cfg.mainWindowConfig->width;
		ecfg.mainWindowHeight	= cfg.mainWindowConfig->height;
		ecfg.projectMediaPath	= cfg.mediaPath;
		ecfg.sharedMediaPath	= cfg.sharedMediaPath;
		const int MEMBER_COUNT_END = __LINE__ - 1;
		MD_STATIC_ASSERT( MEMBER_COUNT_END - MEMBER_COUNT_START == EngineExternalConfig::MEMBER_COUNT );

		ecfg.UpdateXMLDoc( doc );

		SaveXMLDocToFile( doc, cfg.cfgFilePath );
	}

	//------------------------------------------------------------------------

	EXP_IMP
	bool CreateEngine( const EngineConfig& cfg, void (*UserInit)() /*= NULL*/ )
	{
		CreateSystem( cfg.systemConfig );
		
		if( UserInit )
			UserInit();

		ModifyConfig( cfg );

		// they force to load a lib with mode functions
		RenderDriverFunctionsPtr d3d9functions;
		{
			RenderDriverFunctionsConfig rdcfg;
			rdcfg.useD3D10 = false;

			d3d9functions.reset( new RenderDriverFunctions( rdcfg ) );
		}

		ConfigurationDialogOptions opts;
		opts.fullScreen				= cfg.deviceConfig.isFullScreen;
		opts.allowFullScreenChange	= cfg.allowFullScreenChange;

		const WindowConfig& wcfg = *cfg.mainWindowConfig;

		opts.resolutionIndex	= W3DGetModeIndex( wcfg.width, wcfg.height );
		opts.resolutionOptions	= W3DGetAvailableModes();

		if( cfg.showConfigWindow )
		{
			SysConfigure( opts );
		}

		// do not allow windows larger than desktop ( actually, windows doesn't allow it )
		if( !opts.fullScreen )
		{
			UINT32 deskWi, deskHe;
			System::Single().GetDesktopDimmensions( deskWi, deskHe );

			UINT32 moWi, moHe;
			W3DGetModeByIndex( opts.resolutionIndex, moWi, moHe );

			if( deskWi < moWi || deskHe < moHe )
			{
				opts.resolutionIndex = W3DGetModeIndex( deskWi, deskHe );
			}
		}

		if( !opts.proceed )
		{
			W3DShutDown();
			return false;
		}
		else
		{
			W3DGetModeByIndex(	opts.resolutionIndex, 
								cfg.mainWindowConfig->width, 
								cfg.mainWindowConfig->height );

			cfg.deviceConfig.isFullScreen = opts.fullScreen;

			d3d9functions.reset();

			new Engine( cfg );
			return true;
		}
	}

	//------------------------------------------------------------------------

	EXP_IMP void DestroyEngine()
	{
		delete &Engine::Single();
	}

	//------------------------------------------------------------------------

	namespace
	{
		void ModifyConfig( const EngineConfig& cfg )
		{
			if( !cfg.cfgFilePath.empty() )
			{
				XMLDocPtr cdoc = CreateXMLDocFromFile( cfg.cfgFilePath );
				EngineExternalConfig ecfg( cdoc->GetRoot() );

				cfg.mainWindowConfig->width		= ecfg.mainWindowWidth;
				cfg.mainWindowConfig->height	= ecfg.mainWindowHeight;
				cfg.deviceConfig.isFullScreen	= (UINT32)ecfg.fullScreen ? true : false;

				if( !ecfg.projectMediaPath.empty() )
					cfg.mediaPath = ecfg.projectMediaPath;

				if( !ecfg.sharedMediaPath.empty() )
					cfg.sharedMediaPath = ecfg.sharedMediaPath;
			}
		}
	}

	//------------------------------------------------------------------------
	

}