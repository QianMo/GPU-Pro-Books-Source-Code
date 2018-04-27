#ifndef ENGINE_ENGINECONFIG_H_INCLUDED
#define ENGINE_ENGINECONFIG_H_INCLUDED

#include "WrapSys/Src/SystemConfig.h"
#include "Wrap3D/Src/DeviceConfig.h"
#include "Providers/Src/Forw.h"

#include "SceneRender/Src/Forw.h"

namespace Mod
{	
	struct EngineConfig
	{
		EngineConfig();

		bool							showConfigWindow;
		bool							allowFullScreenChange;

		SystemConfig					systemConfig;
		
		// width/height can be modified if config file is provided
		mutable WindowConfigPtr			mainWindowConfig;

		DeviceConfig					deviceConfig;
		String							cfgFilePath;
		mutable String					sharedMediaPath;
		mutable String					mediaPath;
		EffectSubVariationMapPtr		effSubVarMap;

		EffectParamsDataCreator			effectParamsDataCreator;
		EffectSubVariationMapCreator	effectSubVariationMapCreator;

		void (*externalInit)();
	};

	//------------------------------------------------------------------------

	inline 
	EngineConfig::EngineConfig() :
	externalInit( NULL ),
	effectParamsDataCreator( NULL ),
	effectSubVariationMapCreator( NULL ),
	showConfigWindow( true ),
	allowFullScreenChange( true )
	{

	}

}

#endif