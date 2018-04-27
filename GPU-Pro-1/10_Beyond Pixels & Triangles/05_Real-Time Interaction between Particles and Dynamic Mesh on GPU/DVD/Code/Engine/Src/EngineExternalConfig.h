#ifndef ENGINE_ENGINEEXTERNALCONFIG_H_INCLUDED
#define ENGINE_ENGINEEXTERNALCONFIG_H_INCLUDED

#include "Common/Src/Forw.h"
#include "Common/Src/XIElemAttribute.h"

#include "ExportDefs.h"

namespace Mod
{
	struct EngineExternalConfig
	{
		EXP_IMP EngineExternalConfig();
		EXP_IMP EngineExternalConfig( const XMLElemPtr& el );
		EXP_IMP ~EngineExternalConfig();

		EXP_IMP void UpdateXMLDoc( const XMLDocPtr& doc );

		static const UINT32 MEMBERCOUNT_START = __LINE__;
		// -= Reflect extra lines in MEMBER_COUNT '-' correction =-
		UINT32 mainWindowWidth;
		UINT32 mainWindowHeight;
		UINT32 fullScreen;

		String sharedMediaPath;
		String projectMediaPath;
		// -= Reflect extra lines in MEMBER_COUNT '-' correction =-
		static const UINT32 MEMBER_COUNT = __LINE__ - MEMBERCOUNT_START - 4;
	};

}

#endif