#include "Precompiled.h"

#include "Common/Src/XMLAttribConfig.h"
#include "Common/Src/XMLDoc.h"

#include "Common/Src/XMLElemConfig.h"
#include "Common/Src/XMLElem.h"

#include "Common/Src/StringUtils.h"

#include "EngineExternalConfig.h"

namespace Mod
{
	namespace
	{
		// assist help us (CN)
		namespace CN
		{
			const WCHAR* RESOLUTION			= L"resolution";
			const WCHAR* RESOLUTION_X		= L"x";
			const WCHAR* RESOLUTION_Y		= L"y";
			const WCHAR* FULL_SCREEN		= L"full_screen";
			const WCHAR* MEDIA_PATH			= L"media_path";
			const WCHAR* SHARED_MEDIA_PATH	= L"shared_media_path";
			const WCHAR* VAL				= L"val";
		}
	}

	//------------------------------------------------------------------------

	EXP_IMP
	EngineExternalConfig::EngineExternalConfig() :
	mainWindowWidth(	1024 ),
	mainWindowHeight(	768 ),
	fullScreen(			0 ),
	sharedMediaPath(	L"..\\Media\\Shared\\" ),
	projectMediaPath(	L"..\\Media\\Project\\")
	{

	}

	//------------------------------------------------------------------------

	EXP_IMP
	EngineExternalConfig::EngineExternalConfig( const XMLElemPtr& el ) : 
	mainWindowWidth(	XIUInt(		el, CN::RESOLUTION, CN::RESOLUTION_X )		),
	mainWindowHeight(	XIUInt(		el, CN::RESOLUTION, CN::RESOLUTION_Y )		),
	fullScreen(			XIUInt(		el, CN::RESOLUTION, CN::FULL_SCREEN )		),
	sharedMediaPath(	XIString(	el, CN::SHARED_MEDIA_PATH, CN::VAL, L"" )	),
	projectMediaPath(	XIString(	el, CN::MEDIA_PATH, CN::VAL, L"" )			)
	{

	}

	//------------------------------------------------------------------------

	EXP_IMP
	EngineExternalConfig::~EngineExternalConfig()
	{

	}

	//------------------------------------------------------------------------

	EXP_IMP
	void
	EngineExternalConfig::UpdateXMLDoc( const XMLDocPtr& doc )
	{
		const XMLElemPtr& elem = doc->GetRoot();


		XMLElemConfig elcfg;
		XMLAttribConfig atcfg;

		// resolution
		{
			elcfg.name = CN::RESOLUTION;
			
			XMLElemPtr resolutionElem = elem->GetChild( elcfg.name );

			if( !resolutionElem )
				resolutionElem = elem->AddChild( elcfg, 0 );

			atcfg.name	= CN::RESOLUTION_X;
			atcfg.value	= AsString( mainWindowWidth );
			resolutionElem->SetAttrib( atcfg );

			atcfg.name	= CN::RESOLUTION_Y;
			atcfg.value	= AsString( mainWindowHeight );
			resolutionElem->SetAttrib( atcfg );

			atcfg.name	= CN::FULL_SCREEN;
			atcfg.value	= AsString( fullScreen );
			resolutionElem->SetAttrib( atcfg );
		}

		// shared media path
		{
			elcfg.name = CN::MEDIA_PATH;			

			XMLElemPtr mediaPathElem = elem->GetChild( elcfg.name );
			if( !mediaPathElem )
			{
				mediaPathElem = elem->AddChild( elcfg, 0 );
			}

			atcfg.name		= CN::VAL;
			atcfg.value		= projectMediaPath;
			mediaPathElem->SetAttrib( atcfg );
		}

		// project media path
		{
			elcfg.name	= CN::SHARED_MEDIA_PATH;
			
			XMLElemPtr sharedPathElem = elem->GetChild( elcfg.name );

			if( !sharedPathElem )
			{
				sharedPathElem = elem->AddChild( elcfg, 0 );
			}

			atcfg.name		= CN::VAL;
			atcfg.value		= sharedMediaPath;

			sharedPathElem->SetAttrib( atcfg );
		}
	}

}