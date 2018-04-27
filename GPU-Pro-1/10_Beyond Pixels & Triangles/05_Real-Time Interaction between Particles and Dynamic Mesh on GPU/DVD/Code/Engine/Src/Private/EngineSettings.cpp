#include "Precompiled.h"

#include "Common/Src/XMLDoc.h"

#include "Common/Src/XMLElemConfig.h"
#include "Common/Src/XMLElem.h"

#include "Common/Src/XMLAttribConfig.h"
#include "Common/Src/XMLAttrib.h"

#include "Common/Src/XMLDoc.h"
#include "Common/Src/XIElemAttribute.h"

#include "WrapSys/Src/System.h"

#include "EngineSettingsConfig.h"
#include "EngineSettings.h"

#define MD_NAMESPACE EngineSettingsNS
#include "ConfigurableImpl.cpp.h"

namespace Mod
{
	EngineSettings::Data::Data():
	forceModelCache( true ),
	forceEffectCache( true ),
	useDX10( true )
	{

	}

	//------------------------------------------------------------------------

	namespace
	{
		namespace CN
		{
			const WCHAR* FORCE_MODEL_CACHE	= L"force_model_cache";
			const WCHAR* FORCE_EFFECT_CACHE	= L"force_effect_cache";
			const WCHAR* USE_DX10			= L"use_dx10";
			const WCHAR* VAL				= L"val";
		}
	}

	EngineSettings::EngineSettings( const EngineSettingsConfig& cfg ) :
	Parent( cfg )
	{
		if( System::Single().FileExists( cfg.configFile ) )
		{
			XMLDocPtr doc = CreateXMLDocFromFile( cfg.configFile );

			const XMLElemPtr& elem = doc->GetRoot();
			
			mData.forceModelCache	= XIUInt( elem, CN::FORCE_MODEL_CACHE, CN::VAL, 1u ) ? true : false;
			mData.forceEffectCache	= XIUInt( elem, CN::FORCE_EFFECT_CACHE, CN::VAL, 1u ) ? true : false;
			mData.useDX10			= XIUInt( elem, CN::USE_DX10, CN::VAL, 1u ) ? true : false;
		}
	}

	//------------------------------------------------------------------------

	EngineSettings::~EngineSettings() 
	{
	}

	//------------------------------------------------------------------------

	const EngineSettings::Data&
	EngineSettings::Get() const
	{
		return mData;
	}

	//------------------------------------------------------------------------

	void
	EngineSettings::AppendToXMLDoc( const XMLDocPtr& doc )
	{
		const XMLElemPtr& root = doc->GetRoot();

		XMLElemConfig	elcfg;
		XMLAttribConfig	atcfg;

		atcfg.name		= CN::VAL;

		// force model cache
		{
			elcfg.name	= CN::FORCE_MODEL_CACHE;

			XMLElemPtr elem = root->AddChild( elcfg, 0 );
			
			atcfg.value		= mData.forceModelCache ? L"1" : L"0";
			elem->SetAttrib( atcfg );
		}

		// force effect cace
		{
			elcfg.name	= CN::FORCE_EFFECT_CACHE;

			XMLElemPtr elem = root->AddChild( elcfg, 0 );

			atcfg.value		= mData.forceEffectCache ? L"1" : L"0";
			elem->SetAttrib( atcfg );
		}

		// dx10
		{
			elcfg.name	= CN::USE_DX10;

			XMLElemPtr elem = root->AddChild( elcfg, 0 );

			atcfg.value		= mData.useDX10 ? L"1" : L"0";
			elem->SetAttrib( atcfg );			
		}
	
	}
}