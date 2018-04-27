#ifndef ENGINE_ENGINESETTINGS_H_INCLUDED
#define ENGINE_ENGINESETTINGS_H_INCLUDED

#include "Common/Src/Forw.h"

#include "Forw.h"

#include "ExportDefs.h"

#define MD_NAMESPACE EngineSettingsNS
#include "ConfigurableImpl.h"

namespace Mod
{

	class EngineSettings : public EngineSettingsNS::ConfigurableImpl<EngineSettingsConfig>
	{
		// types
	public:
		struct Data
		{
			Data();

			bool forceModelCache;
			bool forceEffectCache;
			bool useDX10;

		};

		// constructors / destructors
	public:
		explicit EngineSettings( const EngineSettingsConfig& cfg );
		~EngineSettings();
	
		// manipulation/ access
	public:
		const Data& Get() const;
		void AppendToXMLDoc( const XMLDocPtr& doc );

		// data
	private:
		Data mData;

	};
}

#endif