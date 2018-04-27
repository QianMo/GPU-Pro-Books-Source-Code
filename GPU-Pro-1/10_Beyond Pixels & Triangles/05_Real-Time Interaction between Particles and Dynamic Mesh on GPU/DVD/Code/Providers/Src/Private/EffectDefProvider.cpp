#include "Precompiled.h"

#include "Common/Src/XMLDoc.h"
#include "Common/Src/XMLDocConfig.h"
#include "Common/Src/XIElemArray.h"

#include "VertexCmptDef.h"

#include "EffectDefConfig.h"
#include "EffectDef.h"

#include "EffectDefProviderConfig.h"
#include "EffectDefProvider.h"

#define MD_NAMESPACE EffectDefProviderNS
#include "ConfigurableImpl.cpp.h"

namespace Mod
{

	//------------------------------------------------------------------------

	template class EffectDefProviderNS::ConfigurableImpl<EffectDefProviderConfig>;

	//------------------------------------------------------------------------

	EXP_IMP
	EffectDefProvider::EffectDefProvider( const EffectDefProviderConfig& cfg ) : 
	Parent( cfg )
	{
		struct Convert
		{
			EffectDefPtr operator () ( const XMLElemPtr& el ) const
			{
				EffectDefConfig cfg;
				cfg.xmlElem = el;

				return EffectDefPtr( new EffectDef( cfg ) );
			}

		} convert;

		AddItemsFromXMLDoc( cfg.docBytes, convert );

		CheckConsistensy();
	}

	//------------------------------------------------------------------------

	EXP_IMP
	EffectDefProvider::~EffectDefProvider()
	{

	}

	//------------------------------------------------------------------------

	void
	EffectDefProvider::CheckConsistensy()
	{
		struct
		{
			void operator() ( const EffectDefPtr& effDef ) const
			{
				const EffectDef::VertexCmpts& output = effDef->GetOutputVertexCmpts();
				const EffectDef::VertexCmpts& input = effDef->GetRequiredVertexCmpts();

				if( !output.empty() )
				{
					for( size_t i = 0, e = input.size(); i < e; i ++ )
					{
						const VertexCmpt& in = input[i];

						// every transformable element must be output from transform stage
						if( in.def->GetType() == VCT::TRANSFORMABLE )
						{
							size_t i = 0, e = output.size();
							for( ; i < e; i ++ )
							{
								const VertexCmpt& o = output[ i ];
								if( in.def->GetFactory() == o.def->GetFactory() && in.idx == o.idx )
									break;
							}
							MD_FERROR_ON_FALSE( i < e );
						}
					}
				}
			}

			EffectDefProvider *This;
		} checker = { this };

		ForConstEach( checker );
	}


}