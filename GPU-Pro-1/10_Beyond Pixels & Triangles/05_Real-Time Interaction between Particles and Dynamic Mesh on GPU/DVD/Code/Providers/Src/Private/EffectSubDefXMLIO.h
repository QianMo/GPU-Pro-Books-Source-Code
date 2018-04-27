#ifndef PROVIDERS_EFFECTSUBDEFXMLIO_H_INCLUDED
#define PROVIDERS_EFFECTSUBDEFXMLIO_H_INCLUDED

namespace Mod
{
	namespace
	{
		struct TypedSubdef
		{
			ESDT::EffectSubDefType	type;
			EffectSubDefPtr			effectSubDef;
		};

		struct ConvertToTypedSubdef
		{
			TypedSubdef operator() ( const XMLElemPtr& elem ) const
			{
				TypedSubdef res;

				EffectSubDefConfig esdcfg;

				esdcfg.xmlElem = elem;

				res.effectSubDef.reset( new EffectSubDef( esdcfg ) );
				
				XIAttString type( elem, L"type" );

				if( type == L"Default" )
				{
					res.type	= ESDT::DEFAULT;
				}
				else
				if( type == L"Transform" )
				{
					res.type	= ESDT::TRANSFORM;
				}
				else
				if( type == L"PostTransform" )
				{
					res.type	= ESDT::POST_TRANSFORM;
				}
				else
					MD_FERROR( L"Unsupported effect subdef!" );

				return res;
			}
		}; 
	}
}

#endif