#include "Precompiled.h"

#include "Common/Src/XIElemAttribute.h"
#include "Common/Src/XIElemArray.h"

#include "EffectVariationConfig.h"
#include "EffectVariation.h"

#include "EffectSubDefConfig.h"
#include "EffectSubDef.h"

#define MD_NAMESPACE EffectVariationNS
#include "ConfigurableImpl.cpp.h"

#include "EffectSubDefXMLIO.h"

namespace Mod
{
	template class EffectVariationNS::ConfigurableImpl<EffectVariationConfig>;

	//------------------------------------------------------------------------

	namespace
	{
		class RenderMethodConverter
		{
			// types
		public:
			typedef Types2< String, RM::RenderMethod > :: Map Map;

			// construction/ destruction
		private:
			RenderMethodConverter();

			// manipulation/ access
		public:
			static RenderMethodConverter& Single();
			RM::RenderMethod GetRenderMethod( const String& rmName );

			// data
		private:
			Map mMap;

		};
	}

	//------------------------------------------------------------------------

	EXP_IMP
	EffectVariation::EffectVariation( const EffectVariationConfig& cfg ) : 
	Parent( cfg ),
	XINamed( cfg.xmlElem )
	{

		struct FillBranchData
		{
			explicit FillBranchData( EffectSubDefs* a_superDefaults ) : superDefaults( a_superDefaults ) {}

			void operator()( EffectSubDefs& data, const XMLElemPtr& elem ) const
			{
				XIElemArray< TypedSubdef, ConvertToTypedSubdef > subdefs( elem, L"SubDef", ConvertToTypedSubdef() );

				MD_FERROR_ON_FALSE( subdefs.size() <= data.size() );

				for( size_t i = 0, e = subdefs.size(); i < e; i ++ )
				{
					const TypedSubdef& ts = subdefs[ i ];
					data[ ts.type ] = ts.effectSubDef;
				}

				// spread defaults
				if( const EffectSubDefPtr& d = data[ ESDT::DEFAULT ] )
				{
					for( size_t i = 0, e = data.size(); i < e; i ++ )
					{
						if( superDefaults )
						{
							if( const EffectSubDefPtr& dest = data[i] )
							{
								if( (*superDefaults)[ i ] )
									EffectSubDefRestrictions::SetEffectSubDefDefaults( *dest, (*superDefaults)[i]->GetData() );
								else
								if( (*superDefaults)[ ESDT::DEFAULT ] )
									EffectSubDefRestrictions::SetEffectSubDefDefaults( *dest, (*superDefaults)[ ESDT::DEFAULT ]->GetData() );
							}
						}
							
						if( i == ESDT::DEFAULT )
							continue;

						if( const EffectSubDefPtr& dest = data[ i ] )
						{
							EffectSubDefRestrictions::SetEffectSubDefDefaults( *dest, d->GetData() );
						}
					}
				}

				// default buffers
				EffectSubDefRestrictions::SetEffectSubDefDefaultBuffers( data );
			}

			EffectSubDefs* superDefaults;
		};

		FillBranchData( NULL )( mDefaultSubDefs, cfg.xmlElem );

		{
			struct Convert
			{
				explicit Convert( EffectSubDefs* a_superDefaults ) : superDefaults( a_superDefaults ) {}

				Branch operator() ( const XMLElemPtr& elem ) const
				{
					Branch result;

					(FillBranchData( superDefaults )) ( result.subDefs, elem );

					struct Convert
					{
						Condition operator() ( const XMLElemPtr& elem ) const
						{
							struct Convert
							{
								ConditionEntry operator()(const XMLElemPtr& elem) const
								{
									ConditionEntry res;

									XIAttString type( elem, L"type" );

									if( type == L"file" )
										res.type = FILE_CONDITION;
									else
									if( type == L"define" )
										res.type = DEFINE_CONDITION;
									else
										MD_FERROR( L"Unsupported condition!" );

									res.val	= XIAttString(elem, L"val" );

									XIAttString subdef( elem, L"subdef" );

									if( subdef == L"Default" )
										res.subDef = ESDT::DEFAULT;
									else
									if( subdef == L"Transform" )
										res.subDef = ESDT::TRANSFORM;
									else
									if( subdef == L"PostTransform" )
										res.subDef = ESDT::POST_TRANSFORM;
									else
										MD_FERROR( L"Unsupported condition!" );

									return res;
								}
							} convert;
							
							XIElemArray< ConditionEntry, Convert > comps( elem, L"Condition", convert );

							return comps;
						}

					}convert;

					XIElemArray< Condition, Convert > comps( elem, L"Conditions", convert );
					result.conditions = comps;

					return result;
				}

				EffectSubDefs* superDefaults;
			}convert ( &mDefaultSubDefs );

			XIElemArray< Branch, Convert > comps( cfg.xmlElem, L"Branch", convert );

			mBranches = comps;
		}

		mSettings.renderMethod		= RenderMethodConverter::Single().GetRenderMethod( XIString( cfg.xmlElem, L"RenderMethod", L"val", L"Default" ) );
		mSettings.passCount			= XIUInt( cfg.xmlElem, L"PassCount", L"val", 1 );

		// get subvariation mask
		{
			mSettings.subVariationMask = 0xffffffff;
			if( const XMLElemPtr& elem = cfg.xmlElem->GetChild( L"ExcludeEffectSubvars" ) )
			{
				XIStrElemArray masks( elem );

				// not supported yet
				MD_FERROR_ON_FALSE( masks.size() == 1 );

				if( masks[0] == L":ALL" )
					mSettings.subVariationMask = 0x00000000;
			}
		}

		mShaderConstantsPluginName	= XIString( cfg.xmlElem, L"ConstantsPlugin", L"name", L"" );
		mVisibilityCheckerName		= XIString( cfg.xmlElem, L"VisibilityChecker", L"name", L"" );
	}

	//------------------------------------------------------------------------

	EXP_IMP
	EffectVariation::~EffectVariation()
	{

	}

	//------------------------------------------------------------------------

	EXP_IMP
	const EffectVariation::Settings&
	EffectVariation::GetSettings() const
	{
		return mSettings;
	}

	//------------------------------------------------------------------------

	EXP_IMP
	const EffectSubDefPtr&
	EffectVariation::GetSubDef( const AquisitionParams& params ) const
	{
		for( size_t i = 0, e = mBranches.size(); i < e; i ++ )
		{
			const Branch& branch = mBranches[ i ];

			for( size_t i = 0, e = branch.conditions.size(); i < e; i ++ )
			{
				const Condition& c = branch.conditions[ i ];

				bool failed = false;

				for( size_t i = 0, e = c.size(); i < e && !failed; i ++ )
				{
					const ConditionEntry& ce = c[ i ];
					switch( ce.type )
					{
					case FILE_CONDITION:
						if( ce.val != params.files[ ce.subDef ] )
						{
							failed = true;
						}
						break;
					case DEFINE_CONDITION:
						{
							AnsiString val = ToAnsiString( ce.val );
							bool found = false;
							if( params.defines[ ce.subDef ] )
							{
								for( size_t i = 0, e = params.defines[ ce.subDef ]->size(); i < e; i ++ )
								{
									if( ( *params.defines[ ce.subDef ] ) [ i ].name == val )
									{
										found = true;
										break;
									}
								}
							}
							if( !found ) failed = true;
						}
						break;
					default:
						MD_FERROR( L"Unsupported condition!" );
					}
				}

				if( !failed )
					return branch.subDefs[ params.type ];
			}
		}

		return mDefaultSubDefs[ params.type ];
	}

	//------------------------------------------------------------------------

	EXP_IMP
	const String&
	EffectVariation::GetShaderConstantsPluginName() const
	{
		return mShaderConstantsPluginName;
	}

	//------------------------------------------------------------------------

	EXP_IMP
	const String&
	EffectVariation::GetVisibilityCheckerName() const
	{
		return mVisibilityCheckerName;
	}

	//------------------------------------------------------------------------

	namespace
	{
		RenderMethodConverter::RenderMethodConverter()
		{
			mMap.insert( Map::value_type( L"Default",				RM::DEFAULT				) );
			mMap.insert( Map::value_type( L"InstancedMultipass",	RM::INSTANCED_MULTIPASS	) );
		}

		//------------------------------------------------------------------------

		/*static*/
		RenderMethodConverter&
		RenderMethodConverter::Single()
		{			
			static RenderMethodConverter single;
			return single;
		}

		//------------------------------------------------------------------------

		RM::RenderMethod
		RenderMethodConverter::GetRenderMethod( const String& rmName )
		{
			Map::const_iterator found = mMap.find( rmName );
			MD_FERROR_ON_FALSE( found != mMap.end() );

			return found->second;
		}

	}


}