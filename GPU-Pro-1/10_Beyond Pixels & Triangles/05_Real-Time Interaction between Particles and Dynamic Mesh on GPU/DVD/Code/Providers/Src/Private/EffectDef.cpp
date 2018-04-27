#include "Precompiled.h"

#include "Common/Src/XIElemAttribute.h"
#include "Common/Src/XIElemArray.h"
#include "Common/Src/VarTypeParser.h"
#include "Common/Src/SyncObjectBlock.h"

#include "VertexCmptDefProvider.h"

#include "EffectDefine.h"

#include "EffectProvider.h"

#include "EffectVariation.h"
#include "EffectVariationMap.h"

#include "EffectSubVariationMap.h"
#include "Providers.h"

#include "EffectSubDefConfig.h"
#include "EffectSubDef.h"

#include "EffectDefConfig.h"
#include "EffectDef.h"

#define MD_NAMESPACE EffectDefNS
#include "ConfigurableImpl.cpp.h"

#include "EffectSubDefXMLIO.h"

namespace Mod
{
	template class EffectDefNS::ConfigurableImpl<EffectDefConfig>;

	//------------------------------------------------------------------------

	EffectDef::VariatedData::VariatedData() :
	initialized( false )
	{
		
	}

	//------------------------------------------------------------------------

	EffectDef::EffectDef( const EffectDefConfig& cfg ) : 
	Parent( cfg ),
	XINamed( cfg.xmlElem )
	{
		// get required components
		{
			struct Convert
			{
				VertexCmpt operator()(const XMLElemPtr& elem) const
				{
					VertexCmpt comp;
					comp.def = prov->GetItem( elem->GetName() );
					comp.idx = XIAttInt( elem, L"idx" );
					return comp;
				}

				VertexCmptDefProviderPtr prov;

			} convert = { Providers::Single().GetVertexCmptDefProv() };

			// required
			{
				XIElemArray< VertexCmpt, Convert > comps( cfg.xmlElem->GetChild(L"VertexCmpts"), convert );
				mRequiredVCs = comps;
			}

			// output
			if( const XMLElemPtr& elem = cfg.xmlElem->GetChild(L"TransformOutput") )
			{
				XIElemArray< VertexCmpt, Convert > comps( elem, convert );
				mOutputVCs = comps;
			}
		}

		// extract subdefs
		{

			XIElemArray< TypedSubdef, ConvertToTypedSubdef > subdefs( cfg.xmlElem, L"SubDef", ConvertToTypedSubdef() );

			MD_FERROR_ON_FALSE( subdefs.size() <= mEffectSubDefs.size() );

			for( size_t i = 0, e = subdefs.size(); i < e; i ++ )
			{
				const TypedSubdef& s = subdefs[ i ];
				mEffectSubDefs[ s.type ] = s.effectSubDef;
			}

			// spread defaults
			if( const EffectSubDefPtr& d = mEffectSubDefs[ ESDT::DEFAULT ] )
			{
				for( size_t i = 0, e = mEffectSubDefs.size(); i < e; i ++ )
				{
					if( i == ESDT::DEFAULT ) continue;
					if( const EffectSubDefPtr& dest = mEffectSubDefs[ i ])
					{
						EffectSubDefRestrictions::SetEffectSubDefDefaults( *dest, d->GetData() );
					}
				}
			}
		}

		// extract default params
		{
			struct Convert
			{
				DefaultParam operator()(const XMLElemPtr& elem) const
				{
					DefaultParam defp;

					defp.value		= XIAttString( elem, L"def" );
					defp.name		= ToAnsiString( XIAttString( elem, L"name" ) );
					defp.type		= VarTypeParser::Single().GetItem( XIAttString( elem, L"type" ) );
					defp.expr_only	= XIAttInt( elem, L"expr_only", 1 ) ? true : false ;

					return defp;
				}

			} convert;

			// required
			{
				if( const XMLElemPtr& defsElem = cfg.xmlElem->GetChild( L"Parameters" ) )
				{
					XIElemArray< DefaultParam, Convert > defs( defsElem, convert );
					mDefaultParams.swap( defs );
				}
			}
		}

		// default buffers
		EffectSubDefRestrictions::SetEffectSubDefDefaultBuffers( mEffectSubDefs );

	}

	//------------------------------------------------------------------------

	EffectDef::~EffectDef()
	{

	}

	//------------------------------------------------------------------------

	EXP_IMP
	const
	EffectDef::VertexCmpts&
	EffectDef::GetRequiredVertexCmpts() const
	{
		return mRequiredVCs;
	}

	//------------------------------------------------------------------------

	EXP_IMP
	const
	EffectDef::VertexCmpts&
	EffectDef::GetOutputVertexCmpts() const
	{
		return mOutputVCs;
	}

	//------------------------------------------------------------------------

	namespace
	{
		EffectVariation::AquisitionParams ConstructAquisitionParams( ESDT::EffectSubDefType type, const EffectDef::EffectSubDefs& subDefs )
		{

			EffectVariation::AquisitionParams params;

			for( size_t i = 0, e = ESDT::EFFECTSUBDEFTYPE_COUNT; i < e; i ++ )
			{
				if( const EffectSubDefPtr& subDef = subDefs[ i ] )
				{
					const EffectSubDef::Data& subDefData = subDef->GetData();

					params.defines	[ i ] = subDefData.defines;
					params.files	[ i ] = subDefData.file;
					params.pools	[ i ] = subDefData.pool;
				}
			}

			params.type	= type;

			return params;
		}
	}

	EXP_IMP
	const EffectPtr&
	EffectDef::GetEffect( ESDT::EffectSubDefType type, EffectVariationID id, EffectVariationID subId )
	{
		MD_FERROR_ON_FALSE( mEffectSubDefs[ type ] );

		EffectVecVec& effects = mEffects[ type ];

		if( id >= effects.size() )
		{
			effects.resize( id + 1 );
		}

		if( subId >= effects[ id ].size() )
		{
			effects[ id ].resize( subId + 1 );
		}

		if( effects[ id ][ subId ] )
			return effects[ id ][ subId ];
		else
		{
			const EffectSubDefPtr& subDef = mEffectSubDefs[ type ];

			MD_FERROR_ON_FALSE( mEffectSubDefs[type] );

			const EffectVariationPtr& effVar = Providers::Single().GetEffectVariationMap()->GetVariationByID( id );

			EffectVariation::AquisitionParams params = ConstructAquisitionParams( type, mEffectSubDefs );

			const EffectSubDefPtr& mergeSubDef = effVar->GetSubDef( params );

			EffectSubDef::Data mergedData = mergeSubDef ? subDef->GetMergedData( mergeSubDef ) : subDef->GetData();

			EffectKey key( mergedData.file );
			key.poolFile = mergedData.pool;

			// own + variations defines
			{
				const EffectDefinesPtr& effDefs = mergedData.defines;
				std::copy( effDefs->begin(), effDefs->end(), std::inserter( key.defines, key.defines.end() ) );
			}

			// sub variation defines ( e.g. num lights )
			{
				const EffectDefines& effDefs = Providers::Single().GetEffectSubVariationMap()->GetEffectDefines( subId, mergedData.subVarBits );
				std::copy( effDefs.begin(), effDefs.end(), std::inserter( key.defines, key.defines.end() ) );
			}

			return ( effects[ id ][ subId ] = Providers::Single().GetEffectProv()->GetItem( key ) );
		}

	}

	//------------------------------------------------------------------------

	EXP_IMP
	EffectSubVariationBits
	EffectDef::GetSubvariationBits( ESDT::EffectSubDefType type, EffectVariationID id )
	{
		ExpandVariationData( type, id );

		VariatedDataVec& vec = mVariatedData[ type ];
		return vec[ id ].bits;
	}

	//------------------------------------------------------------------------

	EXP_IMP
	const BufferTypesSet&
	EffectDef::GetBufferTypesSet( ESDT::EffectSubDefType type, EffectVariationID id )
	{
		ExpandVariationData( type, id );

		VariatedDataVec& vec = mVariatedData[ type ];
		return vec[ id ].requiredBuffers;
	}

	//------------------------------------------------------------------------

	EXP_IMP
	bool
	EffectDef::HasSubDef( ESDT::EffectSubDefType type ) const
	{
		return mEffectSubDefs[ type ] ? true : false;
	}

	//------------------------------------------------------------------------

	EXP_IMP
	const EffectDef::DefaultParams&
	EffectDef::GetDefaultParams() const
	{
		return mDefaultParams;
	}

	//------------------------------------------------------------------------

	void
	EffectDef::ExpandVariationData( ESDT::EffectSubDefType type, EffectVariationID id )
	{
		VariatedDataVec& vec = mVariatedData[ type ];

		if( vec.size() <= id )
		{
			vec.resize( id + 1 );
		}

		if( !vec[ id ].initialized )
		{
			const EffectSubDefPtr& subDef		= mEffectSubDefs[ type ];
			const EffectVariationPtr& effVar	= Providers::Single().GetEffectVariationMap()->GetVariationByID( id );
			const EffectSubDefPtr& mergeSubDef	= effVar->GetSubDef( ConstructAquisitionParams( type, mEffectSubDefs ) );

			const EffectSubDef::Data& origData	= subDef->GetData();

			vec[ id ].bits				= ( mergeSubDef ? mergeSubDef->GetData().subVarBits	: 0xffffffff ) & origData.subVarBits;
			vec[ id ].requiredBuffers	= mergeSubDef ? mergeSubDef->GetData().requiredBuffers : origData.requiredBuffers;
			vec[ id ].initialized		= true;
		}
	}

	//------------------------------------------------------------------------

	bool operator == ( const VertexCmpt& lhs, const VertexCmpt& rhs )
	{
		return lhs.def == rhs.def && lhs.idx == rhs.idx;
	}

	//------------------------------------------------------------------------	


}