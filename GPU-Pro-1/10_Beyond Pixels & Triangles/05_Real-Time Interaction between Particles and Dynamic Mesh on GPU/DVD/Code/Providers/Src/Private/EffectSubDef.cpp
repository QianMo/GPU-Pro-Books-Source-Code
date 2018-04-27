#include "Precompiled.h"

#include "Common/Src/XIElemAttribute.h"
#include "Common/Src/XIElemArray.h"

#include "EffectSubVariationMap.h"
#include "Providers.h"

#include "EffectDefine.h"

#include "EffectSubDefConfig.h"
#include "EffectSubDef.h"

#define MD_NAMESPACE EffectSubDefNS
#include "ConfigurableImpl.cpp.h"

namespace Mod
{
	EffectSubDef::EffectSubDef( const EffectSubDefConfig& cfg ) :
	Parent( cfg ),
	mExplicitSubvarBits( false ),
	mExplicitDefines( false ),
	mExplicitBuffers( false )
	{
		mData.file	= XIString( cfg.xmlElem, L"File", L"val", L"" );
		mData.pool	= XIString( cfg.xmlElem, L"Pool", L"val", L"" );

		// defines
		{
			struct Convert
			{
				EffectDefine operator()(const XMLElemPtr& elem) const
				{
					EffectDefine def;
					def.name	= ToAnsiString( elem->GetName() );
					def.val		= ToAnsiString( XIAttString( elem, L"val", L"" ) );

					return def;
				}

				VertexCmptDefProviderPtr prov;

			} convert = { Providers::Single().GetVertexCmptDefProv() };

			mData.defines.reset( new EffectDefines );
			if( XMLElemPtr defElem = cfg.xmlElem->GetChild( L"Defines" ) )
			{
				mExplicitDefines = true;

				XIElemArray< EffectDefine, Convert > defines( defElem, convert );
				mData.defines->swap( defines );
			}
		}

		// effect subvar bits
		{
			mData.subVarBits = 0x00;

			// TODO : resolve this, this is a gruesome Providers<->SceneRender dependency,
			// perhaps these bits belong elsewhere
			if( const EffectSubVariationMapPtr& effSubVarMap = Providers::Single().GetEffectSubVariationMap() )
			{
				if( const XMLElemPtr& subvarElem = cfg.xmlElem->GetChild( L"Subvars" ) )
				{
					mExplicitSubvarBits = true;

					XIStrElemArray subvars( subvarElem );
					std::for_each( subvars.begin(), subvars.end(), (void (&)( String& ))ToLower );
					mData.subVarBits = effSubVarMap->GetSubVariationBits( subvars );
				}			
			}
		}

		// required buffers
		{
			for( size_t i = 0, e = mData.requiredBuffers.size(); i < e; i ++ )
			{
				mData.requiredBuffers[ i ] = false;
			}

			if( const XMLElemPtr& elem = cfg.xmlElem->GetChild( L"Buffers" ) )
			{
				struct Convert
				{
					BT::BufferType operator() ( const XMLElemPtr& elem ) const
					{
						XIAttString type( elem, L"type" );

						BT::BufferType res( BT::STATIC );

						if( type == L"Transformable" )
						{
							res = BT::TRANSFORMABLE;
						}
						else
						if( type == L"TransformGuides" )
						{
							res = BT::TRANSFORMGUIDES;
						}
						else
						if( type == L"Static" )
						{
							res = BT::STATIC;
						}
						else
						if( type == L"Transformed" )
						{
							res = BT::TRANSFORMED;
						}
						else
							MD_FERROR( L"Unsupported buffer type" );

						return res;
					}
				} convert;

				XIElemArray< BT::BufferType, Convert > buffers( elem, L"Buffer", convert );

				MD_FERROR_ON_FALSE( buffers.size() <= mData.requiredBuffers.size() );

				mExplicitBuffers = true;

				for( size_t i = 0, e = buffers.size(); i < e; i ++ )
				{
					mData.requiredBuffers[ buffers[ i ] ] = true;
				}
			}
		}
	}

	//------------------------------------------------------------------------

	EffectSubDef::~EffectSubDef() 
	{
	}

	//------------------------------------------------------------------------

	EXP_IMP
	const
	EffectSubDef::Data&
	EffectSubDef::GetData() const
	{
		return mData;
	}

	//------------------------------------------------------------------------

	EffectSubDef::Data
	EffectSubDef::GetMergedData( const EffectSubDefPtr& merge ) const
	{
		const EffectSubDef::Data& dominant = merge->GetData();

		EffectSubDef::Data data;

		if( !dominant.file.empty() )
		{
			data = dominant;
		}
		else
		{
			MD_FERROR_ON_FALSE( dominant.pool.empty() );

			data = mData;
			std::copy( dominant.defines->begin(), dominant.defines->end(), std::back_inserter( *data.defines ) );
			data.subVarBits &= dominant.subVarBits;
		}

		return data;
	}

	//------------------------------------------------------------------------

	void
	EffectSubDef::SetDefaults( const EffectSubDef::Data& data )
	{
		if( mData.file.empty() ) mData.file = data.file;
		if( mData.pool.empty() ) mData.pool = data.pool;

		if( !mExplicitSubvarBits )
			mData.subVarBits = data.subVarBits;

		if( !mExplicitDefines )
			*mData.defines = *data.defines;
	}

	//------------------------------------------------------------------------

	void
	EffectSubDef::SetDefaultBuffers( const BufferTypesSet& buffers )
	{
		if( !mExplicitBuffers )
			mData.requiredBuffers = buffers;
	}

	//------------------------------------------------------------------------

	/*static*/
	void
	EffectSubDefRestrictions::SetEffectSubDefDefaults( EffectSubDef& def, const EffectSubDef::Data& data )
	{
		def.SetDefaults( data );
	}

	//------------------------------------------------------------------------

	/*static*/
	void
	EffectSubDefRestrictions::SetEffectSubDefDefaultBuffers( SubDefSet& defs )
	{
		BufferTypesSet buffers;

#define MD_SET_BUFFERS( type ) if( defs[ type ] ) defs[ type ]->SetDefaultBuffers( buffers );

		// default
		{
			const int LINE_GUARD_START = __LINE__ + 3;
			// -- DO NOT ADD UNRELATED LINES --
			buffers[ BT::TRANSFORMABLE		] = true;
			buffers[ BT::TRANSFORMGUIDES	] = true;
			buffers[ BT::STATIC				] = true;
			buffers[ BT::TRANSFORMED		] = false;
			// -- DO NOT ADD UNRELATED LINES --
			MD_STATIC_ASSERT( __LINE__ - LINE_GUARD_START == BT::BUFFERTYPE_COUNT );

			MD_SET_BUFFERS( ESDT::DEFAULT )
		}

		// transform
		{
			const int LINE_GUARD_START = __LINE__ + 3;
			// -- DO NOT ADD UNRELATED LINES --
			buffers[ BT::TRANSFORMABLE		] = true;
			buffers[ BT::TRANSFORMGUIDES	] = true;
			buffers[ BT::STATIC				] = false;
			buffers[ BT::TRANSFORMED		] = false;
			// -- DO NOT ADD UNRELATED LINES --
			MD_STATIC_ASSERT( __LINE__ - LINE_GUARD_START == BT::BUFFERTYPE_COUNT );

			MD_SET_BUFFERS( ESDT::TRANSFORM )
		}

		// post transform
		{
			const int LINE_GUARD_START = __LINE__ + 3;
			// -- DO NOT ADD UNRELATED LINES --
			buffers[ BT::TRANSFORMABLE		] = false;
			buffers[ BT::TRANSFORMGUIDES	] = false;
			buffers[ BT::STATIC				] = true;
			buffers[ BT::TRANSFORMED		] = true;
			// -- DO NOT ADD UNRELATED LINES --
			MD_STATIC_ASSERT( __LINE__ - LINE_GUARD_START == BT::BUFFERTYPE_COUNT );

			MD_SET_BUFFERS( ESDT::POST_TRANSFORM )
		}

#undef MD_SET_BUFFERS		
	}



}