#ifndef PROVIDERS_EFFECTSUBDEF_H_INCLUDED
#define PROVIDERS_EFFECTSUBDEF_H_INCLUDED

#include "Forw.h"

#include "ExportDefs.h"

#define MD_NAMESPACE EffectSubDefNS
#include "ConfigurableImpl.h"

namespace Mod
{

	class EffectSubDef : public EffectSubDefNS::ConfigurableImpl<EffectSubDefConfig>
	{
		friend class EffectSubDefRestrictions;

		// types
	public:
		typedef Types< VertexCmpt > :: Vec							VertexCmpts;

		struct Data
		{
			String					file;
			String					pool;
			EffectDefinesPtr		defines;
			EffectSubVariationBits	subVarBits;
			BufferTypesSet			requiredBuffers;
		};

		// constructors / destructors
	public:
		explicit EffectSubDef( const EffectSubDefConfig& cfg );
		~EffectSubDef();
	
		// manipulation/ access
	public:
		EXP_IMP const Data&	GetData() const;
		Data				GetMergedData( const EffectSubDefPtr& merge ) const;


		// helpers
	private:
		void SetDefaults( const Data& data );
		void SetDefaultBuffers( const BufferTypesSet& buffers );

		// data
	private:

		Data mData;
		bool mExplicitSubvarBits;
		bool mExplicitDefines;
		bool mExplicitBuffers;
	};

	//------------------------------------------------------------------------

	class EffectSubDefRestrictions
	{
		friend class EffectDef;
		friend class EffectVariation;

		typedef TypesI< EffectSubDefPtr, ESDT::EFFECTSUBDEFTYPE_COUNT > :: StaticArray SubDefSet;

		static void SetEffectSubDefDefaults( EffectSubDef& def, const EffectSubDef::Data& data );
		static void SetEffectSubDefDefaultBuffers( SubDefSet& defs );
	};


}

#endif