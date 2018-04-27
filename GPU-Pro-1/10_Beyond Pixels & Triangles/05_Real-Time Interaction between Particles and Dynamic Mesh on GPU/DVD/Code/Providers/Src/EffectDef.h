#ifndef PROVIDERS_EFFECTDEF_H_INCLUDED
#define PROVIDERS_EFFECTDEF_H_INCLUDED

#include "Wrap3D/Src/Forw.h"
#include "WrapSys/Src/Forw.h"

#include "Common/Src/XINamed.h"

#include "Forw.h"

#include "ExportDefs.h"

#define MD_NAMESPACE EffectDefNS
#include "ConfigurableImpl.h"


namespace Mod
{

	struct VertexCmpt
	{
		VertexCmptDefPtr def;
		UINT32 idx;		
	};

	bool operator == ( const VertexCmpt& lhs, const VertexCmpt& rhs );

	class EffectDef :	public EffectDefNS::ConfigurableImpl<EffectDefConfig>,
						public XINamed
	{
		// types
	public:
		typedef Types< VertexCmpt > :: Vec		VertexCmpts;
		typedef Types< AnsiString > :: Vec		ParamNames;

		typedef Types< EffectPtr >				:: Vec EffectVec;
		typedef Types< EffectVec >				:: Vec EffectVecVec;

		struct DefaultParam
		{
			AnsiString		name;
			String			value;
			VarType::Type	type;
			bool			expr_only;
		};

		typedef Types< DefaultParam > :: Vec DefaultParams;

		struct VariatedData
		{
			VariatedData();

			EffectSubVariationBits	bits;
			BufferTypesSet			requiredBuffers;

			bool					initialized;
		};

		typedef Types< VariatedData > :: Vec VariatedDataVec;

		typedef TypesI< EffectVecVec, ESDT::EFFECTSUBDEFTYPE_COUNT > :: StaticArray EffectBundle;

		typedef TypesI< EffectSubDefPtr, ESDT::EFFECTSUBDEFTYPE_COUNT > :: StaticArray EffectSubDefs;

		typedef TypesI< VariatedDataVec, ESDT::EFFECTSUBDEFTYPE_COUNT > :: StaticArray VariatedDataBundle;

		// construction/ destruction
	public:
		explicit EffectDef( const EffectDefConfig& cfg );
		~EffectDef();

		// manipulation/ access
	public:
		EXP_IMP const VertexCmpts&			GetRequiredVertexCmpts() const;
		EXP_IMP const VertexCmpts&			GetOutputVertexCmpts() const;
		EXP_IMP const EffectPtr&			GetEffect( ESDT::EffectSubDefType type, EffectVariationID id, EffectVariationID subId );
		EXP_IMP EffectSubVariationBits		GetSubvariationBits( ESDT::EffectSubDefType type, EffectVariationID id );
		EXP_IMP const BufferTypesSet&		GetBufferTypesSet( ESDT::EffectSubDefType type, EffectVariationID id );
		EXP_IMP bool						HasSubDef( ESDT::EffectSubDefType type ) const;
		EXP_IMP const DefaultParams&		GetDefaultParams() const;

		// helpers
	private:
		void ExpandVariationData( ESDT::EffectSubDefType type, EffectVariationID id );

		// data
	private:
		VertexCmpts						mRequiredVCs;
		VertexCmpts						mOutputVCs;

		EffectSubDefs					mEffectSubDefs;

		EffectBundle					mEffects;
		VariatedDataBundle				mVariatedData;

		DefaultParams					mDefaultParams;
	};
}

#endif