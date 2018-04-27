#ifndef PROVIDERS_EFFECTVARIATION_H_INCLUDED
#define PROVIDERS_EFFECTVARIATION_H_INCLUDED

#include "Forw.h"

#include "Common/Src/XINamed.h"

#include "EffectDefine.h"

#include "ExportDefs.h"

#define MD_NAMESPACE EffectVariationNS
#include "ConfigurableImpl.h"

namespace Mod
{
	namespace RM
	{
		enum RenderMethod
		{
			DEFAULT,
			INSTANCED_MULTIPASS
		};
	}

	class EffectVariation :	public EffectVariationNS::ConfigurableImpl<EffectVariationConfig>,
							public XINamed
	{
		// types
	public:
		typedef Types< EffectDefine > :: Vec Defines;

		enum ConditionType
		{
			FILE_CONDITION,
			DEFINE_CONDITION
		};

		struct Settings
		{
			UINT32					passCount;
			RM::RenderMethod		renderMethod;
			EffectSubVariationBits	subVariationMask; // EffectSet should AND its bits with this mask
		};

		struct ConditionEntry
		{
			ConditionType			type;
			ESDT::EffectSubDefType	subDef;
			String					val;
		};

		// these are ANDed when evaluating
		typedef Types< ConditionEntry > :: Vec Condition;

		// these are ORed when evaluating
		typedef Types< Condition > :: Vec Conditions;

		typedef TypesI< EffectSubDefPtr, ESDT::EFFECTSUBDEFTYPE_COUNT > :: StaticArray EffectSubDefs;

		struct Branch
		{
			Conditions		conditions;
			EffectSubDefs	subDefs;
		};

		struct AquisitionParams
		{
			ESDT::EffectSubDefType	type;

			typedef TypesI< String, ESDT::EFFECTSUBDEFTYPE_COUNT > :: StaticArray			SubDefStrings;
			typedef TypesI< EffectDefinesPtr, ESDT::EFFECTSUBDEFTYPE_COUNT > :: StaticArray	SubDefEffectDefines;

			SubDefStrings			files;
			SubDefStrings			pools;

			SubDefEffectDefines		defines;
		};

		typedef Types< Branch > :: Vec Branches;

		// construction/ destruction
	public:
		EXP_IMP explicit EffectVariation( const EffectVariationConfig& cfg );
		EXP_IMP ~EffectVariation();

		// manipulation/ access
	public:
		EXP_IMP const Settings&			GetSettings() const;
		EXP_IMP const EffectSubDefPtr&	GetSubDef( const AquisitionParams& params ) const;
		EXP_IMP const String&			GetShaderConstantsPluginName() const;
		EXP_IMP const String&			GetVisibilityCheckerName() const;

		// data
	private:
		Settings					mSettings;
		String						mShaderConstantsPluginName;
		String						mVisibilityCheckerName;

		EffectSubDefs				mDefaultSubDefs;

		Branches					mBranches;
	};
}

#endif