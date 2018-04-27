#ifndef PROVIDERS_PROVIDERSFORW_H_INCLUDED
#define PROVIDERS_PROVIDERSFORW_H_INCLUDED

namespace Mod
{

#include "DefDeclareClass.h"

	MOD_DECLARE_CLASS(VertexCmptDef)
	MOD_DECLARE_CLASS(VertexCmptFactory)
	MOD_DECLARE_CLASS(VertexCmptFactoryProvider)
	MOD_DECLARE_CLASS(VertexCmptDefProvider)	
	MOD_DECLARE_CLASS(EffectProvider)
	MOD_DECLARE_CLASS(EffectPoolProvider)
	MOD_DECLARE_CLASS(EffectIncludeProvider)
	MOD_DECLARE_CLASS(CacheMap)
	MOD_DECLARE_CLASS(EffectDefProvider)
	MOD_DECLARE_CLASS(EffectDef)
	MOD_DECLARE_CLASS(EffectSubDef)
	MOD_DECLARE_CLASS(EffectVariationProvider)
	MOD_DECLARE_CLASS(EffectVariation)
	MOD_DECLARE_CLASS(ModelCmptDefProvider)
	MOD_DECLARE_CLASS(ModelCmptDef)
	MOD_DECLARE_CLASS(ModelCmpt)
	MOD_DECLARE_CLASS(VTextModelCmpt)
	MOD_DECLARE_CLASS(SkeletonModelCmpt)
	MOD_DECLARE_CLASS(Model)
	MOD_DECLARE_CLASS(ModelProvider)
	MOD_DECLARE_CLASS(ModelCmptRawDataProvider)
	MOD_DECLARE_CLASS(ModelCmptRawDataImporter)
	MOD_DECLARE_CLASS(ModelCmptRawDataImporterProvider)
	MOD_DECLARE_CLASS(FontProvider)
	MOD_DECLARE_CLASS(VFontProvider)
	MOD_DECLARE_CLASS(VFontDefProvider)
	MOD_DECLARE_CLASS(Font)
	MOD_DECLARE_CLASS(VFont)
	MOD_DECLARE_CLASS(VFontDef)
	MOD_DECLARE_CLASS(TextureProvider)
	MOD_DECLARE_CLASS(TextureLoader)
	MOD_DECLARE_CLASS(RegisteredTextureProvider)
	MOD_DECLARE_CLASS(ShaderResourceProvider)
	MOD_DECLARE_CLASS(Skeleton)
	MOD_DECLARE_CLASS(SkeletonRawDataProvider)
	MOD_DECLARE_CLASS(SkeletonDef)
	MOD_DECLARE_CLASS(SkeletonDefProvider)
	MOD_DECLARE_CLASS(SkeletonRawDataImporter)
	MOD_DECLARE_CLASS(SkeletonRawDataImporterProvider)


	MOD_DECLARE_CLASS(ConstEvalNode)
	MOD_DECLARE_CLASS(ConstEvalNodeProvider)
	MOD_DECLARE_CLASS(ConstExprParser)
	MOD_DECLARE_CLASS(ConstEvalOperationGroup)
	MOD_DECLARE_CLASS(ConstEvalOperationGroupProvider)

	MOD_DECLARE_NON_SHARED_CLASS(Providers)
	MOD_DECLARE_CLASS(EffectVariationMap)
	MOD_DECLARE_CLASS(EffectSubVariationMap)

	MOD_DECLARE_CLASS(EffectParamsData)

	MOD_DECLARE_CLASS(ResourceLoader)
	
	MOD_DECLARE_BOLD_CLASS(VarVariant)
	MOD_DECLARE_BOLD_CLASS(AnimationMix)

	MOD_DECLARE_BOLD_CLASS(PrebuiltModelCmptRawDataImporter)

	MOD_DECLARE_BOLD_STRUCT(VertexIndexModelCmptConfig)
	MOD_DECLARE_BOLD_STRUCT(ModelCmptRawData)
	MOD_DECLARE_BOLD_STRUCT(RawVertexData)
	MOD_DECLARE_BOLD_STRUCT(RawSkeletonData)
	MOD_DECLARE_BOLD_STRUCT(TextureNamesSet)
	MOD_DECLARE_BOLD_STRUCT(ModelCmptEntry)
	MOD_DECLARE_BOLD_STRUCT(AnimationInfo)

	MOD_DECLARE_BOLD_STRUCT(EffectDefine)
	MOD_DECLARE_BOLD_STRUCT(RenderParams)
	MOD_DECLARE_BOLD_STRUCT(EntityParams)

	MOD_DECLARE_BOLD_STRUCT(ModelVertexBuffers)

	struct EffectConfigBase;
	struct EffectProviderConfigBase;

	template <typename AT>
	struct BoneNodeImpl;

	class AnimationTrack;

	typedef BoneNodeImpl<AnimationTrack> BoneNode;

	MOD_DECLARE_SHARED_PTR(BoneNode)

	typedef Types< ModelCmptRawData >	:: Vec ModelCmptRawDataArray;
	MOD_DECLARE_SHARED_PTR(ModelCmptRawDataArray)

	typedef Types< RawSkeletonData >	:: Vec RawSkeletonDataArray;
	MOD_DECLARE_SHARED_PTR(RawSkeletonDataArray)

	typedef Types< SkeletonPtr > :: Vec SkeletonArray;
	MOD_DECLARE_SHARED_PTR(SkeletonArray)

	typedef Types< EffectDefine > :: Vec EffectDefines;
	MOD_DECLARE_SHARED_PTR(EffectDefines)

	class FBXCommon;

#include "UndefDeclareClass.h"

	typedef EffectParamsDataPtr			(*EffectParamsDataCreator)( const EffectParamsDataConfig& );
	typedef EffectSubVariationMapPtr	(*EffectSubVariationMapCreator)( const EffectSubVariationMapConfig& );

	typedef UINT32 EffectVariationID;

	namespace
	{
		namespace EVI
		{
			const EffectVariationID DEFAULT_VAR_ID = 0;
		}

		typedef UINT32 AnimationID;
		const AnimationID WRONG_ANIMATION_ID = AnimationID(-1);
	}

	namespace MCF
	{
		enum ModelComponentFlag
		{
			NO_BBOX_VISIBILITY_CHECKS	= 1
		};
	}

	namespace VCT
	{
		enum VertexCmptType
		{
			TRANSFORMABLE,
			TRANSFORMGUIDES,
			STATIC,
			VERTEXELEMENTTYPE_COUNT
		};
	}

	namespace BT
	{
		enum BufferType
		{
			TRANSFORMABLE,
			TRANSFORMGUIDES,
			STATIC,
			TRANSFORMED,
			BUFFERTYPE_COUNT
		};
	}

	typedef TypesI< bool, BT::BUFFERTYPE_COUNT> :: StaticArray BufferTypesSet;

	namespace ESDT
	{
		enum EffectSubDefType
		{
			DEFAULT,
			TRANSFORM,
			POST_TRANSFORM,
			EFFECTSUBDEFTYPE_COUNT
		};
	}

	namespace ModelCmptType
	{
		enum Type
		{
			DEFAULT,
			VTEXT,
			SKELETON,
			COUNT,
			USER = 0x8000
		};
	}

	namespace LJT
	{
		enum LoadJobType
		{
			MODEL,
			TEXTURE,
			USER	= 1024
		};
	}

	typedef UINT32 ModelComponentFlags;

	typedef UINT32 EffectSubVariationBits;

	struct VertexCmpt;
	struct VertexIndexModelCmptConfigBase;	

}

#endif