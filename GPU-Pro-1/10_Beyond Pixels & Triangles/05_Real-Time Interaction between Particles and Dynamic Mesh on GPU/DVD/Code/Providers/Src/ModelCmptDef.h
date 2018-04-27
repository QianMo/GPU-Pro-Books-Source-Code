#ifndef PROVIDERS_MODELCMPTDEF_H_INCLUDED
#define PROVIDERS_MODELCMPTDEF_H_INCLUDED

#include "Common/Src/XIElemAttribute.h"

#include "Wrap3D/Src/Forw.h"

#include "Forw.h"

#include "ExportDefs.h"

#define MD_NAMESPACE ModelCmptDefNS
#include "ConfigurableImpl.h"


namespace Mod
{
	class ModelCmptDef :	public ModelCmptDefNS::ConfigurableImpl<ModelCmptDefConfig>
	{
		// types
	public:

		typedef Types< InputLayoutPtr > :: Vec InputLayouts;
		typedef Types< InputLayouts > :: Vec InputLayoutsVec;
		typedef TypesI< InputLayoutsVec, ESDT::EFFECTSUBDEFTYPE_COUNT > :: StaticArray InputLayoutsBundle;

		struct InputLayoutLink
		{
			InputLayoutsBundle		inputLayouts;
		};

		typedef Types< EffectDefPtr >		:: Vec SupportedEffects;
		typedef Types< VertexCmpt >			:: Vec RequiredComponents;
		typedef Types< InputLayoutLink >	:: Vec InputLayoutLinks;

		// construction/ destruction
	public:
		explicit ModelCmptDef( const ModelCmptDefConfig& cfg );
		~ModelCmptDef();

		// manipulation/ access
	public:
		EXP_IMP bool					IsEffectSupported( const EffectDefPtr& ptr ) const;
		EXP_IMP void					CreateCachedModelVertexBuffers( const RawVertexDataPtr& vertexData, const String& baseKey ) const;
		EXP_IMP ModelVertexBuffersPtr	CreateModelVertexBuffers( const DevicePtr& dev, const String& baseKey ) const;
		EXP_IMP BufferPtr				CreateTransformBuffer( const DevicePtr& dev, const EffectDefPtr& effDef, UINT64 vertexCount ) const;
		EXP_IMP const SupportedEffects&	GetSupportedEffects() const;
		EXP_IMP RequiredComponents		GetGatheredRequiredComponents() const;
		EXP_IMP UINT32					GetVertexPadding() const;
		EXP_IMP UINT32					GetVertexPaddingMask() const;
		EXP_IMP bool					IsCachable() const;

		EXP_IMP void					SetInputLayout( const DevicePtr& dev, 
														UINT32 effectLink, ESDT::EffectSubDefType subDefType, 
														EffectVariationID effectVar, EffectVariationID effectSubVar );

		EXP_IMP UINT32					GetEffectLink( const EffectDefPtr& effDef ) const;

		// helpers
	private:
		InputLayoutPtr				CreateInputLayout(	const EffectDefPtr& effDef, ESDT::EffectSubDefType subDefType, 
														EffectVariationID effectVar, EffectVariationID effectSubVar );

		// data
	private:
		SupportedEffects			mSupportedEffects;
		InputLayoutLinks			mInputLayoutLinks;

		// if vertexPadding is not 0, vertex size will be padded 
		// so that it is vertexPadding * i, where i is UINT
		XIUInt						mVertexPadding;

		// split buffer into transformable & non-transformable part
		XIUInt						mSplitBuffer;
		
		XIUInt						mCachable;
	};
}

#endif