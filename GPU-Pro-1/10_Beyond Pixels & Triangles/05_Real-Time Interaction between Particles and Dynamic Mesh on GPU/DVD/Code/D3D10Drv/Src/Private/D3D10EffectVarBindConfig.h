#ifndef D3D10DRV_D3D10EFFECTVARBINDCONFIG_H_INCLUDED
#define D3D10DRV_D3D10EFFECTVARBINDCONFIG_H_INCLUDED

#include "Wrap3D/Src/EffectVarBindConfig.h"
#include "Math/Src/Forw.h"

namespace Mod
{

	template< typename T>
	struct ModTypeToDXVarbindType
	{
		// default to matrix result
		typedef ID3D10EffectMatrixVariable Result;
	};

	template <typename T>
	struct D3D10EffectVarBindConfig : EffectVarBindConfig
	{
		// types
	public:
		typedef typename ModTypeToDXVarbindType<T>::Result BindType;
		typedef BindType* BindTypePtr;

		// construction/ destruction
	public:
		explicit D3D10EffectVarBindConfig( BindTypePtr a_bind );

		// data
	public:
		BindTypePtr bind;

		// polymorphism
	private:
		virtual EffectVarBindConfig* Clone() const;
	};

	//------------------------------------------------------------------------

	template <typename T>
	D3D10EffectVarBindConfig<T>::D3D10EffectVarBindConfig( BindTypePtr a_bind ) :
	bind( a_bind )
	{
	}

	//------------------------------------------------------------------------

	template <typename T>
	EffectVarBindConfig*
	D3D10EffectVarBindConfig<T>::Clone() const
	{
		return new D3D10EffectVarBindConfig<T>( *this );
	}

	//------------------------------------------------------------------------	

#define MD_IMPLEMENT_EFFECT_VARBIND_STRUCT(type,result) template<>	struct ModTypeToDXVarbindType<type>	{	typedef result Result;	};
	MD_IMPLEMENT_EFFECT_VARBIND_STRUCT( INT32			, ID3D10EffectScalarVariable			)
	MD_IMPLEMENT_EFFECT_VARBIND_STRUCT( Math::int_vec	, ID3D10EffectScalarVariable			)
	MD_IMPLEMENT_EFFECT_VARBIND_STRUCT( Math::int2		, ID3D10EffectVectorVariable			)
	MD_IMPLEMENT_EFFECT_VARBIND_STRUCT( Math::int3		, ID3D10EffectVectorVariable			)
	MD_IMPLEMENT_EFFECT_VARBIND_STRUCT( Math::int4		, ID3D10EffectVectorVariable			)
	MD_IMPLEMENT_EFFECT_VARBIND_STRUCT( UINT32			, ID3D10EffectScalarVariable			)
	MD_IMPLEMENT_EFFECT_VARBIND_STRUCT( Math::uint_vec	, ID3D10EffectScalarVariable			)
	MD_IMPLEMENT_EFFECT_VARBIND_STRUCT( Math::uint2		, ID3D10EffectVectorVariable			)
	MD_IMPLEMENT_EFFECT_VARBIND_STRUCT( Math::uint3		, ID3D10EffectVectorVariable			)
	MD_IMPLEMENT_EFFECT_VARBIND_STRUCT( Math::uint4		, ID3D10EffectVectorVariable			)
	MD_IMPLEMENT_EFFECT_VARBIND_STRUCT( float			, ID3D10EffectScalarVariable			)
	MD_IMPLEMENT_EFFECT_VARBIND_STRUCT( Math::float_vec	, ID3D10EffectScalarVariable			)
	MD_IMPLEMENT_EFFECT_VARBIND_STRUCT( Math::float2	, ID3D10EffectVectorVariable			)
	MD_IMPLEMENT_EFFECT_VARBIND_STRUCT( Math::float2_vec, ID3D10EffectVectorVariable			)
	MD_IMPLEMENT_EFFECT_VARBIND_STRUCT( Math::float3	, ID3D10EffectVectorVariable			)
	MD_IMPLEMENT_EFFECT_VARBIND_STRUCT( Math::float3_vec, ID3D10EffectVectorVariable			)
	MD_IMPLEMENT_EFFECT_VARBIND_STRUCT( Math::float4	, ID3D10EffectVectorVariable			)
	MD_IMPLEMENT_EFFECT_VARBIND_STRUCT( Math::float4_vec, ID3D10EffectVectorVariable			)
	MD_IMPLEMENT_EFFECT_VARBIND_STRUCT( ShaderResource	, ID3D10EffectShaderResourceVariable	)
	MD_IMPLEMENT_EFFECT_VARBIND_STRUCT( Buffer			, ID3D10EffectConstantBuffer			)
#undef MD_IMPLEMENT_EFFECT_VARBIND_STRUCT


}

#endif
