#ifndef D3D9DRV_D3D9EFFECTVARBINDIMPL_H_INCLUDED
#define D3D9DRV_D3D9EFFECTVARBINDIMPL_H_INCLUDED

#include "Wrap3D/Src/EffectVarBind.h"
#include "Math/Src/Forw.h"
#include "D3D9EffectVarBindConfig.h"

namespace Mod
{
	
	template <typename T>
	class D3D9EffectVarBindImpl : public EffectVarBind 
	{
		// types
	public:
		typedef typename D3D9EffectVarBindConfig::BindType		BindType;
		typedef D3D9EffectVarBindConfig							ConfigType;

		// construction/ destruction
	public:
		explicit D3D9EffectVarBindImpl( const ConfigType& cfg );
		~D3D9EffectVarBindImpl();

		// helpers
	protected:
		BindType	GetBind() const;
		const ConfigType&	GetConfig() const;

		// polymorphism
	private:
		virtual void SetValueImpl( const void * val ) OVERRIDE;
		virtual void UnbindImpl() OVERRIDE;
	};

	//------------------------------------------------------------------------

	typedef D3D9EffectVarBindImpl<INT32>				D3D9EffectIntVarBind;
	typedef D3D9EffectVarBindImpl<Math::int2>			D3D9EffectInt2VarBind;
	typedef D3D9EffectVarBindImpl<Math::int3>			D3D9EffectInt3VarBind;
	typedef D3D9EffectVarBindImpl<Math::int4>			D3D9EffectInt4VarBind;

	typedef D3D9EffectVarBindImpl<UINT32>				D3D9EffectUIntVarBind;
	typedef D3D9EffectVarBindImpl<Math::uint2>			D3D9EffectUInt2VarBind;
	typedef D3D9EffectVarBindImpl<Math::uint3>			D3D9EffectUInt3VarBind;
	typedef D3D9EffectVarBindImpl<Math::uint4>			D3D9EffectUInt4VarBind;

	typedef D3D9EffectVarBindImpl<float>				D3D9EffectFloatVarBind;
	typedef D3D9EffectVarBindImpl<Math::float2>		D3D9EffectFloat2VarBind;
	typedef D3D9EffectVarBindImpl<Math::float3>		D3D9EffectFloat3VarBind;
	typedef D3D9EffectVarBindImpl<Math::float4>		D3D9EffectFloat4VarBind;

	typedef D3D9EffectVarBindImpl<Math::float2x2>		D3D9EffectFloat2x2VarBind;
	typedef D3D9EffectVarBindImpl<Math::float3x3>		D3D9EffectFloat3x3VarBind;
	typedef D3D9EffectVarBindImpl<Math::float4x4>		D3D9EffectFloat4x4VarBind;

	typedef D3D9EffectVarBindImpl<Math::float3x4>		D3D9EffectFloat3x4VarBind;

	typedef D3D9EffectVarBindImpl<Math::int_vec>		D3D9EffectIntVecVarBind;
	typedef D3D9EffectVarBindImpl<Math::uint_vec>		D3D9EffectUIntVecVarBind;
	typedef D3D9EffectVarBindImpl<Math::float_vec>		D3D9EffectFloatVecVarBind;
	typedef D3D9EffectVarBindImpl<Math::float2_vec>	D3D9EffectFloat2VecVarBind;
	typedef D3D9EffectVarBindImpl<Math::float3_vec>	D3D9EffectFloat3VecVarBind;
	typedef D3D9EffectVarBindImpl<Math::float4_vec>	D3D9EffectFloat4VecVarBind;

	typedef D3D9EffectVarBindImpl<Math::float4x4_vec>	D3D9EffectFloat4x4VecVarBind;
	typedef D3D9EffectVarBindImpl<Math::float3x4_vec>	D3D9EffectFloat3x4VecVarBind;

	typedef D3D9EffectVarBindImpl<ShaderResource>		D3D9EffectShaderResourceVarBind;
	typedef D3D9EffectVarBindImpl<Buffer>				D3D9EffectCBufferVarBind;

}

#endif
