#ifndef D3D10DRV_D3D10EFFECTVARBINDIMPL_H_INCLUDED
#define D3D10DRV_D3D10EFFECTVARBINDIMPL_H_INCLUDED

#include "Wrap3D/Src/EffectVarBind.h"
#include "Math/Src/Forw.h"
#include "D3D10EffectVarBindConfig.h"

namespace Mod
{
	
	template <typename T>
	class D3D10EffectVarBindImpl : public EffectVarBind 
	{
		// types
	public:
		typedef typename EffectVarBind Base;
		typedef typename D3D10EffectVarBindConfig<T>::BindType		BindType;
		typedef typename D3D10EffectVarBindConfig<T>::BindTypePtr	BindTypePtr;
		typedef D3D10EffectVarBindConfig<T> ConfigType;

		// construction/ destruction
	public:
		explicit D3D10EffectVarBindImpl( const ConfigType& cfg );
		~D3D10EffectVarBindImpl();

		// helpers
	protected:
		BindTypePtr GetBind() const;

		// polymorphism
	private:
		virtual void SetValueImpl( const void * val ) OVERRIDE;
		virtual void UnbindImpl() OVERRIDE;
	};

	typedef D3D10EffectVarBindImpl<INT32>				D3D10EffectIntVarBind;
	typedef D3D10EffectVarBindImpl<Math::int2>			D3D10EffectInt2VarBind;
	typedef D3D10EffectVarBindImpl<Math::int3>			D3D10EffectInt3VarBind;
	typedef D3D10EffectVarBindImpl<Math::int4>			D3D10EffectInt4VarBind;

	typedef D3D10EffectVarBindImpl<UINT32>				D3D10EffectUIntVarBind;
	typedef D3D10EffectVarBindImpl<Math::uint2>			D3D10EffectUInt2VarBind;
	typedef D3D10EffectVarBindImpl<Math::uint3>			D3D10EffectUInt3VarBind;
	typedef D3D10EffectVarBindImpl<Math::uint4>			D3D10EffectUInt4VarBind;

	typedef D3D10EffectVarBindImpl<float>				D3D10EffectFloatVarBind;
	typedef D3D10EffectVarBindImpl<Math::float2>		D3D10EffectFloat2VarBind;
	typedef D3D10EffectVarBindImpl<Math::float3>		D3D10EffectFloat3VarBind;
	typedef D3D10EffectVarBindImpl<Math::float4>		D3D10EffectFloat4VarBind;

	typedef D3D10EffectVarBindImpl<Math::float2x2>		D3D10EffectFloat2x2VarBind;
	typedef D3D10EffectVarBindImpl<Math::float3x3>		D3D10EffectFloat3x3VarBind;
	typedef D3D10EffectVarBindImpl<Math::float4x4>		D3D10EffectFloat4x4VarBind;

	typedef D3D10EffectVarBindImpl<Math::float3x4>		D3D10EffectFloat3x4VarBind;

	typedef D3D10EffectVarBindImpl<Math::int_vec>		D3D10EffectIntVecVarBind;
	typedef D3D10EffectVarBindImpl<Math::uint_vec>		D3D10EffectUIntVecVarBind;
	typedef D3D10EffectVarBindImpl<Math::float_vec>		D3D10EffectFloatVecVarBind;
	typedef D3D10EffectVarBindImpl<Math::float2_vec>	D3D10EffectFloat2VecVarBind;
	typedef D3D10EffectVarBindImpl<Math::float3_vec>	D3D10EffectFloat3VecVarBind;
	typedef D3D10EffectVarBindImpl<Math::float4_vec>	D3D10EffectFloat4VecVarBind;

	typedef D3D10EffectVarBindImpl<Math::float4x4_vec>	D3D10EffectFloat4x4VecVarBind;
	typedef D3D10EffectVarBindImpl<Math::float3x4_vec>	D3D10EffectFloat3x4VecVarBind;

	typedef D3D10EffectVarBindImpl<ShaderResource>		D3D10EffectShaderResourceVarBind;
	typedef D3D10EffectVarBindImpl<Buffer>				D3D10EffectCBufferVarBind;

}

#endif
