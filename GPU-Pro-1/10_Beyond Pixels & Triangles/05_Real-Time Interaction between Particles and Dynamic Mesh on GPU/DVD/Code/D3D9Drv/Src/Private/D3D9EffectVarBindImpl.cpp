#include "Precompiled.h"
#include "Math/Src/Types.h"

#include "D3D9EffectVarBindConfig.h"
#include "D3D9ShaderResource.h"

#include "D3D9EffectVarBindImpl.h"

//------------------------------------------------------------------------

namespace Mod
{

#define MD_CHECK_ZERO_ARRAYS 0

	using namespace Math;
	
	template <typename T>
	D3D9EffectVarBindImpl<T>::D3D9EffectVarBindImpl( const D3D9EffectVarBindConfig& cfg ) :
	Parent( cfg )
	{

	}

	//------------------------------------------------------------------------

	template <typename T>
	D3D9EffectVarBindImpl<T>::~D3D9EffectVarBindImpl()
	{
	}

	//------------------------------------------------------------------------

	template <typename T>
	typename D3D9EffectVarBindImpl<T>::BindType
	D3D9EffectVarBindImpl<T>::GetBind() const
	{
		const EffectVarBindConfig& cfg = GetConfig();
		MD_CHECK_TYPE(const ConfigType,&cfg);
		return static_cast<const ConfigType&>( cfg ).bind;
	}

	//------------------------------------------------------------------------

	template <typename T>
	typename
	const
	D3D9EffectVarBindImpl<T>::ConfigType&
	D3D9EffectVarBindImpl<T>::GetConfig() const
	{
		return static_cast<const ConfigType&>( Parent::GetConfig() );
	}

	//------------------------------------------------------------------------

	namespace
	{
		template< typename  T>
		bool checkArraySizes( ID3DXBaseEffect* eff, D3DXHANDLE bind, const T& array )
		{
#if MD_CHECK_ZERO_ARRAYS
			MD_FERROR_ON_TRUE( array.empty() );
#endif

			D3DXPARAMETER_DESC desc;
			eff->GetParameterDesc( bind, &desc ); desc;
		
			MD_FERROR_ON_FALSE( array.size() <= desc.Elements );

#if MD_CHECK_ZERO_ARRAYS
			return true;
#else
			return !array.empty();
#endif
		}
	}

	template <>
	void
	D3D9EffectVarBindImpl<INT32>::SetValueImpl( const void * val )
	{
		const ConfigType& cfg = GetConfig();
		cfg.effect->SetInt( cfg.bind, *(const INT32*)val );
	}

	//------------------------------------------------------------------------

	template <>
	void
	D3D9EffectVarBindImpl<int_vec>::SetValueImpl( const void * val )
	{
		const ConfigType& cfg = GetConfig();

		const int_vec& typedVal = *(const int_vec*)val;

		if( checkArraySizes( &*cfg.effect, cfg.bind, typedVal ) )
		{
			cfg.effect->SetIntArray( cfg.bind, const_cast<INT32*>( &typedVal[0] ), (UINT)typedVal.size() );
		}
	}

	//------------------------------------------------------------------------

	template <>
	void
	D3D9EffectVarBindImpl<int2>::SetValueImpl( const void * val )
	{
		const int2& typedVal = *static_cast<const int2*>(val);
		int4 vec( typedVal.x, typedVal.y, 0, 0 );

		const ConfigType& cfg = GetConfig();

		cfg.effect->SetIntArray( cfg.bind, vec.elems, 4 );
	}

	//------------------------------------------------------------------------

	template <>
	void
	D3D9EffectVarBindImpl<int3>::SetValueImpl( const void * val )
	{
		const int3& typedVal = *static_cast<const int3*>(val);
		int4 vec( typedVal.x, typedVal.y, typedVal.z, 0 );

		const ConfigType& cfg = GetConfig();
		cfg.effect->SetIntArray( cfg.bind, vec.elems, 4 );

	}

	//------------------------------------------------------------------------

	template <>
	void
	D3D9EffectVarBindImpl<int4>::SetValueImpl( const void * val )
	{
		const int4& typedVal = *static_cast<const int4*>(val);
		int4 vec( typedVal.x, typedVal.y, typedVal.z, typedVal.w );

		const ConfigType& cfg = GetConfig();
		cfg.effect->SetIntArray( cfg.bind, vec.elems, 4 );
	}

	//------------------------------------------------------------------------

	template <>
	void
	D3D9EffectVarBindImpl<UINT32>::SetValueImpl( const void * val )
	{
		const ConfigType& cfg = GetConfig();
		cfg.effect->SetInt( cfg.bind, *(const UINT32*)val );
	}

	//------------------------------------------------------------------------

	template <>
	void
	D3D9EffectVarBindImpl<uint_vec>::SetValueImpl( const void * val )
	{
		const ConfigType& cfg = GetConfig();

		const uint_vec& typedVal = *(const uint_vec*)val;

		if( checkArraySizes( &*cfg.effect, cfg.bind, typedVal ) )
		{
			cfg.effect->SetIntArray( cfg.bind, static_cast<INT32*>( static_cast<void*>( const_cast<UINT32*>( &typedVal[0] ) ) ), (UINT)typedVal.size() );
		}
	}

	//------------------------------------------------------------------------

	template <>
	void
	D3D9EffectVarBindImpl<uint2>::SetValueImpl( const void * val )
	{
		const uint2& typedVal = *static_cast<const uint2*>(val);
		int4 vec( (INT32)typedVal.x, (INT32)typedVal.y, 0, 0 );

		const ConfigType& cfg = GetConfig();
		cfg.effect->SetIntArray( cfg.bind, vec.elems, 4 );

	}

	//------------------------------------------------------------------------

	template <>
	void
	D3D9EffectVarBindImpl<uint3>::SetValueImpl( const void * val )
	{
		const uint3& typedVal = *static_cast<const uint3*>(val);
		int4 vec( (INT32)typedVal.x, (INT32)typedVal.y, (INT32)typedVal.z, 0 );

		const ConfigType& cfg = GetConfig();
		cfg.effect->SetIntArray( cfg.bind, vec.elems, 4 );
	}

	//------------------------------------------------------------------------

	template <>
	void
	D3D9EffectVarBindImpl<uint4>::SetValueImpl( const void * val )
	{
		const uint4& typedVal = *static_cast<const uint4*>(val);
		int4 vec( (INT32)typedVal.x, (INT32)typedVal.y, (INT32)typedVal.z, (INT32)typedVal.w );

		const ConfigType& cfg = GetConfig();
		cfg.effect->SetIntArray( cfg.bind, vec.elems, 4 );
	}


	//------------------------------------------------------------------------

	template <>
	void
	D3D9EffectVarBindImpl<float>::SetValueImpl( const void * val )
	{
		const ConfigType& cfg = GetConfig();
		cfg.effect->SetFloat( cfg.bind, *(const float*)val );
	}

	//------------------------------------------------------------------------

	template <>
	void
	D3D9EffectVarBindImpl<float_vec>::SetValueImpl( const void * val )
	{
		const float_vec& typedVal = *(const float_vec*)val;

		const ConfigType& cfg = GetConfig();
		if( checkArraySizes( &*cfg.effect, cfg.bind, typedVal ) )
		{
			cfg.effect->SetFloatArray( cfg.bind, const_cast<float*>( &typedVal[0] ), (UINT)typedVal.size() );
		}
	}


	//------------------------------------------------------------------------

	template <>
	void
	D3D9EffectVarBindImpl<float2>::SetValueImpl( const void * val )
	{
		const float2& typedVal = *static_cast<const float2*>(val);
		float4 vec( typedVal.x, typedVal.y, 0, 0 );

		const ConfigType& cfg = GetConfig();

		cfg.effect->SetFloatArray( cfg.bind, vec.elems, 4 );
	}

	//------------------------------------------------------------------------

	namespace
	{
		template< typename T >
		void setFloatArray(	ID3DXEffect* eff, D3DXHANDLE bind, const void * val )
		{
			const T& typedVal = *(const T*)val;

			if( checkArraySizes( eff, bind, typedVal ) )
			{
				eff->SetFloatArray( bind, typedVal[0].elems, (UINT)typedVal.size() * T::value_type::COMPONENT_COUNT );
			}
		}
	}

	template <>
	void
	D3D9EffectVarBindImpl<float2_vec>::SetValueImpl( const void * val )
	{
		const ConfigType& cfg = GetConfig();
		setFloatArray<float2_vec>( &*cfg.effect, cfg.bind, val );
	}

	//------------------------------------------------------------------------

	template <>
	void
	D3D9EffectVarBindImpl<float3>::SetValueImpl( const void * val )
	{
		const float3& typedVal = *static_cast<const float3*>(val);
		float4 vec( typedVal.x, typedVal.y, typedVal.z, 0 );
		
		const ConfigType& cfg = GetConfig();
		cfg.effect->SetFloatArray( cfg.bind, vec.elems, 4 );
	}

	//------------------------------------------------------------------------

	template <>
	void
	D3D9EffectVarBindImpl<float3_vec>::SetValueImpl( const void * val )
	{
		const ConfigType& cfg = GetConfig();
		setFloatArray<float3_vec>( &*cfg.effect, cfg.bind, val );
	}

	//------------------------------------------------------------------------

	template <>
	void
	D3D9EffectVarBindImpl<float4>::SetValueImpl( const void * val )
	{
		const float4& typedVal = *static_cast<const float4*>(val);
		float4 vec( typedVal );

		const ConfigType& cfg = GetConfig();
		cfg.effect->SetFloatArray( cfg.bind, vec.elems, 4 );
	}

	//------------------------------------------------------------------------

	template <>
	void
	D3D9EffectVarBindImpl<float4_vec>::SetValueImpl( const void * val )
	{
		const ConfigType& cfg = GetConfig();
		setFloatArray<float4_vec>( &*cfg.effect, cfg.bind, val );
	}

	//------------------------------------------------------------------------

	template <>
	void
	D3D9EffectVarBindImpl<float2x2>::SetValueImpl( const void * val )
	{
		const float2x2& typedVal = *(const float2x2*)val;

		D3DXMATRIX mat(	typedVal[0].x,	typedVal[0].y,	0,	0,
						typedVal[1].x,	typedVal[1].y,	0,	0,
						0,				0,				1,	0,
						0,				0,				0,	1 );

		const ConfigType& cfg = GetConfig();
		cfg.effect->SetMatrix( cfg.bind, &mat );
	}

	//------------------------------------------------------------------------

	template <>
	void
	D3D9EffectVarBindImpl<float3x3>::SetValueImpl( const void * val )
	{
		const float3x3& typedVal = *(const float3x3*)val;

		D3DXMATRIX mat(	typedVal[0].x,	typedVal[0].y,	typedVal[0].z,	0,
						typedVal[1].x,	typedVal[1].y,	typedVal[1].z,	0,
						typedVal[2].x,	typedVal[2].y,	typedVal[2].z,	0,
						0,				0,				0,				1 );
								
		const ConfigType& cfg = GetConfig();
		cfg.effect->SetMatrix( cfg.bind, &mat );
	}

	//------------------------------------------------------------------------

	namespace
	{
		D3DXMATRIX toDXMatrix( const float4x4& val )
		{
			return D3DXMATRIX(	val[0].x,	val[0].y,	val[0].z,	val[0].w,
								val[1].x,	val[1].y,	val[1].z,	val[1].w,
								val[2].x,	val[2].y,	val[2].z,	val[2].w,
								val[3].x,	val[3].y,	val[3].z,	val[3].w );
		}
	}

	template <>
	void
	D3D9EffectVarBindImpl<float4x4>::SetValueImpl( const void * val )
	{
		const float4x4& typedVal = *(const float4x4*)val;

		const D3DXMATRIX& matrix = toDXMatrix( typedVal);
		const ConfigType& cfg = GetConfig();
		cfg.effect->SetMatrix( cfg.bind, &matrix );
	}

	//------------------------------------------------------------------------

	namespace
	{
		D3DXMATRIX toDXMatrix( const float3x4& mat )
		{
			return D3DXMATRIX(	mat[0].x,	mat[0].y,	mat[0].z,	0,
								mat[1].x,	mat[1].y,	mat[1].z,	0,
								mat[2].x,	mat[2].y,	mat[2].z,	0,
								mat[3].x,	mat[3].y,	mat[3].z,	1 );
		}
	}

	template <>
	void
	D3D9EffectVarBindImpl<float3x4>::SetValueImpl( const void * val )
	{
		const float3x4& typedVal = *(const float3x4*)val;
		const D3DXMATRIX& matrix = toDXMatrix( typedVal);

		const ConfigType& cfg = GetConfig();
		cfg.effect->SetMatrix( cfg.bind, &matrix );
	}

	//------------------------------------------------------------------------

	template <>
	void
	D3D9EffectVarBindImpl<float4x4_vec>::SetValueImpl( const void * val )
	{
		const float4x4_vec& typedVal = *(const float4x4_vec*)val;

		const ConfigType& cfg = GetConfig();

		if( checkArraySizes( &*cfg.effect, cfg.bind, typedVal ) )
		{
			size_t numMats = typedVal.size();

			D3DXMATRIX* dxmats = static_cast<D3DXMATRIX*> ( MD_STACK_ALLOC( sizeof( D3DXMATRIX ) * numMats ) );

			MD_FERROR_ON_FALSE( dxmats );

			for( size_t i = 0, e = numMats; i < e; ++i )
			{
				dxmats[i] = toDXMatrix( typedVal[i] );
			}

			cfg.effect->SetMatrixArray( cfg.bind, dxmats, (UINT)numMats  )

			MD_STACK_FREE( dxmats );
		}
	}

	//------------------------------------------------------------------------

	template <>
	void
	D3D9EffectVarBindImpl<float3x4_vec>::SetValueImpl( const void * val )
	{
		const float3x4_vec& typedVal = *(const float3x4_vec*)val;

		const ConfigType& cfg = GetConfig();

		if( checkArraySizes( &*cfg.effect, cfg.bind, typedVal ) )
		{
			size_t numMats = typedVal.size();

			D3DXMATRIX* dxmats = static_cast<D3DXMATRIX*> ( MD_STACK_ALLOC( sizeof( D3DXMATRIX ) * numMats ) );

			MD_FERROR_ON_FALSE( dxmats );

			for( size_t i = 0, e = numMats; i < e; ++i )
			{
				dxmats[i] = toDXMatrix( typedVal[i] );
			}

			cfg.effect->SetMatrixArray( cfg.bind, dxmats, (UINT)numMats  );

			MD_STACK_FREE( dxmats );
		}
	}

	//------------------------------------------------------------------------

	template <>
	void
	D3D9EffectVarBindImpl<ShaderResource>::SetValueImpl( const void * val )
	{
		const ConfigType& cfg = GetConfig();

		if( const ShaderResource* sr =  static_cast<const ShaderResource*>(val) )
			static_cast<const D3D9ShaderResource*>( sr )->BindTo( &*cfg.effect, cfg.bind );
		else
			D3D9ShaderResource::SetBindToZero( &*cfg.effect, cfg.bind );

	}

	//------------------------------------------------------------------------

	template <typename T>
	void
	D3D9EffectVarBindImpl<T>::UnbindImpl()
	{
		MD_FERROR( L"Can't unbind this bind type!");
	}

	//------------------------------------------------------------------------

	template <>
	void
	D3D9EffectVarBindImpl<ShaderResource>::UnbindImpl()
	{
		const ConfigType& cfg = GetConfig();

		D3D9ShaderResource::SetBindToZero( &*cfg.effect, cfg.bind );
	}

	//------------------------------------------------------------------------

	template class D3D9EffectVarBindImpl<INT32>;
	template class D3D9EffectVarBindImpl<int_vec>;
	template class D3D9EffectVarBindImpl<int2>;
	template class D3D9EffectVarBindImpl<int3>;
	template class D3D9EffectVarBindImpl<int4>;
	template class D3D9EffectVarBindImpl<UINT32>;
	template class D3D9EffectVarBindImpl<uint_vec>;
	template class D3D9EffectVarBindImpl<uint2>;
	template class D3D9EffectVarBindImpl<uint3>;
	template class D3D9EffectVarBindImpl<uint4>;
	template class D3D9EffectVarBindImpl<float>;
	template class D3D9EffectVarBindImpl<float_vec>;
	template class D3D9EffectVarBindImpl<float2>;
	template class D3D9EffectVarBindImpl<float2_vec>;
	template class D3D9EffectVarBindImpl<float3>;
	template class D3D9EffectVarBindImpl<float3_vec>;
	template class D3D9EffectVarBindImpl<float4>;
	template class D3D9EffectVarBindImpl<float4_vec>;
	template class D3D9EffectVarBindImpl<float2x2>;
	template class D3D9EffectVarBindImpl<float3x3>;
	template class D3D9EffectVarBindImpl<float4x4>;
	template class D3D9EffectVarBindImpl<float3x4>;
	template class D3D9EffectVarBindImpl<float4x4_vec>;
	template class D3D9EffectVarBindImpl<float3x4_vec>;
	template class D3D9EffectVarBindImpl<ShaderResource>;

}
