#include "Precompiled.h"
#include "D3D10EffectVarBindImpl.h"
#include "Math/Src/Types.h"
#include "D3D10ShaderResource.h"
#include "D3D10Buffer.h"

//------------------------------------------------------------------------

namespace Mod
{

#define MD_CHECK_ZERO_ARRAYS 0

	using namespace Math;
	
	template <typename T>
	D3D10EffectVarBindImpl<T>::D3D10EffectVarBindImpl( const D3D10EffectVarBindConfig<T>& cfg ) :
	Base( cfg )
	{

	}

	//------------------------------------------------------------------------

	template <typename T>
	D3D10EffectVarBindImpl<T>::~D3D10EffectVarBindImpl()
	{
	}

	//------------------------------------------------------------------------

	template <typename T>
	typename D3D10EffectVarBindImpl<T>::BindTypePtr
	D3D10EffectVarBindImpl<T>::GetBind() const
	{
		const EffectVarBindConfig& cfg = GetConfig();
		MD_CHECK_TYPE(const ConfigType,&cfg);
		return static_cast<const ConfigType&>( cfg ).bind;
	}

	//------------------------------------------------------------------------

	namespace
	{
		template< typename  T>
		bool checkArraySizes( ID3D10EffectVariable* bind, const T& array )
		{
#if MD_CHECK_ZERO_ARRAYS
			MD_FERROR_ON_TRUE( array.empty() );
#endif

			D3D10_EFFECT_TYPE_DESC desc;
			bind->GetType()->GetDesc( &desc );
			
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
	D3D10EffectVarBindImpl<INT32>::SetValueImpl( const void * val )
	{
		GetBind()->SetInt( *(const INT32*)val );
	}

	//------------------------------------------------------------------------

	template <>
	void
	D3D10EffectVarBindImpl<int_vec>::SetValueImpl( const void * val )
	{
		const int_vec& typedVal = *(const int_vec*)val;

		if( checkArraySizes( GetBind(), typedVal ) )
		{
			GetBind()->SetIntArray( const_cast<INT32*>( &typedVal[0] ), 0, (UINT)typedVal.size() );
		}
	}

	//------------------------------------------------------------------------

	template <>
	void
	D3D10EffectVarBindImpl<int2>::SetValueImpl( const void * val )
	{
		const int2& typedVal = *static_cast<const int2*>(val);
		int4 vec( typedVal.x, typedVal.y, 0, 0 );
		GetBind()->SetIntVector( vec.elems );
	}

	//------------------------------------------------------------------------

	template <>
	void
	D3D10EffectVarBindImpl<int3>::SetValueImpl( const void * val )
	{
		const int3& typedVal = *static_cast<const int3*>(val);
		int4 vec( typedVal.x, typedVal.y, typedVal.z, 0 );
		GetBind()->SetIntVector( vec.elems );
	}

	//------------------------------------------------------------------------

	template <>
	void
	D3D10EffectVarBindImpl<int4>::SetValueImpl( const void * val )
	{
		const int4& typedVal = *static_cast<const int4*>(val);
		int4 vec( typedVal.x, typedVal.y, typedVal.z, typedVal.w );
		GetBind()->SetIntVector( vec.elems );
	}

	//------------------------------------------------------------------------

	template <>
	void
	D3D10EffectVarBindImpl<UINT32>::SetValueImpl( const void * val )
	{
		GetBind()->SetInt( *(const UINT32*)val );
	}

	//------------------------------------------------------------------------

	template <>
	void
	D3D10EffectVarBindImpl<uint_vec>::SetValueImpl( const void * val )
	{
		const uint_vec& typedVal = *(const uint_vec*)val;

		if( checkArraySizes( GetBind(), typedVal ) )
		{
			GetBind()->SetIntArray( static_cast<INT32*>( static_cast<void*>( const_cast<UINT32*>( &typedVal[0] ) ) ), 0, (UINT)typedVal.size() );
		}
	}

	//------------------------------------------------------------------------

	template <>
	void
	D3D10EffectVarBindImpl<uint2>::SetValueImpl( const void * val )
	{
		const uint2& typedVal = *static_cast<const uint2*>(val);
		int4 vec( (INT32)typedVal.x, (INT32)typedVal.y, 0, 0 );
		GetBind()->SetIntVector( vec.elems );
	}

	//------------------------------------------------------------------------

	template <>
	void
	D3D10EffectVarBindImpl<uint3>::SetValueImpl( const void * val )
	{
		const uint3& typedVal = *static_cast<const uint3*>(val);
		int4 vec( (INT32)typedVal.x, (INT32)typedVal.y, (INT32)typedVal.z, 0 );
		GetBind()->SetIntVector( vec.elems );
	}

	//------------------------------------------------------------------------

	template <>
	void
	D3D10EffectVarBindImpl<uint4>::SetValueImpl( const void * val )
	{
		const uint4& typedVal = *static_cast<const uint4*>(val);
		int4 vec( (INT32)typedVal.x, (INT32)typedVal.y, (INT32)typedVal.z, (INT32)typedVal.w );
		GetBind()->SetIntVector( vec.elems );
	}


	//------------------------------------------------------------------------

	template <>
	void
	D3D10EffectVarBindImpl<float>::SetValueImpl( const void * val )
	{
		GetBind()->SetFloat( *(const float*)val );
	}

	//------------------------------------------------------------------------

	template <>
	void
	D3D10EffectVarBindImpl<float_vec>::SetValueImpl( const void * val )
	{
		const float_vec& typedVal = *(const float_vec*)val;

		if( checkArraySizes( GetBind(), typedVal ) )
		{
			GetBind()->SetFloatArray( const_cast<float*>( &typedVal[0] ), 0, (UINT)typedVal.size() );
		}
	}


	//------------------------------------------------------------------------

	template <>
	void
	D3D10EffectVarBindImpl<float2>::SetValueImpl( const void * val )
	{
		const float2& typedVal = *static_cast<const float2*>(val);
		float4 vec( typedVal.x, typedVal.y, 0, 0 );
		GetBind()->SetFloatVector( vec.elems );
	}

	//------------------------------------------------------------------------

	namespace
	{
		template< typename T, typename B >
		void setFloatArray( B* bind, const void * val )
		{
			const T& typedVal = *(const T*)val;

			if( checkArraySizes( bind, typedVal ) )
			{
				bind->SetFloatVectorArray( const_cast<float*>( typedVal[0].elems ), 0, (UINT)typedVal.size() );
			}
		}
	}

	template <>
	void
	D3D10EffectVarBindImpl<float2_vec>::SetValueImpl( const void * val )
	{
		setFloatArray<float2_vec>( GetBind(), val );
	}

	//------------------------------------------------------------------------

	template <>
	void
	D3D10EffectVarBindImpl<float3>::SetValueImpl( const void * val )
	{
		const float3& typedVal = *static_cast<const float3*>(val);
		float4 vec( typedVal.x, typedVal.y, typedVal.z, 0 );
		GetBind()->SetFloatVector( vec.elems );
	}

	//------------------------------------------------------------------------

	template <>
	void
	D3D10EffectVarBindImpl<float3_vec>::SetValueImpl( const void * val )
	{
		setFloatArray<float3_vec>( GetBind(), val );
	}

	//------------------------------------------------------------------------

	template <>
	void
	D3D10EffectVarBindImpl<float4>::SetValueImpl( const void * val )
	{
		const float4& typedVal = *static_cast<const float4*>(val);
		float4 vec( typedVal );
		GetBind()->SetFloatVector( vec.elems );
	}

	//------------------------------------------------------------------------

	template <>
	void
	D3D10EffectVarBindImpl<float4_vec>::SetValueImpl( const void * val )
	{
		setFloatArray<float4_vec>( GetBind(), val );
	}

	//------------------------------------------------------------------------

	template <>
	void
	D3D10EffectVarBindImpl<float2x2>::SetValueImpl( const void * val )
	{
		const float2x2& typedVal = *(const float2x2*)val;

		D3DXMATRIX mat(	typedVal[0].x,	typedVal[0].y,	0,	0,
						typedVal[1].x,	typedVal[1].y,	0,	0,
						0,				0,				1,	0,
						0,				0,				0,	1 );
								
		GetBind()->SetMatrix(mat);
	}

	//------------------------------------------------------------------------

	template <>
	void
	D3D10EffectVarBindImpl<float3x3>::SetValueImpl( const void * val )
	{
		const float3x3& typedVal = *(const float3x3*)val;

		D3DXMATRIX mat(	typedVal[0].x,	typedVal[0].y,	typedVal[0].z,	0,
						typedVal[1].x,	typedVal[1].y,	typedVal[1].z,	0,
						typedVal[2].x,	typedVal[2].y,	typedVal[2].z,	0,
						0,				0,				0,				1 );
								
		GetBind()->SetMatrix(mat);
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
	D3D10EffectVarBindImpl<float4x4>::SetValueImpl( const void * val )
	{
		const float4x4& typedVal = *(const float4x4*)val;
		GetBind()->SetMatrix( toDXMatrix( typedVal) );
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
	D3D10EffectVarBindImpl<float3x4>::SetValueImpl( const void * val )
	{
		const float3x4& typedVal = *(const float3x4*)val;
		GetBind()->SetMatrix( toDXMatrix( typedVal) );
	}

	//------------------------------------------------------------------------

	template <>
	void
	D3D10EffectVarBindImpl<float4x4_vec>::SetValueImpl( const void * val )
	{
		const float4x4_vec& typedVal = *(const float4x4_vec*)val;

		if( checkArraySizes( GetBind(), typedVal ) )
		{
			size_t numMats = typedVal.size();

			D3DXMATRIX* dxmats = static_cast<D3DXMATRIX*> ( MD_STACK_ALLOC( sizeof( D3DXMATRIX ) * numMats ) );

			MD_FERROR_ON_FALSE( dxmats );

			for( size_t i = 0, e = numMats; i < e; ++i )
			{
				dxmats[i] = toDXMatrix( typedVal[i] );
			}

			GetBind()->SetMatrixArray( (float*)dxmats, 0, (UINT)numMats );

			MD_STACK_FREE( dxmats );
		}
	}

	//------------------------------------------------------------------------

	template <>
	void
	D3D10EffectVarBindImpl<float3x4_vec>::SetValueImpl( const void * val )
	{
		const float3x4_vec& typedVal = *(const float3x4_vec*)val;

		if( checkArraySizes( GetBind(), typedVal ) )
		{
			size_t numMats = typedVal.size();

			D3DXMATRIX* dxmats = static_cast<D3DXMATRIX*> ( MD_STACK_ALLOC( sizeof( D3DXMATRIX ) * numMats ) );

			MD_FERROR_ON_FALSE( dxmats );

			for( size_t i = 0, e = numMats; i < e; ++i )
			{
				dxmats[i] = toDXMatrix( typedVal[i] );
			}

			GetBind()->SetMatrixArray( (float*)dxmats, 0, (UINT)numMats );

			MD_STACK_FREE( dxmats );
		}
	}

	//------------------------------------------------------------------------

	template <>
	void
	D3D10EffectVarBindImpl<ShaderResource>::SetValueImpl( const void * val )
	{
		if( const ShaderResource* sr =  static_cast<const ShaderResource*>(val) )
			static_cast<const D3D10ShaderResource*>( sr )->BindTo(GetBind());			
		else
			D3D10ShaderResource::SetBindToZero( GetBind() );

	}

	//------------------------------------------------------------------------

	template <>
	void
	D3D10EffectVarBindImpl<Buffer>::SetValueImpl( const void * val )
	{
		if( const Buffer* buf =  static_cast<const Buffer*>(val) )
			static_cast<const D3D10Buffer*>( buf )->BindTo(GetBind());
		else
			D3D10Buffer::SetBindToZero( GetBind() );
	}

	//------------------------------------------------------------------------

	template <typename T>
	void
	D3D10EffectVarBindImpl<T>::UnbindImpl()
	{
		MD_FERROR( L"Can't unbind this bind type!");
	}

	//------------------------------------------------------------------------

	template <>
	void
	D3D10EffectVarBindImpl<ShaderResource>::UnbindImpl()
	{
		D3D10ShaderResource::SetBindToZero( GetBind() );
	}

	//------------------------------------------------------------------------

	template class D3D10EffectVarBindImpl<INT32>;
	template class D3D10EffectVarBindImpl<int_vec>;
	template class D3D10EffectVarBindImpl<int2>;
	template class D3D10EffectVarBindImpl<int3>;
	template class D3D10EffectVarBindImpl<int4>;
	template class D3D10EffectVarBindImpl<UINT32>;
	template class D3D10EffectVarBindImpl<uint_vec>;
	template class D3D10EffectVarBindImpl<uint2>;
	template class D3D10EffectVarBindImpl<uint3>;
	template class D3D10EffectVarBindImpl<uint4>;
	template class D3D10EffectVarBindImpl<float>;
	template class D3D10EffectVarBindImpl<float_vec>;
	template class D3D10EffectVarBindImpl<float2>;
	template class D3D10EffectVarBindImpl<float2_vec>;
	template class D3D10EffectVarBindImpl<float3>;
	template class D3D10EffectVarBindImpl<float3_vec>;
	template class D3D10EffectVarBindImpl<float4>;
	template class D3D10EffectVarBindImpl<float4_vec>;
	template class D3D10EffectVarBindImpl<float2x2>;
	template class D3D10EffectVarBindImpl<float3x3>;
	template class D3D10EffectVarBindImpl<float4x4>;
	template class D3D10EffectVarBindImpl<float3x4>;
	template class D3D10EffectVarBindImpl<float4x4_vec>;
	template class D3D10EffectVarBindImpl<float3x4_vec>;
	template class D3D10EffectVarBindImpl<ShaderResource>;
	template class D3D10EffectVarBindImpl<Buffer>;

}
