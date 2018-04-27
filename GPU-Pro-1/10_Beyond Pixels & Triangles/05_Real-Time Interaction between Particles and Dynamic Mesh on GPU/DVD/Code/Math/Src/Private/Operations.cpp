#include "Precompiled.h"

#include "Frustum.h"

#include "Operations.h"

namespace Mod
{
	namespace Math
	{

		float3x4 to3x4( const float4x4& val )
		{
			float3x4 res;

			res[0] = val.elems[0];
			res[1] = val.elems[1];
			res[2] = val.elems[2];
			res[3] = val.elems[3];

			return res;
		}

		float4x4 to4x4( const float3x4& val )
		{
			float4x4 res;

			res[0] = float4(val.elems[0],0);
			res[1] = float4(val.elems[1],0);
			res[2] = float4(val.elems[2],0);
			res[3] = float4(val.elems[3],1);

			return res;
		}

		namespace
		{
#ifdef MD_USEDXROUTINGS
			D3DXMATRIX toDXMatrix( const float4x4& mat )
			{
				return D3DXMATRIX(	mat[0].x, mat[0].y, mat[0].z, mat[0].w,
									mat[1].x, mat[1].y, mat[1].z, mat[1].w,
									mat[2].x, mat[2].y, mat[2].z, mat[2].w,
									mat[3].x, mat[3].y, mat[3].z, mat[3].w );
			}

			D3DXMATRIX toDXMatrix( const float3x3& mat )
			{
				return D3DXMATRIX(	mat[0].x,	mat[0].y,	mat[0].z,	0,
									mat[1].x,	mat[1].y,	mat[1].z,	0,
									mat[2].x,	mat[2].y,	mat[2].z,	0,
									0,			0,			0,			1 );
			}

			D3DXMATRIX toDXMatrix( const float2x2& mat )
			{
				return D3DXMATRIX(	mat[0].x,	mat[0].y,	0,			0,
									mat[1].x,	mat[1].y,	0,			0,
									0,			0,			1,			0,
									0,			0,			0,			1 );
			}

			D3DXMATRIX toDXMatrix( const float3x4& mat )
			{
				return D3DXMATRIX(	mat[0].x,	mat[0].y,	mat[0].z,	0,
									mat[1].x,	mat[1].y,	mat[1].z,	0,
									mat[2].x,	mat[2].y,	mat[2].z,	0,
									mat[3].x,	mat[3].y,	mat[3].z,	1 );
			}

			float4x4 fromDXMatrix( const D3DXMATRIX& dxMat)
			{
				float4x4 res (
							dxMat.m[0][0], dxMat.m[0][1], dxMat.m[0][2], dxMat.m[0][3],
							dxMat.m[1][0], dxMat.m[1][1], dxMat.m[1][2], dxMat.m[1][3],
							dxMat.m[2][0], dxMat.m[2][1], dxMat.m[2][2], dxMat.m[2][3],
							dxMat.m[3][0], dxMat.m[3][1], dxMat.m[3][2], dxMat.m[3][3]  );

				return res;
			}

			float3x3 fromDXMatrixTo3x3( const D3DXMATRIX& dxMat)
			{
				float3x3 res (
						dxMat.m[0][0], dxMat.m[0][1], dxMat.m[0][2],
						dxMat.m[1][0], dxMat.m[1][1], dxMat.m[1][2],
						dxMat.m[2][0], dxMat.m[2][1], dxMat.m[2][2]	);
				return res;
			}

			float2x2 fromDXMatrixTo2x2( const D3DXMATRIX& dxMat)
			{
				float2x2 res (	dxMat.m[0][0], dxMat.m[0][1],
								dxMat.m[1][0], dxMat.m[1][1] );

				return res;
			}

			float3x4 fromDXMatrixTo3x4( const D3DXMATRIX& dxMat)
			{
				float3x4 res (
								dxMat.m[0][0], dxMat.m[0][1], dxMat.m[0][2],
								dxMat.m[1][0], dxMat.m[1][1], dxMat.m[1][2],
								dxMat.m[2][0], dxMat.m[2][1], dxMat.m[2][2],
								dxMat.m[3][0], dxMat.m[3][1], dxMat.m[3][2] );

				return res;
			}

			D3DXVECTOR4 toDXVec4( const float4& vec )
			{
				return D3DXVECTOR4( vec.x, vec.y, vec.z, vec.w );
			}

			D3DXVECTOR4 toDXVec4( const float3& vec )
			{
				return D3DXVECTOR4( vec.x, vec.y, vec.z, 1 );
			}

			D3DXVECTOR4 toDXVec4ZE( const float3& vec )
			{
				return D3DXVECTOR4( vec.x, vec.y, vec.z, 0 );
			}

			D3DXVECTOR4 toDXVec4( const float2& vec )
			{
				return D3DXVECTOR4( vec.x, vec.y, 0, 1 );
			}

			D3DXVECTOR3 toDXVec3( const float3& vec )
			{
				return D3DXVECTOR3( vec.x, vec.y, vec.z );
			}

			float4 fromDXVec4( const D3DXVECTOR4& vec )
			{
				return float4( vec.x, vec.y, vec.z, vec.w );
			}

			float3 fromDXVec3( const D3DXVECTOR3& vec )
			{
				return float3( vec.x, vec.y, vec.z );
			}

			float3 fromDXVec4To3( const D3DXVECTOR4& vec )
			{
				return float3( vec.x, vec.y, vec.z );
			}

			D3DXVECTOR2 toDXVec2( const float2& vec )
			{
				return D3DXVECTOR2( vec.x, vec.y );
			}

			float2 fromDXVec2( const D3DXVECTOR2& vec )
			{
				return float2( vec.x, vec.y );
			}

			D3DXQUATERNION toDXQuat( const float4& vec )
			{
				return D3DXQUATERNION( vec.x, vec.y, vec.z, vec.w );
			}

			float4 fromDXQuat( const D3DXQUATERNION& quat )
			{
				return float4( quat.x, quat.y, quat.z, quat.w );
			}
#endif
		}

#ifdef MD_USEDXROUTINGS
#define MD_DXMATRIXMULTIPLY									\
		D3DXMATRIX	dxMat1( toDXMatrix( mat1 ) ),			\
					dxMat2( toDXMatrix( mat2 ) ),			\
					dxResult;								\
		D3DXMatrixMultiply( &dxResult, &dxMat1, &dxMat2 );
#endif


		float4x4 mul( const float4x4& mat1, const float4x4& mat2 )
		{
#ifdef MD_USEDXROUTINGS
			MD_DXMATRIXMULTIPLY
			return fromDXMatrix( dxResult );
#else
			throw;
#endif
		}
		//------------------------------------------------------------------------

		float3x3 mul( const float3x3& mat1, const float3x3& mat2 )
		{
#ifdef MD_USEDXROUTINGS
			MD_DXMATRIXMULTIPLY
			return fromDXMatrixTo3x3( dxResult );
#else
			throw;
#endif
		}
		//------------------------------------------------------------------------

		float2x2 mul( const float2x2& mat1, const float2x2& mat2 )
		{
#ifdef MD_USEDXROUTINGS
			MD_DXMATRIXMULTIPLY
			return fromDXMatrixTo2x2( dxResult );
#else
			throw;
#endif
		}

		//------------------------------------------------------------------------

		float4x4 mul( const float3x4& mat1, const float4x4& mat2 )
		{
#ifdef MD_USEDXROUTINGS
			MD_DXMATRIXMULTIPLY
			return fromDXMatrix( dxResult );
#else
			throw;
#endif
		}

		//------------------------------------------------------------------------

		float3x4 mul( const float3x4& mat1, const float3x4& mat2 )
		{
#ifdef MD_USEDXROUTINGS
			MD_DXMATRIXMULTIPLY
			return fromDXMatrixTo3x4( dxResult );
#else
			throw;
#endif
		}

		//------------------------------------------------------------------------

		float4x4 mul( const float4x4& mat1, const float3x4& mat2 )
		{
#ifdef MD_USEDXROUTINGS
			MD_DXMATRIXMULTIPLY
			return fromDXMatrix( dxResult );
#else
			throw;
#endif
		}

#ifdef MD_DXMATRIXMULTIPLY
#undef MD_DXMATRIXMULTIPLY
#endif

		//------------------------------------------------------------------------

		float4 mul( const float4& vec, const float4x4& mat )
		{
#ifdef MD_USEDXROUTINGS
			D3DXVECTOR4 dxVec( toDXVec4( vec ) ), dxResult;
			D3DXMATRIX dxMat( toDXMatrix( mat ) );
			D3DXVec4Transform( &dxResult, &dxVec, &dxMat );

			return fromDXVec4( dxResult );
#else
			throw;
#endif
		}

		//------------------------------------------------------------------------

		float4 mul( const float3& vec, const float4x4& mat )
		{
#ifdef MD_USEDXROUTINGS
			D3DXVECTOR4 dxVec( toDXVec4ZE( vec ) ), dxResult;
			D3DXMATRIX dxMat( toDXMatrix( mat ) );
			D3DXVec4Transform( &dxResult, &dxVec, &dxMat );

			return fromDXVec4( dxResult );
#else
			throw;
#endif
		}

		//------------------------------------------------------------------------

		float4 mul( const float4& vec, const float3x4& mat )
		{
#ifdef MD_USEDXROUTINGS
			D3DXVECTOR4 dxVec( toDXVec4( vec ) ), dxResult;
			D3DXMATRIX dxMat( toDXMatrix( mat ) );
			D3DXVec4Transform( &dxResult, &dxVec, &dxMat );

			return fromDXVec4( dxResult );
#else
			throw;
#endif
		}

		//------------------------------------------------------------------------

		float4 mul( const float3& vec, const float3x4& mat )
		{
#ifdef MD_USEDXROUTINGS
			D3DXVECTOR4 dxVec( toDXVec4ZE( vec ) ), dxResult;
			D3DXMATRIX dxMat( toDXMatrix( mat ) );
			D3DXVec4Transform( &dxResult, &dxVec, &dxMat );

			return fromDXVec4( dxResult );
#else
			throw;
#endif
		}

		//------------------------------------------------------------------------

		float3 mul( const float3& vec, const float3x3& mat )
		{
#ifdef MD_USEDXROUTINGS
			D3DXVECTOR3 dxVec( toDXVec4( vec ) ), dxResult;
			D3DXMATRIX dxMat( toDXMatrix( mat ) );
			D3DXVec3TransformNormal( &dxResult, &dxVec, &dxMat );

			return fromDXVec3( dxResult );
#else
			throw;
#endif
		}
		//------------------------------------------------------------------------

		float2 mul( const float2& vec, const float2x2& mat )
		{
			return float2(	vec.x*mat[0][0] + vec.y*mat[1][0], 
							vec.x*mat[0][1] + vec.y*mat[1][1] );
		}

		//------------------------------------------------------------------------

#define MD_DEF_VEC_MUL_1D(t1,t2,t3)									\
		t1##_vec mul( const t2##_vec& vec, const t3& val )			\
		{															\
			t1##_vec res ( vec.size() );							\
																	\
			for( size_t i = 0, e = vec.size(); i < e; i++ )			\
			{														\
				res[i] = mul( vec[i], val );						\
			}														\
			return res;												\
		}

		MD_DEF_VEC_MUL_1D(float4,float3,float4x4)
		MD_DEF_VEC_MUL_1D(float4,float4,float4x4)
		MD_DEF_VEC_MUL_1D(float4,float3,float3x4)
		MD_DEF_VEC_MUL_1D(float4,float4,float3x4)


#define MD_DEF_VEC_MUL_2D_ASYM(type1,type2,type3)					\
		type1##_vec mul( const type2& val, const type3##_vec& vec )	\
		{															\
			type1##_vec res ( vec.size() );							\
																	\
			for( size_t i = 0, e = vec.size(); i < e; i++ )			\
			{														\
				res[i] = mul( val, vec[i]  );						\
			}														\
			return res;												\
		}


#define MD_DEF_VEC_MUL_2D(type)										\
		MD_DEF_VEC_MUL_1D(type,type,type)							\
		MD_DEF_VEC_MUL_2D_ASYM(type,type,type)

		MD_DEF_VEC_MUL_2D(float2x2)
		MD_DEF_VEC_MUL_2D(float3x3)
		MD_DEF_VEC_MUL_2D(float4x4)

		MD_DEF_VEC_MUL_2D(float3x4)
		
		MD_DEF_VEC_MUL_1D(float4x4,float3x4,float4x4)
		MD_DEF_VEC_MUL_2D_ASYM(float4x4,float4x4,float3x4)

#undef MD_DEF_VEC_MUL_2D
#undef MD_DEF_VEC_MUL_1D

		//------------------------------------------------------------------------

		Frustum	transform( const Frustum& frustum, const float3x4& mat )
		{
			Frustum result;
			float4x4 intTMat = transpose( to4x4( inverse( mat ) ) );

			for( UINT32 i = 0; i < Frustum::NUM_PLANES; i ++ )
			{
				result.planes[i] = mul( frustum.planes[i], mat );
			}

			return result;
		}

		//------------------------------------------------------------------------
		

		float dot( const float4& vec1, const float4& vec2 )
		{
#ifdef MD_USEDXROUTINGS
			D3DXVECTOR4		dxVec1( toDXVec4( vec1 ) ),
							dxVec2( toDXVec4( vec2 ) );

			return D3DXVec4Dot( &dxVec1, &dxVec2 );
#else
			throw;
#endif
		}
		//------------------------------------------------------------------------

		float dot( const float3& vec1, const float3& vec2 )
		{
#ifdef MD_USEDXROUTINGS
			D3DXVECTOR3		dxVec1( toDXVec3( vec1 ) ),
							dxVec2( toDXVec3( vec2 ) );
			return D3DXVec3Dot( &dxVec1, &dxVec2 );
#else
			throw;
#endif
		}

		//------------------------------------------------------------------------

		float dot( const float2& vec1, const float2& vec2 )
		{
			return vec1.x*vec2.x + vec1.y*vec2.y;
		}

		//------------------------------------------------------------------------

		float4	lerp( float4 a,	float4 b, float t )
		{
#ifdef MD_USEDXROUTINGS
			D3DXVECTOR4 dxA( toDXVec4(a) ), dxB( toDXVec4( b ) ),
						dxRes;
			D3DXVec4Lerp( &dxRes, &dxA, &dxB, t );

			return fromDXVec4( dxRes );
#else
			throw;
#endif
		}

		//------------------------------------------------------------------------

		float3 lerp( float3 a,	float3 b,	float t )
		{
#ifdef MD_USEDXROUTINGS
			D3DXVECTOR3 dxA( toDXVec3(a) ), dxB( toDXVec3( b ) ),
						dxRes;
			D3DXVec3Lerp( &dxRes, &dxA, &dxB, t );

			return fromDXVec3( dxRes );
#else
			throw;
#endif
		}

		//------------------------------------------------------------------------
 
		float2 lerp( float2 a,	float2 b,	float t )
		{
#ifdef MD_USEDXROUTINGS
			D3DXVECTOR2 dxA( toDXVec2(a) ), dxB( toDXVec2( b ) ),
						dxRes;
			D3DXVec2Lerp( &dxRes, &dxA, &dxB, t );

			return fromDXVec2( dxRes );
#else
			throw;
#endif
		}

		//------------------------------------------------------------------------

		float lerp( float a, float b, float t )
		{
			return a + ( b - a ) * t;
		}

		//------------------------------------------------------------------------

		float4 quatSLerp( float4 a, float4 b, float t )
		{
#ifdef MD_USEDXROUTINGS
			D3DXQUATERNION	dxA( toDXQuat( a ) ), 
							dxB( toDXQuat( b ) ),
							dxRes;

			D3DXQuaternionSlerp( &dxRes, &dxA, &dxB, t );

			return fromDXQuat( dxRes );
#else
			throw;
#endif
		}

		//------------------------------------------------------------------------

		float4 normalize( const float4& vec )
		{
#ifdef MD_USEDXROUTINGS
			D3DXVECTOR4 dxVec( toDXVec4( vec ) );
			D3DXVec4Normalize( &dxVec, &dxVec );
			return fromDXVec4( dxVec );
#else
			throw;
#endif
		}
		//------------------------------------------------------------------------

		float3 normalize( const float3& vec )
		{
#ifdef MD_USEDXROUTINGS
			D3DXVECTOR3 dxVec( toDXVec3( vec ) );
			D3DXVec3Normalize( &dxVec, &dxVec );
			return fromDXVec3( dxVec );
#else
			throw;
#endif
		}
		//------------------------------------------------------------------------

		float2 normalize( const float2& vec )
		{
#ifdef MD_USEDXROUTINGS
			D3DXVECTOR2 dxVec( toDXVec2( vec ) );
			D3DXVec2Normalize( &dxVec, &dxVec );
			return fromDXVec2( dxVec );
#else
			throw;
#endif
		}

		//------------------------------------------------------------------------

#define MD_DEF_NORM_VEC(type)									\
		type normalize( const type& vec )						\
		{														\
			type result( vec.size() );							\
			for( size_t i = 0, e = vec.size(); i < e; i ++ )	\
			{													\
				result [ i ] = normalize( vec[ i ] );			\
			}													\
			return result;										\
		}

		MD_DEF_NORM_VEC(float4_vec)
		MD_DEF_NORM_VEC(float3_vec)
		MD_DEF_NORM_VEC(float2_vec)

#undef MD_DEF_NORM_VEC


		//------------------------------------------------------------------------

		float length( const float4& vec )
		{
#ifdef MD_USEDXROUTINGS
			D3DXVECTOR4 dxVec( toDXVec4( vec ) );
			return D3DXVec4Length( &dxVec );
#else
			throw;
#endif

		}

		//------------------------------------------------------------------------

		float length( const float3& vec )
		{
#ifdef MD_USEDXROUTINGS
			D3DXVECTOR3 dxVec( toDXVec3( vec ) );
			return D3DXVec3Length( &dxVec );
#else
			throw;
#endif

		}
		//------------------------------------------------------------------------

		float length( const float2& vec )
		{
#ifdef MD_USEDXROUTINGS
			D3DXVECTOR2 dxVec( toDXVec2( vec ) );
			return D3DXVec2Length( &dxVec );
#else
			throw;
#endif
		}
		//------------------------------------------------------------------------


		float3 cross( const float3& vec1, const float3& vec2 )
		{
#ifdef MD_USEDXROUTINGS
			D3DXVECTOR3		dxVec1( toDXVec3( vec1 ) ),
							dxVec2( toDXVec3( vec2 ) ),
							dxResult;
			D3DXVec3Cross( &dxResult, &dxVec1, &dxVec2 );
			return fromDXVec3( dxResult );
#else
			throw;
#endif

		}

		//------------------------------------------------------------------------

		float3 rotate( const float3& vec, const float4& quat )
		{
			float3 uv, uuv;
			float3 qvec = quat; 
			uv	= cross( qvec, vec );
			uuv	= cross( qvec, uv );
			uv	= uv * (2.0f * quat.w);
			uuv	= uuv * 2.0f;

			return vec + uv + uuv;
		}

		//------------------------------------------------------------------------

		float3 getScale( const float4x4& m )
		{
			return float3(	length( float3( m[0][0], m[1][0], m[2][0] ) ),
							length( float3( m[0][1], m[1][1], m[2][1] ) ),
							length( float3( m[0][2], m[1][2], m[2][2] ) ) );
		}

		//------------------------------------------------------------------------

		void m3x4Decompose( const float3x4& m, float3& oT, float4& oRot, float3& oScale )
		{
#ifdef MD_USEDXROUTINGS
			D3DXMATRIX dxMat( toDXMatrix( m ) );
			D3DXVECTOR3 dxT, dxS;
			D3DXQUATERNION dxQuat;

			D3DXMatrixDecompose( &dxS, &dxQuat, &dxT, &dxMat );

			oT = fromDXVec3( dxT );
			oRot = fromDXQuat( dxQuat );
			oScale = fromDXVec3( dxS );
#else
			throw;
#endif
		}

		//------------------------------------------------------------------------

		float4x4 m4x4PerspProj( float w, float h, float zn, float zf )
		{

#ifdef MD_USEDXROUTINGS
			D3DXMATRIX dxMat;
			D3DXMatrixPerspectiveLH( &dxMat, w, h, zn, zf );
			return fromDXMatrix( dxMat );
#else
			throw;
#endif
		}

		//------------------------------------------------------------------------

		float4x4 m4x4OrthoProj( float w, float h, float zn, float zf )
		{
#ifdef MD_USEDXROUTINGS
			D3DXMATRIX dxMat;
			D3DXMatrixOrthoLH( &dxMat, w, h, zn, zf );
			return fromDXMatrix( dxMat );
#else
			throw;
#endif
		}

		//------------------------------------------------------------------------

		float3x4 m3x4RotAxisToAxis( const float3& axisFrom, const float3& axisTo )
		{
			return m3x4RotQuat( quatRotAxisToAxis( axisFrom, axisTo ) );
		}

		//------------------------------------------------------------------------

		float3x4 m3x4RotAxisAngle( const float3& axis, float angle )
		{
#ifdef MD_USEDXROUTINGS
			D3DXMATRIX	dxMat;
			D3DXVECTOR3	dxAxis( toDXVec3( axis ) );
			D3DXMatrixRotationAxis( &dxMat, &dxAxis, angle );
			return fromDXMatrixTo3x4( dxMat );
#else
			throw;
#endif
		}

		//------------------------------------------------------------------------

		float3x4 m3x4RotQuat( const float4& quat)
		{
#ifdef MD_USEDXROUTINGS
			D3DXMATRIX		dxMat;
			D3DXQUATERNION	dxQuat( toDXQuat( quat ) );
			D3DXMatrixRotationQuaternion( &dxMat, &dxQuat );
			return fromDXMatrixTo3x4( dxMat );
#else
			throw;
#endif
		}

		//------------------------------------------------------------------------

		float3x4 m3x4RotX( float alpha )
		{
#ifdef MD_USEDXROUTINGS
			D3DXMATRIX dxMat;
			D3DXMatrixRotationX( &dxMat, alpha );
			return fromDXMatrixTo3x4( dxMat );
#else
			throw;
#endif
		}

		//------------------------------------------------------------------------

		float3x4 m3x4RotY( float alpha )
		{
#ifdef MD_USEDXROUTINGS
			D3DXMATRIX dxMat;
			D3DXMatrixRotationY( &dxMat, alpha );
			return fromDXMatrixTo3x4( dxMat );
#else
			throw;
#endif
		}

		//------------------------------------------------------------------------

		float3x4 m3x4RotZ( float alpha )
		{
#ifdef MD_USEDXROUTINGS
			D3DXMATRIX dxMat;
			D3DXMatrixRotationZ( &dxMat, alpha );
			return fromDXMatrixTo3x4( dxMat );
#else
			throw;
#endif
		}
		//------------------------------------------------------------------------

		float3x4 m3x4RotYawPitchRoll( float yaw, float pitch, float roll )
		{
#ifdef MD_USEDXROUTINGS
			D3DXMATRIX dxMat;
			D3DXMatrixRotationYawPitchRoll( &dxMat, yaw, pitch, roll );
			return fromDXMatrixTo3x4( dxMat );
#else
			throw;
#endif
		}

		//------------------------------------------------------------------------

		float3x4 m3x4Scale( float x, float y, float z )
		{
#ifdef MD_USEDXROUTINGS
			D3DXMATRIX dxMat;
			D3DXMatrixScaling( &dxMat, x, y, z );
			return fromDXMatrixTo3x4( dxMat );
#else
			throw;
#endif
		}

		//------------------------------------------------------------------------

		float3x4 m3x4Transform( const float3& scale, const float4& orient, const float3& translation )
		{
#ifdef MD_USEDXROUTINGS
			D3DXMATRIX		dxMat;
			D3DXVECTOR3		dxScale	( toDXVec3( scale		) ),
							dxTrans	( toDXVec3( translation	) );
			D3DXQUATERNION	dxRot	( toDXQuat( orient		) );
			
			D3DXMatrixTransformation( &dxMat, NULL, NULL, &dxScale, NULL, &dxRot, &dxTrans );
			return fromDXMatrixTo3x4( dxMat );
#else
			throw;
#endif
		}

		//------------------------------------------------------------------------

		float3x4 m3x4View( const float3& pos, float rotX, float rotY, float rotZ )
		{
			float3x4 res = m3x4Translate( -pos );

			res = mul( res, m3x4RotX( rotX ) );
			res = mul( res, m3x4RotY( rotY ) );
			res = mul( res, m3x4RotZ( rotZ ) );

			return res;
		}

		//------------------------------------------------------------------------

		void m3x4SetCol( float3x4& oMat, UINT32 colIdx, const float4& val )
		{
			oMat[0][colIdx] = val.x;
			oMat[1][colIdx] = val.y;
			oMat[2][colIdx] = val.z;
			oMat[3][colIdx] = val.w;
		}

		//------------------------------------------------------------------------

		void m3x3SetCol( float3x3& oMat, UINT32 colIdx, const float3& val )
		{
			oMat[ 0 ][ colIdx ] = val.x;
			oMat[ 1 ][ colIdx ] = val.y;
			oMat[ 2 ][ colIdx ] = val.z;
		}

		//------------------------------------------------------------------------

		float3x4 m3x4Translate( const float3& dir )
		{
			return float3x4(	1,		0,		0,		
								0,		1,		0,		
								0,		0,		1,		
								dir.x,	dir.y,	dir.z );
		}

		//------------------------------------------------------------------------

		float4x4 inverse( const float4x4& mat )
		{
#ifdef MD_USEDXROUTINGS
			D3DXMATRIX	dxMat( toDXMatrix( mat ) ),
						dxResult;
						
			D3DXMatrixInverse( &dxResult, NULL, &dxMat );
			return fromDXMatrix( dxResult );
#else
			throw;
#endif
		}

		//------------------------------------------------------------------------

		float3x4 inverse( const float3x4& mat )
		{
#ifdef MD_USEDXROUTINGS
			D3DXMATRIX	dxMat( toDXMatrix( mat ) ),
						dxResult;
						
			D3DXMatrixInverse( &dxResult, NULL, &dxMat );
			return fromDXMatrixTo3x4( dxResult );
#else
			throw;
#endif
		}

		//------------------------------------------------------------------------

		float4x4 transpose( const float4x4& mat )
		{
			return float4x4(
					mat[0][0], mat[1][0], mat[2][0], mat[3][0],
					mat[0][1], mat[1][1], mat[2][1], mat[3][1],
					mat[0][2], mat[1][2], mat[2][2], mat[3][2],
					mat[0][3], mat[1][3], mat[2][3], mat[3][3]
				);
		}

		//------------------------------------------------------------------------

		float3x4 transpose( const float3x4& mat )
		{
			return float3x4(
					mat[0][0],	mat[1][0],	mat[2][0],
					mat[0][1],	mat[1][1],	mat[2][1],
					mat[0][2],	mat[1][2],	mat[2][2],
					0,			0,			0
				);
		}

		//------------------------------------------------------------------------

		float4x4_vec inverse( const float4x4_vec& mat_arr )
		{
			float4x4_vec res( mat_arr.size() );

			for( size_t i = 0, e = mat_arr.size(); i < e; i++ )
			{
				res[ i ] = inverse( mat_arr[ i ] );
			}

			return res;
		}

		//------------------------------------------------------------------------

		float3x4_vec inverse( const float3x4_vec& mat_arr )
		{
			float3x4_vec res( mat_arr.size() );

			for( size_t i = 0, e = mat_arr.size(); i < e; i++ )
			{
				res[ i ] = inverse( mat_arr[ i ] );
			}

			return res;
		}

		//------------------------------------------------------------------------

		float4 quatRotAxisToAxis( const float3& axisFrom, const float3& axisTo )
		{
			float3 axis = cross( axisFrom, axisTo );
			float3 m = axisFrom + axisTo;

			float la = length( axis );
			float lm = length( m );

			float eps = std::numeric_limits<float>::epsilon();

			float sin_a;
			float cos_a;

			if( la <= eps )
			{
				if( lm <= eps )
				{
					if( fabs( axisFrom.x ) > eps )
						axis = cross( axisFrom, float3(0,1,0) );
					else
						axis = cross( axisFrom, float3(1,0,0) );

					cos_a = 0.f;
					sin_a = 1.f;
				}
				else
				{
					cos_a = 1.f;
					sin_a = 0.f;
				}
			}
			else
			{
				axis	/= la;
				m		/= lm;

				cos_a = dot( m, axisFrom );
				sin_a = length( cross( axisFrom, m ) );
			}

			return float4( sin_a * axis.x, sin_a * axis.y, sin_a * axis.z, cos_a );
		}

		//------------------------------------------------------------------------

		float4 quatRotMat( const float3x4& mat )
		{
#ifdef MD_USEDXROUTINGS
			D3DXMATRIX	dxMat( toDXMatrix( mat ) );
			D3DXQUATERNION dxResult;

			D3DXQuaternionRotationMatrix( &dxResult, &dxMat );
			return fromDXQuat( dxResult );
#else
			throw;
#endif
		}

		//------------------------------------------------------------------------

		float4 quatRotAxisAngle( const float3& axis, float angle )
		{
#ifdef MD_USEDXROUTINGS
			D3DXVECTOR3 dxVec( toDXVec3( axis ) );
			D3DXQUATERNION dxResult;

			D3DXQuaternionRotationAxis( &dxResult, &dxVec, angle );
			return fromDXQuat( dxResult );
#else
			throw;
#endif
		}

		//------------------------------------------------------------------------

		float4 quatRotYawPitchRoll( float yaw, float pitch, float roll )
		{
#ifdef MD_USEDXROUTINGS
			D3DXQUATERNION dxResult;

			D3DXQuaternionRotationYawPitchRoll( &dxResult, yaw, pitch, roll );
			return fromDXQuat( dxResult );
#else
			throw;
#endif
		}

		//------------------------------------------------------------------------

		float4 quatInverse( const float4& quat )
		{
#ifdef MD_USEDXROUTINGS
			D3DXQUATERNION	dxQuat( toDXQuat( quat ) ),
							dxResult;

			D3DXQuaternionInverse( &dxResult, &dxQuat );
			return fromDXQuat( dxResult );
#else
			throw;
#endif
		}

		//------------------------------------------------------------------------

		void
		quatDecompose( const float4& quat, float3& oAxis, float &oAngle )
		{
#ifdef MD_USEDXROUTINGS
			D3DXQUATERNION	dxQuat( toDXQuat( quat ) );
			D3DXVECTOR3		dxResult;

			D3DXQuaternionToAxisAngle( &dxQuat, &dxResult, &oAngle );
			oAxis = fromDXVec3( dxResult );
#else
			throw;
#endif
		}

		//------------------------------------------------------------------------

		float saturate( float val )
		{
			return std::max( std::min( val, 1.f ), 0.f );
		}

		//------------------------------------------------------------------------

		float2 saturate( const float2& val )
		{
			return float2( saturate( val.x ), saturate( val.y ) );
		}

		//------------------------------------------------------------------------

		float3 saturate( const float3& val )
		{
			return float3( saturate( val.x ), saturate( val.y ), saturate( val.z ) );
		}

		//------------------------------------------------------------------------

		float4 saturate( const float4& val )
		{
			return float4(	saturate( val.x ), saturate( val.y ), 
							saturate( val.z ), saturate( val.w ) );
		}

		//------------------------------------------------------------------------

		bool equal( const float4x4& m1, const float4x4& m2, float tolerance )
		{
			for( UINT32 i = 0; i < 4; i ++ )
			for( UINT32 j = 0; j < 4; j ++ )
			{
				if( fabs( m1[i][j] - m2[i][j] ) > tolerance )
					return false;
			}

			return true;
		}

		//------------------------------------------------------------------------

		bool equal( const float3x4& m1, const float3x4& m2, float tolerance )
		{
			for( UINT32 i = 0; i < 4; i ++ )
			for( UINT32 j = 0; j < 3; j ++ )
			{
				if( fabs( m1[i][j] - m2[i][j] ) > tolerance )
					return false;
			}

			return true;
		}

		//------------------------------------------------------------------------

#define MD_DEFINE_COMP_OP(T,func)										\
		T func( T a, T b )												\
		{																\
			T res;														\
			for( int i = 0; i < T::COMPONENT_COUNT; i ++ )				\
			{															\
				res.elems[ i ] = std::func( a.elems[i], b.elems[i] );	\
			}															\
			return res;													\
		}

		MD_DEFINE_COMP_OP(float2,max)
		MD_DEFINE_COMP_OP(float3,max)
		MD_DEFINE_COMP_OP(float4,max)

		MD_DEFINE_COMP_OP(int2,max)
		MD_DEFINE_COMP_OP(int3,max)
		MD_DEFINE_COMP_OP(int4,max)

		MD_DEFINE_COMP_OP(uint2,max)
		MD_DEFINE_COMP_OP(uint3,max)
		MD_DEFINE_COMP_OP(uint4,max)


		MD_DEFINE_COMP_OP(float2,min)
		MD_DEFINE_COMP_OP(float3,min)
		MD_DEFINE_COMP_OP(float4,min)

		MD_DEFINE_COMP_OP(int2,min)
		MD_DEFINE_COMP_OP(int3,min)
		MD_DEFINE_COMP_OP(int4,min)

		MD_DEFINE_COMP_OP(uint2,min)
		MD_DEFINE_COMP_OP(uint3,min)
		MD_DEFINE_COMP_OP(uint4,min)

#undef MD_DEFINE_COMP_OP

		//------------------------------------------------------------------------

		half	ftoh( float val )
		{
			half tmp;
			tmp.FromFloat( val );
			return tmp;
		}

		//------------------------------------------------------------------------

		float	htof( half val )
		{
			return val.AsFloat();
		}


		//------------------------------------------------------------------------

#define MD_DEFINE_F_H_VECTOR_CONVERTION(t1,t2,num,func)	\
		t1##num	func( const t2##num& val )				\
		{												\
			t1##num tmp;								\
			for( int i = 0; i < num; i++)				\
				tmp.elems[i] = func( val.elems[i] );	\
			return tmp;									\
		}

		MD_DEFINE_F_H_VECTOR_CONVERTION(half,float,2,ftoh)
		MD_DEFINE_F_H_VECTOR_CONVERTION(half,float,3,ftoh)
		MD_DEFINE_F_H_VECTOR_CONVERTION(half,float,4,ftoh)

		MD_DEFINE_F_H_VECTOR_CONVERTION(float,half,2,htof)
		MD_DEFINE_F_H_VECTOR_CONVERTION(float,half,3,htof)
		MD_DEFINE_F_H_VECTOR_CONVERTION(float,half,4,htof)

#undef MD_DEFINE_F_H_VECTOR_CONVERTION

		//------------------------------------------------------------------------



		//------------------------------------------------------------------------

		const float4x4 IDENTITY_4x4 (
						1, 0, 0, 0,
						0, 1, 0, 0,
						0, 0, 1, 0,
						0, 0, 0, 1 );

		const float3x4 IDENTITY_3x4 (
						1, 0, 0,
						0, 1, 0,
						0, 0, 1,
						0, 0, 0 );

		const float4 IDENTITY_QUAT(	0, 0, 0, 1 );
	}
}