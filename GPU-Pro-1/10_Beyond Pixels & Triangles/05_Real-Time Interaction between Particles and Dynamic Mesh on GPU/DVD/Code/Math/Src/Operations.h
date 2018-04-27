#ifndef MATH_OPERATIONS_H_INCLUDED
#define MATH_OPERATIONS_H_INCLUDED

#include "Types.h"
#include "BBox.h"

namespace Mod
{
	namespace Math
	{
		float3x4 to3x4( const float4x4& val );
		float4x4 to4x4( const float3x4& val );

		float4x4 mul( const float4x4& mat1, const float4x4& mat2 );
		float3x3 mul( const float3x3& mat1, const float3x3& mat2 );
		float2x2 mul( const float2x2& mat1, const float2x2& mat2 );

		float4x4 mul( const float3x4& mat1, const float4x4& mat2 );
		float4x4 mul( const float4x4& mat1, const float3x4& mat2 );

		float3x4 mul( const float3x4& mat1, const float3x4& mat2 );

		float4 mul( const float4& vec, const float4x4& mat );
		float4 mul( const float3& vec, const float4x4& mat );

		float4 mul( const float4& vec, const float3x4& mat );
		float4 mul( const float3& vec, const float3x4& mat );

		float3 mul( const float3& vec, const float3x3& mat );
		float2 mul( const float2& vec, const float2x2& mat );

		float4_vec mul( const float3_vec& vec, const float4x4& mat );
		float4_vec mul( const float4_vec& vec, const float4x4& mat );

		float4_vec mul( const float3_vec& vec, const float3x4& mat );
		float4_vec mul( const float4_vec& vec, const float3x4& mat );

		float2x2_vec	mul( const float2x2_vec& vec, const float2x2& val );
		float3x3_vec	mul( const float3x3_vec& vec, const float3x3& val );
		float4x4_vec	mul( const float4x4_vec& vec, const float4x4& val );

		float4x4_vec	mul( const float3x4_vec& vec, const float4x4& val );
		float3x4_vec	mul( const float3x4_vec& vec, const float3x4& val );

		float2x2_vec	mul( const float2x2& val, const float2x2_vec& vec );
		float3x3_vec	mul( const float3x3& val, const float3x3_vec& vec );
		float4x4_vec	mul( const float4x4& val, const float4x4_vec& vec );

		float4x4_vec	mul( const float4x4& val, const float3x4_vec& vec );
		float4x4_vec	mul( const float3x4& val, const float4x4_vec& vec );
		float3x4_vec	mul( const float3x4& val, const float3x4_vec& vec );


		Frustum			transform( const Frustum& frustum, const float4x4& mat );

		float dot( const float4& vec1, const float4& vec2 );
		float dot( const float3& vec1, const float3& vec2 );
		float dot( const float2& vec1, const float2& vec2 );

		float4 normalize( const float4& vec );
		float3 normalize( const float3& vec );
		float2 normalize( const float2& vec );

		float4_vec normalize( const float4_vec& vec );
		float3_vec normalize( const float3_vec& vec );
		float2_vec normalize( const float2_vec& vec );

		float4	lerp( float4 a,	float4 b,	float t );
		float3	lerp( float3 a,	float3 b,	float t );
		float2	lerp( float2 a,	float2 b,	float t );
		float	lerp( float a,	float b,	float t );

		float4 quatSLerp( float4 a, float4 b, float t );

		float length( const float4& vec );
		float length( const float3& vec );
		float length( const float2& vec );

		float3 cross( const float3& vec1, const float3& vec2 );
		float3 rotate( const float3& vec, const float4& quat );

		// NOTE: post rotational scale is returned
		float3 getScale( const float4x4& m );
		void m3x4Decompose( const float3x4& m, float3& oT, float4& oRot, float3& oScale );
		float4x4 m4x4PerspProj( float w, float h, float zn, float zf );
		float4x4 m4x4OrthoProj( float w, float h, float zn, float zf );
		float3x4 m3x4RotAxisToAxis( const float3& axisFrom, const float3& axisTo );
		float3x4 m3x4RotAxisAngle( const float3& axis, float angle );
		float3x4 m3x4RotQuat( const float4& quat);
		float3x4 m3x4RotX( float alpha );
		float3x4 m3x4RotY( float alpha );
		float3x4 m3x4RotZ( float alpha );
		float3x4 m3x4RotYawPitchRoll( float yaw, float pitch, float roll );
		float3x4 m3x4Scale( float x, float y, float z );
		float3x4 m3x4Translate( const float3& dir );
		float3x4 m3x4Transform( const float3& scale, const float4& orient, const float3& translation );
		float3x4 m3x4View( const float3& pos, float rotX, float rotY, float rotZ );

		void m3x4SetCol( float3x4& oMat, UINT32 colIdx, const float4& val );
		void m3x3SetCol( float3x3& oMat, UINT32 colIdx, const float3& val );

		float4x4 inverse( const float4x4& mat );
		float3x4 inverse( const float3x4& mat );

		float4x4 transpose( const float4x4& mat );
		float3x4 transpose( const float3x4& mat );

		float4x4_vec inverse( const float4x4_vec& mat_arr );
		float3x4_vec inverse( const float3x4_vec& mat_arr );

		// warning: assumes axisFrom & axisTo to be normalized
		float4 quatRotAxisToAxis( const float3& axisFrom, const float3& axisTo );
		float4 quatRotMat( const float3x4& mat );
		float4 quatRotAxisAngle( const float3& axis, float angle );
		float4 quatRotYawPitchRoll( float yaw, float pitch, float roll );
		float4 quatInverse( const float4& quat );

		void quatDecompose( const float4& quat, float3& oAxis, float& oAngle );

		//------------------------------------------------------------------------

		float saturate( float val );
		float2 saturate( const float2& val );
		float3 saturate( const float3& val );
		float4 saturate( const float4& val );

		//------------------------------------------------------------------------

		bool equal( const float4x4& m1, const float4x4& m2, float tolerance );
		bool equal( const float3x4& m1, const float3x4& m2, float tolerance );

		//------------------------------------------------------------------------

		float2 max( float2 a, float2 b );
		float3 max( float3 a, float3 b );
		float4 max( float4 a, float4 b );

		int2 max( int2 a, int2 b );
		int3 max( int3 a, int3 b );
		int4 max( int4 a, int4 b );

		uint2 max( uint2 a, uint2 b );
		uint3 max( uint3 a, uint3 b );
		uint4 max( uint4 a, uint4 b );

		//------------------------------------------------------------------------

		float2 min( float2 a, float2 b );
		float3 min( float3 a, float3 b );
		float4 min( float4 a, float4 b );

		int2 min( int2 a, int2 b );
		int3 min( int3 a, int3 b );
		int4 min( int4 a, int4 b );

		uint2 min( uint2 a, uint2 b );
		uint3 min( uint3 a, uint3 b );
		uint4 min( uint4 a, uint4 b );

		//------------------------------------------------------------------------
		// half/float convertions
		half	ftoh( float val );
		half2	ftoh( const float2& val );
		half3	ftoh( const float3& val );
		half4	ftoh( const float4& val );

		float	htof( half val );
		float2	htof( const half2& val );
		float3	htof( const half3& val );
		float4	htof( const half4& val );

		//------------------------------------------------------------------------
		// constants

		extern const float4x4 IDENTITY_4x4;
		extern const float3x4 IDENTITY_3x4;
		extern const float4 IDENTITY_QUAT;

		namespace
		{
			const float HALF_PI_F = float(0.5*3.14159265358979323846);
			const float PI_F = 3.14159265358979323846f;
			const float TWO_PI_F = float(2*3.14159265358979323846);
			const float SQRT_2_F = 1.4142135623730950488f;
			const float HALF_SQRT_2_F = float(0.5*1.4142135623730950488);

			const float3 ZERO_3(0,0,0);
			const float4 ZERO_4(0,0,0,0);
			const float4x4 ZERO_4x4( 0,0,0,0,  0,0,0,0, 0,0,0,0, 0,0,0,0 );
			const float3x4 ZERO_3x4( 0,0,0,  0,0,0, 0,0,0, 0,0,0 );
			const float3x3 ZERO_3x3( 0,0,0, 0,0,0, 0,0,0 );
			const float2x2 ZERO_2x2( 0,0, 0,0  );

			const float FLOAT_MAX = std::numeric_limits<float>::max();

			const float3 MAX_3( FLOAT_MAX, FLOAT_MAX, FLOAT_MAX );

			const BBox BBOX_GROW_INIT( MAX_3, -MAX_3 );
		}
	}
}

#endif