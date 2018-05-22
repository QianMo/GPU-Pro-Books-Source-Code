/**
 *	@file
 *	@brief		Basic vector class (non-optimized).
 *	@author		Alan Chambers
 *	@date		2011
**/

#ifndef VECTOR4_H
#define VECTOR4_H

#ifdef _OSX
#define ALIGN( x ) __attribute__( ( packed ( x ) ) )
#else
#define ALIGN( x )
#endif

class Vector4
{
public:
								Vector4( void );
								Vector4( const Vector4& vec );
								Vector4( float x, float y, float z, float w=0.0f );

	static Vector4				Cross( const Vector4& v1, const Vector4& v2 );

	static float				Dot( const Vector4& v1, const Vector4& v2 );

	float						GetX( void ) const;
	float						GetY( void ) const;
	float						GetZ( void ) const;
	float						GetW( void ) const;

	float						Length( void ) const;

	Vector4&					Normalize( void );

	void						Scale( float scale );
	void						SetX( float x );
	void						SetY( float y );
	void						SetZ( float z );
	void						SetW( float w );
	float						SquareLength( void ) const;
	
	Vector4&					operator=( const Vector4& vec );
	Vector4						operator*( const Vector4& vec ) const;
	Vector4						operator-( const Vector4& vec ) const;
	Vector4						operator+( const Vector4& vec ) const;
	Vector4						operator*( float scale ) const;
	
	static const Vector4		MAX;										//!> @brief FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX
	static const Vector4		MIN;										//!> @brief FLT_MIN, FLT_MIN, FLT_MIN, FLT_MIN
	static const Vector4		NEG_XAXIS;									//!> @brief -1, 0, 0, 0
	static const Vector4		NEG_YAXIS;									//!> @brief 0, -1, 0, 0
	static const Vector4		NEG_ZAXIS;									//!> @brief 0, 0, -1, 0
	static const Vector4		XAXIS;										//!> @brief 1, 0, 0, 0
	static const Vector4		YAXIS;										//!> @brief 0, 1, 0, 0
	static const Vector4		ZAXIS;										//!> @brief 0, 0, 1, 0
	static const Vector4		ORIGIN;										//!> @brief 0, 0, 0, 1
	static const Vector4		ZERO;										//!> @brief 0, 0, 0, 0
	
private:
	float						m_x;
	float						m_y;
	float						m_z;
	float						m_w;
}	ALIGN( 16 );

#include "Vector4.inl"

#endif
