/**
 *	@file
 *	@brief		A basic matrix class (non-optimized).
 *	@author		Alan Chambers
 *	@date		2011
**/

#ifndef MATRIX4_H
#define MATRIX4_H

#include "Vector4.h"

class Matrix4
{
public:
								Matrix4( void );
								Matrix4( const Matrix4& mat );
	explicit					Matrix4( const float* mat );
	explicit					Matrix4( const Vector4& x_axis, const Vector4& y_axis, const Vector4& z_axis, const Vector4& w_axis );

	const Vector4&				GetTranslation( void ) const;
	const Vector4&				GetXAxis( void ) const;
	const Vector4&				GetYAxis( void ) const;
	const Vector4&				GetZAxis( void ) const;

	void						SetRotateX( float radians );
	void						SetRotateY( float radians );
	void						SetRotateZ( float radians );
	void						SetScale( float x, float y, float z );
	void						SetTranslation( const Vector4& trans );
	void						SetXAxis( const Vector4& x_axis );
	void						SetYAxis( const Vector4& y_axis );
	void						SetZAxis( const Vector4& z_axis );
	
	void						Transpose( void );

	Vector4						operator*( const Vector4& vec ) const;
	Matrix4						operator*( const Matrix4& mat ) const;
	Matrix4&					operator=( const Matrix4& mat );
	Matrix4&					operator*=( const Matrix4& mat );

	static const Matrix4		IDENTITY;
	static const Matrix4		ZERO;

private:
	Vector4						m_x;
	Vector4						m_y;
	Vector4						m_z;
	Vector4						m_t;
};

#include "Matrix4.inl"

#endif
