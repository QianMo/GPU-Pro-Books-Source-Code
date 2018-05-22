#include "Matrix4.h"

#include <cstdio>
#include <cmath>
#include <cfloat>

//---------------------------------------------------------------------------------------------------------------------------------

const Vector4 Vector4::MAX( FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX );
const Vector4 Vector4::MIN( FLT_MIN, FLT_MIN, FLT_MIN, FLT_MIN );
const Vector4 Vector4::NEG_XAXIS( -1.0f, 0.0f, 0.0f, 0.0f );
const Vector4 Vector4::NEG_YAXIS( 0.0f, -1.0f, 0.0f, 0.0f );
const Vector4 Vector4::NEG_ZAXIS( 0.0f, 0.0f, -1.0f, 0.0f );
const Vector4 Vector4::XAXIS( 1.0f, 0.0f, 0.0f, 0.0f );
const Vector4 Vector4::YAXIS( 0.0f, 1.0f, 0.0f, 0.0f );
const Vector4 Vector4::ZAXIS( 0.0f, 0.0f, 1.0f, 0.0f );
const Vector4 Vector4::ORIGIN( 0.0f, 0.0f, 0.0f, 1.0f );
const Vector4 Vector4::ZERO( 0.0f, 0.0f, 0.0f, 0.0f );

//---------------------------------------------------------------------------------------------------------------------------------

const Matrix4 Matrix4::IDENTITY( Vector4::XAXIS, Vector4::YAXIS, Vector4::ZAXIS, Vector4::ORIGIN );
const Matrix4 Matrix4::ZERO( Vector4::ZERO, Vector4::ZERO, Vector4::ZERO, Vector4::ZERO );

//---------------------------------------------------------------------------------------------------------------------------------

void Matrix4::SetRotateX( float radians )
{
	const float cosr = cos( radians );
	const float sinr = sin( radians );

	m_x = Vector4( 1.0f, 0.0f, 0.0f );
	m_y = Vector4( 0.0f, cosr, -sinr );
	m_z = Vector4( 0.0f, -sinr, cosr );
}

//---------------------------------------------------------------------------------------------------------------------------------

void Matrix4::SetRotateY( float radians )
{
	const float cosr = cos( radians );
	const float sinr = sin( radians );

	m_x = Vector4( cosr, 0.0f, sinr );
	m_y = Vector4( 0.0f, 1.0f, 0.0f );
	m_z = Vector4( -sinr, 0.0f, cosr );
}

//---------------------------------------------------------------------------------------------------------------------------------

void Matrix4::SetRotateZ( float radians )
{
	const float cosr = cos( radians );
	const float sinr = sin( radians );

	m_x = Vector4( cosr, -sinr, 0.0f );
	m_y = Vector4( sinr, cosr, 0.0f );
	m_z = Vector4( 0.0f, 0.0f, 1.0f );
}
	
//---------------------------------------------------------------------------------------------------------------------------------

void Matrix4::SetScale( float x, float y, float z )
{
	m_x.SetX( x );
	m_y.SetY( y );
	m_z.SetZ( z );
}

//---------------------------------------------------------------------------------------------------------------------------------

void Matrix4::Transpose( void )
{
	const float xy = m_x.GetY();
	const float xz = m_x.GetZ();
	const float xw = m_x.GetW();
	const float yz = m_y.GetZ();
	const float yw = m_y.GetW();
	const float zw = m_z.GetW();

	m_x.SetY( m_y.GetX() );
	m_x.SetZ( m_z.GetX() );
	m_x.SetW( m_t.GetX() );
	m_y.SetX( xy );
	m_y.SetZ( m_z.GetY() );
	m_y.SetW( m_t.GetY() );
	m_z.SetX( xz );
	m_z.SetY( yz );
	m_z.SetW( m_t.GetZ() );
	m_t.SetX( xw );
	m_t.SetY( yw );
	m_t.SetZ( zw );
}

//---------------------------------------------------------------------------------------------------------------------------------

Matrix4& Matrix4::operator=( const Matrix4& mat )
{
	m_x = mat.m_x;
	m_y = mat.m_y;
	m_z = mat.m_z;
	m_t = mat.m_t;

	return *this;
}

//---------------------------------------------------------------------------------------------------------------------------------

Matrix4& Matrix4::operator*=( const Matrix4& mat )
{
	Matrix4 temp( *this );

	m_x = mat * temp.GetXAxis();
	m_y = mat * temp.GetYAxis();
	m_z = mat * temp.GetZAxis();
	m_t = mat * temp.GetTranslation();

	return *this;
}

//---------------------------------------------------------------------------------------------------------------------------------

Matrix4 Matrix4::operator*( const Matrix4& mat ) const
{
	Matrix4 ret;

	ret.m_x = mat * m_x;
	ret.m_y = mat * m_y;
	ret.m_z = mat * m_z;
	ret.m_t = mat * m_t;

	return ret;
}

//---------------------------------------------------------------------------------------------------------------------------------

Vector4 Matrix4::operator*( const Vector4& vec ) const
{
	float x = vec.GetX() * m_x.GetX() + vec.GetY() * m_y.GetX() + vec.GetZ() * m_z.GetX() + vec.GetW() * m_t.GetX();
	float y = vec.GetX() * m_x.GetY() + vec.GetY() * m_y.GetY() + vec.GetZ() * m_z.GetY() + vec.GetW() * m_t.GetY();
	float z = vec.GetX() * m_x.GetZ() + vec.GetY() * m_y.GetZ() + vec.GetZ() * m_z.GetZ() + vec.GetW() * m_t.GetZ();
	float t = vec.GetX() * m_x.GetW() + vec.GetY() * m_y.GetW() + vec.GetZ() * m_z.GetW() + vec.GetW() * m_t.GetW();

	return Vector4( x, y, z, t );
}

//---------------------------------------------------------------------------------------------------------------------------------
