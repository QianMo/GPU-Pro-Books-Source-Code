#include "Vector4.h"

#include <cstdlib>
#include <cfloat>
#include <cstdio>
#include <cmath>

//---------------------------------------------------------------------------------------------------------------------------------

Vector4 Vector4::Cross( const Vector4& v1, const Vector4& v2 )
{
	float x = v1.GetY() * v2.GetZ() - v1.GetZ() * v2.GetY();
	float y = v1.GetZ() * v2.GetX() - v1.GetX() * v2.GetZ();
	float z = v1.GetX() * v2.GetY() - v1.GetY() * v2.GetX();

	return Vector4( x, y, z );
}

//---------------------------------------------------------------------------------------------------------------------------------

float Vector4::Dot( const Vector4& v1, const Vector4& v2 )
{
	float dot;
	
	float x = v1.GetX() * v2.GetX();
	float y = v1.GetY() * v2.GetY();
	float z = v1.GetZ() * v2.GetZ();
	
	dot = x + y + z;

	return dot;
}

//---------------------------------------------------------------------------------------------------------------------------------

float Vector4::Length( void ) const
{
	float length;
	
	length = sqrtf( SquareLength() );

	return length;
}

//---------------------------------------------------------------------------------------------------------------------------------

Vector4& Vector4::Normalize( void )
{
	float length = Length();

	length = 1.0f / length;

	Scale( length );
	
	return *this;
}

//---------------------------------------------------------------------------------------------------------------------------------

void Vector4::Scale( float scale )
{
	m_x *= scale;
	m_y *= scale;
	m_z *= scale;
}

//---------------------------------------------------------------------------------------------------------------------------------

float Vector4::SquareLength( void ) const
{
	float length;
	
	float x = m_x * m_x;
	float y = m_y * m_y;
	float z = m_z * m_z;
	
	length = x + y + z;

	return length;
}

//---------------------------------------------------------------------------------------------------------------------------------

Vector4& Vector4::operator=( const Vector4& vec )
{
	m_x = vec.m_x;
	m_y = vec.m_y;
	m_z = vec.m_z;
	m_w = vec.m_w;

	return *this;
}

//---------------------------------------------------------------------------------------------------------------------------------

Vector4 Vector4::operator*( const Vector4& vec ) const
{
	float x = m_x * vec.GetX();
	float y = m_y * vec.GetY();
	float z = m_z * vec.GetZ();
	float w = m_w * vec.GetW();
	
	return Vector4( x, y, z, w );
}

//---------------------------------------------------------------------------------------------------------------------------------

Vector4 Vector4::operator-( const Vector4& vec ) const
{
	float x = m_x - vec.GetX();
	float y = m_y - vec.GetY();
	float z = m_z - vec.GetZ();
	float w = m_w - vec.GetW();

	return Vector4( x, y, z, w );
}

//---------------------------------------------------------------------------------------------------------------------------------

Vector4 Vector4::operator+( const Vector4& vec ) const
{
	float x = m_x + vec.GetX();
	float y = m_y + vec.GetY();
	float z = m_z + vec.GetZ();
	float w = m_w + vec.GetW();

	return Vector4( x, y, z, w );
}

//---------------------------------------------------------------------------------------------------------------------------------

Vector4 Vector4::operator*( float scale ) const
{
	float x = m_x * scale;
	float y = m_y * scale;
	float z = m_z * scale;
	float w = m_w * scale;
	
	return Vector4( x, y, z, w );
}

//---------------------------------------------------------------------------------------------------------------------------------
