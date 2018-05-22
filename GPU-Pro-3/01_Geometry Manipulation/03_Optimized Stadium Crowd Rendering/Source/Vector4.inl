#ifndef VECTOR4_H
#error "Do not include Vector4.inl directly!"
#endif

inline Vector4::Vector4( void ) :
	m_x( 0.0f ),
	m_y( 0.0f ),
	m_z( 0.0f ),
	m_w( 0.0f )
{

}

inline Vector4::Vector4( const Vector4& vec ) :
	m_x( vec.GetX() ),
	m_y( vec.GetY() ),
	m_z( vec.GetZ() ),
	m_w( vec.GetW() )
{

}

inline Vector4::Vector4( float x, float y, float z, float w ) :
	m_x( x ),
	m_y( y ),
	m_z( z ),
	m_w( w )
{

}

inline float Vector4::GetX( void ) const
{
	return m_x;
}

inline float Vector4::GetY( void ) const
{
	return m_y;
}

inline float Vector4::GetZ( void ) const
{
	return m_z;
}

inline float Vector4::GetW( void ) const
{
	return m_w;
}

inline void Vector4::SetX( float x )
{
	m_x = x;
}

inline void Vector4::SetY( float y )
{
	m_y = y;
}

inline void Vector4::SetZ( float z )
{
	m_z = z;
}

inline void Vector4::SetW( float w )
{
	m_w = w;
}
