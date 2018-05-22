#ifndef MATRIX4_H
#error "Do not include Matrix4.inl directly!"
#endif

inline Matrix4::Matrix4( void ) :
	m_x( Vector4::XAXIS ),
	m_y( Vector4::YAXIS ),
	m_z( Vector4::ZAXIS ),
	m_t( Vector4::ORIGIN )
{

}

inline Matrix4::Matrix4( const Matrix4& mat ) :
	m_x( mat.GetXAxis() ),
	m_y( mat.GetYAxis() ),
	m_z( mat.GetZAxis() ),
	m_t( mat.GetTranslation() )
{

}

inline Matrix4::Matrix4( const float* pMat ) :
	m_x( pMat[0], pMat[1], pMat[2], pMat[3] ),
	m_y( pMat[4], pMat[5], pMat[6], pMat[7] ),
	m_z( pMat[8], pMat[9], pMat[10], pMat[11] ),
	m_t( pMat[12], pMat[13], pMat[14], pMat[15] )	
{

}

inline Matrix4::Matrix4( const Vector4& x, const Vector4& y, const Vector4& z, const Vector4& t ) :
	m_x( x ),
	m_y( y ),
	m_z( z ),
	m_t( t )
{

}

inline const Vector4& Matrix4::GetXAxis( void ) const
{
	return m_x;
}

inline const Vector4& Matrix4::GetYAxis( void ) const
{
	return m_y;
}

inline const Vector4& Matrix4::GetZAxis( void ) const
{
	return m_z;
}

inline const Vector4& Matrix4::GetTranslation( void ) const
{
	return m_t;
}

inline void Matrix4::SetXAxis( const Vector4& x )
{
	m_x = x;
}

inline void Matrix4::SetYAxis( const Vector4& y )
{
	m_y = y;
}

inline void Matrix4::SetZAxis( const Vector4& z )
{
	m_z = z;
}

inline void Matrix4::SetTranslation( const Vector4& t )
{
	m_t = t;
}
