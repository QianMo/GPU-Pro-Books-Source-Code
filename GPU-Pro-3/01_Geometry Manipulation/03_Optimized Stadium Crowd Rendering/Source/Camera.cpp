#include "Camera.h"

#include <cmath>

//---------------------------------------------------------------------------------------------------------------------------------

Camera::Camera( void ) :
	m_fov( 45.0f ),
	m_aspect( 1.0f ),
	m_near( 1.0f ),
	m_far( 10000.0f )
{

}

//---------------------------------------------------------------------------------------------------------------------------------

void Camera::Create( float fov, float aspect, float zNear, float zFar )
{
	const float y = 1.0f / tanf( fov * 0.5f );
	const float x = ( y / aspect );
	const float z = ( zFar + zNear ) / ( zNear - zFar );
	const float tz = ( 2.0f * zNear * zFar ) / ( zNear - zFar );
	const float tw = -1.0f;

	m_proj.SetXAxis( Vector4( x,	0.0f,	0.0f,	0.0f ) );
	m_proj.SetYAxis( Vector4( 0.0f,	y,		0.0f,	0.0f ) );
	m_proj.SetZAxis( Vector4( 0.0f,	0.0f,	z,		tw	 ) );
	m_proj.SetTranslation( Vector4( 0.0f,	0.0f,	tz,		0.0f ) );
	
	m_fov = fov;
	m_aspect = aspect;
	m_near = zNear;
	m_far = zFar;
}

//---------------------------------------------------------------------------------------------------------------------------------

void Camera::Move( const Vector4& dir )
{
	Vector4 x_axis = m_world.GetXAxis() * dir.GetX();
	Vector4 y_axis = m_world.GetYAxis() * dir.GetY();
	Vector4 z_axis = m_world.GetZAxis() * dir.GetZ();
	Vector4 pos = m_world.GetTranslation() + x_axis + y_axis + z_axis;
	m_world.SetTranslation( pos );
}

//---------------------------------------------------------------------------------------------------------------------------------

void Camera::Update( void )
{
	Vector4 z_axis = m_world.GetZAxis();
	z_axis.Normalize();
	Vector4 x_axis = Vector4::Cross( Vector4::YAXIS, z_axis );
	x_axis.Normalize();
	Vector4 y_axis = Vector4::Cross( z_axis, x_axis );
	y_axis.Normalize();

	Vector4 pos = m_world.GetTranslation();

 	const float x = Vector4::Dot( x_axis, pos );
 	const float y = Vector4::Dot( y_axis, pos );
 	const float z = Vector4::Dot( z_axis, pos );

	m_view.SetXAxis( Vector4( x_axis.GetX(), y_axis.GetX(), z_axis.GetX() ) );
	m_view.SetYAxis( Vector4( x_axis.GetY(), y_axis.GetY(), z_axis.GetY() ) );
	m_view.SetZAxis( Vector4( x_axis.GetZ(), y_axis.GetZ(), z_axis.GetZ() ) );	
	m_view.SetTranslation( Vector4( -x, -y, -z, 1.0f ) );
}

//---------------------------------------------------------------------------------------------------------------------------------

void Camera::RotateX( const float angle )
{
	Matrix4 rot;

	rot.SetRotateX( angle );

	m_world = rot * m_world;
}

//---------------------------------------------------------------------------------------------------------------------------------

void Camera::RotateY( const float angle )
{
	Matrix4 rot;

	rot.SetRotateY( angle );
	
	m_world = rot * m_world;
}

//---------------------------------------------------------------------------------------------------------------------------------

void Camera::SetPosition( const Vector4& pos )
{
	m_world.SetTranslation( pos );
}

//---------------------------------------------------------------------------------------------------------------------------------
