/**
 *	@file
 *	@brief		Basic camera functionality.
 *	@author		Alan Chambers
 *	@date		2011
**/

#ifndef CAMERA_H
#define CAMERA_H

#include "Matrix4.h"

class Camera
{
public:
								Camera( void );
								Camera( const char* name );

	void						Create( float fov, float aspect, float near, float far );
	
	float						GetAspect( void ) const;
	float						GetFar( void ) const;
	float						GetFOV( void ) const;
	float						GetNear( void ) const;
	const Matrix4&				GetProjectionMatrix( void ) const;
	const Matrix4&				GetViewMatrix( void ) const;

	void						Update( void );
	
	void						Move( const Vector4& dir );

	void						SetPosition( const Vector4& pos );

	void						RotateX( const float angle );
	void						RotateY( const float angle );

private:
	float						m_fov;
	float						m_aspect;
	float						m_near;
	float						m_far;
	Matrix4						m_world;
	Matrix4						m_view;
	Matrix4						m_proj;
};

#include "Camera.inl"

#endif
