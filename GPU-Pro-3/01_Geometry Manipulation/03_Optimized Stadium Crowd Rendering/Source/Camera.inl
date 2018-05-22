#ifndef CAMERA_H
#error "Do not include camera.inl directly!"
#endif

inline float Camera::GetAspect( void ) const
{
	return m_aspect;
}

inline float Camera::GetFar( void ) const
{
	return m_far;
}

inline float Camera::GetFOV( void ) const
{
	return m_aspect;
}

inline float Camera::GetNear( void ) const
{
	return m_near;
}

inline const Matrix4& Camera::GetProjectionMatrix( void ) const
{
	return m_proj;
}

inline const Matrix4& Camera::GetViewMatrix( void ) const
{
	return m_view;
}
