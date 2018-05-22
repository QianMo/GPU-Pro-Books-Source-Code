#ifndef SHADER_H
#error "Do not include Shader.inl directly!"
#endif

inline CGbuffer Shader::Buffer::GetId( void ) const
{
	return m_id;
}
