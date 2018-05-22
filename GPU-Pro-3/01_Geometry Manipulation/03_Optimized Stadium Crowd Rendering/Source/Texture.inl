#ifndef TEXTURE_H
#error "Do not include Texture.inl directly!"
#endif

inline unsigned int Texture::GetBpp( void ) const
{
	return m_bpp;
}

inline unsigned int Texture::GetHeight( void ) const
{
	return m_height;
}

inline unsigned int Texture::GetId( void ) const
{
	return m_id;
}

inline const void* Texture::GetPixelData( void ) const
{
	return m_data;
}

inline unsigned int Texture::GetWidth( void ) const
{
	return m_width;
}
