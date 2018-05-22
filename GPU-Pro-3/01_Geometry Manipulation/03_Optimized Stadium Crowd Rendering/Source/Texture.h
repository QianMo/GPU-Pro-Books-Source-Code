/**
 *	@file
 *	@brief		A basic texture class that supports BMP loading.
 *	@author		Alan Chambers
 *	@date		2011
**/

#ifndef TEXTURE_H
#define TEXTURE_H

class Texture
{
public:
								Texture( void );
								~Texture( void );

	void						Bind( unsigned int tex_unit );

	unsigned int				GetBpp( void ) const;
	unsigned int				GetHeight( void ) const;
	unsigned int				GetId( void ) const;
	const void*					GetPixelData( void ) const;
	unsigned int				GetWidth( void ) const;
	
	bool						LoadBMP( const char* file_name );

private:
	unsigned int				m_width;
	unsigned int				m_height;
	unsigned int				m_bpp;
	unsigned int				m_size;
	void*						m_data;
	unsigned int				m_id;
};

#include "Texture.inl"

#endif
