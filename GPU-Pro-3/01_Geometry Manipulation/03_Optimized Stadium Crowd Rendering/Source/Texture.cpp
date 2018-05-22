#include "Texture.h"

#include <memory.h>
#include <algorithm>
#include <stdio.h>

#ifdef _WIN32
#include <windows.h>
#include <gl/glew.h>
#else
#include <opengl/gl.h>
#endif
#include <stdio.h>

#ifdef _WIN32
#define ATTR_PACK
#else
#define ATTR_PACK __attribute__( ( packed ) )
#endif

//---------------------------------------------------------------------------------------------------------------------------------

#ifdef _WIN32
#pragma pack( push, 1 )
#endif

struct FileHeader
{
	unsigned short		signature;
	unsigned int		size;
	unsigned short		reserved1;
	unsigned short		reserved2;
	unsigned int		bit_offset;
}	ATTR_PACK;

struct InfoHeader
{
	unsigned int		size;
	unsigned int		width;
	unsigned int		height;
	unsigned short		plane_count;
	unsigned short		bit_count;
	unsigned int		compression;
	unsigned int		image_size;
	unsigned int		x_pixels_per_meter;
	unsigned int		y_pixels_per_meter;
	unsigned int		clr_used;
	unsigned int		clr_important;
}	ATTR_PACK;

#ifdef _WIN32
#pragma pack( pop )
#endif

//---------------------------------------------------------------------------------------------------------------------------------

Texture::Texture( void ) :
	m_width( 0 ),
	m_height( 0 ),
	m_bpp( 0 ),
	m_size( 0 ),
	m_data( 0 ),
	m_id( 0 )
{

}

//---------------------------------------------------------------------------------------------------------------------------------

Texture::~Texture( void )
{
	if( m_data )
		free( m_data );
}

//---------------------------------------------------------------------------------------------------------------------------------

void Texture::Bind( unsigned int tex_unit )
{
	glActiveTexture( GL_TEXTURE0+tex_unit );
	glEnable( GL_TEXTURE_2D );
	glBindTexture( GL_TEXTURE_2D, m_id );
}

//---------------------------------------------------------------------------------------------------------------------------------

bool Texture::LoadBMP( const char* file_name )
{
	FILE* in = fopen( file_name, "rb" );
	
	if( !in )
		return false;
	
	FileHeader file_header;	
 	InfoHeader info_header;
	
	fread( &file_header, sizeof( file_header ), 1, in );
 	fread( &info_header, sizeof( info_header ), 1, in );

	m_width = info_header.width;
	m_height = info_header.height;
	m_bpp = info_header.bit_count;
	m_size = m_width * m_height * ( m_bpp / 8 );
	m_data = malloc( m_size );

	if( !m_data )
		return false;
	
	fread( m_data, m_size, 1, in );
	fclose( in );

	//Allocate our 
	glGenTextures( 1, &m_id );
	glBindTexture( GL_TEXTURE_2D, m_id );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
	glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, m_width, m_height, 0, GL_BGRA, GL_UNSIGNED_BYTE, m_data );

	return true;
}

//---------------------------------------------------------------------------------------------------------------------------------
