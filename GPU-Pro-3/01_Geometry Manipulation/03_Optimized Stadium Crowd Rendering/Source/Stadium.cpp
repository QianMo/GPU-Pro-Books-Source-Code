#include "Stadium.h"
#include "StadiumShader.h"
#include "Camera.h"

#ifdef _WIN32
#include <windows.h>
#include <gl/glew.h>
#else
#include <opengl/gl.h>
#endif

//---------------------------------------------------------------------------------------------------------------------------------

struct Vertex
{
	float x, y, z, w;
	unsigned int color;
};

//---------------------------------------------------------------------------------------------------------------------------------

Stadium::Stadium( void ) :
	m_vb( 0 ),
	m_ib( 0 )
{

}

//---------------------------------------------------------------------------------------------------------------------------------

Stadium::~Stadium( void )
{

}

//---------------------------------------------------------------------------------------------------------------------------------

void Stadium::Create( unsigned int width, unsigned int height, unsigned int length )
{
	float x = static_cast<float>( length ) * 0.5f;
	float y = static_cast<float>( height );
	float z = static_cast<float>( width ) * 0.5f;

	static const unsigned int vert_count = 12;
	Vertex vertex_data[vert_count] =
	{
		{ -x, 0.0f, +z, 1.0f, 0xffc1c1c1 },
		{ +x, 0.0f, +z, 1.0f, 0xffc1c1c1 },
		{ +x, 0.0f, -z, 1.0f, 0xffc1c1c1 },
		{ -x, 0.0f, -z, 1.0f, 0xffc1c1c1 },
		{ -x-y, +y, +z+y, 1.0f, 0xffc1c1c1 },
		{ +x+y, +y, +z+y, 1.0f, 0xffc1c1c1 },
		{ +x+y, +y, -z-y, 1.0f, 0xffc1c1c1 },
		{ -x-y, +y, -z-y, 1.0f, 0xffc1c1c1 },
		{ -x, 0.0f, +z, 1.0f, 0xff33d066 },
		{ +x, 0.0f, +z, 1.0f, 0xff33d066 },
		{ +x, 0.0f, -z, 1.0f, 0xff33d066 },
		{ -x, 0.0f, -z, 1.0f, 0xff33d066 },
	};
	static const unsigned int tristrip_count = 15;
	static unsigned char index_data[tristrip_count] =
	{
		5, 0, 4, 3, 7, 2, 6, 1, 5, 0, 0, 8, 11, 9, 10
	};

	glGenBuffers( 1, &m_vb );
	glBindBuffer( GL_ARRAY_BUFFER, m_vb );
	glBufferData( GL_ARRAY_BUFFER, sizeof( vertex_data ), &vertex_data[0], GL_STATIC_DRAW );
	glBindBuffer( GL_ARRAY_BUFFER, 0 );
	glGenBuffers( 1, &m_ib );
	glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, m_ib );
	glBufferData( GL_ELEMENT_ARRAY_BUFFER, sizeof( index_data ), &index_data[0], GL_STATIC_DRAW );
	glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, 0 );

	m_vertex_shader.Create( stadium_shader, "vp40", "main_vs" );
	m_fragment_shader.Create( stadium_shader, "fp40", "main_fs" );
}

//---------------------------------------------------------------------------------------------------------------------------------

void Stadium::Render( const Camera& camera )
{
	m_vertex_shader.Bind();
	m_fragment_shader.Bind();

	Matrix4 world_view_proj = camera.GetViewMatrix() * camera.GetProjectionMatrix();
	m_vertex_shader.SetParameter( "world_view_proj", &world_view_proj, 16 );

	glBindBuffer( GL_ARRAY_BUFFER_ARB, m_vb );
	glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, m_ib );
	glVertexPointer( 4, GL_FLOAT, sizeof( Vertex ), 0 );
	glEnableClientState( GL_VERTEX_ARRAY );
	glColorPointer( 4, GL_UNSIGNED_BYTE, sizeof( Vertex ), (void*)( sizeof( float ) * 4 ) );
	glEnableClientState( GL_COLOR_ARRAY );
	glIndexPointer( GL_UNSIGNED_BYTE, sizeof( char ), 0 );
	glEnableClientState( GL_INDEX_ARRAY );
	glDrawElements( GL_TRIANGLE_STRIP, 15, GL_UNSIGNED_BYTE, 0 );
	glDisableClientState( GL_VERTEX_ARRAY );
	glDisableClientState( GL_COLOR_ARRAY );
}

//---------------------------------------------------------------------------------------------------------------------------------

void Stadium::Shutdown( void )
{
	glDeleteBuffers( 1, &m_vb );
	glDeleteBuffers( 1, &m_ib );
}

//---------------------------------------------------------------------------------------------------------------------------------
