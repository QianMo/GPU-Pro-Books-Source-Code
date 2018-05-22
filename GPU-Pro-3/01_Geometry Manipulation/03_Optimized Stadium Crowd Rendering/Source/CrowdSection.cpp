#include "CrowdSection.h"
#include "Shader.h"
#include "Camera.h"

#ifdef _WIN32
#include <windows.h>
#include <gl/glew.h>
#else
#include <opengl/gl.h>
#endif

#include <time.h>

//---------------------------------------------------------------------------------------------------------------------------------

struct Position
{
	float x, y, z, w;
};

struct Property
{
	float density, seat_id, dude_id, shirt_id;
};

//---------------------------------------------------------------------------------------------------------------------------------

static const float x_offset = 0.75f;
static const float y_offset = 0.5f;
static const float z_offset = 0.5f;
static const unsigned int quad_count = 4;		//The 4 corners of the quadrilateral
static const unsigned int character_count = 9;

//---------------------------------------------------------------------------------------------------------------------------------

CrowdSection::CrowdSection( void ) :
	m_quad_id( 0 ),
	m_seat_count( 0 ),
	m_world( Matrix4::IDENTITY ),
	m_mirror( 0.0f )
{

}

//---------------------------------------------------------------------------------------------------------------------------------

CrowdSection::~CrowdSection( void )
{

}

//---------------------------------------------------------------------------------------------------------------------------------

Vector4 CrowdSection::ComputeCameraPosition( const Camera& world_camera )
{
	//The following is psuedo-code that featured in the book to compute
	//the camera position for rendering out the crowd atlas.
#ifdef ENABLE_ATLAS_RENDERING
	Vector3 camera_zaxis( -world_camera.getForwardAxis() );
	Vector3 crowd_xaxis = crowd_matrix.getRightAxis();
	Vector3 crowd_zaxis = crowd_matrix.getForwardAxis();
	camera_zaxis.setY( 0.0f );
	camera_zaxis = normalize( camera_zaxis );
	float x_dot = dot( camera_zaxis, crowd_xaxis );
	float z_dot = dot( camera_zaxis, crowd_zaxis );
	float angle = fast_acos( z_dot );
	angle = x_dot >= 0 ? angle : angle * -1; 
	Vector3 crowd_edge_pos = Vector3( 0.0f, 0.0f, crowd_bounds.getZ() ) * crowd_matrix;
	Vector3 camera_wpos = world_camera.getPosition();
	camera_wpos.setY( 0.0f );
	Vector3 scale_vector = camera_wpos – crowd_edge_pos;
	float scalar = dot( scale_vector, crowd_zaxis );
	angle *= ( 0.5f / ( abs( scalar ) + 1.0f ) ) + 0.5f;
	float pitch = world_camera.getPitch();

	Vector3 camera_fwd = Matrix3::rotationX( -pitch ).getForwardAxis();
	Vector3 camera_dir = Matrix3::rotationY( angle ) * camera_fwd;
	Vector3 camera_pos = camera_dir * MIN_CAMERA_DIST;

	return camera_pos;
#else
	return Vector4();
#endif
}

//---------------------------------------------------------------------------------------------------------------------------------

void CrowdSection::Create( const Vector4& offset )
{
	static Position quad_data[quad_count] =
	{
		{ 0.0f, 0.0f, 0.0f, 0.0f },
		{ 1.0f, 0.0f, 0.0f, 0.0f },
		{ 1.0f, 1.0f, 0.0f, 0.0f },
		{ 0.0f, 1.0f, 0.0f, 0.0f },
	};

	glGenBuffers( 1, &m_quad_id );
	glBindBuffer( GL_ARRAY_BUFFER, m_quad_id );
	glBufferData( GL_ARRAY_BUFFER, sizeof( quad_data ), &quad_data[0], GL_STATIC_DRAW );
	glBindBuffer( GL_ARRAY_BUFFER, 0 );

	//Our crowd sections always faces the origin.
	Vector4 z_axis = offset - Vector4::ORIGIN;
	z_axis.SetY( 0.0f );
	z_axis.Normalize();
	Vector4 x_axis = Vector4::Cross( Vector4::YAXIS, z_axis );
	x_axis.Normalize();
	Vector4 y_axis = Vector4::Cross( z_axis, x_axis );
	y_axis.Normalize();
	m_world.SetXAxis( x_axis );
	m_world.SetYAxis( y_axis );
	m_world.SetZAxis( z_axis );	
	m_world.SetTranslation( offset );

 	if( IsMirror() )
 	{
		m_mirror = 1.0f;
		//At the this point we should construct our light transformation matrix so the
		//x component of the right axis is negated to compensate for the normal values
		//not being in the right space for mirrored sections.
 	}
}

//---------------------------------------------------------------------------------------------------------------------------------

//Ideally this data would be loaded from an exported crowd mesh.
//This data is procedurally generated only so we can illustrate what to
//do with the data once it's loaded.
void CrowdSection::GenerateGeometry( int width, int height )
{
	width = static_cast<int>( static_cast<float>( width ) / x_offset );
	height = static_cast<int>( static_cast<float>( height ) / y_offset );

	unsigned int element_count = width * height;
	std::vector<Position> positions;
	std::vector<Property> properties;

	positions.resize( element_count );
	properties.resize( element_count );

	//Generate position data
	for( int x=0; x<width; ++x )
	{
		for( int y=0; y<height; ++y )
		{
			Position& pos = positions[x+width*y];

			pos.x = x_offset * static_cast<float>( x - ( width / 2 ) );
			pos.y = y_offset * static_cast<float>( y );
			pos.z = z_offset * static_cast<float>( y );
			pos.w = 1.0f;
		}
	}

	//Randomize the crowd dudes at startup.
	srand( static_cast<unsigned int>( time( 0 ) ) );

	//Generate color data
	for( int x=0; x<width; ++x )
	{
		for( int y=0; y<height; ++y )
		{
			Property& property = properties[x*height+y];

			property.density = static_cast<float>( rand() ) / RAND_MAX;
			property.seat_id = static_cast<float>( rand() ) / RAND_MAX;
			//Generate a random dude between 0 and our max character count.
			//The seat is rendered into slot 0 so we shift it across by 1.
			//We then expand it into the full range of a char so we don't
			//have to do it in the shader i.e. we can use the float value
			//direct as a uv input.
			property.dude_id = static_cast<float>( ( rand() % character_count ) ) / 10.0f;
			property.shirt_id = 0;
		}
	}	

	unsigned int pos_size = positions.size() * sizeof( Position );
	unsigned int prop_size = properties.size() * sizeof( Property );

	m_buffers[0].Create( &positions[0], pos_size );
	m_buffers[1].Create( &properties[0], prop_size );
	m_seat_count = positions.size();
}

//---------------------------------------------------------------------------------------------------------------------------------

bool CrowdSection::IsMirror( void ) const
{
	//The following is psuedo-code that featured in the book to compute
	//which half of the stadium the section lies in to identify a mirror.
	Vector4 x_axis( 1.0f, 0.0f, 0.0f, 0.0f );
	Vector4 z_axis( 0.0f, 0.0f, 1.0f, 0.0f );
	Vector4 dir = m_world.GetZAxis();

	dir.SetY( 0.0f );
	dir.Normalize();
	float dp = Vector4::Dot( dir, z_axis ) + Vector4::Dot( dir, x_axis );

	return dp >= 0.0f;
}

//---------------------------------------------------------------------------------------------------------------------------------

void CrowdSection::RenderAtlas( const Camera& world_camera ) const
{
	(void) world_camera;

	//ComputeCameraPosition with the world camera.
	//Setup the offscreen camera and scene.
	//Loop through and render each one into an appropriate slot of the atlas.
}

//---------------------------------------------------------------------------------------------------------------------------------

void CrowdSection::RenderBillboards( Shader& vertex_shader, Shader& fragment_shader ) const
{
	fragment_shader.SetParameter( "world", &m_world, 16 );
	vertex_shader.SetParameter( "world", &m_world, 16 );
	vertex_shader.SetParameter( "mirror", &m_mirror, 1 );
	vertex_shader.Bind( 0, m_buffers[0] );
	vertex_shader.Bind( 1, m_buffers[1] );

	glBindBuffer( GL_ARRAY_BUFFER_ARB, m_quad_id );
	glVertexAttribPointer( 0, 4, GL_FLOAT, GL_FALSE, 0, 0 );
	glEnableVertexAttribArray( 0 );
	glDrawArraysInstancedARB( GL_QUADS, 0, quad_count, m_seat_count  );
	glDisableVertexAttribArray( 0 );
}

//---------------------------------------------------------------------------------------------------------------------------------

void CrowdSection::Shutdown( void )
{
	m_buffers[1].Shutdown();
	m_buffers[2].Shutdown();
	glDeleteBuffers( 1, &m_quad_id );
	m_quad_id = 0;
	m_seat_count = 0;
}

//---------------------------------------------------------------------------------------------------------------------------------
