#include "Crowd.h"
#include "CrowdShader.h"
#include "Camera.h"

#ifdef _WIN32
#include <windows.h>
#include <gl/glew.h>
#else
#include <opengl/gl.h>
#endif

//---------------------------------------------------------------------------------------------------------------------------------

#ifdef _WIN32
	static const char* s_base_map_file_path = "base_map.bmp";
	static const char* s_normal_map_file_path = "normal_map.bmp";
#else
	//Weird xcode path correction to compensate for broken working directory functionality.
	//You may need to change this if the textures are not loading.
	static const char* s_base_map_file_path = "../../../../Data/base_map.bmp";
	static const char* s_normal_map_file_path = "../../../../Data/normal_map.bmp";
#endif

//---------------------------------------------------------------------------------------------------------------------------------

Crowd::Crowd( void ) :
	m_light_dir( -0.1f, -0.1f, -0.7f ),
	m_light_col( 1.09f, 1.01f, 1.04f ),
	m_light_amb( 0.75f, 0.69f, 0.66f ),
	m_density( 50 )
{
	m_light_dir.Normalize();
}

//---------------------------------------------------------------------------------------------------------------------------------

Crowd::~Crowd( void )
{

}

//---------------------------------------------------------------------------------------------------------------------------------

void Crowd::AdjustDensity( unsigned int percent )
{
	m_density += percent;
	m_density = m_density < 0 ? 0 : ( m_density > 100 ? 100 : m_density );
}

//---------------------------------------------------------------------------------------------------------------------------------

void Crowd::Create( unsigned int main_width, unsigned int main_height, unsigned int side_width, unsigned int side_height )
{
	m_base_map.LoadBMP( s_base_map_file_path );
	m_normal_map.LoadBMP( s_normal_map_file_path );
	m_vertex_shader.Create( crowd_shader, "gp4vp", "main_vs" );
	m_fragment_shader.Create( crowd_shader, "fp40", "main_fs" );
	m_mesh.Create( main_width, main_height, side_width, side_height );
}

//---------------------------------------------------------------------------------------------------------------------------------

void Crowd::Render( const Camera& camera )
{
	RenderAtlas( camera );
	RenderBillboards( camera );
	RenderDeferred();
}

//---------------------------------------------------------------------------------------------------------------------------------

void Crowd::RenderAtlas( const Camera& world_camera )
{
	for( unsigned int i=0; i<m_mesh.GetGroupCount(); ++i )
	{
		m_mesh.GetGroup( i ).RenderAtlas( world_camera );
	}
}

//---------------------------------------------------------------------------------------------------------------------------------

void Crowd::RenderBillboards( const Camera& world_camera )
{
	unsigned int base_map_id = m_base_map.GetId();
	unsigned int normal_map_id = m_normal_map.GetId();
	float density = static_cast<float>( m_density ) * 0.01f;

	m_vertex_shader.Bind();
	m_fragment_shader.Bind();
	m_base_map.Bind( 0 );
	m_normal_map.Bind( 1 );
	m_vertex_shader.SetParameter( "view", &world_camera.GetViewMatrix(), 16 );
	m_vertex_shader.SetParameter( "proj", &world_camera.GetProjectionMatrix(), 16 );
	m_vertex_shader.SetParameter( "density", &density, 1 );
	m_fragment_shader.SetParameter( "base_map", &base_map_id, 1 );
	m_fragment_shader.SetParameter( "normal_map", &normal_map_id, 1 );
	m_fragment_shader.SetParameter( "light_dir", &m_light_dir, 4 );
	m_fragment_shader.SetParameter( "light_col", &m_light_col, 4 );
	m_fragment_shader.SetParameter( "light_amb", &m_light_amb, 4 );

	for( unsigned int i=0; i<m_mesh.GetGroupCount(); ++i )
	{
		m_mesh.GetGroup( i ).RenderBillboards( m_vertex_shader, m_fragment_shader );
	}
}

//---------------------------------------------------------------------------------------------------------------------------------

void Crowd::RenderDeferred( void )
{
	//Render a quad, with stencil hardware set to EQUAL
}

//---------------------------------------------------------------------------------------------------------------------------------

void Crowd::Shutdown( void )
{
	m_mesh.Shutdown();
}

//---------------------------------------------------------------------------------------------------------------------------------
	
void Crowd::Update( float dt )
{
	//Animate all the models so they are set in the correct pose.
	//Note that all sections adopt the same pose but render from
	//different camera angles.
}

//---------------------------------------------------------------------------------------------------------------------------------
