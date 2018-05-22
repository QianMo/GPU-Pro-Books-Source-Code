#include "CrowdMesh.h"

//---------------------------------------------------------------------------------------------------------------------------------

static const unsigned int s_max_groups = 2;

//---------------------------------------------------------------------------------------------------------------------------------

CrowdMesh::CrowdMesh( void )
{
	m_groups.reserve( s_max_groups );
}

//---------------------------------------------------------------------------------------------------------------------------------

CrowdMesh::~CrowdMesh( void )
{

}

//---------------------------------------------------------------------------------------------------------------------------------

void CrowdMesh::Create( unsigned int main_width, unsigned int main_height, unsigned int side_width, unsigned int side_height )
{
	if( m_groups.size() != 0 )
		return;

	m_groups.resize( 2 );

	float mx = static_cast<float>( main_width ) * 0.5f;
	float sx = static_cast<float>( side_width ) * 0.5f;

	m_groups[0].Create( 2 );
	m_groups[0].GetSection( 0 ).GenerateGeometry( main_width, main_height );
	m_groups[0].GetSection( 0 ).Create( Vector4( +sx, 0.0f, 0.0f, 1.0f ) );
	m_groups[0].GetSection( 1 ).GenerateGeometry( main_width, main_height );
	m_groups[0].GetSection( 1 ).Create( Vector4( -sx, 0.0f, 0.0f, 1.0f ) );

	m_groups[1].Create( 2 );
	m_groups[1].GetSection( 0 ).GenerateGeometry( side_width, side_height );
	m_groups[1].GetSection( 0 ).Create( Vector4( 0.0f, 0.0f, +mx, 1.0f ) );
	m_groups[1].GetSection( 1 ).GenerateGeometry( side_width, side_height );
	m_groups[1].GetSection( 1 ).Create( Vector4( 0.0f, 0.0f, -mx, 1.0f ) );
}

//---------------------------------------------------------------------------------------------------------------------------------

void CrowdMesh::Shutdown( void )
{
	unsigned int group_count = m_groups.size();
	for( unsigned int i=0; i<group_count; ++i )
	{
		m_groups[i].Shutdown();
	}
	m_groups.clear();
}

//---------------------------------------------------------------------------------------------------------------------------------
