#include "CrowdGroup.h"

#include <algorithm>

//---------------------------------------------------------------------------------------------------------------------------------

static const unsigned int s_max_sections = 2;

//---------------------------------------------------------------------------------------------------------------------------------

CrowdGroup::CrowdGroup( void )
{
	m_sections.reserve( s_max_sections );
}

//---------------------------------------------------------------------------------------------------------------------------------

CrowdGroup::~CrowdGroup( void )
{

}

//---------------------------------------------------------------------------------------------------------------------------------
	
void CrowdGroup::Create( unsigned int section_count )
{
	if( m_sections.size() != 0 )
		return;

	if( section_count > s_max_sections )
		section_count = s_max_sections;

	m_sections.resize( section_count );
}

//---------------------------------------------------------------------------------------------------------------------------------

void CrowdGroup::RenderAtlas( const Camera& world_camera )
{
	//Render the models with respect to only the base (non-mirrored) section.
	//We'll just re-use these renders for the mirrored section in the Billboard Phase.
	m_sections[0].RenderAtlas( world_camera );
}

//---------------------------------------------------------------------------------------------------------------------------------

void CrowdGroup::RenderBillboards( Shader& vertex_shader, Shader& fragment_shader )
{
	for( unsigned int i=0; i<m_sections.size(); ++i )
	{
		m_sections[i].RenderBillboards( vertex_shader, fragment_shader );
	}
}

//---------------------------------------------------------------------------------------------------------------------------------

void CrowdGroup::Shutdown( void )
{
	unsigned int section_count = m_sections.size();
	for( unsigned int i=0; i<section_count; ++i )
	{
		m_sections[i].Shutdown();
	}
	m_sections.clear();
}

//---------------------------------------------------------------------------------------------------------------------------------
