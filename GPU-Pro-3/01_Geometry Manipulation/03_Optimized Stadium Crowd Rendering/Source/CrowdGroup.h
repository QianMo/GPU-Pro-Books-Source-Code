/**
 *	@file
 *	@brief		A group that controls the symmetrical rendering optimizations for opposing crowd sections.
 *	@author		Alan Chambers
 *	@date		2011
**/

#ifndef CROWDGROUP_H
#define CROWDGROUP_H

#include "CrowdSection.h"

class Shader;

class CrowdGroup
{
public:
									CrowdGroup( void );
									~CrowdGroup( void );
	
	void							Create( unsigned int section_count );

	CrowdSection&					GetSection( unsigned int index );
	unsigned int					GetSectionCount( void ) const;

	void							RenderAtlas( const Camera& world_camera );
	void							RenderBillboards( Shader& vertex_shader, Shader& fragment_shader );

	void							Shutdown( void );
	
private:
	std::vector<CrowdSection>		m_sections;
};

#include "CrowdGroup.inl"

#endif
