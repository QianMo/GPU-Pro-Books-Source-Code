/**
 *	@file
 *	@brief		Crowd mesh data. Ideally this would be loaded in on a per-stadium basis and allocate the appropriate amount of groups.
 *	@author		Alan Chambers
 *	@date		2011
**/

#ifndef CROWDMESH_H
#define CROWDMESH_H

#include "CrowdGroup.h"

class CrowdMesh
{
public:
									CrowdMesh( void );
									~CrowdMesh( void );
	
	void							Create( unsigned int main_width, unsigned int main_height, unsigned int side_width, unsigned int side_height );

	CrowdGroup&						GetGroup( unsigned int index );
	unsigned int					GetGroupCount( void ) const;

	void							Shutdown( void );

private:
	std::vector<CrowdGroup>			m_groups;
};

#include "CrowdMesh.inl"

#endif
