/**
 *	@file
 *	@brief		Crowd class that controls how the game interacts with the crowd.
 *	@author		Alan Chambers
 *	@date		2011
**/

#ifndef CROWD_H
#define CROWD_H

#include "CrowdMesh.h"
#include "Shader.h"
#include "Texture.h"

class Camera;

class Crowd
{
public:	
									Crowd( void );
									~Crowd( void );
	
	void							Create( unsigned int main_width, unsigned int main_height, unsigned int side_width, unsigned int side_height );

	void							AdjustDensity( unsigned int percent );

	void							Render( const Camera& world_camera );

	void							Shutdown( void );
	
	void							Update( float dt );

protected:
	void							RenderAtlas( const Camera& world_camera );
	void							RenderBillboards( const Camera& world_camera );
	void							RenderDeferred( void );

private:
	Shader							m_vertex_shader;
	Shader							m_fragment_shader;
	Texture							m_base_map;
	Texture							m_normal_map;
	Vector4							m_light_dir;
	Vector4							m_light_col;
	Vector4							m_light_amb;
	int								m_density;
	CrowdMesh						m_mesh;
};

#endif
