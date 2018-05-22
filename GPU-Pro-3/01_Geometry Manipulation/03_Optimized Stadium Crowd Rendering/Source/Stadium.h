/**
 *	@file
 *	@brief		A class that allows us to generate a low-poly mesh to represent a really basic stadium.
 *	@author		Alan Chambers
 *	@date		2011
**/

#ifndef PITCH_H
#define PITCH_H

#include "Shader.h"

class Camera;

class Stadium
{
public:
									Stadium( void );
									~Stadium( void );
	
	void							Create( unsigned int width, unsigned int height, unsigned int length );

	void							Render( const Camera& camera );

	void							Shutdown( void );

private:
	unsigned int					m_vb;
	unsigned int					m_ib;
	Shader							m_vertex_shader;
	Shader							m_fragment_shader;
};

#endif
