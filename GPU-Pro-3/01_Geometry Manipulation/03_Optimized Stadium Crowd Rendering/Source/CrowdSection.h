/**
 *	@file
 *	@brief		A section of the crowd mesh that we render our model set from with regard to the camera and large scale viewpoints.
 *	@author		Alan Chambers
 *	@date		2011
**/

#ifndef CROWDSECTION_H
#define CROWDSECTION_H

#include "Matrix4.h"
#include "Shader.h"
#include "Vector4.h"

#ifdef _WIN32
#include <windows.h>
#include <gl/glew.h>
#else
#include <opengl/gl.h>
#endif

#include <vector>

class Shader;
class Camera;

class CrowdSection
{
public:
									CrowdSection( void );
									~CrowdSection( void );

	void							Create( const Vector4& offset );

									//Ideally the crowd mesh data would be loaded in on a per-stadium basis.
	void							GenerateGeometry( int width, int height );

	bool							IsMirror( void ) const;

	void							RenderAtlas( const Camera& world_camera ) const;
	void							RenderBillboards( Shader& vertex_shader, Shader& fragment_shader ) const;

	void							Shutdown( void );
	
protected:
	Vector4							ComputeCameraPosition( const Camera& camera );

private:
	unsigned int					m_quad_id;		//@brief OpenGL stream id for our instanced quads.
	unsigned int					m_seat_count;	//@brief Number of seats in this section.
	Shader::Buffer					m_buffers[2];	//@brief One uniform buffer for each stream (positions and properties).
	Matrix4							m_world;		//@brief World matrix for the section.
	float							m_mirror;		//@brief Controls whether to use mirrored rendering for this section.
	//std::vector<CrowdModel*>		m_models;		//@brief Each crowd section can have it's own list of renderable models that is a subset of those in the crowd manager.
};

#endif
