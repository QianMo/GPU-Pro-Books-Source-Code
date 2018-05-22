/**
 *	@file
 *	@brief		Basic API for updating the components of the crowd system.
 *	@author		Alan Chambers
 *	@date		2011
**/

#ifndef APP_H
#define APP_H

#include "Camera.h"
#include "Crowd.h"
#include "Stadium.h"

class App
{
public:
									App( void );
									~App( void );
	
	static App*						Get( void ) { return m_app; }
	
	void							HandleInput( unsigned char key );
	
	void							Initialize( void );
	
	void							Render( void );

	void							Shutdown( void );
	
	void							Update( float dt );
		
private:
	Camera							m_camera;
	Stadium							m_stadium;
	Crowd							m_crowd;
	static App*						m_app;
};

#endif
