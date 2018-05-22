/**
 *	@file
 *	@ingroup	Simple Cg shader setup.
 *	@author		Alan Chambers
 *	@date		2011
**/

#ifndef SHADER_H
#define SHADER_H

#include <cg/cg.h>

class Shader
{
public:
								Shader( void );
								~Shader( void );
	
	class Buffer
	{
	public:
								Buffer( void );
								~Buffer( void );

		void					Create( const void* buffer, unsigned int size );

		CGbuffer				GetId( void ) const;

		void					Shutdown( void );

	private:
		CGbuffer				m_id;
	};

	void						Bind( void ) const;
	void						Bind( unsigned int index, const Buffer& buffer ) const;

    bool                        Create( const char* code, const char* profile, const char* entry="main" );

	static void					Initialize( void );
	static bool					IsProfileSupported( const char* profile );

	void						Release( void );
	
	void						SetParameter( const char* name, const void* param, unsigned int size );
	static void					Shutdown( void );
	    
private:
	CGprogram					m_program;
	CGprofile					m_profile;
	static CGcontext			ms_context;
};

#include "Shader.inl"

#endif
