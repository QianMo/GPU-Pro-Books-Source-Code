#include "Shader.h"
#include <cg/cggl.h>
#include <iostream>

//---------------------------------------------------------------------------------------------------------------------------------

CGcontext Shader::ms_context = 0;

//---------------------------------------------------------------------------------------------------------------------------------

Shader::Buffer::Buffer( void ) :
m_id( 0 )
{

}

//---------------------------------------------------------------------------------------------------------------------------------

Shader::Buffer::~Buffer( void )
{

}

//---------------------------------------------------------------------------------------------------------------------------------

void Shader::Buffer::Create( const void* data, unsigned int size )
{
	m_id = cgCreateBuffer( ms_context, size, data, CG_BUFFER_USAGE_STATIC_DRAW );
}

//---------------------------------------------------------------------------------------------------------------------------------

void Shader::Buffer::Shutdown( void )
{
	cgDestroyBuffer( m_id );
}

//---------------------------------------------------------------------------------------------------------------------------------

Shader::Shader( void ) :
	m_program( 0 ),
	m_profile( CG_PROFILE_VP40 )
{
	
}

//---------------------------------------------------------------------------------------------------------------------------------
	
Shader::~Shader( void )
{
	Release();
}

//---------------------------------------------------------------------------------------------------------------------------------

void Shader::Bind( void ) const
{
	cgGLLoadProgram( m_program );
	cgGLBindProgram( m_program );
	cgGLEnableProfile( m_profile );
}

//---------------------------------------------------------------------------------------------------------------------------------

void Shader::Bind( unsigned int index, const Buffer& buffer ) const
{
	cgSetProgramBuffer( m_program, index, buffer.GetId() );
}

//---------------------------------------------------------------------------------------------------------------------------------

bool Shader::Create( const char* code, const char* profile, const char* entry_point )
{
	m_profile = cgGetProfile( profile );
	
	if( m_profile == 0 )
		return false;
	
	m_program = cgCreateProgram( ms_context, CG_SOURCE, code, m_profile, entry_point, 0 );
		
	if( m_program == 0 )
	{
		const char* lastListing = cgGetLastListing( ms_context );
		
		if( lastListing )
		{
			std::cout << lastListing << std::endl;
		}
		
		return false;
	}

	return true;
}

//---------------------------------------------------------------------------------------------------------------------------------

void Shader::Initialize( void )
{
	ms_context = cgCreateContext();
}

//---------------------------------------------------------------------------------------------------------------------------------

bool Shader::IsProfileSupported( const char* profile )
{
	CGprofile cg_profile = cgGetProfile( profile );

	return cgGLIsProfileSupported( cg_profile ) == CG_TRUE;
}

//---------------------------------------------------------------------------------------------------------------------------------

void Shader::Release( void )
{
	cgDestroyProgram( m_program );
}

//---------------------------------------------------------------------------------------------------------------------------------

void Shader::SetParameter( const char* name, const void* parameter, unsigned int size )
{
	CGparameter param = cgGetNamedParameter( m_program, name );
	CGtype type = cgGetParameterType( param );
	
	if( type == CG_SAMPLER2D )
	{
		cgGLSetTextureParameter( param, *static_cast<const unsigned int*>( parameter ) );
		cgGLEnableTextureParameter( param );
	}
	else
	{
		cgSetParameterValuefc( param, size, static_cast<const float*>( parameter ) );
	}
}

//---------------------------------------------------------------------------------------------------------------------------------

void Shader::Shutdown( void )
{
	cgDestroyContext( ms_context );
}
	
//---------------------------------------------------------------------------------------------------------------------------------
