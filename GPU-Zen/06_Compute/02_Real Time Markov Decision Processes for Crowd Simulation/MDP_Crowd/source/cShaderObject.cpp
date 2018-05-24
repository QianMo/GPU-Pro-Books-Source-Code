/*

Copyright 2013,2014 Sergio Ruiz, Benjamin Hernandez

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>

In case you, or any of your employees or students, publish any article or
other material resulting from the use of this  software, that publication
must cite the following references:

Sergio Ruiz, Benjamin Hernandez, Adriana Alvarado, and Isaac Rudomin. 2013.
Reducing Memory Requirements for Diverse Animated Crowds. In Proceedings of
Motion on Games (MIG '13). ACM, New York, NY, USA, , Article 55 , 10 pages.
DOI: http://dx.doi.org/10.1145/2522628.2522901

Sergio Ruiz and Benjamin Hernandez. 2015. A Parallel Solver for Markov Decision Process
in Crowd Simulations. Fourteenth Mexican International Conference on Artificial
Intelligence (MICAI), Cuernavaca, 2015, pp. 107-116.
DOI: 10.1109/MICAI.2015.23

*/
#include "cGlslManager.h"

//=======================================================================================
//
ShaderObject::ShaderObject( LogManager*	log_manager_,
							char*		fragment 	)
{
	shader_id			= 0;
	num_vars			= 0;
	log_manager			= log_manager_;
	shaderOK			= init( fragment );
}
//
//=======================================================================================
//
ShaderObject::ShaderObject( LogManager*	log_manager_,
							char*		vertex,
							char*		fragment     )
{
	shader_id			= 0;
	num_vars			= 0;
	log_manager			= log_manager_;
	shaderOK			= init( vertex, fragment );
}
//
//=======================================================================================
//
#ifdef USE_GEOMETRY_SHADERS
ShaderObject::ShaderObject( LogManager*		log_manager_,
							char*			vertex,
							char*			fragment,
							char*			geometry,
							unsigned int	inPrimitive,
							unsigned int	outPrimitive    )
{
	shader_id			= 0;
	num_vars			= 0;
	log_manager			= log_manager_;
	shaderOK			= init( vertex, fragment, geometry, inPrimitive, outPrimitive );
}
//
//=======================================================================================
//
ShaderObject::ShaderObject( LogManager*		log_manager_,
							char*			vertex,
							char*			geometry,
							unsigned int	inPrimitive,
							unsigned int	outPrimitive    )
{
	shader_id			= 0;
	num_vars			= 0;
	log_manager			= log_manager_;
	shaderOK			= init( vertex, geometry, inPrimitive, outPrimitive );
}
#endif
//
//=======================================================================================
//
ShaderObject::~ShaderObject( void )
{
	if( shader_id != 0 )
	{
		glDeleteShader( shader_id );
	}
}
//
//=======================================================================================
//
char* ShaderObject::loadShaderCode( const char* filename )
{
	FILE*			shFile		= 0;
	char*			fileData	= 0;
	unsigned long	len			= 0;
	unsigned long	readLen		= 0;

	shFile = fopen( filename , "r" );
	if ( !shFile )
	{
		return NULL;
	}

	fseek( shFile , 0 , SEEK_END );
	len = ftell( shFile );
	rewind( shFile );

	fileData = (char*)malloc(len+1);
	if( fileData == NULL )
	{
		return NULL;
	}

	readLen = fread( fileData, 1, len, shFile );
	fclose( shFile );
	fileData[readLen] = 0;
	return fileData;
}
//
//=======================================================================================
//
bool ShaderObject::init( char* fragment )
{
	int			progLinkSuccess;
	char* 		Shader;
	GLuint		fragmentHandler;
	GLuint		programObject;
	string_utils_buffer_ptr = 0;
	shader_id				= 0;

	if( (Shader = loadShaderCode( fragment )) == NULL )
	{
		log_manager->log( LogManager::LERROR, "Unable to load fragment shader \"%s\"!", fragment );
		return false;
	}
	fragmentHandler = glCreateShader( GL_FRAGMENT_SHADER );
	glShaderSource( fragmentHandler, 1, (const GLchar**)&Shader, NULL );
	free( Shader );
	glCompileShader( fragmentHandler );
	programObject = glCreateProgram();
	glAttachShader( programObject, fragmentHandler );
	glLinkProgram( programObject );
	glDeleteShader( fragmentHandler );
    glGetProgramiv( programObject, GL_LINK_STATUS, &progLinkSuccess );
	if( !progLinkSuccess )
	{
		glGetProgramiv( programObject, GL_INFO_LOG_LENGTH, &progLinkSuccess );
		Shader = (char*)malloc( progLinkSuccess );
		glGetProgramInfoLog( programObject, progLinkSuccess, NULL, Shader );
		log_manager->log( LogManager::LERROR, "Shader \"%s\" could not be linked: \"%s\"!", fragment, Shader );
		free( Shader );
        return false;
	}
	string name;
	StringUtils::getNameFromPath( fragment, name );
	log_manager->log( 	LogManager::GLSL_MANAGER,
						"Build succeeded for \"%s\".",
						name.c_str()					);
	shader_id = programObject;
	glUseProgram( shader_id );
	num_vars = 0;
	return initVarInfo();
}
//
//=======================================================================================
//
bool ShaderObject::init( char* vertex, char* fragment )
{
	int			progLinkSuccess;
	char*		Shader = NULL;
	GLuint		vertexHandler;
	GLuint		fragmentHandler;
	GLuint		programObject;
	string_utils_buffer_ptr = 0;
	shader_id				= 0;
	if ( (Shader = loadShaderCode( vertex )) == NULL )
	{
		log_manager->log( LogManager::LERROR, "Unable to load vertex shader \"%s\"!", vertex );
		return false;
	}
	vertexHandler = glCreateShader( GL_VERTEX_SHADER );
	glShaderSource( vertexHandler, 1, (const GLchar**)&Shader, NULL );
	free( Shader );
	if( (Shader = loadShaderCode( fragment )) == NULL )
	{
		log_manager->log( LogManager::LERROR, "Unable to load fragment shader \"%s\"!", fragment );
		return false;
	}
	fragmentHandler = glCreateShader( GL_FRAGMENT_SHADER );
	glShaderSource( fragmentHandler, 1, (const GLchar**)&Shader, NULL );
    free( Shader );
	glCompileShader( vertexHandler );
	glCompileShader( fragmentHandler );
	programObject = glCreateProgram();
	glAttachShader( programObject, vertexHandler );
	glAttachShader( programObject, fragmentHandler );
	glLinkProgram( programObject );
	glDeleteShader( vertexHandler );
	glDeleteShader( fragmentHandler );
    glGetProgramiv( programObject, GL_LINK_STATUS, &progLinkSuccess );
	if( !progLinkSuccess )
	{
		glGetProgramiv( programObject, GL_INFO_LOG_LENGTH, &progLinkSuccess );
		Shader = (char*)malloc( progLinkSuccess );
		glGetProgramInfoLog( programObject, progLinkSuccess, NULL, Shader );
		log_manager->log( LogManager::LERROR, "Shader \"%s\"-\"%s\" could not be linked: \"%s\"!", vertex, fragment, Shader );
		free( Shader );
        return false;
	}
	string nV = StringUtils::getStrNameFromPath( vertex );
	string nF = StringUtils::getStrNameFromPath( fragment );
	log_manager->log( 	LogManager::GLSL_MANAGER,
						"Build succeeded for \"%s\"-\"%s\".",
						nV.c_str(),
						nF.c_str() 							);
	shader_id = programObject;
	glUseProgram( shader_id );
	num_vars = 0;
	return initVarInfo();
}
//
//=======================================================================================
//
#ifdef USE_GEOMETRY_SHADERS
bool ShaderObject::init( char*			vertex,
						 char*			fragment,
						 char*			geometry,
						 unsigned int	inPrimitive,
						 unsigned int	outPrimitive    )
{
	int			progLinkSuccess;
	char* 		Shader;
	GLuint		vertexHandler;
	GLuint		fragmentHandler;
	GLuint		geometryHandler;
	GLuint		programObject;
	string_utils_buffer_ptr = 0;
	shader_id				= 0;

	if( (Shader = loadShaderCode( vertex )) == NULL )
	{
		log_manager->log( LogManager::LERROR, "Unable to load vertex shader \"%s\"!", vertex );
		return false;
	}
	vertexHandler = glCreateShader( GL_VERTEX_SHADER );
	glShaderSource( vertexHandler, 1, (const GLchar**)&Shader, NULL	);
	free( Shader );
	if( (Shader = loadShaderCode( geometry )) == NULL )
	{
		log_manager->log( LogManager::LERROR, "Unable to load geometry shader \"%s\"!", geometry );
		return false;
	}
	geometryHandler = glCreateShader( GL_GEOMETRY_SHADER );
	glShaderSource( geometryHandler, 1, (const GLchar**)&Shader, NULL );
	free( Shader );
	if( (Shader = loadShaderCode( fragment )) == NULL )
	{
		log_manager->log( LogManager::LERROR, "Unable to load fragment shader \"%s\"!", fragment );
		return false;
	}
	fragmentHandler = glCreateShader( GL_FRAGMENT_SHADER );
	glShaderSource( fragmentHandler, 1, (const GLchar**)&Shader, NULL );
	free( Shader );

	glCompileShader( vertexHandler   );
	glCompileShader( geometryHandler );
	glCompileShader( fragmentHandler );

	programObject = glCreateProgram();
	glAttachShader( programObject, vertexHandler   );
	glAttachShader( programObject, geometryHandler );
	glAttachShader( programObject, fragmentHandler );

    glProgramParameteri( programObject, GL_GEOMETRY_INPUT_TYPE,   inPrimitive  );
	glProgramParameteri( programObject, GL_GEOMETRY_OUTPUT_TYPE,  outPrimitive );
	glProgramParameteri( programObject, GL_GEOMETRY_VERTICES_OUT, 4            );

	glLinkProgram( programObject );

	glDeleteShader( vertexHandler   );
	glDeleteShader( geometryHandler );
	glDeleteShader( fragmentHandler );

    glGetProgramiv( programObject, GL_LINK_STATUS, &progLinkSuccess );

	if( !progLinkSuccess )
	{
		glGetProgramiv( programObject, GL_INFO_LOG_LENGTH, &progLinkSuccess );
		Shader = (char*)malloc( progLinkSuccess );
		glGetProgramInfoLog( programObject, progLinkSuccess, NULL, Shader );
		log_manager->log( LogManager::LERROR, "Unable to link shader: \"%s\"-\"%s\"-\"%s\": \"%s\"!", vertex, geometry, fragment, Shader );
		free( Shader );
        return false;
	}

	string nV = StringUtils::getStrNameFromPath( vertex );
	string nG = StringUtils::getStrNameFromPath( geometry );
	string nF = StringUtils::getStrNameFromPath( fragment );
	log_manager->log( 	LogManager::GLSL_MANAGER,
						"Build succeeded for \"%s\"-\"%s\"-\"%s\".",
						nV.c_str(),
						nG.c_str(),
						nF.c_str() 									);
	shader_id = programObject;
	glUseProgram( shader_id );
	num_vars = 0;
	return initVarInfo();
}
//
//=======================================================================================
//
bool ShaderObject::init( char*			vertex,
						 char*			geometry,
						 unsigned int	inPrimitive,
						 unsigned int	outPrimitive    )
{
	int			progLinkSuccess;
	char*		Shader;
	GLuint		vertexHandler;
	GLuint		geometryHandler;
	GLuint		programObject;
	string_utils_buffer_ptr = 0;
	shader_id				= 0;

	if( (Shader = loadShaderCode( vertex )) == NULL )
	{
		log_manager->log( LogManager::LERROR, "Unable to load vertex shader \"%s\"!", vertex );
		return false;
	}
	vertexHandler = glCreateShader( GL_VERTEX_SHADER );
	glShaderSource( vertexHandler, 1, (const GLchar**)&Shader, NULL );
	free( Shader );
	if( (Shader = loadShaderCode( geometry )) == NULL )
	{
		log_manager->log( LogManager::LERROR, "Unable to load geometry shader \"%s\"!", geometry );
		return false;
	}
	geometryHandler = glCreateShader( GL_GEOMETRY_SHADER );
	glShaderSource( geometryHandler, 1, (const GLchar**)&Shader, NULL );
	free( Shader );
	glCompileShader( vertexHandler   );
	glCompileShader( geometryHandler );

	programObject = glCreateProgram();
	glAttachShader( programObject, vertexHandler   );
	glAttachShader( programObject, geometryHandler );

	glProgramParameteri( programObject, GL_GEOMETRY_INPUT_TYPE_EXT,   inPrimitive  );
	glProgramParameteri( programObject, GL_GEOMETRY_OUTPUT_TYPE_EXT,  outPrimitive );
	int temp;
	glGetIntegerv( GL_MAX_GEOMETRY_OUTPUT_VERTICES, &temp );
	glProgramParameteri( programObject, GL_GEOMETRY_VERTICES_OUT_EXT, temp );

	glLinkProgram( programObject );

	glDeleteShader( vertexHandler   );
	glDeleteShader( geometryHandler );

    glGetProgramiv( programObject, GL_LINK_STATUS, &progLinkSuccess );

	if( !progLinkSuccess )
	{
		glGetProgramiv( programObject, GL_INFO_LOG_LENGTH, &progLinkSuccess );
		Shader = (char*)malloc( progLinkSuccess );
		glGetProgramInfoLog( programObject, progLinkSuccess, NULL, Shader );
		log_manager->log( LogManager::GLSL_MANAGER, "Unable to link shader: \"%s\"-\"%s\": \"%s\"!", vertex, geometry, Shader );
		free( Shader );
        return false;
	}
	string nV = StringUtils::getStrNameFromPath( vertex );
	string nG = StringUtils::getStrNameFromPath( geometry );
	log_manager->log( 	LogManager::GLSL_MANAGER,
						"Build succeeded for \"%s\"-\"%s\".",
						nV.c_str(),
						nG.c_str() 							);
	shader_id = programObject;
	glUseProgram( shader_id );
	num_vars = 0;
	return initVarInfo();
}
#endif
//
//=======================================================================================
//
bool ShaderObject::initVarInfo( void )
{
	char *buffer;
	int   len = 0;
	int   bufLen = 0;
	const char* Shader;
	if( !shader_id ) return false;
	num_vars = 0;
	int progValid;
	glGetProgramiv( shader_id, GL_ACTIVE_UNIFORMS, &num_vars );
	glValidateProgram( shader_id );
	glGetProgramiv( shader_id, GL_VALIDATE_STATUS, &progValid );
	if( progValid != GL_TRUE )
	{
		glGetProgramiv( shader_id, GL_INFO_LOG_LENGTH, &progValid );
		Shader = new char[progValid];
		glGetProgramInfoLog( shader_id, progValid, NULL, (char*)Shader );
		log_manager->log( LogManager::LERROR, "INVALID_PROGRAM: \"%s\"!", Shader );
		delete [] Shader;
		Shader = NULL;
		return false;
	}

	glGetProgramiv( shader_id, GL_ACTIVE_UNIFORM_MAX_LENGTH, &bufLen   );
	glValidateProgram( shader_id );
	glGetProgramiv( shader_id, GL_VALIDATE_STATUS, &progValid );
	if( progValid != GL_TRUE )
	{
		glGetProgramiv( shader_id, GL_INFO_LOG_LENGTH, &progValid );
		Shader = new char[progValid];
		glGetProgramInfoLog( shader_id, progValid, NULL, (char*)Shader );
		log_manager->log( LogManager::LERROR, "INVALID_PROGRAM: \"%s\"!", Shader );
		delete [] Shader;
		Shader = NULL;
		return false;
	}
	buffer = new char[bufLen + 1];
	for( int i = 0; i < num_vars; ++i )
	{
		variable_ids[i]		= i;
		glGetActiveUniform( shader_id,
							i,
							bufLen,
							&len,
							&variable_sizes[i],
							&variable_types[i],
							buffer );
		buffer[len + 1]		= 0;
		variable_names[i]	= registerString( buffer );
	}
	delete buffer;
	return true;
}
//
//=======================================================================================
//
void ShaderObject::setUniformi( char* name, int value )
{
	for( int i = 0; i < num_vars; ++i )
	{
		if( strcmp( name, variable_names[i] ) == 0 )
		{
			glUniform1i( variable_ids[i], value );
			break;
		}
	}
}
//
//=======================================================================================
//
void ShaderObject::setUniformf( char* name, float value )
{
	for( int i = 0; i < num_vars; ++i )
	{
		if( strcmp( name, variable_names[i] ) == 0 )
		{
			glUniform1f( variable_ids[i], value );
			break;
		}
	}
}
//
//=======================================================================================
//
void ShaderObject::setUniformiv( char* name, int* value, int size )
{
	unsigned int i = glGetUniformLocation( shader_id, name );
	switch( size )
	{
		case 1:
			glUniform1iv( i, 1, value );
			break;
		case 2:
			glUniform2iv( i, 1, value );
			break;
		case 3:
			glUniform3iv( i, 1, value );
			break;
		case 4:
			glUniform4iv( i, 1, value );
			break;
	}
}
//
//=======================================================================================
//
void ShaderObject::setUniformfv( char* name, float* value, int size )
{
	unsigned int i = glGetUniformLocation( shader_id, name );
	switch( size )
	{
		case 1:
			glUniform1fv( i, 1, value );
			break;
		case 2:
			glUniform2fv( i, 1, value );
			break;
		case 3:
			glUniform3fv( i, 1, value );
			break;
		case 4:
			glUniform4fv( i, 1, value );
			break;
		case 7:
			glUniform1fv( i, 7, value );
			break;
	}
}
//
//=======================================================================================
//
void ShaderObject::setUniformMatrix( char* name, float* value, int size )
{
	unsigned int i = glGetUniformLocation( shader_id, name );
	switch( size )
	{
		case 2:
			glUniformMatrix2fv( i, 1, GL_FALSE, value );
			break;
		case 3:
			glUniformMatrix3fv( i, 1, GL_FALSE, value );
			break;
		case 4:
			glUniformMatrix4fv( i, 1, GL_FALSE, value );
			break;
	}
}
//
//=======================================================================================
//
void ShaderObject::activate( void )
{
	glUseProgram( shader_id );
}
//
//=======================================================================================
//
void ShaderObject::deactivate( void )
{
	glUseProgram( 0 );
}
//
//=======================================================================================
//
void ShaderObject::setBindableUniform( char* name, unsigned int buffer )
{
	for( int i = 0; i < num_vars; ++i )
	{
		if( strcmp( name, variable_names[i] ) == 0 )
		{
			glUniformBufferEXT( shader_id, variable_ids[i], buffer );
			break;
		}
	}
}
//
//=======================================================================================
//
void ShaderObject::unSetBindableUniform( char* name )
{
	for( int i = 0; i < num_vars; ++i )
	{
		if( strcmp( name, variable_names[i] ) == 0 )
		{
			glUniformBufferEXT( shader_id, variable_ids[i], 0 );
			break;
		}
	}
}
//
//=======================================================================================
//
int ShaderObject::getBindableUniformSize( char* name )
{
	for( int i = 0; i < num_vars; ++i )
	{
		if( strcmp( name, variable_names[i] ) == 0 )
		{
			return glGetUniformBufferSizeEXT( shader_id, variable_ids[i] );
		}
	}
	return -1;
}
//
//=======================================================================================
//
void ShaderObject::getMaxBindableUniformSize( int& val )
{
	if( GL_MAX_BINDABLE_UNIFORM_SIZE_EXT )
	{
		glGetIntegerv( GL_MAX_BINDABLE_UNIFORM_SIZE_EXT, &val );
	}
	else
	{
		log_manager->log( LogManager::LERROR, "Extension \"GL_MAX_BINDABLE_UNIFORM_SIZE\" not supported!" );
		val = -1;
	}
}
//
//=======================================================================================
//
void ShaderObject::getMaxVertexBindableUniforms( int& val )
{
	if( GL_MAX_VERTEX_BINDABLE_UNIFORMS_EXT )
	{
		glGetIntegerv( GL_MAX_VERTEX_BINDABLE_UNIFORMS_EXT, &val );
	}
	else
	{
		log_manager->log( LogManager::LERROR, "Extension \"GL_MAX_VERTEX_BINDABLE_UNIFORMS\" not supported!" );
		val = -1;
	}
}
//
//=======================================================================================
//
void ShaderObject::getMaxFragmentBindableUniforms( int& val )
{
	if( GL_MAX_FRAGMENT_BINDABLE_UNIFORMS_EXT )
	{
		glGetIntegerv( GL_MAX_FRAGMENT_BINDABLE_UNIFORMS_EXT, &val );
	}
	else
	{
		log_manager->log( LogManager::LERROR, "Extension \"GL_MAX_FRAGMENT_BINDABLE_UNIFORMS\" not supported!" );
		val = -1;
	}
}
//
//=======================================================================================
//
void ShaderObject::getMaxGeometryBindableUniforms( int& val )
{
	if( GL_MAX_GEOMETRY_BINDABLE_UNIFORMS_EXT )
	{
		glGetIntegerv( GL_MAX_GEOMETRY_BINDABLE_UNIFORMS_EXT, &val );
	}
	else
	{
		log_manager->log( LogManager::LERROR, "Extension \"GL_MAX_GEOMETRY_BINDABLE_UNIFORMS\" not supported!" );
		val = -1;
	}
}
//
//=======================================================================================
//
char* ShaderObject::registerString( char* str )
{
	int		len;
	char*	retVal;
	len = (int)strlen( str );
	strncpy( &string_utils_buffer[string_utils_buffer_ptr], str, len + 1 );
	retVal = &string_utils_buffer[string_utils_buffer_ptr];
	string_utils_buffer_ptr += len + 1;
	return retVal;
}
//
//=======================================================================================
//
bool ShaderObject::isShaderOk( void )
{
	return shaderOK;
}
//
//=======================================================================================
