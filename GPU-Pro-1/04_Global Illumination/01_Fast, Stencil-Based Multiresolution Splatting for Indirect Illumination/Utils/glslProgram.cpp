/**************************************************************************
** glslProgram.cpp                                                       **
** ------------                                                          **
**                                                                       **
** This file implements a class for loading and initializing a _single_  **
**   GLSL program (which consists of a set of vertex, geometry, and/or   **
**   fragment shaders linked into a "program")                           **
**                                                                       **
** Please see the header file for further "README" type notes.           **
**                                                                       **
** Chris Wyman (12/4/2007)                                               **
**************************************************************************/

#include "glslProgram.h"

#pragma warning( disable: 4996 )

/*
#define BIND_UNKNOWN      0 
#define BIND_FLOAT        1
#define BIND_VEC2         2
#define BIND_VEC3         3
#define BIND_VEC4         4
#define BIND_MAT2         5
#define BIND_MAT3         6
#define BIND_MAT4         7
#define BIND_TEX2D_PTR    8
#define BIND_MAX          8

class GLSLBindings
{
public:
	GLint uniformLocation;
	GLuint bindingType; 
	float *boundC_variable;
	GLuint textureID;
	GLenum textureUnit;
	char *shaderVarName;
};
*/

GLSLProgram::GLSLProgram( bool verboseError, PathList *path ) :
	verbose(verboseError), vertShaderID(0), geomShaderID(0), fragShaderID(0),
	isLinked(false), enabled(false), shaderEnableFlags(0), shaderDisableFlags(0)
{
	shaderSearchPath = path;
	vertShaderFile = NULL;
	geomShaderFile = NULL;
	fragShaderFile = NULL;
	geomVerticesOut = 0; // OpenGL default... Silly, because it gives a linker error at 0!
	geomInputType = GL_TRIANGLES;
	geomOutputType = GL_TRIANGLE_STRIP;
	programID = glCreateProgram();
	if (programID == 0)
	{
		fprintf(stderr, "***Fatal Error: glCreateProgram() failed in GLSLProgram constructor!\n");
		exit(0);
	}
}

GLSLProgram::GLSLProgram( char *vShader, char *gShader, char *fShader, bool verboseErrors, PathList *path, bool linkImmediately ) :
	verbose(verboseErrors), vertShaderID(0), geomShaderID(0), fragShaderID(0), isLinked(false), enabled(false),
	shaderEnableFlags(0), shaderDisableFlags(0)
{
	geomVerticesOut = 0; // OpenGL default... Silly, because it gives a linker error at 0!
	shaderSearchPath = path;
	vertShaderFile = geomShaderFile = fragShaderFile = NULL;
	geomInputType = GL_TRIANGLES;
	geomOutputType = GL_TRIANGLE_STRIP;
	programID = glCreateProgram();
	if (programID == 0)
	{
		fprintf(stderr, "***Fatal Error: glCreateProgram() failed in GLSLProgram constructor!\n");
		exit(0);
	}
	if ( vShader ) SetVertexShader( vShader );
	if ( gShader ) SetGeometryShader( gShader );
	if ( fShader ) SetFragmentShader( fShader );
	if ( linkImmediately ) LinkProgram();
}


GLSLProgram::~GLSLProgram()
{
	// If we passed in a shader via filename (i.e. so this class loaded
	//    the shader from file) then we are responsible for cleaning up,
	//    which means (a) free the OpenGL shader, (b) free the duplicated
	//    storage for the filename.
	if (vertShaderFile) 
	{ 
		free(vertShaderFile); 
		glDeleteShader( vertShaderID ); 
	}
	if (geomShaderFile) 
	{
		free(geomShaderFile);
		glDeleteShader( geomShaderID );
	}
	if (fragShaderFile) 
	{
		free(fragShaderFile);
		glDeleteShader( fragShaderID );
	}

	for (unsigned int i=0; i < autoBindUniforms.Size(); i++)
	{
		if (autoBindUniforms[i]->shaderVarName)
			free (autoBindUniforms[i]->shaderVarName);
		delete autoBindUniforms[i];
	}
}


bool GLSLProgram::SetVertexShader  ( char *shaderFilename )
{
	// Load the shader code
	char *tmpString = ReturnFileAsString( shaderFilename );
	if (!tmpString && verbose) 
		fprintf(stderr, "***Error: Unable to load vertex shader '%s'!\n", shaderFilename );
	if (!tmpString) return false;

	// Copy the filename (now that we know it is valid) in case we need to reload.
	vertShaderFile = strdup( shaderFilename );

	// Create an OpenGL shader object, give OpenGL the source, and free the temp memory
	vertShaderID = glCreateShader( GL_VERTEX_SHADER );
	glShaderSource( vertShaderID, 1, (const char **)&tmpString, 0 );
	free( tmpString );

	// Copile the shader, and check if there were compilation errors
	GLint compiled = 0;
	glCompileShader( vertShaderID );
	glGetShaderiv( vertShaderID, GL_COMPILE_STATUS, &compiled);

	// If there were compile errors, we might want to print them out.
	if (!compiled)
	{
		PrintCompilerError( vertShaderID, vertShaderFile );
		return false;
	}

	glAttachShader( programID, vertShaderID );
	return true;
}

bool GLSLProgram::SetGeometryShader( char *shaderFilename )
{
	// Load the shader code
	char *tmpString = ReturnFileAsString( shaderFilename );
	if (!tmpString && verbose) 
		fprintf(stderr, "***Error: Unable to load geometry shader '%s'!\n", shaderFilename );
	if (!tmpString) return false;

	// Copy the filename (now that we know it is valid) in case we need to reload.
	geomShaderFile = strdup( shaderFilename );

	// Create an OpenGL shader object, give OpenGL the source, and free the temp memory
	geomShaderID = glCreateShader( GL_GEOMETRY_SHADER_EXT );
	glShaderSource( geomShaderID, 1, (const char **)&tmpString, 0 );
	free( tmpString );

	// Copile the shader, and check if there were compilation errors
	GLint compiled = 0;
	glCompileShader( geomShaderID );
	glGetShaderiv( geomShaderID, GL_COMPILE_STATUS, &compiled);

	// Print out compiler errors.
	if (!compiled)
	{
		PrintCompilerError( geomShaderID, geomShaderFile );
		return false;
	}

	glAttachShader( programID, geomShaderID );
	return true;
}

bool GLSLProgram::SetFragmentShader( char *shaderFilename )
{
	// Load the shader code
	char *tmpString = ReturnFileAsString( shaderFilename );
	if (!tmpString && verbose) 
		fprintf(stderr, "***Error: Unable to load fragment shader '%s'!\n", shaderFilename );
	if (!tmpString) return false;

	// Copy the filename (now that we know it is valid) in case we need to reload.
	fragShaderFile = strdup( shaderFilename );

	// Create an OpenGL shader object, give OpenGL the source, and free the temp memory
	fragShaderID = glCreateShader( GL_FRAGMENT_SHADER );
	glShaderSource( fragShaderID, 1, (const char **)&tmpString, 0 );
	free( tmpString );

	// Copile the shader, and check if there were compilation errors
	GLint compiled = 0;
	glCompileShader( fragShaderID );
	glGetShaderiv( fragShaderID, GL_COMPILE_STATUS, &compiled);

	// If there were compile errors, we might want to print them out.
	if (!compiled)
	{
		PrintCompilerError( fragShaderID, fragShaderFile );
		return false;
	}

	//free( tmpString );

	glAttachShader( programID, fragShaderID );
	return true;
}

// Reloads shaders from files, compiles them all, and links the result
bool GLSLProgram::ReloadShaders( void )
{
	GLint compiled=0, linked=0;

	// Make sure this isn't currently bound!
	if (enabled) DisableShader();

	// Reload all shaders (that have non-zero shaders attached and we are responsible for (non-zero filename))
	char *tmpString;
	if (vertShaderID > 0 && vertShaderFile)
	{
		tmpString = ReturnFileAsString( vertShaderFile );
		if (!tmpString && verbose) fprintf(stderr, "***Error: Unable to load vertex shader '%s'!\n", vertShaderFile );
		if (!tmpString) return false;
		glShaderSource( vertShaderID, 1, (const char **)&tmpString, 0 );
		free( tmpString );
		glCompileShader( vertShaderID );
		glGetShaderiv( vertShaderID, GL_COMPILE_STATUS, &compiled);
		if (!compiled)
		{
			PrintCompilerError( vertShaderID, vertShaderFile );
			return false;
		}
	}
	if (geomShaderID > 0 && geomShaderFile)
	{
		tmpString = ReturnFileAsString( geomShaderFile );
		if (!tmpString && verbose) fprintf(stderr, "***Error: Unable to load geometry shader '%s'!\n", geomShaderFile );
		if (!tmpString) return false;
		glShaderSource( geomShaderID, 1, (const char **)&tmpString, 0 );
		free( tmpString );
		glCompileShader( geomShaderID );
		glGetShaderiv( geomShaderID, GL_COMPILE_STATUS, &compiled);
		if (!compiled)
		{
			PrintCompilerError( geomShaderID, geomShaderFile );
			return false;
		}
	}
	if (fragShaderID > 0 && fragShaderFile)
	{
		tmpString = ReturnFileAsString( fragShaderFile );
		if (!tmpString && verbose) fprintf(stderr, "***Error: Unable to load fragment shader '%s'!\n", fragShaderFile );
		if (!tmpString) return false;
		glShaderSource( fragShaderID, 1, (const char **)&tmpString, 0 );
		free( tmpString );
		glCompileShader( fragShaderID );
		glGetShaderiv( fragShaderID, GL_COMPILE_STATUS, &compiled);
		if (!compiled)
		{
			PrintCompilerError( fragShaderID, fragShaderFile );
			return false;
		}
	}

	glLinkProgram( programID );
	glGetProgramiv( programID, GL_LINK_STATUS, &linked);
	if (!linked)
	{
		PrintLinkerError();
		return ( isLinked = false );
	}

	ResetAutomaticBindings();
	return ( isLinked = true );
}

// Relinks the program and checks for any errors.
bool GLSLProgram::LinkProgram( void )
{
	GLint linked=0;
	glLinkProgram( programID );
	glGetProgramiv( programID, GL_LINK_STATUS, &linked);
	if (!linked)
	{
		PrintLinkerError();
		return ( isLinked = false );
	}
	return ( isLinked = true );
}


// Return the OpenGL shader id (e.g., from glCreateShader()) for one of the three shader types
GLuint GLSLProgram::GetShaderID( GLenum type )
{
	switch( type )
	{
	case GL_VERTEX_SHADER:
		return vertShaderID;
	case GL_GEOMETRY_SHADER_EXT:
		return geomShaderID;
	case GL_FRAGMENT_SHADER:
		return fragShaderID;
	default:
		fprintf( stderr, "***Fatal Error: Unknown type passed to GLSLProgram::GetShaderID! (0x%X)\n", type);
		exit(0);
	}
	return 0;
}


// Open a shader, read its contents into a string in memory.
char *GLSLProgram::ReturnFileAsString( char *filename )
{
	/* open the file */
	FILE *file;
	
	if (shaderSearchPath)
		file = shaderSearchPath->OpenFileInPath( filename, "rb" );
	else
		file = fopen( filename, "rb" );

	if (!file) return 0;	

	/* get the length of the file */
	fseek( file, 0, SEEK_END );
	long size = ftell( file );
	rewind( file );

	/* allocate memory for the shader string */
	char *shaderMemory = (char *)calloc( (size+1), sizeof(char) );
	if (!shaderMemory) { fclose(file); return 0; }
	memset( shaderMemory, 0, (size+1) );

	/* read shader data from the file */
	fread (shaderMemory,1,size,file);
	shaderMemory[size] = 0;

	/* clean up and return the data */
	fclose(file);

	return shaderMemory;
}


// For testing (or eliminating all warnings)
void GLSLProgram::OutputErrorLogs( void )
{
	GLsizei theLen;
	char buf[4096];
	if ( vertShaderID > 0 )
	{
		glGetShaderInfoLog( vertShaderID, 4095, &theLen, buf );
		printf("Vertex Log: %s\n", buf );
	}
	if ( geomShaderID > 0 )
	{
		glGetShaderInfoLog( geomShaderID, 4095, &theLen, buf );
		printf("Geometry Log: %s\n", buf );
	}
	if ( fragShaderID > 0 )
	{
		glGetShaderInfoLog( fragShaderID, 4095, &theLen, buf );
		printf("Geometry Log: %s\n", buf );
	}
	glGetProgramInfoLog( programID, 4095, &theLen, buf );
	printf("Linker Log: %s\n", buf );
}

// Grabs the shader log (e.g., compiler errors) for a specified shader. 
//     (The filename is just there to be pretty)
void GLSLProgram::PrintCompilerError( GLuint shaderID, char *filename )
{
	if (!verbose) return;
	char buf[4096];
	GLsizei theLen;
	fprintf(stderr, "(((GLSL Compile Error!)))....When compiling '%s'....\n", filename );
	fprintf(stderr, "-----------------------------------------------------------------\n");
	glGetShaderInfoLog( shaderID, 4095, &theLen, buf );
	fprintf(stderr, "%s\n", buf);
}

// Grabs the program log (e.g., linker errors).  It prints name of the files
//     combined to link the program.  NOTE:  If another class manages the shaders
//     (e.g., this one does not load shaders with a filename), the name printed will
//     be the relatively useless "<Shader ID #xx>"
void GLSLProgram::PrintLinkerError( void )
{
	if (!verbose) return;
	char buf[4096], shaderNum[50];
	GLsizei theLen;
	fprintf(stderr, "(((GLSL Link Error!)))....When linking....\n" );
	sprintf( shaderNum, "<Shader ID #%d>", vertShaderID );
	fprintf(stderr, "                  Vertex: '%s',\n", 
		vertShaderID == 0 ? "<FIXED-FUNCTION>" : ( vertShaderFile ? vertShaderFile : shaderNum ) );
	fprintf(stderr, "                Geometry: '%s',\n", 
		geomShaderID == 0 ? "<FIXED-FUNCTION>" : ( geomShaderFile ? geomShaderFile : shaderNum ) );
	fprintf(stderr, "            and Fragment: '%s'\n", 
		fragShaderID == 0 ? "<FIXED-FUNCTION>" : ( fragShaderFile ? fragShaderFile : shaderNum ) );
	fprintf(stderr, "-----------------------------------------------------------------\n");
	glGetProgramInfoLog( programID, 4095, &theLen, buf );
	fprintf(stderr, "%s\n", buf);
}


void GLSLProgram::GeometryShaderSettings( GLenum inputType, int maxEmittedVerts, GLenum outputType )
{
	if (!geomShaderID) return;

	glProgramParameteriEXT( programID, GL_GEOMETRY_INPUT_TYPE_EXT, inputType );
	glProgramParameteriEXT( programID, GL_GEOMETRY_OUTPUT_TYPE_EXT, outputType );
	glProgramParameteriEXT( programID, GL_GEOMETRY_VERTICES_OUT_EXT, maxEmittedVerts );

	isLinked = false;

	if (verbose)
	{
		if (maxEmittedVerts == 0)
			fprintf(stderr, "***Error: GeometryShaderSettings() called with zero 'maxEmittedVerts'!\n");
		if ((inputType != GL_POINTS) && (inputType != GL_TRIANGLES) && (inputType != GL_LINES) &&
			(inputType != GL_LINES_ADJACENCY_EXT) && (inputType != GL_TRIANGLES_ADJACENCY_EXT))
			fprintf(stderr, "***Error: GeometryShaderSettings() called with invalid 'inputType'!\n");
		if ((outputType != GL_POINTS) && (outputType != GL_LINE_STRIP) && (outputType != GL_TRIANGLE_STRIP))
			fprintf(stderr, "***Error: GeometryShaderSettings() called with invalid 'outputType'!\n");
	}

}


void GLSLProgram::BindAndEnableTexture( char *shaderTextureName, GLint textureID, GLenum location, GLenum type )
{
	GLint loc = glGetUniformLocation( programID, shaderTextureName );
	if (loc != -1) glUniform1i( loc, location-GL_TEXTURE0 );
	glActiveTexture( location );
	glBindTexture( type, textureID );
	glEnable( type );
}


void GLSLProgram::DisableTexture( GLenum location, GLenum type )
{
	glActiveTexture( location );
	glDisable( type );
	glBindTexture( type, 0 );
}


bool GLSLProgram::EnableShader( void )
{
	// Enable the relevant program.
	glUseProgram( programID );

	// If there's uniforms we want to bind when we enable the program, loop thru them.
	for (unsigned int i=0; i < autoBindUniforms.Size(); i++)
	{
		// Check that this uniform corresponds to a variable...
		if ( (autoBindUniforms[i]->uniformLocation < 0) ) continue;

		// It does.  Check what type
		if ( (autoBindUniforms[i]->bindingType >= BIND_FLOAT) &&  // It's 1, 2, 3, or 4 vector to pass,
			 (autoBindUniforms[i]->bindingType <= BIND_MAT4) )    // or it's a matrix type.
			SetUniform( autoBindUniforms[i]->uniformLocation, 
						autoBindUniforms[i]->bindingType, 
						autoBindUniforms[i]->boundC_variable );
		else if (autoBindUniforms[i]->bindingType == BIND_TEX2D_PTR) //It's a 2D texture, but we have a pointer
		{                                                            //   to the texID, not the texID itself
			glUniform1i( autoBindUniforms[i]->uniformLocation, autoBindUniforms[i]->textureUnit-GL_TEXTURE0 );
			glActiveTexture( autoBindUniforms[i]->textureUnit );
			// The following line is a sneaky cast.  Your compiler may complain, but it should be OK unless
			//    you compile this on a machine where:  sizeof(GLuint) != sizeof(GLuint *)
			glBindTexture( GL_TEXTURE_2D, *((GLuint *)autoBindUniforms[i]->textureID) );
			glEnable( GL_TEXTURE_2D );
		}
		else // It's a texture of some sort to bind, and we have the constant GL texture identifier already
		{
			glUniform1i( autoBindUniforms[i]->uniformLocation, autoBindUniforms[i]->textureUnit-GL_TEXTURE0 );
			glActiveTexture( autoBindUniforms[i]->textureUnit );
			glBindTexture( autoBindUniforms[i]->bindingType, autoBindUniforms[i]->textureID );
			glEnable( autoBindUniforms[i]->bindingType );
		}

	}

	// Check if we want to change OpenGL state (enable/disable) with the shader.
	if (shaderEnableFlags || shaderDisableFlags) SetProgramSpecificEnablesAndDisables();

	// Ok.  We're done.
	return( enabled = true );
}

bool GLSLProgram::DisableShader( void )
{
	// Disable the program
	glUseProgram( 0 );

	// If we changed the state with EnableShader(), pop the state off.
	if (shaderEnableFlags || shaderDisableFlags) glPopAttrib();

	// If we enabled any textures in EnableShader(), disable them.
	for (unsigned int i=0; i < autoBindUniforms.Size(); i++)
		if ( (autoBindUniforms[i]->bindingType > BIND_MAX) )  // Then we need to disable a texture
		{
			glActiveTexture( autoBindUniforms[i]->textureUnit );
			glBindTexture( autoBindUniforms[i]->bindingType, 0 );
			glDisable( autoBindUniforms[i]->bindingType );
		}

	// OK, we're done.
	enabled = false;
	return true;
}


int GLSLProgram::SetParameter( char *paramName, float x )
{
	GLint location = glGetUniformLocation( programID, paramName );
	if (location != -1) glUniform1f( location, x );
	return location;
}

int GLSLProgram::SetParameter( char *paramName, float x, float y )
{
	GLint location = glGetUniformLocation( programID, paramName );
	if (location != -1) glUniform2f( location, x, y );
	return location;
}

int GLSLProgram::SetParameter( char *paramName, float x, float y, float z )
{
	GLint location = glGetUniformLocation( programID, paramName );
	if (location != -1) glUniform3f( location, x, y, z );
	return location;
}

int GLSLProgram::SetParameter( char *paramName, float x, float y, float z, float w )
{
	GLint location = glGetUniformLocation( programID, paramName );
	if (location != -1) glUniform4f( location, x, y, z, w );
	return location;
}

void GLSLProgram::SetUniform( GLint location, int arraySize, float *array )
{
	switch( arraySize )
	{
	case 4:
		glUniform4fv( location, 1, array );
		break;
	case 3:
		glUniform3fv( location, 1, array );
		break;
	case 2:
		glUniform2fv( location, 1, array );
		break;
	case 1:
		glUniform1fv( location, 1, array );
		break;
	case BIND_MAT4:
		glUniformMatrix4fv( location, 1, false, array );
		break;
	}
}

int GLSLProgram::SetParameterv( char *paramName, int arraySize, float *array )
{
	GLint location = glGetUniformLocation( programID, paramName );
	if (location == -1) return -1;
	if (arraySize == 1)
		glUniform1fv( location, 1, array );
	else if (arraySize == 2)
		glUniform2fv( location, 1, array );
	else if (arraySize == 3)
		glUniform3fv( location, 1, array );
	else glUniform4fv( location, 1, array );
	return location;
}

int GLSLProgram::Set4x4MatrixParameterv( char *paramName, float *array )
{
	GLint location = glGetUniformLocation( programID, paramName );
	if (location == -1) return -1;
	glUniformMatrix4fv( location, 1, false, array );
	return location;
}

void GLSLProgram::SetTextureBinding( char *shaderTextureName, GLenum location )
{
	GLint loc = glGetUniformLocation( programID, shaderTextureName );
	if (loc != -1) glUniform1i( loc, location-GL_TEXTURE0 );
}


void GLSLProgram::ResetAutomaticBindings( void )
{
	for (unsigned int i=0; i < autoBindUniforms.Size(); i++)
	{
		if (!autoBindUniforms[i]->shaderVarName) continue;
		autoBindUniforms[i]->uniformLocation = glGetUniformLocation( programID, 
													autoBindUniforms[i]->shaderVarName );
	}
}


void GLSLProgram::SetupAutomaticBinding( char *paramName, int arraySize, float *array )
{
	unsigned int idx = autoBindUniforms.Add( new GLSLBindings() );
	autoBindUniforms[idx]->uniformLocation = glGetUniformLocation( programID, paramName );
	autoBindUniforms[idx]->bindingType = arraySize;
	autoBindUniforms[idx]->boundC_variable = array;
	autoBindUniforms[idx]->shaderVarName = strdup( paramName );
}

void GLSLProgram::SetupAutomatic4x4MatrixBinding( char *paramName, float *array )
{
	unsigned int idx = autoBindUniforms.Add( new GLSLBindings() );
	autoBindUniforms[idx]->uniformLocation = glGetUniformLocation( programID, paramName );
	autoBindUniforms[idx]->bindingType = BIND_MAT4;
	autoBindUniforms[idx]->boundC_variable = array;
	autoBindUniforms[idx]->shaderVarName = strdup( paramName );
}

void GLSLProgram::SetupAutomaticTextureBinding( char *shaderTextureName, GLuint textureID, GLenum location, GLenum type )
{
	unsigned int idx = autoBindUniforms.Add( new GLSLBindings() );
	autoBindUniforms[idx]->uniformLocation = glGetUniformLocation( programID, shaderTextureName );
	autoBindUniforms[idx]->textureUnit = location;
	autoBindUniforms[idx]->bindingType = type;
	autoBindUniforms[idx]->textureID   = textureID;
	autoBindUniforms[idx]->shaderVarName = strdup( shaderTextureName );
	glUniform1i( autoBindUniforms[idx]->uniformLocation, autoBindUniforms[idx]->textureUnit-GL_TEXTURE0 );
}

void GLSLProgram::SetupAutomaticTexture_Experimental( char *shaderTextureName, GLuint *textureID, GLenum location )
{
	unsigned int idx = autoBindUniforms.Add( new GLSLBindings() );
	autoBindUniforms[idx]->uniformLocation = glGetUniformLocation( programID, shaderTextureName );
	autoBindUniforms[idx]->textureUnit = location;
	autoBindUniforms[idx]->bindingType = BIND_TEX2D_PTR;
	// The following line is a sneaky cast.  Your compiler may complain, but it should be OK unless
	//    you compile this on a machine where:  sizeof(GLuint) != sizeof(GLuint *)
	autoBindUniforms[idx]->textureID   = (GLuint)textureID;
	autoBindUniforms[idx]->shaderVarName = strdup( shaderTextureName );
	glUniform1i( autoBindUniforms[idx]->uniformLocation, autoBindUniforms[idx]->textureUnit-GL_TEXTURE0 );
}

void GLSLProgram::SetProgramSpecificEnablesAndDisables( void )
{
	glPushAttrib( GL_ENABLE_BIT );
	if ( shaderEnableFlags & GLSL_BLEND )				glEnable( GL_BLEND );
	if ( shaderEnableFlags & GLSL_DEPTH_TEST )			glEnable( GL_DEPTH_TEST );
	if ( shaderEnableFlags & GLSL_STENCIL_TEST )		glEnable( GL_STENCIL_TEST );
	if ( shaderEnableFlags & GLSL_ALPHA_TEST )			glEnable( GL_ALPHA_TEST );
	if ( shaderEnableFlags & GLSL_CULL_FACE )			glEnable( GL_CULL_FACE );
	if ( shaderEnableFlags & GLSL_LIGHTING )			glEnable( GL_LIGHTING );
	if ( shaderEnableFlags & GLSL_VARY_POINT_SIZE )		glEnable( GL_VERTEX_PROGRAM_POINT_SIZE );	
	if ( shaderDisableFlags & GLSL_BLEND )				glDisable( GL_BLEND );
	if ( shaderDisableFlags & GLSL_DEPTH_TEST )			glDisable( GL_DEPTH_TEST );
	if ( shaderDisableFlags & GLSL_STENCIL_TEST )		glDisable( GL_STENCIL_TEST );
	if ( shaderDisableFlags & GLSL_ALPHA_TEST )			glDisable( GL_ALPHA_TEST );
	if ( shaderDisableFlags & GLSL_CULL_FACE )			glDisable( GL_CULL_FACE );
	if ( shaderDisableFlags & GLSL_LIGHTING )			glDisable( GL_LIGHTING );
	if ( shaderDisableFlags & GLSL_VARY_POINT_SIZE )	glDisable( GL_VERTEX_PROGRAM_POINT_SIZE );	
}

