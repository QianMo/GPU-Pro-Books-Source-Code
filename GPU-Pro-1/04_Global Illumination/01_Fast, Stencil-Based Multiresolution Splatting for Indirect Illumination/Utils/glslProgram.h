/**************************************************************************
** glslProgram.h                                                         **
** ------------                                                          **
**                                                                       **
** This header includes a class for loading and initializing a _single_  **
**   GLSL program (which consists of a set of vertex, geometry, and/or   **
**   fragment shaders linked into a "program")                           **
**                                                                       **
** Please note that this is not the most efficient implementation.  One  **
**   is likely to reuse the same vertex shader in multiple programs, for **
**   example, and using this class requires duplicating the state for    **
**   that shader for each program.                                       **
**                                                                       **
** However, this use of a "program class" simplifies state setup and     **
**   facilitates error checking for program-specific state that was hard **
**   to check for in an earlier class that contained *all* the needed    **
**   GLSL shaders.  I also think this will be straightforward to extend  **
**   should additional programmable stages be introduced in the future.  **
**                                                                       **
** To make this more efficient, a wrapper class (like my earlier         **
**   'glslShaders' class) could be writted to eliminate duplicated state **
**   and oversee necessary correspondances between shaders.              **
**                                                                       **
** Chris Wyman (12/4/2007)                                               **
**************************************************************************/

#ifndef __GLSL_PROGRAM_H
#define __GLSL_PROGRAM_H

// This file contains a simple class for a path list that implements a 
//    fopen() replacement which searches for files in multiple directories.
//    This is not needed.  In fact a NULL pathList passed to the constructor
//    uses a standard fopen(), so this dependance should be easy to remove.
#include "searchPathList.h"

// Other needed headers
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "Utils/GLee.h"
#include <GL/glut.h>

// The following two lines are used for some of the advanced functionality.
//    If you don't use the advanced functionality, you can remove all 
//    references to both from the class and .cpp file
#include "DataTypes/Array1D.h"
class GLSLBindings;

// Begin the definition of the GLSLProgram class
class GLSLProgram
{
public:
	// Setup a program.  There are two choices.  Pass shader filenames to the 
	//     constructor or do it after the fact with the Set*Shader() methods.
	// "verboseErrors" determines if compiler errors appear on the console
	// "path" specifies a list of directories to search for shader files.
	GLSLProgram( bool verboseErrors=false, PathList *path = 0 );
	GLSLProgram( char *vShader, char *gShader, char *fShader, 
		         bool verboseErrors=false, PathList *path = 0, bool linkImmediately = true );
	~GLSLProgram();

	// Assocate a shader from a particuar file (Error => returns 'false')
	bool SetVertexShader  ( char *shaderFilename );
	bool SetGeometryShader( char *shaderFilename );
	bool SetFragmentShader( char *shaderFilename );

	// Enable or disable this program (Error => returns 'false')
	bool EnableShader( void );
	bool DisableShader( void );
	inline bool IsEnabled( void ) const { return enabled; }

	// Reloads, compiles, and links program shaders (Error => returns 'false')
	bool ReloadShaders( void );

	// Some changes below do not take effect until the program is relinked
	bool LinkProgram( void );
	inline bool IsLinked( void ) { return isLinked; } 

	// Set geometry shader params.  Requires relinking (call LinkProgram())
	void GeometryShaderSettings( GLenum inputType, 
		                         int maxEmittedVerts, 
								 GLenum outputType );

	// Set uniform shader parameters.  The number of parameters (or array size)
	//    should match the size of the variable in the GLSL shader.
	int SetParameter( char *paramName, float x );
	int SetParameter( char *paramName, float x, float y );
	int SetParameter( char *paramName, float x, float y, float z );
	int SetParameter( char *paramName, float x, float y, float z, float w );
	int SetParameterv( char *paramName, int arraySize, float *array );

	// Set uniform shader matrix parameter.  This should be possible for non 
	//    4x4 matrices, but it was not clear to me how to do this efficiently
	//    in a single method.  Only the need for 4x4 matrices has arisen, so
	//    that's the only one implemented.  Array should be in OpenGL order.
	int Set4x4MatrixParameterv( char *paramName, float *array );

	// These are shortcuts for utilizing textures with this shader.  To use a
	//     texture in GLSL, you need four steps:  1) Bind name to a texture 
	//     unit, 2) glActiveTexture(texUnit), 3) glBindTexture(texType,texID),
	//     4) glEnable(texType). 
	// BindAndEnableTexture() does all of these, but isn't very efficient.
	// DisableTexture() does cleanup necessary after BindAndEnableTexture().
	// SetTextureBinding() does just step #1, forcing you to do #2-#4.
	void BindAndEnableTexture( char *shaderTextureName, 
		                       GLint textureID, 
		                       GLenum location = GL_TEXTURE0, 
							   GLenum type = GL_TEXTURE_2D );
	void DisableTexture( GLenum location = GL_TEXTURE0, 
		                 GLenum type = GL_TEXTURE_2D );
	void SetTextureBinding( char *shaderTextureName, 
		                    GLenum location = GL_TEXTURE0 );

	/***********************************************************************/
	/*                           ADVANCED STUFF                            */
	/***********************************************************************/

	// Sets up an automatic binding so that *every* time EnableShader() is 
	//    called: SetParameterv() is called, Set4x4MatrixParameterv() is 
	//    called, and/or a texture is bound with BindAndEnableTexture(), 
	//    making the calling of shaders much cleaner in your code.  The data
	//    is re-read from the variables each time, so the values can be 
	//    updated between shaders instantiations.
	// PLEASE NOTE:  Pointers should not point to values on the stack!
	void SetupAutomaticBinding( char *paramName, int arraySize, float *array);
	void SetupAutomatic4x4MatrixBinding( char *paramName, float *array );
	void SetupAutomaticTextureBinding( char *shaderTextureName, 
		                               GLuint textureID, 
									   GLenum location = GL_TEXTURE0, 
									   GLenum type = GL_TEXTURE_2D );
	void SetupAutomaticTexture_Experimental( char *shaderTextureName, 
		                                     GLuint *textureID, 
											 GLenum location = GL_TEXTURE0 );

	// Sometimes it's easier to tell the shader what state it needs... and 
	//    let EnableShader() and DisableShader() handle the enabling.  Before
	//    changing state PushAttrib() is used, and PopAttrib() is called in 
	//    DisableShader().  Thus these are non-destructive to current state.
	inline void SetProgramEnables( unsigned int progEnableFlags = 0 )  
	            { shaderEnableFlags = progEnableFlags; }
	inline void SetProgramDisables( unsigned int progDisableFlags = 0 )  
	            { shaderDisableFlags = progDisableFlags; }

	// Read out OpenGL program ID (i.e., from glCreateProgram() or 
	//    glCreateShader()).  For the shader IDs, pass in: GL_VERTEX_SHADER,
	//    GL_GEOMETRY_SHADER_EXT, or GL_FRAGMENT_SHADER
	inline GLuint GetProgramID( void ) { return programID; } 
	GLuint GetShaderID( GLenum type );  

	// Changes if verbose error messages are on/off  
	inline void VerboseOff() { verbose = false; }
	inline void VerboseOn()  { verbose = true; }

	// Get shader filenames.  
	inline char *GetVertexShader  ( void )   { return vertShaderFile; }
	inline char *GetGeometryShader( void )   { return geomShaderFile; }
	inline char *GetFragmentShader( void )   { return fragShaderFile; }

	// Output all the error logs for this shader program
	void OutputErrorLogs( void );

	inline Array1D< GLSLBindings * > *GetAutomaticShaderBindings( void ) { return &autoBindUniforms; }

/***************************************************************************/
/*                            PRIVATE STUFF                                */
/* -- should not need to modify, unless you're removing dependencies on -- */ 
/* -- PathList, Array1D, or GLSLBindings, or implementing more advanced -- */
/* -- functionality.                                                    -- */
/***************************************************************************/
private:
	PathList *shaderSearchPath;  // Where to search for shader files
	bool verbose;                // Are error messages output?

	GLuint programID;            // OpenGL ID for this program
	GLuint vertShaderID;         // OpenGL ID for associated vertex shader
	GLuint geomShaderID;         // OpenGL ID for associated geom shader
	GLuint fragShaderID;         // OpenGL ID for associated frag shader

	bool enabled;                // Program Enabled() and not Disabled()?
	bool isLinked;               // Does the program need relinking?

	// Information needed for reloading...  These store the unqualified 
	//    filenames.  If these are NULL, either we're using fixed function 
	//    pipeline OR someone else is responsible for reloading the file 
	//    bound to the correct shaderID.  If these are non-NULL, we will 
	//    reload these files in ReloadShaders().
	char *vertShaderFile, *geomShaderFile, *fragShaderFile;  

	// Parameters for geometry shader input and output.  
	int geomVerticesOut, geomInputType, geomOutputType;

	// Variables automatically bound upon calling EnableShader()
	Array1D< GLSLBindings * > autoBindUniforms;

	// Special functionality that absolutely must be enabled or disabled?
	unsigned int shaderEnableFlags, shaderDisableFlags;

	// Private utility functions
	char *ReturnFileAsString( char *filename );     // Searches the path
	void PrintCompilerError( GLuint shaderID, char *filename );
	void PrintLinkerError( void );
public:
	void SetUniform( GLint location, int arraySize, float *array );
private:
	void SetProgramSpecificEnablesAndDisables( void );

	// This needs to be called in ReloadShaders() to make sure the bindings
	//    are set to the appropriate shader variables.
	void ResetAutomaticBindings( void );
};




// Enable/disable flags (stored in shaderEnableFlags and shaderDisableFlags).
//    This is state that is modified when EnableShader() is called. 
// NOTE:  These can be OR'd together, so if you add more, make sure to
//    maintain this ability!  If you add more, you need to change the code
//    in SetProgramSpecificEnablesAndDisables().
#define GLSL_NO_SPECIAL_STATE	0x00000000
#define GLSL_BLEND				0x00000001
#define GLSL_DEPTH_TEST			0x00000002
#define GLSL_STENCIL_TEST		0x00000004
#define GLSL_ALPHA_TEST			0x00000008
#define GLSL_CULL_FACE			0x00000010
#define GLSL_LIGHTING			0x00000020
#define GLSL_VARY_POINT_SIZE    0x00000040

// A simple class to store bindings inside the shader
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


#endif