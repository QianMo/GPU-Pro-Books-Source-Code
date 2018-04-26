/******************************************************************/
/* ProgramPathLists.h                                             */
/* -----------------------                                        */
/*                                                                */
/* This class defines a wrapper for storing directories where     */
/*    various file types may be stored.  For the most robust I/O  */
/*    routines, use the utilities provided by this class to open  */
/*    existing files, as they will search all the valid locations */
/*    before reporting failure.                                   */
/*                                                                */
/* Chris Wyman (03/30/2007)                                       */
/******************************************************************/


#ifndef PROGRAM_PATH_LISTS_H
#define PROGRAM_PATH_LISTS_H

#include <stdio.h>
#include <stdlib.h>
#include "searchPathList.h"

class ProgramSearchPaths
{
	PathList modelPath;       /* list of directories to search for models   */
	PathList texturePath;     /* list of directories to search for textures */
	PathList shaderPath;      /* list of directories to search for shaders  */
	PathList scenePath;       /* list of directories to search for scenes   */

public:
	ProgramSearchPaths() {}

	/* All of the input character strings should be ended with a '/' ! */
	inline void AddShaderPath ( char *path )  { shaderPath.AddPath( path );  }
	inline void AddTexturePath( char *path )  { texturePath.AddPath( path ); }
	inline void AddModelPath  ( char *path )  { modelPath.AddPath( path );   }
	inline void AddScenePath  ( char *path )  { scenePath.AddPath( path );   }

	/* Functions to open various file types.  The paths are checked in the order */
	/*    they were Add()'ed above.  If a file with the name 'filename' isn't    */
	/*    in any of the specified directories, a NULL pointer is returned.       */
	inline FILE *OpenShader ( char *filename, char *mode )  { return shaderPath.OpenFileInPath( filename, mode ); }
	inline FILE *OpenTexture( char *filename, char *mode )  { return texturePath.OpenFileInPath( filename, mode ); }
	inline FILE *OpenModel  ( char *filename, char *mode )  { return modelPath.OpenFileInPath( filename, mode ); }
	inline FILE *OpenScene  ( char *filename, char *mode )  { return scenePath.OpenFileInPath( filename, mode ); }

	/* Tries to open the file specified in all the paths until it finds a matching */
	/*    file.  It then returns the string "path/filename" of the match (which,   */
	/*    of course, could be used as direct input to fopen() to open the file)    */
	/* Note:  These pointers are to malloc()'d memory, so they must be free()'d!!  */
	inline char *GetShaderPath ( char *filename )  { return shaderPath.GetCompletePath( filename ); }
	inline char *GetTexturePath( char *filename )  { return texturePath.GetCompletePath( filename ); }
	inline char *GetModelPath  ( char *filename )  { return modelPath.GetCompletePath( filename ); }
	inline char *GetScenePath  ( char *filename )  { return scenePath.GetCompletePath( filename ); }


	// Advanced use.  Beware using this.
	PathList *GetShaderPathList( void )			   { return &shaderPath; }
};


#endif

