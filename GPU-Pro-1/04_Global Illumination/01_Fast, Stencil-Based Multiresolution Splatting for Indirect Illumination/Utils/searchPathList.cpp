
#include "searchPathList.h"
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#pragma warning( disable: 4996 )

PathList::~PathList()
{
	for (int i=0; i<numPaths; i++)
		free( searchPaths[i] );
	free ( searchPaths );
}

PathList::PathList( char *path ) :
  numPaths(0), searchPaths(0), maxNumPaths(0)
{
	AddPath( path );
}

char *PathList::GetCompletePath( char *filename )
{
	FILE*     file;
	int i;
	char buf[1024];

	/* open the file */
	file = fopen( filename, "r" );
	sprintf(buf,"%s",filename);

	/* if we couldn't find it, search the alternate paths */
	if (!file){
		for (i=0;i<GetNumPaths(); i++)
		{
			sprintf( buf, "%s%s", GetPath(i), filename );
			file = fopen( buf, "r" );
			if (file) break;
		}
	}

	if (!file) return NULL;
	fclose( file );
	return strdup( buf );
}

FILE *PathList::OpenFileInPath( char *filename, char *mode )
{
	FILE*     file;
	int i;
	char buf[1024];

	/* open the file */
	file = fopen( filename, mode );

	/* if we couldn't find it, search the alternate paths */
	if (!file) 
		for (i=0;i<GetNumPaths(); i++)
		{
			sprintf( buf, "%s%s", GetPath(i), filename );
			file = fopen( buf, mode );
			if (file) break;
		}

	return file;
}

void PathList::ResizePathList( void )
{
	unsigned int newSize = MAX( 2*maxNumPaths, 4 );
	char **tmp = (char **)malloc( newSize * sizeof( char * ) );
	assert( tmp );

	// Copy the existing data
	for (int i=0; i<numPaths; i++)
		tmp[i] = searchPaths[i];

	// Zero out new data
	for (unsigned int i=numPaths; i < newSize; i++)
		tmp[i] = 0;

	// Free the old data and copy the pointers over
	free( searchPaths );
	searchPaths = tmp;
	maxNumPaths = newSize;
}

void PathList::AddPath( char *path )
{
	if (numPaths >= maxNumPaths)
		ResizePathList();

	searchPaths[numPaths++] = strdup( path );
}