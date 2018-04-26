/************************************************
** searchPathList.h                            **
** -------------------                         **
**                                             **
** This is a very basic class that stores      **
**    character strings to directory paths     **
**    for files you might want to search for.  **
**    When you decide to open a file, you can  **
**    simply call OpenFileInPath( name, "r" )  **
**    instead of fopen( name, "r" ), and it    **
**    opens the first file with that name      **
**    encountered in the search path.  The     **
**    path list includes "." (the current dir) **
**    by default.                              **
**                                             **
** Chris Wyman (9/29/2006)                     **
************************************************/


#ifndef __SearchPathList_h__
#define __SearchPathList_h__

#include <stdio.h>
#include <stdlib.h>

class PathList {

private:
	int numPaths;
    char **searchPaths;
    int maxNumPaths;

	void ResizePathList( void );

public:

	PathList( void ) { numPaths = 0; searchPaths = 0; maxNumPaths = 0; }
	PathList( char *path );
	~PathList(); 

	inline int GetNumPaths( void ) { return numPaths; }
	inline char *GetPath( int i ) { return ( i >=0 && i<numPaths ? searchPaths[i] : 0 ); }

	// Adds a path to search to the list.  This function calls strdup() to copy the path.
	void AddPath( char *path );

	// Searches the current directory, followed by every directory in the path (in the 
	//    order specified by AddPath()).  The first existant file "filename" is returned.
	//    If no such file exists, it returns NULL (just like fopen()).
	FILE *OpenFileInPath( char *filename, char *mode );

	// Searches the current directory, followed by every directory in the path, and
	//    returns a character array containing the full pathname of the first file
	//    discovered.  If no such file exists, it returns NULL!  The return values
	//    is created using strdup(), so it must be free()'d.
	char *GetCompletePath( char *filename );

}; 

#ifndef MAX
#define MAX(x,y)  ((x)>(y)?(x):(y))
#endif

#endif