#ifndef __SIMPLEMODELLIB_H
#define __SIMPLEMODELLIB_H

#include "Utils/GLee.h"
#include <GL/glut.h>

// Definitions that can be passed to the HalfEdgeModel constructor
//    Note:  The .hem file is the native time for this representation.
//    Other types may load *very* slowly, given that this code has not
//    been optimized for creation of half-edge models.  It is recommended
//    to use/write a conversion program as a preprocess and always load
//    .hem files with this class in actual applications.
#define TYPE_HEM_FILE    0
#define TYPE_OBJ_FILE    1
#define TYPE_M_FILE      2  // Not currently implemented
#define TYPE_SMF_FILE    3

// Parameters you can give to CreateOpenGLDisplayList().  These can be
//    or'd together.  Note WITH_NORMALS and WITH_FACET_NORMALS are mutually
//    exclusive (WITH_NORMAL overrides).  WITH_VERTICES is always used. 
#define WITH_VERTICES               0x00
#define WITH_NORMALS                0x01
#define WITH_FACET_NORMALS          0x02

// Parameters passed to CreateOpenGLDisplayList() specifying the primitive
//    type used for rendering.  These are all mutually exclusive, and if
//    none is specified, TRIANGLES are used.
#define USE_TRIANGLES               0x10
#define USE_TRIANGLE_ADJACENCY      0x20
#define USE_LINES                   0x40
#define USE_POINTS					0x80

// When used as a parameter to CreateOpenGLDisplayList() with USE_LINES, it 
//     puts the facet norms of adjacent faces into the texture coordinates 
//     for units GL_TEXCOORD6 and GL_TEXCOORD7! 
#define WITH_ADJACENT_FACE_NORMS    0x100

class HalfEdgeModel
{
public:
	// Create the object, load the model from file.  Note loading a non-HEM file
	//     potentially requires a conversion, which can sometimes be (quite) slow.
	HalfEdgeModel( char *filename, int fileType=TYPE_HEM_FILE );
	~HalfEdgeModel();

	// Frees model memory associated with the half edge model.  OpenGL display
	//    lists and/or vertex buffer objects are left intact
	void FreeNonGLMemory( void );

	// Check if this is valid.  It might be invalid if the load failed.
	bool IsValid( void ) { return solid != 0; }

	// Output the internally stored model in my custom ".hem" model format
	bool SaveAsHEM( char *outputFilename );

	// Creates a display list that can be executed using CallList()
	//    It returns 0 on an error (e.g., model corrupted), and returns
	//    the OpenGL list ID if you'd rather call it directly.
	GLuint CreateOpenGLDisplayList( unsigned int flags = WITH_VERTICES|USE_TRIANGLES, bool deleteOldList=true );

	// Creates a display list that can be executed using CallList()
	//    It returns 0 on an error (e.g., model corrupted), and returns
	//    the OpenGL VBO ID if you'd rather call it directly.
	GLuint CreateOpenGLVBO( unsigned int flags = WITH_VERTICES|USE_TRIANGLES );

	// Executes the OpenGL display list created with parameters from the last call
	//    to CreateOpenGLDisplayList().  False is returned if a list has not been
	//    created, or creation failed due to model corruption (should be rare).
	bool CallList( int type=USE_TRIANGLES );

	// Draws geometry using a previously created VBO
	bool CallVBO( int type=USE_TRIANGLES );


private:
	// Note:  Private data are all represented as void pointers, in order to avoid
	//        polluting the namespace with the varied data types introduced by the
	//        underlying library I use.  Thus, in the class methods, these must all
	//        be cast to the correct type.  Obviously, this does not affect you if
	//        you only use the public methods.
	void *solid;    // Type "Solid *"

	// Store information about a currently constructed display list
	GLuint triList, edgeList, pointList, adjList;
	GLuint triVBO, edgeVBO;
	GLuint vboTriCount, vboEdgeCount, vboEdgeComponents, vboTriComponents;

	// Internal methods to create various types of display lists and VBOs
	GLuint CreateTriangleDisplayList( unsigned int flags );
	GLuint CreateEdgeDisplayList( unsigned int flags );
	GLuint CreatePointDisplayList( unsigned int flags );
	GLuint CreateTriangleAdjacencyDisplayList( unsigned int flags );
	GLuint CreateEdgeVBO( unsigned int flags );
	GLuint CreateTriangleVBO( unsigned int flags );

	// Computes a triangle/plane normal from a "Face *".  This assumes the
	//    underlying facet (a "Face" structure) is planar.  If it is not, the
	//    wrong normal will be returned, since it always uses the same 3 facet
	//    vertices to compute the normal.  False is returned if the structure
	//    is corrupted and (thus) cannot compute a normal (resultNorm = undefined).
	bool ComputeFaceNormal( float *resultNorm, void *face );
};


#endif