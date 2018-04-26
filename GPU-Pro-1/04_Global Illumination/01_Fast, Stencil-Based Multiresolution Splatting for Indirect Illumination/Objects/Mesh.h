/******************************************************************/
/* Mesh.h                                                        */
/* -----------------------                                        */
/*                                                                */
/* The file defines an object container (i.e., an object that     */
/*     contains numerous others).  The intersection technique     */
/*     simply loops through all objects that have been added to   */
/*     the group, calls its Intersect() routine, and returns the  */
/*     closest object.  Obviously, this is NOT the most efficient */
/*     container class possible.                                  */
/*                                                                */
/* Chris Wyman (10/26/2006)                                       */
/******************************************************************/

#ifndef __MESH_H__
#define __MESH_H__

#include "Objects/Object.h"
#include "Utils/TextParsing.h"
#include "DataTypes/Array1D.h"
#include "DataTypes/Matrix4x4.h"
#include "Utils/ModelIO/SimpleModelLib.h"

#define MESH_RENDER_AS_DISPLAY_LIST       0
#define MESH_RENDER_AS_VBO_VERTEX_ARRAY   1

struct _GLMmodel;

class Mesh : public Object {
protected:
	char *filename, *lowResFile;
	HalfEdgeModel *hem, *hem_lowRes;
	_GLMmodel     *glm, *glm_lowRes;

	Array1D<Object *> objs;
	Matrix4x4 meshXForm;

	int modelType, renderMode;
	GLuint displayListID, displayListID_low;
	GLuint elementVBO, interleavedVertDataVBO, elementCount;
	GLuint elementVBO_low, interleavedVertDataVBO_low, elementCount_low;
public:
	// Set up a mesh
	Mesh( Material *matl=0 );   
	Mesh( char *linePtr, FILE *f, Scene *s );

	// Free all the memory inside this mesh.
	~Mesh();

	// The basic operation every object must do:  Draw itself.   
	virtual void Draw( Scene *s, 
		               unsigned int matlFlags, 
					   unsigned int optionFlags=OBJECT_OPTION_NONE );
	virtual void DrawOnly( Scene *s, 
		                   unsigned int propertyFlags, 
						   unsigned int matlFlags, 
						   unsigned int optionFlags=OBJECT_OPTION_NONE );

	// Preprocess each of the individual objects
	virtual void Preprocess( Scene *s );
	virtual bool NeedsPreprocessing( void ) { return (displayListID==0 && interleavedVertDataVBO==0); }
};





#endif




