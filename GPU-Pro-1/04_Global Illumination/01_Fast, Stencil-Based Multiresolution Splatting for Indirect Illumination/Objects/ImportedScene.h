/******************************************************************/
/* ImportedScene.h                                                */
/* -----------------------                                        */
/*                                                                */
/* The file defines an container that contains an imported scene. */
/*    An imported "Scene" is conceptually different than a "Mesh" */
/*    in the sense that a scene may have multiple objects or      */
/*    components, each with a separate material type.  While this */
/*    distinction is somewhat arbitrary, and thus classifies many */
/*    true "objects" as scenes, it simplifies mesh construction   */
/*    somewhat.                                                   */
/*                                                                */
/* Eventually, "Imported Scene" and "Mesh" might want to be       */
/*    merged back into one construct, for user clarity.           */
/*                                                                */
/* Chris Wyman (04/17/2009)                                       */
/******************************************************************/

#ifndef __IMPORTED_SCENE_H__
#define __IMPORTED_SCENE_H__

#include "Objects/Object.h"
#include "Utils/TextParsing.h"
#include "DataTypes/Array1D.h"
#include "DataTypes/Matrix4x4.h"
#include "Utils/ModelIO/SimpleModelLib.h"
#include "Objects/Mesh.h"


class ImportedScene : public Object {
protected:
	char *filename;
	int modelType, renderMode;
	_GLMmodel     *glm;

	Array1D<Object *> objs;
	Matrix4x4 sceneXForm;

	GLuint interleavedFormat;
public:
	// Set up a mesh
	ImportedScene( Material *matl=0 ) { printf("Error: ImportedScene( Material *) called!  This is undefined!\n"); }
	ImportedScene( char *linePtr, FILE *f, Scene *s );

	// Free all the memory inside this mesh.
	~ImportedScene();

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
	virtual bool NeedsPreprocessing( void ) { return true; }
};





#endif




