/******************************************************************/
/* ImportedSceneComponent.h                                       */
/* -----------------------                                        */
/*                                                                */
/* The file defines an container that stores a scene component,   */
/*    namely a set of triangles or other primitves that all have  */
/*    the same material propoerty.  This is not really different  */
/*    then the "mesh" object, except it requires different        */
/*    initialization techniques.                                  */
/*                                                                */
/* Eventually, "Imported Scene" and "Mesh" might want to be       */
/*    merged back into one construct, for user clarity.           */
/*                                                                */
/* Chris Wyman (04/17/2009)                                       */
/******************************************************************/

#ifndef __IMPORTED_SCENE_COMPONENT_H__
#define __IMPORTED_SCENE_COMPONENT_H__

#include "Objects/Object.h"
#include "Utils/TextParsing.h"
#include "DataTypes/Array1D.h"
#include "DataTypes/Matrix4x4.h"
#include "Objects/ImportedScene.h"
#include "Utils/ModelIO/glm.h"

class ImportedSceneComponent : public Object {
protected:
	int renderMode;
	GLuint displayListID;
	GLuint interleavedVertDataVBO, elementCount;
	GLenum interleavedFormat;
public:
	// Importing a null component.  Currently unsupported
	ImportedSceneComponent( Material *matl=0 ) { printf("Error: ImportedSceneComponent( Material *) called!  This is undefined!\n"); } 

	// For importing a component to be drawn as a display list
	ImportedSceneComponent( Scene *s, _GLMmodel *glm, _GLMgroup *component ); 
	
	// For importing a component to be drawn as a VBO-based vertex array
	ImportedSceneComponent( Scene *s, _GLMmodel *glm, _GLMgroup *component, GLenum interleavedFormat );
		                    

	// Free all the memory inside this mesh.
	~ImportedSceneComponent();

	// The basic operation every object must do:  Draw itself.   
	virtual void Draw( Scene *s, 
		               unsigned int matlFlags, 
					   unsigned int optionFlags=OBJECT_OPTION_NONE );
	virtual void DrawOnly( Scene *s, 
		                   unsigned int propertyFlags, 
						   unsigned int matlFlags, 
						   unsigned int optionFlags=OBJECT_OPTION_NONE );

	// Preprocess each of the individual objects
	//    Technically, these might need preprocessing, but because of how scene components
	//    are initialized (i.e., during ImportedScene::Preprocess()), the constructor 
	//    can do anything that might have required preprocessing normally.  Thus these
	//    are false, and a no-op.
	virtual void Preprocess( Scene *s ) {};
	virtual bool NeedsPreprocessing( void ) { return false; }
};





#endif




