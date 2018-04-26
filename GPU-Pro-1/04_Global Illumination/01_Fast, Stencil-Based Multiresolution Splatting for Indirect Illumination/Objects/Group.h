/******************************************************************/
/* Group.h                                                        */
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

#ifndef GROUP_H
#define GROUP_H

#include "Objects/Object.h"
#include "Utils/TextParsing.h"
#include "DataTypes/Array1D.h"
#include "DataTypes/Matrix4x4.h"

class Group : public Object {
protected:
	Array1D<Object *> objs;
	Array1D<Object *> updatableObjs;
	bool needsPreprocessing, needsFrameUpdates;

	Matrix4x4 groupXForm;
public:
	// Set up a default (empty) group.
	Group( Material *matl=0 );   

	// Currently, one cannot read a group of objects from a file.  If you wish to
	//    define how this would be done, you can change this.
	Group( FILE *f, Scene *s );

	// Free all the memory inside this group.
	virtual ~Group();

	// Add an object to the group.
	void Add( Object *obj );

	// Get an object from the group.  (Note: no bounds checking is done)
	Object *Get( int i );
	const Object *Get( int i ) const;

	// Get the number of objects in the group.
	inline int GetSize( void ) const   { return objs.Size(); }

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
	virtual bool NeedsPreprocessing( void ) { return needsPreprocessing; }

	// Functions to see if geometry wants to update itself every frame.
	virtual bool NeedPerFrameUpdates( void ) { return needsFrameUpdates; }
	virtual void Update( float currentTime ); 
};





#endif




