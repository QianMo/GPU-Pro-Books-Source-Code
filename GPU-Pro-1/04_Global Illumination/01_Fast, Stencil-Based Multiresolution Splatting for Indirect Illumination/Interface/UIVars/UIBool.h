/******************************************************************/
/* UIBool.h                                                       */
/* -----------------------                                        */
/*                                                                */
/* There's nearly always values in your program that need to      */
/*     change according to some user input.  Often these values   */
/*     require all sorts of logic to update, set, get, or         */
/*     change in response to user interface.                      */
/* This class, and others like it, inherit from UIVariable and    */
/*     try to avoid spreading UI interface logic out over the     */
/*     entire code, and instead encapsulate it in these classes.  */
/*     The additional advantage here, is that derived classes     */
/*     from UIVariable can all be treated identically, avoiding   */
/*     special-casing depending on the variable type.             */
/*                                                                */
/* Realize that this class adds overhead, and shouldn't be used   */
/*     for all variables.  Just those that might vary with input  */
/*     or, perhaps, those defined in an input or script file.     */
/*                                                                */
/* Chris Wyman (02/12/2009)                                       */
/******************************************************************/


#ifndef UIBOOL_H
#define UIBOOL_H

#include <stdio.h>
#include <stdlib.h>
#include "UIVariable.h"

class UIBool : public UIVariable {
public:
	UIBool( char *name=0 ) : UIVariable(name) {}
	UIBool( bool value, char *name=0 ) : UIVariable(name), value(value) {}
	UIBool( FILE *f, Scene *s ) { printf("UIBool constructor error!"); }
	virtual ~UIBool() {}

	// Get and set values
	//     Note:  We can get/set values with typical class accessor methods
	inline void SetValue( bool newValue )				{ value = newValue; }
	inline bool GetValue( void ) const					{ return value; }
	inline bool *GetValuePtr( void )					{ return &value; }


	//     Note:  We can also use UIBool just like a standard bool, thanks
	//            the the magic of overloaded assignment and casting operators
	inline UIBool& operator=( bool newValue )			{ value = newValue; return *this; }
	inline operator const bool()						{ return value; }

	// This method updates the current variable, if appropriate
	virtual bool UpdateFromInput( unsigned int inputKey );

	// This method adds a key for the variable to respond to
	inline void AddKeyResponse( unsigned int key )		{ keys.Add( key ); }
protected:
	bool value;

private:
	// Explicitly outlaw copy-constructors or UIVariable assignment
	//     by declaring undefined methods.
	UIBool( const UIBool& );
	UIBool& operator=( const UIBool& );
};


#endif