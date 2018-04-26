/******************************************************************/
/* UIInt.h	                                                      */
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


#ifndef UIINT_H
#define UIINT_H

#include <stdio.h>
#include <stdlib.h>
#include "UIVariable.h"

class UIInt : public UIVariable {
public:
	UIInt( char *name=0 ) : UIVariable(name), value(0), _max(INT_MAX), _min(INT_MIN) {}
	UIInt( int value, char *name=0 ) : UIVariable(name), value(value), _max(INT_MAX), _min(INT_MIN) {}
	UIInt( int value, int _max, int _min, char *name=0 ) : UIVariable(name), value(value), _max(_max), _min(_min) {}
	UIInt( FILE *f, Scene *s ) { printf("UIInt constructor error!"); }
	virtual ~UIInt() {}

	// Get and set values
	//     Note:  We can get/set values with typical class accessor methods
	void SetValue( int newValue );
	inline int GetValue( void ) const								{ return value; }
	inline int *GetValuePtr( void )									{ return &value; }


	// Set the max and min values
	inline void SetValueRange( int newMin, int newMax )				{ _max=newMax; _min=newMin; value=(value>newMax?newMax:value); value=(value<newMin?newMin:value);}

	//     Note:  We can also use UIInt just like a standard bool, thanks
	//            the the magic of overloaded assignment and casting operators
	inline UIInt& operator=( int newValue )							{ SetValue( newValue ); return *this; }
	inline operator const int()										{ return value; }

	// This method updates the current variable, if appropriate
	virtual bool UpdateFromInput( unsigned int inputKey );

	// This method adds a key for the variable to respond to
	inline void AddKeyResponse( unsigned int key, int response )	{ keys.Add( key ); responses.Add( response ); }
protected:
	int value, _min, _max;
	Array1D<int> responses;

private:
	// Explicitly outlaw copy-constructors or UIVariable assignment
	//     by declaring undefined methods.
	UIInt( const UIInt& );
	UIInt& operator=( const UIInt& );
};


#endif