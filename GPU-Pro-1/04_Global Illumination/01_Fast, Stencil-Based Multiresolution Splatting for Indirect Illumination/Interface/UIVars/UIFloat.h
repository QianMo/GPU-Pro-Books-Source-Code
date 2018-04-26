/******************************************************************/
/* UIFloat.h	                                                  */
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


#ifndef UIFLOAT_H
#define UIFLOAT_H

#include <stdio.h>
#include <stdlib.h>
#include "UIVariable.h"

class UIFloat : public UIVariable {
public:
	UIFloat( char *name=0 ) : UIVariable(name), _min(FLT_MIN), _max(FLT_MAX) {}
	UIFloat( float value, char *name=0 ) : UIVariable(name), value(value), _min(FLT_MIN), _max(FLT_MAX) {}
	UIFloat( float value, float _max, float _min, char *name=0 ) : UIVariable(name), value(value), _max(_max), _min(_min) {}
	UIFloat( FILE *f, Scene *s ) { printf("UIFloat constructor error!"); }
	virtual ~UIFloat() {}

	// Get and set values
	//     Note:  We can get/set values with typical class accessor methods
	void SetValue( float newValue );							
	inline float GetValue( void ) const								{ return value; }
	inline float *GetValuePtr( void )								{ return &value; }

	// Set the max and min values
	inline void SetValueRange( float newMin, float newMax )			{ _max=newMax; _min=newMin; value=(value>newMax?newMax:value); value=(value<newMin?newMin:value);}

	//     Note:  We can also use UIInt just like a standard bool, thanks
	//            the the magic of overloaded assignment and casting operators
	inline UIFloat& operator=( float newValue )						{ SetValue( newValue ); return *this; }
	inline operator const float()									{ return value; }

	// This method updates the current variable, if appropriate
	virtual bool UpdateFromInput( unsigned int inputKey );

	// This method adds a key for the variable to respond to
	inline void AddKeyResponse( unsigned int key, float response )	{ keys.Add( key ); responses.Add( response ); }
protected:
	float value, _max, _min;
	Array1D<float> responses;

private:
	// Explicitly outlaw copy-constructors or UIVariable assignment
	//     by declaring undefined methods.
	UIFloat( const UIFloat& );
	UIFloat& operator=( const UIFloat& );
};


#endif