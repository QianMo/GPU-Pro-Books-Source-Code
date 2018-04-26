/******************************************************************/
/* UIVariable.h                                                   */
/* -----------------------                                        */
/*                                                                */
/* There's nearly always values in your program that need to      */
/*     change according to some user input.  Often these values   */
/*     require all sorts of logic to update, set, get, or         */
/*     change in response to user interface.                      */
/* This class tries to avoid spreading this logic out over the    */
/*     entire code, and instead encapsulate it all in a class.    */
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

#include "UIVariable.h"

bool UIVariable::RespondsToInput( unsigned int inputKey )
{
	for (unsigned int i=0; i< keys.Size(); i++)
		if (inputKey == keys[i]) return true;
	return false;
}


void UIVariable::RemoveKeyResponse( unsigned int key )
{
	// We should really shrink the array, not just overwrite
	//    the keystroke with a NULL.  But this should be an infrequently
	//    used function, and Array1D does not currently support 
	//    removing array entries.
	for (unsigned int i=0; i< keys.Size(); i++)
		if (key == keys[i]) keys[i] = KEY_UNKNOWN;
}

