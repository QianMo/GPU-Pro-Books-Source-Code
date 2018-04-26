/******************************************************************/
/* TextParsing.h                                                  */
/* -----------------------                                        */
/*                                                                */
/* The file defines a few text parsing routines I use to read in  */
/*      scenes from file.                                         */
/*                                                                */
/* Chris Wyman (10/26/2006)                                       */
/******************************************************************/

#ifndef TEXTPARSING_H
#define TEXTPARSING_H

// takes an entire string, and makes it all lowercase 
void MakeLower( char *buf );

// takes an entire string, and makes it all uppercase
void MakeUpper( char *buf );


// Returns a ptr to the first non-whitespace character in a string
//   (For all these functions, "whitespace" means ' ', '\t', '\n', AND '\r', which means
//    they should handle Unix files under Windows and Windows files under Unix)
char *StripLeadingWhiteSpace( char *string );
 
// Returns a ptr to the first non-whitespace character in a string...
// Also includes the special character(s) when it considers whitespace 
// (useful for stripping off commas, parenthesis, etc)
char *StripLeadingSpecialWhiteSpace( char *string, char special=' ', char special2=' ' );


// Returns the first 'token' (string of non-whitespace characters) in 
// the buffer, and returns a pointer to the next non-whitespace 
// character (if any) in the string
char *StripLeadingTokenToBuffer( char *string, char *buf );


// Returns the first 'token' (string of non-whitespace characters) in 
// the buffer, and returns a pointer to the next non-whitespace 
// character (if any) in the string
//
// Also, this function stops if it encounters either of the special
// characters passed in by the user, and returns the token thus far...
// This is useful for reading till a comma or a parenthesis, etc
//
// Returns the reason it stopped (as a character) if passed a pointer
// to a character.
char *StripLeadingSpecialTokenToBuffer ( char *string, char *buf, 
					 char special=' ', char special2=' ', char *reason=0);


// Works the same ways as StripLeadingTokenToBuffer, except instead of 
// returning a string, it returns basically atof( StripLeadingTokenToBuffer ), 
// with some minor cleaning beforehand to make sure commas, parens and 
// other junk don't interfere.
char *StripLeadingNumber( char *string, float *result );

// Checks to see if a character is a white space.
int IsWhiteSpace( char c );

#endif

