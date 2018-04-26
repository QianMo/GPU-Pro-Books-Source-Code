/******************************************************************/
/* TextParsing.cpp                                                */
/* -----------------------                                        */
/*                                                                */
/* The file defines a few text parsing routines I use to read in  */
/*      scenes from file.                                         */
/*                                                                */
/* Chris Wyman (10/26/2006)                                       */
/******************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>


/* takes an entire string, and makes it all lowercase */
void MakeLower( char *buf )
{
  char *tmp = buf;

  while ( tmp[0] != 0 )
    {
      *tmp = (char)tolower( *tmp );
      tmp++;
    }
}

/* takes an entire string, and makes it all uppercase */
void MakeUpper( char *buf )
{
  char *tmp = buf;

  while ( tmp[0] != 0 )
    {
      *tmp = (char)toupper( *tmp );
      tmp++;
    }
}

/* check to see if a character is a white space. */
/*    NOTE: no functions in this file use this!  */
int IsWhiteSpace( char c )
{
	if (c == ' ' || c == '\t' || c == '\n' || c == '\r') return 1;
	return 0;
}

/* 
** Returns a ptr to the first non-whitespace character 
** in a string
*/
char *StripLeadingWhiteSpace( char *string )
{
  char *tmp = string;

  while ( (tmp[0] == ' ') ||
	  (tmp[0] == '\t') ||
	  (tmp[0] == '\n') ||
	  (tmp[0] == '\r') )
    tmp++;

  return tmp;
}

/* 
** Returns a ptr to the first non-whitespace character 
** in a string...  Also includes the special character(s) 
** when it considers whitespace (useful for stripping
** off commas, parenthesis, etc)
*/
char *StripLeadingSpecialWhiteSpace( char *string, char special, char special2 )
{
  char *tmp = string;

  while ( (tmp[0] == ' ') ||
	  (tmp[0] == '\t') ||
	  (tmp[0] == '\n') ||
	  (tmp[0] == '\r') ||
	  (tmp[0] == special) ||
	  (tmp[0] == special2) )
    tmp++;
  
  return tmp;
}

/*
** Returns the first 'token' (string of non-whitespace
** characters) in the buffer, and returns a pointer to the
** next non-whitespace character (if any) in the string
*/
char *StripLeadingTokenToBuffer( char *string, char *buf )
{
  char *tmp = string;
  char *out = buf;

  while ( (tmp[0] != ' ') &&
	  (tmp[0] != '\t') &&
	  (tmp[0] != '\n') &&
	  (tmp[0] != '\r') &&
	  (tmp[0] != 0) )
    {
      *(out++) = *(tmp++); 
    }
  *out = 0;

  return StripLeadingWhiteSpace( tmp );
}


/*
** Returns the first 'token' (string of non-whitespace
** characters) in the buffer, and returns a pointer to the
** next non-whitespace character (if any) in the string
**
** Also, this function stops if it encounters either of the special
** characters passed in by the user, and returns the token thus far...
** This is useful for reading till a comma or a parenthesis, etc
**
** Returns the reason it stopped (as a character) if passed a pointer
** to a character.
*/
char *StripLeadingSpecialTokenToBuffer ( char *string, char *buf, 
					 char special, char special2, char *reason)
{
  char *tmp = string;
  char *out = buf;

  while ( (tmp[0] != ' ') &&
	  (tmp[0] != '\t') &&
	  (tmp[0] != '\n') &&
	  (tmp[0] != '\r') &&
	  (tmp[0] != 0) &&
	  (tmp[0] != special) &&
	  (tmp[0] != special2) )
    {
      *(out++) = *(tmp++); 
    }
  
  if (reason) *reason = tmp[0];
  *out = 0;

  return StripLeadingSpecialWhiteSpace( tmp, tmp[0], 0 );
}

/*
** works the same ways as StripLeadingTokenToBuffer,
** except instead of returning a string, it returns
** basically atof( StripLeadingTokenToBuffer ), with
** some minor cleaning beforehand to make sure commas,
** parens and other junk don't interfere.
*/
char *StripLeadingNumber( char *string, float *result )
{
  char *tmp = string;
  char buf[80];
  char *ptr = buf;
  char *ptr2;

  tmp = StripLeadingTokenToBuffer( tmp, buf );
  
  /* find the beginning of the number */
  while( (ptr[0] != '-') &&
	 (ptr[0] != '.') &&
	 ((ptr[0]-'0' < 0) ||
	  (ptr[0]-'9' > 0)) )
    ptr++;

  /* find the end of the number */
  ptr2 = ptr;
  while( (ptr2[0] == '-') ||
	 (ptr2[0] == '.') ||
	 ((ptr2[0]-'0' >= 0) && (ptr2[0]-'9' <= 0)) )
    ptr2++;

  /* put a null at the end of the number */
  ptr2[0] = 0;

  *result = (float)atof(ptr);

  return tmp;
}
