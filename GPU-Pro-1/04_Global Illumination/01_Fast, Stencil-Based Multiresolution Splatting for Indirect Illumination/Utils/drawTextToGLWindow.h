/************************************************************************/
/* drawTextToGLWindow.h                                                 */
/* ------------------                                                   */
/*                                                                      */
/* This file contains prototypes for a number of utility functions that */
/*    use GLUT to draw simple text into the OpenGL window.              */
/*                                                                      */
/* PLEASE NOTE: The functions are actually _quite_ slow, mainly because */
/*    they output one character at a time, using a character bitmap.    */
/*    The glutStrokeCharacter() function is much faster but I find it   */
/*    (a) uglier and (b) harder to control.                             */
/*                                                                      */
/* Typically I use this code simply to display a frame counter, and in  */
/*    non-framerate demanding locations (e.g., a help screen).  It      */
/*    appears DisplayString() takes about 1 ms for even one 10 char     */
/*    string.                                                           */
/*                                                                      */
/* Chris Wyman (12/7/2007)                                              */
/************************************************************************/


#ifndef __DRAWTEXTTOGLWINDOW_H__
#define __DRAWTEXTTOGLWINDOW_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// I assume you're using GLEW.  If not, you can remove the first include.
//    But if you *ARE* using it, it needs to be included before GLUT.
#include "Utils/GLee.h"
#include <GL/glut.h>

// Prints a string at the current glRasterPos() using the specified font.
void PrintString(char *str, void *font=GLUT_BITMAP_HELVETICA_12);


// Display a string at a particular raster position on the screen.  This
//     function takes care of all the matrix manipulations and resetting GL
//     state to make sure it appears on screen where it should.  This means
//     if you repeatedly call this function you will make lots of duplicate
//     calls and you would be better off writing a function to call 
//     PrintString() directly.  If you specify the screen dimentions, the
//     function need not look them up using glutGet().
void DisplayString( int rasterPosX, int rasterPosY, char *str, 
				    int screenWidth=-1, int screenHeight=-1 );


// Displays a framerate counter in the lower left corner of your window
//     If you specify the screen dimensions, the function will not look
//     them up (by calling glutGet()) which will speed execution.
void DisplayTimer( float fps, int screenWidth=-1, int screenHeight=-1 );


#endif


