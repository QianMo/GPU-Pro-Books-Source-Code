/******************************************************************/
/* imageIO.h                                                      */
/* -----------------------                                        */
/*                                                                */
/* Most of the publically accessible function in these image i/o  */
/*    routines are really useless unless you really understand    */
/*    the file format.  This header includes the really useful    */
/*    function prototypes -- essentially the dumbed-down way to   */
/*    load and write images.                                      */
/*                                                                */
/* Chris Wyman (01/30/2008)                                       */
/******************************************************************/

#ifndef __IMAGE_IO_H__
#define __IMAGE_IO_H__

// Reads '.rgb' and '.rgba' files  (SGI file format)
//   -> 'components' is the number of unsigned chars per pixel.
unsigned char *ReadRGB( const char *name, int *width, int *height, int *components, bool invertY=false );

// Reads '.hdr' and '.rgbe' files  (Greg Ward's format).
//   -> Note the header parser is a bit fragile...
float *ReadHDR( char *filename, int *width, int *height );

// Reads '.bmp' files (Windows bitmaps)
unsigned char *ReadBMP( char *f, int *width, int *height, bool invertY=false );

// Reads '.ppm' '.pgm' and (to some extent) '.pbm' files (Portable Pixel Maps)
//   -> 'mode' can be ignored, since this code always returns 3 chars per pixel
unsigned char *ReadPPM( char *f, int *mode, int *width, int *height, bool invertY=false );


// Error code used by image I/O routines.  Can be useful elsewhere too!
void FatalError( char *msg );    // a fatal, terminating error 
void Error( char *msg );         // a non-fatal, non-terminating error 
void Warning( char *msg );       // a warning to the user...  

// These versions first insert the 2nd string into the first, then print.
void FatalError( char *formatStr, char *insertThisStr );
void Error( char *formatStr, char *insertThisStr );
void Warning( char *formatStr, char *insertThisStr );

#endif

