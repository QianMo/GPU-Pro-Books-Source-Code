/********************************************
** loadHDR.cpp                             **
** -----------                             **
**                                         **
** Loads an high dynamic range image using **
**    commands described in rgbe.cpp.      **
**                                         **
** Chris Wyman (9/07/2006)                 **
********************************************/

#include <stdio.h>
#include <stdlib.h>
#include "rgbe.h"

void FatalHDRError( char *str, char *fname )
{
  printf( str, fname );
  exit(-1);
}

float *ReadHDR(char *filename, int *w, int *h )
{
  FILE *f;
  float *tmpbuf;
  int width, height;

  /* open the file */
  f = fopen( filename, "rb" );
  if (!f) FatalHDRError( "Unable to load image file \"%s\" in loadHDR()!\n", filename ); 

  /* read in header information */
  if (RGBE_ReadHeader( f, &width, &height, NULL ) != RGBE_RETURN_SUCCESS)
    FatalHDRError( "Unable to read HDR header in image \"%s\"!\n", filename );

  /* allocate memory, both temporary & final */
  tmpbuf = (float *)malloc( 3 * width * height * sizeof(float) );
  if (!tmpbuf ) FatalHDRError( "Unable to allocate memory for image \"%s\"!\n", filename );

  /* read in the image data */
  if (RGBE_ReadPixels_RLE(f, tmpbuf, width, height) != RGBE_RETURN_SUCCESS)
    FatalHDRError( "Unable to read HDR data in image \"%s\"!\n", filename );
  
  /* clse the file */
  fclose(f);
 
  *w = width;
  *h = height;
  return tmpbuf;
}
