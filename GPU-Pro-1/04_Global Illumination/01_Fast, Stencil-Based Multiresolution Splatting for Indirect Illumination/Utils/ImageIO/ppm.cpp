#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "ppm.h"

/* 
** check if integer specified is a valid 
** image mode 
*/
int IsValidMode( int mode )
{
  if (mode==PPM_ASCII) return 1;
  if (mode==PGM_ASCII) return 1;
  if (mode==PBM_ASCII) return 1;
  if (mode==PPM_RAW) return 1;
  if (mode==PGM_RAW) return 1;
  if (mode==PBM_RAW) return 1;
  return 0;
}

/* is this mode a raw mode? */
int RawMode( int mode )
{
  if (mode==PPM_RAW) return 1;
  if (mode==PGM_RAW) return 1;
  if (mode==PBM_RAW) return 1;
  return 0;
}

/* is this mode a raw mode? */
int ASCIIMode( int mode )
{
  if (mode==PPM_ASCII) return 1;
  if (mode==PGM_ASCII) return 1;
  if (mode==PBM_ASCII) return 1;
  return 0;
}


/* a fatal, terminating error */
void FatalError( char *msg )
{
  char buf[512];  
  sprintf( buf, "Fatal Error: %s\n", msg );
  fprintf( stderr, buf );
  exit(-1);
}

void FatalError( char *formatStr, char *insertThisStr )
{
  char buf[512];  
  sprintf( buf, formatStr, insertThisStr );
  FatalError( buf );
}

/* a non-fatal, non-terminating error */
void Error( char *msg )
{
  char buf[512];  
  sprintf( buf, "Error: %s\n", msg );
  fprintf( stderr, buf );
}

void Error( char *formatStr, char *insertThisStr )
{
  char buf[512];  
  sprintf( buf, formatStr, insertThisStr );
  Error( buf );
}

/* a warning to the user...  */
void Warning( char *msg )
{
  char buf[512];  
  sprintf( buf, "Warning: %s\n", msg );
  fprintf( stderr, buf );
}

void Warning( char *formatStr, char *insertThisStr )
{
  char buf[512];  
  sprintf( buf, formatStr, insertThisStr );
  Warning( buf );
}



/* read a texture from a file */
unsigned char *ReadPPM( char *f, int *mode, int *width, int *height, bool invertY )
{
  FILE *infile;
  char string[256], buf[80];
  unsigned char *texImage;
  int i, j, count=0, img_max;
  long img_size;
  int r, g, b, bw, c=0, raw=0, ascii=0;
    
  /* open file containing texture */
  if ((infile = fopen(f, "rb")) == NULL) {
    sprintf(buf, "LIBGFX: Can't open file '%s'!", f);
    FatalError( buf );
  }
  
  /* read and discard first line (the P3 or P6, etc...) */
  fgets(string, 256, infile);
  *mode = string[1]-'0';
  if ((!IsValidMode(*mode)) || (string[0] != 'P' && string[0] != 'p'))
    {
      sprintf(buf, "LIBGFX: Invalid PPM format specification in '%s'!", f);
      FatalError(buf);
    }
  raw = RawMode( *mode );
  ascii = ASCIIMode( *mode );

  /* discard all the comments at the top of the file               */
  /* if there are comments elsewhere, this code doesn't handle it! */
  fgets(string, 256, infile);
  while (string[0] == '#')
     fgets(string, 256, infile);

  /* read image size and max component value */
  sscanf(string, "%d %d", width, height);
  
  /* PBMs are just 1's and 0's, so there's no max_component */
  if (*mode != PBM_RAW && *mode != PBM_ASCII)
    {
      fscanf(infile, "%d ", &img_max);
      if ((raw && img_max > 255) || (img_max <= 0))
	  FatalError( "LIBGFX: Invalid value for maximum image color!" );
    }

  /* allocate texture array */
  img_size = 3*(*height)*(*width);
  if ((texImage = (unsigned char *)calloc(img_size, sizeof(char))) == NULL)
    FatalError("LIBGFX: Cannot allocate memory for image!");
  
  /* read image data */
  for (i=0; i < *height; i++) {
    for (j=0; j < *width; j++) {
	  int loc = invertY ? 3*((*width)*(*height-1-i) + j) : 3*((*width)*i + j);
      if (raw)
	    {
		  if (*mode==PBM_RAW)
	      {
	        int bit_pos;
	        if ((count%8)==0) bw=fgetc(infile);
	        bit_pos = 7-(count%8);
	        r=g=b=((bw & (1 << bit_pos))?0:255);
	        count++;
	        if ((j+1)==*width) count=0;
	      }
	    else if (*mode==PGM_RAW)
	      r=g=b=fgetc(infile);
	    else if (*mode==PPM_RAW)
	      {
	        r = fgetc(infile);
	        g = fgetc(infile);
	        b = fgetc(infile);
	      }
	  //texImage[c++] = r;
	  //texImage[c++] = g;
	  //texImage[c++] = b;
	  texImage[loc+0] = r;
	  texImage[loc+1] = g;
	  texImage[loc+2] = b;
	  }
      else /* then ASCII mode */
	  {
	    if (*mode==PBM_ASCII)
	      {
	        fscanf(infile, "%d", &r);
	        if (r!=1 && r!=0) r=0;
	        r=g=b=r*255;
	      }
	    else if (*mode==PGM_ASCII)
	      {
	        fscanf(infile, "%d", &r);
	        r=g=b=r;
	      }
	    else if (*mode==PPM_ASCII)
	    fscanf(infile, "%d %d %d", &r, &g, &b);
	    //texImage[c++] = r;
	    //texImage[c++] = g;
	    //texImage[c++] = b;
		texImage[loc+0] = r;
		texImage[loc+1] = g;
		texImage[loc+2] = b;
	  }
    }
  }
 
  fclose( infile );
  
  return texImage;
}



/* 
** write the image with a given width & height to a file called filename,
** the data is given as a stream of chars (unsigned bytes) in the pointer 
** ptr.
*/
int WritePPM( char *f, int mode, int width, int height, unsigned char *ptr )
{
  time_t thetime;
  int x, y, i;
  unsigned char r, g, b, bw=0;
  int invertedHeight = 0;
  FILE *out;

  if (height < 0) 
  {
	invertedHeight = 1;
	height = -height;
  }

  if (mode<0) 
    {
      Error("LIBGFX: Bad image format type!");
      return GFXIO_UNSUPPORTED;
    }
  if (mode==PBM_RAW || mode==PBM_ASCII)
    Warning("LIBGFX: Distortions occur converting to PBM format!");

  out = fopen(f, "wb");
  if (!out) {
    char buf[256];
    sprintf( buf, "LIBGFX: Unable to open file '%s', output lost!", f );
    Error( buf );
    return GFXIO_OPENERROR;
  }
  
  fprintf(out, "P%d\n", mode);  
  thetime = time(0);
  fprintf(out, "# File created by Chris Wyman's PPM Library on %s",
	  ctime(&thetime));
  fprintf(out, "%d %d\n", width, height);
  /* PBM's are just 1's and 0's, so there's no max component entry */
  if (mode!=PBM_RAW && mode!=PBM_ASCII)
    fprintf(out, "%d\n", 255); 
  
  for (   y = (invertedHeight? height-1 : 0)     ; 
	      (invertedHeight? y >= 0 : y < height)  ; 
		  (invertedHeight? y-- : y++)            ) {
    for (x = 0; x < width; x++) {
      r = *(ptr+3*(y*width+x));  
      g = *(ptr+3*(y*width+x)+1);
      b = *(ptr+3*(y*width+x)+2);

      if (mode == PPM_RAW)
        fprintf(out, "%c%c%c", r, g, b);
      else if (mode == PPM_ASCII)
	  {
	    fprintf(out, "%d %d %d ", r, g, b);
	    if (((++i) % 5) == 0) fprintf( out, "\n" ); 
	  }
      else if (mode == PGM_RAW)
        fprintf(out, "%c", ((r+g+b)/3));
      else if (mode == PGM_ASCII)
	  {
	    fprintf(out, "%d ", (r+g+b)/3);
	    if (((++i) % 15) == 0) fprintf( out, "\n" ); 
	  }
      else if (mode == PBM_ASCII)
	  {
	    fprintf(out, "%d ", ( (((r+g+b)/3)<128) ? 1 : 0));
	    if (((++i) % 15) == 0) fprintf( out, "\n" ); 
	  }
      else if (mode == PBM_RAW)
	  {
	    char add=0;
	    if (((r+g+b)/3)<128) add=1;
	    bw = (bw << 1)+add;
	    if (((++i) % 8) == 0) 
	    {
	      fprintf( out, "%c", bw );
	      bw = 0;
	    }
	    else if ((x+1)==width) 
	    /* there's padding at the end of rows, evidently... grr! */
	    {
	      int bits=8-(i%8);
	      bw = bw<<bits;
	      fprintf( out, "%c", bw );
	      bw=i=0;
	    }
	  }
    }
  }
  fprintf(out, "\n");
  fclose(out);

  return GFXIO_OK;
}