/******************************************************************/
/* Texture.cpp                                                    */
/* -----------------------                                        */
/*                                                                */
/* The file defines an image class that stores a texture.         */
/*     This is very similar to the image class, but also defines  */
/*     access patterns for the texture with interpolation.        */
/*                                                                */
/* Chris Wyman (10/26/2006)                                       */
/******************************************************************/

#include "DataTypes/Texture.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "Utils/ProgramPathLists.h"

extern ProgramSearchPaths *paths;

Texture::Texture(int width, int height) :
	width(width), height(height), type( TEXTURE_TYPE_UNKNOWN ), fileName(0),
		wrap_u( TEXTURE_CLAMP ), wrap_v( TEXTURE_CLAMP )
{
	// Allocate memory
	image1D = new RGBAColor[width*height];
	image2D = new RGBAColor *[height];
	for (int i=0;i<height;i++)
		image2D[i] = image1D+i*width;
}

Texture::Texture(char *filename, float scale) : wrap_u( TEXTURE_CLAMP ), wrap_v( TEXTURE_CLAMP )
{
	char *ptr;

	if (!Color::IsRGBColor())
	{
		printf("Error in Texture::Texture():  Currently unable to load textures using spectral colors!\n");
		exit(0);
	}

	// Identify the type of file
	ptr = strrchr( filename, '.' );
	char buf[16];
	strncpy( buf, ptr, 16 );
	MakeLower( buf );
	if (!strcmp(buf, ".ppm"))			type = TEXTURE_TYPE_PPM;
	else if (!strcmp(buf, ".rgb"))		type = TEXTURE_TYPE_RGB;
	else if (!strcmp(buf, ".rgba"))		type = TEXTURE_TYPE_RGBA;
	else if (!strcmp(buf, ".hdr"))		type = TEXTURE_TYPE_HDR;
	else if (!strcmp(buf, ".csv"))		type = TEXTURE_TYPE_CSV;

	if (type == TEXTURE_TYPE_CSV)
		printf("Warning!  CSV loading poorly tested in Texture class!\n");

	if (type == TEXTURE_TYPE_UNKNOWN)
	{
		printf("Error in Texture::Texture():  Unable to load files of type '%s'!\n", ptr );
		exit(0);
	}

	fileName = paths->GetTexturePath( filename );
	if (!fileName) 
	{
		printf("Error in Texture::Texture():  Unable to open texture!\n");
		exit(0);
	}

	if (type == TEXTURE_TYPE_PPM) LoadPPM( fileName, scale );
	else if (type == TEXTURE_TYPE_RGB) LoadRGB( fileName, scale );
	else if (type == TEXTURE_TYPE_RGBA) LoadRGB( fileName, scale );
	else if (type == TEXTURE_TYPE_HDR) LoadHDR( fileName, scale );
	else if (type == TEXTURE_TYPE_CSV) LoadCSV( fileName, scale );
}

void Texture::LoadRGB( char *filename, float scale )
{
	int components;
	unsigned char *data = ReadRGB(filename, &width, &height, &components);

	if (!data)
	{
		printf("Error in Texture::LoadRGB(): Unknown error in ReadRGB! (NULL return)\n");
		exit(0);
	}

	// Allocate memory
	image1D = new RGBAColor[width*height];
	image2D = new RGBAColor *[height];
	for (int i=0;i<height;i++)
		image2D[i] = image1D+i*width;

	unsigned char *ptr = data;
	for (int i=0; i < height; i++) 
	{
		for (int j=0; j < width; j++) 
		{
			if (components < 3)
				image2D[i][j] = Color( scale*ptr[0]/255.0f, scale*ptr[0]/255.0f, scale*ptr[0]/255.0f );
			else if (components == 3)
				image2D[i][j] = Color( scale*ptr[0]/255.0f, scale*ptr[1]/255.0f, scale*ptr[2]/255.0f );
			else 
				image2D[i][j] = RGBAColor( scale*ptr[0]/255.0f, scale*ptr[1]/255.0f, scale*ptr[2]/255.0f, ptr[3]/255.0f );
			ptr += 4; //components;
		}
    }

	free( data );
}


void Texture::LoadPPM( char *filename, float scale )
{
	int mode, img_max;
	char buf[256];

	FILE *f = fopen( filename, "rb" );
	fgets( buf, 256, f );
	mode = buf[1]-'0';
	if (mode != 3 && mode != 6)
	{
		printf("Error in Texture::LoadPPM():  Only P3 and P6 .ppm files supported!\n");
		exit(0);
	}
	// Discard any comments here.
	fgets(buf, 256, f);
	while (buf[0] == '#')
		fgets(buf, 256, f);
	// read the width and height
	sscanf( buf, "%d %d", &width, &height );
	// read the max component field
	fscanf(f, "%d ", &img_max);
    if ((mode==6 && img_max > 255) || (img_max <= 0))
	{
		printf("Error in Texture::LoadPPM():  Invalid value for .ppm maximum image color!\n");
		exit(0);
	}

	// Allocate memory
	image1D = new RGBAColor[width*height];
	image2D = new RGBAColor *[height];
	for (int i=0;i<height;i++)
		image2D[i] = image1D+i*width;

	unsigned int r,g,b;
	for (int i=0; i < height; i++) 
	{
		for (int j=0; j < width; j++) 
		{
			if (mode==6)  /* RAW mode */
				{ r = fgetc(f); g = fgetc(f); b = fgetc(f); }
			else          /* ASCII mode */
				fscanf(f, "%d %d %d", &r, &g, &b);
			image2D[height-i-1][j] = Color( scale*r/(float)img_max, scale*g/(float)img_max, scale*b/(float)img_max );
		}
    }
	fclose( f );
}


void Texture::LoadHDR( char *filename, float scale )
{
	float *data = ReadHDR( filename, &width, &height );
	if (!data)
	{
		printf("Unknown error while loading '%s'!\n", filename);
		exit(0);
	}

	// Allocate memory
	image1D = new RGBAColor[width*height];
	image2D = new RGBAColor *[height];
	for (int i=0;i<height;i++)
		image2D[i] = image1D+i*width;

	float *ptr = data;
	for (int i=0; i < height; i++) 
	{
		for (int j=0; j < width; j++) 
		{
			image2D[i][j] = Color( scale*ptr[0], scale*ptr[1], scale*ptr[2] );
			ptr+=3;
		}
    }

	free( data );
}

void Texture::LoadCSV( char *filename, float scale )
{
	float fData;
	float fTexRange[4];
	int i, j, count=0, success=0;

	if (scale != 1.0f)
		printf("Warning: Texture::LoadCSV() only supports a scale of 1!\n");

	FILE *fDataFile = fopen( filename, "rb" );
	if (!fDataFile)
	{
		printf("Unknown error while loading '%s'!\n", filename);
		exit(0);
	}
	
	fscanf( fDataFile, "%d", &width );
	fscanf( fDataFile, "%d", &height );
	fscanf( fDataFile, "%f", &fTexRange[0] );
	fscanf( fDataFile, "%f", &fTexRange[1] );
	fscanf( fDataFile, "%f", &fTexRange[2] );
	fscanf( fDataFile, "%f", &fTexRange[3] );

	printf("%d %d %f %f %f %f\n", width, height, fTexRange[0], fTexRange[1], fTexRange[2], fTexRange[3] );

	// Allocate memory
	image1D = new RGBAColor[width*height];
	image2D = new RGBAColor *[height];
	for (int i=0;i<height;i++)
		image2D[i] = image1D+i*width;

	for (j=0; j<height; j++)
	{
		for (i=0; i<width-1; i++)
		{
			fscanf( fDataFile, "%f,", &fData );
			image2D[j][i] = Color( fData, fData, fData );
		}
		fscanf( fDataFile, "%f", &fData );
		image2D[j][width-1] = Color( fData, fData, fData );
	}

	fclose( fDataFile );
}


Texture::Texture(FILE *f, Scene *s) 
{
	printf("Texture:Texture() not implemented for reading from scene files.\n");
	exit(0);
}


Texture::~Texture()
{

}


Color Texture::IndexTextureAt(float x, float y) const
{
	float w, h; 

	switch( wrap_u ) {
		case TEXTURE_REPEAT:	w = Repeat( x*(width-1), width-1 );		break;
		case TEXTURE_CLAMP:		w = Clamp( x*(width-1), width-1 );		break;	
		case TEXTURE_MIRROR:	w = Mirror( x*(width-1), width-1 );		break;	
	};
	switch( wrap_v ) {
		case TEXTURE_REPEAT:	h = Repeat( y*(height-1), height-1 );	break;
		case TEXTURE_CLAMP:		h = Clamp( y*(height-1), height-1 );	break;	
		case TEXTURE_MIRROR:	h = Mirror( y*(height-1), height-1 );	break;	
	};

	if (w<1||w>width-2||h<1||h>height-2)
		return image2D[(int)h][(int)w];

	float facW = w-floor(w);
	float facH = h-floor(h);
	return (image2D[(int)h][(int)w]*(1-facH)+image2D[1+(int)h][(int)w]*facH)*(1-facW) +
		   (image2D[(int)h][1+(int)w]*(1-facH)+image2D[1+(int)h][1+(int)w]*facH)*facW;
}

float Texture::AlphaAt( float x, float y ) const
{
	int w, h; 

	switch( wrap_u ) {
		case TEXTURE_REPEAT:	w = (int)Repeat( x*(width-1), width-1 );	break;
		case TEXTURE_CLAMP:		w = (int)Clamp( x*(width-1), width-1 );		break;	
		case TEXTURE_MIRROR:	w = (int)Mirror( x*(width-1), width-1 );	break;	
	};
	switch( wrap_v ) {
		case TEXTURE_REPEAT:	h = (int)Repeat( y*(height-1), height-1 );	break;
		case TEXTURE_CLAMP:		h = (int)Clamp( y*(height-1), height-1 );	break;	
		case TEXTURE_MIRROR:	h = (int)Mirror( y*(height-1), height-1 );	break;	
	};

	return image2D[h][w].Alpha();
}


void Texture::Save(char* filename, float gamma)
{
	FILE *file;
	int count=0;
	int r, g, b;

	file = fopen( filename, "w" );
	if (!file) 
		{ 
			printf( "Error: Unable to write to %s!\n", filename ); 
			return; 
		}

	fprintf( file, "P3\n# Image output from Image::save()!\n%d %d\n%d\n", width, height, 255 );

	for (int j=height-1;j>=0;j--)
		for (int i=0;i<width;i++)
		{
			float tmp = pow( image2D[j][i].Red(), 1.0f/gamma );
			r = (int)(tmp*255);
			tmp = pow( image2D[j][i].Green(), 1.0f/gamma );
			g = (int)(tmp*255);
			tmp = pow( image2D[j][i].Blue(), 1.0f/gamma );
			b = (int)(tmp*255);
			r = ( r>255 ? 255 : (r<0 ? 0 : r) );
			g = ( g>255 ? 255 : (g<0 ? 0 : g) );
			b = ( b>255 ? 255 : (b<0 ? 0 : b) );
			fprintf( file, "%d %d %d ", r, g, b );
			if ((++count % 5) == 0)
				fprintf( file, "\n"); 
		}

	fclose( file );
}


float Texture::Clamp( float coord, int max ) const
{
	// Pretty simple, check if < 0, then check if > max
	coord = (coord < 0 ? 0 : coord );
	return (coord > max ? max : coord );
}

float Texture::Repeat( float coord, int max ) const
{
	// Coord could be arbitrarily far from the range 0..max, so we'll push it into the
	//    range 0..max if it's positive and -max..0 if negative.  Then if it's negative,
	//    we add max to bring into the range 0..max
	int div = (int)(coord / max);
	float clamp_negMax_to_max = coord - div * max;
	float zero_to_max = clamp_negMax_to_max < 0 ? clamp_negMax_to_max + max : clamp_negMax_to_max;
	return zero_to_max;
}

float Texture::Mirror( float coord, int max ) const
{
	// Same as above, only when we count how many units of "max" the coordinate is from 
	//    the origin, we use the oddity (i.e., div&0x1) to determine if it's in a 'mirrored'
	//    or standard coordinate range
	int div = (int)(coord / max);
	float clamp_negMax_to_max = coord - div * max;
	float zero_to_max = clamp_negMax_to_max < 0 ? clamp_negMax_to_max + max : clamp_negMax_to_max;
	return (div & 0x1 ? max-zero_to_max : zero_to_max);
}

