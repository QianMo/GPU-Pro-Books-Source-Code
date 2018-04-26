/******************************************************************/
/* Texture.h                                                      */
/* -----------------------                                        */
/*                                                                */
/* The file defines an image class that stores a texture.         */
/*     This is very similar to the image class, but also defines  */
/*     access patterns for the texture with interpolation.        */
/*                                                                */
/* Chris Wyman (10/26/2006)                                       */
/******************************************************************/

#ifndef TEXTURE_H
#define TEXTURE_H


#include <stdio.h>
#include <stdlib.h>
#include "DataTypes/Color.h"
#include "DataTypes/RGBAColor.h"
#include "Utils/TextParsing.h"
#include "Utils/ImageIO/imageIO.h"

class Scene;

#define TEXTURE_TYPE_UNKNOWN  0
#define TEXTURE_TYPE_PPM      1
#define TEXTURE_TYPE_RGB      2
#define TEXTURE_TYPE_RGBA     3
#define TEXTURE_TYPE_HDR      4
#define TEXTURE_TYPE_CSV      5
#define TEXTURE_TYPE_BMP      6

#define TEXTURE_REPEAT        0
#define TEXTURE_CLAMP         1
#define TEXTURE_MIRROR        2

class Texture
{
private:
	RGBAColor* image1D;   // image1D[i]    accesses location (i%width, i/width)
    RGBAColor** image2D;  // image2D[y][x] accesses location (x,y)
	char *fileName;
    int width, height, type;
	unsigned char interpolationMethod;
	unsigned char wrap_u, wrap_v;

	float Clamp( float coord, int max ) const;
	float Repeat( float coord, int max ) const;
	float Mirror( float coord, int max ) const;

	void LoadPPM( char *filename, float scale );
	void LoadRGB( char *filename, float scale );
	void LoadHDR( char *filename, float scale );
	void LoadCSV( char *filename, float scale );
public:
    Texture(int width, int height);
	Texture(char *filename, float scale=1.0);  // loads a PPM file
	Texture(FILE *f, Scene *s);
    ~Texture();

	// Get size of the texture
	inline int GetWidth() const  { return width;  }
	inline int GetHeight() const { return height; }

	inline void SetWrapS( int wrapMode )  { wrap_u = wrapMode; } 
	inline void SetWrapT( int wrapMode )  { wrap_v = wrapMode; } 

	inline bool HasAlpha( void ) const { return type == TEXTURE_TYPE_RGBA; }

	// Returns the value at (x,y) with no interpolation!
    inline RGBAColor& operator()(int x, int y) { return image2D[y][x]; }

	// Returns the value interpolated to (x,y)
	//   NOTE: this operator assumes x, y in [0..1] covers the entire texture!
	//         and NOT x in [0..width] and y in [0..height]
	Color IndexTextureAt(float x, float y) const;
	float AlphaAt( float x, float y ) const;

	// Assumes x & y are in [0..1] (not outside!)
	inline Color FastIndexTextureAt(float x, float y) const { return image2D[(int)(y*(height-1))][(int)(x*(width-1))]; }

	// This is going away soon.  Just for testing.
	void Save(char* filename, float gamma=1.0f);

	inline char *GetFilename( void ) { return fileName; }
};



#endif
