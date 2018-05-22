#pragma once

#include <FreeImage.h>
#include <math.h>
#include <map>

#include "cglibdefines.h"

#define IMAGE_STATUS_OK        1
#define IMAGE_STATUS_ERROR     0

using namespace std;

class Texture2D
{
public:
	Texture2D (void);
	Texture2D (int width, int height, int bitsperchannel, int channels);
	Texture2D (char *filename, bool tiling = true, bool mipmap = true, bool smooth = true, bool compression = false);
	~Texture2D (void);

	int load (char *filename, bool tiling = true, bool mipmap = true, bool smooth = true, bool compression = false);
	int save (char *filename);
	void build ();
	void erase ();

	unsigned char * getDataPtr ()       { return (image_type==FIT_BITMAP) ? FreeImage_GetBits (bitmap) : NULL; }
	int             getWidth ()         { return size[0]; }
	int             getHeight ()        { return size[1]; }
	float           getPageWidth ()     { return pagesize[0]; }
	float           getPageHeight ()    { return pagesize[1]; }
	int             getDpiX ()          { return (int) floor (dpi[0]); }
	int             getDpiY ()          { return (int) floor (dpi[1]); }
	int             getBytesPerPixel () { return (int) bpp; }
	int             getBitsPerPixel ()  { return (int) bitspp; }
	char          * getFileName ()      { return fileName; }
	unsigned int    getID ()            { return id; }
	unsigned int    getNumComponents () { return components; }

	void            setTiling (int s, int t);

	void            setAnisotropy (int);
	int             getAnisotropy ()    { return anisotropy; }
	int             getMaxAnisotropicFilter ();

    void dump (void)
    {
        fprintf (stdout, "\tTexture (0x%x) \"%s\" info:\n",
                (unsigned int) getDataPtr (), getFileName ());
        fprintf (stdout, "\t\tid     : %d\n", getID ());
        fprintf (stdout, "\t\twidth  : %d\n", getWidth ());
        fprintf (stdout, "\t\theight : %d\n", getHeight ());
        fprintf (stdout, "\t\tcomps  : %d\n", getNumComponents ());
        fprintf (stdout, "\t\tbyt pp : %d\n", getBytesPerPixel ());
        fprintf (stdout, "\t\tbit pp : %d\n", getBitsPerPixel ());
        fflush  (stdout);
    }

protected:
	FREE_IMAGE_FORMAT   format;
	FREE_IMAGE_TYPE     image_type;
	FREE_IMAGE_COLOR_TYPE color_type;
	FIBITMAP          * bitmap;
	unsigned char       bpp;
	unsigned char       bitspp;
	unsigned char       ordering;
	char              * fileName;
	int                 size[2];
	float               dpi[2];
	float               pagesize[2];
	unsigned int        id;
	unsigned int        components;
	bool                tiling,
                        mipmap,
                        smooth,
                        compression;
	int                 anisotropy;
	int                 maxAnisotropicFilter;
};

class TextureManager3D
{
private:
	TextureManager3D ();
	~TextureManager3D ();
	static map <char*, Texture2D *> textures;
	static TextureManager3D *instance;

public:
	static TextureManager3D *getInstance ();
	static int loadTexture (char *filename, bool tiling = true, 
		                    bool mipmap = true, bool smooth = true, bool compression = false);
	static void addTexture (Texture2D *tex);
	static int  getTexture (char *filename);
	static Texture2D * getFullTexture (char *filename);
	static bool hasTexture (Texture2D *tex);
	static bool hasTexture (char *filename);
	static void removeTexture (char *filename);
	static void removeTexture (Texture2D *tex);
	static int  getNumTextures ();
	static long getMemoryUsed ();
	static void init ();
	static void cleanup ();
};

