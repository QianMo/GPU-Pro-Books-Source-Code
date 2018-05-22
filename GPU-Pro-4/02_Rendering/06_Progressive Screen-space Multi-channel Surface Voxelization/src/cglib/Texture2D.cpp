
#include <string.h>
#include <stdlib.h>

#ifdef WIN32
    #include <GL/glew.h>
#else
    // this should be defined before glext.h
    #define GL_GLEXT_PROTOTYPES

    #include <GL/gl.h>
    #include <GL/glu.h>
    #include <GL/glext.h>
#endif

#include "Texture2D.h"
#include "cglibdefines.h"

extern int getPowerOfTwo (int val);

Texture2D::Texture2D (void)
{
	bitmap = NULL;
	fileName = NULL;
	size[0] = size[1] = 0;
	components = bpp = 3;
	tiling = mipmap = smooth = true;
	compression = false;
	id = 0;
}

Texture2D::Texture2D (char *filename, bool tiling, bool mipmap, bool smooth, bool compression)
{
	Texture2D ();

	this->tiling = tiling;
	this->mipmap = mipmap;
	this->smooth = smooth;
	this->compression = compression;
	
	if (load (filename)==IMAGE_STATUS_OK)
		build ();
	else
	{
		EAZD_TRACE ("Texture2D::load() : ERROR - Could not load image file \"" << filename << "\".");
		EAZD_ASSERTALWAYS (false);
	}
}

Texture2D::Texture2D (int width, int height, int bitsperchannel, int channels)
{
	Texture2D ();

	bitmap = FreeImage_Allocate (width, height, bitsperchannel*channels);
	format = FIF_BMP;
	image_type = FIT_BITMAP;
	switch (channels)
	{
	case 1: color_type = FIC_MINISBLACK; break;
	case 3: color_type = FIC_RGB; break;
	case 4: color_type = FIC_RGBALPHA; break;
	default: color_type = FIC_RGB;
	}
	components = channels;
	bpp = bitsperchannel*channels/8;
	size[0] = width;
	size[1] = height;
	dpi[0] = 72;
	dpi[1] = 72;
	build ();
}

Texture2D::~Texture2D (void)
{
	if (bitmap)
		FreeImage_Unload (bitmap);
	if (fileName)
		free (fileName);
    if (glIsTexture (id))
		glDeleteTextures (1, &id);
}

int Texture2D::save (char *filename)
{
	if (!bitmap)
		return IMAGE_STATUS_OK;
	
	format = FreeImage_GetFIFFromFilename (filename);
	if (format == FIF_UNKNOWN)
		return IMAGE_STATUS_ERROR;
	
	if (fileName)
		free (fileName);
	fileName = STR_DUP (filename);
	
	if (image_type == FIT_BITMAP)
	{
		if (FreeImage_FIFSupportsWriting (format) &&
			FreeImage_FIFSupportsExportBPP (format, bitspp))
		{
			FreeImage_Save (format, bitmap, filename);
		}
	}
	else
	{
		if (FreeImage_FIFSupportsExportType (format, image_type))
			FreeImage_Save (format, bitmap, filename);
	}
	return IMAGE_STATUS_OK;
}

int Texture2D::load (char *filename, bool tiling, bool mipmap, bool smooth, bool compression)
{
	fileName = STR_DUP (filename);
	
	format = FreeImage_GetFileType (filename);
	if (format == FIF_UNKNOWN)
		return IMAGE_STATUS_ERROR;
	
	bitmap = FreeImage_Load (format, filename);
	if (! bitmap)
		return IMAGE_STATUS_ERROR;

	image_type = FreeImage_GetImageType (bitmap);
	bitspp = FreeImage_GetBPP (bitmap);
	bpp = bitspp/8;
	components = bpp;
	size[0] = FreeImage_GetWidth (bitmap);
	size[1] = FreeImage_GetHeight (bitmap);
	dpi[0] = FreeImage_GetDotsPerMeterX (bitmap)/39.37f;
	dpi[1] = FreeImage_GetDotsPerMeterY (bitmap)/39.37f;
	color_type = FreeImage_GetColorType (bitmap);

	return IMAGE_STATUS_OK;
}

void Texture2D::build ()
{
	EAZD_ASSERTALWAYS (bitmap);

    if (mipmap && compression)
		EAZD_ASSERTALWAYS (false);

    if (glIsTexture (id))
		glDeleteTextures (1, &id);
	glGenTextures (1, &id);
	glBindTexture (GL_TEXTURE_2D, id);
	glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, tiling ? GL_REPEAT : GL_CLAMP);
	glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, tiling ? GL_REPEAT : GL_CLAMP);
	glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, mipmap ? GL_LINEAR_MIPMAP_LINEAR : GL_LINEAR);
	glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, smooth ? GL_LINEAR : GL_NEAREST);

	unsigned char * bits = getDataPtr ();
	EAZD_ASSERTALWAYS (bits);

#if 0
    // if the image is already compressed on disk we have to use glCompressedTexImage2D
    if (image is compressed)
    {
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

#ifdef WIN32
        if (GL_TRUE != glewGetExtension ("GL_SGIS_generate_mipmap"))
#else
        if (strstr ((const char *) glGetString (GL_EXTENSIONS), "GL_SGIS_generate_mipmap") == NULL)
#endif
        {
            // TODO:
            // this is wrong because if we reading a compressed image file from
            // disk then it is most probably a dds file which inherently stores
            // its mipmaps in the file. It is just a matter of accessing them.

            glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST);
            glTexParameteri (GL_TEXTURE_2D, GL_GENERATE_MIPMAP_SGIS, GL_TRUE);
        }

        int image_size = ((size[0]+3)/4)*((size[1]+3)/4)*16;
        if (pixelFormat == GL_COMPRESSED_RGBA_S3TC_DXT1_EXT)
            image_size /= 2;

        glCompressedTexImage2DARB (GL_TEXTURE_2D, 0,
                pixelFormat, size[0], size[1], 0, image_size, bits);
    } // if image is compressed

    // otherwise load it in the reqular manner
    else
    {
#endif
        // TODO:
        // the internalFormat and pixelFormat should not be set explicitly
        // but read from the image structure to be more accurate

        if (mipmap)
        {
            gluBuild2DMipmaps (GL_TEXTURE_2D, bpp, size[0], size[1],
				(bpp==3) ? GL_BGR : (bpp==4) ? GL_BGRA_EXT:GL_INTENSITY,
                               GL_UNSIGNED_BYTE, bits);
        }
        else if (compression)
        {
            glTexImage2D (GL_TEXTURE_2D, 0,
				(bpp==3) ? GL_COMPRESSED_RGB : (bpp==4) ? GL_COMPRESSED_RGBA:GL_COMPRESSED_INTENSITY,
                          size[0], size[1], 0, (bpp==3) ? GL_BGR : GL_BGRA_EXT,
                          GL_UNSIGNED_BYTE, bits);

            // check if compression took place
            int compressed;
            // request a generic compression method
            glGetTexLevelParameteriv (GL_TEXTURE_2D, 0, GL_TEXTURE_COMPRESSED_ARB, &compressed);
            if (compressed == GL_TRUE)
            {
                GLint internalFormat;
                // get the actual compression method used
                glGetTexLevelParameteriv (GL_TEXTURE_2D, 0, GL_TEXTURE_INTERNAL_FORMAT, &internalFormat);
#if 0
                // read back the texture to save it
                GLint compressed_size;
                glGetTexLevelParameteriv (GL_TEXTURE_2D, 0, GL_TEXTURE_COMPRESSED_IMAGE_SIZE_ARB, &compressed_size);
                unsigned char * img = (unsigned char *) malloc (compressed_size * sizeof (unsigned char));
                glGetCompressedTexImageARB (GL_TEXTURE_2D, 0, img);
#endif
            }
            else
                EAZD_PRINT ("Texture2D::build() : INFO - Texture \"" << getFileName () << "\" was not compressed.");
        }
        else
        {
            glTexImage2D (GL_TEXTURE_2D, 0, (bpp==3) ? GL_RGB8 : GL_RGBA8,
                      size[0], size[1], 0, (bpp==3) ? GL_BGR : GL_BGRA_EXT,
                      GL_UNSIGNED_BYTE, bits);
        }
#if 0
    }
#endif
}

void Texture2D::setTiling (int s, int t)
{
    if (glIsTexture (id))
    {
        glBindTexture   (GL_TEXTURE_2D, id);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, s);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, t);
    }
    else
		EAZD_TRACE ("Texture2D::setTiling() : ERROR - Texture has not been created from image file \"" << fileName << "\".");
}

int Texture2D::getMaxAnisotropicFilter ()
{
    static bool initialized = false;

    if (! initialized)
    {
        glGetIntegerv (GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT,
                       &maxAnisotropicFilter);
        initialized = true;
    }

    return maxAnisotropicFilter;
}

void Texture2D::setAnisotropy (int level)
{
    EAZD_ASSERTALWAYS (level);

    if (glIsTexture (id))
    {
        anisotropy = CLAMP (getPowerOfTwo (level), 0, getMaxAnisotropicFilter ());

        glBindTexture (GL_TEXTURE_2D, id);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, anisotropy);
    }
    else
		EAZD_TRACE ("Texture2D::setAnisotropy() : ERROR - Texture has not been created from image file \"" << fileName << "\".");
}

void Texture2D::erase ()
{
	if (bitmap)
	{
		FreeImage_Unload (bitmap);
		bitmap = NULL;
	}
}

//---------------------------------------------------------------

TextureManager3D::TextureManager3D ()
{
}

TextureManager3D::~TextureManager3D ()
{
	map <char *, Texture2D *>::const_iterator iter;

	for (iter=textures.begin (); iter != textures.end (); ++iter)
		delete iter->second;
	textures.clear ();
}

TextureManager3D * TextureManager3D::getInstance ()
{
	init ();
	return instance;
}

void TextureManager3D::init ()
{
	if (! instance)
    {
		instance = new TextureManager3D ();
/*
        // check if texture compression is supported
        GLint * compressed_formats;
        GLint   num_compressed_formats;

        glGetIntegerv (GL_NUM_COMPRESSED_TEXTURE_FORMATS, &num_compressed_formats);
        EAZD_PRINT ("TextureManager3D::init() : INFO - Number of compressed texture formats supported: " << num_compressed_formats);

        compressed_formats = (GLint *) malloc (num_compressed_formats *
                                               sizeof (GLint));
        glGetIntegerv (GL_COMPRESSED_TEXTURE_FORMATS, compressed_formats);
        for (int i = 0; i < num_compressed_formats; i++)
        {
            if (compressed_formats[i] == GL_COMPRESSED_RGB)
			{
				EAZD_PRINT ("\tGL_COMPRESSED_RGB");
			}
            else if (compressed_formats[i] == GL_COMPRESSED_RGBA)
			{
                EAZD_PRINT ("\tGL_COMPRESSED_RGBA");
			}
            else if (compressed_formats[i] == GL_COMPRESSED_ALPHA)
			{
				EAZD_PRINT ("\tGL_COMPRESSED_ALPHA");
			}
            else if (compressed_formats[i] == GL_COMPRESSED_LUMINANCE)
			{
				EAZD_PRINT ("\tGL_COMPRESSED_LUMINANCE");
			}
            else if (compressed_formats[i] == GL_COMPRESSED_LUMINANCE_ALPHA)
			{
				EAZD_PRINT ("\tGL_COMPRESSED_LUMINANCE_ALPHA");
			}
            else if (compressed_formats[i] == GL_COMPRESSED_INTENSITY)
			{
				EAZD_PRINT ("\tGL_COMPRESSED_INTENSITY");
			}
            else if (compressed_formats[i] == GL_COMPRESSED_RGB_S3TC_DXT1_EXT)
			{
                EAZD_PRINT ("\tGL_COMPRESSED_RGB_S3TC_DXT1_EXT");
			}
            else if (compressed_formats[i] == GL_COMPRESSED_RGBA_S3TC_DXT1_EXT)
			{
                EAZD_PRINT ("\tGL_COMPRESSED_RGBA_S3TC_DXT1_EXT");
			}
            else if (compressed_formats[i] == GL_COMPRESSED_RGBA_S3TC_DXT3_EXT)
			{
                EAZD_PRINT ("\tGL_COMPRESSED_RGBA_S3TC_DXT3_EXT");
			}
            else if (compressed_formats[i] == GL_COMPRESSED_RGBA_S3TC_DXT5_EXT)
			{
                EAZD_PRINT ("\tGL_COMPRESSED_RGBA_S3TC_DXT5_EXT");
			}
            else
			{
                EAZD_PRINT ("\tOther compression format (0x" << hex << compressed_formats[i] << dec << ") used.");
			}
        }
*/
        const GLubyte *str = glGetString (GL_EXTENSIONS);

        if (! ((strstr ((const char *) str, "GL_ARB_texture_compression")      != NULL) &&
               (strstr ((const char *) str, "GL_EXT_texture_compression_s3tc") != NULL)
            ))
        {
            EAZD_TRACE ("TextureManager3D::init() : ERROR - Texture Compression is not supported");
        }
    }
}

int TextureManager3D::loadTexture (char *filename, bool tiling,
                                   bool mipmap, bool smooth, bool compression)
{
	map <char *, Texture2D *>::const_iterator iter;
	iter = textures.find (filename);
	if (iter!=textures.end ())
		return iter->second->getID ();
	else
	{
		Texture2D * tex = new Texture2D (filename, tiling, mipmap, smooth, compression);
		textures.insert (textures.begin (), pair <char *, Texture2D *> (filename, tex));
		
		if (tex==NULL)
			return 0;
		else
			return tex->getID ();
	}
}

void TextureManager3D::addTexture (Texture2D *tex)
{
	map <char *, Texture2D *>::const_iterator iter;
	iter = textures.find (tex->getFileName ());
	if (iter==textures.end ())
   	    textures.insert (textures.begin (), pair <char *, Texture2D *> (tex->getFileName (), tex));
}

int TextureManager3D::getTexture (char *filename)
{
	map <char *, Texture2D *>::const_iterator iter;
	iter = textures.find (filename);
	if (iter!=textures.end ())
		return iter->second->getID ();
	else
		return 0;
}

Texture2D * TextureManager3D::getFullTexture (char *filename)
{
	map <char *, Texture2D *>::const_iterator iter;
	iter = textures.find (filename);
	if (iter!=textures.end ())
		return iter->second;
	else
		return NULL;
}

bool TextureManager3D::hasTexture (char *filename)
{
	map <char *, Texture2D *>::const_iterator iter;
	iter = textures.find (filename);
	if (iter!=textures.end ())
		return true;
	else
		return false;
}
	
bool TextureManager3D::hasTexture (Texture2D *tex)
{
	return hasTexture (tex->getFileName ());
}

int TextureManager3D::getNumTextures ()
{
	return textures.size ();
}

long TextureManager3D::getMemoryUsed ()
{
	long nbytes = 0;
	map <char *, Texture2D *>::const_iterator iter;
	for (iter=textures.begin (); iter != textures.end (); ++iter)
		nbytes+= iter->second->getBytesPerPixel ()*
		         iter->second->getWidth ()*
				 iter->second->getHeight ();
	return nbytes;
}

void TextureManager3D::removeTexture (char *filename)
{
	textures.erase (filename);
}

void TextureManager3D::removeTexture (Texture2D *tex)
{
	removeTexture (tex->getFileName ());
}

void TextureManager3D::cleanup ()
{
	map <char *, Texture2D *>::const_iterator iter;
	for (iter=textures.begin (); iter != textures.end (); ++iter)
		iter->second->erase ();
}

map <char*, Texture2D *> TextureManager3D::textures = map <char*, Texture2D *> ();
TextureManager3D *TextureManager3D::instance = NULL;

