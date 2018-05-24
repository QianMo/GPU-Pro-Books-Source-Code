#include <stdafx.h>
#include <Demo.h>
#include <OGL_Texture.h>

static const OGL_TexFormat oglTexFormats[] = 
{
  0, 0, 0,
  GL_ALPHA8, GL_ALPHA, GL_UNSIGNED_BYTE,
  GL_R8, GL_LUMINANCE, GL_UNSIGNED_BYTE,
  GL_RG8, GL_LUMINANCE_ALPHA, GL_UNSIGNED_BYTE,
  GL_RGB8, GL_BGR, GL_UNSIGNED_BYTE,
  GL_SRGB8, GL_BGR, GL_UNSIGNED_BYTE,
  GL_RGBA8,	GL_BGRA, GL_UNSIGNED_BYTE,
  GL_SRGB8_ALPHA8,	GL_BGRA, GL_UNSIGNED_BYTE,
  GL_RGB8, GL_RGB, GL_UNSIGNED_BYTE,
  GL_RGB32UI, GL_RGB, GL_UNSIGNED_INT,
  GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE,
  GL_SRGB8_ALPHA8, GL_RGBA, GL_UNSIGNED_BYTE,
  GL_RGBA32UI, GL_RGBA, GL_UNSIGNED_INT, 
  GL_R16, GL_LUMINANCE, GL_UNSIGNED_SHORT,
  GL_RG16, GL_LUMINANCE_ALPHA, GL_UNSIGNED_SHORT,
  GL_R32UI, GL_LUMINANCE, GL_UNSIGNED_INT,
  GL_RGB16, GL_RGB, GL_UNSIGNED_SHORT,
  GL_RGBA16, GL_RGBA, GL_UNSIGNED_SHORT,
  GL_RGB16F, GL_RGB, GL_HALF_FLOAT,
  GL_RGBA16F, GL_RGBA, GL_HALF_FLOAT,
  GL_RGB32F, GL_RGB, GL_FLOAT,
  GL_RGBA32F, GL_RGBA, GL_FLOAT,
  GL_R16F, GL_LUMINANCE, GL_HALF_FLOAT,
  GL_RG16F, GL_LUMINANCE_ALPHA, GL_HALF_FLOAT,
  GL_R32F, GL_LUMINANCE, GL_FLOAT,
  GL_RG32F, GL_LUMINANCE_ALPHA, GL_FLOAT,
  GL_DU8DV8_ATI, GL_DUDV_ATI, GL_BYTE,
  0, 0, 0, // No UVWQ8 support
  0, 0, 0, // No UV16 support
  0, 0, 0, // No UVWQ16 support
  GL_DEPTH_COMPONENT16, GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT,
  GL_DEPTH24_STENCIL8, GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8,
  GL_R3_G3_B2, GL_RGB, GL_UNSIGNED_BYTE_3_3_2,
  GL_RGB5, GL_RGB, GL_UNSIGNED_SHORT_5_6_5,
  GL_RGB5_A1, GL_RGBA, GL_UNSIGNED_SHORT_5_5_5_1,
  GL_RGB10_A2, GL_RGBA, GL_UNSIGNED_INT_2_10_10_10_REV,
  0, 0, 0, // No mixed formats in OpenGL
  0, 0, 0,
  GL_COMPRESSED_RGB_S3TC_DXT1_EXT, 0, 0,
  GL_COMPRESSED_SRGB_S3TC_DXT1_EXT, 0, 0,
  GL_COMPRESSED_RGBA_S3TC_DXT3_EXT, 0, 0,
  GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT, 0, 0,
  GL_COMPRESSED_RGBA_S3TC_DXT5_EXT, 0, 0,
  GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT, 0, 0,
  0, 0, 0, // ATI1N not yet supported
  GL_COMPRESSED_LUMINANCE_ALPHA_3DC_ATI, 0, 0 // ATI2N
};


void OGL_Texture::Release()
{
  if(textureName > 0)
    glDeleteTextures(1, &textureName);  
}

bool OGL_Texture::LoadFromFile(const char *fileName)
{	
  strcpy(name, fileName);
  
  Image image;
  if(!image.Load(fileName))
    return false;

  const unsigned int width = image.GetWidth();
  const unsigned int height = image.GetHeight();
  const unsigned int depth = image.GetDepth();

  format = image.GetFormat();
  if(format == TEX_FORMAT_NONE)
    return false;

  const unsigned int numMipMaps = image.GetNumMipMaps();

  if(image.IsCube())
    target = GL_TEXTURE_CUBE_MAP;
  else if(image.Is3D())
    target = GL_TEXTURE_3D;
  else if(image.Is2D())
    target = GL_TEXTURE_2D;
  else
    target = GL_TEXTURE_1D;
  glGenTextures(1, &textureName);
  glBindTexture(target, textureName);

  for(unsigned int i=0; i<image.GetNumMipMaps(); i++)
  {
    if(image.IsCube())
    {
      unsigned int size = image.GetSize(i)/6;
      for(unsigned int j=0; j<6; j++)
      {
        if(image.IsCompressed())
        {
          glCompressedTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X+j, i, oglTexFormats[format].internalFormat,
           image.GetWidth(i), image.GetHeight(i), 0, size, image.GetData(i)+(j*size));
        }
        else
        {
          glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X+j, i, oglTexFormats[format].internalFormat, image.GetWidth(i), image.GetHeight(i),
           0, oglTexFormats[format].srcFormat, oglTexFormats[format].srcType, image.GetData(i)+(j*size));
        }
      }
    } 
    else if(image.Is3D())
    {
      if(image.IsCompressed())
      {
        glCompressedTexImage3D(GL_TEXTURE_3D, i, oglTexFormats[format].internalFormat, image.GetWidth(i), image.GetHeight(i),
         image.GetDepth(i), 0, image.GetSize(i), image.GetData(i));
      }
      else
      {
        glTexImage3D(GL_TEXTURE_3D, i, oglTexFormats[format].internalFormat, image.GetWidth(i), image.GetHeight(i), image.GetDepth(i), 0,
         oglTexFormats[format].srcFormat, oglTexFormats[format].srcType, image.GetData(i));
      }
    } 
    else if(image.Is2D())
    {
      if(image.IsCompressed())
      {
        glCompressedTexImage2D(GL_TEXTURE_2D, i, oglTexFormats[format].internalFormat, image.GetWidth(i),
         image.GetHeight(i), 0, image.GetSize(i), image.GetData(i));
      }
      else 
      {
        glTexImage2D(GL_TEXTURE_2D, i, oglTexFormats[format].internalFormat, image.GetWidth(i), image.GetHeight(i), 0,
         oglTexFormats[format].srcFormat, oglTexFormats[format].srcType, image.GetData(i));
      }
    } 
    else  
    {
      if(image.IsCompressed())
      {
        glCompressedTexImage1D(GL_TEXTURE_1D, i, oglTexFormats[format].internalFormat, image.GetWidth(i), 0,
         image.GetSize(i), image.GetData(i));
      }
      else
      {
        glTexImage1D(GL_TEXTURE_1D, i, oglTexFormats[format].internalFormat, image.GetWidth(i), 0,
         oglTexFormats[format].srcFormat, oglTexFormats[format].srcType, image.GetData(i));
      }
    }
  }

  return true;
}

bool OGL_Texture::CreateRenderable(unsigned int width, unsigned int height, unsigned int depth, texFormats format, unsigned int rtFlags)
{
  strcpy(name, "renderTargetTexture");
  this->format = format;

  glGenTextures(1, &textureName);
  if((rtFlags & CUBEMAP_RTF) == 0)
  {
    if(depth == 1)
    {
      target = GL_TEXTURE_2D;
      glBindTexture(target, textureName);
      glTexStorage2D(target, 1, oglTexFormats[format].internalFormat, width, height);
    }
    else
    {
      target = GL_TEXTURE_2D_ARRAY;
      glBindTexture(target, textureName);
      glTexStorage3D(target, 1, oglTexFormats[format].internalFormat, width, height, depth);
    }
  }
  else
  {
    if(depth == 1)
    {
      target = GL_TEXTURE_CUBE_MAP;
      glBindTexture(target, textureName);
      glTexStorage2D(target, 1, oglTexFormats[format].internalFormat, width, height);
    }
    else
    {
      target = GL_TEXTURE_CUBE_MAP_ARRAY;
      glBindTexture(target, textureName);
      glTexStorage3D(target, 1, oglTexFormats[format].internalFormat, width, height, depth*6);
    }
  }

  return true;
}

void OGL_Texture::Bind(textureBP bindingPoint) const
{
  glActiveTexture(GL_TEXTURE0+bindingPoint);
  glBindTexture(target, textureName);
}

OGL_TexFormat OGL_Texture::GetOglTexFormat(texFormats texFormat)
{
  return oglTexFormats[texFormat];
}





