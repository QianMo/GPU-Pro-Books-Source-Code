#include "EmptyTexture.h"
#include "GLError.h"

GLuint EmptyTexture::create2D(int width, int height,
                              GLint internalFormat, GLenum pixelFormat, GLenum pixelType,
                              GLint minFilter, GLint magFilter, GLvoid* data)
{
   GLuint tex;

	V(glGenTextures(1, &tex));
	V(glBindTexture(GL_TEXTURE_2D, tex));
	V(glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER, minFilter));
   V(glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER, magFilter));
	V(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
	V(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
	V(glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, width, height, 0, pixelFormat, pixelType, data));

	return tex;
}
GLuint EmptyTexture::create2DRect(int width, int height,
                              GLint internalFormat, GLenum pixelFormat, GLenum pixelType,
                              GLint minFilter, GLint magFilter, GLvoid* data)
{
   GLuint tex;

	V(glGenTextures(1, &tex));
	V(glBindTexture(GL_TEXTURE_RECTANGLE_ARB, tex));
	V(glTexParameteri(GL_TEXTURE_RECTANGLE_ARB,GL_TEXTURE_MIN_FILTER, minFilter));
   V(glTexParameteri(GL_TEXTURE_RECTANGLE_ARB,GL_TEXTURE_MAG_FILTER, magFilter));
	V(glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
	V(glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
	V(glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, internalFormat, width, height, 0, pixelFormat, pixelType, data));

	return tex;
}

void EmptyTexture::createMipmaps2D(int width, int height,
                                   int baseLevel, int maxLevel,
                                   GLint internalFormat, GLenum pixelFormat, GLenum pixelType)
{
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL,  maxLevel);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
   int div = 2;
   for(int level = 1; level <= maxLevel; level++)
   {
      glTexImage2D(GL_TEXTURE_2D, level, internalFormat, width/div, height/div, 0, pixelFormat, pixelType, 0);
      div = div << 1;
   }

}

GLuint EmptyTexture::create3D(int width, int height, int depth,
                              GLint internalFormat, GLenum pixelFormat, GLenum pixelType,
                              GLint minFilter, GLint magFilter)
{
   GLuint tex;
   glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_3D, tex);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, magFilter);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, minFilter);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE );
	V(glTexImage3D(GL_TEXTURE_3D, 0, internalFormat, width, height, depth, 0, pixelFormat, pixelType, 0));

   return tex;
}

GLuint EmptyTexture::create2DLayered(int width, int height, int layers,
                              GLint internalFormat, GLenum pixelFormat, GLenum pixelType,
                              GLint minFilter, GLint magFilter)
{
   GLuint tex;
   glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_2D_ARRAY_EXT, tex);
	glTexParameteri(GL_TEXTURE_2D_ARRAY_EXT, GL_TEXTURE_MAG_FILTER, magFilter);
	glTexParameteri(GL_TEXTURE_2D_ARRAY_EXT, GL_TEXTURE_MIN_FILTER, minFilter);
	glTexParameteri(GL_TEXTURE_2D_ARRAY_EXT, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
	glTexParameteri(GL_TEXTURE_2D_ARRAY_EXT, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
	//glTexParameteri(GL_TEXTURE_2D_ARRAY_EXT, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE );
	V(glTexImage3D(GL_TEXTURE_2D_ARRAY_EXT, 0, internalFormat, width, height, layers, 0, pixelFormat, pixelType, 0));

   return tex;
}

GLuint EmptyTexture::create1D(int width,
                              GLint internalFormat, GLenum pixelFormat, GLenum pixelType,
                              GLint minFilter, GLint magFilter,
                              GLvoid* data)
{
	GLuint tex;

	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_1D, tex);
	glTexParameteri(GL_TEXTURE_1D,GL_TEXTURE_MIN_FILTER, minFilter);
   glTexParameteri(GL_TEXTURE_1D,GL_TEXTURE_MAG_FILTER, magFilter);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	V(glTexImage1D(GL_TEXTURE_1D, 0, internalFormat, width, 0, pixelFormat, pixelType, data));

	return tex;

}

GLuint EmptyTexture::createCubeMap(int width, int height,
                                   GLint internalFormat, GLenum pixelFormat, GLenum pixelType,
                                   GLint minFilter, GLint magFilter)
{
   GLuint tex;

   V(glGenTextures(1, &tex));
   V(glBindTexture(GL_TEXTURE_CUBE_MAP, tex)); 

   V(glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
   V(glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
   V(glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE));

   V(glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, magFilter));
   V(glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, minFilter));

   V(glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X, 0, internalFormat, width, height, 0, pixelFormat, pixelType, 0));
   V(glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_X, 0, internalFormat, width, height, 0, pixelFormat, pixelType, 0));

   V(glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Y, 0, internalFormat, width, height, 0, pixelFormat, pixelType, 0));
   V(glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, 0, internalFormat, width, height, 0, pixelFormat, pixelType, 0));   

   V(glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Z, 0, internalFormat, width, height, 0, pixelFormat, pixelType, 0));
   V(glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Z, 0, internalFormat, width, height, 0, pixelFormat, pixelType, 0));

   return tex;
}

void EmptyTexture::setComparisonModesShadow(GLenum target)
{
   // set comparison modes (also needed for shaders)
   // compare texture value with r-component of tex-coord
   glTexParameteri(target, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_R_TO_TEXTURE); 
   // set comparison: (r/q <= texture (s/q, t/q ) ) ? 1: 0
   glTexParameteri(target, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
   GLfloat white[4] = {1, 1, 1, 1};
   GLfloat black[4] = {0, 0, 0, 1};
   setClampToBorder(target, 0, 0, 0, 1); // black

}

void EmptyTexture::setClampToBorder(GLenum target, float borderColorR, float borderColorG, float borderColorB, float borderColorA)
{
   GLfloat color[4] = {borderColorR, borderColorG, borderColorB, borderColorA};
   glTexParameterfv(target, GL_TEXTURE_BORDER_COLOR, color);
   glTexParameteri(target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
   glTexParameteri(target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
}