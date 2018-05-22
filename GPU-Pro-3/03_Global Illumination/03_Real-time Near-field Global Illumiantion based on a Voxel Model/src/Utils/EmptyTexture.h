#ifndef EMPTYTEXTURE_H
#define EMPTYTEXTURE_H

#include "OpenGL.h"

class EmptyTexture
{
public:

   /// Creates an empty OpenGL 2D Texture and returns its ID. 
   /// This ID is bound to the active texture slot.
   /// \param width: width of texture in pixels
   /// \param height: height of texture in pixels
   ///
   /// \param internalFormat: internal representation of the color components (GL_RGBA32F_ARB,...)
   /// \param pixelFormat: order of pixel data stored in the memory (GL_RGB, GL_RGBA, ...)
   /// \param pixelType: type of the  pixel data stored in the memory (GL_BYTE, GL_FLOAT, ...) 
   /// \return The texture's ID.
   ///
   static GLuint create2D(int width, int height,
      GLint internalFormat, GLenum pixelFormat, GLenum pixelType,
      GLint minFilter = GL_NEAREST, GLint magFilter = GL_NEAREST,
      GLvoid* data = 0);

   static GLuint create2DRect(int width, int height,
      GLint internalFormat, GLenum pixelFormat, GLenum pixelType,
      GLint minFilter = GL_NEAREST, GLint magFilter = GL_NEAREST,
      GLvoid* data = 0);


   /// Creates and attaches a set of mipmap levels to the currently bound texture.
   /// \param width: width of bound texture in pixels (= mipmap level 0)
   /// \param height: height of bound texture in pixels (= mipmap level 0)
   /// \param baseLevel: sets a user defined finest mipmap level
   /// \param maxLevel: sets a user defined coarsest mipmap level
   static void createMipmaps2D(int width, int height,
      int baseLevel, int maxLevel,
      GLint internalFormat, GLenum pixelFormat, GLenum pixelType);

   /// Creates an empty OpenGL 3D Texture and returns its ID. 
   /// This ID is bound to the active texture slot.
   /// \param width: width of texture in pixels
   /// \param height: height of texture in pixels
   /// \param depth: depth of texture in pixels
   ///
   /// \param internalFormat: internal representation of the color components (GL_RGBA32F_ARB,...)
   /// \param pixelFormat: order of pixel data stored in the memory (GL_RGB, GL_RGBA, ...)
   /// \param pixelType: type of the  pixel data stored in the memory (GL_BYTE, GL_FLOAT, ...) 
   /// \return The texture's ID.
   ///
   static GLuint create3D(int width, int height, int depth,
      GLint internalFormat, GLenum pixelFormat, GLenum pixelType,
      GLint minFilter = GL_NEAREST, GLint magFilter = GL_NEAREST);

   static GLuint create2DLayered(int width, int height, int layers,
      GLint internalFormat, GLenum pixelFormat, GLenum pixelType,
      GLint minFilter = GL_NEAREST, GLint magFilter = GL_NEAREST);

   /// Creates an empty OpenGL 1D Texture and returns its ID. 
   /// This ID is bound to the active texture slot.
   /// \param width: width of texture in pixels
   ///
   /// \param internalFormat: internal representation of the color components (GL_RGBA32F_ARB,...)
   /// \param pixelFormat: order of pixel data stored in the memory (GL_RGB, GL_RGBA, ...)
   /// \param pixelType: type of the  pixel data stored in the memory (GL_BYTE, GL_FLOAT, ...) 
   /// \return The texture's ID.
   ///
   static GLuint create1D(int width,
      GLint internalFormat, GLenum pixelFormat, GLenum pixelType,
      GLint minFilter = GL_NEAREST, GLint magFilter = GL_NEAREST,
      GLvoid* data = 0);


   static GLuint createCubeMap(int width, int height,
      GLint internalFormat, GLenum pixelFormat, GLenum pixelType,
      GLint minFilter = GL_NEAREST, GLint magFilter = GL_NEAREST);

   static void setComparisonModesShadow(GLenum target);

   static void setClampToBorder(GLenum target, float borderColorR, float borderColorG, float borderColorB, float borderColorA);


private:
   EmptyTexture(){}
};



#endif
