/************************************************
** framebufferObject.h                         **
** -------------------                         **
**                                             **
** This is the frame-work for general purpose  **
**   initialization of a framebuffer object,   **
**   as specified in the OpenGL extension:     **
**       GL_EXT_FRAMEBUFFER_OBJECT             **
**                                             **
** Since this is an OpenGL extension, not WGL, **
**   it should be much more portable (and      **
**   supposedly) faster than p-buffers and     **
**   render-to-texture.                        **
**                                             **
** Chris Wyman (4/27/2005)                     **
************************************************/

#ifndef ___FRAMEBUFFER_OBJECT_H
#define ___FRAMEBUFFER_OBJECT_H

#include "Utils/GLee.h"

#define FBO_DEPTH         0
#define FBO_STENCIL       10
#define FBO_COLOR(x)      (20+(x))
#define FBO_COLOR0        20
#define FBO_COLOR1        21
#define FBO_COLOR2        22
#define FBO_COLOR3        23
#define FBO_COLOR4        24
#define FBO_COLOR5        25
#define FBO_COLOR6        26
#define FBO_COLOR7        27


void AllowUnclampedColors();

class FrameBuffer
{
private:
	GLuint ID;         // from glGenFramebuffersEXT()
	GLuint *colorIDs;  // these are the GL texture IDs for the color buffers
	GLuint depthID;    // the GL texture ID for the depth buffer (_OR_ the depth buffer ID from glBindRenderbufferEXT)
	GLuint stencilID;  // the GL buffer ID for the stencil buffer (from glBindRenderbufferEXT)
	GLint maxColorBuffers;   // the max number of color buffer textures (i.e., size of colorIDs[] array)
	GLuint prevFrameBuf;     // a very primitive method for allowing nested BindBuffer() calls. (BE CAREFUL)
	int width, height;       // width & height of ALL textures/buffers in framebuffer object.  (MAY BE INCORRECT if used with the default Framebuffer() constructor!)
	char fbName[80];         // a simple ascii text name for the buffer -- useful for debugging

	GLenum *colorType;
	GLenum depthType, stencilType;
	unsigned int includedBuffers;
	int automaticMipmapsEnabled;
	int numColorAttachments;

	// Internal functions for saving PPM images/screencaptures
	void FrameToPGM( char *f, unsigned char *data, int width, int height );
	void FrameToPPM( char *f, unsigned char *data, int width, int height );

	// Test values for use with texture array FBOs.  These may be broken, and should not be exposed!
	int depth;
public:
	FrameBuffer( char *name=0 );                           // avoid the use of this constructor whenever possible
	FrameBuffer( int width, int height, char *name=0 );    // use this constructor if you want to setup the textures yourself
	FrameBuffer( GLenum type, int width, int height, int depth, 
				 GLuint colorBufType, int numColorBufs, int hasZbuf, 
		         bool enableAutomaticMipmaps, char *name=0 );
	FrameBuffer( int test );
	~FrameBuffer();

	// This is useful for debugging.
	//    It displays (in the currently active framebuffer [WARNING: CANNOT BE THIS ONE])
	//    the specified attachment as a full screen image (i.e., using gluOrtho2D()).
	//    This overwrites whatever was currently there.
	void DisplayAsFullScreenTexture( int attachment, bool blending, float lod=0.0f );  // attachment is one of the FBO_xxx #define's from above
	void DisplayAlphaAsFullScreenTexture( int attachment, float lod=0.0f );

	// Output images from the FBO to a file.
	void SaveColorImage( int attachment, char *filename );
	void SaveDepthImage( char *filename );
	void SaveStencilImage( char *filename );

	// Clears all the buffers in the framebuffer (based on the values currently set
	//     by fuctions like glClearColor(), etc).  This function simply calls glClear()
	//     with the appropriate bits for the buffers allocated in this FBO!
	inline void ClearBuffers( void ) { glClear( includedBuffers ); }

	// Functions to setup the appropriate buffers to draw into
	void DrawBuffers( int cBuf1, int cBuf2=0, int cBuf3=0, int cBuf4=0 );

	// Check to see if there are any errors.  Returns GL_FRAMEBUFFER_COMPLETE_EXT if OK.
	//    Prints messages to stdout if printMessage == 1
    GLenum CheckFramebufferStatus( int printMessage=0 );

	// Attach textures to various attachment points
	int AttachColorTexture( GLuint colorTexID, int colorBuffer=0 );
	int AttachDepthTexture( GLuint depthTexID );
	int AttachStencilTexture( GLuint stencilTexID );  // CURRENTLY UNSUPPORTED. DO NOT USE!

	// Attach renderbuffers to various attachment points
	//    (note these SHOULD replace textures, but it may not be guaranteed,
	//     so you might want to unbind textures before binding a renderbuffer)
	int AttachColorBuffer( GLuint colorBufID, int colorBuffer=0 );   // Use only if you know what you're doing!
	int AttachDepthBuffer( GLuint depthBufID );                      // Use only if you know what you're doing!
	int AttachStencilBuffer( GLuint stencilBufID );                  // Use only if you know what you're doing!

	// Functionality for drawing custom mipmap levels.
	void DrawToColorMipmapLevel( GLuint colorBuffer, GLuint level ); // Use only if you know what you're doing!
	void DoneDrawingMipmapLevels( GLuint colorBuffer );              // Use only if you know what you're doing!
	void TemporarilyUnattachAllBuffersExcept( GLuint colorBuffer );  // Use only if you know what you're doing!
	void ReattachAllBuffers( void );                                 // Use only if you know what you're doing!

	// attachment is one of the FBO_xxx #defined from above
	//   NOTE: This only works if automatic mipmapping has been enabled for this attachment! 
	//         (and OGL supports it -- though we try for all attachments colorBuf, depthBuf, stencilBuf)
	void AutomaticallyGenerateMipmaps( int attachment );            

	// Bind/unbind the current framebuffer.  These functions store the currently
	//    bound framebuffer during a BindBuffer() and rebind it upon an UnbindBuffer()
	//    (which allows for a very basic nesting of BindBuffer() calls)
	GLuint BindBuffer( void );   // Commands after this draw into THIS framebuffer.
	int UnbindBuffer( void );    // Commands after this draw into the normal framebuffer (e.g., screen)

	// Queries to return the texture/renderbuffer ID of the various attachments
	//
	//    You may thus use one of the color buffers as a GL texture using:
	//          glBindTexture( GL_TEXTURE_2D, framebuffer->GetColorTextureID( 0 ) );
	//    Beware you may NOT have this framebuffer bound (via BindBuffer()) at the 
	//    same time you're using one of the textures (via glBindTexture()).  This means
	//    when you're done using framebuffer->GetColorTextureID(0), you should call
	//          glBindTexture( GL_TEXTURE_2D, 0 );
	inline GLuint GetColorTextureID( int level=0 ) { return (level < maxColorBuffers && level >= 0 ? colorIDs[level] : -1); }
	inline GLuint GetDepthTextureID( void )        { return depthID; }
	inline GLuint GetStencilTextureID( void )      { return stencilID; }
	GLuint GetTextureID( int attachment );         // attachment is one of the FBO_xxx #define's frome above
	GLenum GetTextureType( int attachment );

	inline int GetNumColorTextures( void )         { return numColorAttachments; }

	// Plays with the texture parameters of various attachments in a painless way
	void SetAttachmentClamping( int attachment, GLint sWrap, GLint tWrap );
	void SetAttachmentFiltering( int attachment, GLint minFilter, GLint magFilter );

	// Gets some identifying information about the framebuffer object.
	inline int GetWidth( void )	const			   { return width; }      
	inline int GetHeight( void ) const			   { return height; }   
	inline float GetAspectRatio( void ) const	   { return width/(float)height; }
	inline GLuint GetBufferID( void ) const	       { return ID; }
	inline char *GetName( void )				   { return fbName; }

	// Allows you to set the size, particularly useful if you use the default constructor, 
	//      and want to set it up correctly afterwards!
	inline void SetSize( int newWidth, int newHeight ) { width = newWidth; height = newHeight; }

	// This resizes a framebuffer object that already exists and has textures
	//      already associated with it.
	void ResizeExistingFBO( int newWidth, int newHeight );
};



#endif