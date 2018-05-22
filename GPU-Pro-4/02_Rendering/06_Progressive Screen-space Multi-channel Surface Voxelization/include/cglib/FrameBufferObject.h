// 
// Copyright (c) 2005, 
// Aaron Lefohn	(lefohn@cs.ucdavis.edu)
// Robert Strzodka (strzodka@stanford.edu)
// Adam Moerschell (atmoerschell@ucdavis.edu)
// All rights reserved.
// 
// This software is licensed under the BSD open-source license. See
// http://www.opensource.org/licenses/bsd-license.php for more detail.
// 
////////////////////////////////////////////////////////////////////////////////

#ifndef _FRAMEBUFFER_OBJECT_H
#define _FRAMEBUFFER_OBJECT_H

#include <iostream>
#include <GL/glew.h>

#define GL_TEXTURE_LAYER 98765

/*!
FramebufferObject Class. This class encapsulates the FramebufferObject
(FBO) OpenGL spec. See the official spec at:
	http://oss.sgi.com/projects/ogl-sample/registry/EXT/framebuffer_object.txt

for details.

A framebuffer object (FBO) is conceptually a structure containing pointers
to GPU memory. The memory pointed to is either an OpenGL texture or an
OpenGL RenderBuffer. FBOs can be used to render to one or more textures,
share depth buffers between multiple sets of color buffers/textures and
are a complete replacement for pbuffers.

Performance Notes:
  1) It is more efficient (but not required) to call Bind ()
     on an FBO before making multiple method calls. For example:
		
      FramebufferObject fbo;
      fbo.Bind ();
      fbo.AttachTexture (GL_TEXTURE_2D, texId0, GL_COLOR_ATTACHMENT0);
      fbo.AttachTexture (GL_TEXTURE_2D, texId1, GL_COLOR_ATTACHMENT1);
      fbo.IsValid ();

    To provide a complete encapsulation, the following usage
    pattern works correctly but is less efficient:

      FramebufferObject fbo;
      // NOTE : No Bind () call
      fbo.AttachTexture (GL_TEXTURE_2D, texId0, GL_COLOR_ATTACHMENT0);
      fbo.AttachTexture (GL_TEXTURE_2D, texId1, GL_COLOR_ATTACHMENT1);
      fbo.IsValid ();

    The first usage pattern binds the FBO only once, whereas
    the second usage binds/unbinds the FBO for each method call.

  2) Use FramebufferObject::Disable () sparingly. We have intentionally
     left out an "Unbind ()" method because it is largely unnecessary
     and encourages rendundant Bind/Unbind coding. Binding an FBO is
     usually much faster than enabling/disabling a pbuffer, but is
     still a costly operation. When switching between multiple FBOs
     and a visible OpenGL framebuffer, the following usage pattern 
     is recommended:

      FramebufferObject fbo1, fbo2;
      fbo1.Bind ();
        ... Render ...
      // NOTE : No Unbind/Disable here...

      fbo2.Bind ();
        ... Render ...

      // Disable FBO rendering and return to visible window
      // OpenGL framebuffer.
      FramebufferObject::Disable ();
*/

class FramebufferObject
{
public:
	/// Ctor/Dtor
	FramebufferObject ();
	virtual ~FramebufferObject ();

	/// Bind this FBO as current render target
	void Bind ();

	/// Bind a texture to the "attachment" point of this FBO
	virtual void AttachTexture (GLenum texTarget, 
		GLuint texId,
		GLenum attachment = GL_COLOR_ATTACHMENT0,
		int mipLevel      = 0,
		int zSlice        = 0);

	/// Bind an array of textures to multiple "attachment" points of this FBO
	///  - By default, the first 'numTextures' attachments are used,
	///    starting with GL_COLOR_ATTACHMENT0
	virtual void AttachTextures (int numTextures, 
		GLenum texTarget[], 
		GLuint texId[],
		GLenum attachment[] = NULL,
		int mipLevel[]      = NULL,
		int zSlice[]        = NULL);

	/// Bind a render buffer to the "attachment" point of this FBO
	virtual void AttachRenderBuffer (GLuint buffId,
		GLenum attachment = GL_COLOR_ATTACHMENT0);

	/// Bind an array of render buffers to corresponding "attachment" points
	/// of this FBO.
	/// - By default, the first 'numBuffers' attachments are used,
	///   starting with GL_COLOR_ATTACHMENT0
	virtual void AttachRenderBuffers (int numBuffers, GLuint buffId[],
		GLenum attachment[] = NULL);

	/// Free any resource bound to the "attachment" point of this FBO
	void Unattach (GLenum attachment);

	/// Free any resources bound to any attachment points of this FBO
	void UnattachAll ();

	/// Is this FBO currently a valid render target?
	///  - Sends output to std::cerr by default but can
	///    be a user-defined C++ stream
	///
	/// NOTE : This function works correctly in debug build
	///        mode but always returns "true" if NDEBUG is
	///        is defined (optimized builds)
#ifndef NDEBUG
	bool IsValid (std::ostream& ostr = std::cerr);
#else
	bool IsValid (std::ostream& ostr = std::cerr) { return true; }
#endif

	/// check if FBOs are supported
	static bool fboSupported ();

	/// BEGIN : Accessors

	/// Get the FBO ID
	GLuint GetID () { return m_fboId; }

	/// Is attached type GL_RENDERBUFFER or GL_TEXTURE?
	GLenum GetAttachedType (GLenum attachment);

	/// What is the Id of Renderbuffer/texture currently 
	/// attached to "attachement?"
	GLuint GetAttachedId (GLenum attachment);

	/// Which mipmap level is currently attached to "attachement?"
	GLint  GetAttachedMipLevel (GLenum attachment);

	/// Which cube face is currently attached to "attachment?"
	GLint  GetAttachedCubeFace (GLenum attachment);

	/// Which z-slice is currently attached to "attachment?"
	GLint  GetAttachedZSlice (GLenum attachment);

	/// END : Accessors

	/// BEGIN : Static methods global to all FBOs

	/// Return number of color attachments permitted
	static int GetMaxColorAttachments ();

	/// Disable all FBO rendering and return to traditional,
	/// windowing-system controlled framebuffer
	///  NOTE:
	///     This is NOT an "unbind" for this specific FBO, but rather
	///     disables all FBO rendering. This call is intentionally "static"
	///     and named "Disable" instead of "Unbind" for this reason. The
	///     motivation for this strange semantic is performance. Providing
	///     "Unbind" would likely lead to a large number of unnecessary
	///     FBO enablings/disabling.
	static void Disable ();

	/// END : Static methods global to all FBOs

protected:
	void  _GuardedBind ();
	void  _GuardedUnbind ();
	void  _FramebufferTextureND (GLenum attachment, GLenum texTarget, 
								 GLuint texId, int mipLevel, int zSlice);
	static GLuint _GenerateFboId ();
	static int m_fboSupported;

private:
	GLuint m_fboId;
	GLint  m_savedFboId;
};

#endif
