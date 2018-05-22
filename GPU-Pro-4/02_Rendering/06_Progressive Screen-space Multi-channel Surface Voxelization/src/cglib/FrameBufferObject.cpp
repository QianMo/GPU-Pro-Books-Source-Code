//
// Copyright (c) 2005, 
// Aaron Lefohn    (lefohn@cs.ucdavis.edu)
// Robert Strzodka (strzodka@stanford.edu)
// Adam Moerschell (atmoerschell@ucdavis.edu)
// All rights reserved.
// 
// This software is licensed under the BSD open-source license. See
// http://www.opensource.org/licenses/bsd-license.php for more detail.
// 
////////////////////////////////////////////////////////////////////////////////

#include <iostream>

#include "cglibdefines.h"
#include "FrameBufferObject.h"

#ifdef WIN32
#define FRAME_BUFFER_OBJECT \
    (glewGetExtension ("GL_EXT_framebuffer_object"))
#else
#define FRAME_BUFFER_OBJECT \
    (strstr ((const char *) glGetString (GL_EXTENSIONS), "GL_EXT_framebuffer_object") != NULL)
#endif

FramebufferObject::FramebufferObject ()
{
	EAZD_ASSERTALWAYS (fboSupported ());

	m_fboId = _GenerateFboId ();
	m_savedFboId = 0;

	// Bind this FBO so that it actually gets created now
	_GuardedBind ();
	_GuardedUnbind ();
}

FramebufferObject::~FramebufferObject ()
{
	if (glIsFramebuffer (m_fboId))
		glDeleteFramebuffers (1, &m_fboId);
}

int FramebufferObject::m_fboSupported = -1;

bool FramebufferObject::fboSupported ()
{
	if (m_fboSupported < 0)
		m_fboSupported = FRAME_BUFFER_OBJECT ? 1 : 0;

	return m_fboSupported == 1;
}

void FramebufferObject::Bind ()
{
	glBindFramebuffer (GL_FRAMEBUFFER, m_fboId);
}

void FramebufferObject::Disable ()
{
	glBindFramebuffer (GL_FRAMEBUFFER, 0);
}

void
FramebufferObject::AttachTexture (GLenum texTarget, GLuint texId, 
								  GLenum attachment, int mipLevel, int zSlice)
{
	_GuardedBind ();

#if 0
#ifndef NDEBUG
	if (GetAttachedId (attachment) != texId)
	{
#endif

	_FramebufferTextureND (attachment, texTarget, texId, mipLevel, zSlice);

#ifndef NDEBUG
	}
	else
	{
		std::cerr << "FramebufferObject::AttachTexture PERFORMANCE WARNING:\n"
			 << "\tRedundant bind of texture (id = " << texId << ").\n"
			 << "\tHINT : Compile with -DNDEBUG to remove this warning.\n";
	}
#endif
#else

	_FramebufferTextureND (attachment, texTarget, texId, mipLevel, zSlice);

#endif

	_GuardedUnbind ();
}

void
FramebufferObject::AttachTextures (int numTextures, GLenum texTarget[], GLuint texId[],
								   GLenum attachment[], int mipLevel[], int zSlice[])
{
	for (int i = 0; i < numTextures; ++i)
		AttachTexture (texTarget[i], texId[i], 
			attachment ? attachment[i] : (GL_COLOR_ATTACHMENT0 + i), 
			mipLevel ? mipLevel[i] : 0, 
			zSlice ? zSlice[i] : 0);
}

void
FramebufferObject::AttachRenderBuffer (GLuint buffId, GLenum attachment)
{
	_GuardedBind ();

	glFramebufferRenderbuffer (GL_FRAMEBUFFER, attachment, GL_RENDERBUFFER, buffId);

#if 0
#ifndef NDEBUG
	if (GetAttachedId (attachment) != buffId)
	{
#endif

	glFramebufferRenderbuffer (GL_FRAMEBUFFER, attachment, GL_RENDERBUFFER, buffId);

#ifndef NDEBUG
	}
	else {
		std::cerr << "FramebufferObject::AttachRenderBuffer PERFORMANCE WARNING:\n"
			<< "\tRedundant bind of Renderbuffer (id = " << buffId << ")\n"
			<< "\tHINT : Compile with -DNDEBUG to remove this warning.\n";
	}
#endif
#endif

	_GuardedUnbind ();
}

void
FramebufferObject::AttachRenderBuffers (int numBuffers, GLuint buffId[], GLenum attachment[])
{
	for (int i = 0; i < numBuffers; ++i)
		AttachRenderBuffer (buffId[i], 
			attachment ? attachment[i] : (GL_COLOR_ATTACHMENT0 + i));
}

void
FramebufferObject::Unattach (GLenum attachment)
{
	_GuardedBind ();

	switch (GetAttachedType (attachment))
	{
		case GL_NONE:
			break;
		case GL_RENDERBUFFER:
			AttachRenderBuffer (0, attachment);
			break;
		case GL_TEXTURE:
			AttachTexture (GL_TEXTURE_2D, 0, attachment);
			break;
		default:
			std::cerr << "FramebufferObject::unbind_attachment ERROR: Unknown attached resource type\n";
	}

	_GuardedUnbind ();
}

void
FramebufferObject::UnattachAll ()
{
	int numAttachments = GetMaxColorAttachments ();

	for (int i = 0; i < numAttachments; ++i)
		Unattach (GL_COLOR_ATTACHMENT0 + i);
}

GLint FramebufferObject::GetMaxColorAttachments ()
{
	GLint maxAttach = 0;
	glGetIntegerv (GL_MAX_COLOR_ATTACHMENTS, &maxAttach);

	GLint maxBuffers = 0;
	glGetIntegerv (GL_MAX_DRAW_BUFFERS, &maxBuffers);

	EAZD_ASSERTALWAYS (maxAttach == maxBuffers);

	return maxAttach;
}

GLuint FramebufferObject::_GenerateFboId ()
{
	GLuint id = 0;
	glGenFramebuffers (1, &id);

	return id;
}

void FramebufferObject::_GuardedBind ()
{
	// Only binds if m_fboId is different than the currently bound FBO
	glGetIntegerv (GL_FRAMEBUFFER_BINDING, &m_savedFboId);

	if (m_fboId != (GLuint) m_savedFboId)
		glBindFramebuffer (GL_FRAMEBUFFER, m_fboId);
}

void FramebufferObject::_GuardedUnbind ()
{
	// Returns FBO binding to the previously enabled FBO
	if (m_fboId != (GLuint) m_savedFboId)
		glBindFramebuffer (GL_FRAMEBUFFER, (GLuint) m_savedFboId);
}

void
FramebufferObject::_FramebufferTextureND (GLenum attachment, GLenum texTarget,
										  GLuint texId, int mipLevel, int zSlice)
{
		 if (texTarget == GL_TEXTURE_LAYER)
		glFramebufferTextureARB (GL_FRAMEBUFFER, attachment, texId, mipLevel);
	else if (texTarget == GL_TEXTURE_1D)
		glFramebufferTexture1D (GL_FRAMEBUFFER, attachment,
			GL_TEXTURE_1D, texId, mipLevel);
	else if (texTarget == GL_TEXTURE_3D)
		glFramebufferTexture3D (GL_FRAMEBUFFER, attachment,
			GL_TEXTURE_3D, texId, mipLevel, zSlice);
	else
		// Default is GL_TEXTURE_2D, GL_TEXTURE_RECTANGLE_ARB, or cube faces
		glFramebufferTexture2D (GL_FRAMEBUFFER, attachment,
			texTarget, texId, mipLevel);
}

#ifndef NDEBUG
bool FramebufferObject::IsValid (std::ostream& ostr)
{
#define CASE(format) case format: ostr << "FramebufferObject::IsValid () ERROR:\n\t" << #format << "\n"; isOK = false; break;

	_GuardedBind ();

	bool isOK = false;

	switch (glCheckFramebufferStatus (GL_FRAMEBUFFER))
	{                                          
		case GL_FRAMEBUFFER_COMPLETE: // Everything's OK
			isOK = true;
			break;
		CASE (GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT)
		CASE (GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS_EXT)
		CASE (GL_FRAMEBUFFER_INCOMPLETE_FORMATS_EXT)
		CASE (GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER)
		CASE (GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER)
		CASE (GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT)
		CASE (GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS)
		CASE (GL_FRAMEBUFFER_INCOMPLETE_LAYER_COUNT_ARB)
		CASE (GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE)
		CASE (GL_FRAMEBUFFER_UNSUPPORTED)
		default:
			ostr << "FramebufferObject::IsValid () ERROR:\n\tUnknown ERROR\n";
			isOK = false;
	}

	_GuardedUnbind ();

	return isOK;

#undef CASE
}
#endif // NDEBUG

/// Accessors
GLenum FramebufferObject::GetAttachedType (GLenum attachment)
{
	// Returns GL_RENDERBUFFER or GL_TEXTURE
	_GuardedBind ();

	GLint type = 0;
	glGetFramebufferAttachmentParameteriv (GL_FRAMEBUFFER, attachment,
		GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE, &type);

	_GuardedUnbind ();

	return GLenum (type);
}

GLuint FramebufferObject::GetAttachedId (GLenum attachment)
{
	_GuardedBind ();

	GLint id = 0;
	glGetFramebufferAttachmentParameteriv (GL_FRAMEBUFFER, attachment,
		GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME, &id);

	_GuardedUnbind ();

	return GLuint (id);
}

GLint FramebufferObject::GetAttachedMipLevel (GLenum attachment)
{
	_GuardedBind ();

	GLint level = 0;
	glGetFramebufferAttachmentParameteriv (GL_FRAMEBUFFER, attachment,
		GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_LEVEL, &level);

	_GuardedUnbind ();

	return level;
}

GLint FramebufferObject::GetAttachedCubeFace (GLenum attachment)
{
	_GuardedBind ();

	GLint level = 0;
	glGetFramebufferAttachmentParameteriv (GL_FRAMEBUFFER, attachment,
		GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_CUBE_MAP_FACE, &level);

	_GuardedUnbind ();

	return level;
}

GLint FramebufferObject::GetAttachedZSlice (GLenum attachment)
{
	_GuardedBind ();

	GLint slice = 0;
	glGetFramebufferAttachmentParameteriv (GL_FRAMEBUFFER, attachment,
		GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_3D_ZOFFSET_EXT, &slice);

	_GuardedUnbind ();

	return slice;
}
