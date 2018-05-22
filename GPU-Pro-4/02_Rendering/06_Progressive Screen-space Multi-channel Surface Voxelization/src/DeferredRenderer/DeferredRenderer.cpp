//////////////////////////////////////////////////////////////////////////////
//                                                                          //
//  Scene Graph 3D                                                          //
//  Georgios Papaioannou, 2009                                              //
//                                                                          //
//  This is a free, extensible scene graph management library that works    //
//  along with the EaZD deferred renderer. Both libraries and their source  //
//  code are free. If you use this code as is or any part of it in any kind //
//  of project or product, please acknowledge the source and its author.    //
//                                                                          //
//  For manuals, help and instructions, please visit:                       //
//  http://graphics.cs.aueb.gr/graphics/                                    //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////

#ifdef WIN32
	#include <windows.h>
#endif

#include <string.h>
#include <math.h>
#include <time.h>
#include <typeinfo>

#include <GL/glew.h>

#include "DeferredRenderer.h"
#include "SceneGraph.h"
#include "GlobalIlluminationRenderer_IV.h"

#ifdef USING_GLUT	// defined in DeferredRenderer.h
	#include <GL/glut.h>
#endif

#ifndef WIN32
#include <sys/time.h>

inline size_t strcpy_s(
      char *strDestination,
      size_t sizeInBytes,
      const char *strSource)
{
    if (strlen(strSource)+1 > sizeInBytes)
        return 1;
    strcpy(strDestination, strSource);

    return 0;
}

inline size_t strcat_s(
      char *strDestination,
      size_t sizeInBytes,
      const char *strSource)
{
    if (strlen(strSource) + strlen(strDestination) + 1 > sizeInBytes)
        return 1;
    strcat(strDestination, strSource);

    return 0;
}
#endif

float getHighResTime() 
{
#ifdef WIN32
	LARGE_INTEGER ticksPerSecond;
	LARGE_INTEGER tick;
	bool available = QueryPerformanceFrequency(&ticksPerSecond);
	if (!available)
		return 0.0;
	QueryPerformanceCounter(&tick);
	
	return (float)(1000.0f*((double)tick.QuadPart/(double)ticksPerSecond.QuadPart));
#else
	return 0.0f;
#endif
}



//------------ Helper functions. Copied from the MESA implementation CVS -------
/*
 * Compute inverse of 4x4 transformation matrix.
 * Code contributed by Jacques Leroy jle@star.be
 * Return GL_TRUE for success, GL_FALSE for failure (singular matrix)
 */
GLboolean
invert_matrix(const GLdouble * m, GLdouble * out)
{
/* NB. OpenGL Matrices are COLUMN major. */
#define SWAP_ROWS(a, b) { GLdouble *_tmp = a; (a)=(b); (b)=_tmp; }
#define MAT(m,r,c) (m)[(c)*4+(r)]

   GLdouble wtmp[4][8];
   GLdouble m0, m1, m2, m3, s;
   GLdouble *r0, *r1, *r2, *r3;

   r0 = wtmp[0], r1 = wtmp[1], r2 = wtmp[2], r3 = wtmp[3];

   r0[0] = MAT(m, 0, 0), r0[1] = MAT(m, 0, 1),
      r0[2] = MAT(m, 0, 2), r0[3] = MAT(m, 0, 3),
      r0[4] = 1.0, r0[5] = r0[6] = r0[7] = 0.0,
      r1[0] = MAT(m, 1, 0), r1[1] = MAT(m, 1, 1),
      r1[2] = MAT(m, 1, 2), r1[3] = MAT(m, 1, 3),
      r1[5] = 1.0, r1[4] = r1[6] = r1[7] = 0.0,
      r2[0] = MAT(m, 2, 0), r2[1] = MAT(m, 2, 1),
      r2[2] = MAT(m, 2, 2), r2[3] = MAT(m, 2, 3),
      r2[6] = 1.0, r2[4] = r2[5] = r2[7] = 0.0,
      r3[0] = MAT(m, 3, 0), r3[1] = MAT(m, 3, 1),
      r3[2] = MAT(m, 3, 2), r3[3] = MAT(m, 3, 3),
      r3[7] = 1.0, r3[4] = r3[5] = r3[6] = 0.0;

   /* choose pivot - or die */
   if (fabs(r3[0]) > fabs(r2[0]))
      SWAP_ROWS(r3, r2);
   if (fabs(r2[0]) > fabs(r1[0]))
      SWAP_ROWS(r2, r1);
   if (fabs(r1[0]) > fabs(r0[0]))
      SWAP_ROWS(r1, r0);
   if (0.0 == r0[0])
      return GL_FALSE;

   /* eliminate first variable     */
   m1 = r1[0] / r0[0];
   m2 = r2[0] / r0[0];
   m3 = r3[0] / r0[0];
   s = r0[1];
   r1[1] -= m1 * s;
   r2[1] -= m2 * s;
   r3[1] -= m3 * s;
   s = r0[2];
   r1[2] -= m1 * s;
   r2[2] -= m2 * s;
   r3[2] -= m3 * s;
   s = r0[3];
   r1[3] -= m1 * s;
   r2[3] -= m2 * s;
   r3[3] -= m3 * s;
   s = r0[4];
   if (s != 0.0) {
      r1[4] -= m1 * s;
      r2[4] -= m2 * s;
      r3[4] -= m3 * s;
   }
   s = r0[5];
   if (s != 0.0) {
      r1[5] -= m1 * s;
      r2[5] -= m2 * s;
      r3[5] -= m3 * s;
   }
   s = r0[6];
   if (s != 0.0) {
      r1[6] -= m1 * s;
      r2[6] -= m2 * s;
      r3[6] -= m3 * s;
   }
   s = r0[7];
   if (s != 0.0) {
      r1[7] -= m1 * s;
      r2[7] -= m2 * s;
      r3[7] -= m3 * s;
   }

   /* choose pivot - or die */
   if (fabs(r3[1]) > fabs(r2[1]))
      SWAP_ROWS(r3, r2);
   if (fabs(r2[1]) > fabs(r1[1]))
      SWAP_ROWS(r2, r1);
   if (0.0 == r1[1])
      return GL_FALSE;

   /* eliminate second variable */
   m2 = r2[1] / r1[1];
   m3 = r3[1] / r1[1];
   r2[2] -= m2 * r1[2];
   r3[2] -= m3 * r1[2];
   r2[3] -= m2 * r1[3];
   r3[3] -= m3 * r1[3];
   s = r1[4];
   if (0.0 != s) {
      r2[4] -= m2 * s;
      r3[4] -= m3 * s;
   }
   s = r1[5];
   if (0.0 != s) {
      r2[5] -= m2 * s;
      r3[5] -= m3 * s;
   }
   s = r1[6];
   if (0.0 != s) {
      r2[6] -= m2 * s;
      r3[6] -= m3 * s;
   }
   s = r1[7];
   if (0.0 != s) {
      r2[7] -= m2 * s;
      r3[7] -= m3 * s;
   }

   /* choose pivot - or die */
   if (fabs(r3[2]) > fabs(r2[2]))
      SWAP_ROWS(r3, r2);
   if (0.0 == r2[2])
      return GL_FALSE;

   /* eliminate third variable */
   m3 = r3[2] / r2[2];
   r3[3] -= m3 * r2[3], r3[4] -= m3 * r2[4],
      r3[5] -= m3 * r2[5], r3[6] -= m3 * r2[6], r3[7] -= m3 * r2[7];

   /* last check */
   if (0.0 == r3[3])
      return GL_FALSE;

   s = 1.0 / r3[3];		/* now back substitute row 3 */
   r3[4] *= s;
   r3[5] *= s;
   r3[6] *= s;
   r3[7] *= s;

   m2 = r2[3];			/* now back substitute row 2 */
   s = 1.0 / r2[2];
   r2[4] = s * (r2[4] - r3[4] * m2), r2[5] = s * (r2[5] - r3[5] * m2),
      r2[6] = s * (r2[6] - r3[6] * m2), r2[7] = s * (r2[7] - r3[7] * m2);
   m1 = r1[3];
   r1[4] -= r3[4] * m1, r1[5] -= r3[5] * m1,
      r1[6] -= r3[6] * m1, r1[7] -= r3[7] * m1;
   m0 = r0[3];
   r0[4] -= r3[4] * m0, r0[5] -= r3[5] * m0,
      r0[6] -= r3[6] * m0, r0[7] -= r3[7] * m0;

   m1 = r1[2];			/* now back substitute row 1 */
   s = 1.0 / r1[1];
   r1[4] = s * (r1[4] - r2[4] * m1), r1[5] = s * (r1[5] - r2[5] * m1),
      r1[6] = s * (r1[6] - r2[6] * m1), r1[7] = s * (r1[7] - r2[7] * m1);
   m0 = r0[2];
   r0[4] -= r2[4] * m0, r0[5] -= r2[5] * m0,
      r0[6] -= r2[6] * m0, r0[7] -= r2[7] * m0;

   m0 = r0[1];			/* now back substitute row 0 */
   s = 1.0 / r0[0];
   r0[4] = s * (r0[4] - r1[4] * m0), r0[5] = s * (r0[5] - r1[5] * m0),
      r0[6] = s * (r0[6] - r1[6] * m0), r0[7] = s * (r0[7] - r1[7] * m0);

   MAT(out, 0, 0) = r0[4];
   MAT(out, 0, 1) = r0[5], MAT(out, 0, 2) = r0[6];
   MAT(out, 0, 3) = r0[7], MAT(out, 1, 0) = r1[4];
   MAT(out, 1, 1) = r1[5], MAT(out, 1, 2) = r1[6];
   MAT(out, 1, 3) = r1[7], MAT(out, 2, 0) = r2[4];
   MAT(out, 2, 1) = r2[5], MAT(out, 2, 2) = r2[6];
   MAT(out, 2, 3) = r2[7], MAT(out, 3, 0) = r3[4];
   MAT(out, 3, 1) = r3[5], MAT(out, 3, 2) = r3[6];
   MAT(out, 3, 3) = r3[7];

   return GL_TRUE;

#undef MAT
#undef SWAP_ROWS
}

/*
 * Perform a 4x4 matrix multiplication  (product = a x b).
 * Input:  a, b - matrices to multiply
 * Output:  product - product of a and b
 */
void
matmul(GLdouble * product, const GLdouble * a, const GLdouble * b)
{
   /* This matmul was contributed by Thomas Malik */
   GLdouble temp[16];
   GLint i;

#define A(row,col)  a[(col<<2)+row]
#define B(row,col)  b[(col<<2)+row]
#define T(row,col)  temp[(col<<2)+row]

   /* i-te Zeile */
   for (i = 0; i < 4; i++)
   {
      T(i, 0) = A(i, 0) * B(0, 0) + A(i, 1) * B(1, 0) + A(i, 2) * B(2, 0) + A(i, 3) * B(3, 0);
      T(i, 1) = A(i, 0) * B(0, 1) + A(i, 1) * B(1, 1) + A(i, 2) * B(2, 1) + A(i, 3) * B(3, 1);
      T(i, 2) = A(i, 0) * B(0, 2) + A(i, 1) * B(1, 2) + A(i, 2) * B(2, 2) + A(i, 3) * B(3, 2);
      T(i, 3) = A(i, 0) * B(0, 3) + A(i, 1) * B(1, 3) + A(i, 2) * B(2, 3) + A(i, 3) * B(3, 3);
   }

#undef A
#undef B
#undef T
   memcpy(product, temp, 16 * sizeof(GLdouble));
}

//-------------------- end here (MESA) ---------------------------


DeferredRenderer::~DeferredRenderer()
{
	if (gi_renderer)
		delete gi_renderer;
	
	if (buffer[0]!=0)
		glDeleteTextures(8, buffer);
	
	if (glIsFramebufferEXT(FBO))
		glDeleteFramebuffersEXT(1, &FBO);
	
	if (glIsFramebufferEXT(framebuffer_FBO))
		glDeleteFramebuffersEXT(1, &framebuffer_FBO);

	if (glIsFramebufferEXT(glow_FBO))
		glDeleteFramebuffersEXT(1, &glow_FBO);

	shadow_fbo.release();

delete t_g_buffer;
}

DeferredRenderer::DeferredRenderer()
{
	// set the default render targets (render buffer 0)
    buffer_bits = DR_BUFFER_ALBEDO | DR_BUFFER_CSS_DEPTH | DR_BUFFER_ECS_NORMAL | DR_BUFFER_LIGHTS | DR_BUFFER_SPECULAR | DR_BUFFER_GLOW;
	width = height = 512;
	is_fixed_size = false;
	buffer_scale = 1.0f;
	buffer[0] = buffer[1] = buffer[2] =
    buffer[3] = buffer[4] = buffer[5] =
	buffer[6] = buffer[7] = buffer[8] = 0;
	is_initialized = false;
	draw_callback = NULL;
	camera_callback = NULL;
	num_lights = 0;
	global_ambient[0] = global_ambient[1] = global_ambient[2] = 0.2f;
	gi_renderer = NULL;
	calc_matrix = false;
	shadow_method = DR_SHADOW_GAUSS;
	noise = 0;
	memset(light_array_pos,0,3*DR_NUM_LIGHTS*sizeof(float));
	memset(light_array_dir,0,3*DR_NUM_LIGHTS*sizeof(float));
	memset(light_array_col,0,3*DR_NUM_LIGHTS*sizeof(float));
	memset(light_array_attn,0,DR_NUM_LIGHTS*sizeof(int));
	memset(light_array_active,0,DR_NUM_LIGHTS*sizeof(int));
	memset(light_array_range,0,DR_NUM_LIGHTS*sizeof(float));

	timer_total = counter_total = 0.0;
	frames = 0;
	root = NULL;
	ao_buffer_ratio = 1.0f;
	background[0] = background[1] = background[2] = 0.0f;
    volumebuffer_resolution = 32;
	use_gi = false;
	gi_method = DR_GI_METHOD_NONE;
	hdr_method = DR_HDR_AUTO;
	units_per_meter = 1.0f;

	glow_FBO = AO_FBO = framebuffer_FBO = 0;

t_g_buffer = new Timer("G-Buffer creation", 10);
}

int DeferredRenderer::init(int request_mask)
{
    buffer_bits = request_mask;
	return init();
}

int DeferredRenderer::resizeBuffers()
{
	glEnable(GL_TEXTURE_2D);

	// generate and bind buffer textures...
	if (buffer[0]!=0)
		glDeleteTextures(9, buffer);
	glGenTextures(9, buffer);

	if (glIsFramebufferEXT(FBO))
		glDeleteFramebuffersEXT(1, &FBO);
	glGenFramebuffersEXT(1, &FBO);
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT,FBO);

	actual_width = (int)floor(buffer_scale * width  + 0.5);
    actual_height = (int)floor(buffer_scale * height + 0.5);

    // Color and Alpha RGBA 8,8,8,8 --> Render Target 0
	glBindTexture(GL_TEXTURE_2D, buffer[DR_TARGET_COLOR]);
//	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, actual_width, actual_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, actual_width, actual_height, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT,
                              GL_TEXTURE_2D, buffer[DR_TARGET_COLOR], 0);
	CHECK_GL_ERROR();

	// Normals Luminance/Alpha 16,16 --> Render Target 1
	// GL_LUMINANCE_ALPHA or FLOAT_RG16_NV are not respected --> slow (S/W implementation ?)
	glBindTexture(GL_TEXTURE_2D, buffer[DR_TARGET_NORMAL]);
//	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, actual_width, actual_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, actual_width, actual_height, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT1_EXT,
                              GL_TEXTURE_2D, buffer[DR_TARGET_NORMAL], 0);
	CHECK_GL_ERROR();
	
    // Specular intensity/power RGBA 8,8,8,8 --> Render Target 2
	glBindTexture(GL_TEXTURE_2D, buffer[DR_TARGET_SPECULAR]);
//	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, actual_width, actual_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, actual_width, actual_height, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT2_EXT,
                              GL_TEXTURE_2D, buffer[DR_TARGET_SPECULAR], 0);
	CHECK_GL_ERROR();

	// Illumination/Effect channel RGBA 8,8,8,8 --> Render Target 3
	glBindTexture(GL_TEXTURE_2D, buffer[DR_TARGET_LIGHTING]);
//	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, actual_width, actual_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, actual_width, actual_height, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT3_EXT,
                              GL_TEXTURE_2D, buffer[DR_TARGET_LIGHTING], 0);
	CHECK_GL_ERROR();

	multipleRenderTargets[0] = GL_COLOR_ATTACHMENT0_EXT;
	multipleRenderTargets[1] = GL_COLOR_ATTACHMENT1_EXT;
	multipleRenderTargets[2] = GL_COLOR_ATTACHMENT2_EXT;
	multipleRenderTargets[3] = GL_COLOR_ATTACHMENT3_EXT;

	// Bind depth texture --> 24bits depth buffer
	glBindTexture(GL_TEXTURE_2D, buffer[DR_TARGET_DEPTH]);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F_NV, actual_width, actual_height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT,
		                      GL_TEXTURE_2D, buffer[DR_TARGET_DEPTH], 0);
	CHECK_GL_ERROR();
#if 1
	if (glCheckFramebufferStatus(GL_FRAMEBUFFER_EXT)!=GL_FRAMEBUFFER_COMPLETE_EXT)
	{
		EAZD_TRACE ("DeferredRenderer::resizeBuffers() : ERROR - incomplete frame buffer object (MRT).");
		return DR_ERROR_BAD_FBO;
	}
#else
#define CASE(format) case format: EAZD_TRACE ("DeferredRenderer::resizeBuffers() : ERROR - \n\t" << #format); return DR_ERROR_BAD_FBO;

	switch (glCheckFramebufferStatus (GL_FRAMEBUFFER))
	{                                          
		case GL_FRAMEBUFFER_COMPLETE: // Everything's OK
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
			EAZD_TRACE ("DeferredRenderer::resizeBuffers() : ERROR - \n\tUnknown ERROR");
			return DR_ERROR_BAD_FBO;
	}
#endif
	if (glIsFramebufferEXT(framebuffer_FBO))
		glDeleteFramebuffersEXT(1, &framebuffer_FBO);
	glGenFramebuffersEXT(1, &framebuffer_FBO);
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT,framebuffer_FBO);
	
	// Final frame buffer: RGBA 12,12,12,12 --> separate color attachment 0 (2nd pass)
	glBindTexture(GL_TEXTURE_2D, buffer[DR_TARGET_FRAMEBUFFER]);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_R11F_G11F_B10F, width, height, 0, GL_RGB, GL_FLOAT, NULL);
//	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_FLOAT, NULL);
//	glPixelStorei(GL_PACK_ALIGNMENT,8);
//	glPixelStorei(GL_UNPACK_ALIGNMENT,8);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT,
                              GL_TEXTURE_2D, buffer[DR_TARGET_FRAMEBUFFER], 0);
	CHECK_GL_ERROR();

	// If HDR rendering is set to auto, then build mipamaps (and keep building them at 
	// regular intervals) so that a mean intensity can be calculated. 
	if (hdr_method==DR_HDR_AUTO)
	{
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		glGenerateMipmapEXT(GL_TEXTURE_2D); 
	}
	else
	{
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    }

	if (glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT)!=GL_FRAMEBUFFER_COMPLETE_EXT)
	{
		EAZD_TRACE ("DeferredRenderer::resizeBuffers() : ERROR - incomplete frame buffer object (final buffer).");
		return DR_ERROR_BAD_FBO;
	}

	// glow effect frame buffer: RGBA 8,8,8,8 --> separate color attachment 0
	if (glIsFramebufferEXT(glow_FBO))
		glDeleteFramebuffersEXT(1, &glow_FBO);
	glGenFramebuffersEXT(1, &glow_FBO);
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT,glow_FBO);

	glBindTexture(GL_TEXTURE_2D, buffer[DR_TARGET_GLOW]);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width/8, height/8, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT,
                              GL_TEXTURE_2D, buffer[DR_TARGET_GLOW], 0);
	CHECK_GL_ERROR();

	if (glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT)!=GL_FRAMEBUFFER_COMPLETE_EXT)
	{
		EAZD_TRACE ("DeferredRenderer::resizeBuffers() : ERROR - incomplete frame buffer object (glow effect).");
		return DR_ERROR_BAD_FBO;
	}

	// ambient occlusion/illum frame buffer: RGBA 8,8,8,8 --> separate color attachment 0
	if (glIsFramebufferEXT(AO_FBO))
		glDeleteFramebuffersEXT(1, &AO_FBO);
	glGenFramebuffersEXT(1, &AO_FBO);
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT,AO_FBO);

	glBindTexture(GL_TEXTURE_2D, buffer[DR_TARGET_AO]);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, (GLsizei) (ao_buffer_ratio*actual_width), (GLsizei) (ao_buffer_ratio*actual_height), 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT,
                              GL_TEXTURE_2D, buffer[DR_TARGET_AO], 0);
	CHECK_GL_ERROR();

	if (glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT)!=GL_FRAMEBUFFER_COMPLETE_EXT)
	{
		EAZD_TRACE ("DeferredRenderer::resizeBuffers() : ERROR - incomplete frame buffer object (AO).");
		return DR_ERROR_BAD_FBO;
	}

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glGenerateMipmapEXT(GL_TEXTURE_2D); 
	glBindTexture(GL_TEXTURE_2D, 0);
	glDisable(GL_TEXTURE_2D);
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT,0);

	return DR_ERROR_NONE;
}

int DeferredRenderer::init()
{
	const GLubyte *str;
	if (! (str = glGetString (GL_EXTENSIONS)))
        return DR_ERROR_NOT_SUPPORTED;

    if (! ((strstr ((const char *) str, "GL_ARB_shading_language_100")      != NULL) &&
           (strstr ((const char *) str, "GL_ARB_texture_non_power_of_two")  != NULL) &&
           (strstr ((const char *) str, "GL_ARB_multitexture")              != NULL)
        ))
	{
		EAZD_TRACE ("DeferredRenderer::init() : ERROR - Shaders are not supported");
		return DR_ERROR_NOT_SUPPORTED;
	}

	glewInit();

    // query maximum number of render buffers supported by driver
	glGetIntegerv(GL_MAX_DRAW_BUFFERS, &maxbuffers);
	printf("DeferredRenderer::init() : OpenGL info - Maximum render targets available: %d\n", maxbuffers);
	if (maxbuffers<4)
		return DR_ERROR_INSUFFICIENT_BUFFERS;
	int maxtextures;
	glGetIntegerv(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS_ARB, &maxtextures);
	printf("DeferredRenderer::init() : OpenGL info - Maximum texture units supported in shader: %d\n", maxtextures);
	if (maxtextures<5)
		return DR_ERROR_INSUFFICIENT_TEXTURE_UNITS;
	int i;
	glGetIntegerv (GL_MAX_COLOR_ATTACHMENTS_EXT, &i);
	printf ("DeferredRenderer::init() : OpenGL info - Maximum number of color output buffers (fbo's): %d\n", i);
	
    // query maximum texture sizes
	glGetIntegerv (GL_MAX_TEXTURE_SIZE, &i);
	printf ("DeferredRenderer::init() : OpenGL info - Maximum texture size: %d\n", i);
	glGetIntegerv (GL_MAX_3D_TEXTURE_SIZE, &i);
	printf ("DeferredRenderer::init() : OpenGL info - Maximum 3D texture size: %d\n", i);

	// generate frame buffer object for MRT
	glGenFramebuffersEXT(1, &FBO);
	// generate frame buffer object for final rendering
	glGenFramebuffersEXT(1, &framebuffer_FBO);
	
	//build noise texture
	unsigned char *noisemap;
	noisemap = (unsigned char*)malloc(256*256*3*sizeof(unsigned char));
	long k;
	SRAND(0);
	for (k=0; k<(256*256); k++)
	{
		noisemap[k*3+0]=(unsigned char)floor(255.0f*RAND_0TO1());
		noisemap[k*3+1]=(unsigned char)floor(255.0f*RAND_0TO1());
		noisemap[k*3+2]=(unsigned char)floor(255.0f*RAND_0TO1());
	}
	if (noise!=0)
		glDeleteTextures(1, &noise);
	glGenTextures(1, &noise);
	glBindTexture(GL_TEXTURE_2D, noise);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 256, 256, 0, GL_RGB, GL_UNSIGNED_BYTE, noisemap);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	free(noisemap);

	int err = resizeBuffers();
	if (err!=DR_ERROR_NONE)
	{
		EAZD_TRACE ("DeferredRenderer::init() : ERROR - renderer not properly initialized.");
		return err;
	}
	
	buildShaders();

	is_initialized = true;
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT,0);

	return DR_ERROR_NONE;
}

void DeferredRenderer::setBufferScale(float scale)
{
	if (scale<0)
		return;
	buffer_scale = scale;
	if (resizeBuffers() != DR_ERROR_NONE)
		EAZD_TRACE ("DeferredRenderer::setBufferScale() : ERROR - renderer not properly initialized.");
}

void DeferredRenderer::setFixedBufferSize(int w, int h)
{
    buffer_scale = 1.0f;
	width = w;
	height =h;
	is_fixed_size = true;
	if (resizeBuffers() != DR_ERROR_NONE)
		EAZD_TRACE ("DeferredRenderer::setFixedBufferSize() : ERROR - renderer not properly initialized.");
}

void DeferredRenderer::setFreeBufferSize()
{
    buffer_scale = 1.0f;
	is_fixed_size = false;
	if (resizeBuffers() != DR_ERROR_NONE)
		EAZD_TRACE ("DeferredRenderer::setFreeBufferSize() : ERROR - renderer not properly initialized.");
}

void DeferredRenderer::setCameraCallback(void(*cm)(void))
{
	camera_callback = cm;
}

void displayMessageStroke(char * msg, int len, int x, int y)
{
	glEnable(GL_COLOR_MATERIAL);
	glDisable(GL_TEXTURE_2D);
	glDisable(GL_TEXTURE_3D);
	glDisable(GL_LIGHTING);

	glColor3f(1,1,1);
	glLineWidth(1.0f);
	glPushMatrix();
		glTranslatef(x, y, 0);
		glScalef(0.12,0.10,1);
		for (int i = 0; i<len; i++)
			glutStrokeCharacter(GLUT_STROKE_ROMAN, msg[i]);
	glPopMatrix();
	glDisable(GL_COLOR_MATERIAL);
	glEnable(GL_LIGHTING);
}

void displayMessageBitmap(char * msg, int len, int x, int y)
{
	glEnable(GL_COLOR_MATERIAL);
	glDisable(GL_TEXTURE_2D);
	glDisable(GL_TEXTURE_3D);
	glDisable(GL_LIGHTING);

	glColor3f(1,1,1);
	glLineWidth(1.0f);
	glPushMatrix();
		glRasterPos3f (x, y, 0);
		for (int i = 0; i<len; i++)
			glutBitmapCharacter(GLUT_BITMAP_9_BY_15, msg[i]);
	glPopMatrix();
	glDisable(GL_COLOR_MATERIAL);
	glEnable(GL_LIGHTING);
}

void DeferredRenderer::showStatistics()
{
	GlobalIlluminationRendererIV * gi_iv = dynamic_cast<GlobalIlluminationRendererIV*>(gi_renderer);
	if (! gi_iv)
		return;

	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
	glDrawBuffer(GL_BACK);
	glActiveTextureARB(GL_TEXTURE0);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_LIGHTING);
	glBindTexture(GL_TEXTURE_2D,0);
    glDisable(GL_TEXTURE_2D);

#ifdef USING_GLUT
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glOrtho(0.0, width, 0.0, height, -1, 1);
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	char message[80];
//	sprintf(message,"frame time: %3.3f ms, %3.3f fps",(float)timer_total, getFrameRate ());

	sprintf(message,"%s: %3.3f ms", t_g_buffer->getName ().c_str (), t_g_buffer->getMsTime ());
	displayMessageBitmap(message, strlen(message), 10, height-17);

	sprintf(message,"Volume Buffer resolution: %d^3", volumebuffer_resolution);
	displayMessageBitmap(message, strlen(message), 10, height-34);

	if (gi_iv->getInjectCamera() && gi_iv->t_inc_camera_injection)
	{
	sprintf(message,"%s: %3.3f ms", gi_iv->t_inc_camera_injection->getName ().c_str (), gi_iv->t_inc_camera_injection->getMsTime ());
	displayMessageBitmap(message, strlen(message), 10, height-51);
	}

	if (gi_iv->getInjectCamera() && gi_iv->t_inc_camera_cleanup)
	{
	sprintf(message,"%s: %3.3f ms", gi_iv->t_inc_camera_cleanup->getName ().c_str (), gi_iv->t_inc_camera_cleanup->getMsTime ());
	displayMessageBitmap(message, strlen(message), 10, height-68);
	}

	if (gi_iv->getInjectLights() && gi_iv->t_inc_light_injection)
	{
	sprintf(message,"%s: %3.3f ms", gi_iv->t_inc_light_injection->getName ().c_str (), gi_iv->t_inc_light_injection->getMsTime ());
	displayMessageBitmap(message, strlen(message), 10, height-85);
	}

	if (gi_iv->getInjectLights() && gi_iv->t_inc_light_cleanup)
	{
	sprintf(message,"%s: %3.3f ms", gi_iv->t_inc_light_cleanup->getName ().c_str (), gi_iv->t_inc_light_cleanup->getMsTime ());
	displayMessageBitmap(message, strlen(message), 10, height-102);
	}

	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
#endif

	glEnable(GL_DEPTH_TEST);
}

void DeferredRenderer::statistics()
{
	double stop;
	frames=(frames+1)%100;
	if (frames==0)
	{
#ifdef WIN32
		// retrieves the system time (time since the system started)
		// in milliseconds
		stop = timeGetTime();
#else
        gettimeofday(&tp, NULL);
		// tv_usec returns microseconds so we convert it to milliseconds
		stop = (double) (tp.tv_sec * 1000. + tp.tv_usec / 1000.);
#endif
		timer_total = (stop-counter_total)/100.0;
		counter_total = stop;
	}
}

void DeferredRenderer::drawShadowMaps()
{
	// Shadow maps
	for (int i=0; i<num_lights; i++)
	{
		if ( !lights[i].needsUpdate() ||
			 !lights[i].isActive() ||
			 !lights[i].isShadowEnabled() )
			 continue;
		glPushAttrib(GL_VIEWPORT_BIT );
		lights[i].setupShadowMap();
		glClampColor (GL_CLAMP_VERTEX_COLOR, GL_FALSE);
		glClampColor (GL_CLAMP_FRAGMENT_COLOR, GL_FALSE);
		glClampColor (GL_CLAMP_READ_COLOR, GL_FALSE);
		glEnable(GL_DEPTH_TEST);
		glDepthFunc(GL_LEQUAL);
		glDisable(GL_TEXTURE_2D);
		glDisable(GL_LIGHTING);
		
		glDepthMask(1);
		if (lights[i].hasExtendedData())
		{
			glColorMask(1,1,1,1);
			shader_ShadowMap.setCurrentLight(lights+i);
			shader_ShadowMap.start();
		}
		else
			glColorMask(0,0,0,0);
		
		if (!root)
			draw_callback();
		else
		{
			root->setRenderMode(SCENE_GRAPH_RENDER_MODE_NORMAL);
			root->clear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
			root->draw();
		}
		
		if (lights[i].hasExtendedData())
		{
			shader_ShadowMap.stop();
		}
		
		glColorMask(1,1,1,1);
		glEnable(GL_DEPTH_TEST);
		glDepthFunc(GL_LEQUAL);
		glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
		glPopAttrib();

		/*//glBindTexture(GL_TEXTURE_2D,lights[i].getShadowMap());
		//glGenerateMipmapEXT(GL_TEXTURE_2D); 
		glBindTexture(GL_TEXTURE_2D,lights[i].getShadowMapColorBuffer());
		glGenerateMipmapEXT(GL_TEXTURE_2D); 
		glBindTexture(GL_TEXTURE_2D,lights[i].getShadowMapNormalBuffer());
		glGenerateMipmapEXT(GL_TEXTURE_2D); 
		*/
		glBindTexture(GL_TEXTURE_2D,0);
		if ( lights[i].isShadowEnabled() )
			calc_matrix = true;
	}
}

void DeferredRenderer::drawMultipleRenderTargets()
{
	// Multiple render targets
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glEnable(GL_TEXTURE_2D);
	glDisable(GL_CULL_FACE);

	glPushAttrib(GL_VIEWPORT_BIT );
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, FBO);
    glViewport(0,0,actual_width, actual_height);
    glDrawBuffers(4, multipleRenderTargets);
	
	//clear the buffers
	glDisable(GL_BLEND);
	glDisable(GL_DEPTH_TEST);
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glOrtho(-1,1,-1,1,10,-10);
	glDisable(GL_LIGHTING);
	glFrontFace(GL_CCW);
	
t_g_buffer->start();

	shader_ClearMRT.start();
	
	glBegin(GL_QUADS);
        glNormal3f(0,0,1);
		glTexCoord2f(0,1);	glVertex3f(-1,1,0);
		glTexCoord2f(0,0);	glVertex3f(-1,-1,0);
		glTexCoord2f(1,0);	glVertex3f(1,-1,0);
		glTexCoord2f(1,1);	glVertex3f(1,1,0);
	glEnd();
	glDisable(GL_BLEND);
	
	shader_ClearMRT.stop();
	
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
	glEnable(GL_DEPTH_TEST);
	
	//glEnable(GL_MULTISAMPLE);
	//glSampleCoverage(GL_SAMPLE_ALPHA_TO_COVERAGE,false);
	glEnableIndexedEXT(0,GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_ALPHA_TEST);
	glAlphaFunc(GL_GREATER,0.01f);
	glActiveTextureARB(GL_TEXTURE5);
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, noise);
	
	shader_MRT.start();
	
	if (!root) // if no scene graph is defined, use draw callbacks
	{
		camera_callback();
		draw_callback();
	}
	else // call scene graph draw methods instead
	{
		root->getActiveCamera()->setupViewProjection(width, height);
		root->setRenderMode(SCENE_GRAPH_RENDER_MODE_NORMAL);
		root->clear(GL_DEPTH_BUFFER_BIT);
		root->draw();
	}
	
	shader_MRT.stop();

t_g_buffer->stop();

	glDisable(GL_BLEND);
	//glDisable(GL_SAMPLE_ALPHA_TO_COVERAGE);
	//glDisable(GL_MULTISAMPLE);
	glDisable(GL_CULL_FACE);

	for (int i=0;i<16;i++)
	{
		glActiveTextureARB(GL_TEXTURE0+i);
		glDisable(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, 0);
	}
	glPopAttrib();
	glDrawBuffer(GL_BACK);
	glDisable(GL_TEXTURE_2D);
	
}

void DeferredRenderer::drawGlobalIllumination()
{
	glActiveTextureARB(GL_TEXTURE4);
	glBindTexture(GL_TEXTURE_2D, buffer[DR_TARGET_NORMAL]);
	glActiveTextureARB(GL_TEXTURE5);
	glBindTexture(GL_TEXTURE_2D, buffer[DR_TARGET_DEPTH]);
	glActiveTextureARB(GL_TEXTURE6);
	glBindTexture(GL_TEXTURE_2D, noise);
	
	if (use_gi && gi_renderer)
	{
		gi_renderer->draw();
		//glBindTexture(GL_TEXTURE_2D,buffer[DR_TARGET_AO]);
		//glGenerateMipmapEXT(GL_TEXTURE_2D); 
		glBindTexture(GL_TEXTURE_2D,0);
	}
}

void DeferredRenderer::drawLighting()
{
	glPushAttrib(GL_VIEWPORT_BIT );
    
	glDisable(GL_DEPTH_TEST);
	glMatrixMode(GL_TEXTURE);
	glPushMatrix();
	glLoadIdentity();
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glOrtho(-1,1,-1,1,10,-10);
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	glDisable(GL_TEXTURE_2D);
	
	glEnable(GL_BLEND);
	glBlendEquation(GL_FUNC_ADD);
	int blending_src, blending_dst;
	glGetIntegerv(GL_BLEND_SRC,&blending_src);
	glGetIntegerv(GL_BLEND_DST,&blending_dst);
	glBlendFunc(GL_ONE,GL_ONE);
	
	glActiveTextureARB(GL_TEXTURE6);
	glBindTexture(GL_TEXTURE_2D, noise);
	glActiveTextureARB(GL_TEXTURE7);
	glBindTexture(GL_TEXTURE_2D, buffer[DR_TARGET_NORMAL]);
	glActiveTextureARB(GL_TEXTURE8);
	glBindTexture(GL_TEXTURE_2D, buffer[DR_TARGET_DEPTH]);
	glActiveTextureARB(GL_TEXTURE9);
	glBindTexture(GL_TEXTURE_2D, buffer[DR_TARGET_SPECULAR]);
	
	// Light sources contribution
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, FBO);
    glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT+DR_TARGET_LIGHTING);
	glViewport(0,0,actual_width, actual_height);
    glClearColor(0,0,0,0);
	glClear(GL_COLOR_BUFFER_BIT);

	shader_Lighting.setModelViewMatrixInverse(matrix_MVP_inverse);
	shader_Lighting.setProjectionMatrix(matrix_P);
	shader_Lighting.setEyePosition(eye);
	
	// lights
	for (int i=0; i<num_lights; i++)
	{
		if (!lights[i].isActive())
			continue;

		shader_Lighting.setCurrentLight(&(lights[i]));

		if (lights[i].isShadowEnabled())
			shader_Lighting.setLightMatrix(light_array_matrix[i]);
		
		glActiveTextureARB(GL_TEXTURE0);
		glEnable(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, lights[i].getShadowMap());
	
		shader_Lighting.start();
		
		float *lp = lights[i].getTransformedPosition();
		Vector3D camera_light = Vector3D(lp[0],lp[1],lp[2]);
		camera_light -= eye;

		if (light_array_attn[i]!=0 && camera_light.length()>2*light_array_range[i])
		{
			glMatrixMode(GL_PROJECTION);
			glLoadMatrixd(matrix_MVP);
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			glTranslatef(lp[0],lp[1],lp[2]);
			glEnable(GL_CULL_FACE);
			glFrontFace(GL_CW);
			glutSolidSphere(light_array_range[i]*1.2f,16,16);
			glFrontFace(GL_CCW);
		}
		else
		{
			glMatrixMode(GL_PROJECTION);
			glLoadIdentity();
			glOrtho(-1,1,-1,1,10,-10);
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			glFrontFace(GL_CCW);
			glBegin(GL_QUADS);
				glColor3f(1,1,1);
				glNormal3f(0,0,1);
				glTexCoord2f(0,1);	glVertex3f(-1,1,0);
				glTexCoord2f(0,0);	glVertex3f(-1,-1,0);
				glTexCoord2f(1,0);	glVertex3f(1,-1,0);
				glTexCoord2f(1,1);	glVertex3f(1,1,0);
			glEnd();
		}

		shader_Lighting.stop();
	}

// Global illumination or Ambient Occlusion pass
	glPushAttrib(GL_VIEWPORT_BIT );
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, AO_FBO);
    glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);
	glViewport(0,0, (GLsizei)(ao_buffer_ratio*actual_width), (GLsizei)(ao_buffer_ratio*actual_height));
    
	//glClearColor(0,0,0,1);
	//if (!use_gi)
		//glClear(GL_COLOR_BUFFER_BIT);
	
	if (use_gi)
		drawGlobalIllumination();
	
	glPopAttrib();

	glBlendFunc(blending_src,blending_dst);
	glBindTexture(GL_TEXTURE_2D,0);
	glEnable(GL_DEPTH_TEST);
	glPopAttrib();
	
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_TEXTURE);
	glPopMatrix();
	
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
	glDrawBuffer(GL_BACK);
}

void DeferredRenderer::drawTransparency()
{
	glPushAttrib(GL_VIEWPORT_BIT );
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, framebuffer_FBO);
    glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);
//	glDisable(GL_ALPHA_TEST);
	
	glViewport(0,0,actual_width, actual_height);
    glDisable(GL_DEPTH_TEST);
	glDepthMask(0);
	glMatrixMode(GL_TEXTURE);
	glPushMatrix();
	glLoadIdentity();
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	glDisable(GL_TEXTURE_2D);
	
//	glEnable(GL_BLEND);
//	glBlendEquation(GL_FUNC_ADD);
//	int blending_src, blending_dst;
//	glGetIntegerv(GL_BLEND_SRC,&blending_src);
//	glGetIntegerv(GL_BLEND_DST,&blending_dst);
//	glBlendFunc(GL_ONE,GL_ONE);
//	glActiveTextureARB(GL_TEXTURE0_ARB);
	
	shader_Trans.setModelViewMatrixInverse(matrix_MVP_inverse);
	shader_Trans.setEyePosition(eye);
	
	// lights
	for (int i=0; i<num_lights; i++)
	{
		if (!lights[i].isActive())
			continue;
		
		shader_Trans.setCurrentLight(&(lights[i]));
		if (lights[i].isShadowEnabled())
			shader_Trans.setLightMatrix(light_array_matrix[i]);
		
		glActiveTextureARB(GL_TEXTURE7);
		glBindTexture(GL_TEXTURE_2D, lights[i].getShadowMap());
		glActiveTextureARB(GL_TEXTURE8);
		glBindTexture(GL_TEXTURE_2D, buffer[DR_TARGET_DEPTH]);
		glActiveTextureARB(GL_TEXTURE9);
		glBindTexture(GL_TEXTURE_2D, buffer[DR_TARGET_SPECULAR]);
	
		shader_Trans.start();
	
		
		if (!root) // if no scene graph is defined, use draw callbacks
		{
			camera_callback();
			draw_callback();
		}
		else // call scene graph draw methods instead
		{
			root->getActiveCamera()->setupViewProjection(width, height);
			root->setRenderMode(SCENE_GRAPH_RENDER_MODE_TRANSPARENCY);
			root->draw();
		}
	
		shader_Trans.stop();
	}

//	glBlendFunc(blending_src,blending_dst);
//	glBindTexture(GL_TEXTURE_2D,0);
	glEnable(GL_DEPTH_TEST);
	glDepthMask(1);
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_TEXTURE);
	glPopMatrix();
//	glDisable(GL_BLEND);
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
	glDrawBuffer(GL_BACK);
	glPopAttrib();
}

void DeferredRenderer::drawFrameBuffer()
{
	// Render frame buffer
	glPushAttrib(GL_VIEWPORT_BIT );
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, framebuffer_FBO);
    glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);
	glViewport(0,0,width, height);
    glEnable(GL_TEXTURE_2D);
	glDisable(GL_DEPTH_TEST);
	glMatrixMode(GL_TEXTURE);
	glPushMatrix();
	glLoadIdentity();
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glOrtho(-1,1,-1,1,10,-10);
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	glDisable(GL_TEXTURE_2D);
	glDisable(GL_TEXTURE_3D);
	glDisable(GL_BLEND);
	
    glActiveTextureARB(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D, buffer[0]);
	
	glActiveTextureARB(GL_TEXTURE3);
	glBindTexture(GL_TEXTURE_2D, buffer[1]);
	
	glActiveTextureARB(GL_TEXTURE4);
	glBindTexture(GL_TEXTURE_2D, buffer[2]);
	
	glActiveTextureARB(GL_TEXTURE5);
	glBindTexture(GL_TEXTURE_2D, buffer[3]);
	
	glActiveTextureARB(GL_TEXTURE6);
	glBindTexture(GL_TEXTURE_2D, buffer[4]);

	glActiveTextureARB(GL_TEXTURE7);
	glBindTexture(GL_TEXTURE_2D, buffer[8]);

	glActiveTextureARB(GL_TEXTURE8);
	glBindTexture(GL_TEXTURE_2D, noise);
	
	glActiveTextureARB(GL_TEXTURE0_ARB);

	shader_FB.start();
	 
	glFrontFace(GL_CCW);
	glBegin(GL_QUADS);
        glColor3f(1,1,1);
	    glNormal3f(0,0,1);
		glTexCoord2f(0,1);
		glVertex3f(-1,1,0);
		glTexCoord2f(0,0);
		glVertex3f(-1,-1,0);
		glTexCoord2f(1,0);
		glVertex3f(1,-1,0);
		glTexCoord2f(1,1);
		glVertex3f(1,1,0);
	glEnd();
	
	shader_FB.stop();

	glBindTexture(GL_TEXTURE_2D,0);
	glEnable(GL_DEPTH_TEST);
	glPopAttrib();
	
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_TEXTURE);
	glPopMatrix();
	
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
	glDrawBuffer(GL_BACK);
	
	if (hdr_method==DR_HDR_AUTO)
	{
		glBindTexture(GL_TEXTURE_2D,buffer[DR_TARGET_FRAMEBUFFER]);
		glGenerateMipmapEXT(GL_TEXTURE_2D); 
	}
	glBindTexture(GL_TEXTURE_2D,0);
}

void DeferredRenderer::transformCoordinates()
{
	// camera matrices for shadow map indexing and light calculations
	glGetDoublev(GL_PROJECTION_MATRIX,matrix_P);
	glGetDoublev(GL_MODELVIEW_MATRIX,matrix_MV);
	matmul(matrix_MVP, matrix_P, matrix_MV);

	//if (calc_matrix)
	invert_matrix(matrix_MVP,matrix_MVP_inverse);
	invert_matrix(matrix_P,matrix_P_inverse);
	invert_matrix(matrix_MV,matrix_MV_inverse);

    eye[3] = matrix_MV_inverse[15];
	eye[0] = matrix_MV_inverse[12]/eye[3];
	eye[1] = matrix_MV_inverse[13]/eye[3];
	eye[2] = matrix_MV_inverse[14]/eye[3];

	for (int i=0; i<num_lights; i++)
	{
		if(!lights[i].isActive())
			continue;
		lights[i].update();

		light_array_pos[3*i+0]=lights[i].getTransformedPosition()[0];
		light_array_pos[3*i+1]=lights[i].getTransformedPosition()[1];
		light_array_pos[3*i+2]=lights[i].getTransformedPosition()[2];
		
		if (!lights[i].isShadowEnabled())
			continue;
		//GLdouble lmat[16];
		matmul(light_array_matrix[i], lights[i].getProjectionMatrix(), lights[i].getModelviewMatrix());
	}

	if (root) 
		bbox = root->getBBox();
	else
		bbox.setSymmetrical(Vector3D(0,0,0),Vector3D(200,200,200));
}

void DeferredRenderer::finalRender()
{
	// render glow
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, glow_FBO);
    glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);
	glDisable(GL_DEPTH_TEST);
	glViewport(0,0,width/8, height/8);
    glMatrixMode(GL_TEXTURE);
	glPushMatrix();
	glLoadIdentity();
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glOrtho(-1,1,-1,1,10,-10);
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	glDisable(GL_TEXTURE_2D);
	glDisable(GL_BLEND);
	
	glActiveTextureARB(GL_TEXTURE5);
	glBindTexture(GL_TEXTURE_2D, buffer[DR_TARGET_FRAMEBUFFER]);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 2);
	
	glActiveTextureARB(GL_TEXTURE0_ARB);
	
	shader_Glow.start();
	
	glFrontFace(GL_CCW);
	glBegin(GL_QUADS);
        glColor3f(1,1,1);
	    glNormal3f(0,0,1);
		glTexCoord2f(0,1);
		glVertex3f(-1,1,0);
		glTexCoord2f(0,0);
		glVertex3f(-1,-1,0);
		glTexCoord2f(1,0);
		glVertex3f(1,-1,0);
		glTexCoord2f(1,1);
		glVertex3f(1,1,0);
	glEnd();
	
	shader_Glow.stop();

	// Render to screen
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
    glDrawBuffer(GL_BACK);
	glViewport(0,0,width, height);
	glDisable(GL_BLEND);
    glActiveTextureARB(GL_TEXTURE6);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	glBindTexture(GL_TEXTURE_2D, buffer[DR_TARGET_FRAMEBUFFER]);
	glActiveTextureARB(GL_TEXTURE7);
	glBindTexture(GL_TEXTURE_2D, buffer[DR_TARGET_GLOW]);
	glActiveTextureARB(GL_TEXTURE8);
	glBindTexture(GL_TEXTURE_2D, buffer[DR_TARGET_DEPTH]);
	glActiveTextureARB(GL_TEXTURE0_ARB);

	shader_PostProcess.setProjectionMatrixInverse(matrix_P_inverse);
	shader_PostProcess.start();
	
	glFrontFace(GL_CCW);
	glBegin(GL_QUADS);
        glColor3f(1,1,1);
	    glNormal3f(0,0,1);
		glTexCoord2f(0,1);
		glVertex3f(-1,1,0);
		glTexCoord2f(0,0);
		glVertex3f(-1,-1,0);
		glTexCoord2f(1,0);
		glVertex3f(1,-1,0);
		glTexCoord2f(1,1);
		glVertex3f(1,1,0);
	glEnd();
	
	shader_PostProcess.stop();

	glEnable(GL_DEPTH_TEST);
	
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_TEXTURE);
	glPopMatrix();
}

void DeferredRenderer::draw()
{
	if (root==NULL && (draw_callback == NULL || camera_callback == NULL))
		return;
    if (!is_initialized)
		return;
	
	CHECK_GL_ERROR();
	// Selectively draw shadow maps that need update
	drawShadowMaps();
	force_update_shadows = false;
	CHECK_GL_ERROR();

	// Fill the render target buffers
	drawMultipleRenderTargets();
	CHECK_GL_ERROR();
	
	// Acquire camera matrices and readjust light ECS coords
	transformCoordinates();
	CHECK_GL_ERROR();

	// Render illumination effects --> lighting render target
	drawLighting();
	CHECK_GL_ERROR();
	
	// Composition of the frame buffer
	drawFrameBuffer();
	CHECK_GL_ERROR();

	// Transparent surface rendering
	drawTransparency();
	CHECK_GL_ERROR();
	
	// Post-processing effects and display
	finalRender();
	CHECK_GL_ERROR();
	
	// accumulate and present statistics
	statistics();
	CHECK_GL_ERROR();
}

void DeferredRenderer::show(int target)
{
	static float sweep = 0.9f;
	static int sm = 0; 
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
	glDrawBuffer(GL_BACK);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_LIGHTING);
	glDisable(GL_BLEND);
	glColor4f(1,1,1,1);

	for (int i=0;i<16;i++)
	{
		glActiveTextureARB(GL_TEXTURE0+i);
		glDisable(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, 0);
	}

	glClear(GL_DEPTH_BUFFER_BIT|GL_COLOR_BUFFER_BIT); 

	glMatrixMode(GL_TEXTURE);
	glLoadIdentity();
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(-1,1,-1,1,10,-10);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	
	if (target==DR_TARGET_GLOW)	// using this slot to display the generated voxels of the volume
	{
		GlobalIlluminationRendererIV * gi_iv = dynamic_cast<GlobalIlluminationRendererIV*>(gi_renderer);

		if (root && gi_iv!=NULL)
		{
			glEnable(GL_DEPTH_TEST);
			glClearColor(0,0,0,1.0);
			root->getActiveCamera()->setupViewProjection(width, height);
			root->clear(GL_DEPTH_BUFFER_BIT|GL_COLOR_BUFFER_BIT);

			// draw the normals buffer as voxels in the scene
			gi_iv->drawTexture3D (gi_iv->getNormalBuffer());
			glDisable(GL_DEPTH_TEST);
		}
	}
	else if (target==DR_TARGET_VBO)
	{
		glEnable(GL_DEPTH_TEST);
		GlobalIlluminationRendererIV * gi_iv = dynamic_cast<GlobalIlluminationRendererIV*>(gi_renderer);

		if (root && gi_iv!=NULL)
		{
			root->getActiveCamera()->setupViewProjection(width, height);
			glClearColor(0,0,0,1.0);
			root->clear(GL_DEPTH_BUFFER_BIT|GL_COLOR_BUFFER_BIT);

			BBox3D bx;
			bx = gi_iv->getBBox();

		//	glColor4f(1,0,0,1);
			glMatrixMode(GL_TEXTURE);
			glLoadIdentity();
			glMatrixMode(GL_MODELVIEW);
			glPushMatrix();
			glTranslatef(bx.getCenter().x,bx.getCenter().y,bx.getCenter().z);
			glScalef(bx.getSize().x,bx.getSize().y,bx.getSize().z);
#ifdef USING_GLUT
			glutWireCube(1);
#endif
			glPopMatrix();

			glDisable(GL_CULL_FACE);
		//	glColor4f(0,1,0,0.5);
			glPolygonMode(GL_FRONT_AND_BACK,GL_LINE);
			root->setRenderMode(SCENE_GRAPH_RENDER_MODE_NORMAL);
			glDisable(GL_TEXTURE_3D);
			root->draw();

			glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
			glDisable(GL_CULL_FACE);
			glEnable(GL_BLEND);
			glBlendEquation(GL_ADD);
			glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);

			// draw the z sweep plane in gray
			glBegin(GL_LINE_LOOP);
				glColor4f (1,0.5,0,0.5);
				glNormal3f (0,0,1);
				glVertex3f (bx.getMin().x, bx.getMax().y, bx.getMin().z + bx.getSize().z * sweep);
				glVertex3f (bx.getMin().x, bx.getMin().y, bx.getMin().z + bx.getSize().z * sweep);
				glVertex3f (bx.getMax().x, bx.getMin().y, bx.getMin().z + bx.getSize().z * sweep);
				glVertex3f (bx.getMax().x, bx.getMax().y, bx.getMin().z + bx.getSize().z * sweep);
			glEnd();

			// draw the y sweep plane in gray
			glBegin(GL_LINE_LOOP);
				glNormal3f (0,1,0);
				glVertex3f (bx.getMin().x, bx.getMin().y + bx.getSize().y * sweep, bx.getMax().z);
				glVertex3f (bx.getMin().x, bx.getMin().y + bx.getSize().y * sweep, bx.getMin().z);
				glVertex3f (bx.getMax().x, bx.getMin().y + bx.getSize().y * sweep, bx.getMin().z);
				glVertex3f (bx.getMax().x, bx.getMin().y + bx.getSize().y * sweep, bx.getMax().z);
			glEnd();

			// draw the x sweep plane in gray
			glBegin(GL_LINE_LOOP);
				glNormal3f (1,0,0);
				glVertex3f (bx.getMin().x + bx.getSize().x * sweep, bx.getMin().y, bx.getMax().z);
				glVertex3f (bx.getMin().x + bx.getSize().x * sweep, bx.getMin().y, bx.getMin().z);
				glVertex3f (bx.getMin().x + bx.getSize().x * sweep, bx.getMax().y, bx.getMin().z);
				glVertex3f (bx.getMin().x + bx.getSize().x * sweep, bx.getMax().y, bx.getMax().z);
			glEnd();

			shader_ViewPhotonMap.start();

			glBegin(GL_QUADS);
				// z sweep plane
				glColor4f (1,1,1,1);
				glNormal3f (0,0,1);
				glTexCoord3f (0,1,sweep);	glVertex3f (bx.getMin().x, bx.getMax().y, bx.getMin().z + bx.getSize().z * sweep);
				glTexCoord3f (0,0,sweep);	glVertex3f (bx.getMin().x, bx.getMin().y, bx.getMin().z + bx.getSize().z * sweep);
				glTexCoord3f (1,0,sweep);	glVertex3f (bx.getMax().x, bx.getMin().y, bx.getMin().z + bx.getSize().z * sweep);
				glTexCoord3f (1,1,sweep);	glVertex3f (bx.getMax().x, bx.getMax().y, bx.getMin().z + bx.getSize().z * sweep);

				// z sweep plane
				glNormal3f (0,1,0);
				glTexCoord3f (0,sweep,1);	glVertex3f (bx.getMin().x, bx.getMin().y + bx.getSize().y * sweep, bx.getMax().z);
				glTexCoord3f (0,sweep,0);	glVertex3f (bx.getMin().x, bx.getMin().y + bx.getSize().y * sweep, bx.getMin().z);
				glTexCoord3f (1,sweep,0);	glVertex3f (bx.getMax().x, bx.getMin().y + bx.getSize().y * sweep, bx.getMin().z);
				glTexCoord3f (1,sweep,1);	glVertex3f (bx.getMax().x, bx.getMin().y + bx.getSize().y * sweep, bx.getMax().z);

				// x sweep plane
				glNormal3f (1,0,0);
				glTexCoord3f (sweep,0,1);	glVertex3f (bx.getMin().x + bx.getSize().x * sweep, bx.getMin().y, bx.getMax().z);
				glTexCoord3f (sweep,0,0);	glVertex3f (bx.getMin().x + bx.getSize().x * sweep, bx.getMin().y, bx.getMin().z);
				glTexCoord3f (sweep,1,0);	glVertex3f (bx.getMin().x + bx.getSize().x * sweep, bx.getMax().y, bx.getMin().z);
				glTexCoord3f (sweep,1,1);	glVertex3f (bx.getMin().x + bx.getSize().x * sweep, bx.getMax().y, bx.getMax().z);
			glEnd();

			shader_ViewPhotonMap.stop();

			sweep+=0.0025f;
			if (sweep>=1.0f)
				sweep=0.0f;
		}
		glDisable(GL_DEPTH_TEST);
	}
#if 1
	else if (target==DR_TARGET_DEPTH)
	{
		shader_ViewDepthBuffer.start();

		glBegin(GL_QUADS);
			glNormal3f(0,0,1);
			glTexCoord2f(0,1);	glVertex3f(-1, 1,0);
			glTexCoord2f(0,0);	glVertex3f(-1,-1,0);
			glTexCoord2f(1,0);	glVertex3f( 1,-1,0);
			glTexCoord2f(1,1);	glVertex3f( 1, 1,0);
		glEnd();

		shader_ViewDepthBuffer.stop();
	}
#endif
	else
	{
		glActiveTextureARB(GL_TEXTURE0_ARB);
		glEnable(GL_TEXTURE_2D);

		if (target==DR_TARGET_SHADOWMAP)
		{
			glBindTexture(GL_TEXTURE_2D,lights[sm].getShadowMap());
		//	glBindTexture(GL_TEXTURE_2D,lights[sm].getShadowMapColorBuffer());
		//	glBindTexture(GL_TEXTURE_2D,lights[sm].getShadowMapNormalBuffer());

			sm=(sm+1)%num_lights;
		}
		else
			glBindTexture(GL_TEXTURE_2D,buffer[target]);

		glDisable(GL_LIGHTING);
		glFrontFace(GL_CCW);
		glBegin(GL_QUADS);
			glColor3f(1,1,1);
			glNormal3f(0,0,1);
			glTexCoord2f(0,1);	glVertex3f(-1,1,0);
			glTexCoord2f(0,0);	glVertex3f(-1,-1,0);
			glTexCoord2f(1,0);	glVertex3f(1,-1,0);
			glTexCoord2f(1,1);	glVertex3f(1,1,0);
		glEnd();
		
		glBindTexture(GL_TEXTURE_2D,0);
		glDisable(GL_TEXTURE_2D);
	}

#ifdef USING_GLUT
	static char * DR_TARGET_NAMES[128];
	static bool DR_TARGET_NAMES_inited = false;
	if (! DR_TARGET_NAMES_inited)
	{
		DR_TARGET_NAMES[0] = (char *) "Albedo";
		DR_TARGET_NAMES[1] = (char *) "Normal";
		DR_TARGET_NAMES[2] = (char *) "Specular";
		DR_TARGET_NAMES[3] = (char *) "Lighting";
		DR_TARGET_NAMES[4] = (char *) "Depth (linearized)";
		DR_TARGET_NAMES[5] = (char *) "Framebuffer";
		DR_TARGET_NAMES[6] = (char *) "Shadowmap";
	//	DR_TARGET_NAMES[6] = (char *) "Shadowmap color";
	//	DR_TARGET_NAMES[6] = (char *) "Shadowmap normal";
		DR_TARGET_NAMES[7] = (char *) "Normals Volume buffer";
		DR_TARGET_NAMES[8] = (char *) "Ambient";
		DR_TARGET_NAMES[9] = (char *) "Photon map";

		DR_TARGET_NAMES_inited = true;
	}

	char message[32];
	strcpy_s(message,32,DR_TARGET_NAMES[target]);
	int slen = strlen( DR_TARGET_NAMES[target] );
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glOrtho(0.0, width, 0.0, height, -1, 1);
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	glColor3f(1,1,1);
	glRasterPos3f(10, 10,0.1);
	for (int i = 0; i<slen; i++)
    {
		glutBitmapCharacter(GLUT_BITMAP_9_BY_15, message[i]);
    }
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
#endif

	glEnable(GL_DEPTH_TEST);
}

void DeferredRenderer::setDrawCallback(void (*drw)() )
{
	draw_callback = drw;
}

void DeferredRenderer::resize(int w, int h)
{
	width = w>0?w:1;
	height = h>0?h:1;
	if (resizeBuffers() != DR_ERROR_NONE)
		EAZD_TRACE ("DeferredRenderer::resize() : ERROR - renderer not properly initialized.");
}

void DeferredRenderer::setDOF(float distance, float range)
{
	focal_distance = distance;
	focal_range = range;
}

void DeferredRenderer::initLighting()
{
	// Instanciate the appropriate GI renderer according to GI method.
	float gi_factor, gi_range;
	int gi_bounces, gi_samples;
	char params[64];
	bool gi_reset = false;
	if (gi_renderer)
	{	
		// keep gi settings for new gi renderer
		gi_reset = true;
		gi_bounces = gi_renderer->getBounces();
		gi_samples = gi_renderer->getNumSamples();
		gi_factor  = gi_renderer->getFactor();
		gi_range   = gi_renderer->getRange();
		if (gi_renderer->getParamString())
			strcpy(params,gi_renderer->getParamString());
		
		SAFEDELETE (gi_renderer);
	}

	gi_renderer = new GlobalIlluminationRendererIV();
	if (!gi_renderer)
		is_initialized = false;
	else if (gi_reset)
	{
		gi_renderer->setBounces(gi_bounces);
		gi_renderer->setNumSamples(gi_samples);
		gi_renderer->setFactor(gi_factor);
		gi_renderer->setRange(gi_range);
		gi_renderer->setParamString(params);
	}
	
	if (use_gi)
	{
		ambient_blending = DR_AMBIENT_BLEND_ADD;
		is_initialized &= gi_renderer->init(this);
	}
}

void DeferredRenderer::buildShaders()
{
	printf ("DeferredRenderer::buildShaders\n");

	// (re-)initialization of the shaders
	is_initialized  = shader_MRT.init(this);
	is_initialized &= shader_ClearMRT.init(this);
	is_initialized &= shader_FB.init(this);
	is_initialized &= shader_Lighting.init(this);
	is_initialized &= shader_PostProcess.init(this);
	is_initialized &= shader_Glow.init(this);
	is_initialized &= shader_Trans.init(this);
	is_initialized &= shader_ShadowMap.init(this);
	is_initialized &= shader_ViewDepthBuffer.init(this);
	is_initialized &= shader_ViewPhotonMap.init(this);

	initLighting();

	printf ("End of DeferredRenderer::buildShaders\n");
}

GLuint DeferredRenderer::getBuffer(int id)
{
	return buffer[id];
}

int DeferredRenderer::createLight()
{
	if (num_lights>=DR_NUM_LIGHTS)
		return -1;

	lights[num_lights].enable(false);
	light_array_dir[3*num_lights+2]=-1;
	light_array_col[3*num_lights]=
	light_array_col[3*num_lights+1]=
	light_array_col[3*num_lights+2]= 1;
	light_array_attn[num_lights]=0;
	light_array_active[num_lights]=0;
	light_array_range[num_lights]=100.0f;

	num_lights++;
	buildShaders();
	return num_lights-1;
}

void DeferredRenderer::setLightRanges(int light, float _near, float _far)
{
	lights[light].setRanges(_near,_far);
	light_array_range[light]=_far;
}

void DeferredRenderer::setLightSkipFrames(int light, int sf)
{
	lights[light].skipFrames(sf);
}

void DeferredRenderer::setLightColor(int light, float r, float g, float b)
{
	lights[light].setColor(r,g,b);
	light_array_col[3*light]=r*lights[light].getIntensity();
	light_array_col[3*light+1]=g*lights[light].getIntensity();
	light_array_col[3*light+2]= b*lights[light].getIntensity();
}

void DeferredRenderer::setLightAmbient(int light, float r, float g, float b)
{
	lights[light].setAmbient(r,g,b);
}

void DeferredRenderer::setLightSize(int light, float sz)
{
	lights[light].setSize(sz);
}

void DeferredRenderer::setLightCone(int light, float a)
{
	lights[light].setCone(a);
}

void GenericRenderer::attachLightData(int light, void * dt)
{
	lights[light].data=dt;
}

void DeferredRenderer::setLightIntensity(int light, int intens)
{
	light_array_col[3*light]*=intens/lights[light].getIntensity();
	light_array_col[3*light+1]*=intens/lights[light].getIntensity();
	light_array_col[3*light+2]*=intens/lights[light].getIntensity();
	lights[light].setIntensity(intens);
}

void DeferredRenderer::enableLight(int light, bool ltenable)
{
	lights[light].enable(ltenable);
	light_array_active[light] = ltenable?1:0;
}

void DeferredRenderer::enableLightShadows(int light, bool sh)
{
	lights[light].enableShadows(sh);
	lights[light].setShadowFBO(shadow_fbo.getInstance());
}

void DeferredRenderer::setShadowMethod(int method)
{
	if (shadow_method != method)
	{
		shadow_method = method;
		buildShaders();
	}
}

void DeferredRenderer::setLightAttenuation(int light, bool attn)
{
	lights[light].setAttenuation(attn);
	light_array_attn[light] = attn?1:0;
}

void DeferredRenderer::setShadowResolution(int light, int res)
{
	lights[light].setShadowResolution(res);
}


void DeferredRenderer::setLightPosition(int light, float x, float y, float z)
{
	lights[light].setPosition(x,y,z);
	light_array_pos[3*light+0]=x;
	light_array_pos[3*light+1]=y;
	light_array_pos[3*light+2]=z;
	float * tgt = lights[light].getTarget();
	float dir[3];
	dir[0]= tgt[0]-x; dir[1]= tgt[1]-y; dir[2]= tgt[2]-z;
	float len = sqrt(dir[0]*dir[0]+dir[1]*dir[1]+dir[2]*dir[2]);
	dir[0]/=len;
	dir[1]/=len;
	dir[2]/=len;
	memcpy(light_array_dir+3*light,dir,3*sizeof(float));
}
	
void DeferredRenderer::setLightTarget(int light, float x, float y, float z)
{
	lights[light].setTarget(x,y,z);
	float * pos = lights[light].getPosition();
	float dir[3];
	dir[0]= x-pos[0]; dir[1]= y-pos[1]; dir[2]= y-pos[2];
	float len = sqrt(dir[0]*dir[0]+dir[1]*dir[1]+dir[2]*dir[2]);
	dir[0]/=len;
	dir[1]/=len;
	dir[2]/=len;
	memcpy(light_array_dir+3*light,dir,3*sizeof(float));
}

//------------------------------------------------------------------------------

GLuint DRShadowFBO::getInstance()
{
	if (!glIsFramebufferEXT(fbo))
		glGenFramebuffersEXT(1, &fbo);

	return fbo;
}

void DRShadowFBO::release()
{
	if (glIsFramebufferEXT(fbo))
		glDeleteFramebuffersEXT(1, &fbo);
}

//------------------------------------------------------------------------------
GLuint DRShadowFBO::fbo = 0;

DRLight::DRLight()
{
	memset(matrix_shadow_modelview,0,16*sizeof(GLfloat));
	memset(matrix_shadow_projection,0,16*sizeof(GLfloat));
	matrix_shadow_modelview[0] = matrix_shadow_modelview[5] =
	matrix_shadow_modelview[10] = matrix_shadow_modelview[15] = 1.0f;
	matrix_shadow_projection[0] = matrix_shadow_projection[5] =
	matrix_shadow_projection[10] = matrix_shadow_projection[15] = 1.0f;
	active = 0;
	shadow_active = 0;
	intensity = 1.0f;
	color[0] = color[1] = color[2] = 1.0f;
	ambient[0] = ambient[1] = ambient[2] = 0.05f;
	light_pos[0] = light_pos[2] = 0.0f;
	light_pos[1] = 100.0f;
    light_tgt[0] = light_tgt[1] = light_tgt[2] = 0.0f;
	light_pos_transformed[0] =  light_pos[0];
	light_pos_transformed[1] =  light_pos[1];
	light_pos_transformed[2] =  light_pos[2];
	light_tgt_transformed[0] =  light_tgt[0];
	light_tgt_transformed[1] =  light_tgt[1];
	light_tgt_transformed[2] =  light_tgt[2];
	light_near = 1.0f;
	light_far =  100.0f;
	apperture =  45.0f;
	is_attenuating = false;
	shadow_method = DR_SHADOW_GAUSS;
	shadow_res = 1024;
	shadow_map = 0;
	color_map = 0;
	normal_map = 0;
	shadow_FBO = 0;
	update_shadow_map = false;
	size = 0.0f;
	skip_frames = 0;
	frame_loop = 0;
	data = NULL;
	gi_enabled = false;
	has_extended_data = false;
	is_cone = false;
}

DRLight::~DRLight()
{
	if (glIsTexture(shadow_map))
		glDeleteTextures(1,&shadow_map);
	if (glIsTexture(color_map))
		glDeleteTextures(1,&color_map);
	if (glIsTexture(normal_map))
		glDeleteTextures(1,&normal_map);
}

void DRLight::setPosition(float lpx, float lpy, float lpz)
{
	light_pos[0] = lpx;
	light_pos[1] = lpy;
	light_pos[2] = lpz;
	if (shadow_active)
		update_shadow_map = true;
}

void DRLight::setTarget(float ltx, float lty, float ltz)
{
	light_tgt[0] = ltx;
	light_tgt[1] = lty;
	light_tgt[2] = ltz;
	if (shadow_active)
		update_shadow_map = true;
}

void DRLight::setRanges(float _near, float _far)
{
	light_near = _near;
	light_far = _far;
	if (shadow_active)
		update_shadow_map = true;
}

void DRLight::setAttenuation(bool attn)
{
	is_attenuating = attn;
}

void DRLight::enable(bool en)
{
	int prev = active;
	active = en?1:0;
	if ( (active-prev==1) && shadow_active)
		update_shadow_map = true;
}

void DRLight::enableShadows(bool shadows)
{
	shadow_active = shadows?1:0;
	if (shadow_active && shadow_FBO==0)
	if ( active && shadow_active)
	{
		update_shadow_map = true;
		if (!glIsTexture(shadow_map))
		{
			glGenTextures(1,&shadow_map);
			glBindTexture(GL_TEXTURE_2D, shadow_map);
	        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, shadow_res, shadow_res, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
			glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
			glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
			glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
			glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
			//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
			//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR_MIPMAP_LINEAR);
			glBindTexture(GL_TEXTURE_2D, 0);
		}
		if (!glIsTexture(color_map))
		{
			glGenTextures(1,&color_map);
			glBindTexture(GL_TEXTURE_2D, color_map);
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, shadow_res, shadow_res, 0, GL_RGBA, GL_FLOAT, NULL);
			glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
			glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR_MIPMAP_LINEAR);
			glBindTexture(GL_TEXTURE_2D, 0);
		}
		if (has_extended_data && !glIsTexture(normal_map))
		{
			glGenTextures(1,&normal_map);
			glBindTexture(GL_TEXTURE_2D, normal_map);
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, shadow_res, shadow_res, 0, GL_RGBA, GL_FLOAT, NULL);
			glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
			glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR_MIPMAP_LINEAR);
			glBindTexture(GL_TEXTURE_2D, 0);
		}
	}
}

void DRLight::setupShadowMap()
{
	CHECK_GL_ERROR();

	//if (!glIsFramebufferEXT(shadow_FBO))
	//	EAZD_TRACE("Your Shadow FBO is not valid !!!");
	//glBindTexture(GL_TEXTURE_2D, color_map);
	//glGenerateMipmapEXT(GL_TEXTURE_2D); 
	glBindTexture(GL_TEXTURE_2D, 0);
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, shadow_FBO);
	glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT,
		GL_TEXTURE_2D, shadow_map, 0);
	if (has_extended_data)
	{
		glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT,
			GL_TEXTURE_2D, color_map, 0);
		glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT1_EXT,
			GL_TEXTURE_2D, normal_map, 0);
	}
	CHECK_GL_ERROR();
	
	unsigned int attachments[]={GL_COLOR_ATTACHMENT0_EXT,GL_COLOR_ATTACHMENT1_EXT};
	if (has_extended_data)
		glDrawBuffers(2,attachments);
//	if (glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT)!=GL_FRAMEBUFFER_COMPLETE_EXT)
//		EAZD_TRACE ("DRLight::setupShadowMap() : ERROR - Light FBO init problem.");
	CHECK_GL_ERROR();
	
	glViewport(0,0,shadow_res,shadow_res);
	glMatrixMode(GL_PROJECTION);
	//glPushMatrix();
	glLoadIdentity();
	//gluPerspective(apperture,1,light_near,light_far);
	float h=light_near*tan(3.1415936*apperture/180.0f);
	glFrustum(-h,h,-h,h,light_near,light_far);
	glGetDoublev(GL_PROJECTION_MATRIX,matrix_shadow_projection);
	glMatrixMode(GL_MODELVIEW);
	//glPushMatrix();
	glLoadIdentity();
	if ( ( fabs(light_pos_transformed[0]-light_tgt_transformed[0])<0.001 ) &&
		 ( fabs(light_pos_transformed[2]-light_tgt_transformed[2])<0.001 ) )
		gluLookAt(light_pos_transformed[0],light_pos_transformed[1],light_pos_transformed[2],
			      light_tgt_transformed[0],light_tgt_transformed[1],light_tgt_transformed[2],
				  0,0,1);
	else
		gluLookAt(light_pos_transformed[0],light_pos_transformed[1],light_pos_transformed[2],
			      light_tgt_transformed[0],light_tgt_transformed[1],light_tgt_transformed[2],
				  0,1,0);
	glGetDoublev(GL_MODELVIEW_MATRIX,matrix_shadow_modelview);
	//glPopMatrix();
	//glMatrixMode(GL_PROJECTION);
	//glPopMatrix();
	update_shadow_map = false;
	frame_loop = (frame_loop+1)%(1+skip_frames);
}

void DRLight::transform()
{
	Node3D *nd = (Node3D*)data;
	if (nd)
	{
		Matrix4D mt = nd->getTransform();
		mt.transpose();
		Vector3D newpoint =  mt * Vector3D(light_pos);
		light_pos_transformed[0]=newpoint.x;
		light_pos_transformed[1]=newpoint.y;
		light_pos_transformed[2]=newpoint.z;
		newpoint =  mt * Vector3D(light_tgt);
		light_tgt_transformed[0]=newpoint.x;
		light_tgt_transformed[1]=newpoint.y;
		light_tgt_transformed[2]=newpoint.z;
		// evaluate transformed range:
		float dist = Vector3D(light_pos).distance(Vector3D(light_tgt));
		float dist_new = newpoint.distance(Vector3D(light_pos_transformed));
		float ratio = dist_new/dist;
		light_near_transformed = light_near * ratio;
		light_far_transformed = light_far * ratio;
	}
	else
	{
		light_pos_transformed[0] = light_pos[0];
		light_pos_transformed[1] = light_pos[1];
		light_pos_transformed[2] = light_pos[2];
		light_tgt_transformed[0] = light_tgt[0];
		light_tgt_transformed[1] = light_tgt[1];
		light_tgt_transformed[2] = light_tgt[2];
		light_near_transformed = light_near;
		light_far_transformed = light_far;
	}
}

void DRLight::enableExtendedData(bool e)
{
	has_extended_data = e;
	setShadowResolution(shadow_res);
	setupShadowMap();
}

void DRLight::setShadowResolution(unsigned int res)
{
	if (res<=0)
		return;
	shadow_res = res;
	if (glIsTexture(shadow_map))
		glDeleteTextures(1,&shadow_map);
	glGenTextures(1,&shadow_map);
	glBindTexture(GL_TEXTURE_2D, shadow_map);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, shadow_res, shadow_res, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glBindTexture(GL_TEXTURE_2D, 0);
#define CLAMP_TO_BORDER	
	if (has_extended_data)
	{
#ifdef CLAMP_TO_BORDER		
		GLfloat bc[4];
		bc[0] = bc[1] = bc[2] = 0.0f;
		bc[3] = 1.0f;
		glTexParameterfv(GL_TEXTURE_2D,GL_TEXTURE_BORDER_COLOR,bc);
#endif
		if (glIsTexture(color_map))
			glDeleteTextures(1,&color_map);
		glGenTextures(1,&color_map);
		glBindTexture(GL_TEXTURE_2D, color_map);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, shadow_res, shadow_res, 0, GL_RGBA, GL_FLOAT, NULL);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
#ifdef CLAMP_TO_BORDER		
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
#else
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
#endif
		//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		glBindTexture(GL_TEXTURE_2D, 0);

		if (glIsTexture(normal_map))
			glDeleteTextures(1,&normal_map);
		glGenTextures(1,&normal_map);
		glBindTexture(GL_TEXTURE_2D, normal_map);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, shadow_res, shadow_res, 0, GL_RGBA, GL_FLOAT, NULL);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
#ifdef CLAMP_TO_BORDER		
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
#else
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
#endif
		glBindTexture(GL_TEXTURE_2D, 0);
	}
}

void DRLight::setShadowSamplingMethod(int method)
{
	if (method>0 && method<4)
		shadow_method = method;
}

void DRLight::setColor(float r, float g, float b)
{
	color[0] = r;
	color[1] = g;
	color[2] = b;
}

void DRLight::setAmbient(float r, float g, float b)
{
	ambient[0] = r;
	ambient[1] = g;
	ambient[2] = b;
}

void DRLight::setIntensity(float bright)
{
	intensity = bright;
}

void DRLight::setShadowFBO(GLuint fbo)
{
	shadow_FBO = fbo;
}

