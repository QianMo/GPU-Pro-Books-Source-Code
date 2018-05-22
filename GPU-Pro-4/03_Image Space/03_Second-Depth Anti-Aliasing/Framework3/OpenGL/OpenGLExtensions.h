
/* * * * * * * * * * * * * Author's note * * * * * * * * * * * *\
*   _       _   _       _   _       _   _       _     _ _ _ _   *
*  |_|     |_| |_|     |_| |_|_   _|_| |_|     |_|  _|_|_|_|_|  *
*  |_|_ _ _|_| |_|     |_| |_|_|_|_|_| |_|     |_| |_|_ _ _     *
*  |_|_|_|_|_| |_|     |_| |_| |_| |_| |_|     |_|   |_|_|_|_   *
*  |_|     |_| |_|_ _ _|_| |_|     |_| |_|_ _ _|_|  _ _ _ _|_|  *
*  |_|     |_|   |_|_|_|   |_|     |_|   |_|_|_|   |_|_|_|_|    *
*                                                               *
*                     http://www.humus.name                     *
*                                                                *
* This file is a part of the work done by Humus. You are free to   *
* use the code in any way you like, modified, unmodified or copied   *
* into your own work. However, I expect you to respect these points:  *
*  - If you use this file and its contents unmodified, or use a major *
*    part of this file, please credit the author and leave this note. *
*  - For use in anything commercial, please request my approval.     *
*  - Share your work and ideas too as much as you can.             *
*                                                                *
\* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef _OPENGLEXTENSIONS_H_
#define _OPENGLEXTENSIONS_H_

#include "../Platform.h"

#if defined(_WIN32)

#include <GL/gl.h>

#define wglxGetProcAddress wglGetProcAddress

#elif defined(LINUX)

#define GLX_GLXEXT_PROTOTYPES
#define GL_GLEXT_PROTOTYPES

#ifndef GL_GLEXT_PROTOTYPES
#define GL_ARB_multitexture
#endif

#include <GL/glx.h>
#include <GL/glxext.h>

#ifndef GL_GLEXT_PROTOTYPES
#undef GL_ARB_multitexture
#define PROTOTYPES
#endif

#ifndef INT_PTR
#define INT_PTR int
#endif

#ifndef HANDLE
#define HANDLE void *
#endif


#ifdef GLX_VERSION_1_4
#define wglxGetProcAddress(a) glXGetProcAddress((const GLubyte *) (a))
#else
#define wglxGetProcAddress(a) glXGetProcAddressARB((const GLubyte *) (a))
#endif

#elif defined(__APPLE__)

#define wglxGetProcAddress NSGLGetProcAddress

#include <AGL/agl.h>

#endif


#ifndef APIENTRY
#define APIENTRY
#endif

extern bool GL_ARB_depth_texture_supported;
extern bool GL_ARB_draw_buffers_supported;
extern bool GL_ARB_fragment_program_supported;
extern bool GL_ARB_fragment_shader_supported;
extern bool GL_ARB_multisample_supported;
extern bool GL_ARB_multitexture_supported;
extern bool GL_ARB_occlusion_query_supported;
extern bool GL_ARB_point_parameters_supported;
extern bool GL_ARB_shader_objects_supported;
extern bool GL_ARB_shading_language_100_supported;
extern bool GL_ARB_shadow_supported;
extern bool GL_ARB_shadow_ambient_supported;
extern bool GL_ARB_texture_compression_supported;
extern bool GL_ARB_texture_cube_map_supported;
extern bool GL_ARB_texture_env_add_supported;
extern bool GL_ARB_texture_env_combine_supported;
extern bool GL_ARB_texture_env_crossbar_supported;
extern bool GL_ARB_texture_env_dot3_supported;
extern bool GL_ARB_transpose_matrix_supported;
extern bool GL_ARB_vertex_buffer_object_supported;
extern bool GL_ARB_vertex_program_supported;
extern bool GL_ARB_vertex_shader_supported;
extern bool GL_ARB_window_pos_supported;

extern bool GL_ATI_fragment_shader_supported;
extern bool GL_ATI_separate_stencil_supported;
extern bool GL_ATI_texture_compression_3dc_supported;
extern bool GL_ATI_texture_float_supported;
extern bool GL_ATI_texture_mirror_once_supported;

extern bool GL_EXT_blend_color_supported;
extern bool GL_EXT_blend_func_separate_supported;
extern bool GL_EXT_blend_minmax_supported;
extern bool GL_EXT_blend_subtract_supported;
extern bool GL_EXT_draw_range_elements_supported;
extern bool GL_EXT_fog_coord_supported;
extern bool GL_EXT_framebuffer_object_supported;
extern bool GL_EXT_multi_draw_arrays_supported;
extern bool GL_EXT_packed_pixels_supported;
extern bool GL_EXT_packed_depth_stencil_supported;
extern bool GL_EXT_stencil_wrap_supported;
extern bool GL_EXT_secondary_color_supported;
extern bool GL_EXT_texture3D_supported;
extern bool GL_EXT_texture_compression_s3tc_supported;
extern bool GL_EXT_texture_edge_clamp_supported;
extern bool GL_EXT_texture_filter_anisotropic_supported;
extern bool GL_EXT_texture_lod_bias_supported;

extern bool GL_HP_occlusion_test_supported;

extern bool GL_NV_blend_square_supported;

extern bool GL_SGIS_generate_mipmap_supported;

#if defined(_WIN32)

extern bool WGL_ARB_extensions_string_supported;
extern bool WGL_ARB_make_current_read_supported;
extern bool WGL_ARB_multisample_supported;
extern bool WGL_ARB_pbuffer_supported;
extern bool WGL_ARB_pixel_format_supported;
extern bool WGL_ARB_render_texture_supported;

extern bool WGL_ATI_pixel_format_float_supported;

extern bool WGL_EXT_swap_control_supported;

#define RenderTexture_supported WGL_ARB_render_texture_supported
#define FloatRenderTexture_supported (RenderTexture_supported && WGL_ATI_pixel_format_float_supported)

#elif defined(LINUX)

extern bool GLX_ATI_pixel_format_float_supported;
extern bool GLX_ATI_render_texture_supported;

#define RenderTexture_supported GLX_ATI_render_texture_supported
#define FloatRenderTexture_supported (RenderTexture_supported && GLX_ATI_pixel_format_float_supported)

#elif defined(__APPLE__)

#endif


#define GLSL_supported (GL_ARB_shader_objects_supported && GL_ARB_shading_language_100_supported && GL_ARB_fragment_shader_supported && GL_ARB_vertex_shader_supported)

extern bool GL_1_1_supported;
extern bool GL_1_2_supported;
extern bool GL_1_3_supported;
extern bool GL_1_4_supported;
extern bool GL_1_5_supported;
extern bool GL_2_0_supported;

extern int GLMajorVersion;
extern int GLMinorVersion;
extern int GLReleaseVersion;

#define GLVER(major, minor) ((GLMajorVersion << 8) + GLMinorVersion >= (major << 8) + minor)

#if !defined(GL_ARB_draw_buffers) || defined(PROTOTYPES)
#define GL_ARB_draw_buffers_PROTOTYPES
#endif

#if !defined(GL_ARB_fragment_program) || defined(PROTOTYPES)
#define GL_ARB_fragment_program_PROTOTYPES
#endif

#if !defined(GL_ARB_multisample) || defined(PROTOTYPES)
#define GL_ARB_multisample_PROTOTYPES
#endif

#if !defined(GL_ARB_multitexture) || defined(PROTOTYPES)
#define GL_ARB_multitexture_PROTOTYPES
#endif

#if !defined(GL_ARB_occlusion_query) || defined(PROTOTYPES)
#define GL_ARB_occlusion_query_PROTOTYPES
#endif

#if !defined(GL_ARB_point_parameters) || defined(PROTOTYPES)
#define GL_ARB_point_parameters_PROTOTYPES
#endif

#if !defined(GL_ARB_shader_objects) || defined(PROTOTYPES)
#define GL_ARB_shader_objects_PROTOTYPES
#endif

#if !defined(GL_ARB_texture_compression) || defined(PROTOTYPES)
#define GL_ARB_texture_compression_PROTOTYPES
#endif

#if !defined(GL_ARB_transpose_matrix) || defined(PROTOTYPES)
#define GL_ARB_transpose_matrix_PROTOTYPES
#endif

#if !defined(GL_ARB_vertex_buffer_object) || defined(PROTOTYPES)
#define GL_ARB_vertex_buffer_object_PROTOTYPES
#endif

#if !defined(GL_ARB_vertex_program) || defined(PROTOTYPES)
#define GL_ARB_vertex_program_PROTOTYPES
#endif

#if !defined(GL_ARB_vertex_shader) || defined(PROTOTYPES)
#define GL_ARB_vertex_shader_PROTOTYPES
#endif

#if !defined(GL_ARB_window_pos) || defined(PROTOTYPES)
#define GL_ARB_window_pos_PROTOTYPES
#endif

#if !defined(GL_ATI_fragment_shader) || defined(PROTOTYPES)
#define GL_ATI_fragment_shader_PROTOTYPES
#endif

#if !defined(GL_ATI_separate_stencil) || defined(PROTOTYPES)
#define GL_ATI_separate_stencil_PROTOTYPES
#endif

#if !defined(GL_EXT_blend_color) || defined(PROTOTYPES)
#define GL_EXT_blend_color_PROTOTYPES
#endif

#if !defined(GL_EXT_blend_func_separate) || defined(PROTOTYPES)
#define GL_EXT_blend_func_separate_PROTOTYPES
#endif

#if !defined(GL_EXT_blend_minmax) || defined(PROTOTYPES)
#define GL_EXT_blend_minmax_PROTOTYPES
#endif

#if !defined(GL_EXT_draw_range_elements) || defined(PROTOTYPES)
#define GL_EXT_draw_range_elements_PROTOTYPES
#endif

#if !defined(GL_EXT_fog_coord) || defined(PROTOTYPES)
#define GL_EXT_fog_coord_PROTOTYPES
#endif

#if !defined(GL_EXT_framebuffer_object) || defined(PROTOTYPES)
#define GL_EXT_framebuffer_object_PROTOTYPES
#endif

#if !defined(GL_EXT_framebuffer_object) || defined(PROTOTYPES)
#define GL_EXT_multi_draw_arrays_PROTOTYPES
#endif

#if !defined(GL_EXT_secondary_color) || defined(PROTOTYPES)
#define GL_EXT_secondary_color_PROTOTYPES
#endif

#if !defined(GL_EXT_texture3D) || defined(PROTOTYPES)
#define GL_EXT_texture3D_PROTOTYPES
#endif






#ifndef GL_ARB_depth_texture
#define GL_ARB_depth_texture

#define GL_DEPTH_COMPONENT16_ARB  0x81A5
#define GL_DEPTH_COMPONENT24_ARB  0x81A6
#define GL_DEPTH_COMPONENT32_ARB  0x81A7
#define GL_TEXTURE_DEPTH_SIZE_ARB 0x884A
#define GL_DEPTH_TEXTURE_MODE_ARB 0x884B
#define GL_DEPTH_COMPONENT16      0x81A5
#define GL_DEPTH_COMPONENT24      0x81A6
#define GL_DEPTH_COMPONENT32      0x81A7
#define GL_TEXTURE_DEPTH_SIZE     0x884A
#define GL_DEPTH_TEXTURE_MODE     0x884B

#endif



#ifndef GL_ARB_draw_buffers
#define GL_ARB_draw_buffers

#define GL_MAX_DRAW_BUFFERS_ARB 0x8824
#define GL_DRAW_BUFFER0_ARB     0x8825
#define GL_DRAW_BUFFER1_ARB     0x8826
#define GL_DRAW_BUFFER2_ARB     0x8827
#define GL_DRAW_BUFFER3_ARB     0x8828
#define GL_DRAW_BUFFER4_ARB     0x8829
#define GL_DRAW_BUFFER5_ARB     0x882A
#define GL_DRAW_BUFFER6_ARB     0x882B
#define GL_DRAW_BUFFER7_ARB     0x882C
#define GL_DRAW_BUFFER8_ARB     0x882D
#define GL_DRAW_BUFFER9_ARB     0x882E
#define GL_DRAW_BUFFER10_ARB    0x882F
#define GL_DRAW_BUFFER11_ARB    0x8830
#define GL_DRAW_BUFFER12_ARB    0x8831
#define GL_DRAW_BUFFER13_ARB    0x8832
#define GL_DRAW_BUFFER14_ARB    0x8833
#define GL_DRAW_BUFFER15_ARB    0x8834

typedef GLvoid (APIENTRY * PFNGLDRAWBUFFERSARBPROC) (GLsizei n, const GLenum *bufs);
#endif

#ifdef GL_ARB_draw_buffers_PROTOTYPES
extern PFNGLDRAWBUFFERSARBPROC glDrawBuffersARB;
#endif



#ifndef GL_ARB_fragment_program
#define GL_ARB_fragment_program

#define GL_FRAGMENT_PROGRAM_ARB                    0x8804
#define GL_PROGRAM_ALU_INSTRUCTIONS_ARB            0x8805
#define GL_PROGRAM_TEX_INSTRUCTIONS_ARB            0x8806
#define GL_PROGRAM_TEX_INDIRECTIONS_ARB            0x8807
#define GL_PROGRAM_NATIVE_ALU_INSTRUCTIONS_ARB     0x8808
#define GL_PROGRAM_NATIVE_TEX_INSTRUCTIONS_ARB     0x8809
#define GL_PROGRAM_NATIVE_TEX_INDIRECTIONS_ARB     0x880A
#define GL_MAX_PROGRAM_ALU_INSTRUCTIONS_ARB        0x880B
#define GL_MAX_PROGRAM_TEX_INSTRUCTIONS_ARB        0x880C
#define GL_MAX_PROGRAM_TEX_INDIRECTIONS_ARB        0x880D
#define GL_MAX_PROGRAM_NATIVE_ALU_INSTRUCTIONS_ARB 0x880E
#define GL_MAX_PROGRAM_NATIVE_TEX_INSTRUCTIONS_ARB 0x880F
#define GL_MAX_PROGRAM_NATIVE_TEX_INDIRECTIONS_ARB 0x8810
#define GL_MAX_TEXTURE_COORDS_ARB                  0x8871
#define GL_MAX_TEXTURE_IMAGE_UNITS_ARB             0x8872

#endif


#ifndef GL_ARB_fragment_shader
#define GL_ARB_fragment_shader

#define GL_FRAGMENT_SHADER_ARB                 0x8B30
#define GL_MAX_FRAGMENT_UNIFORM_COMPONENTS_ARB 0x8B49

#endif

#ifndef GL_ARB_half_float_pixel
#define GL_ARB_half_float_pixel

#define GL_HALF_FLOAT_ARB 0x140B

#endif

#ifndef GL_ARB_multisample
#define GL_ARB_multisample

#define GL_MULTISAMPLE_ARB              0x809D
#define GL_SAMPLE_ALPHA_TO_COVERAGE_ARB 0x809E
#define GL_SAMPLE_ALPHA_TO_ONE_ARB      0x809F
#define GL_SAMPLE_COVERAGE_ARB          0x80A0
#define GL_SAMPLE_BUFFERS_ARB           0x80A8
#define GL_SAMPLES_ARB                  0x80A9
#define GL_SAMPLE_COVERAGE_VALUE_ARB    0x80AA
#define GL_SAMPLE_COVERAGE_INVERT_ARB   0x80AB
#define GL_MULTISAMPLE_BIT_ARB          0x20000000

typedef GLvoid (APIENTRY * PFNGLSAMPLECOVERAGEARBPROC)(GLclampf value, GLboolean invert);

#endif

#ifdef GL_ARB_multisample_PROTOTYPES
extern PFNGLSAMPLECOVERAGEARBPROC glSampleCoverageARB;
#endif


#ifndef GL_ARB_multitexture
#define GL_ARB_multitexture

#define GL_ACTIVE_TEXTURE_ARB        0x84E0
#define GL_CLIENT_ACTIVE_TEXTURE_ARB 0x84E1
#define GL_MAX_TEXTURE_UNITS_ARB     0x84E2
#define GL_TEXTURE0_ARB              0x84C0
#define GL_TEXTURE1_ARB              0x84C1
#define GL_TEXTURE2_ARB              0x84C2
#define GL_TEXTURE3_ARB              0x84C3
#define GL_TEXTURE4_ARB              0x84C4
#define GL_TEXTURE5_ARB              0x84C5
#define GL_TEXTURE6_ARB              0x84C6
#define GL_TEXTURE7_ARB              0x84C7
#define GL_TEXTURE8_ARB              0x84C8
#define GL_TEXTURE9_ARB              0x84C9
#define GL_TEXTURE10_ARB             0x84CA
#define GL_TEXTURE11_ARB             0x84CB
#define GL_TEXTURE12_ARB             0x84CC
#define GL_TEXTURE13_ARB             0x84CD
#define GL_TEXTURE14_ARB             0x84CE
#define GL_TEXTURE15_ARB             0x84CF
#define GL_TEXTURE16_ARB             0x84D0
#define GL_TEXTURE17_ARB             0x84D1
#define GL_TEXTURE18_ARB             0x84D2
#define GL_TEXTURE19_ARB             0x84D3
#define GL_TEXTURE20_ARB             0x84D4
#define GL_TEXTURE21_ARB             0x84D5
#define GL_TEXTURE22_ARB             0x84D6
#define GL_TEXTURE23_ARB             0x84D7
#define GL_TEXTURE24_ARB             0x84D8
#define GL_TEXTURE25_ARB             0x84D9
#define GL_TEXTURE26_ARB             0x84DA
#define GL_TEXTURE27_ARB             0x84DB
#define GL_TEXTURE28_ARB             0x84DC
#define GL_TEXTURE29_ARB             0x84DD
#define GL_TEXTURE30_ARB             0x84DE
#define GL_TEXTURE31_ARB             0x84DF

typedef GLvoid (APIENTRY * PFNGLACTIVETEXTUREARBPROC) (GLenum texture);
typedef GLvoid (APIENTRY * PFNGLCLIENTACTIVETEXTUREARBPROC) (GLenum texture);
typedef GLvoid (APIENTRY * PFNGLMULTITEXCOORD1DARBPROC) (GLenum texture, GLdouble s);
typedef GLvoid (APIENTRY * PFNGLMULTITEXCOORD1DVARBPROC) (GLenum texture, const GLdouble *v);
typedef GLvoid (APIENTRY * PFNGLMULTITEXCOORD1FARBPROC) (GLenum texture, GLfloat s);
typedef GLvoid (APIENTRY * PFNGLMULTITEXCOORD1FVARBPROC) (GLenum texture, const GLfloat *v);
typedef GLvoid (APIENTRY * PFNGLMULTITEXCOORD1IARBPROC) (GLenum texture, GLint s);
typedef GLvoid (APIENTRY * PFNGLMULTITEXCOORD1IVARBPROC) (GLenum texture, const GLint *v);
typedef GLvoid (APIENTRY * PFNGLMULTITEXCOORD1SARBPROC) (GLenum texture, GLshort s);
typedef GLvoid (APIENTRY * PFNGLMULTITEXCOORD1SVARBPROC) (GLenum texture, const GLshort *v);
typedef GLvoid (APIENTRY * PFNGLMULTITEXCOORD2DARBPROC) (GLenum texture, GLdouble s, GLdouble t);
typedef GLvoid (APIENTRY * PFNGLMULTITEXCOORD2DVARBPROC) (GLenum texture, const GLdouble *v);
typedef GLvoid (APIENTRY * PFNGLMULTITEXCOORD2FARBPROC) (GLenum texture, GLfloat s, GLfloat t);
typedef GLvoid (APIENTRY * PFNGLMULTITEXCOORD2FVARBPROC) (GLenum texture, const GLfloat *v);
typedef GLvoid (APIENTRY * PFNGLMULTITEXCOORD2IARBPROC) (GLenum texture, GLint s, GLint t);
typedef GLvoid (APIENTRY * PFNGLMULTITEXCOORD2IVARBPROC) (GLenum texture, const GLint *v);
typedef GLvoid (APIENTRY * PFNGLMULTITEXCOORD2SARBPROC) (GLenum texture, GLshort s, GLshort t);
typedef GLvoid (APIENTRY * PFNGLMULTITEXCOORD2SVARBPROC) (GLenum texture, const GLshort *v);
typedef GLvoid (APIENTRY * PFNGLMULTITEXCOORD3DARBPROC) (GLenum texture, GLdouble s, GLdouble t, GLdouble r);
typedef GLvoid (APIENTRY * PFNGLMULTITEXCOORD3DVARBPROC) (GLenum texture, const GLdouble *v);
typedef GLvoid (APIENTRY * PFNGLMULTITEXCOORD3FARBPROC) (GLenum texture, GLfloat s, GLfloat t, GLfloat r);
typedef GLvoid (APIENTRY * PFNGLMULTITEXCOORD3FVARBPROC) (GLenum texture, const GLfloat *v);
typedef GLvoid (APIENTRY * PFNGLMULTITEXCOORD3IARBPROC) (GLenum texture, GLint s, GLint t, GLint r);
typedef GLvoid (APIENTRY * PFNGLMULTITEXCOORD3IVARBPROC) (GLenum texture, const GLint *v);
typedef GLvoid (APIENTRY * PFNGLMULTITEXCOORD3SARBPROC) (GLenum texture, GLshort s, GLshort t, GLshort r);
typedef GLvoid (APIENTRY * PFNGLMULTITEXCOORD3SVARBPROC) (GLenum texture, const GLshort *v);
typedef GLvoid (APIENTRY * PFNGLMULTITEXCOORD4DARBPROC) (GLenum texture, GLdouble s, GLdouble t, GLdouble r, GLdouble q);
typedef GLvoid (APIENTRY * PFNGLMULTITEXCOORD4DVARBPROC) (GLenum texture, const GLdouble *v);
typedef GLvoid (APIENTRY * PFNGLMULTITEXCOORD4FARBPROC) (GLenum texture, GLfloat s, GLfloat t, GLfloat r, GLfloat q);
typedef GLvoid (APIENTRY * PFNGLMULTITEXCOORD4FVARBPROC) (GLenum texture, const GLfloat *v);
typedef GLvoid (APIENTRY * PFNGLMULTITEXCOORD4IARBPROC) (GLenum texture, GLint s, GLint t, GLint r, GLint q);
typedef GLvoid (APIENTRY * PFNGLMULTITEXCOORD4IVARBPROC) (GLenum texture, const GLint *v);
typedef GLvoid (APIENTRY * PFNGLMULTITEXCOORD4SARBPROC) (GLenum texture, GLshort s, GLshort t, GLshort r, GLshort q);
typedef GLvoid (APIENTRY * PFNGLMULTITEXCOORD4SVARBPROC) (GLenum texture, const GLshort *v);
#endif

#ifdef GL_ARB_multitexture_PROTOTYPES
extern PFNGLACTIVETEXTUREARBPROC glActiveTextureARB;
extern PFNGLCLIENTACTIVETEXTUREARBPROC glClientActiveTextureARB;
extern PFNGLMULTITEXCOORD1DARBPROC  glMultiTexCoord1dARB;
extern PFNGLMULTITEXCOORD1DVARBPROC glMultiTexCoord1dvARB;
extern PFNGLMULTITEXCOORD1FARBPROC  glMultiTexCoord1fARB;
extern PFNGLMULTITEXCOORD1FVARBPROC glMultiTexCoord1fvARB;
extern PFNGLMULTITEXCOORD1IARBPROC  glMultiTexCoord1iARB;
extern PFNGLMULTITEXCOORD1IVARBPROC glMultiTexCoord1ivARB;
extern PFNGLMULTITEXCOORD1SARBPROC  glMultiTexCoord1sARB;
extern PFNGLMULTITEXCOORD1SVARBPROC glMultiTexCoord1svARB;
extern PFNGLMULTITEXCOORD2DARBPROC  glMultiTexCoord2dARB;
extern PFNGLMULTITEXCOORD2DVARBPROC glMultiTexCoord2dvARB;
extern PFNGLMULTITEXCOORD2FARBPROC  glMultiTexCoord2fARB;
extern PFNGLMULTITEXCOORD2FVARBPROC glMultiTexCoord2fvARB;
extern PFNGLMULTITEXCOORD2IARBPROC  glMultiTexCoord2iARB;
extern PFNGLMULTITEXCOORD2IVARBPROC glMultiTexCoord2ivARB;
extern PFNGLMULTITEXCOORD2SARBPROC  glMultiTexCoord2sARB;
extern PFNGLMULTITEXCOORD2SVARBPROC glMultiTexCoord2svARB;
extern PFNGLMULTITEXCOORD3DARBPROC  glMultiTexCoord3dARB;
extern PFNGLMULTITEXCOORD3DVARBPROC glMultiTexCoord3dvARB;
extern PFNGLMULTITEXCOORD3FARBPROC  glMultiTexCoord3fARB;
extern PFNGLMULTITEXCOORD3FVARBPROC glMultiTexCoord3fvARB;
extern PFNGLMULTITEXCOORD3IARBPROC  glMultiTexCoord3iARB;
extern PFNGLMULTITEXCOORD3IVARBPROC glMultiTexCoord3ivARB;
extern PFNGLMULTITEXCOORD3SARBPROC  glMultiTexCoord3sARB;
extern PFNGLMULTITEXCOORD3SVARBPROC glMultiTexCoord3svARB;
extern PFNGLMULTITEXCOORD4DARBPROC  glMultiTexCoord4dARB;
extern PFNGLMULTITEXCOORD4DVARBPROC glMultiTexCoord4dvARB;
extern PFNGLMULTITEXCOORD4FARBPROC  glMultiTexCoord4fARB;
extern PFNGLMULTITEXCOORD4FVARBPROC glMultiTexCoord4fvARB;
extern PFNGLMULTITEXCOORD4IARBPROC  glMultiTexCoord4iARB;
extern PFNGLMULTITEXCOORD4IVARBPROC glMultiTexCoord4ivARB;
extern PFNGLMULTITEXCOORD4SARBPROC  glMultiTexCoord4sARB;
extern PFNGLMULTITEXCOORD4SVARBPROC glMultiTexCoord4svARB;
#endif


#ifndef GL_ARB_occlusion_query
#define GL_ARB_occlusion_query

#define GL_SAMPLES_PASSED_ARB         0x8914
#define GL_QUERY_COUNTER_BITS_ARB     0x8864
#define GL_CURRENT_QUERY_ARB          0x8865
#define GL_QUERY_RESULT_ARB           0x8866
#define GL_QUERY_RESULT_AVAILABLE_ARB 0x8867

typedef void (APIENTRY * PFNGLGENQUERIESARBPROC) (GLsizei n, GLuint *ids);
typedef void (APIENTRY * PFNGLDELETEQUERIESARBPROC) (GLsizei n, const GLuint *ids);
typedef void (APIENTRY * PFNGLISQUERYARBPROC) (GLuint id);
typedef void (APIENTRY * PFNGLBEGINQUERYARBPROC) (GLenum target, GLuint id);
typedef void (APIENTRY * PFNGLENDQUERYARBPROC) (GLenum target);
typedef void (APIENTRY * PFNGLGETQUERYIVARBPROC) (GLenum target, GLenum pname, GLint *params);
typedef void (APIENTRY * PFNGLGETQUERYOBJECTIVARBPROC) (GLuint id, GLenum pname, GLint *params);
typedef void (APIENTRY * PFNGLGETQUERYOBJECTUIVARBPROC) (GLuint id, GLenum pname, GLuint *params);
#endif

#ifdef GL_ARB_occlusion_query_PROTOTYPES
extern PFNGLGENQUERIESARBPROC glGenQueriesARB;
extern PFNGLDELETEQUERIESARBPROC glDeleteQueriesARB;
extern PFNGLISQUERYARBPROC glIsQueryARB;
extern PFNGLBEGINQUERYARBPROC glBeginQueryARB;
extern PFNGLENDQUERYARBPROC glEndQueryARB;
extern PFNGLGETQUERYIVARBPROC glGetQueryivARB;
extern PFNGLGETQUERYOBJECTIVARBPROC glGetQueryObjectivARB;
extern PFNGLGETQUERYOBJECTUIVARBPROC glGetQueryObjectuivARB;
#endif


#ifndef GL_ARB_point_parameters
#define GL_ARB_point_parameters

#define GL_POINT_SIZE_MIN_ARB                0x8126
#define GL_POINT_SIZE_MAX_ARB                0x8127
#define GL_POINT_FADE_THRESHOLD_SIZE_ARB     0x8128
#define GL_POINT_DISTANCE_ATTENUATION_ARB    0x8129

typedef GLvoid (APIENTRY * PFNGLPOINTPARAMETERFARBPROC)  (GLenum pname, GLfloat param);
typedef GLvoid (APIENTRY * PFNGLPOINTPARAMETERFVARBPROC)  (GLenum pname, GLfloat *params);
typedef GLvoid (APIENTRY * PFNGLPOINTPARAMETERIARBPROC)  (GLenum pname, GLint param);
typedef GLvoid (APIENTRY * PFNGLPOINTPARAMETERIVARBPROC)  (GLenum pname, GLint *params);
#endif

#ifdef GL_ARB_point_parameters_PROTOTYPES
extern PFNGLPOINTPARAMETERFARBPROC  glPointParameterfARB;
extern PFNGLPOINTPARAMETERFVARBPROC glPointParameterfvARB;
#endif


#ifndef GL_ARB_shader_objects
#define GL_ARB_shader_objects

#define GL_PROGRAM_OBJECT_ARB                   0x8B40

#define GL_OBJECT_TYPE_ARB                      0x8B4E
#define GL_OBJECT_SUBTYPE_ARB                   0x8B4F
#define GL_OBJECT_DELETE_STATUS_ARB             0x8B80
#define GL_OBJECT_COMPILE_STATUS_ARB            0x8B81
#define GL_OBJECT_LINK_STATUS_ARB               0x8B82
#define GL_OBJECT_VALIDATE_STATUS_ARB           0x8B83
#define GL_OBJECT_INFO_LOG_LENGTH_ARB           0x8B84
#define GL_OBJECT_ATTACHED_OBJECTS_ARB          0x8B85
#define GL_OBJECT_ACTIVE_UNIFORMS_ARB           0x8B86
#define GL_OBJECT_ACTIVE_UNIFORM_MAX_LENGTH_ARB 0x8B87
#define GL_OBJECT_SHADER_SOURCE_LENGTH_ARB      0x8B88

#define GL_SHADER_OBJECT_ARB                    0x8B48

#define GL_FLOAT_VEC2_ARB   0x8B50
#define GL_FLOAT_VEC3_ARB   0x8B51
#define GL_FLOAT_VEC4_ARB   0x8B52
#define GL_INT_VEC2_ARB     0x8B53
#define GL_INT_VEC3_ARB     0x8B54
#define GL_INT_VEC4_ARB     0x8B55
#define GL_BOOL_ARB         0x8B56
#define GL_BOOL_VEC2_ARB    0x8B57
#define GL_BOOL_VEC3_ARB    0x8B58
#define GL_BOOL_VEC4_ARB    0x8B59
#define GL_FLOAT_MAT2_ARB   0x8B5A
#define GL_FLOAT_MAT3_ARB   0x8B5B
#define GL_FLOAT_MAT4_ARB   0x8B5C
#define GL_SAMPLER_1D_ARB   0x8B5D
#define GL_SAMPLER_2D_ARB   0x8B5E
#define GL_SAMPLER_3D_ARB   0x8B5F
#define GL_SAMPLER_CUBE_ARB 0x8B60
#define GL_SAMPLER_1D_SHADOW_ARB      0x8B61
#define GL_SAMPLER_2D_SHADOW_ARB      0x8B62
#define GL_SAMPLER_2D_RECT_ARB        0x8B63
#define GL_SAMPLER_2D_RECT_SHADOW_ARB 0x8B64

typedef char         GLcharARB;
typedef unsigned int GLhandleARB;

typedef GLvoid      (APIENTRY *PFNGLDELETEOBJECTARBPROC)(GLhandleARB obj);
typedef GLhandleARB (APIENTRY *PFNGLGETHANDLEARBPROC)(GLenum pname);
typedef GLvoid      (APIENTRY *PFNGLDETACHOBJECTARBPROC)(GLhandleARB containerObj, GLhandleARB attachedObj);
typedef GLhandleARB (APIENTRY *PFNGLCREATESHADEROBJECTARBPROC)(GLenum shaderType);
typedef GLvoid      (APIENTRY *PFNGLSHADERSOURCEARBPROC)(GLhandleARB shaderObj, GLsizei count, const GLcharARB **string, const GLint *length);
typedef GLvoid      (APIENTRY *PFNGLCOMPILESHADERARBPROC)(GLhandleARB shaderObj);
typedef GLhandleARB (APIENTRY *PFNGLCREATEPROGRAMOBJECTARBPROC)(GLvoid);
typedef GLvoid      (APIENTRY *PFNGLATTACHOBJECTARBPROC)(GLhandleARB containerObj, GLhandleARB obj);
typedef GLvoid      (APIENTRY *PFNGLLINKPROGRAMARBPROC)(GLhandleARB programObj);
typedef GLvoid      (APIENTRY *PFNGLUSEPROGRAMOBJECTARBPROC)(GLhandleARB programObj);
typedef GLvoid      (APIENTRY *PFNGLVALIDATEPROGRAMARBPROC)(GLhandleARB programObj);
typedef GLvoid      (APIENTRY *PFNGLUNIFORM1FARBPROC)(GLint location, GLfloat v0);
typedef GLvoid      (APIENTRY *PFNGLUNIFORM2FARBPROC)(GLint location, GLfloat v0, GLfloat v1);
typedef GLvoid      (APIENTRY *PFNGLUNIFORM3FARBPROC)(GLint location, GLfloat v0, GLfloat v1, GLfloat v2);
typedef GLvoid      (APIENTRY *PFNGLUNIFORM4FARBPROC)(GLint location, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3);
typedef GLvoid      (APIENTRY *PFNGLUNIFORM1IARBPROC)(GLint location, GLint v0);
typedef GLvoid      (APIENTRY *PFNGLUNIFORM2IARBPROC)(GLint location, GLint v0, GLint v1);
typedef GLvoid      (APIENTRY *PFNGLUNIFORM3IARBPROC)(GLint location, GLint v0, GLint v1, GLint v2);
typedef GLvoid      (APIENTRY *PFNGLUNIFORM4IARBPROC)(GLint location, GLint v0, GLint v1, GLint v2, GLint v3);
typedef GLvoid      (APIENTRY *PFNGLUNIFORM1FVARBPROC)(GLint location, GLsizei count, GLfloat *value);
typedef GLvoid      (APIENTRY *PFNGLUNIFORM2FVARBPROC)(GLint location, GLsizei count, GLfloat *value);
typedef GLvoid      (APIENTRY *PFNGLUNIFORM3FVARBPROC)(GLint location, GLsizei count, GLfloat *value);
typedef GLvoid      (APIENTRY *PFNGLUNIFORM4FVARBPROC)(GLint location, GLsizei count, GLfloat *value);
typedef GLvoid      (APIENTRY *PFNGLUNIFORM1IVARBPROC)(GLint location, GLsizei count, GLint *value);
typedef GLvoid      (APIENTRY *PFNGLUNIFORM2IVARBPROC)(GLint location, GLsizei count, GLint *value);
typedef GLvoid      (APIENTRY *PFNGLUNIFORM3IVARBPROC)(GLint location, GLsizei count, GLint *value);
typedef GLvoid      (APIENTRY *PFNGLUNIFORM4IVARBPROC)(GLint location, GLsizei count, GLint *value);
typedef GLvoid      (APIENTRY *PFNGLUNIFORMMATRIX2FVARBPROC)(GLint location, GLsizei count, GLboolean transpose, GLfloat *value);
typedef GLvoid      (APIENTRY *PFNGLUNIFORMMATRIX3FVARBPROC)(GLint location, GLsizei count, GLboolean transpose, GLfloat *value);
typedef GLvoid      (APIENTRY *PFNGLUNIFORMMATRIX4FVARBPROC)(GLint location, GLsizei count, GLboolean transpose, GLfloat *value);
typedef GLvoid      (APIENTRY *PFNGLGETOBJECTPARAMETERFVARBPROC)(GLhandleARB obj, GLenum pname, GLfloat *params);
typedef GLvoid      (APIENTRY *PFNGLGETOBJECTPARAMETERIVARBPROC)(GLhandleARB obj, GLenum pname, GLint *params);
typedef GLvoid      (APIENTRY *PFNGLGETINFOLOGARBPROC)(GLhandleARB obj, GLsizei maxLength, GLsizei *length, GLcharARB *infoLog);
typedef GLvoid      (APIENTRY *PFNGLGETATTACHEDOBJECTSARBPROC)(GLhandleARB containerObj, GLsizei maxCount, GLsizei *count, GLhandleARB *obj);
typedef GLint       (APIENTRY *PFNGLGETUNIFORMLOCATIONARBPROC)(GLhandleARB programObj, const GLcharARB *name);
typedef GLvoid      (APIENTRY *PFNGLGETACTIVEUNIFORMARBPROC)(GLhandleARB programObj, GLuint index, GLsizei maxLength, GLsizei *length, GLint *size, GLenum *type, GLcharARB *name);
typedef GLvoid      (APIENTRY *PFNGLGETUNIFORMFVARBPROC)(GLhandleARB programObj, GLint location, GLfloat *params);
typedef GLvoid      (APIENTRY *PFNGLGETUNIFORMIVARBPROC)(GLhandleARB programObj, GLint location, GLint *params);
typedef GLvoid      (APIENTRY *PFNGLGETSHADERSOURCEARBPROC)(GLhandleARB obj, GLsizei maxLength, GLsizei *length, GLcharARB *source);
#endif

#ifdef GL_ARB_shader_objects_PROTOTYPES
extern PFNGLDELETEOBJECTARBPROC glDeleteObjectARB;
extern PFNGLGETHANDLEARBPROC glGetHandleARB;
extern PFNGLDETACHOBJECTARBPROC glDetachObjectARB;
extern PFNGLCREATESHADEROBJECTARBPROC glCreateShaderObjectARB;
extern PFNGLSHADERSOURCEARBPROC glShaderSourceARB;
extern PFNGLCOMPILESHADERARBPROC glCompileShaderARB;
extern PFNGLCREATEPROGRAMOBJECTARBPROC glCreateProgramObjectARB;
extern PFNGLATTACHOBJECTARBPROC glAttachObjectARB;
extern PFNGLLINKPROGRAMARBPROC glLinkProgramARB;
extern PFNGLUSEPROGRAMOBJECTARBPROC glUseProgramObjectARB;
extern PFNGLVALIDATEPROGRAMARBPROC glValidateProgramARB;

extern PFNGLUNIFORM1FARBPROC glUniform1fARB;
extern PFNGLUNIFORM2FARBPROC glUniform2fARB;
extern PFNGLUNIFORM3FARBPROC glUniform3fARB;
extern PFNGLUNIFORM4FARBPROC glUniform4fARB;
extern PFNGLUNIFORM1IARBPROC glUniform1iARB;
extern PFNGLUNIFORM2IARBPROC glUniform2iARB;
extern PFNGLUNIFORM3IARBPROC glUniform3iARB;
extern PFNGLUNIFORM4IARBPROC glUniform4iARB;
extern PFNGLUNIFORM1FVARBPROC glUniform1fvARB;
extern PFNGLUNIFORM2FVARBPROC glUniform2fvARB;
extern PFNGLUNIFORM3FVARBPROC glUniform3fvARB;
extern PFNGLUNIFORM4FVARBPROC glUniform4fvARB;
extern PFNGLUNIFORM1IVARBPROC glUniform1ivARB;
extern PFNGLUNIFORM2IVARBPROC glUniform2ivARB;
extern PFNGLUNIFORM3IVARBPROC glUniform3ivARB;
extern PFNGLUNIFORM4IVARBPROC glUniform4ivARB;

extern PFNGLUNIFORMMATRIX2FVARBPROC glUniformMatrix2fvARB;
extern PFNGLUNIFORMMATRIX3FVARBPROC glUniformMatrix3fvARB;
extern PFNGLUNIFORMMATRIX4FVARBPROC glUniformMatrix4fvARB;

extern PFNGLGETOBJECTPARAMETERFVARBPROC glGetObjectParameterfvARB;
extern PFNGLGETOBJECTPARAMETERIVARBPROC glGetObjectParameterivARB;
extern PFNGLGETINFOLOGARBPROC glGetInfoLogARB;
extern PFNGLGETATTACHEDOBJECTSARBPROC glGetAttachedObjectsARB;
extern PFNGLGETUNIFORMLOCATIONARBPROC glGetUniformLocationARB;
extern PFNGLGETACTIVEUNIFORMARBPROC glGetActiveUniformARB;
extern PFNGLGETUNIFORMFVARBPROC glGetUniformfvARB;
extern PFNGLGETUNIFORMIVARBPROC glGetUniformivARB;
extern PFNGLGETSHADERSOURCEARBPROC glGetShaderSourceARB;
#endif


#ifndef GL_ARB_shading_language_100
#define GL_ARB_shading_language_100

#define GL_SHADING_LANGUAGE_VERSION_ARB 0x8B8C

#endif

#ifndef GL_ARB_shadow
#define GL_ARB_shadow

#define GL_TEXTURE_COMPARE_MODE_ARB     0x884C
#define GL_TEXTURE_COMPARE_FUNC_ARB     0X884D
#define GL_COMPARE_R_TO_TEXTURE_ARB     0x884E

#endif


#ifndef GL_ARB_shadow_ambient
#define GL_ARB_shadow_ambient

#define GL_TEXTURE_COMPARE_FAIL_VALUE_ARB 0x80BF
#define GL_TEXTURE_COMPARE_FAIL_VALUE     0x80BF

#endif


#ifndef GL_ARB_texture_compression
#define GL_ARB_texture_compression

#define GL_COMPRESSED_ALPHA_ARB               0x84E9
#define GL_COMPRESSED_LUMINANCE_ARB           0x84EA
#define GL_COMPRESSED_LUMINANCE_ALPHA_ARB     0x84EB
#define GL_COMPRESSED_INTENSITY_ARB           0x84EC
#define GL_COMPRESSED_RGB_ARB                 0x84ED
#define GL_COMPRESSED_RGBA_ARB                0x84EE
#define GL_TEXTURE_COMPRESSION_HINT_ARB       0x84EF
#define GL_TEXTURE_COMPRESSED_IMAGE_SIZE_ARB  0x86A0
#define GL_TEXTURE_COMPRESSED_ARB             0x86A1
#define GL_NUM_COMPRESSED_TEXTURE_FORMATS_ARB 0x86A2
#define GL_COMPRESSED_TEXTURE_FORMATS_ARB     0x86A3

typedef GLvoid (APIENTRY * PFNGLCOMPRESSEDTEXIMAGE1DARBPROC)(GLenum target, GLint level, GLenum internalFormat, GLsizei width, GLint border, GLsizei imageSize, const GLvoid *data);
typedef GLvoid (APIENTRY * PFNGLCOMPRESSEDTEXIMAGE2DARBPROC)(GLenum target, GLint level, GLenum internalFormat, GLsizei width, GLsizei height, GLint border, GLsizei imageSize, const GLvoid *data);
typedef GLvoid (APIENTRY * PFNGLCOMPRESSEDTEXIMAGE3DARBPROC)(GLenum target, GLint level, GLenum internalFormat, GLsizei width, GLsizei height, GLsizei depth, GLint border, GLsizei imageSize, const GLvoid *data);
typedef GLvoid (APIENTRY * PFNGLCOMPRESSEDTEXSUBIMAGE1DARBPROC)(GLenum target, GLint level, GLint xoffset, GLsizei width, GLenum format, GLsizei imageSize, const GLvoid *data);
typedef GLvoid (APIENTRY * PFNGLCOMPRESSEDTEXSUBIMAGE2DARBPROC)(GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLsizei imageSize, const GLvoid *data);
typedef GLvoid (APIENTRY * PFNGLCOMPRESSEDTEXSUBIMAGE3DARBPROC)(GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLsizei imageSize, const GLvoid *data);
typedef GLvoid (APIENTRY * PFNGLGETCOMPRESSEDTEXIMAGEARBPROC)(GLenum target, GLint lod, GLvoid *img);
#endif

#ifdef GL_ARB_texture_compression_PROTOTYPES
extern PFNGLCOMPRESSEDTEXIMAGE1DARBPROC    glCompressedTexImage1DARB;
extern PFNGLCOMPRESSEDTEXIMAGE2DARBPROC    glCompressedTexImage2DARB;
extern PFNGLCOMPRESSEDTEXIMAGE3DARBPROC    glCompressedTexImage3DARB;
extern PFNGLCOMPRESSEDTEXSUBIMAGE1DARBPROC glCompressedTexSubImage1DARB;
extern PFNGLCOMPRESSEDTEXSUBIMAGE2DARBPROC glCompressedTexSubImage2DARB;
extern PFNGLCOMPRESSEDTEXSUBIMAGE3DARBPROC glCompressedTexSubImage3DARB;
extern PFNGLGETCOMPRESSEDTEXIMAGEARBPROC   glGetCompressedTexImageARB;
#endif


#ifndef GL_ARB_texture_cube_map
#define GL_ARB_texture_cube_map

#define GL_NORMAL_MAP_ARB                  0x8511
#define GL_REFLECTION_MAP_ARB              0x8512
#define GL_TEXTURE_CUBE_MAP_ARB            0x8513
#define GL_TEXTURE_BINDING_CUBE_MAP_ARB    0x8514
#define GL_TEXTURE_CUBE_MAP_POSITIVE_X_ARB 0x8515
#define GL_TEXTURE_CUBE_MAP_NEGATIVE_X_ARB 0x8516
#define GL_TEXTURE_CUBE_MAP_POSITIVE_Y_ARB 0x8517
#define GL_TEXTURE_CUBE_MAP_NEGATIVE_Y_ARB 0x8518
#define GL_TEXTURE_CUBE_MAP_POSITIVE_Z_ARB 0x8519
#define GL_TEXTURE_CUBE_MAP_NEGATIVE_Z_ARB 0x851A
#define GL_PROXY_TEXTURE_CUBE_MAP_ARB      0x851B
#define GL_MAX_CUBE_MAP_TEXTURE_SIZE_ARB   0x851C

#endif


#ifndef GL_ARB_texture_env_combine
#define GL_ARB_texture_env_combine

#define GL_COMBINE_ARB        0x8570
#define GL_COMBINE_RGB_ARB    0x8571
#define GL_COMBINE_ALPHA_ARB  0x8572
#define GL_SOURCE0_RGB_ARB    0x8580
#define GL_SOURCE1_RGB_ARB    0x8581
#define GL_SOURCE2_RGB_ARB    0x8582
#define GL_SOURCE0_ALPHA_ARB  0x8588
#define GL_SOURCE1_ALPHA_ARB  0x8589
#define GL_SOURCE2_ALPHA_ARB  0x858A
#define GL_OPERAND0_RGB_ARB   0x8590
#define GL_OPERAND1_RGB_ARB   0x8591
#define GL_OPERAND2_RGB_ARB   0x8592
#define GL_OPERAND0_ALPHA_ARB 0x8598
#define GL_OPERAND1_ALPHA_ARB 0x8599
#define GL_OPERAND2_ALPHA_ARB 0x859A
#define GL_RGB_SCALE_ARB      0x8573
#define GL_ADD_SIGNED_ARB     0x8574
#define GL_INTERPOLATE_ARB    0x8575
#define GL_CONSTANT_ARB       0x8576
#define GL_PRIMARY_COLOR_ARB  0x8577
#define GL_PREVIOUS_ARB       0x8578
#define GL_SUBTRACT_ARB       0x84E7

#endif


#ifndef GL_ARB_texture_env_dot3
#define GL_ARB_texture_env_dot3

#define GL_DOT3_RGB_ARB  0x86AE
#define GL_DOT3_RGBA_ARB 0x86AF

#endif


#ifndef GL_ARB_transpose_matrix
#define GL_ARB_transpose_matrix

#define GL_TRANSPOSE_MODELVIEW_MATRIX_ARB  0x84E3
#define GL_TRANSPOSE_PROJECTION_MATRIX_ARB 0x84E4
#define GL_TRANSPOSE_TEXTURE_MATRIX_ARB    0x84E5
#define GL_TRANSPOSE_COLOR_MATRIX_ARB      0x84E6

typedef GLvoid (APIENTRY * PFNGLLOADTRANSPOSEMATRIXFARBPROC)(const GLfloat m[16]);
typedef GLvoid (APIENTRY * PFNGLLOADTRANSPOSEMATRIXDARBPROC)(const GLdouble m[16]);
typedef GLvoid (APIENTRY * PFNGLMULTTRANSPOSEMATRIXFARBPROC)(const GLfloat m[16]);
typedef GLvoid (APIENTRY * PFNGLMULTTRANSPOSEMATRIXDARBPROC)(const GLdouble m[16]);
#endif

#ifdef GL_ARB_transpose_matrix_PROTOTYPES
extern PFNGLLOADTRANSPOSEMATRIXFARBPROC glLoadTransposeMatrixfARB;
extern PFNGLLOADTRANSPOSEMATRIXDARBPROC glLoadTransposeMatrixdARB;
extern PFNGLMULTTRANSPOSEMATRIXFARBPROC glMultTransposeMatrixfARB;
extern PFNGLMULTTRANSPOSEMATRIXDARBPROC glMultTransposeMatrixdARB;
#endif



#ifndef BUFFER_OFFSET
#define BUFFER_OFFSET(i) ((char *) NULL + (i))
#endif

#ifndef GL_ARB_vertex_buffer_object
#define GL_ARB_vertex_buffer_object

typedef INT_PTR GLintptrARB;
typedef INT_PTR GLsizeiptrARB;

#define GL_ARRAY_BUFFER_ARB           0x8892
#define GL_ELEMENT_ARRAY_BUFFER_ARB   0x8893

#define GL_ARRAY_BUFFER_BINDING_ARB                 0x8894
#define GL_ELEMENT_ARRAY_BUFFER_BINDING_ARB         0x8895
#define GL_VERTEX_ARRAY_BUFFER_BINDING_ARB          0x8896
#define GL_NORMAL_ARRAY_BUFFER_BINDING_ARB          0x8897
#define GL_COLOR_ARRAY_BUFFER_BINDING_ARB           0x8898
#define GL_INDEX_ARRAY_BUFFER_BINDING_ARB           0x8899
#define GL_TEXTURE_COORD_ARRAY_BUFFER_BINDING_ARB   0x889A
#define GL_EDGE_FLAG_ARRAY_BUFFER_BINDING_ARB       0x889B
#define GL_SECONDARY_COLOR_ARRAY_BUFFER_BINDING_ARB 0x889C
#define GL_FOG_COORDINATE_ARRAY_BUFFER_BINDING_ARB  0x889D
#define GL_WEIGHT_ARRAY_BUFFER_BINDING_ARB          0x889E

#define GL_VERTEX_ATTRIB_ARRAY_BUFFER_BINDING_ARB   0x889F

#define GL_STREAM_DRAW_ARB        0x88E0
#define GL_STREAM_READ_ARB        0x88E1
#define GL_STREAM_COPY_ARB        0x88E2
#define GL_STATIC_DRAW_ARB        0x88E4
#define GL_STATIC_READ_ARB        0x88E5
#define GL_STATIC_COPY_ARB        0x88E6
#define GL_DYNAMIC_DRAW_ARB       0x88E8
#define GL_DYNAMIC_READ_ARB       0x88E9
#define GL_DYNAMIC_COPY_ARB       0x88EA

#define GL_READ_ONLY_ARB          0x88B8
#define GL_WRITE_ONLY_ARB         0x88B9
#define GL_READ_WRITE_ARB         0x88BA

#define GL_BUFFER_SIZE_ARB        0x8764
#define GL_BUFFER_USAGE_ARB       0x8765
#define GL_BUFFER_ACCESS_ARB      0x88BB
#define GL_BUFFER_MAPPED_ARB      0x88BC

#define GL_BUFFER_MAP_POINTER_ARB 0x88BD

typedef GLvoid      (APIENTRY * PFNGLBINDBUFFERARBPROC) (GLenum target, GLuint buffer);
typedef GLvoid      (APIENTRY * PFNGLDELETEBUFFERSARBPROC) (GLsizei n, const GLuint *buffers);
typedef GLvoid      (APIENTRY * PFNGLGENBUFFERSARBPROC) (GLsizei n, GLuint *buffers);
typedef GLboolean   (APIENTRY * PFNGLISBUFFERARBPROC) (GLuint buffer);
typedef GLvoid      (APIENTRY * PFNGLBUFFERDATAARBPROC) (GLenum target, GLsizeiptrARB size, const GLvoid *data, GLenum usage);
typedef GLvoid      (APIENTRY * PFNGLBUFFERSUBDATAARBPROC) (GLenum target, GLintptrARB offset, GLsizeiptrARB size, const GLvoid *data);
typedef GLvoid      (APIENTRY * PFNGLGETBUFFERSUBDATAARBPROC) (GLenum target, GLintptrARB offset, GLsizeiptrARB size, GLvoid *data);
typedef GLvoid *    (APIENTRY * PFNGLMAPBUFFERARBPROC) (GLenum target, GLenum access);
typedef GLboolean   (APIENTRY * PFNGLUNMAPBUFFERARBPROC) (GLenum target);
typedef GLvoid      (APIENTRY * PFNGLGETBUFFERPARAMETERIVARBPROC) (GLenum target, GLenum pname, GLint *params);
typedef GLvoid      (APIENTRY * PFNGLGETBUFFERPOINTERVARBPROC) (GLenum target, GLenum pname, GLvoid **params);
#endif

#ifdef GL_ARB_vertex_buffer_object_PROTOTYPES
extern PFNGLBINDBUFFERARBPROC glBindBufferARB;
extern PFNGLDELETEBUFFERSARBPROC glDeleteBuffersARB;
extern PFNGLGENBUFFERSARBPROC glGenBuffersARB;
extern PFNGLISBUFFERARBPROC glIsBufferARB;

extern PFNGLBUFFERDATAARBPROC glBufferDataARB;
extern PFNGLBUFFERSUBDATAARBPROC glBufferSubDataARB;
extern PFNGLGETBUFFERSUBDATAARBPROC glGetBufferSubDataARB;

extern PFNGLMAPBUFFERARBPROC glMapBufferARB;
extern PFNGLUNMAPBUFFERARBPROC glUnmapBufferARB;

extern PFNGLGETBUFFERPARAMETERIVARBPROC glGetBufferParameterivARB;
extern PFNGLGETBUFFERPOINTERVARBPROC glGetBufferPointervARB;
#endif


#ifndef GL_ARB_vertex_program
#define GL_ARB_vertex_program

#define GL_VERTEX_PROGRAM_ARB                       0x8620
#define GL_VERTEX_PROGRAM_POINT_SIZE_ARB            0x8642
#define GL_VERTEX_PROGRAM_TWO_SIDE_ARB              0x8643
#define GL_COLOR_SUM_ARB                            0x8458
#define GL_PROGRAM_FORMAT_ASCII_ARB                 0x8875
#define GL_VERTEX_ATTRIB_ARRAY_ENABLED_ARB          0x8622
#define GL_VERTEX_ATTRIB_ARRAY_SIZE_ARB             0x8623
#define GL_VERTEX_ATTRIB_ARRAY_STRIDE_ARB           0x8624
#define GL_VERTEX_ATTRIB_ARRAY_TYPE_ARB             0x8625
#define GL_VERTEX_ATTRIB_ARRAY_NORMALIZED_ARB       0x886A
#define GL_CURRENT_VERTEX_ATTRIB_ARB                0x8626
#define GL_VERTEX_ATTRIB_ARRAY_POINTER_ARB          0x8645
#define GL_PROGRAM_LENGTH_ARB                       0x8627
#define GL_PROGRAM_FORMAT_ARB                       0x8876
#define GL_PROGRAM_BINDING_ARB                      0x8677
#define GL_PROGRAM_INSTRUCTIONS_ARB                 0x88A0
#define GL_MAX_PROGRAM_INSTRUCTIONS_ARB             0x88A1
#define GL_PROGRAM_NATIVE_INSTRUCTIONS_ARB          0x88A2
#define GL_MAX_PROGRAM_NATIVE_INSTRUCTIONS_ARB      0x88A3
#define GL_PROGRAM_TEMPORARIES_ARB                  0x88A4
#define GL_MAX_PROGRAM_TEMPORARIES_ARB              0x88A5
#define GL_PROGRAM_NATIVE_TEMPORARIES_ARB           0x88A6
#define GL_MAX_PROGRAM_NATIVE_TEMPORARIES_ARB       0x88A7
#define GL_PROGRAM_PARAMETERS_ARB                   0x88A8
#define GL_MAX_PROGRAM_PARAMETERS_ARB               0x88A9
#define GL_PROGRAM_NATIVE_PARAMETERS_ARB            0x88AA
#define GL_MAX_PROGRAM_NATIVE_PARAMETERS_ARB        0x88AB
#define GL_PROGRAM_ATTRIBS_ARB                      0x88AC
#define GL_MAX_PROGRAM_ATTRIBS_ARB                  0x88AD
#define GL_PROGRAM_NATIVE_ATTRIBS_ARB               0x88AE
#define GL_MAX_PROGRAM_NATIVE_ATTRIBS_ARB           0x88AF
#define GL_PROGRAM_ADDRESS_REGISTERS_ARB            0x88B0
#define GL_MAX_PROGRAM_ADDRESS_REGISTERS_ARB        0x88B1
#define GL_PROGRAM_NATIVE_ADDRESS_REGISTERS_ARB     0x88B2
#define GL_MAX_PROGRAM_NATIVE_ADDRESS_REGISTERS_ARB 0x88B3
#define GL_MAX_PROGRAM_LOCAL_PARAMETERS_ARB         0x88B4
#define GL_MAX_PROGRAM_ENV_PARAMETERS_ARB           0x88B5
#define GL_PROGRAM_UNDER_NATIVE_LIMITS_ARB          0x88B6
#define GL_PROGRAM_STRING_ARB                       0x8628
#define GL_PROGRAM_ERROR_POSITION_ARB               0x864B
#define GL_CURRENT_MATRIX_ARB                       0x8641
#define GL_TRANSPOSE_CURRENT_MATRIX_ARB             0x88B7
#define GL_CURRENT_MATRIX_STACK_DEPTH_ARB           0x8640
#define GL_MAX_VERTEX_ATTRIBS_ARB                   0x8869
#define GL_MAX_PROGRAM_MATRICES_ARB                 0x862F
#define GL_MAX_PROGRAM_MATRIX_STACK_DEPTH_ARB       0x862E
#define GL_PROGRAM_ERROR_STRING_ARB                 0x8874
#define GL_MATRIX0_ARB  0x88C0
#define GL_MATRIX1_ARB  0x88C1
#define GL_MATRIX2_ARB  0x88C2
#define GL_MATRIX3_ARB  0x88C3
#define GL_MATRIX4_ARB  0x88C4
#define GL_MATRIX5_ARB  0x88C5
#define GL_MATRIX6_ARB  0x88C6
#define GL_MATRIX7_ARB  0x88C7
#define GL_MATRIX8_ARB  0x88C8
#define GL_MATRIX9_ARB  0x88C9
#define GL_MATRIX10_ARB 0x88CA
#define GL_MATRIX11_ARB 0x88CB
#define GL_MATRIX12_ARB 0x88CC
#define GL_MATRIX13_ARB 0x88CD
#define GL_MATRIX14_ARB 0x88CE
#define GL_MATRIX15_ARB 0x88CF
#define GL_MATRIX16_ARB 0x88D0
#define GL_MATRIX17_ARB 0x88D1
#define GL_MATRIX18_ARB 0x88D2
#define GL_MATRIX19_ARB 0x88D3
#define GL_MATRIX20_ARB 0x88D4
#define GL_MATRIX21_ARB 0x88D5
#define GL_MATRIX22_ARB 0x88D6
#define GL_MATRIX23_ARB 0x88D7
#define GL_MATRIX24_ARB 0x88D8
#define GL_MATRIX25_ARB 0x88D9
#define GL_MATRIX26_ARB 0x88DA
#define GL_MATRIX27_ARB 0x88DB
#define GL_MATRIX28_ARB 0x88DC
#define GL_MATRIX29_ARB 0x88DD
#define GL_MATRIX30_ARB 0x88DE
#define GL_MATRIX31_ARB 0x88DF

typedef GLvoid (APIENTRY *PFNGLVERTEXATTRIB1SARBPROC)(GLuint index, GLshort x);
typedef GLvoid (APIENTRY *PFNGLVERTEXATTRIB1FARBPROC)(GLuint index, GLfloat x);
typedef GLvoid (APIENTRY *PFNGLVERTEXATTRIB1DARBPROC)(GLuint index, GLdouble x);
typedef GLvoid (APIENTRY *PFNGLVERTEXATTRIB2SARBPROC)(GLuint index, GLshort x, GLshort y);
typedef GLvoid (APIENTRY *PFNGLVERTEXATTRIB2FARBPROC)(GLuint index, GLfloat x, GLfloat y);
typedef GLvoid (APIENTRY *PFNGLVERTEXATTRIB2DARBPROC)(GLuint index, GLdouble x, GLdouble y);
typedef GLvoid (APIENTRY *PFNGLVERTEXATTRIB3SARBPROC)(GLuint index, GLshort x, GLshort y, GLshort z);
typedef GLvoid (APIENTRY *PFNGLVERTEXATTRIB3FARBPROC)(GLuint index, GLfloat x, GLfloat y, GLfloat z);
typedef GLvoid (APIENTRY *PFNGLVERTEXATTRIB3DARBPROC)(GLuint index, GLdouble x, GLdouble y, GLdouble z);
typedef GLvoid (APIENTRY *PFNGLVERTEXATTRIB4SARBPROC)(GLuint index, GLshort x, GLshort y, GLshort z, GLshort w);
typedef GLvoid (APIENTRY *PFNGLVERTEXATTRIB4FARBPROC)(GLuint index, GLfloat x, GLfloat y, GLfloat z, GLfloat w);
typedef GLvoid (APIENTRY *PFNGLVERTEXATTRIB4DARBPROC)(GLuint index, GLdouble x, GLdouble y, GLdouble z, GLdouble w);
typedef GLvoid (APIENTRY *PFNGLVERTEXATTRIB4NUBARBPROC)(GLuint index, GLubyte x, GLubyte y, GLubyte z, GLubyte w);
typedef GLvoid (APIENTRY *PFNGLVERTEXATTRIB1SVARBPROC)(GLuint index, const GLshort *v);
typedef GLvoid (APIENTRY *PFNGLVERTEXATTRIB1FVARBPROC)(GLuint index, const GLfloat *v);
typedef GLvoid (APIENTRY *PFNGLVERTEXATTRIB1DVARBPROC)(GLuint index, const GLdouble *v);
typedef GLvoid (APIENTRY *PFNGLVERTEXATTRIB2SVARBPROC)(GLuint index, const GLshort *v);
typedef GLvoid (APIENTRY *PFNGLVERTEXATTRIB2FVARBPROC)(GLuint index, const GLfloat *v);
typedef GLvoid (APIENTRY *PFNGLVERTEXATTRIB2DVARBPROC)(GLuint index, const GLdouble *v);
typedef GLvoid (APIENTRY *PFNGLVERTEXATTRIB3SVARBPROC)(GLuint index, const GLshort *v);
typedef GLvoid (APIENTRY *PFNGLVERTEXATTRIB3FVARBPROC)(GLuint index, const GLfloat *v);
typedef GLvoid (APIENTRY *PFNGLVERTEXATTRIB3DVARBPROC)(GLuint index, const GLdouble *v);
typedef GLvoid (APIENTRY *PFNGLVERTEXATTRIB4BVARBPROC)(GLuint index, const GLbyte *v);
typedef GLvoid (APIENTRY *PFNGLVERTEXATTRIB4SVARBPROC)(GLuint index, const GLshort *v);
typedef GLvoid (APIENTRY *PFNGLVERTEXATTRIB4IVARBPROC)(GLuint index, const GLint *v);
typedef GLvoid (APIENTRY *PFNGLVERTEXATTRIB4UBVARBPROC)(GLuint index, const GLubyte *v);
typedef GLvoid (APIENTRY *PFNGLVERTEXATTRIB4USVARBPROC)(GLuint index, const GLushort *v);
typedef GLvoid (APIENTRY *PFNGLVERTEXATTRIB4UIVARBPROC)(GLuint index, const GLuint *v);
typedef GLvoid (APIENTRY *PFNGLVERTEXATTRIB4FVARBPROC)(GLuint index, const GLfloat *v);
typedef GLvoid (APIENTRY *PFNGLVERTEXATTRIB4DVARBPROC)(GLuint index, const GLdouble *v);
typedef GLvoid (APIENTRY *PFNGLVERTEXATTRIB4NBVARBPROC)(GLuint index, const GLbyte *v);
typedef GLvoid (APIENTRY *PFNGLVERTEXATTRIB4NSVARBPROC)(GLuint index, const GLshort *v);
typedef GLvoid (APIENTRY *PFNGLVERTEXATTRIB4NIVARBPROC)(GLuint index, const GLint *v);
typedef GLvoid (APIENTRY *PFNGLVERTEXATTRIB4NUBVARBPROC)(GLuint index, const GLubyte *v);
typedef GLvoid (APIENTRY *PFNGLVERTEXATTRIB4NUSVARBPROC)(GLuint index, const GLushort *v);
typedef GLvoid (APIENTRY *PFNGLVERTEXATTRIB4NUIVARBPROC)(GLuint index, const GLuint *v);
typedef GLvoid (APIENTRY *PFNGLVERTEXATTRIBPOINTERARBPROC)(GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, const GLvoid *pointer);
typedef GLvoid (APIENTRY *PFNGLENABLEVERTEXATTRIBARRAYARBPROC)(GLuint index);
typedef GLvoid (APIENTRY *PFNGLDISABLEVERTEXATTRIBARRAYARBPROC)(GLuint index);
typedef GLvoid (APIENTRY *PFNGLPROGRAMSTRINGARBPROC)(GLenum target, GLenum format, GLsizei len, const GLvoid *string);
typedef GLvoid (APIENTRY *PFNGLBINDPROGRAMARBPROC)(GLenum target, GLuint program);
typedef GLvoid (APIENTRY *PFNGLDELETEPROGRAMSARBPROC)(GLsizei n, const GLuint *programs);
typedef GLvoid (APIENTRY *PFNGLGENPROGRAMSARBPROC)(GLsizei n, GLuint *programs);
typedef GLvoid (APIENTRY *PFNGLPROGRAMENVPARAMETER4FARBPROC)(GLenum target, GLuint index, GLfloat x, GLfloat y, GLfloat z, GLfloat w);
typedef GLvoid (APIENTRY *PFNGLPROGRAMENVPARAMETER4DARBPROC)(GLenum target, GLuint index, GLdouble x, GLdouble y, GLdouble z, GLdouble w);
typedef GLvoid (APIENTRY *PFNGLPROGRAMENVPARAMETER4FVARBPROC)(GLenum target, GLuint index, const GLfloat *params);
typedef GLvoid (APIENTRY *PFNGLPROGRAMENVPARAMETER4DVARBPROC)(GLenum target, GLuint index, const GLdouble *params);
typedef GLvoid (APIENTRY *PFNGLPROGRAMLOCALPARAMETER4FARBPROC)(GLenum target, GLuint index, GLfloat x, GLfloat y, GLfloat z, GLfloat w);
typedef GLvoid (APIENTRY *PFNGLPROGRAMLOCALPARAMETER4DARBPROC)(GLenum target, GLuint index, GLdouble x, GLdouble y, GLdouble z, GLdouble w);
typedef GLvoid (APIENTRY *PFNGLPROGRAMLOCALPARAMETER4FVARBPROC)(GLenum target, GLuint index, const GLfloat *params);
typedef GLvoid (APIENTRY *PFNGLPROGRAMLOCALPARAMETER4DVARBPROC)(GLenum target, GLuint index, const GLdouble *params);
typedef GLvoid (APIENTRY *PFNGLGETPROGRAMENVPARAMETERFVARBPROC)(GLenum target, GLuint index, GLfloat *params);
typedef GLvoid (APIENTRY *PFNGLGETPROGRAMENVPARAMETERDVARBPROC)(GLenum target, GLuint index, GLdouble *params);
typedef GLvoid (APIENTRY *PFNGLGETPROGRAMLOCALPARAMETERFVARBPROC)(GLenum target, GLuint index, GLfloat *params);
typedef GLvoid (APIENTRY *PFNGLGETPROGRAMLOCALPARAMETERDVARBPROC)(GLenum target, GLuint index, GLdouble *params);
typedef GLvoid (APIENTRY *PFNGLGETPROGRAMIVARBPROC)(GLenum target, GLenum pname, GLint *params);
typedef GLvoid (APIENTRY *PFNGLGETPROGRAMSTRINGARBPROC)(GLenum target, GLenum pname, GLvoid *string);
typedef GLvoid (APIENTRY *PFNGLGETVERTEXATTRIBDVARBPROC)(GLuint index, GLenum pname, GLdouble *params);
typedef GLvoid (APIENTRY *PFNGLGETVERTEXATTRIBFVARBPROC)(GLuint index, GLenum pname, GLfloat *params);
typedef GLvoid (APIENTRY *PFNGLGETVERTEXATTRIBIVARBPROC)(GLuint index, GLenum pname, GLint *params);
typedef GLvoid (APIENTRY *PFNGLGETVERTEXATTRIBPOINTERVARBPROC)(GLuint index, GLenum pname, GLvoid **pointer);
typedef GLboolean (APIENTRY *PFNGLISPROGRAMARBPROC)(GLuint program);
#endif

#if defined(GL_ARB_vertex_program_PROTOTYPES) && defined(GL_ARB_fragment_program_PROTOTYPES)
extern PFNGLVERTEXATTRIB1SARBPROC glVertexAttrib1sARB;
extern PFNGLVERTEXATTRIB1FARBPROC glVertexAttrib1fARB;
extern PFNGLVERTEXATTRIB1DARBPROC glVertexAttrib1dARB;
extern PFNGLVERTEXATTRIB2SARBPROC glVertexAttrib2sARB;
extern PFNGLVERTEXATTRIB2FARBPROC glVertexAttrib2fARB;
extern PFNGLVERTEXATTRIB2DARBPROC glVertexAttrib2dARB;
extern PFNGLVERTEXATTRIB3SARBPROC glVertexAttrib3sARB;
extern PFNGLVERTEXATTRIB3FARBPROC glVertexAttrib3fARB;
extern PFNGLVERTEXATTRIB3DARBPROC glVertexAttrib3dARB;
extern PFNGLVERTEXATTRIB4SARBPROC glVertexAttrib4sARB;
extern PFNGLVERTEXATTRIB4FARBPROC glVertexAttrib4fARB;
extern PFNGLVERTEXATTRIB4DARBPROC glVertexAttrib4dARB;
extern PFNGLVERTEXATTRIB4NUBARBPROC glVertexAttrib4NubARB;
extern PFNGLVERTEXATTRIB1SVARBPROC glVertexAttrib1svARB;
extern PFNGLVERTEXATTRIB1FVARBPROC glVertexAttrib1fvARB;
extern PFNGLVERTEXATTRIB1DVARBPROC glVertexAttrib1dvARB;
extern PFNGLVERTEXATTRIB2SVARBPROC glVertexAttrib2svARB;
extern PFNGLVERTEXATTRIB2FVARBPROC glVertexAttrib2fvARB;
extern PFNGLVERTEXATTRIB2DVARBPROC glVertexAttrib2dvARB;
extern PFNGLVERTEXATTRIB3SVARBPROC glVertexAttrib3svARB;
extern PFNGLVERTEXATTRIB3FVARBPROC glVertexAttrib3fvARB;
extern PFNGLVERTEXATTRIB3DVARBPROC glVertexAttrib3dvARB;
extern PFNGLVERTEXATTRIB4BVARBPROC glVertexAttrib4bvARB;
extern PFNGLVERTEXATTRIB4SVARBPROC glVertexAttrib4svARB;
extern PFNGLVERTEXATTRIB4IVARBPROC glVertexAttrib4ivARB;
extern PFNGLVERTEXATTRIB4UBVARBPROC glVertexAttrib4ubvARB;
extern PFNGLVERTEXATTRIB4USVARBPROC glVertexAttrib4usvARB;
extern PFNGLVERTEXATTRIB4UIVARBPROC glVertexAttrib4uivARB;
extern PFNGLVERTEXATTRIB4FVARBPROC glVertexAttrib4fvARB;
extern PFNGLVERTEXATTRIB4DVARBPROC glVertexAttrib4dvARB;
extern PFNGLVERTEXATTRIB4NBVARBPROC glVertexAttrib4NbvARB;
extern PFNGLVERTEXATTRIB4NSVARBPROC glVertexAttrib4NsvARB;
extern PFNGLVERTEXATTRIB4NIVARBPROC glVertexAttrib4NivARB;
extern PFNGLVERTEXATTRIB4NUBVARBPROC glVertexAttrib4NubvARB;
extern PFNGLVERTEXATTRIB4NUSVARBPROC glVertexAttrib4NusvARB;
extern PFNGLVERTEXATTRIB4NUIVARBPROC glVertexAttrib4NuivARB;
extern PFNGLVERTEXATTRIBPOINTERARBPROC glVertexAttribPointerARB;
extern PFNGLENABLEVERTEXATTRIBARRAYARBPROC glEnableVertexAttribArrayARB;
extern PFNGLDISABLEVERTEXATTRIBARRAYARBPROC glDisableVertexAttribArrayARB;
extern PFNGLPROGRAMSTRINGARBPROC glProgramStringARB;
extern PFNGLBINDPROGRAMARBPROC glBindProgramARB;
extern PFNGLDELETEPROGRAMSARBPROC glDeleteProgramsARB;
extern PFNGLGENPROGRAMSARBPROC glGenProgramsARB;
extern PFNGLPROGRAMENVPARAMETER4DARBPROC glProgramEnvParameter4dARB;
extern PFNGLPROGRAMENVPARAMETER4DVARBPROC glProgramEnvParameter4dvARB;
extern PFNGLPROGRAMENVPARAMETER4FARBPROC glProgramEnvParameter4fARB;
extern PFNGLPROGRAMENVPARAMETER4FVARBPROC glProgramEnvParameter4fvARB;
extern PFNGLPROGRAMLOCALPARAMETER4DARBPROC glProgramLocalParameter4dARB;
extern PFNGLPROGRAMLOCALPARAMETER4DVARBPROC glProgramLocalParameter4dvARB;
extern PFNGLPROGRAMLOCALPARAMETER4FARBPROC glProgramLocalParameter4fARB;
extern PFNGLPROGRAMLOCALPARAMETER4FVARBPROC glProgramLocalParameter4fvARB;
extern PFNGLGETPROGRAMENVPARAMETERDVARBPROC glGetProgramEnvParameterdvARB;
extern PFNGLGETPROGRAMENVPARAMETERFVARBPROC glGetProgramEnvParameterfvARB;
extern PFNGLGETPROGRAMLOCALPARAMETERDVARBPROC glGetProgramLocalParameterdvARB;
extern PFNGLGETPROGRAMLOCALPARAMETERFVARBPROC glGetProgramLocalParameterfvARB;
extern PFNGLGETPROGRAMIVARBPROC glGetProgramivARB;
extern PFNGLGETPROGRAMSTRINGARBPROC glGetProgramStringARB;
extern PFNGLGETVERTEXATTRIBDVARBPROC glGetVertexAttribdvARB;
extern PFNGLGETVERTEXATTRIBFVARBPROC glGetVertexAttribfvARB;
extern PFNGLGETVERTEXATTRIBIVARBPROC glGetVertexAttribivARB;
extern PFNGLGETVERTEXATTRIBPOINTERVARBPROC glGetVertexAttribPointervARB;
extern PFNGLISPROGRAMARBPROC glIsProgramARB;
#endif

#ifndef GL_ARB_vertex_shader
#define GL_ARB_vertex_shader

#define GL_VERTEX_SHADER_ARB                        0x8B31

#define GL_MAX_VERTEX_UNIFORM_COMPONENTS_ARB        0x8B4A
#define GL_MAX_VARYING_FLOATS_ARB                   0x8B4B
#define GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS_ARB       0x8B4C
#define GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS_ARB     0x8B4D

#define GL_OBJECT_ACTIVE_ATTRIBUTES_ARB             0x8B89
#define GL_OBJECT_ACTIVE_ATTRIBUTE_MAX_LENGTH_ARB   0x8B8A

typedef GLvoid (APIENTRY *PFNGLBINDATTRIBLOCATIONARBPROC)(GLhandleARB programObj, GLuint index, const GLcharARB *name);
typedef GLvoid (APIENTRY *PFNGLGETACTIVEATTRIBARBPROC)(GLhandleARB programObj, GLuint index, GLsizei maxLength, GLsizei *length, GLint *size, GLenum *type, GLcharARB *name);
typedef GLint  (APIENTRY *PFNGLGETATTRIBLOCATIONARBPROC)(GLhandleARB programObj, const GLcharARB *name);
#endif

#ifdef GL_ARB_vertex_shader_PROTOTYPES
extern PFNGLBINDATTRIBLOCATIONARBPROC glBindAttribLocationARB;
extern PFNGLGETACTIVEATTRIBARBPROC    glGetActiveAttribARB;
extern PFNGLGETATTRIBLOCATIONARBPROC  glGetAttribLocationARB;
#endif


#ifndef GL_ARB_window_pos
#define GL_ARB_window_pos

typedef GLvoid (APIENTRY * PFNGLWINDOWPOS2DARBPROC) (double x, double y);
typedef GLvoid (APIENTRY * PFNGLWINDOWPOS2FARBPROC) (float x, float y);
typedef GLvoid (APIENTRY * PFNGLWINDOWPOS2IARBPROC) (int x, int y);
typedef GLvoid (APIENTRY * PFNGLWINDOWPOS2SARBPROC) (short x, short y);
typedef GLvoid (APIENTRY * PFNGLWINDOWPOS2IVARBPROC) (const int *p);
typedef GLvoid (APIENTRY * PFNGLWINDOWPOS2SVARBPROC) (const short *p);
typedef GLvoid (APIENTRY * PFNGLWINDOWPOS2FVARBPROC) (const float *p);
typedef GLvoid (APIENTRY * PFNGLWINDOWPOS2DVARBPROC) (const double *p);
typedef GLvoid (APIENTRY * PFNGLWINDOWPOS3IARBPROC) (int x, int y, int z);
typedef GLvoid (APIENTRY * PFNGLWINDOWPOS3SARBPROC) (short x, short y, short z);
typedef GLvoid (APIENTRY * PFNGLWINDOWPOS3FARBPROC) (float x, float y, float z);
typedef GLvoid (APIENTRY * PFNGLWINDOWPOS3DARBPROC) (double x, double y, double z);
typedef GLvoid (APIENTRY * PFNGLWINDOWPOS3IVARBPROC) (const int *p);
typedef GLvoid (APIENTRY * PFNGLWINDOWPOS3SVARBPROC) (const short *p);
typedef GLvoid (APIENTRY * PFNGLWINDOWPOS3FVARBPROC) (const float *p);
typedef GLvoid (APIENTRY * PFNGLWINDOWPOS3DVARBPROC) (const double *p);
#endif

#ifdef GL_ARB_window_pos_PROTOTYPES
extern PFNGLWINDOWPOS2DARBPROC  glWindowPos2dARB;
extern PFNGLWINDOWPOS2FARBPROC  glWindowPos2fARB;
extern PFNGLWINDOWPOS2IARBPROC  glWindowPos2iARB;
extern PFNGLWINDOWPOS2SARBPROC  glWindowPos2sARB;
extern PFNGLWINDOWPOS2IVARBPROC glWindowPos2ivARB;
extern PFNGLWINDOWPOS2SVARBPROC glWindowPos2svARB;
extern PFNGLWINDOWPOS2FVARBPROC glWindowPos2fvARB;
extern PFNGLWINDOWPOS2DVARBPROC glWindowPos2dvARB;
extern PFNGLWINDOWPOS3IARBPROC  glWindowPos3iARB;
extern PFNGLWINDOWPOS3SARBPROC  glWindowPos3sARB;
extern PFNGLWINDOWPOS3FARBPROC  glWindowPos3fARB;
extern PFNGLWINDOWPOS3DARBPROC  glWindowPos3dARB;
extern PFNGLWINDOWPOS3IVARBPROC glWindowPos3ivARB;
extern PFNGLWINDOWPOS3SVARBPROC glWindowPos3svARB;
extern PFNGLWINDOWPOS3FVARBPROC glWindowPos3fvARB;
extern PFNGLWINDOWPOS3DVARBPROC glWindowPos3dvARB;
#endif


#ifndef GL_ATI_fragment_shader
#define GL_ATI_fragment_shader

#define GL_FRAGMENT_SHADER_ATI  0x8920
#define GL_REG_0_ATI   0x8921
#define GL_REG_1_ATI   0x8922
#define GL_REG_2_ATI   0x8923
#define GL_REG_3_ATI   0x8924
#define GL_REG_4_ATI   0x8925
#define GL_REG_5_ATI   0x8926
#define GL_REG_6_ATI   0x8927
#define GL_REG_7_ATI   0x8928
#define GL_REG_8_ATI   0x8929
#define GL_REG_9_ATI   0x892A
#define GL_REG_10_ATI  0x892B
#define GL_REG_11_ATI  0x892C
#define GL_REG_12_ATI  0x892D
#define GL_REG_13_ATI  0x892E
#define GL_REG_14_ATI  0x892F
#define GL_REG_15_ATI  0x8930
#define GL_REG_16_ATI  0x8931
#define GL_REG_17_ATI  0x8932
#define GL_REG_18_ATI  0x8933
#define GL_REG_19_ATI  0x8934
#define GL_REG_20_ATI  0x8935
#define GL_REG_21_ATI  0x8936
#define GL_REG_22_ATI  0x8937
#define GL_REG_23_ATI  0x8938
#define GL_REG_24_ATI  0x8939
#define GL_REG_25_ATI  0x893A
#define GL_REG_26_ATI  0x893B
#define GL_REG_27_ATI  0x893C
#define GL_REG_28_ATI  0x893D
#define GL_REG_29_ATI  0x893E
#define GL_REG_30_ATI  0x893F
#define GL_REG_31_ATI  0x8940
#define GL_CON_0_ATI   0x8941
#define GL_CON_1_ATI   0x8942
#define GL_CON_2_ATI   0x8943
#define GL_CON_3_ATI   0x8944
#define GL_CON_4_ATI   0x8945
#define GL_CON_5_ATI   0x8946
#define GL_CON_6_ATI   0x8947
#define GL_CON_7_ATI   0x8948
#define GL_CON_8_ATI   0x8949
#define GL_CON_9_ATI   0x894A
#define GL_CON_10_ATI  0x894B
#define GL_CON_11_ATI  0x894C
#define GL_CON_12_ATI  0x894D
#define GL_CON_13_ATI  0x894E
#define GL_CON_14_ATI  0x894F
#define GL_CON_15_ATI  0x8950
#define GL_CON_16_ATI  0x8951
#define GL_CON_17_ATI  0x8952
#define GL_CON_18_ATI  0x8953
#define GL_CON_19_ATI  0x8954
#define GL_CON_20_ATI  0x8955
#define GL_CON_21_ATI  0x8956
#define GL_CON_22_ATI  0x8957
#define GL_CON_23_ATI  0x8958
#define GL_CON_24_ATI  0x8959
#define GL_CON_25_ATI  0x895A
#define GL_CON_26_ATI  0x895B
#define GL_CON_27_ATI  0x895C
#define GL_CON_28_ATI  0x895D
#define GL_CON_29_ATI  0x895E
#define GL_CON_30_ATI  0x895F
#define GL_CON_31_ATI  0x8960
#define GL_MOV_ATI      0x8961
#define GL_ADD_ATI      0x8963
#define GL_MUL_ATI      0x8964
#define GL_SUB_ATI      0x8965
#define GL_DOT3_ATI     0x8966
#define GL_DOT4_ATI     0x8967
#define GL_MAD_ATI      0x8968
#define GL_LERP_ATI     0x8969
#define GL_CND_ATI      0x896A
#define GL_CND0_ATI     0x896B
#define GL_DOT2_ADD_ATI 0x896C
#define GL_SECONDARY_INTERPOLATOR_ATI            0x896D
#define GL_NUM_FRAGMENT_REGISTERS_ATI            0x896E
#define GL_NUM_FRAGMENT_CONSTANTS_ATI            0x896F
#define GL_NUM_PASSES_ATI                        0x8970
#define GL_NUM_INSTRUCTIONS_PER_PASS_ATI         0x8971
#define GL_NUM_INSTRUCTIONS_TOTAL_ATI            0x8972
#define GL_NUM_INPUT_INTERPOLATOR_COMPONENTS_ATI 0x8973
#define GL_NUM_LOOPBACK_COMPONENTS_ATI           0x8974
#define GL_COLOR_ALPHA_PAIRING_ATI               0x8975
#define GL_SWIZZLE_STR_ATI      0x8976
#define GL_SWIZZLE_STQ_ATI      0x8977
#define GL_SWIZZLE_STR_DR_ATI   0x8978
#define GL_SWIZZLE_STQ_DQ_ATI   0x8979
#define GL_SWIZZLE_STRQ_ATI     0x897A
#define GL_SWIZZLE_STRQ_DQ_ATI  0x897B
#define GL_RED_BIT_ATI          0x00000001
#define GL_GREEN_BIT_ATI        0x00000002
#define GL_BLUE_BIT_ATI         0x00000004
#define GL_2X_BIT_ATI           0x00000001
#define GL_4X_BIT_ATI           0x00000002
#define GL_8X_BIT_ATI           0x00000004
#define GL_HALF_BIT_ATI         0x00000008
#define GL_QUARTER_BIT_ATI      0x00000010
#define GL_EIGHTH_BIT_ATI       0x00000020
#define GL_SATURATE_BIT_ATI     0x00000040
#define GL_COMP_BIT_ATI         0x00000002
#define GL_NEGATE_BIT_ATI       0x00000004
#define GL_BIAS_BIT_ATI         0x00000008

typedef GLuint (APIENTRY *PFNGLGENFRAGMENTSHADERSATIPROC)(GLuint range);
typedef GLvoid (APIENTRY *PFNGLBINDFRAGMENTSHADERATIPROC)(GLuint id);
typedef GLvoid (APIENTRY *PFNGLDELETEFRAGMENTSHADERATIPROC)(GLuint id);
typedef GLvoid (APIENTRY *PFNGLBEGINFRAGMENTSHADERATIPROC)(GLvoid);
typedef GLvoid (APIENTRY *PFNGLENDFRAGMENTSHADERATIPROC)(GLvoid);
typedef GLvoid (APIENTRY *PFNGLPASSTEXCOORDATIPROC)(GLuint dst, GLuint coord, GLenum swizzle);
typedef GLvoid (APIENTRY *PFNGLSAMPLEMAPATIPROC)(GLuint dst, GLuint interp, GLenum swizzle);
typedef GLvoid (APIENTRY *PFNGLCOLORFRAGMENTOP1ATIPROC)(GLenum op, GLuint dst, GLuint dstMask, GLuint dstMod, GLuint arg1, GLuint arg1Rep, GLuint arg1Mod);
typedef GLvoid (APIENTRY *PFNGLCOLORFRAGMENTOP2ATIPROC)(GLenum op, GLuint dst, GLuint dstMask, GLuint dstMod, GLuint arg1, GLuint arg1Rep, GLuint arg1Mod, GLuint arg2, GLuint arg2Rep, GLuint arg2Mod);
typedef GLvoid (APIENTRY *PFNGLCOLORFRAGMENTOP3ATIPROC)(GLenum op, GLuint dst, GLuint dstMask, GLuint dstMod, GLuint arg1, GLuint arg1Rep, GLuint arg1Mod, GLuint arg2, GLuint arg2Rep, GLuint arg2Mod, GLuint arg3, GLuint arg3Rep, GLuint arg3Mod);
typedef GLvoid (APIENTRY *PFNGLALPHAFRAGMENTOP1ATIPROC)(GLenum op, GLuint dst, GLuint dstMod, GLuint arg1, GLuint arg1Rep, GLuint arg1Mod);
typedef GLvoid (APIENTRY *PFNGLALPHAFRAGMENTOP2ATIPROC)(GLenum op, GLuint dst, GLuint dstMod, GLuint arg1, GLuint arg1Rep, GLuint arg1Mod, GLuint arg2, GLuint arg2Rep, GLuint arg2Mod);
typedef GLvoid (APIENTRY *PFNGLALPHAFRAGMENTOP3ATIPROC)(GLenum op, GLuint dst, GLuint dstMod, GLuint arg1, GLuint arg1Rep, GLuint arg1Mod, GLuint arg2, GLuint arg2Rep, GLuint arg2Mod, GLuint arg3, GLuint arg3Rep, GLuint arg3Mod);
typedef GLvoid (APIENTRY *PFNGLSETFRAGMENTSHADERCONSTANTATIPROC)(GLuint dst, const GLfloat *value);
#endif

#ifdef GL_ATI_fragment_shader_PROTOTYPES
extern PFNGLGENFRAGMENTSHADERSATIPROC   glGenFragmentShadersATI;
extern PFNGLBINDFRAGMENTSHADERATIPROC   glBindFragmentShaderATI;
extern PFNGLDELETEFRAGMENTSHADERATIPROC glDeleteFragmentShaderATI;
extern PFNGLBEGINFRAGMENTSHADERATIPROC  glBeginFragmentShaderATI;
extern PFNGLENDFRAGMENTSHADERATIPROC    glEndFragmentShaderATI;
extern PFNGLPASSTEXCOORDATIPROC         glPassTexCoordATI;
extern PFNGLSAMPLEMAPATIPROC            glSampleMapATI;

extern PFNGLCOLORFRAGMENTOP1ATIPROC glColorFragmentOp1ATI;
extern PFNGLCOLORFRAGMENTOP2ATIPROC glColorFragmentOp2ATI;
extern PFNGLCOLORFRAGMENTOP3ATIPROC glColorFragmentOp3ATI;

extern PFNGLALPHAFRAGMENTOP1ATIPROC glAlphaFragmentOp1ATI;
extern PFNGLALPHAFRAGMENTOP2ATIPROC glAlphaFragmentOp2ATI;
extern PFNGLALPHAFRAGMENTOP3ATIPROC glAlphaFragmentOp3ATI;

extern PFNGLSETFRAGMENTSHADERCONSTANTATIPROC glSetFragmentShaderConstantATI;
#endif


#ifndef GL_ATI_separate_stencil
#define GL_ATI_separate_stencil

#define GL_STENCIL_BACK_FUNC_ATI            0x8800
#define GL_STENCIL_BACK_FAIL_ATI            0x8801
#define GL_STENCIL_BACK_PASS_DEPTH_FAIL_ATI 0x8802
#define GL_STENCIL_BACK_PASS_DEPTH_PASS_ATI 0x8803

typedef GLvoid (APIENTRY *PFNGLSTENCILOPSEPARATEATIPROC)(GLenum face, GLenum sfail, GLenum dpfail, GLenum dppass);
typedef GLvoid (APIENTRY *PFNGLSTENCILFUNCSEPARATEATIPROC)(GLenum frontfunc, GLenum backfunc, GLint ref, GLuint mask);
#endif

#ifdef GL_ATI_separate_stencil_PROTOTYPES
extern PFNGLSTENCILOPSEPARATEATIPROC   glStencilOpSeparateATI;
extern PFNGLSTENCILFUNCSEPARATEATIPROC glStencilFuncSeparateATI;
#endif


#ifndef GL_ATI_texture_compression_3dc
#define GL_ATI_texture_compression_3dc

#define GL_COMPRESSED_LUMINANCE_ALPHA_3DC_ATI 0x8837

#endif


#ifndef GL_ATI_texture_float
#define GL_ATI_texture_float

#define GL_RGBA_FLOAT32_ATI            0x8814
#define GL_RGB_FLOAT32_ATI             0x8815
#define GL_ALPHA_FLOAT32_ATI           0x8816
#define GL_INTENSITY_FLOAT32_ATI       0x8817
#define GL_LUMINANCE_FLOAT32_ATI       0x8818
#define GL_LUMINANCE_ALPHA_FLOAT32_ATI 0x8819
#define GL_RGBA_FLOAT16_ATI            0x881A
#define GL_RGB_FLOAT16_ATI             0x881B
#define GL_ALPHA_FLOAT16_ATI           0x881C
#define GL_INTENSITY_FLOAT16_ATI       0x881D
#define GL_LUMINANCE_FLOAT16_ATI       0x881E
#define GL_LUMINANCE_ALPHA_FLOAT16_ATI 0x881F

#endif


#ifndef GL_ATI_texture_mirror_once
#define GL_ATI_texture_mirror_once

#define GL_MIRROR_CLAMP_ATI         0x8742
#define GL_MIRROR_CLAMP_TO_EDGE_ATI 0x8743

#endif


#ifndef GL_EXT_blend_color
#define GL_EXT_blend_color

#define GL_CONSTANT_COLOR_EXT           0x8001
#define GL_ONE_MINUS_CONSTANT_COLOR_EXT 0x8002
#define GL_CONSTANT_ALPHA_EXT           0x8003
#define GL_ONE_MINUS_CONSTANT_ALPHA_EXT 0x8004
#define GL_BLEND_COLOR_EXT              0x8005

typedef GLvoid (APIENTRY * PFNGLBLENDCOLOREXTPROC) (GLclampf red, GLclampf green, GLclampf blue, GLclampf alpha);
#endif

#ifdef GL_EXT_blend_color_PROTOTYPES
extern PFNGLBLENDCOLOREXTPROC glBlendColorEXT;
#endif


#ifndef GL_EXT_blend_func_separate
#define GL_EXT_blend_func_separate

#define GL_BLEND_DST_RGB_EXT    0x80C8
#define GL_BLEND_SRC_RGB_EXT    0x80C9
#define GL_BLEND_DST_ALPHA_EXT  0x80CA
#define GL_BLEND_SRC_ALPHA_EXT  0x80CB

typedef GLvoid (APIENTRY * PFNGLBLENDFUNCSEPARATEEXTPROC) (GLenum sfactorRGB, GLenum dfactorRGB, GLenum sfactorAlpha, GLenum dfactorAlpha);
#endif

#ifdef GL_EXT_blend_func_separate_PROTOTYPES
extern PFNGLBLENDFUNCSEPARATEEXTPROC glBlendFuncSeparateEXT;
#endif


#ifndef GL_EXT_blend_minmax
#define GL_EXT_blend_minmax

#define GL_FUNC_ADD_EXT 0x8006
#define GL_MIN_EXT      0x8007
#define GL_MAX_EXT      0x8008
#define GL_BLEND_EQUATION_EXT 0x8009

typedef GLvoid (APIENTRY * PFNGLBLENDEQUATIONEXTPROC) (GLenum mode);
#endif

#ifdef GL_EXT_blend_minmax_PROTOTYPES
extern PFNGLBLENDEQUATIONEXTPROC glBlendEquationEXT;
#endif


#ifndef GL_EXT_blend_subtract
#define GL_EXT_blend_subtract

#define GL_FUNC_SUBTRACT_EXT         0x800A
#define GL_FUNC_REVERSE_SUBTRACT_EXT 0x800B

#endif


#ifndef GL_EXT_draw_range_elements
#define GL_EXT_draw_range_elements

#define GL_MAX_ELEMENTS_VERTICES_EXT 0x80E8
#define GL_MAX_ELEMENTS_INDICES_EXT  0x80E9

typedef GLvoid (APIENTRY * PFNGLDRAWRANGEELEMENTSEXTPROC) (GLenum  mode, GLuint start, GLuint end, GLsizei count, GLenum type, const GLvoid *indices);
#endif

#ifdef GL_EXT_draw_range_elements_PROTOTYPES
extern PFNGLDRAWRANGEELEMENTSEXTPROC glDrawRangeElementsEXT;
#endif


#ifndef GL_EXT_fog_coord
#define GL_EXT_fog_coord

#define GL_FOG_COORDINATE_SOURCE_EXT         0x8450
#define GL_FOG_COORDINATE_EXT                0x8451
#define GL_FRAGMENT_DEPTH_EXT                0x8452
#define GL_CURRENT_FOG_COORDINATE_EXT        0x8453
#define GL_FOG_COORDINATE_ARRAY_TYPE_EXT     0x8454
#define GL_FOG_COORDINATE_ARRAY_STRIDE_EXT   0x8455
#define GL_FOG_COORDINATE_ARRAY_POINTER_EXT  0x8456
#define GL_FOG_COORDINATE_ARRAY_EXT          0x8457

typedef GLvoid (APIENTRY * PFNGLFOGCOORDFEXTPROC) (GLfloat f);
typedef GLvoid (APIENTRY * PFNGLFOGCOORDDEXTPROC) (GLdouble f);
typedef GLvoid (APIENTRY * PFNGLFOGCOORDFVEXTPROC) (const GLfloat *v);
typedef GLvoid (APIENTRY * PFNGLFOGCOORDDVEXTPROC) (const GLdouble *v);
typedef GLvoid (APIENTRY * PFNGLFOGCOORDPOINTEREXTPROC) (GLenum type, GLsizei stride, GLvoid *pointer);
#endif

#ifdef GL_EXT_fog_coord_PROTOTYPES
extern PFNGLFOGCOORDFEXTPROC  glFogCoordfEXT;
extern PFNGLFOGCOORDDEXTPROC  glFogCoorddEXT;
extern PFNGLFOGCOORDFVEXTPROC glFogCoordfvEXT;
extern PFNGLFOGCOORDDVEXTPROC glFogCoorddvEXT;
extern PFNGLFOGCOORDPOINTEREXTPROC glFogCoordPointerEXT;
#endif


#ifndef GL_EXT_framebuffer_object
#define GL_EXT_framebuffer_object

#define GL_FRAMEBUFFER_EXT     0x8D40
#define GL_RENDERBUFFER_EXT    0x8D41
#define GL_STENCIL_INDEX_EXT   0x8D45
#define GL_STENCIL_INDEX1_EXT  0x8D46
#define GL_STENCIL_INDEX4_EXT  0x8D47
#define GL_STENCIL_INDEX8_EXT  0x8D48
#define GL_STENCIL_INDEX16_EXT 0x8D49

#define GL_RENDERBUFFER_WIDTH_EXT           0x8D42
#define GL_RENDERBUFFER_HEIGHT_EXT          0x8D43
#define GL_RENDERBUFFER_INTERNAL_FORMAT_EXT 0x8D44

#define GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE_EXT           0x8CD0
#define GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME_EXT           0x8CD1
#define GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_LEVEL_EXT         0x8CD2
#define GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_CUBE_MAP_FACE_EXT 0x8CD3
#define GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_3D_ZOFFSET_EXT    0x8CD4

#define GL_COLOR_ATTACHMENT0_EXT  0x8CE0
#define GL_COLOR_ATTACHMENT1_EXT  0x8CE1
#define GL_COLOR_ATTACHMENT2_EXT  0x8CE2
#define GL_COLOR_ATTACHMENT3_EXT  0x8CE3
#define GL_COLOR_ATTACHMENT4_EXT  0x8CE4
#define GL_COLOR_ATTACHMENT5_EXT  0x8CE5
#define GL_COLOR_ATTACHMENT6_EXT  0x8CE6
#define GL_COLOR_ATTACHMENT7_EXT  0x8CE7
#define GL_COLOR_ATTACHMENT8_EXT  0x8CE8
#define GL_COLOR_ATTACHMENT9_EXT  0x8CE9
#define GL_COLOR_ATTACHMENT10_EXT 0x8CEA
#define GL_COLOR_ATTACHMENT11_EXT 0x8CEB
#define GL_COLOR_ATTACHMENT12_EXT 0x8CEC
#define GL_COLOR_ATTACHMENT13_EXT 0x8CED
#define GL_COLOR_ATTACHMENT14_EXT 0x8CEE
#define GL_COLOR_ATTACHMENT15_EXT 0x8CEF
#define GL_DEPTH_ATTACHMENT_EXT   0x8D00
#define GL_STENCIL_ATTACHMENT_EXT 0x8D20

#define GL_FRAMEBUFFER_COMPLETE_EXT                        0x8CD5
#define GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT_EXT           0x8CD6
#define GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT_EXT   0x8CD7
#define GL_FRAMEBUFFER_INCOMPLETE_DUPLICATE_ATTACHMENT_EXT 0x8CD8
#define GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS_EXT           0x8CD9
#define GL_FRAMEBUFFER_INCOMPLETE_FORMATS_EXT              0x8CDA
#define GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER_EXT          0x8CDB
#define GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER_EXT          0x8CDC

#define GL_FRAMEBUFFER_UNSUPPORTED_EXT  0x8CDD
#define GL_FRAMEBUFFER_STATUS_ERROR_EXT 0x8CDE
#define GL_FRAMEBUFFER_BINDING_EXT      0x8CA6
#define GL_RENDERBUFFER_BINDING_EXT     0x8CA7
#define GL_MAX_COLOR_ATTACHMENTS_EXT    0x8CDF
#define GL_MAX_RENDERBUFFER_SIZE_EXT    0x84E8

#define GL_INVALID_FRAMEBUFFER_OPERATION_EXT 0x0506

typedef GLboolean (APIENTRY * PFNGLISRENDERBUFFEREXTPROC)(GLuint renderbuffer);
typedef GLvoid    (APIENTRY * PFNGLBINDRENDERBUFFEREXTPROC)(GLenum target, GLuint renderbuffer);
typedef GLvoid    (APIENTRY * PFNGLDELETERENDERBUFFERSEXTPROC)(GLsizei n, const GLuint *renderbuffers);
typedef GLvoid    (APIENTRY * PFNGLGENRENDERBUFFERSEXTPROC)(GLsizei n, GLuint *renderbuffers);
typedef GLvoid    (APIENTRY * PFNGLRENDERBUFFERSTORAGEEXTPROC)(GLenum target, GLenum internalformat, GLsizei width, GLsizei height);
typedef GLvoid    (APIENTRY * PFNGLGETRENDERBUFFERPARAMETERIVEXTPROC)(GLenum target, GLenum pname, GLint *params);
typedef GLboolean (APIENTRY * PFNGLISFRAMEBUFFEREXTPROC)(GLuint framebuffer);
typedef GLvoid    (APIENTRY * PFNGLBINDFRAMEBUFFEREXTPROC)(GLenum target, GLuint framebuffer);
typedef GLvoid    (APIENTRY * PFNGLDELETEFRAMEBUFFERSEXTPROC)(GLsizei n, const GLuint *framebuffers);
typedef GLvoid    (APIENTRY * PFNGLGENFRAMEBUFFERSEXTPROC)(GLsizei n, GLuint *framebuffers);
typedef GLenum    (APIENTRY * PFNGLCHECKFRAMEBUFFERSTATUSEXTPROC)(GLenum target);
typedef GLvoid    (APIENTRY * PFNGLFRAMEBUFFERTEXTURE1DEXTPROC)(GLenum target, GLenum attachment, GLenum textarget, GLuint texture, GLint level);
typedef GLvoid    (APIENTRY * PFNGLFRAMEBUFFERTEXTURE2DEXTPROC)(GLenum target, GLenum attachment, GLenum textarget, GLuint texture, GLint level);
typedef GLvoid    (APIENTRY * PFNGLFRAMEBUFFERTEXTURE3DEXTPROC)(GLenum target, GLenum attachment, GLenum textarget, GLuint texture, GLint level, GLint zoffset);
typedef GLvoid    (APIENTRY * PFNGLFRAMEBUFFERRENDERBUFFEREXTPROC)(GLenum target, GLenum attachment, GLenum renderbuffertarget, GLuint renderbuffer);
typedef GLvoid    (APIENTRY * PFNGLGETFRAMEBUFFERATTACHMENTPARAMETERIVEXTPROC)(GLenum target, GLenum attachment, GLenum pname, GLint *params);
typedef GLvoid    (APIENTRY * PFNGLGENERATEMIPMAPEXTPROC)(GLenum target);
#endif

#ifdef GL_EXT_framebuffer_object_PROTOTYPES
extern PFNGLISRENDERBUFFEREXTPROC             glIsRenderbufferEXT;
extern PFNGLBINDRENDERBUFFEREXTPROC           glBindRenderbufferEXT;
extern PFNGLDELETERENDERBUFFERSEXTPROC        glDeleteRenderbuffersEXT;
extern PFNGLGENRENDERBUFFERSEXTPROC           glGenRenderbuffersEXT;
extern PFNGLRENDERBUFFERSTORAGEEXTPROC        glRenderbufferStorageEXT;
extern PFNGLGETRENDERBUFFERPARAMETERIVEXTPROC glGetRenderbufferParameterivEXT;
extern PFNGLISFRAMEBUFFEREXTPROC              glIsFramebufferEXT;
extern PFNGLBINDFRAMEBUFFEREXTPROC            glBindFramebufferEXT;
extern PFNGLDELETEFRAMEBUFFERSEXTPROC         glDeleteFramebuffersEXT;
extern PFNGLGENFRAMEBUFFERSEXTPROC            glGenFramebuffersEXT;
extern PFNGLCHECKFRAMEBUFFERSTATUSEXTPROC     glCheckFramebufferStatusEXT;
extern PFNGLFRAMEBUFFERTEXTURE1DEXTPROC       glFramebufferTexture1DEXT;
extern PFNGLFRAMEBUFFERTEXTURE2DEXTPROC       glFramebufferTexture2DEXT;
extern PFNGLFRAMEBUFFERTEXTURE3DEXTPROC       glFramebufferTexture3DEXT;
extern PFNGLFRAMEBUFFERRENDERBUFFEREXTPROC    glFramebufferRenderbufferEXT;
extern PFNGLGETFRAMEBUFFERATTACHMENTPARAMETERIVEXTPROC glGetFramebufferAttachmentParameterivEXT;
extern PFNGLGENERATEMIPMAPEXTPROC             glGenerateMipmapEXT;
#endif


#ifndef GL_EXT_multi_draw_arrays
#define GL_EXT_multi_draw_arrays

typedef GLvoid (APIENTRY * PFNGLMULTIDRAWARRAYSEXTPROC) (GLenum mode, GLint *first, GLsizei *count, GLsizei primcount);
typedef GLvoid (APIENTRY * PFNGLMULTIDRAWELEMENTSEXTPROC) (GLenum mode, GLsizei *count, GLenum type, const GLvoid **indices, GLsizei primcount);
#endif

#ifdef GL_EXT_multi_draw_arrays_PROTOTYPES
extern PFNGLMULTIDRAWARRAYSEXTPROC   glMultiDrawArraysEXT;
extern PFNGLMULTIDRAWELEMENTSEXTPROC glMultiDrawElementsEXT;
#endif


#ifndef GL_EXT_packed_pixels
#define GL_EXT_packed_pixels

#define GL_UNSIGNED_BYTE_3_3_2_EXT     0x8032
#define GL_UNSIGNED_SHORT_4_4_4_4_EXT  0x8033
#define GL_UNSIGNED_SHORT_5_5_5_1_EXT  0x8034
#define GL_UNSIGNED_INT_8_8_8_8_EXT    0x8035
#define GL_UNSIGNED_INT_10_10_10_2_EXT 0x8036

#endif


#ifndef GL_EXT_packed_depth_stencil
#define GL_EXT_packed_depth_stencil

#define GL_DEPTH_STENCIL_EXT        0x84F9
#define GL_UNSIGNED_INT_24_8_EXT    0x84FA
#define GL_DEPTH24_STENCIL8_EXT     0x88F0
#define GL_TEXTURE_STENCIL_SIZE_EXT 0x88F1

#endif


#ifndef GL_EXT_secondary_color
#define GL_EXT_secondary_color

#define GL_COLOR_SUM_EXT                     0x8458
#define GL_CURRENT_SECONDARY_COLOR_EXT       0x8459
#define GL_SECONDARY_COLOR_ARRAY_SIZE_EXT    0x845A
#define GL_SECONDARY_COLOR_ARRAY_TYPE_EXT    0x845B
#define GL_SECONDARY_COLOR_ARRAY_STRIDE_EXT  0x845C
#define GL_SECONDARY_COLOR_ARRAY_POINTER_EXT 0x845D
#define GL_SECONDARY_COLOR_ARRAY_EXT         0x845E

typedef GLvoid (APIENTRY * PFNGLSECONDARYCOLOR3FEXTPROC) (GLfloat r, GLfloat g, GLfloat b);
typedef GLvoid (APIENTRY * PFNGLSECONDARYCOLOR3DEXTPROC) (GLdouble r, GLdouble g, GLdouble b);
typedef GLvoid (APIENTRY * PFNGLSECONDARYCOLOR3BEXTPROC) (GLbyte r, GLbyte g, GLbyte b);
typedef GLvoid (APIENTRY * PFNGLSECONDARYCOLOR3SEXTPROC) (GLshort r, GLshort g, GLshort b);
typedef GLvoid (APIENTRY * PFNGLSECONDARYCOLOR3IEXTPROC) (GLint r, GLint g, GLint b);
typedef GLvoid (APIENTRY * PFNGLSECONDARYCOLOR3UBEXTPROC)(GLubyte r, GLubyte g, GLubyte b);
typedef GLvoid (APIENTRY * PFNGLSECONDARYCOLOR3USEXTPROC)(GLushort r, GLushort g, GLushort b);
typedef GLvoid (APIENTRY * PFNGLSECONDARYCOLOR3UIEXTPROC)(GLuint r, GLuint g, GLuint b);

typedef GLvoid (APIENTRY * PFNGLSECONDARYCOLOR3FVEXTPROC) (const GLfloat *v);
typedef GLvoid (APIENTRY * PFNGLSECONDARYCOLOR3DVEXTPROC) (const GLdouble *v);
typedef GLvoid (APIENTRY * PFNGLSECONDARYCOLOR3BVEXTPROC) (const GLbyte *v);
typedef GLvoid (APIENTRY * PFNGLSECONDARYCOLOR3SVEXTPROC) (const GLshort *v);
typedef GLvoid (APIENTRY * PFNGLSECONDARYCOLOR3IVEXTPROC) (const GLint *v);
typedef GLvoid (APIENTRY * PFNGLSECONDARYCOLOR3UBVEXTPROC)(const GLubyte *v);
typedef GLvoid (APIENTRY * PFNGLSECONDARYCOLOR3USVEXTPROC)(const GLushort *v);
typedef GLvoid (APIENTRY * PFNGLSECONDARYCOLOR3UIVEXTPROC)(const GLuint *v);

typedef GLvoid (APIENTRY * PFNGLSECONDARYCOLORPOINTEREXTPROC)(GLint size, GLenum type, GLsizei stride, GLvoid *pointer);
#endif

#ifdef GL_EXT_secondary_color_PROTOTYPES
extern PFNGLSECONDARYCOLOR3FEXTPROC glSecondaryColor3fEXT;
extern PFNGLSECONDARYCOLOR3DEXTPROC glSecondaryColor3dEXT;
extern PFNGLSECONDARYCOLOR3BEXTPROC glSecondaryColor3bEXT;
extern PFNGLSECONDARYCOLOR3SEXTPROC glSecondaryColor3sEXT;
extern PFNGLSECONDARYCOLOR3IEXTPROC glSecondaryColor3iEXT;
extern PFNGLSECONDARYCOLOR3UBEXTPROC glSecondaryColor3ubEXT;
extern PFNGLSECONDARYCOLOR3USEXTPROC glSecondaryColor3usEXT;
extern PFNGLSECONDARYCOLOR3UIEXTPROC glSecondaryColor3uiEXT;

extern PFNGLSECONDARYCOLOR3FVEXTPROC glSecondaryColor3fvEXT;
extern PFNGLSECONDARYCOLOR3DVEXTPROC glSecondaryColor3dvEXT;
extern PFNGLSECONDARYCOLOR3BVEXTPROC glSecondaryColor3bvEXT;
extern PFNGLSECONDARYCOLOR3SVEXTPROC glSecondaryColor3svEXT;
extern PFNGLSECONDARYCOLOR3IVEXTPROC glSecondaryColor3ivEXT;
extern PFNGLSECONDARYCOLOR3UBVEXTPROC glSecondaryColor3ubvEXT;
extern PFNGLSECONDARYCOLOR3USVEXTPROC glSecondaryColor3usvEXT;
extern PFNGLSECONDARYCOLOR3UIVEXTPROC glSecondaryColor3uivEXT;

extern PFNGLSECONDARYCOLORPOINTEREXTPROC glSecondaryColorPointerEXT;
#endif


#ifndef GL_EXT_stencil_wrap
#define GL_EXT_stencil_wrap

#define GL_INCR_WRAP_EXT 0x8507
#define GL_DECR_WRAP_EXT 0x8508

#endif


#ifndef GL_EXT_texture3D
#define GL_EXT_texture3D

#define GL_TEXTURE_BINDING_3D_EXT  0x806A
#define GL_PACK_SKIP_IMAGES_EXT    0x806B
#define GL_PACK_IMAGE_HEIGHT_EXT   0x806C
#define GL_UNPACK_SKIP_IMAGES_EXT  0x806D
#define GL_UNPACK_IMAGE_HEIGHT_EXT 0x806E
#define GL_TEXTURE_3D_EXT          0x806F
#define GL_PROXY_TEXTURE_3D_EXT    0x8070
#define GL_TEXTURE_DEPTH_EXT       0x8071
#define GL_TEXTURE_WRAP_R_EXT      0x8072
#define GL_MAX_3D_TEXTURE_SIZE_EXT 0x8073

typedef GLvoid (APIENTRY * PFNGLTEXIMAGE3DEXTPROC)(GLenum target, GLint level, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth, GLint border, GLenum format, GLenum type, const GLvoid *pixels);
typedef GLvoid (APIENTRY * PFNGLTEXSUBIMAGE3DPROC)(GLenum target, GLint lod, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei w, GLsizei h, GLsizei d, GLenum format, GLenum type, const GLvoid *buf);
typedef GLvoid (APIENTRY * PFNGLCOPYTEXSUBIMAGE3DPROC)(GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLint x, GLint y, GLsizei width, GLsizei height);
#endif

#ifdef GL_EXT_texture3D_PROTOTYPES
extern PFNGLTEXIMAGE3DEXTPROC glTexImage3DEXT;
#endif


#ifndef GL_EXT_texture_compression_s3tc
#define GL_EXT_texture_compression_s3tc

#define GL_COMPRESSED_RGB_S3TC_DXT1_EXT  0x83F0
#define GL_COMPRESSED_RGBA_S3TC_DXT1_EXT 0x83F1
#define GL_COMPRESSED_RGBA_S3TC_DXT3_EXT 0x83F2
#define GL_COMPRESSED_RGBA_S3TC_DXT5_EXT 0x83F3

#endif


#ifndef GL_EXT_texture_edge_clamp
#define GL_EXT_texture_edge_clamp

#define GL_CLAMP_TO_EDGE_EXT 0x812F

#endif


#ifndef GL_EXT_texture_filter_anisotropic
#define GL_EXT_texture_filter_anisotropic

#define GL_TEXTURE_MAX_ANISOTROPY_EXT     0x84FE
#define GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT 0x84FF

#endif


#ifndef GL_EXT_texture_lod_bias
#define GL_EXT_texture_lod_bias

#define GL_TEXTURE_FILTER_CONTROL_EXT 0x8500
#define GL_TEXTURE_LOD_BIAS_EXT       0x8501
#define GL_MAX_TEXTURE_LOD_BIAS_EXT   0x84FD

#endif


#ifndef GL_HP_occlusion_test
#define GL_HP_occlusion_test

#define GL_OCCLUSION_TEST_HP        0x8165
#define GL_OCCLUSION_TEST_RESULT_HP 0x8166

#endif


#ifndef GL_SGIS_generate_mipmap
#define GL_SGIS_generate_mipmap

#define GL_GENERATE_MIPMAP_SGIS      0x8191
#define GL_GENERATE_MIPMAP           0x8191
#define GL_GENERATE_MIPMAP_HINT_SGIS 0x8192
#define GL_GENERATE_MIPMAP_HINT      0x8192

#endif


#if defined(_WIN32)


#ifndef WGL_ARB_extensions_string
#define WGL_ARB_extensions_string

typedef const char * (WINAPI * PFNWGLGETEXTENSIONSSTRINGARBPROC) (HDC hdc);

extern PFNWGLGETEXTENSIONSSTRINGARBPROC wglGetExtensionsStringARB;

#endif


#ifndef WGL_ARB_multisample
#define WGL_ARB_multisample

#define WGL_SAMPLE_BUFFERS_ARB 0x2041
#define WGL_SAMPLES_ARB        0x2042

#endif


#ifndef WGL_ARB_make_current_read
#define WGL_ARB_make_current_read

#define GL_ERROR_INVALID_PIXEL_TYPE_ARB   0x2043
#define GL_ERROR_INCOMPATIBLE_DEVICE_CONTEXTS_ARB 0x2054

typedef BOOL (WINAPI * PFNWGLMAKECONTEXTCURRENTARBPROC) (HDC hDrawDC, HDC hReadDC, HGLRC hglrc);
typedef HDC (WINAPI * PFNWGLGETCURRENTREADDCARBPROC) (void);

extern PFNWGLMAKECONTEXTCURRENTARBPROC wglMakeContextCurrentARB;
extern PFNWGLGETCURRENTREADDCARBPROC   wglGetCurrentReadDCARB;

#endif


#ifndef WGL_ARB_pbuffer
#define WGL_ARB_pbuffer

#define WGL_DRAW_TO_PBUFFER_ARB    0x202D
#define WGL_MAX_PBUFFER_PIXELS_ARB 0x202E
#define WGL_MAX_PBUFFER_WIDTH_ARB  0x202F
#define WGL_MAX_PBUFFER_HEIGHT_ARB 0x2030
#define WGL_PBUFFER_LARGEST_ARB    0x2033
#define WGL_PBUFFER_WIDTH_ARB      0x2034
#define WGL_PBUFFER_HEIGHT_ARB     0x2035
#define WGL_PBUFFER_LOST_ARB       0x2036

DECLARE_HANDLE(HPBUFFERARB);

typedef HPBUFFERARB (WINAPI * PFNWGLCREATEPBUFFERARBPROC) (HDC hDC, int iPixelFormat, int iWidth, int iHeight, const int *piAttribList);
typedef HDC (WINAPI * PFNWGLGETPBUFFERDCARBPROC) (HPBUFFERARB hPbuffer);
typedef int (WINAPI * PFNWGLRELEASEPBUFFERDCARBPROC) (HPBUFFERARB hPbuffer, HDC hDC);
typedef BOOL (WINAPI * PFNWGLDESTROYPBUFFERARBPROC) (HPBUFFERARB hPbuffer);
typedef BOOL (WINAPI * PFNWGLQUERYPBUFFERARBPROC) (HPBUFFERARB hPbuffer, int iAttribute, int *piValue);

extern PFNWGLCREATEPBUFFERARBPROC    wglCreatePbufferARB;
extern PFNWGLGETPBUFFERDCARBPROC     wglGetPbufferDCARB;
extern PFNWGLRELEASEPBUFFERDCARBPROC wglReleasePbufferDCARB;
extern PFNWGLDESTROYPBUFFERARBPROC   wglDestroyPbufferARB;
extern PFNWGLQUERYPBUFFERARBPROC     wglQueryPbufferARB;

#endif


#ifndef WGL_ARB_pixel_format
#define WGL_ARB_pixel_format

#define WGL_NUMBER_PIXEL_FORMATS_ARB    0x2000
#define WGL_DRAW_TO_WINDOW_ARB          0x2001
#define WGL_DRAW_TO_BITMAP_ARB          0x2002
#define WGL_ACCELERATION_ARB            0x2003
#define WGL_NEED_PALETTE_ARB            0x2004
#define WGL_NEED_SYSTEM_PALETTE_ARB     0x2005
#define WGL_SWAP_LAYER_BUFFERS_ARB      0x2006
#define WGL_SWAP_METHOD_ARB             0x2007
#define WGL_NUMBER_OVERLAYS_ARB         0x2008
#define WGL_NUMBER_UNDERLAYS_ARB        0x2009
#define WGL_TRANSPARENT_ARB             0x200A
#define WGL_TRANSPARENT_RED_VALUE_ARB   0x2037
#define WGL_TRANSPARENT_GREEN_VALUE_ARB 0x2038
#define WGL_TRANSPARENT_BLUE_VALUE_ARB  0x2039
#define WGL_TRANSPARENT_ALPHA_VALUE_ARB 0x203A
#define WGL_TRANSPARENT_INDEX_VALUE_ARB 0x203B
#define WGL_SHARE_DEPTH_ARB             0x200C
#define WGL_SHARE_STENCIL_ARB           0x200D
#define WGL_SHARE_ACCUM_ARB             0x200E
#define WGL_SUPPORT_GDI_ARB             0x200F
#define WGL_SUPPORT_OPENGL_ARB          0x2010
#define WGL_DOUBLE_BUFFER_ARB           0x2011
#define WGL_STEREO_ARB                  0x2012
#define WGL_PIXEL_TYPE_ARB              0x2013
#define WGL_COLOR_BITS_ARB              0x2014
#define WGL_RED_BITS_ARB                0x2015
#define WGL_RED_SHIFT_ARB               0x2016
#define WGL_GREEN_BITS_ARB              0x2017
#define WGL_GREEN_SHIFT_ARB             0x2018
#define WGL_BLUE_BITS_ARB               0x2019
#define WGL_BLUE_SHIFT_ARB              0x201A
#define WGL_ALPHA_BITS_ARB              0x201B
#define WGL_ALPHA_SHIFT_ARB             0x201C
#define WGL_ACCUM_BITS_ARB              0x201D
#define WGL_ACCUM_RED_BITS_ARB          0x201E
#define WGL_ACCUM_GREEN_BITS_ARB        0x201F
#define WGL_ACCUM_BLUE_BITS_ARB         0x2020
#define WGL_ACCUM_ALPHA_BITS_ARB        0x2021
#define WGL_DEPTH_BITS_ARB              0x2022
#define WGL_STENCIL_BITS_ARB            0x2023
#define WGL_AUX_BUFFERS_ARB             0x2024
#define WGL_NO_ACCELERATION_ARB         0x2025
#define WGL_GENERIC_ACCELERATION_ARB    0x2026
#define WGL_FULL_ACCELERATION_ARB       0x2027
#define WGL_SWAP_EXCHANGE_ARB           0x2028
#define WGL_SWAP_COPY_ARB               0x2029
#define WGL_SWAP_UNDEFINED_ARB          0x202A
#define WGL_TYPE_RGBA_ARB               0x202B
#define WGL_TYPE_COLORINDEX_ARB         0x202C

typedef BOOL (WINAPI * PFNWGLGETPIXELFORMATATTRIBIVARBPROC) (HDC hdc, int iPixelFormat, int iLayerPlane, UINT nAttributes, const int *piAttributes, int *piValues);
typedef BOOL (WINAPI * PFNWGLGETPIXELFORMATATTRIBFVARBPROC) (HDC hdc, int iPixelFormat, int iLayerPlane, UINT nAttributes, const int *piAttributes, FLOAT *pfValues);
typedef BOOL (WINAPI * PFNWGLCHOOSEPIXELFORMATARBPROC) (HDC hdc, const int *piAttribIList, const FLOAT *pfAttribFList, UINT nMaxFormats, int *piFormats, UINT *nNumFormats);

extern PFNWGLGETPIXELFORMATATTRIBIVARBPROC wglGetPixelFormatAttribivARB;
extern PFNWGLGETPIXELFORMATATTRIBFVARBPROC wglGetPixelFormatAttribfvARB;
extern PFNWGLCHOOSEPIXELFORMATARBPROC      wglChoosePixelFormatARB;

#endif


#ifndef WGL_ARB_render_texture
#define WGL_ARB_render_texture

#define WGL_BIND_TO_TEXTURE_RGB_ARB    0x2070
#define WGL_BIND_TO_TEXTURE_RGBA_ARB   0x2071
#define WGL_TEXTURE_FORMAT_ARB         0x2072
#define WGL_TEXTURE_TARGET_ARB         0x2073
#define WGL_MIPMAP_TEXTURE_ARB         0x2074
#define WGL_TEXTURE_RGB_ARB            0x2075
#define WGL_TEXTURE_RGBA_ARB           0x2076
#define WGL_NO_TEXTURE_ARB             0x2077
#define WGL_TEXTURE_CUBE_MAP_ARB       0x2078
#define WGL_TEXTURE_1D_ARB             0x2079
#define WGL_TEXTURE_2D_ARB             0x207A
#define WGL_MIPMAP_LEVEL_ARB           0x207B
#define WGL_CUBE_MAP_FACE_ARB          0x207C
#define WGL_TEXTURE_CUBE_MAP_POSITIVE_X_ARB 0x207D
#define WGL_TEXTURE_CUBE_MAP_NEGATIVE_X_ARB 0x207E
#define WGL_TEXTURE_CUBE_MAP_POSITIVE_Y_ARB 0x207F
#define WGL_TEXTURE_CUBE_MAP_NEGATIVE_Y_ARB 0x2080
#define WGL_TEXTURE_CUBE_MAP_POSITIVE_Z_ARB 0x2081
#define WGL_TEXTURE_CUBE_MAP_NEGATIVE_Z_ARB 0x2082
#define WGL_FRONT_LEFT_ARB             0x2083
#define WGL_FRONT_RIGHT_ARB            0x2084
#define WGL_BACK_LEFT_ARB              0x2085
#define WGL_BACK_RIGHT_ARB             0x2086
#define WGL_AUX0_ARB                   0x2087
#define WGL_AUX1_ARB                   0x2088
#define WGL_AUX2_ARB                   0x2089
#define WGL_AUX3_ARB                   0x208A
#define WGL_AUX4_ARB                   0x208B
#define WGL_AUX5_ARB                   0x208C
#define WGL_AUX6_ARB                   0x208D
#define WGL_AUX7_ARB                   0x208E
#define WGL_AUX8_ARB                   0x208F
#define WGL_AUX9_ARB                   0x2090

typedef BOOL (WINAPI * PFNWGLBINDTEXIMAGEARBPROC) (HPBUFFERARB hPbuffer, int iBuffer);
typedef BOOL (WINAPI * PFNWGLRELEASETEXIMAGEARBPROC) (HPBUFFERARB hPbuffer, int iBuffer);
typedef BOOL (WINAPI * PFNWGLSETPBUFFERATTRIBARBPROC) (HPBUFFERARB hPbuffer, const int *piAttribList);

extern PFNWGLBINDTEXIMAGEARBPROC     wglBindTexImageARB;
extern PFNWGLRELEASETEXIMAGEARBPROC  wglReleaseTexImageARB;
extern PFNWGLSETPBUFFERATTRIBARBPROC wglSetPbufferAttribARB;

#endif


#ifndef WGL_ATI_pixel_format_float
#define WGL_ATI_pixel_format_float

#define WGL_TYPE_RGBA_FLOAT_ATI 0x21A0

#endif


#ifndef WGL_EXT_swap_control
#define WGL_EXT_swap_control

typedef BOOL (WINAPI * PFNWGLSWAPINTERVALEXTPROC) (int interval);
typedef int (WINAPI * PFNWGLGETSWAPINTERVALEXTPROC) (void);

extern PFNWGLSWAPINTERVALEXTPROC wglSwapIntervalEXT;
extern PFNWGLGETSWAPINTERVALEXTPROC wglGetSwapIntervalEXT;

#endif


#elif defined(LINUX)

#ifndef GLX_ATI_pixel_format_float
#define GLX_ATI_pixel_format_float

#define GLX_RGBA_FLOAT_ATI_BIT 0x00000100

#endif

#ifndef GLX_ATI_render_texture
#define GLX_ATI_render_texture

#define GLX_BIND_TO_TEXTURE_RGB_ATI         0x9800
#define GLX_BIND_TO_TEXTURE_RGBA_ATI        0x9801
#define GLX_TEXTURE_FORMAT_ATI              0x9802
#define GLX_TEXTURE_TARGET_ATI              0x9803
#define GLX_MIPMAP_TEXTURE_ATI              0x9804
#define GLX_TEXTURE_RGB_ATI                 0x9805
#define GLX_TEXTURE_RGBA_ATI                0x9806
#define GLX_NO_TEXTURE_ATI                  0x9807
#define GLX_TEXTURE_CUBE_MAP_ATI            0x9808
#define GLX_TEXTURE_1D_ATI                  0x9809
#define GLX_TEXTURE_2D_ATI                  0x980A
#define GLX_MIPMAP_LEVEL_ATI                0x980B
#define GLX_CUBE_MAP_FACE_ATI               0x980C
#define GLX_TEXTURE_CUBE_MAP_POSITIVE_X_ATI 0x980D
#define GLX_TEXTURE_CUBE_MAP_NEGATIVE_X_ATI 0x980E
#define GLX_TEXTURE_CUBE_MAP_POSITIVE_Y_ATI 0x980F
#define GLX_TEXTURE_CUBE_MAP_NEGATIVE_Y_ATI 0x9810
#define GLX_TEXTURE_CUBE_MAP_POSITIVE_Z_ATI 0x9811
#define GLX_TEXTURE_CUBE_MAP_NEGATIVE_Z_ATI 0x9812
#define GLX_FRONT_LEFT_ATI                  0x9813
#define GLX_FRONT_RIGHT_ATI                 0x9814
#define GLX_BACK_LEFT_ATI                   0x9815
#define GLX_BACK_RIGHT_ATI                  0x9816
#define GLX_AUX0_ATI                        0x9817
#define GLX_AUX1_ATI                        0x9818
#define GLX_AUX2_ATI                        0x9819
#define GLX_AUX3_ATI                        0x981A
#define GLX_AUX4_ATI                        0x981B
#define GLX_AUX5_ATI                        0x981C
#define GLX_AUX6_ATI                        0x981D
#define GLX_AUX7_ATI                        0x981E
#define GLX_AUX8_ATI                        0x981F
#define GLX_AUX9_ATI                        0x9820
#define GLX_BIND_TO_TEXTURE_LUMINANCE_ATI   0x9821
#define GLX_BIND_TO_TEXTURE_INTENSITY_ATI   0x9822

typedef void (* PFNGLXBINDTEXIMAGEATIPROC)(Display *dpy, GLXPbuffer pbuf, int buffer);
typedef void (* PFNGLXRELEASETEXIMAGEATIPROC)(Display *dpy, GLXPbuffer pbuf, int buffer);
typedef void (* PFNGLXDRAWABLEATTRIBATIPROC)(Display *dpy, GLXDrawable draw, const int *attrib_list);

extern PFNGLXBINDTEXIMAGEATIPROC    glXBindTexImageATI;
extern PFNGLXRELEASETEXIMAGEATIPROC glXReleaseTexImageATI;
extern PFNGLXDRAWABLEATTRIBATIPROC  glXDrawableAttribATI;

#endif

#elif defined(__APPLE__)

#endif









#ifndef GL_VERSION_1_2
#define GL_VERSION_1_2
#define GL_VERSION_1_2_PROTOTYPES

#define GL_TEXTURE_BINDING_3D    0x806A
#define GL_PACK_SKIP_IMAGES      0x806B
#define GL_PACK_IMAGE_HEIGHT     0x806C
#define GL_UNPACK_SKIP_IMAGES    0x806D
#define GL_UNPACK_IMAGE_HEIGHT   0x806E
#define GL_TEXTURE_3D            0x806F
#define GL_PROXY_TEXTURE_3D      0x8070
#define GL_TEXTURE_DEPTH         0x8071
#define GL_TEXTURE_WRAP_R        0x8072
#define GL_MAX_3D_TEXTURE_SIZE   0x8073

#define GL_BGR    0x80E0
#define GL_BGRA   0x80E1

#define GL_UNSIGNED_BYTE_3_3_2           0x8032
#define GL_UNSIGNED_SHORT_4_4_4_4        0x8033
#define GL_UNSIGNED_SHORT_5_5_5_1        0x8034
#define GL_UNSIGNED_INT_8_8_8_8          0x8035
#define GL_UNSIGNED_INT_10_10_10_2       0x8036
#define GL_UNSIGNED_BYTE_2_3_3_REV       0x8362
#define GL_UNSIGNED_SHORT_5_6_5          0x8363
#define GL_UNSIGNED_SHORT_5_6_5_REV      0x8364
#define GL_UNSIGNED_SHORT_4_4_4_4_REV    0x8365
#define GL_UNSIGNED_SHORT_1_5_5_5_REV    0x8366
#define GL_UNSIGNED_INT_8_8_8_8_REV      0x8367
#define GL_UNSIGNED_INT_2_10_10_10_REV   0x8368

#define GL_RESCALE_NORMAL   0x803A

#define GL_LIGHT_MODEL_COLOR_CONTROL   0x81F8
#define GL_SINGLE_COLOR                0x81F9
#define GL_SEPARATE_SPECULAR_COLOR     0x81FA

#define GL_CLAMP_TO_EDGE   0x812F

#define GL_TEXTURE_MIN_LOD      0x813A
#define GL_TEXTURE_MAX_LOD      0x813B
#define GL_TEXTURE_BASE_LEVEL   0x813C
#define GL_TEXTURE_MAX_LEVEL    0x813D

#define GL_MAX_ELEMENTS_VERTICES   0x80E8
#define GL_MAX_ELEMENTS_INDICES    0x80E9

#define GL_CONSTANT_COLOR             0x8001
#define GL_ONE_MINUS_CONSTANT_COLOR   0x8002
#define GL_CONSTANT_ALPHA             0x8003
#define GL_ONE_MINUS_CONSTANT_ALPHA   0x8004
#define GL_BLEND_COLOR                0x8005

#define GL_FUNC_ADD                0x8006
#define GL_MIN                     0x8007
#define GL_MAX                     0x8008
#define GL_BLEND_EQUATION          0x8009
#define GL_FUNC_SUBTRACT           0x800A
#define GL_FUNC_REVERSE_SUBTRACT   0x800B

extern PFNGLTEXIMAGE3DEXTPROC glTexImage3D;
extern PFNGLTEXSUBIMAGE3DPROC glTexSubImage3D;
extern PFNGLCOPYTEXSUBIMAGE3DPROC glCopyTexSubImage3D;
extern PFNGLDRAWRANGEELEMENTSEXTPROC glDrawRangeElements;
extern PFNGLBLENDCOLOREXTPROC glBlendColor;
extern PFNGLBLENDEQUATIONEXTPROC glBlendEquation;

#endif // GL_VERSION_1_2





#ifndef GL_VERSION_1_3
#define GL_VERSION_1_3
#define GL_VERSION_1_3_PROTOTYPES

#define GL_TEXTURE0    0x84C0
#define GL_TEXTURE1    0x84C1
#define GL_TEXTURE2    0x84C2
#define GL_TEXTURE3    0x84C3
#define GL_TEXTURE4    0x84C4
#define GL_TEXTURE5    0x84C5
#define GL_TEXTURE6    0x84C6
#define GL_TEXTURE7    0x84C7
#define GL_TEXTURE8    0x84C8
#define GL_TEXTURE9    0x84C9
#define GL_TEXTURE10   0x84CA
#define GL_TEXTURE11   0x84CB
#define GL_TEXTURE12   0x84CC
#define GL_TEXTURE13   0x84CD
#define GL_TEXTURE14   0x84CE
#define GL_TEXTURE15   0x84CF
#define GL_TEXTURE16   0x84D0
#define GL_TEXTURE17   0x84D1
#define GL_TEXTURE18   0x84D2
#define GL_TEXTURE19   0x84D3
#define GL_TEXTURE20   0x84D4
#define GL_TEXTURE21   0x84D5
#define GL_TEXTURE22   0x84D6
#define GL_TEXTURE23   0x84D7
#define GL_TEXTURE24   0x84D8
#define GL_TEXTURE25   0x84D9
#define GL_TEXTURE26   0x84DA
#define GL_TEXTURE27   0x84DB
#define GL_TEXTURE28   0x84DC
#define GL_TEXTURE29   0x84DD
#define GL_TEXTURE30   0x84DE
#define GL_TEXTURE31   0x84DF
#define GL_ACTIVE_TEXTURE          0x84E0
#define GL_CLIENT_ACTIVE_TEXTURE   0x84E1
#define GL_MAX_TEXTURE_UNITS       0x84E2

#define GL_COMPRESSED_ALPHA                 0x84E9
#define GL_COMPRESSED_LUMINANCE             0x84EA
#define GL_COMPRESSED_LUMINANCE_ALPHA       0x84EB
#define GL_COMPRESSED_INTENSITY             0x84EC
#define GL_COMPRESSED_RGB                   0x84ED
#define GL_COMPRESSED_RGBA                  0x84EE
#define GL_TEXTURE_COMPRESSION_HINT         0x84EF
#define GL_TEXTURE_COMPRESSED_IMAGE_SIZE    0x86A0
#define GL_TEXTURE_COMPRESSED               0x86A1
#define GL_NUM_COMPRESSED_TEXTURE_FORMATS   0x86A2
#define GL_COMPRESSED_TEXTURE_FORMATS       0x86A3

#define GL_NORMAL_MAP                    0x8511
#define GL_REFLECTION_MAP                0x8512
#define GL_TEXTURE_CUBE_MAP              0x8513
#define GL_TEXTURE_BINDING_CUBE_MAP      0x8514
#define GL_TEXTURE_CUBE_MAP_POSITIVE_X   0x8515
#define GL_TEXTURE_CUBE_MAP_NEGATIVE_X   0x8516
#define GL_TEXTURE_CUBE_MAP_POSITIVE_Y   0x8517
#define GL_TEXTURE_CUBE_MAP_NEGATIVE_Y   0x8518
#define GL_TEXTURE_CUBE_MAP_POSITIVE_Z   0x8519
#define GL_TEXTURE_CUBE_MAP_NEGATIVE_Z   0x851A
#define GL_PROXY_TEXTURE_CUBE_MAP        0x851B
#define GL_MAX_CUBE_MAP_TEXTURE_SIZE     0x851C

#define GL_MULTISAMPLE                0x809D
#define GL_SAMPLE_ALPHA_TO_COVERAGE   0x809E
#define GL_SAMPLE_ALPHA_TO_ONE        0x809F
#define GL_SAMPLE_COVERAGE            0x80A0
#define GL_SAMPLE_BUFFERS             0x80A8
#define GL_SAMPLES                    0x80A9
#define GL_SAMPLE_COVERAGE_VALUE      0x80AA
#define GL_SAMPLE_COVERAGE_INVERT     0x80AB
#define GL_MULTISAMPLE_BIT            0x20000000

#define GL_COMBINE          0x8570
#define GL_COMBINE_RGB      0x8571
#define GL_COMBINE_ALPHA    0x8572
#define GL_SOURCE0_RGB      0x8580
#define GL_SOURCE1_RGB      0x8581
#define GL_SOURCE2_RGB      0x8582
#define GL_SOURCE0_ALPHA    0x8588
#define GL_SOURCE1_ALPHA    0x8589
#define GL_SOURCE2_ALPHA    0x858A
#define GL_OPERAND0_RGB     0x8590
#define GL_OPERAND1_RGB     0x8591
#define GL_OPERAND2_RGB     0x8592
#define GL_OPERAND0_ALPHA   0x8598
#define GL_OPERAND1_ALPHA   0x8599
#define GL_OPERAND2_ALPHA   0x859A
#define GL_RGB_SCALE        0x8573
#define GL_ADD_SIGNED       0x8574
#define GL_INTERPOLATE      0x8575
#define GL_SUBTRACT         0x84E7
#define GL_CONSTANT         0x8576
#define GL_PRIMARY_COLOR    0x8577
#define GL_PREVIOUS         0x8578

#define GL_DOT3_RGB    0x86AE
#define GL_DOT3_RGBA   0x86AF

#define GL_CLAMP_TO_BORDER   0x812D

#define GL_TRANSPOSE_MODELVIEW_MATRIX    0x84E3
#define GL_TRANSPOSE_PROJECTION_MATRIX   0x84E4
#define GL_TRANSPOSE_TEXTURE_MATRIX      0x84E5
#define GL_TRANSPOSE_COLOR_MATRIX        0x84E6

extern PFNGLACTIVETEXTUREARBPROC glActiveTexture;
extern PFNGLCLIENTACTIVETEXTUREARBPROC glClientActiveTexture;
extern PFNGLMULTITEXCOORD1DARBPROC  glMultiTexCoord1d;
extern PFNGLMULTITEXCOORD1DVARBPROC glMultiTexCoord1dv;
extern PFNGLMULTITEXCOORD1FARBPROC  glMultiTexCoord1f;
extern PFNGLMULTITEXCOORD1FVARBPROC glMultiTexCoord1fv;
extern PFNGLMULTITEXCOORD1IARBPROC  glMultiTexCoord1i;
extern PFNGLMULTITEXCOORD1IVARBPROC glMultiTexCoord1iv;
extern PFNGLMULTITEXCOORD1SARBPROC  glMultiTexCoord1s;
extern PFNGLMULTITEXCOORD1SVARBPROC glMultiTexCoord1sv;
extern PFNGLMULTITEXCOORD2DARBPROC  glMultiTexCoord2d;
extern PFNGLMULTITEXCOORD2DVARBPROC glMultiTexCoord2dv;
extern PFNGLMULTITEXCOORD2FARBPROC  glMultiTexCoord2f;
extern PFNGLMULTITEXCOORD2FVARBPROC glMultiTexCoord2fv;
extern PFNGLMULTITEXCOORD2IARBPROC  glMultiTexCoord2i;
extern PFNGLMULTITEXCOORD2IVARBPROC glMultiTexCoord2iv;
extern PFNGLMULTITEXCOORD2SARBPROC  glMultiTexCoord2s;
extern PFNGLMULTITEXCOORD2SVARBPROC glMultiTexCoord2sv;
extern PFNGLMULTITEXCOORD3DARBPROC  glMultiTexCoord3d;
extern PFNGLMULTITEXCOORD3DVARBPROC glMultiTexCoord3dv;
extern PFNGLMULTITEXCOORD3FARBPROC  glMultiTexCoord3f;
extern PFNGLMULTITEXCOORD3FVARBPROC glMultiTexCoord3fv;
extern PFNGLMULTITEXCOORD3IARBPROC  glMultiTexCoord3i;
extern PFNGLMULTITEXCOORD3IVARBPROC glMultiTexCoord3iv;
extern PFNGLMULTITEXCOORD3SARBPROC  glMultiTexCoord3s;
extern PFNGLMULTITEXCOORD3SVARBPROC glMultiTexCoord3sv;
extern PFNGLMULTITEXCOORD4DARBPROC  glMultiTexCoord4d;
extern PFNGLMULTITEXCOORD4DVARBPROC glMultiTexCoord4dv;
extern PFNGLMULTITEXCOORD4FARBPROC  glMultiTexCoord4f;
extern PFNGLMULTITEXCOORD4FVARBPROC glMultiTexCoord4fv;
extern PFNGLMULTITEXCOORD4IARBPROC  glMultiTexCoord4i;
extern PFNGLMULTITEXCOORD4IVARBPROC glMultiTexCoord4iv;
extern PFNGLMULTITEXCOORD4SARBPROC  glMultiTexCoord4s;
extern PFNGLMULTITEXCOORD4SVARBPROC glMultiTexCoord4sv;

extern PFNGLCOMPRESSEDTEXIMAGE1DARBPROC glCompressedTexImage1D;
extern PFNGLCOMPRESSEDTEXIMAGE2DARBPROC glCompressedTexImage2D;
extern PFNGLCOMPRESSEDTEXIMAGE3DARBPROC glCompressedTexImage3D;
extern PFNGLCOMPRESSEDTEXSUBIMAGE1DARBPROC glCompressedTexSubImage1D;
extern PFNGLCOMPRESSEDTEXSUBIMAGE2DARBPROC glCompressedTexSubImage2D;
extern PFNGLCOMPRESSEDTEXSUBIMAGE3DARBPROC glCompressedTexSubImage3D;
extern PFNGLGETCOMPRESSEDTEXIMAGEARBPROC glGetCompressedTexImage;

extern PFNGLSAMPLECOVERAGEARBPROC glSampleCoverage;

extern PFNGLLOADTRANSPOSEMATRIXFARBPROC glLoadTransposeMatrixf;
extern PFNGLLOADTRANSPOSEMATRIXDARBPROC glLoadTransposeMatrixd;
extern PFNGLMULTTRANSPOSEMATRIXFARBPROC glMultTransposeMatrixf;
extern PFNGLMULTTRANSPOSEMATRIXDARBPROC glMultTransposeMatrixd;

#endif // GL_VERSION_1_3






#ifndef GL_VERSION_1_4
#define GL_VERSION_1_4
#define GL_VERSION_1_4_PROTOTYPES

#define GL_GENERATE_MIPMAP        0x8191
#define GL_GENERATE_MIPMAP_HINT   0x8192

#define GL_DEPTH_COMPONENT16    0x81A5
#define GL_DEPTH_COMPONENT24    0x81A6
#define GL_DEPTH_COMPONENT32    0x81A7
#define GL_TEXTURE_DEPTH_SIZE   0x884A
#define GL_DEPTH_TEXTURE_MODE   0x884B

#define GL_TEXTURE_COMPARE_MODE   0x884C
#define GL_TEXTURE_COMPARE_FUNC   0x884D
#define GL_COMPARE_R_TO_TEXTURE   0x884E

#define GL_FOG_COORDINATE_SOURCE          0x8450
#define GL_FOG_COORDINATE                 0x8451
#define GL_FRAGMENT_DEPTH                 0x8452
#define GL_CURRENT_FOG_COORDINATE         0x8453
#define GL_FOG_COORDINATE_ARRAY_TYPE      0x8454
#define GL_FOG_COORDINATE_ARRAY_STRIDE    0x8455
#define GL_FOG_COORDINATE_ARRAY_POINTER   0x8456
#define GL_FOG_COORDINATE_ARRAY           0x8457

#define GL_POINT_SIZE_MIN               0x8126
#define GL_POINT_SIZE_MAX               0x8127
#define GL_POINT_FADE_THRESHOLD_SIZE    0x8128
#define GL_POINT_DISTANCE_ATTENUATION   0x8129

#define GL_COLOR_SUM                       0x8458
#define GL_CURRENT_SECONDARY_COLOR         0x8459
#define GL_SECONDARY_COLOR_ARRAY_SIZE      0x845A
#define GL_SECONDARY_COLOR_ARRAY_TYPE      0x845B
#define GL_SECONDARY_COLOR_ARRAY_STRIDE    0x845C
#define GL_SECONDARY_COLOR_ARRAY_POINTER   0x845D
#define GL_SECONDARY_COLOR_ARRAY           0x845E

#define GL_BLEND_DST_RGB     0x80C8
#define GL_BLEND_SRC_RGB     0x80C9
#define GL_BLEND_DST_ALPHA   0x80CA
#define GL_BLEND_SRC_ALPHA   0x80CB

#define GL_INCR_WRAP   0x8507
#define GL_DECR_WRAP   0x8508

#define GL_TEXTURE_FILTER_CONTROL   0x8500
#define GL_TEXTURE_LOD_BIAS         0x8501
#define GL_MAX_TEXTURE_LOD_BIAS     0x84FD

#define GL_MIRRORED_REPEAT   0x8370

extern PFNGLFOGCOORDFEXTPROC  glFogCoordf;
extern PFNGLFOGCOORDDEXTPROC  glFogCoordd;
extern PFNGLFOGCOORDFVEXTPROC glFogCoordfv;
extern PFNGLFOGCOORDDVEXTPROC glFogCoorddv;
extern PFNGLFOGCOORDPOINTEREXTPROC glFogCoordPointer;

extern PFNGLMULTIDRAWARRAYSEXTPROC glMultiDrawArrays;
extern PFNGLMULTIDRAWELEMENTSEXTPROC glMultiDrawElements;

extern PFNGLPOINTPARAMETERFARBPROC glPointParameterf;
extern PFNGLPOINTPARAMETERFVARBPROC glPointParameterfv;

extern PFNGLSECONDARYCOLOR3FEXTPROC glSecondaryColor3f;
extern PFNGLSECONDARYCOLOR3DEXTPROC glSecondaryColor3d;
extern PFNGLSECONDARYCOLOR3BEXTPROC glSecondaryColor3b;
extern PFNGLSECONDARYCOLOR3SEXTPROC glSecondaryColor3s;
extern PFNGLSECONDARYCOLOR3IEXTPROC glSecondaryColor3i;
extern PFNGLSECONDARYCOLOR3UBEXTPROC glSecondaryColor3ub;
extern PFNGLSECONDARYCOLOR3USEXTPROC glSecondaryColor3us;
extern PFNGLSECONDARYCOLOR3UIEXTPROC glSecondaryColor3ui;

extern PFNGLSECONDARYCOLOR3FVEXTPROC glSecondaryColor3fv;
extern PFNGLSECONDARYCOLOR3DVEXTPROC glSecondaryColor3dv;
extern PFNGLSECONDARYCOLOR3BVEXTPROC glSecondaryColor3bv;
extern PFNGLSECONDARYCOLOR3SVEXTPROC glSecondaryColor3sv;
extern PFNGLSECONDARYCOLOR3IVEXTPROC glSecondaryColor3iv;
extern PFNGLSECONDARYCOLOR3UBVEXTPROC glSecondaryColor3ubv;
extern PFNGLSECONDARYCOLOR3USVEXTPROC glSecondaryColor3usv;
extern PFNGLSECONDARYCOLOR3UIVEXTPROC glSecondaryColor3uiv;

extern PFNGLSECONDARYCOLORPOINTEREXTPROC glSecondaryColorPointer;

extern PFNGLBLENDFUNCSEPARATEEXTPROC glBlendFuncSeparate;

extern PFNGLWINDOWPOS2DARBPROC  glWindowPos2d;
extern PFNGLWINDOWPOS2FARBPROC  glWindowPos2f;
extern PFNGLWINDOWPOS2IARBPROC  glWindowPos2i;
extern PFNGLWINDOWPOS2SARBPROC  glWindowPos2s;
extern PFNGLWINDOWPOS2IVARBPROC glWindowPos2iv;
extern PFNGLWINDOWPOS2SVARBPROC glWindowPos2sv;
extern PFNGLWINDOWPOS2FVARBPROC glWindowPos2fv;
extern PFNGLWINDOWPOS2DVARBPROC glWindowPos2dv;
extern PFNGLWINDOWPOS3IARBPROC  glWindowPos3i;
extern PFNGLWINDOWPOS3SARBPROC  glWindowPos3s;
extern PFNGLWINDOWPOS3FARBPROC  glWindowPos3f;
extern PFNGLWINDOWPOS3DARBPROC  glWindowPos3d;
extern PFNGLWINDOWPOS3IVARBPROC glWindowPos3iv;
extern PFNGLWINDOWPOS3SVARBPROC glWindowPos3sv;
extern PFNGLWINDOWPOS3FVARBPROC glWindowPos3fv;
extern PFNGLWINDOWPOS3DVARBPROC glWindowPos3dv;

#endif // GL_VERSION_1_4





#ifndef GL_VERSION_1_5
#define GL_VERSION_1_5
#define GL_VERSION_1_5_PROTOTYPES

#define GL_ARRAY_BUFFER                   0x8892
#define GL_ELEMENT_ARRAY_BUFFER           0x8893
#define GL_ARRAY_BUFFER_BINDING           0x8894
#define GL_ELEMENT_ARRAY_BUFFER_BINDING   0x8895
#define GL_VERTEX_ARRAY_BUFFER_BINDING    0x8896
#define GL_NORMAL_ARRAY_BUFFER_BINDING    0x8897
#define GL_COLOR_ARRAY_BUFFER_BINDING     0x8898
#define GL_INDEX_ARRAY_BUFFER_BINDING     0x8899
#define GL_TEXTURE_COORD_ARRAY_BUFFER_BINDING     0x889A
#define GL_EDGE_supported_ARRAY_BUFFER_BINDING         0x889B
#define GL_SECONDARY_COLOR_ARRAY_BUFFER_BINDING   0x889C
#define GL_FOG_COORDINATE_ARRAY_BUFFER_BINDING    0x889D
#define GL_WEIGHT_ARRAY_BUFFER_BINDING            0x889E
#define GL_VERTEX_ATTRIB_ARRAY_BUFFER_BINDING     0x889F
#define GL_STREAM_DRAW          0x88E0
#define GL_STREAM_READ          0x88E1
#define GL_STREAM_COPY          0x88E2
#define GL_STATIC_DRAW          0x88E4
#define GL_STATIC_READ          0x88E5
#define GL_STATIC_COPY          0x88E6
#define GL_DYNAMIC_DRAW         0x88E8
#define GL_DYNAMIC_READ         0x88E9
#define GL_DYNAMIC_COPY         0x88EA
#define GL_READ_ONLY            0x88B8
#define GL_WRITE_ONLY           0x88B9
#define GL_READ_WRITE           0x88BA
#define GL_BUFFER_SIZE          0x8764
#define GL_BUFFER_USAGE         0x8765
#define GL_BUFFER_ACCESS        0x88BB
#define GL_BUFFER_MAPPED        0x88BC
#define GL_BUFFER_MAP_POINTER   0x88BD

#define GL_QUERY_COUNTER_BITS       0x8864
#define GL_CURRENT_QUERY            0x8865
#define GL_QUERY_RESULT             0x8866
#define GL_QUERY_RESULT_AVAILABLE   0x8867
#define GL_SAMPLES_PASSED           0x8914

#define GL_FOG_COORD_SRC       GL_FOG_COORDINATE_SOURCE
#define GL_FOG_COORD           GL_FOG_COORDINATE
#define GL_CURRENT_FOG_COORD   GL_CURRENT_FOG_COORDINATE
#define GL_FOG_COORD_ARRAY_TYPE             GL_FOG_COORDINATE_ARRAY_TYPE
#define GL_FOG_COORD_ARRAY_STRIDE           GL_FOG_COORDINATE_ARRAY_STRIDE
#define GL_FOG_COORD_ARRAY_POINTER          GL_FOG_COORDINATE_ARRAY_POINTER
#define GL_FOG_COORD_ARRAY                  GL_FOG_COORDINATE_ARRAY
#define GL_FOG_COORD_ARRAY_BUFFER_BINDING   GL_FOG_COORDINATE_ARRAY_BUFFER_BINDING
#define GL_SRC0_RGB     GL_SOURCE0_RGB
#define GL_SRC1_RGB     GL_SOURCE1_RGB
#define GL_SRC2_RGB     GL_SOURCE2_RGB
#define GL_SRC0_ALPHA   GL_SOURCE0_ALPHA
#define GL_SRC1_ALPHA   GL_SOURCE1_ALPHA
#define GL_SRC2_ALPHA   GL_SOURCE2_ALPHA

extern PFNGLBINDBUFFERARBPROC    glBindBuffer;
extern PFNGLDELETEBUFFERSARBPROC glDeleteBuffers;
extern PFNGLGENBUFFERSARBPROC    glGenBuffers;
extern PFNGLISBUFFERARBPROC      glIsBuffer;
extern PFNGLBUFFERDATAARBPROC    glBufferData;
extern PFNGLBUFFERSUBDATAARBPROC glBufferSubData;
extern PFNGLGETBUFFERSUBDATAARBPROC glGetBufferSubData;
extern PFNGLMAPBUFFERARBPROC   glMapBuffer;
extern PFNGLUNMAPBUFFERARBPROC glUnmapBuffer;
extern PFNGLGETBUFFERPARAMETERIVARBPROC glGetBufferParameteriv;
extern PFNGLGETBUFFERPOINTERVARBPROC    glGetBufferPointerv;

extern PFNGLGENQUERIESARBPROC    glGenQueries;
extern PFNGLDELETEQUERIESARBPROC glDeleteQueries;
extern PFNGLISQUERYARBPROC    glIsQuery;
extern PFNGLBEGINQUERYARBPROC glBeginQuery;
extern PFNGLENDQUERYARBPROC   glEndQuery;
extern PFNGLGETQUERYIVARBPROC glGetQueryiv;
extern PFNGLGETQUERYOBJECTIVARBPROC  glGetQueryObjectiv;
extern PFNGLGETQUERYOBJECTUIVARBPROC glGetQueryObjectuiv;

#endif // GL_VERSION_1_5





#ifndef GL_VERSION_2_0
#define GL_VERSION_2_0
#define GL_VERSION_2_0_PROTOTYPES

typedef GLvoid (APIENTRY *PFNGLSTENCILOPSEPARATEPROC)(GLenum face, GLenum sfail, GLenum dpfail, GLenum dppass);
typedef GLvoid (APIENTRY *PFNGLSTENCILFUNCSEPARATEPROC)(GLenum face, GLenum func, GLint ref, GLuint mask);
typedef GLvoid (APIENTRY *PFNGLSTENCILMASKSEPARATEPROC)(GLenum face, GLuint mask);
typedef GLvoid (APIENTRY *PFNGLBLENDEQUATIONSEPARATEPROC)(GLenum modeRGB, GLenum modeAlpha);

#define GL_PROGRAM_OBJECT   0x8B40
#define GL_SHADER_OBJECT    0x8B48

#define GL_OBJECT_TYPE                        0x8B4E
#define GL_OBJECT_SUBTYPE                     0x8B4F
#define GL_OBJECT_DELETE_STATUS               0x8B80
#define GL_OBJECT_COMPILE_STATUS              0x8B81
#define GL_OBJECT_LINK_STATUS                 0x8B82
#define GL_OBJECT_VALIDATE_STATUS             0x8B83
#define GL_OBJECT_INFO_LOG_LENGTH             0x8B84
#define GL_OBJECT_ATTACHED_OBJECTS            0x8B85
#define GL_OBJECT_ACTIVE_UNIFORMS             0x8B86
#define GL_OBJECT_ACTIVE_UNIFORM_MAX_LENGTH   0x8B87
#define GL_OBJECT_SHADER_SOURCE_LENGTH        0x8B88

#define GL_FLOAT_VEC2   0x8B50
#define GL_FLOAT_VEC3   0x8B51
#define GL_FLOAT_VEC4   0x8B52
#define GL_INT_VEC2     0x8B53
#define GL_INT_VEC3     0x8B54
#define GL_INT_VEC4     0x8B55
#define GL_BOOL         0x8B56
#define GL_BOOL_VEC2    0x8B57
#define GL_BOOL_VEC3    0x8B58
#define GL_BOOL_VEC4    0x8B59
#define GL_FLOAT_MAT2   0x8B5A
#define GL_FLOAT_MAT3   0x8B5B
#define GL_FLOAT_MAT4   0x8B5C
#define GL_SAMPLER_1D   0x8B5D
#define GL_SAMPLER_2D   0x8B5E
#define GL_SAMPLER_3D   0x8B5F
#define GL_SAMPLER_CUBE 0x8B60
#define GL_SAMPLER_1D_SHADOW      0x8B61
#define GL_SAMPLER_2D_SHADOW      0x8B62
#define GL_SAMPLER_2D_RECT        0x8B63
#define GL_SAMPLER_2D_RECT_SHADOW 0x8B64


#define GL_VERTEX_SHADER     0x8B31
#define GL_FRAGMENT_SHADER   0x8B30

#define GL_MAX_FRAGMENT_UNIFORM_COMPONENTS_ARB   0x8B49
#define GL_MAX_VERTEX_UNIFORM_COMPONENTS         0x8B4A
#define GL_MAX_VARYING_FLOATS                    0x8B4B
#define GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS        0x8B4C
#define GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS      0x8B4D

#define GL_OBJECT_ACTIVE_ATTRIBUTES             0x8B89
#define GL_OBJECT_ACTIVE_ATTRIBUTE_MAX_LENGTH   0x8B8A

#define GL_SHADING_LANGUAGE_VERSION   0x8B8C

extern PFNGLDELETEOBJECTARBPROC glDeleteProgram;
extern PFNGLDELETEOBJECTARBPROC glDeleteShader;
//extern PFNGLGETHANDLEARBPROC glGetHandle;
extern PFNGLDETACHOBJECTARBPROC glDetachShader;
extern PFNGLCREATESHADEROBJECTARBPROC glCreateShader;
extern PFNGLSHADERSOURCEARBPROC glShaderSource;
extern PFNGLCOMPILESHADERARBPROC glCompileShader;
extern PFNGLCREATEPROGRAMOBJECTARBPROC glCreateProgram;
extern PFNGLATTACHOBJECTARBPROC glAttachShader;
extern PFNGLLINKPROGRAMARBPROC glLinkProgram;
extern PFNGLUSEPROGRAMOBJECTARBPROC glUseProgram;
extern PFNGLVALIDATEPROGRAMARBPROC glValidateProgram;
extern PFNGLUNIFORM1FARBPROC glUniform1f;
extern PFNGLUNIFORM2FARBPROC glUniform2f;
extern PFNGLUNIFORM3FARBPROC glUniform3f;
extern PFNGLUNIFORM4FARBPROC glUniform4f;
extern PFNGLUNIFORM1IARBPROC glUniform1i;
extern PFNGLUNIFORM2IARBPROC glUniform2i;
extern PFNGLUNIFORM3IARBPROC glUniform3i;
extern PFNGLUNIFORM4IARBPROC glUniform4i;
extern PFNGLUNIFORM1FVARBPROC glUniform1fv;
extern PFNGLUNIFORM2FVARBPROC glUniform2fv;
extern PFNGLUNIFORM3FVARBPROC glUniform3fv;
extern PFNGLUNIFORM4FVARBPROC glUniform4fv;
extern PFNGLUNIFORM1IVARBPROC glUniform1iv;
extern PFNGLUNIFORM2IVARBPROC glUniform2iv;
extern PFNGLUNIFORM3IVARBPROC glUniform3iv;
extern PFNGLUNIFORM4IVARBPROC glUniform4iv;
extern PFNGLUNIFORMMATRIX2FVARBPROC glUniformMatrix2fv;
extern PFNGLUNIFORMMATRIX3FVARBPROC glUniformMatrix3fv;
extern PFNGLUNIFORMMATRIX4FVARBPROC glUniformMatrix4fv;
//extern PFNGLGETOBJECTPARAMETERFVARBPROC glGetObjectParameterfv;
//extern PFNGLGETOBJECTPARAMETERIVARBPROC glGetObjectParameteriv;
//extern PFNGLGETINFOLOGARBPROC glGetInfoLog;
//extern PFNGLGETATTACHEDOBJECTSARBPROC glGetAttachedObjects;
extern PFNGLGETUNIFORMLOCATIONARBPROC glGetUniformLocation;
extern PFNGLGETACTIVEUNIFORMARBPROC glGetActiveUniform;
extern PFNGLGETUNIFORMFVARBPROC glGetUniformfv;
extern PFNGLGETUNIFORMIVARBPROC glGetUniformiv;
extern PFNGLGETSHADERSOURCEARBPROC glGetShaderSource;

extern PFNGLBINDATTRIBLOCATIONARBPROC glBindAttribLocation;
extern PFNGLGETACTIVEATTRIBARBPROC    glGetActiveAttrib;
extern PFNGLGETATTRIBLOCATIONARBPROC  glGetAttribLocation;
extern PFNGLVERTEXATTRIB1SARBPROC glVertexAttrib1s;
extern PFNGLVERTEXATTRIB1FARBPROC glVertexAttrib1f;
extern PFNGLVERTEXATTRIB1DARBPROC glVertexAttrib1d;
extern PFNGLVERTEXATTRIB2SARBPROC glVertexAttrib2s;
extern PFNGLVERTEXATTRIB2FARBPROC glVertexAttrib2f;
extern PFNGLVERTEXATTRIB2DARBPROC glVertexAttrib2d;
extern PFNGLVERTEXATTRIB3SARBPROC glVertexAttrib3s;
extern PFNGLVERTEXATTRIB3FARBPROC glVertexAttrib3f;
extern PFNGLVERTEXATTRIB3DARBPROC glVertexAttrib3d;
extern PFNGLVERTEXATTRIB4SARBPROC glVertexAttrib4s;
extern PFNGLVERTEXATTRIB4FARBPROC glVertexAttrib4f;
extern PFNGLVERTEXATTRIB4DARBPROC glVertexAttrib4d;
extern PFNGLVERTEXATTRIB4NUBARBPROC glVertexAttrib4Nub;
extern PFNGLVERTEXATTRIB1SVARBPROC glVertexAttrib1sv;
extern PFNGLVERTEXATTRIB1FVARBPROC glVertexAttrib1fv;
extern PFNGLVERTEXATTRIB1DVARBPROC glVertexAttrib1dv;
extern PFNGLVERTEXATTRIB2SVARBPROC glVertexAttrib2sv;
extern PFNGLVERTEXATTRIB2FVARBPROC glVertexAttrib2fv;
extern PFNGLVERTEXATTRIB2DVARBPROC glVertexAttrib2dv;
extern PFNGLVERTEXATTRIB3SVARBPROC glVertexAttrib3sv;
extern PFNGLVERTEXATTRIB3FVARBPROC glVertexAttrib3fv;
extern PFNGLVERTEXATTRIB3DVARBPROC glVertexAttrib3dv;
extern PFNGLVERTEXATTRIB4BVARBPROC glVertexAttrib4bv;
extern PFNGLVERTEXATTRIB4SVARBPROC glVertexAttrib4sv;
extern PFNGLVERTEXATTRIB4IVARBPROC glVertexAttrib4iv;
extern PFNGLVERTEXATTRIB4UBVARBPROC glVertexAttrib4ubv;
extern PFNGLVERTEXATTRIB4USVARBPROC glVertexAttrib4usv;
extern PFNGLVERTEXATTRIB4UIVARBPROC glVertexAttrib4uiv;
extern PFNGLVERTEXATTRIB4FVARBPROC glVertexAttrib4fv;
extern PFNGLVERTEXATTRIB4DVARBPROC glVertexAttrib4dv;
extern PFNGLVERTEXATTRIB4NBVARBPROC glVertexAttrib4Nbv;
extern PFNGLVERTEXATTRIB4NSVARBPROC glVertexAttrib4Nsv;
extern PFNGLVERTEXATTRIB4NIVARBPROC glVertexAttrib4Niv;
extern PFNGLVERTEXATTRIB4NUBVARBPROC glVertexAttrib4Nubv;
extern PFNGLVERTEXATTRIB4NUSVARBPROC glVertexAttrib4Nusv;
extern PFNGLVERTEXATTRIB4NUIVARBPROC glVertexAttrib4Nuiv;
extern PFNGLVERTEXATTRIBPOINTERARBPROC glVertexAttribPointer;
extern PFNGLENABLEVERTEXATTRIBARRAYARBPROC glEnableVertexAttribArray;
extern PFNGLDISABLEVERTEXATTRIBARRAYARBPROC glDisableVertexAttribArray;
extern PFNGLGETVERTEXATTRIBDVARBPROC glGetVertexAttribdv;
extern PFNGLGETVERTEXATTRIBFVARBPROC glGetVertexAttribfv;
extern PFNGLGETVERTEXATTRIBIVARBPROC glGetVertexAttribiv;
extern PFNGLGETVERTEXATTRIBPOINTERVARBPROC glGetVertexAttribPointerv;

extern PFNGLDRAWBUFFERSARBPROC glDrawBuffers;

extern PFNGLSTENCILOPSEPARATEPROC   glStencilOpSeparate;
extern PFNGLSTENCILFUNCSEPARATEPROC glStencilFuncSeparate;
extern PFNGLSTENCILMASKSEPARATEPROC glStencilMaskSeparate;

extern PFNGLBLENDEQUATIONSEPARATEPROC glBlendEquationSeparate;

#endif // GL_VERSION_2_0





#if defined(_WIN32)
void initExtensions(HDC hdc);
#elif defined(LINUX)
void initExtensions(Display *display);
#elif defined(__APPLE__)
void initExtensions();
#endif

#endif // _OPENGLEXTENSIONS_H_
