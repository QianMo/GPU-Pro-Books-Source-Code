
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

#include "OpenGLExtensions.h"

#include <stdlib.h>
#include <string.h>

bool GL_ARB_depth_texture_supported        = false;
bool GL_ARB_draw_buffers_supported         = false;
bool GL_ARB_fragment_program_supported     = false;
bool GL_ARB_fragment_shader_supported      = false;
bool GL_ARB_multisample_supported          = false;
bool GL_ARB_multitexture_supported         = false;
bool GL_ARB_occlusion_query_supported      = false;
bool GL_ARB_point_parameters_supported     = false;
bool GL_ARB_shader_objects_supported       = false;
bool GL_ARB_shading_language_100_supported = false;
bool GL_ARB_shadow_supported               = false;
bool GL_ARB_shadow_ambient_supported       = false;
bool GL_ARB_texture_compression_supported  = false;
bool GL_ARB_texture_cube_map_supported     = false;
bool GL_ARB_texture_env_add_supported      = false;
bool GL_ARB_texture_env_combine_supported  = false;
bool GL_ARB_texture_env_crossbar_supported = false;
bool GL_ARB_texture_env_dot3_supported     = false;
bool GL_ARB_transpose_matrix_supported     = false;
bool GL_ARB_vertex_buffer_object_supported = false;
bool GL_ARB_vertex_program_supported       = false;
bool GL_ARB_vertex_shader_supported        = false;
bool GL_ARB_window_pos_supported           = false;

bool GL_ATI_fragment_shader_supported         = false;
bool GL_ATI_separate_stencil_supported        = false;
bool GL_ATI_texture_compression_3dc_supported = false;
bool GL_ATI_texture_float_supported           = false;
bool GL_ATI_texture_mirror_once_supported     = false;

bool GL_EXT_blend_color_supported                = false;
bool GL_EXT_blend_func_separate_supported        = false;
bool GL_EXT_blend_minmax_supported               = false;
bool GL_EXT_blend_subtract_supported             = false;
bool GL_EXT_draw_range_elements_supported        = false;
bool GL_EXT_fog_coord_supported                  = false;
bool GL_EXT_framebuffer_object_supported         = false;
bool GL_EXT_multi_draw_arrays_supported          = false;
bool GL_EXT_packed_pixels_supported              = false;
bool GL_EXT_packed_depth_stencil_supported       = false;
bool GL_EXT_secondary_color_supported            = false;
bool GL_EXT_stencil_wrap_supported               = false;
bool GL_EXT_texture3D_supported                  = false;
bool GL_EXT_texture_compression_s3tc_supported   = false;
bool GL_EXT_texture_edge_clamp_supported         = false;
bool GL_EXT_texture_filter_anisotropic_supported = false;
bool GL_EXT_texture_lod_bias_supported           = false;

bool GL_HP_occlusion_test_supported    = false;

bool GL_NV_blend_square_supported      = false;

bool GL_SGIS_generate_mipmap_supported = false;

#if defined(_WIN32)

bool WGL_ARB_extensions_string_supported  = false;
bool WGL_ARB_make_current_read_supported  = false;
bool WGL_ARB_multisample_supported        = false;
bool WGL_ARB_pbuffer_supported            = false;
bool WGL_ARB_pixel_format_supported       = false;
bool WGL_ARB_render_texture_supported     = false;

bool WGL_ATI_pixel_format_float_supported = false;

bool WGL_EXT_swap_control_supported       = false;

#elif defined(LINUX)

bool GLX_ATI_pixel_format_float_supported = false;
bool GLX_ATI_render_texture_supported = false;

#elif defined(__APPLE__)

#import <mach-o/dyld.h>
#import <stdlib.h>
#import <string.h>

void *NSGLGetProcAddress(const char *name){
    NSSymbol symbol;

    // Prepend a '_' for the Unix C symbol mangling convention
    char *symbolName = (char *) malloc(strlen (name) + 2);
    strcpy(symbolName + 1, name);
    symbolName[0] = '_';
    symbol = NULL;
    if (NSIsSymbolNameDefined(symbolName))
        symbol = NSLookupAndBindSymbol(symbolName);
    free(symbolName);

    return symbol ? NSAddressOfSymbol(symbol) : NULL;
}

#endif

bool GL_1_1_supported = false;
bool GL_1_2_supported = false;
bool GL_1_3_supported = false;
bool GL_1_4_supported = false;
bool GL_1_5_supported = false;
bool GL_2_0_supported = false;

int GLMajorVersion   = 1;
int GLMinorVersion   = 0;
int GLReleaseVersion = 0;




// GL_ARB_draw_buffers
#ifdef GL_ARB_draw_buffers_PROTOTYPES
PFNGLDRAWBUFFERSARBPROC glDrawBuffersARB = NULL;
#endif

// GL_ARB_multisample
#ifdef GL_ARB_multisample_PROTOTYPES
PFNGLSAMPLECOVERAGEARBPROC glSampleCoverageARB = NULL;
#endif

// GL_ARB_multitexture
#ifdef GL_ARB_multitexture_PROTOTYPES
PFNGLACTIVETEXTUREARBPROC glActiveTextureARB = NULL;
PFNGLCLIENTACTIVETEXTUREARBPROC glClientActiveTextureARB = NULL;
PFNGLMULTITEXCOORD1DARBPROC  glMultiTexCoord1dARB  = NULL;
PFNGLMULTITEXCOORD1DVARBPROC glMultiTexCoord1dvARB = NULL;
PFNGLMULTITEXCOORD1FARBPROC  glMultiTexCoord1fARB  = NULL;
PFNGLMULTITEXCOORD1FVARBPROC glMultiTexCoord1fvARB = NULL;
PFNGLMULTITEXCOORD1IARBPROC  glMultiTexCoord1iARB  = NULL;
PFNGLMULTITEXCOORD1IVARBPROC glMultiTexCoord1ivARB = NULL;
PFNGLMULTITEXCOORD1SARBPROC  glMultiTexCoord1sARB  = NULL;
PFNGLMULTITEXCOORD1SVARBPROC glMultiTexCoord1svARB = NULL;
PFNGLMULTITEXCOORD2DARBPROC  glMultiTexCoord2dARB  = NULL;
PFNGLMULTITEXCOORD2DVARBPROC glMultiTexCoord2dvARB = NULL;
PFNGLMULTITEXCOORD2FARBPROC  glMultiTexCoord2fARB  = NULL;
PFNGLMULTITEXCOORD2FVARBPROC glMultiTexCoord2fvARB = NULL;
PFNGLMULTITEXCOORD2IARBPROC  glMultiTexCoord2iARB  = NULL;
PFNGLMULTITEXCOORD2IVARBPROC glMultiTexCoord2ivARB = NULL;
PFNGLMULTITEXCOORD2SARBPROC  glMultiTexCoord2sARB  = NULL;
PFNGLMULTITEXCOORD2SVARBPROC glMultiTexCoord2svARB = NULL;
PFNGLMULTITEXCOORD3DARBPROC  glMultiTexCoord3dARB  = NULL;
PFNGLMULTITEXCOORD3DVARBPROC glMultiTexCoord3dvARB = NULL;
PFNGLMULTITEXCOORD3FARBPROC  glMultiTexCoord3fARB  = NULL;
PFNGLMULTITEXCOORD3FVARBPROC glMultiTexCoord3fvARB = NULL;
PFNGLMULTITEXCOORD3IARBPROC  glMultiTexCoord3iARB  = NULL;
PFNGLMULTITEXCOORD3IVARBPROC glMultiTexCoord3ivARB = NULL;
PFNGLMULTITEXCOORD3SARBPROC  glMultiTexCoord3sARB  = NULL;
PFNGLMULTITEXCOORD3SVARBPROC glMultiTexCoord3svARB = NULL;
PFNGLMULTITEXCOORD4DARBPROC  glMultiTexCoord4dARB  = NULL;
PFNGLMULTITEXCOORD4DVARBPROC glMultiTexCoord4dvARB = NULL;
PFNGLMULTITEXCOORD4FARBPROC  glMultiTexCoord4fARB  = NULL;
PFNGLMULTITEXCOORD4FVARBPROC glMultiTexCoord4fvARB = NULL;
PFNGLMULTITEXCOORD4IARBPROC  glMultiTexCoord4iARB  = NULL;
PFNGLMULTITEXCOORD4IVARBPROC glMultiTexCoord4ivARB = NULL;
PFNGLMULTITEXCOORD4SARBPROC  glMultiTexCoord4sARB  = NULL;
PFNGLMULTITEXCOORD4SVARBPROC glMultiTexCoord4svARB = NULL;
#endif

// GL_ARB_occlusion_query
#ifdef GL_ARB_occlusion_query_PROTOTYPES
PFNGLGENQUERIESARBPROC glGenQueriesARB = NULL;
PFNGLDELETEQUERIESARBPROC glDeleteQueriesARB = NULL;
PFNGLISQUERYARBPROC glIsQueryARB = NULL;
PFNGLBEGINQUERYARBPROC glBeginQueryARB = NULL;
PFNGLENDQUERYARBPROC glEndQueryARB = NULL;
PFNGLGETQUERYIVARBPROC glGetQueryivARB = NULL;
PFNGLGETQUERYOBJECTIVARBPROC glGetQueryObjectivARB = NULL;
PFNGLGETQUERYOBJECTUIVARBPROC glGetQueryObjectuivARB = NULL;
#endif

// GL_ARB_point_parameters
#ifdef GL_ARB_point_parameters_PROTOTYPES
PFNGLPOINTPARAMETERFARBPROC  glPointParameterfARB = NULL;
PFNGLPOINTPARAMETERFVARBPROC glPointParameterfvARB = NULL;
#endif

// GL_ARB_shader_objects
#ifdef GL_ARB_shader_objects_PROTOTYPES
PFNGLDELETEOBJECTARBPROC glDeleteObjectARB = NULL;
PFNGLGETHANDLEARBPROC glGetHandleARB = NULL;
PFNGLDETACHOBJECTARBPROC glDetachObjectARB = NULL;
PFNGLCREATESHADEROBJECTARBPROC glCreateShaderObjectARB = NULL;
PFNGLSHADERSOURCEARBPROC glShaderSourceARB = NULL;
PFNGLCOMPILESHADERARBPROC glCompileShaderARB = NULL;
PFNGLCREATEPROGRAMOBJECTARBPROC glCreateProgramObjectARB = NULL;
PFNGLATTACHOBJECTARBPROC glAttachObjectARB = NULL;
PFNGLLINKPROGRAMARBPROC glLinkProgramARB = NULL;
PFNGLUSEPROGRAMOBJECTARBPROC glUseProgramObjectARB = NULL;
PFNGLVALIDATEPROGRAMARBPROC glValidateProgramARB = NULL;

PFNGLUNIFORM1FARBPROC glUniform1fARB = NULL;
PFNGLUNIFORM2FARBPROC glUniform2fARB = NULL;
PFNGLUNIFORM3FARBPROC glUniform3fARB = NULL;
PFNGLUNIFORM4FARBPROC glUniform4fARB = NULL;
PFNGLUNIFORM1IARBPROC glUniform1iARB = NULL;
PFNGLUNIFORM2IARBPROC glUniform2iARB = NULL;
PFNGLUNIFORM3IARBPROC glUniform3iARB = NULL;
PFNGLUNIFORM4IARBPROC glUniform4iARB = NULL;
PFNGLUNIFORM1FVARBPROC glUniform1fvARB = NULL;
PFNGLUNIFORM2FVARBPROC glUniform2fvARB = NULL;
PFNGLUNIFORM3FVARBPROC glUniform3fvARB = NULL;
PFNGLUNIFORM4FVARBPROC glUniform4fvARB = NULL;
PFNGLUNIFORM1IVARBPROC glUniform1ivARB = NULL;
PFNGLUNIFORM2IVARBPROC glUniform2ivARB = NULL;
PFNGLUNIFORM3IVARBPROC glUniform3ivARB = NULL;
PFNGLUNIFORM4IVARBPROC glUniform4ivARB = NULL;

PFNGLUNIFORMMATRIX2FVARBPROC glUniformMatrix2fvARB = NULL;
PFNGLUNIFORMMATRIX3FVARBPROC glUniformMatrix3fvARB = NULL;
PFNGLUNIFORMMATRIX4FVARBPROC glUniformMatrix4fvARB = NULL;

PFNGLGETOBJECTPARAMETERFVARBPROC glGetObjectParameterfvARB = NULL;
PFNGLGETOBJECTPARAMETERIVARBPROC glGetObjectParameterivARB = NULL;
PFNGLGETINFOLOGARBPROC glGetInfoLogARB = NULL;
PFNGLGETATTACHEDOBJECTSARBPROC glGetAttachedObjectsARB = NULL;
PFNGLGETUNIFORMLOCATIONARBPROC glGetUniformLocationARB = NULL;
PFNGLGETACTIVEUNIFORMARBPROC glGetActiveUniformARB = NULL;
PFNGLGETUNIFORMFVARBPROC glGetUniformfvARB = NULL;
PFNGLGETUNIFORMIVARBPROC glGetUniformivARB = NULL;
PFNGLGETSHADERSOURCEARBPROC glGetShaderSourceARB = NULL;
#endif

// GL_ARB_texture_compression
#ifdef GL_ARB_texture_compression_PROTOTYPES
PFNGLCOMPRESSEDTEXIMAGE1DARBPROC    glCompressedTexImage1DARB    = NULL;
PFNGLCOMPRESSEDTEXIMAGE2DARBPROC    glCompressedTexImage2DARB    = NULL;
PFNGLCOMPRESSEDTEXIMAGE3DARBPROC    glCompressedTexImage3DARB    = NULL;
PFNGLCOMPRESSEDTEXSUBIMAGE1DARBPROC glCompressedTexSubImage1DARB = NULL;
PFNGLCOMPRESSEDTEXSUBIMAGE2DARBPROC glCompressedTexSubImage2DARB = NULL;
PFNGLCOMPRESSEDTEXSUBIMAGE3DARBPROC glCompressedTexSubImage3DARB = NULL;
PFNGLGETCOMPRESSEDTEXIMAGEARBPROC   glGetCompressedTexImageARB   = NULL;
#endif

// GL_ARB_transpose_matrix
#ifdef GL_ARB_transpose_matrix_PROTOTYPES
PFNGLLOADTRANSPOSEMATRIXFARBPROC glLoadTransposeMatrixfARB = NULL;
PFNGLLOADTRANSPOSEMATRIXDARBPROC glLoadTransposeMatrixdARB = NULL;
PFNGLMULTTRANSPOSEMATRIXFARBPROC glMultTransposeMatrixfARB = NULL;
PFNGLMULTTRANSPOSEMATRIXDARBPROC glMultTransposeMatrixdARB = NULL;
#endif

// GL_ARB_vertex_buffer_object
#ifdef GL_ARB_vertex_buffer_object_PROTOTYPES
PFNGLBINDBUFFERARBPROC glBindBufferARB;
PFNGLDELETEBUFFERSARBPROC glDeleteBuffersARB;
PFNGLGENBUFFERSARBPROC glGenBuffersARB;
PFNGLISBUFFERARBPROC glIsBufferARB;

PFNGLBUFFERDATAARBPROC glBufferDataARB;
PFNGLBUFFERSUBDATAARBPROC glBufferSubDataARB;
PFNGLGETBUFFERSUBDATAARBPROC glGetBufferSubDataARB;

PFNGLMAPBUFFERARBPROC glMapBufferARB;
PFNGLUNMAPBUFFERARBPROC glUnmapBufferARB;

PFNGLGETBUFFERPARAMETERIVARBPROC glGetBufferParameterivARB;
PFNGLGETBUFFERPOINTERVARBPROC glGetBufferPointervARB;
#endif


// GL_ARB_vertex_program
#ifdef GL_ARB_vertex_program_PROTOTYPES
PFNGLVERTEXATTRIB1SARBPROC glVertexAttrib1sARB = NULL;
PFNGLVERTEXATTRIB1FARBPROC glVertexAttrib1fARB = NULL;
PFNGLVERTEXATTRIB1DARBPROC glVertexAttrib1dARB = NULL;
PFNGLVERTEXATTRIB2SARBPROC glVertexAttrib2sARB = NULL;
PFNGLVERTEXATTRIB2FARBPROC glVertexAttrib2fARB = NULL;
PFNGLVERTEXATTRIB2DARBPROC glVertexAttrib2dARB = NULL;
PFNGLVERTEXATTRIB3SARBPROC glVertexAttrib3sARB = NULL;
PFNGLVERTEXATTRIB3FARBPROC glVertexAttrib3fARB = NULL;
PFNGLVERTEXATTRIB3DARBPROC glVertexAttrib3dARB = NULL;
PFNGLVERTEXATTRIB4SARBPROC glVertexAttrib4sARB = NULL;
PFNGLVERTEXATTRIB4FARBPROC glVertexAttrib4fARB = NULL;
PFNGLVERTEXATTRIB4DARBPROC glVertexAttrib4dARB = NULL;
PFNGLVERTEXATTRIB4NUBARBPROC glVertexAttrib4NubARB = NULL;
PFNGLVERTEXATTRIB1SVARBPROC glVertexAttrib1svARB = NULL;
PFNGLVERTEXATTRIB1FVARBPROC glVertexAttrib1fvARB = NULL;
PFNGLVERTEXATTRIB1DVARBPROC glVertexAttrib1dvARB = NULL;
PFNGLVERTEXATTRIB2SVARBPROC glVertexAttrib2svARB = NULL;
PFNGLVERTEXATTRIB2FVARBPROC glVertexAttrib2fvARB = NULL;
PFNGLVERTEXATTRIB2DVARBPROC glVertexAttrib2dvARB = NULL;
PFNGLVERTEXATTRIB3SVARBPROC glVertexAttrib3svARB = NULL;
PFNGLVERTEXATTRIB3FVARBPROC glVertexAttrib3fvARB = NULL;
PFNGLVERTEXATTRIB3DVARBPROC glVertexAttrib3dvARB = NULL;
PFNGLVERTEXATTRIB4BVARBPROC glVertexAttrib4bvARB = NULL;
PFNGLVERTEXATTRIB4SVARBPROC glVertexAttrib4svARB = NULL;
PFNGLVERTEXATTRIB4IVARBPROC glVertexAttrib4ivARB = NULL;
PFNGLVERTEXATTRIB4UBVARBPROC glVertexAttrib4ubvARB = NULL;
PFNGLVERTEXATTRIB4USVARBPROC glVertexAttrib4usvARB = NULL;
PFNGLVERTEXATTRIB4UIVARBPROC glVertexAttrib4uivARB = NULL;
PFNGLVERTEXATTRIB4FVARBPROC glVertexAttrib4fvARB = NULL;
PFNGLVERTEXATTRIB4DVARBPROC glVertexAttrib4dvARB = NULL;
PFNGLVERTEXATTRIB4NBVARBPROC glVertexAttrib4NbvARB = NULL;
PFNGLVERTEXATTRIB4NSVARBPROC glVertexAttrib4NsvARB = NULL;
PFNGLVERTEXATTRIB4NIVARBPROC glVertexAttrib4NivARB = NULL;
PFNGLVERTEXATTRIB4NUBVARBPROC glVertexAttrib4NubvARB = NULL;
PFNGLVERTEXATTRIB4NUSVARBPROC glVertexAttrib4NusvARB = NULL;
PFNGLVERTEXATTRIB4NUIVARBPROC glVertexAttrib4NuivARB = NULL;
PFNGLVERTEXATTRIBPOINTERARBPROC glVertexAttribPointerARB = NULL;
PFNGLENABLEVERTEXATTRIBARRAYARBPROC glEnableVertexAttribArrayARB = NULL;
PFNGLDISABLEVERTEXATTRIBARRAYARBPROC glDisableVertexAttribArrayARB = NULL;
PFNGLPROGRAMSTRINGARBPROC glProgramStringARB = NULL;
PFNGLBINDPROGRAMARBPROC glBindProgramARB = NULL;
PFNGLDELETEPROGRAMSARBPROC glDeleteProgramsARB = NULL;
PFNGLGENPROGRAMSARBPROC glGenProgramsARB = NULL;
PFNGLPROGRAMENVPARAMETER4DARBPROC glProgramEnvParameter4dARB = NULL;
PFNGLPROGRAMENVPARAMETER4DVARBPROC glProgramEnvParameter4dvARB = NULL;
PFNGLPROGRAMENVPARAMETER4FARBPROC glProgramEnvParameter4fARB = NULL;
PFNGLPROGRAMENVPARAMETER4FVARBPROC glProgramEnvParameter4fvARB = NULL;
PFNGLPROGRAMLOCALPARAMETER4DARBPROC glProgramLocalParameter4dARB = NULL;
PFNGLPROGRAMLOCALPARAMETER4DVARBPROC glProgramLocalParameter4dvARB = NULL;
PFNGLPROGRAMLOCALPARAMETER4FARBPROC glProgramLocalParameter4fARB = NULL;
PFNGLPROGRAMLOCALPARAMETER4FVARBPROC glProgramLocalParameter4fvARB = NULL;
PFNGLGETPROGRAMENVPARAMETERDVARBPROC glGetProgramEnvParameterdvARB = NULL;
PFNGLGETPROGRAMENVPARAMETERFVARBPROC glGetProgramEnvParameterfvARB = NULL;
PFNGLGETPROGRAMLOCALPARAMETERDVARBPROC glGetProgramLocalParameterdvARB = NULL;
PFNGLGETPROGRAMLOCALPARAMETERFVARBPROC glGetProgramLocalParameterfvARB = NULL;
PFNGLGETPROGRAMIVARBPROC glGetProgramivARB = NULL;
PFNGLGETPROGRAMSTRINGARBPROC glGetProgramStringARB = NULL;
PFNGLGETVERTEXATTRIBDVARBPROC glGetVertexAttribdvARB = NULL;
PFNGLGETVERTEXATTRIBFVARBPROC glGetVertexAttribfvARB = NULL;
PFNGLGETVERTEXATTRIBIVARBPROC glGetVertexAttribivARB = NULL;
PFNGLGETVERTEXATTRIBPOINTERVARBPROC glGetVertexAttribPointervARB = NULL;
PFNGLISPROGRAMARBPROC glIsProgramARB = NULL;
#endif

// GL_ARB_vertex_shader
#ifdef GL_ARB_vertex_shader_PROTOTYPES
PFNGLBINDATTRIBLOCATIONARBPROC glBindAttribLocationARB = NULL;
PFNGLGETACTIVEATTRIBARBPROC    glGetActiveAttribARB = NULL;
PFNGLGETATTRIBLOCATIONARBPROC  glGetAttribLocationARB = NULL;
#endif

// GL_ARB_window_pos
#ifdef GL_ARB_window_pos_PROTOTYPES
PFNGLWINDOWPOS2DARBPROC  glWindowPos2dARB  = NULL;
PFNGLWINDOWPOS2FARBPROC  glWindowPos2fARB  = NULL;
PFNGLWINDOWPOS2IARBPROC  glWindowPos2iARB  = NULL;
PFNGLWINDOWPOS2SARBPROC  glWindowPos2sARB  = NULL;
PFNGLWINDOWPOS2IVARBPROC glWindowPos2ivARB = NULL;
PFNGLWINDOWPOS2SVARBPROC glWindowPos2svARB = NULL;
PFNGLWINDOWPOS2FVARBPROC glWindowPos2fvARB = NULL;
PFNGLWINDOWPOS2DVARBPROC glWindowPos2dvARB = NULL;
PFNGLWINDOWPOS3IARBPROC  glWindowPos3iARB  = NULL;
PFNGLWINDOWPOS3SARBPROC  glWindowPos3sARB  = NULL;
PFNGLWINDOWPOS3FARBPROC  glWindowPos3fARB  = NULL;
PFNGLWINDOWPOS3DARBPROC  glWindowPos3dARB  = NULL;
PFNGLWINDOWPOS3IVARBPROC glWindowPos3ivARB = NULL;
PFNGLWINDOWPOS3SVARBPROC glWindowPos3svARB = NULL;
PFNGLWINDOWPOS3FVARBPROC glWindowPos3fvARB = NULL;
PFNGLWINDOWPOS3DVARBPROC glWindowPos3dvARB = NULL;
#endif

// GL_ATI_fragment_shader
#ifdef GL_ATI_fragment_shader_PROTOTYPES
PFNGLGENFRAGMENTSHADERSATIPROC   glGenFragmentShadersATI   = NULL;
PFNGLBINDFRAGMENTSHADERATIPROC   glBindFragmentShaderATI   = NULL;
PFNGLDELETEFRAGMENTSHADERATIPROC glDeleteFragmentShaderATI = NULL;
PFNGLBEGINFRAGMENTSHADERATIPROC  glBeginFragmentShaderATI  = NULL;
PFNGLENDFRAGMENTSHADERATIPROC    glEndFragmentShaderATI    = NULL;
PFNGLPASSTEXCOORDATIPROC         glPassTexCoordATI         = NULL;
PFNGLSAMPLEMAPATIPROC            glSampleMapATI            = NULL;

PFNGLCOLORFRAGMENTOP1ATIPROC glColorFragmentOp1ATI = NULL;
PFNGLCOLORFRAGMENTOP2ATIPROC glColorFragmentOp2ATI = NULL;
PFNGLCOLORFRAGMENTOP3ATIPROC glColorFragmentOp3ATI = NULL;

PFNGLALPHAFRAGMENTOP1ATIPROC glAlphaFragmentOp1ATI = NULL;
PFNGLALPHAFRAGMENTOP2ATIPROC glAlphaFragmentOp2ATI = NULL;
PFNGLALPHAFRAGMENTOP3ATIPROC glAlphaFragmentOp3ATI = NULL;

PFNGLSETFRAGMENTSHADERCONSTANTATIPROC glSetFragmentShaderConstantATI = NULL;
#endif

// GL_ATI_separate_stencil
#ifdef GL_ATI_separate_stencil_PROTOTYPES
PFNGLSTENCILOPSEPARATEATIPROC   glStencilOpSeparateATI   = NULL;
PFNGLSTENCILFUNCSEPARATEATIPROC glStencilFuncSeparateATI = NULL;
#endif

// GL_EXT_blend_color
#ifdef GL_EXT_blend_color_PROTOTYPES
PFNGLBLENDCOLOREXTPROC glBlendColorEXT = NULL;
#endif

// GL_EXT_blend_func_separate
#ifdef GL_EXT_blend_func_separate_PROTOTYPES
PFNGLBLENDFUNCSEPARATEEXTPROC glBlendFuncSeparateEXT = NULL;
#endif

// GL_EXT_blend_minmax/GL_EXT_blend_subtract
#ifdef GL_EXT_blend_minmax_PROTOTYPES
PFNGLBLENDEQUATIONEXTPROC glBlendEquationEXT = NULL;
#endif

// GL_EXT_draw_range_elements
#ifdef GL_EXT_draw_range_elements_PROTOTYPES
PFNGLDRAWRANGEELEMENTSEXTPROC glDrawRangeElementsEXT = NULL;
#endif

// GL_EXT_fog_coord
#ifdef GL_EXT_fog_coord_PROTOTYPES
PFNGLFOGCOORDFEXTPROC  glFogCoordfEXT  = NULL;
PFNGLFOGCOORDDEXTPROC  glFogCoorddEXT  = NULL;
PFNGLFOGCOORDFVEXTPROC glFogCoordfvEXT = NULL;
PFNGLFOGCOORDDVEXTPROC glFogCoorddvEXT = NULL;
PFNGLFOGCOORDPOINTEREXTPROC glFogCoordPointerEXT = NULL;
#endif

// GL_EXT_framebuffer_object
#ifdef GL_EXT_framebuffer_object_PROTOTYPES
PFNGLISRENDERBUFFEREXTPROC             glIsRenderbufferEXT = NULL;
PFNGLBINDRENDERBUFFEREXTPROC           glBindRenderbufferEXT = NULL;
PFNGLDELETERENDERBUFFERSEXTPROC        glDeleteRenderbuffersEXT = NULL;
PFNGLGENRENDERBUFFERSEXTPROC           glGenRenderbuffersEXT = NULL;
PFNGLRENDERBUFFERSTORAGEEXTPROC        glRenderbufferStorageEXT = NULL;
PFNGLGETRENDERBUFFERPARAMETERIVEXTPROC glGetRenderbufferParameterivEXT = NULL;
PFNGLISFRAMEBUFFEREXTPROC              glIsFramebufferEXT = NULL;
PFNGLBINDFRAMEBUFFEREXTPROC            glBindFramebufferEXT = NULL;
PFNGLDELETEFRAMEBUFFERSEXTPROC         glDeleteFramebuffersEXT = NULL;
PFNGLGENFRAMEBUFFERSEXTPROC            glGenFramebuffersEXT = NULL;
PFNGLCHECKFRAMEBUFFERSTATUSEXTPROC     glCheckFramebufferStatusEXT = NULL;
PFNGLFRAMEBUFFERTEXTURE1DEXTPROC       glFramebufferTexture1DEXT = NULL;
PFNGLFRAMEBUFFERTEXTURE2DEXTPROC       glFramebufferTexture2DEXT = NULL;
PFNGLFRAMEBUFFERTEXTURE3DEXTPROC       glFramebufferTexture3DEXT = NULL;
PFNGLFRAMEBUFFERRENDERBUFFEREXTPROC    glFramebufferRenderbufferEXT = NULL;
PFNGLGETFRAMEBUFFERATTACHMENTPARAMETERIVEXTPROC glGetFramebufferAttachmentParameterivEXT = NULL;
PFNGLGENERATEMIPMAPEXTPROC             glGenerateMipmapEXT = NULL;
#endif

// GL_EXT_multi_draw_arrays
#ifdef GL_EXT_multi_draw_arrays_PROTOTYPES
PFNGLMULTIDRAWARRAYSEXTPROC   glMultiDrawArraysEXT   = NULL;
PFNGLMULTIDRAWELEMENTSEXTPROC glMultiDrawElementsEXT = NULL;
#endif

// GL_EXT_secondary_color
#ifdef GL_EXT_secondary_color_PROTOTYPES
PFNGLSECONDARYCOLOR3FEXTPROC glSecondaryColor3fEXT = NULL;
PFNGLSECONDARYCOLOR3DEXTPROC glSecondaryColor3dEXT = NULL;
PFNGLSECONDARYCOLOR3BEXTPROC glSecondaryColor3bEXT = NULL;
PFNGLSECONDARYCOLOR3SEXTPROC glSecondaryColor3sEXT = NULL;
PFNGLSECONDARYCOLOR3IEXTPROC glSecondaryColor3iEXT = NULL;
PFNGLSECONDARYCOLOR3UBEXTPROC glSecondaryColor3ubEXT = NULL;
PFNGLSECONDARYCOLOR3USEXTPROC glSecondaryColor3usEXT = NULL;
PFNGLSECONDARYCOLOR3UIEXTPROC glSecondaryColor3uiEXT = NULL;

PFNGLSECONDARYCOLOR3FVEXTPROC glSecondaryColor3fvEXT = NULL;
PFNGLSECONDARYCOLOR3DVEXTPROC glSecondaryColor3dvEXT = NULL;
PFNGLSECONDARYCOLOR3BVEXTPROC glSecondaryColor3bvEXT = NULL;
PFNGLSECONDARYCOLOR3SVEXTPROC glSecondaryColor3svEXT = NULL;
PFNGLSECONDARYCOLOR3IVEXTPROC glSecondaryColor3ivEXT = NULL;
PFNGLSECONDARYCOLOR3UBVEXTPROC glSecondaryColor3ubvEXT = NULL;
PFNGLSECONDARYCOLOR3USVEXTPROC glSecondaryColor3usvEXT = NULL;
PFNGLSECONDARYCOLOR3UIVEXTPROC glSecondaryColor3uivEXT = NULL;

PFNGLSECONDARYCOLORPOINTEREXTPROC glSecondaryColorPointerEXT = NULL;
#endif

// GL_EXT_texture3D
#ifdef GL_EXT_texture3D_PROTOTYPES
PFNGLTEXIMAGE3DEXTPROC glTexImage3DEXT = NULL;
#endif

#if defined(_WIN32)

// WGL_ARB_extensions_string
PFNWGLGETEXTENSIONSSTRINGARBPROC wglGetExtensionsStringARB = NULL;

// WGL_ARB_make_current_read
PFNWGLMAKECONTEXTCURRENTARBPROC wglMakeContextCurrentARB = NULL;
PFNWGLGETCURRENTREADDCARBPROC wglGetCurrentReadDCARB = NULL;

// WGL_ARB_pbuffer
PFNWGLCREATEPBUFFERARBPROC    wglCreatePbufferARB    = NULL;
PFNWGLGETPBUFFERDCARBPROC     wglGetPbufferDCARB     = NULL;
PFNWGLRELEASEPBUFFERDCARBPROC wglReleasePbufferDCARB = NULL;
PFNWGLDESTROYPBUFFERARBPROC   wglDestroyPbufferARB   = NULL;
PFNWGLQUERYPBUFFERARBPROC     wglQueryPbufferARB     = NULL;

// WGL_ARB_pixel_format
PFNWGLGETPIXELFORMATATTRIBIVARBPROC wglGetPixelFormatAttribivARB = NULL;
PFNWGLGETPIXELFORMATATTRIBFVARBPROC wglGetPixelFormatAttribfvARB = NULL;
PFNWGLCHOOSEPIXELFORMATARBPROC      wglChoosePixelFormatARB      = NULL;

// WGL_ARB_render_texture
PFNWGLBINDTEXIMAGEARBPROC     wglBindTexImageARB     = NULL;
PFNWGLRELEASETEXIMAGEARBPROC  wglReleaseTexImageARB  = NULL;
PFNWGLSETPBUFFERATTRIBARBPROC wglSetPbufferAttribARB = NULL;

// WGL_EXT_swap_control
PFNWGLSWAPINTERVALEXTPROC wglSwapIntervalEXT = NULL;
PFNWGLGETSWAPINTERVALEXTPROC wglGetSwapIntervalEXT = NULL;


#elif defined(LINUX)

// GLX_ATI_render_texture
PFNGLXBINDTEXIMAGEATIPROC    glXBindTexImageATI    = NULL;
PFNGLXRELEASETEXIMAGEATIPROC glXReleaseTexImageATI = NULL;
PFNGLXDRAWABLEATTRIBATIPROC  glXDrawableAttribATI  = NULL;

#elif defined(__APPLE__)

#endif







#ifdef GL_VERSION_1_2_PROTOTYPES
PFNGLTEXIMAGE3DEXTPROC glTexImage3D = NULL;
PFNGLTEXSUBIMAGE3DPROC glTexSubImage3D = NULL;
PFNGLCOPYTEXSUBIMAGE3DPROC glCopyTexSubImage3D = NULL;
PFNGLDRAWRANGEELEMENTSEXTPROC glDrawRangeElements = NULL;
PFNGLBLENDCOLOREXTPROC glBlendColor = NULL;
PFNGLBLENDEQUATIONEXTPROC glBlendEquation = NULL;
#endif // GL_VERSION_1_2_PROTOTYPES



#ifdef GL_VERSION_1_3_PROTOTYPES
PFNGLACTIVETEXTUREARBPROC glActiveTexture = NULL;
PFNGLCLIENTACTIVETEXTUREARBPROC glClientActiveTexture = NULL;
PFNGLMULTITEXCOORD1DARBPROC  glMultiTexCoord1d  = NULL;
PFNGLMULTITEXCOORD1DVARBPROC glMultiTexCoord1dv = NULL;
PFNGLMULTITEXCOORD1FARBPROC  glMultiTexCoord1f  = NULL;
PFNGLMULTITEXCOORD1FVARBPROC glMultiTexCoord1fv = NULL;
PFNGLMULTITEXCOORD1IARBPROC  glMultiTexCoord1i  = NULL;
PFNGLMULTITEXCOORD1IVARBPROC glMultiTexCoord1iv = NULL;
PFNGLMULTITEXCOORD1SARBPROC  glMultiTexCoord1s  = NULL;
PFNGLMULTITEXCOORD1SVARBPROC glMultiTexCoord1sv = NULL;
PFNGLMULTITEXCOORD2DARBPROC  glMultiTexCoord2d  = NULL;
PFNGLMULTITEXCOORD2DVARBPROC glMultiTexCoord2dv = NULL;
PFNGLMULTITEXCOORD2FARBPROC  glMultiTexCoord2f  = NULL;
PFNGLMULTITEXCOORD2FVARBPROC glMultiTexCoord2fv = NULL;
PFNGLMULTITEXCOORD2IARBPROC  glMultiTexCoord2i  = NULL;
PFNGLMULTITEXCOORD2IVARBPROC glMultiTexCoord2iv = NULL;
PFNGLMULTITEXCOORD2SARBPROC  glMultiTexCoord2s  = NULL;
PFNGLMULTITEXCOORD2SVARBPROC glMultiTexCoord2sv = NULL;
PFNGLMULTITEXCOORD3DARBPROC  glMultiTexCoord3d  = NULL;
PFNGLMULTITEXCOORD3DVARBPROC glMultiTexCoord3dv = NULL;
PFNGLMULTITEXCOORD3FARBPROC  glMultiTexCoord3f  = NULL;
PFNGLMULTITEXCOORD3FVARBPROC glMultiTexCoord3fv = NULL;
PFNGLMULTITEXCOORD3IARBPROC  glMultiTexCoord3i  = NULL;
PFNGLMULTITEXCOORD3IVARBPROC glMultiTexCoord3iv = NULL;
PFNGLMULTITEXCOORD3SARBPROC  glMultiTexCoord3s  = NULL;
PFNGLMULTITEXCOORD3SVARBPROC glMultiTexCoord3sv = NULL;
PFNGLMULTITEXCOORD4DARBPROC  glMultiTexCoord4d  = NULL;
PFNGLMULTITEXCOORD4DVARBPROC glMultiTexCoord4dv = NULL;
PFNGLMULTITEXCOORD4FARBPROC  glMultiTexCoord4f  = NULL;
PFNGLMULTITEXCOORD4FVARBPROC glMultiTexCoord4fv = NULL;
PFNGLMULTITEXCOORD4IARBPROC  glMultiTexCoord4i  = NULL;
PFNGLMULTITEXCOORD4IVARBPROC glMultiTexCoord4iv = NULL;
PFNGLMULTITEXCOORD4SARBPROC  glMultiTexCoord4s  = NULL;
PFNGLMULTITEXCOORD4SVARBPROC glMultiTexCoord4sv = NULL;

PFNGLCOMPRESSEDTEXIMAGE1DARBPROC    glCompressedTexImage1D    = NULL;
PFNGLCOMPRESSEDTEXIMAGE2DARBPROC    glCompressedTexImage2D    = NULL;
PFNGLCOMPRESSEDTEXIMAGE3DARBPROC    glCompressedTexImage3D    = NULL;
PFNGLCOMPRESSEDTEXSUBIMAGE1DARBPROC glCompressedTexSubImage1D = NULL;
PFNGLCOMPRESSEDTEXSUBIMAGE2DARBPROC glCompressedTexSubImage2D = NULL;
PFNGLCOMPRESSEDTEXSUBIMAGE3DARBPROC glCompressedTexSubImage3D = NULL;
PFNGLGETCOMPRESSEDTEXIMAGEARBPROC   glGetCompressedTexImage   = NULL;

PFNGLSAMPLECOVERAGEARBPROC glSampleCoverage = NULL;

PFNGLLOADTRANSPOSEMATRIXFARBPROC glLoadTransposeMatrixf = NULL;
PFNGLLOADTRANSPOSEMATRIXDARBPROC glLoadTransposeMatrixd = NULL;
PFNGLMULTTRANSPOSEMATRIXFARBPROC glMultTransposeMatrixf = NULL;
PFNGLMULTTRANSPOSEMATRIXDARBPROC glMultTransposeMatrixd = NULL;
#endif // GL_VERSION_1_3_PROTOTYPES


#ifdef GL_VERSION_1_4_PROTOTYPES
PFNGLFOGCOORDFEXTPROC  glFogCoordf = NULL;
PFNGLFOGCOORDDEXTPROC  glFogCoordd = NULL;
PFNGLFOGCOORDFVEXTPROC glFogCoordfv = NULL;
PFNGLFOGCOORDDVEXTPROC glFogCoorddv = NULL;
PFNGLFOGCOORDPOINTEREXTPROC glFogCoordPointer = NULL;

PFNGLMULTIDRAWARRAYSEXTPROC   glMultiDrawArrays = NULL;
PFNGLMULTIDRAWELEMENTSEXTPROC glMultiDrawElements = NULL;

PFNGLPOINTPARAMETERFARBPROC  glPointParameterf = NULL;
PFNGLPOINTPARAMETERFVARBPROC glPointParameterfv = NULL;

PFNGLSECONDARYCOLOR3FEXTPROC glSecondaryColor3f = NULL;
PFNGLSECONDARYCOLOR3DEXTPROC glSecondaryColor3d = NULL;
PFNGLSECONDARYCOLOR3BEXTPROC glSecondaryColor3b = NULL;
PFNGLSECONDARYCOLOR3SEXTPROC glSecondaryColor3s = NULL;
PFNGLSECONDARYCOLOR3IEXTPROC glSecondaryColor3i = NULL;
PFNGLSECONDARYCOLOR3UBEXTPROC glSecondaryColor3ub = NULL;
PFNGLSECONDARYCOLOR3USEXTPROC glSecondaryColor3us = NULL;
PFNGLSECONDARYCOLOR3UIEXTPROC glSecondaryColor3ui = NULL;

PFNGLSECONDARYCOLOR3FVEXTPROC glSecondaryColor3fv = NULL;
PFNGLSECONDARYCOLOR3DVEXTPROC glSecondaryColor3dv = NULL;
PFNGLSECONDARYCOLOR3BVEXTPROC glSecondaryColor3bv = NULL;
PFNGLSECONDARYCOLOR3SVEXTPROC glSecondaryColor3sv = NULL;
PFNGLSECONDARYCOLOR3IVEXTPROC glSecondaryColor3iv = NULL;
PFNGLSECONDARYCOLOR3UBVEXTPROC glSecondaryColor3ubv = NULL;
PFNGLSECONDARYCOLOR3USVEXTPROC glSecondaryColor3usv = NULL;
PFNGLSECONDARYCOLOR3UIVEXTPROC glSecondaryColor3uiv = NULL;

PFNGLSECONDARYCOLORPOINTEREXTPROC glSecondaryColorPointer = NULL;

PFNGLBLENDFUNCSEPARATEEXTPROC glBlendFuncSeparate = NULL;

PFNGLWINDOWPOS2DARBPROC  glWindowPos2d = NULL;
PFNGLWINDOWPOS2FARBPROC  glWindowPos2f = NULL;
PFNGLWINDOWPOS2IARBPROC  glWindowPos2i = NULL;
PFNGLWINDOWPOS2SARBPROC  glWindowPos2s = NULL;
PFNGLWINDOWPOS2IVARBPROC glWindowPos2iv = NULL;
PFNGLWINDOWPOS2SVARBPROC glWindowPos2sv = NULL;
PFNGLWINDOWPOS2FVARBPROC glWindowPos2fv = NULL;
PFNGLWINDOWPOS2DVARBPROC glWindowPos2dv = NULL;
PFNGLWINDOWPOS3IARBPROC  glWindowPos3i = NULL;
PFNGLWINDOWPOS3SARBPROC  glWindowPos3s = NULL;
PFNGLWINDOWPOS3FARBPROC  glWindowPos3f = NULL;
PFNGLWINDOWPOS3DARBPROC  glWindowPos3d = NULL;
PFNGLWINDOWPOS3IVARBPROC glWindowPos3iv = NULL;
PFNGLWINDOWPOS3SVARBPROC glWindowPos3sv = NULL;
PFNGLWINDOWPOS3FVARBPROC glWindowPos3fv = NULL;
PFNGLWINDOWPOS3DVARBPROC glWindowPos3dv = NULL;
#endif // GL_VERSION_1_4_PROTOTYPES


#ifdef GL_VERSION_1_5_PROTOTYPES
PFNGLBINDBUFFERARBPROC    glBindBuffer = NULL;
PFNGLDELETEBUFFERSARBPROC glDeleteBuffers = NULL;
PFNGLGENBUFFERSARBPROC    glGenBuffers = NULL;
PFNGLISBUFFERARBPROC      glIsBuffer = NULL;
PFNGLBUFFERDATAARBPROC    glBufferData = NULL;
PFNGLBUFFERSUBDATAARBPROC glBufferSubData = NULL;
PFNGLGETBUFFERSUBDATAARBPROC glGetBufferSubData = NULL;
PFNGLMAPBUFFERARBPROC   glMapBuffer = NULL;
PFNGLUNMAPBUFFERARBPROC glUnmapBuffer = NULL;
PFNGLGETBUFFERPARAMETERIVARBPROC glGetBufferParameteriv = NULL;
PFNGLGETBUFFERPOINTERVARBPROC    glGetBufferPointerv = NULL;

PFNGLGENQUERIESARBPROC    glGenQueries = NULL;
PFNGLDELETEQUERIESARBPROC glDeleteQueries = NULL;
PFNGLISQUERYARBPROC    glIsQuery = NULL;
PFNGLBEGINQUERYARBPROC glBeginQuery = NULL;
PFNGLENDQUERYARBPROC   glEndQuery = NULL;
PFNGLGETQUERYIVARBPROC glGetQueryiv = NULL;
PFNGLGETQUERYOBJECTIVARBPROC  glGetQueryObjectiv = NULL;
PFNGLGETQUERYOBJECTUIVARBPROC glGetQueryObjectuiv = NULL;
#endif // GL_VERSION_1_5_PROTOTYPES


#ifdef GL_VERSION_2_0_PROTOTYPES
PFNGLDELETEOBJECTARBPROC glDeleteProgram = NULL;
PFNGLDELETEOBJECTARBPROC glDeleteShader  = NULL;
//PFNGLGETHANDLEARBPROC glGetHandle = NULL;
PFNGLDETACHOBJECTARBPROC glDetachShader = NULL;
PFNGLCREATESHADEROBJECTARBPROC glCreateShader = NULL;
PFNGLSHADERSOURCEARBPROC glShaderSource = NULL;
PFNGLCOMPILESHADERARBPROC glCompileShader = NULL;
PFNGLCREATEPROGRAMOBJECTARBPROC glCreateProgram = NULL;
PFNGLATTACHOBJECTARBPROC glAttachShader = NULL;
PFNGLLINKPROGRAMARBPROC glLinkProgram = NULL;
PFNGLUSEPROGRAMOBJECTARBPROC glUseProgram = NULL;
PFNGLVALIDATEPROGRAMARBPROC glValidateProgram = NULL;
PFNGLUNIFORM1FARBPROC glUniform1f = NULL;
PFNGLUNIFORM2FARBPROC glUniform2f = NULL;
PFNGLUNIFORM3FARBPROC glUniform3f = NULL;
PFNGLUNIFORM4FARBPROC glUniform4f = NULL;
PFNGLUNIFORM1IARBPROC glUniform1i = NULL;
PFNGLUNIFORM2IARBPROC glUniform2i = NULL;
PFNGLUNIFORM3IARBPROC glUniform3i = NULL;
PFNGLUNIFORM4IARBPROC glUniform4i = NULL;
PFNGLUNIFORM1FVARBPROC glUniform1fv = NULL;
PFNGLUNIFORM2FVARBPROC glUniform2fv = NULL;
PFNGLUNIFORM3FVARBPROC glUniform3fv = NULL;
PFNGLUNIFORM4FVARBPROC glUniform4fv = NULL;
PFNGLUNIFORM1IVARBPROC glUniform1iv = NULL;
PFNGLUNIFORM2IVARBPROC glUniform2iv = NULL;
PFNGLUNIFORM3IVARBPROC glUniform3iv = NULL;
PFNGLUNIFORM4IVARBPROC glUniform4iv = NULL;
PFNGLUNIFORMMATRIX2FVARBPROC glUniformMatrix2fv = NULL;
PFNGLUNIFORMMATRIX3FVARBPROC glUniformMatrix3fv = NULL;
PFNGLUNIFORMMATRIX4FVARBPROC glUniformMatrix4fv = NULL;
//PFNGLGETOBJECTPARAMETERFVARBPROC glGetObjectParameterfv = NULL;
//PFNGLGETOBJECTPARAMETERIVARBPROC glGetObjectParameteriv = NULL;
//PFNGLGETINFOLOGARBPROC glGetInfoLog = NULL;
//PFNGLGETATTACHEDOBJECTSARBPROC glGetAttachedObjects = NULL;
PFNGLGETUNIFORMLOCATIONARBPROC glGetUniformLocation = NULL;
PFNGLGETACTIVEUNIFORMARBPROC glGetActiveUniform = NULL;
PFNGLGETUNIFORMFVARBPROC glGetUniformfv = NULL;
PFNGLGETUNIFORMIVARBPROC glGetUniformiv = NULL;
PFNGLGETSHADERSOURCEARBPROC glGetShaderSource = NULL;

PFNGLBINDATTRIBLOCATIONARBPROC glBindAttribLocation = NULL;
PFNGLGETACTIVEATTRIBARBPROC    glGetActiveAttrib = NULL;
PFNGLGETATTRIBLOCATIONARBPROC  glGetAttribLocation = NULL;
PFNGLVERTEXATTRIB1SARBPROC glVertexAttrib1s = NULL;
PFNGLVERTEXATTRIB1FARBPROC glVertexAttrib1f = NULL;
PFNGLVERTEXATTRIB1DARBPROC glVertexAttrib1d = NULL;
PFNGLVERTEXATTRIB2SARBPROC glVertexAttrib2s = NULL;
PFNGLVERTEXATTRIB2FARBPROC glVertexAttrib2f = NULL;
PFNGLVERTEXATTRIB2DARBPROC glVertexAttrib2d = NULL;
PFNGLVERTEXATTRIB3SARBPROC glVertexAttrib3s = NULL;
PFNGLVERTEXATTRIB3FARBPROC glVertexAttrib3f = NULL;
PFNGLVERTEXATTRIB3DARBPROC glVertexAttrib3d = NULL;
PFNGLVERTEXATTRIB4SARBPROC glVertexAttrib4s = NULL;
PFNGLVERTEXATTRIB4FARBPROC glVertexAttrib4f = NULL;
PFNGLVERTEXATTRIB4DARBPROC glVertexAttrib4d = NULL;
PFNGLVERTEXATTRIB4NUBARBPROC glVertexAttrib4Nub = NULL;
PFNGLVERTEXATTRIB1SVARBPROC glVertexAttrib1sv = NULL;
PFNGLVERTEXATTRIB1FVARBPROC glVertexAttrib1fv = NULL;
PFNGLVERTEXATTRIB1DVARBPROC glVertexAttrib1dv = NULL;
PFNGLVERTEXATTRIB2SVARBPROC glVertexAttrib2sv = NULL;
PFNGLVERTEXATTRIB2FVARBPROC glVertexAttrib2fv = NULL;
PFNGLVERTEXATTRIB2DVARBPROC glVertexAttrib2dv = NULL;
PFNGLVERTEXATTRIB3SVARBPROC glVertexAttrib3sv = NULL;
PFNGLVERTEXATTRIB3FVARBPROC glVertexAttrib3fv = NULL;
PFNGLVERTEXATTRIB3DVARBPROC glVertexAttrib3dv = NULL;
PFNGLVERTEXATTRIB4BVARBPROC glVertexAttrib4bv = NULL;
PFNGLVERTEXATTRIB4SVARBPROC glVertexAttrib4sv = NULL;
PFNGLVERTEXATTRIB4IVARBPROC glVertexAttrib4iv = NULL;
PFNGLVERTEXATTRIB4UBVARBPROC glVertexAttrib4ubv = NULL;
PFNGLVERTEXATTRIB4USVARBPROC glVertexAttrib4usv = NULL;
PFNGLVERTEXATTRIB4UIVARBPROC glVertexAttrib4uiv = NULL;
PFNGLVERTEXATTRIB4FVARBPROC glVertexAttrib4fv = NULL;
PFNGLVERTEXATTRIB4DVARBPROC glVertexAttrib4dv = NULL;
PFNGLVERTEXATTRIB4NBVARBPROC glVertexAttrib4Nbv = NULL;
PFNGLVERTEXATTRIB4NSVARBPROC glVertexAttrib4Nsv = NULL;
PFNGLVERTEXATTRIB4NIVARBPROC glVertexAttrib4Niv = NULL;
PFNGLVERTEXATTRIB4NUBVARBPROC glVertexAttrib4Nubv = NULL;
PFNGLVERTEXATTRIB4NUSVARBPROC glVertexAttrib4Nusv = NULL;
PFNGLVERTEXATTRIB4NUIVARBPROC glVertexAttrib4Nuiv = NULL;
PFNGLVERTEXATTRIBPOINTERARBPROC glVertexAttribPointer = NULL;
PFNGLENABLEVERTEXATTRIBARRAYARBPROC glEnableVertexAttribArray = NULL;
PFNGLDISABLEVERTEXATTRIBARRAYARBPROC glDisableVertexAttribArray = NULL;
PFNGLGETVERTEXATTRIBDVARBPROC glGetVertexAttribdv = NULL;
PFNGLGETVERTEXATTRIBFVARBPROC glGetVertexAttribfv = NULL;
PFNGLGETVERTEXATTRIBIVARBPROC glGetVertexAttribiv = NULL;
PFNGLGETVERTEXATTRIBPOINTERVARBPROC glGetVertexAttribPointerv = NULL;

PFNGLDRAWBUFFERSARBPROC glDrawBuffers = NULL;

PFNGLSTENCILOPSEPARATEPROC   glStencilOpSeparate   = NULL;
PFNGLSTENCILFUNCSEPARATEPROC glStencilFuncSeparate = NULL;
PFNGLSTENCILMASKSEPARATEPROC glStencilMaskSeparate = NULL;

PFNGLBLENDEQUATIONSEPARATEPROC glBlendEquationSeparate = NULL;

#endif // GL_VERSION_2_0_PROTOTYPES


bool extensionSupported(char *extStr, const char *extension){
	if (extStr){
		size_t len = strlen(extension);
		while ((extStr = strstr(extStr, extension)) != NULL){
			extStr += len;
			if (*extStr == ' ' || *extStr == '\0') return true;
		}
	}
	return false;
}

#define isExtensionSupported(str) extensionSupported(extensions, str)

#if defined(_WIN32)

#define isWGLXExtensionSupported(str) (extensionSupported(wglxExtensions, str) || extensionSupported(extensions, str))

void initExtensions(HDC hdc){
	char *wglxExtensions = NULL;

	if (((wglGetExtensionsStringARB = (PFNWGLGETEXTENSIONSSTRINGARBPROC) wglxGetProcAddress("wglGetExtensionsStringARB")) != NULL) ||
		((wglGetExtensionsStringARB = (PFNWGLGETEXTENSIONSSTRINGARBPROC) wglxGetProcAddress("wglGetExtensionsStringEXT")) != NULL)){
		WGL_ARB_extensions_string_supported = true;
		wglxExtensions = (char *) wglGetExtensionsStringARB(hdc);
	}

#elif defined(LINUX)

#define isWGLXExtensionSupported(str) extensionSupported(wglxExtensions, str)

#include <stdio.h>

void initExtensions(Display *display){
	char *wglxExtensions = (char *) glXGetClientString(display, GLX_EXTENSIONS);

#elif defined(__APPLE__)

#define isWGLXExtensionSupported(str) false

void initExtensions(){

#endif

	char *extensions = (char *) glGetString(GL_EXTENSIONS);

	GL_ARB_depth_texture_supported = isExtensionSupported("GL_ARB_depth_texture");

	if (GL_ARB_draw_buffers_supported = isExtensionSupported("GL_ARB_draw_buffers")){
#ifdef GL_ARB_draw_buffers_PROTOTYPES
		glDrawBuffersARB = (PFNGLDRAWBUFFERSARBPROC) wglxGetProcAddress("glDrawBuffersARB");
#endif
	}

	GL_ARB_fragment_program_supported = isExtensionSupported("GL_ARB_fragment_program");
	GL_ARB_fragment_shader_supported  = isExtensionSupported("GL_ARB_fragment_shader");

	if (GL_ARB_multisample_supported = isExtensionSupported("GL_ARB_multisample")){
#ifdef GL_ARB_multisample_PROTOTYPES
		glSampleCoverageARB = (PFNGLSAMPLECOVERAGEARBPROC) wglxGetProcAddress("glSampleCoverageARB");
#endif
	}

	if (GL_ARB_multitexture_supported = isExtensionSupported("GL_ARB_multitexture")){
#ifdef GL_ARB_multitexture_PROTOTYPES
		glActiveTextureARB = (PFNGLACTIVETEXTUREARBPROC) wglxGetProcAddress("glActiveTextureARB");
		glClientActiveTextureARB = (PFNGLCLIENTACTIVETEXTUREARBPROC) wglxGetProcAddress("glClientActiveTextureARB");
		glMultiTexCoord1dARB  = (PFNGLMULTITEXCOORD1DARBPROC)  wglxGetProcAddress("glMultiTexCoord1dARB");
		glMultiTexCoord1dvARB = (PFNGLMULTITEXCOORD1DVARBPROC) wglxGetProcAddress("glMultiTexCoord1dvARB");
		glMultiTexCoord1fARB  = (PFNGLMULTITEXCOORD1FARBPROC)  wglxGetProcAddress("glMultiTexCoord1fARB");
		glMultiTexCoord1fvARB = (PFNGLMULTITEXCOORD1FVARBPROC) wglxGetProcAddress("glMultiTexCoord1fvARB");
		glMultiTexCoord1iARB  = (PFNGLMULTITEXCOORD1IARBPROC)  wglxGetProcAddress("glMultiTexCoord1iARB");
		glMultiTexCoord1ivARB = (PFNGLMULTITEXCOORD1IVARBPROC) wglxGetProcAddress("glMultiTexCoord1ivARB");
		glMultiTexCoord1sARB  = (PFNGLMULTITEXCOORD1SARBPROC)  wglxGetProcAddress("glMultiTexCoord1sARB");
		glMultiTexCoord1svARB = (PFNGLMULTITEXCOORD1SVARBPROC) wglxGetProcAddress("glMultiTexCoord1svARB");
		glMultiTexCoord2dARB  = (PFNGLMULTITEXCOORD2DARBPROC)  wglxGetProcAddress("glMultiTexCoord2dARB");
		glMultiTexCoord2dvARB = (PFNGLMULTITEXCOORD2DVARBPROC) wglxGetProcAddress("glMultiTexCoord2dvARB");
		glMultiTexCoord2fARB  = (PFNGLMULTITEXCOORD2FARBPROC)  wglxGetProcAddress("glMultiTexCoord2fARB");
		glMultiTexCoord2fvARB = (PFNGLMULTITEXCOORD2FVARBPROC) wglxGetProcAddress("glMultiTexCoord2fvARB");
		glMultiTexCoord2iARB  = (PFNGLMULTITEXCOORD2IARBPROC)  wglxGetProcAddress("glMultiTexCoord2iARB");
		glMultiTexCoord2ivARB = (PFNGLMULTITEXCOORD2IVARBPROC) wglxGetProcAddress("glMultiTexCoord2ivARB");
		glMultiTexCoord2sARB  = (PFNGLMULTITEXCOORD2SARBPROC)  wglxGetProcAddress("glMultiTexCoord2sARB");
		glMultiTexCoord2svARB = (PFNGLMULTITEXCOORD2SVARBPROC) wglxGetProcAddress("glMultiTexCoord2svARB");
		glMultiTexCoord3dARB  = (PFNGLMULTITEXCOORD3DARBPROC)  wglxGetProcAddress("glMultiTexCoord3dARB");
		glMultiTexCoord3dvARB = (PFNGLMULTITEXCOORD3DVARBPROC) wglxGetProcAddress("glMultiTexCoord3dvARB");
		glMultiTexCoord3fARB  = (PFNGLMULTITEXCOORD3FARBPROC)  wglxGetProcAddress("glMultiTexCoord3fARB");
		glMultiTexCoord3fvARB = (PFNGLMULTITEXCOORD3FVARBPROC) wglxGetProcAddress("glMultiTexCoord3fvARB");
		glMultiTexCoord3iARB  = (PFNGLMULTITEXCOORD3IARBPROC)  wglxGetProcAddress("glMultiTexCoord3iARB");
		glMultiTexCoord3ivARB = (PFNGLMULTITEXCOORD3IVARBPROC) wglxGetProcAddress("glMultiTexCoord3ivARB");
		glMultiTexCoord3sARB  = (PFNGLMULTITEXCOORD3SARBPROC)  wglxGetProcAddress("glMultiTexCoord3sARB");
		glMultiTexCoord3svARB = (PFNGLMULTITEXCOORD3SVARBPROC) wglxGetProcAddress("glMultiTexCoord3svARB");
		glMultiTexCoord4dARB  = (PFNGLMULTITEXCOORD4DARBPROC)  wglxGetProcAddress("glMultiTexCoord4dARB");
		glMultiTexCoord4dvARB = (PFNGLMULTITEXCOORD4DVARBPROC) wglxGetProcAddress("glMultiTexCoord4dvARB");
		glMultiTexCoord4fARB  = (PFNGLMULTITEXCOORD4FARBPROC)  wglxGetProcAddress("glMultiTexCoord4fARB");
		glMultiTexCoord4fvARB = (PFNGLMULTITEXCOORD4FVARBPROC) wglxGetProcAddress("glMultiTexCoord4fvARB");
		glMultiTexCoord4iARB  = (PFNGLMULTITEXCOORD4IARBPROC)  wglxGetProcAddress("glMultiTexCoord4iARB");
		glMultiTexCoord4ivARB = (PFNGLMULTITEXCOORD4IVARBPROC) wglxGetProcAddress("glMultiTexCoord4ivARB");
		glMultiTexCoord4sARB  = (PFNGLMULTITEXCOORD4SARBPROC)  wglxGetProcAddress("glMultiTexCoord4sARB");
		glMultiTexCoord4svARB = (PFNGLMULTITEXCOORD4SVARBPROC) wglxGetProcAddress("glMultiTexCoord4svARB");
#endif
	}

	if (GL_ARB_occlusion_query_supported = isExtensionSupported("GL_ARB_occlusion_query")){
#ifdef GL_ARB_occlusion_query_PROTOTYPES
		glGenQueriesARB    = (PFNGLGENQUERIESARBPROC)    wglxGetProcAddress("glGenQueriesARB");
		glDeleteQueriesARB = (PFNGLDELETEQUERIESARBPROC) wglxGetProcAddress("glDeleteQueriesARB");
		glIsQueryARB       = (PFNGLISQUERYARBPROC)       wglxGetProcAddress("glIsQueryARB");
		glBeginQueryARB    = (PFNGLBEGINQUERYARBPROC)    wglxGetProcAddress("glBeginQueryARB");
		glEndQueryARB      = (PFNGLENDQUERYARBPROC)      wglxGetProcAddress("glEndQueryARB");
		glGetQueryivARB    = (PFNGLGETQUERYIVARBPROC)    wglxGetProcAddress("glGetQueryivARB");
		glGetQueryObjectivARB  = (PFNGLGETQUERYOBJECTIVARBPROC)  wglxGetProcAddress("glGetQueryObjectivARB");
		glGetQueryObjectuivARB = (PFNGLGETQUERYOBJECTUIVARBPROC) wglxGetProcAddress("glGetQueryObjectuivARB");
#endif
	}

	if (GL_ARB_point_parameters_supported = isExtensionSupported("GL_ARB_point_parameters")){
#ifdef GL_ARB_point_parameters_PROTOTYPES
		glPointParameterfARB  = (PFNGLPOINTPARAMETERFARBPROC)  wglxGetProcAddress("glPointParameterfARB");
		glPointParameterfvARB = (PFNGLPOINTPARAMETERFVARBPROC) wglxGetProcAddress("glPointParameterfvARB");
#endif
	}

	if (GL_ARB_shader_objects_supported = isExtensionSupported("GL_ARB_shader_objects")){
#ifdef GL_ARB_shader_objects_PROTOTYPES
		glDeleteObjectARB        = (PFNGLDELETEOBJECTARBPROC)        wglxGetProcAddress("glDeleteObjectARB");
		glGetHandleARB           = (PFNGLGETHANDLEARBPROC)           wglxGetProcAddress("glGetHandleARB");
		glDetachObjectARB        = (PFNGLDETACHOBJECTARBPROC)        wglxGetProcAddress("glDetachObjectARB");
		glCreateShaderObjectARB  = (PFNGLCREATESHADEROBJECTARBPROC)  wglxGetProcAddress("glCreateShaderObjectARB");
		glShaderSourceARB        = (PFNGLSHADERSOURCEARBPROC)        wglxGetProcAddress("glShaderSourceARB");
		glCompileShaderARB       = (PFNGLCOMPILESHADERARBPROC)       wglxGetProcAddress("glCompileShaderARB");
		glCreateProgramObjectARB = (PFNGLCREATEPROGRAMOBJECTARBPROC) wglxGetProcAddress("glCreateProgramObjectARB");
		glAttachObjectARB        = (PFNGLATTACHOBJECTARBPROC)        wglxGetProcAddress("glAttachObjectARB");
		glLinkProgramARB         = (PFNGLLINKPROGRAMARBPROC)         wglxGetProcAddress("glLinkProgramARB");
		glUseProgramObjectARB    = (PFNGLUSEPROGRAMOBJECTARBPROC)    wglxGetProcAddress("glUseProgramObjectARB");
		glValidateProgramARB     = (PFNGLVALIDATEPROGRAMARBPROC)     wglxGetProcAddress("glValidateProgramARB");

		glUniform1fARB = (PFNGLUNIFORM1FARBPROC) wglxGetProcAddress("glUniform1fARB");
		glUniform2fARB = (PFNGLUNIFORM2FARBPROC) wglxGetProcAddress("glUniform2fARB");
		glUniform3fARB = (PFNGLUNIFORM3FARBPROC) wglxGetProcAddress("glUniform3fARB");
		glUniform4fARB = (PFNGLUNIFORM4FARBPROC) wglxGetProcAddress("glUniform4fARB");

		glUniform1iARB = (PFNGLUNIFORM1IARBPROC) wglxGetProcAddress("glUniform1iARB");
		glUniform2iARB = (PFNGLUNIFORM2IARBPROC) wglxGetProcAddress("glUniform2iARB");
		glUniform3iARB = (PFNGLUNIFORM3IARBPROC) wglxGetProcAddress("glUniform3iARB");
		glUniform4iARB = (PFNGLUNIFORM4IARBPROC) wglxGetProcAddress("glUniform4iARB");

		glUniform1fvARB = (PFNGLUNIFORM1FVARBPROC) wglxGetProcAddress("glUniform1fvARB");
		glUniform2fvARB = (PFNGLUNIFORM2FVARBPROC) wglxGetProcAddress("glUniform2fvARB");
		glUniform3fvARB = (PFNGLUNIFORM3FVARBPROC) wglxGetProcAddress("glUniform3fvARB");
		glUniform4fvARB = (PFNGLUNIFORM4FVARBPROC) wglxGetProcAddress("glUniform4fvARB");

		glUniform1ivARB = (PFNGLUNIFORM1IVARBPROC) wglxGetProcAddress("glUniform1ivARB");
		glUniform2ivARB = (PFNGLUNIFORM2IVARBPROC) wglxGetProcAddress("glUniform2ivARB");
		glUniform3ivARB = (PFNGLUNIFORM3IVARBPROC) wglxGetProcAddress("glUniform3ivARB");
		glUniform4ivARB = (PFNGLUNIFORM4IVARBPROC) wglxGetProcAddress("glUniform4ivARB");

		glUniformMatrix2fvARB = (PFNGLUNIFORMMATRIX2FVARBPROC) wglxGetProcAddress("glUniformMatrix2fvARB");
		glUniformMatrix3fvARB = (PFNGLUNIFORMMATRIX3FVARBPROC) wglxGetProcAddress("glUniformMatrix3fvARB");
		glUniformMatrix4fvARB = (PFNGLUNIFORMMATRIX4FVARBPROC) wglxGetProcAddress("glUniformMatrix4fvARB");

		glGetObjectParameterfvARB = (PFNGLGETOBJECTPARAMETERFVARBPROC) wglxGetProcAddress("glGetObjectParameterfvARB");
		glGetObjectParameterivARB = (PFNGLGETOBJECTPARAMETERIVARBPROC) wglxGetProcAddress("glGetObjectParameterivARB");
		glGetInfoLogARB           = (PFNGLGETINFOLOGARBPROC)           wglxGetProcAddress("glGetInfoLogARB");
		glGetAttachedObjectsARB   = (PFNGLGETATTACHEDOBJECTSARBPROC)   wglxGetProcAddress("glGetAttachedObjectsARB");
		glGetUniformLocationARB   = (PFNGLGETUNIFORMLOCATIONARBPROC)   wglxGetProcAddress("glGetUniformLocationARB");
		glGetActiveUniformARB     = (PFNGLGETACTIVEUNIFORMARBPROC)     wglxGetProcAddress("glGetActiveUniformARB");
		glGetUniformfvARB         = (PFNGLGETUNIFORMFVARBPROC)         wglxGetProcAddress("glGetUniformfvARB");
		glGetUniformivARB         = (PFNGLGETUNIFORMIVARBPROC)         wglxGetProcAddress("glGetUniformivARB");
		glGetShaderSourceARB      = (PFNGLGETSHADERSOURCEARBPROC)      wglxGetProcAddress("glGetShaderSourceARB");
#endif
	}

	GL_ARB_shading_language_100_supported = isExtensionSupported("GL_ARB_shading_language_100");

	GL_ARB_shadow_supported         = isExtensionSupported("GL_ARB_shadow");
	GL_ARB_shadow_ambient_supported = isExtensionSupported("GL_ARB_shadow_ambient");

	if (GL_ARB_texture_compression_supported = isExtensionSupported("GL_ARB_texture_compression")){
#ifdef GL_ARB_texture_compression_PROTOTYPES
		glCompressedTexImage1DARB     = (PFNGLCOMPRESSEDTEXIMAGE1DARBPROC)    wglxGetProcAddress("glCompressedTexImage1DARB");
		glCompressedTexImage2DARB     = (PFNGLCOMPRESSEDTEXIMAGE2DARBPROC)    wglxGetProcAddress("glCompressedTexImage2DARB");
		glCompressedTexImage3DARB     = (PFNGLCOMPRESSEDTEXIMAGE3DARBPROC)    wglxGetProcAddress("glCompressedTexImage3DARB");
		glCompressedTexSubImage1DARB  = (PFNGLCOMPRESSEDTEXSUBIMAGE1DARBPROC) wglxGetProcAddress("glCompressedTexSubImage1DARB");
		glCompressedTexSubImage2DARB  = (PFNGLCOMPRESSEDTEXSUBIMAGE2DARBPROC) wglxGetProcAddress("glCompressedTexSubImage2DARB");
		glCompressedTexSubImage3DARB  = (PFNGLCOMPRESSEDTEXSUBIMAGE3DARBPROC) wglxGetProcAddress("glCompressedTexSubImage3DARB");
		glGetCompressedTexImageARB    = (PFNGLGETCOMPRESSEDTEXIMAGEARBPROC)   wglxGetProcAddress("glGetCompressedTexImageARB");
#endif
	}

	GL_ARB_texture_cube_map_supported     = isExtensionSupported("GL_ARB_texture_cube_map");
    GL_ARB_texture_env_add_supported      = isExtensionSupported("GL_ARB_texture_env_add");
	GL_ARB_texture_env_combine_supported  = isExtensionSupported("GL_ARB_texture_env_combine");
	GL_ARB_texture_env_crossbar_supported = isExtensionSupported("GL_ARB_texture_env_crossbar");
	GL_ARB_texture_env_dot3_supported     = isExtensionSupported("GL_ARB_texture_env_dot3");

	if (GL_ARB_transpose_matrix_supported = isExtensionSupported("GL_ARB_transpose_matrix")){
#ifdef GL_ARB_transpose_matrix_PROTOTYPES
		glLoadTransposeMatrixfARB = (PFNGLLOADTRANSPOSEMATRIXFARBPROC) wglxGetProcAddress("glLoadTransposeMatrixfARB");
		glLoadTransposeMatrixdARB = (PFNGLLOADTRANSPOSEMATRIXDARBPROC) wglxGetProcAddress("glLoadTransposeMatrixdARB");
		glMultTransposeMatrixfARB = (PFNGLMULTTRANSPOSEMATRIXFARBPROC) wglxGetProcAddress("glMultTransposeMatrixfARB");
		glMultTransposeMatrixdARB = (PFNGLMULTTRANSPOSEMATRIXDARBPROC) wglxGetProcAddress("glMultTransposeMatrixdARB");
#endif
	}

	if (GL_ARB_vertex_buffer_object_supported = isExtensionSupported("GL_ARB_vertex_buffer_object")){
#ifdef GL_ARB_vertex_buffer_object_PROTOTYPES
		glBindBufferARB    = (PFNGLBINDBUFFERARBPROC)    wglxGetProcAddress("glBindBufferARB");
		glDeleteBuffersARB = (PFNGLDELETEBUFFERSARBPROC) wglxGetProcAddress("glDeleteBuffersARB");
		glGenBuffersARB    = (PFNGLGENBUFFERSARBPROC)    wglxGetProcAddress("glGenBuffersARB");
		glIsBufferARB      = (PFNGLISBUFFERARBPROC)      wglxGetProcAddress("glIsBufferARB");

		glBufferDataARB       = (PFNGLBUFFERDATAARBPROC)       wglxGetProcAddress("glBufferDataARB");
		glBufferSubDataARB    = (PFNGLBUFFERSUBDATAARBPROC)    wglxGetProcAddress("glBufferSubDataARB");
		glGetBufferSubDataARB = (PFNGLGETBUFFERSUBDATAARBPROC) wglxGetProcAddress("glGetBufferSubDataARB");

		glMapBufferARB   = (PFNGLMAPBUFFERARBPROC)   wglxGetProcAddress("glMapBufferARB");
		glUnmapBufferARB = (PFNGLUNMAPBUFFERARBPROC) wglxGetProcAddress("glUnmapBufferARB");

		glGetBufferParameterivARB = (PFNGLGETBUFFERPARAMETERIVARBPROC) wglxGetProcAddress("glGetBufferParameterivARB");
		glGetBufferPointervARB    = (PFNGLGETBUFFERPOINTERVARBPROC)    wglxGetProcAddress("glGetBufferPointervARB");
#endif
	}

	GL_ARB_vertex_program_supported = isExtensionSupported("GL_ARB_vertex_program");

	if (GL_ARB_vertex_program_supported || GL_ARB_fragment_program_supported){
#if defined(GL_ARB_vertex_program_PROTOTYPES) && defined(GL_ARB_fragment_program_PROTOTYPES)
		glProgramStringARB  = (PFNGLPROGRAMSTRINGARBPROC)  wglxGetProcAddress("glProgramStringARB");
		glBindProgramARB    = (PFNGLBINDPROGRAMARBPROC)    wglxGetProcAddress("glBindProgramARB");
		glDeleteProgramsARB = (PFNGLDELETEPROGRAMSARBPROC) wglxGetProcAddress("glDeleteProgramsARB");
		glGenProgramsARB    = (PFNGLGENPROGRAMSARBPROC)    wglxGetProcAddress("glGenProgramsARB");
		glProgramEnvParameter4dARB    = (PFNGLPROGRAMENVPARAMETER4DARBPROC)    wglxGetProcAddress("glProgramEnvParameter4dARB");
		glProgramEnvParameter4dvARB   = (PFNGLPROGRAMENVPARAMETER4DVARBPROC)   wglxGetProcAddress("glProgramEnvParameter4dvARB");
		glProgramEnvParameter4fARB    = (PFNGLPROGRAMENVPARAMETER4FARBPROC)    wglxGetProcAddress("glProgramEnvParameter4fARB");
		glProgramEnvParameter4fvARB   = (PFNGLPROGRAMENVPARAMETER4FVARBPROC)   wglxGetProcAddress("glProgramEnvParameter4fvARB");
		glProgramLocalParameter4dARB  = (PFNGLPROGRAMLOCALPARAMETER4DARBPROC)  wglxGetProcAddress("glProgramLocalParameter4dARB");
		glProgramLocalParameter4dvARB = (PFNGLPROGRAMLOCALPARAMETER4DVARBPROC) wglxGetProcAddress("glProgramLocalParameter4dvARB");
		glProgramLocalParameter4fARB  = (PFNGLPROGRAMLOCALPARAMETER4FARBPROC)  wglxGetProcAddress("glProgramLocalParameter4fARB");
		glProgramLocalParameter4fvARB = (PFNGLPROGRAMLOCALPARAMETER4FVARBPROC) wglxGetProcAddress("glProgramLocalParameter4fvARB");
		glGetProgramEnvParameterdvARB = (PFNGLGETPROGRAMENVPARAMETERDVARBPROC) wglxGetProcAddress("glGetProgramEnvParameterdvARB");
		glGetProgramEnvParameterfvARB = (PFNGLGETPROGRAMENVPARAMETERFVARBPROC) wglxGetProcAddress("glGetProgramEnvParameterfvARB");
		glGetProgramLocalParameterdvARB = (PFNGLGETPROGRAMLOCALPARAMETERDVARBPROC) wglxGetProcAddress("glGetProgramLocalParameterdvARB");
		glGetProgramLocalParameterfvARB = (PFNGLGETPROGRAMLOCALPARAMETERFVARBPROC) wglxGetProcAddress("glGetProgramLocalParameterfvARB");
		glGetProgramivARB     = (PFNGLGETPROGRAMIVARBPROC)     wglxGetProcAddress("glGetProgramivARB");
		glGetProgramStringARB = (PFNGLGETPROGRAMSTRINGARBPROC) wglxGetProcAddress("glGetProgramStringARB");
		glIsProgramARB        = (PFNGLISPROGRAMARBPROC)        wglxGetProcAddress("glIsProgramARB");
#endif
	}

	if (GL_ARB_vertex_shader_supported = isExtensionSupported("GL_ARB_vertex_shader")){
#ifdef GL_ARB_vertex_shader_PROTOTYPES
		glBindAttribLocationARB = (PFNGLBINDATTRIBLOCATIONARBPROC) wglxGetProcAddress("glBindAttribLocationARB");
		glGetActiveAttribARB    = (PFNGLGETACTIVEATTRIBARBPROC)    wglxGetProcAddress("glGetActiveAttribARB");
		glGetAttribLocationARB  = (PFNGLGETATTRIBLOCATIONARBPROC)  wglxGetProcAddress("glGetAttribLocationARB");
#endif
	}

	if (GL_ARB_vertex_program_supported || GL_ARB_vertex_shader_supported){
#ifdef GL_ARB_vertex_program_PROTOTYPES
		glVertexAttrib1sARB = (PFNGLVERTEXATTRIB1SARBPROC) wglxGetProcAddress("glVertexAttrib1sARB");
		glVertexAttrib1fARB = (PFNGLVERTEXATTRIB1FARBPROC) wglxGetProcAddress("glVertexAttrib1fARB");
		glVertexAttrib1dARB = (PFNGLVERTEXATTRIB1DARBPROC) wglxGetProcAddress("glVertexAttrib1dARB");
		glVertexAttrib2sARB = (PFNGLVERTEXATTRIB2SARBPROC) wglxGetProcAddress("glVertexAttrib2sARB");
		glVertexAttrib2fARB = (PFNGLVERTEXATTRIB2FARBPROC) wglxGetProcAddress("glVertexAttrib2fARB");
		glVertexAttrib2dARB = (PFNGLVERTEXATTRIB2DARBPROC) wglxGetProcAddress("glVertexAttrib2dARB");
		glVertexAttrib3sARB = (PFNGLVERTEXATTRIB3SARBPROC) wglxGetProcAddress("glVertexAttrib3sARB");
		glVertexAttrib3fARB = (PFNGLVERTEXATTRIB3FARBPROC) wglxGetProcAddress("glVertexAttrib3fARB");
		glVertexAttrib3dARB = (PFNGLVERTEXATTRIB3DARBPROC) wglxGetProcAddress("glVertexAttrib3dARB");
		glVertexAttrib4sARB = (PFNGLVERTEXATTRIB4SARBPROC) wglxGetProcAddress("glVertexAttrib4sARB");
		glVertexAttrib4fARB = (PFNGLVERTEXATTRIB4FARBPROC) wglxGetProcAddress("glVertexAttrib4fARB");
		glVertexAttrib4dARB = (PFNGLVERTEXATTRIB4DARBPROC) wglxGetProcAddress("glVertexAttrib4dARB");
		glVertexAttrib4NubARB = (PFNGLVERTEXATTRIB4NUBARBPROC) wglxGetProcAddress("glVertexAttrib4NubARB");
		glVertexAttrib1svARB  = (PFNGLVERTEXATTRIB1SVARBPROC)  wglxGetProcAddress("glVertexAttrib1svARB");
		glVertexAttrib1fvARB  = (PFNGLVERTEXATTRIB1FVARBPROC)  wglxGetProcAddress("glVertexAttrib1fvARB");
		glVertexAttrib1dvARB  = (PFNGLVERTEXATTRIB1DVARBPROC)  wglxGetProcAddress("glVertexAttrib1dvARB");
		glVertexAttrib2svARB  = (PFNGLVERTEXATTRIB2SVARBPROC)  wglxGetProcAddress("glVertexAttrib2svARB");
		glVertexAttrib2fvARB  = (PFNGLVERTEXATTRIB2FVARBPROC)  wglxGetProcAddress("glVertexAttrib2fvARB");
		glVertexAttrib2dvARB  = (PFNGLVERTEXATTRIB2DVARBPROC)  wglxGetProcAddress("glVertexAttrib2dvARB");
		glVertexAttrib3svARB  = (PFNGLVERTEXATTRIB3SVARBPROC)  wglxGetProcAddress("glVertexAttrib3svARB");
		glVertexAttrib3fvARB  = (PFNGLVERTEXATTRIB3FVARBPROC)  wglxGetProcAddress("glVertexAttrib3fvARB");
		glVertexAttrib3dvARB  = (PFNGLVERTEXATTRIB3DVARBPROC)  wglxGetProcAddress("glVertexAttrib3dvARB");
		glVertexAttrib4bvARB  = (PFNGLVERTEXATTRIB4BVARBPROC)  wglxGetProcAddress("glVertexAttrib4bvARB");
		glVertexAttrib4svARB  = (PFNGLVERTEXATTRIB4SVARBPROC)  wglxGetProcAddress("glVertexAttrib4svARB");
		glVertexAttrib4ivARB  = (PFNGLVERTEXATTRIB4IVARBPROC)  wglxGetProcAddress("glVertexAttrib4ivARB");
		glVertexAttrib4ubvARB = (PFNGLVERTEXATTRIB4UBVARBPROC) wglxGetProcAddress("glVertexAttrib4ubvARB");
		glVertexAttrib4usvARB = (PFNGLVERTEXATTRIB4USVARBPROC) wglxGetProcAddress("glVertexAttrib4usvARB");
		glVertexAttrib4uivARB = (PFNGLVERTEXATTRIB4UIVARBPROC) wglxGetProcAddress("glVertexAttrib4uivARB");
		glVertexAttrib4fvARB  = (PFNGLVERTEXATTRIB4FVARBPROC)  wglxGetProcAddress("glVertexAttrib4fvARB");
		glVertexAttrib4dvARB  = (PFNGLVERTEXATTRIB4DVARBPROC)  wglxGetProcAddress("glVertexAttrib4dvARB");
		glVertexAttrib4NbvARB = (PFNGLVERTEXATTRIB4NBVARBPROC) wglxGetProcAddress("glVertexAttrib4NbvARB");
		glVertexAttrib4NsvARB = (PFNGLVERTEXATTRIB4NSVARBPROC) wglxGetProcAddress("glVertexAttrib4NsvARB");
		glVertexAttrib4NivARB = (PFNGLVERTEXATTRIB4NIVARBPROC) wglxGetProcAddress("glVertexAttrib4NivARB");
		glVertexAttrib4NubvARB = (PFNGLVERTEXATTRIB4NUBVARBPROC) wglxGetProcAddress("glVertexAttrib4NubvARB");
		glVertexAttrib4NusvARB = (PFNGLVERTEXATTRIB4NUSVARBPROC) wglxGetProcAddress("glVertexAttrib4NusvARB");
		glVertexAttrib4NuivARB = (PFNGLVERTEXATTRIB4NUIVARBPROC) wglxGetProcAddress("glVertexAttrib4NuivARB");

		glVertexAttribPointerARB = (PFNGLVERTEXATTRIBPOINTERARBPROC) wglxGetProcAddress("glVertexAttribPointerARB");
		glEnableVertexAttribArrayARB = (PFNGLENABLEVERTEXATTRIBARRAYARBPROC) wglxGetProcAddress("glEnableVertexAttribArrayARB");
		glDisableVertexAttribArrayARB = (PFNGLDISABLEVERTEXATTRIBARRAYARBPROC) wglxGetProcAddress("glDisableVertexAttribArrayARB");

		glGetVertexAttribdvARB = (PFNGLGETVERTEXATTRIBDVARBPROC) wglxGetProcAddress("glGetVertexAttribdvARB");
		glGetVertexAttribfvARB = (PFNGLGETVERTEXATTRIBFVARBPROC) wglxGetProcAddress("glGetVertexAttribfvARB");
		glGetVertexAttribivARB = (PFNGLGETVERTEXATTRIBIVARBPROC) wglxGetProcAddress("glGetVertexAttribivARB");
		glGetVertexAttribPointervARB = (PFNGLGETVERTEXATTRIBPOINTERVARBPROC) wglxGetProcAddress("glGetVertexAttribPointervARB");
#endif
	}

	if (GL_ARB_window_pos_supported = isExtensionSupported("GL_ARB_window_pos")){
#ifdef GL_ARB_window_pos_PROTOTYPES
		glWindowPos2dARB  = (PFNGLWINDOWPOS2DARBPROC)  wglxGetProcAddress("glWindowPos2dARB");
		glWindowPos2fARB  = (PFNGLWINDOWPOS2FARBPROC)  wglxGetProcAddress("glWindowPos2fARB");
		glWindowPos2iARB  = (PFNGLWINDOWPOS2IARBPROC)  wglxGetProcAddress("glWindowPos2iARB");
		glWindowPos2sARB  = (PFNGLWINDOWPOS2SARBPROC)  wglxGetProcAddress("glWindowPos2sARB");
		glWindowPos2ivARB = (PFNGLWINDOWPOS2IVARBPROC) wglxGetProcAddress("glWindowPos2ivARB");
		glWindowPos2svARB = (PFNGLWINDOWPOS2SVARBPROC) wglxGetProcAddress("glWindowPos2svARB");
		glWindowPos2fvARB = (PFNGLWINDOWPOS2FVARBPROC) wglxGetProcAddress("glWindowPos2fvARB");
		glWindowPos2dvARB = (PFNGLWINDOWPOS2DVARBPROC) wglxGetProcAddress("glWindowPos2dvARB");
		glWindowPos3iARB  = (PFNGLWINDOWPOS3IARBPROC)  wglxGetProcAddress("glWindowPos3iARB");
		glWindowPos3sARB  = (PFNGLWINDOWPOS3SARBPROC)  wglxGetProcAddress("glWindowPos3sARB");
		glWindowPos3fARB  = (PFNGLWINDOWPOS3FARBPROC)  wglxGetProcAddress("glWindowPos3fARB");
		glWindowPos3dARB  = (PFNGLWINDOWPOS3DARBPROC)  wglxGetProcAddress("glWindowPos3dARB");
		glWindowPos3ivARB = (PFNGLWINDOWPOS3IVARBPROC) wglxGetProcAddress("glWindowPos3ivARB");
		glWindowPos3svARB = (PFNGLWINDOWPOS3SVARBPROC) wglxGetProcAddress("glWindowPos3svARB");
		glWindowPos3fvARB = (PFNGLWINDOWPOS3FVARBPROC) wglxGetProcAddress("glWindowPos3fvARB");
		glWindowPos3dvARB = (PFNGLWINDOWPOS3DVARBPROC) wglxGetProcAddress("glWindowPos3dvARB");
#endif
	}

	if (GL_ATI_fragment_shader_supported = isExtensionSupported("GL_ATI_fragment_shader")){
#ifdef GL_ATI_fragment_shader_PROTOTYPES
		glGenFragmentShadersATI   = (PFNGLGENFRAGMENTSHADERSATIPROC)   wglxGetProcAddress("glGenFragmentShadersATI");
		glBindFragmentShaderATI   = (PFNGLBINDFRAGMENTSHADERATIPROC)   wglxGetProcAddress("glBindFragmentShaderATI");
		glDeleteFragmentShaderATI = (PFNGLDELETEFRAGMENTSHADERATIPROC) wglxGetProcAddress("glDeleteFragmentShaderATI");
		glBeginFragmentShaderATI  = (PFNGLBEGINFRAGMENTSHADERATIPROC)  wglxGetProcAddress("glBeginFragmentShaderATI");
		glEndFragmentShaderATI    = (PFNGLENDFRAGMENTSHADERATIPROC)    wglxGetProcAddress("glEndFragmentShaderATI");
		glPassTexCoordATI         = (PFNGLPASSTEXCOORDATIPROC)         wglxGetProcAddress("glPassTexCoordATI");
		glSampleMapATI            = (PFNGLSAMPLEMAPATIPROC)            wglxGetProcAddress("glSampleMapATI");

		glColorFragmentOp1ATI = (PFNGLCOLORFRAGMENTOP1ATIPROC) wglxGetProcAddress("glColorFragmentOp1ATI");
		glColorFragmentOp2ATI = (PFNGLCOLORFRAGMENTOP2ATIPROC) wglxGetProcAddress("glColorFragmentOp2ATI");
		glColorFragmentOp3ATI = (PFNGLCOLORFRAGMENTOP3ATIPROC) wglxGetProcAddress("glColorFragmentOp3ATI");

		glAlphaFragmentOp1ATI = (PFNGLALPHAFRAGMENTOP1ATIPROC) wglxGetProcAddress("glAlphaFragmentOp1ATI");
		glAlphaFragmentOp2ATI = (PFNGLALPHAFRAGMENTOP2ATIPROC) wglxGetProcAddress("glAlphaFragmentOp2ATI");
		glAlphaFragmentOp3ATI = (PFNGLALPHAFRAGMENTOP3ATIPROC) wglxGetProcAddress("glAlphaFragmentOp3ATI");

		glSetFragmentShaderConstantATI = (PFNGLSETFRAGMENTSHADERCONSTANTATIPROC) wglxGetProcAddress("glSetFragmentShaderConstantATI");
#endif
	}

	if (GL_ATI_separate_stencil_supported = isExtensionSupported("GL_ATI_separate_stencil")){
#ifdef GL_ATI_separate_stencil_PROTOTYPES
		glStencilOpSeparateATI   = (PFNGLSTENCILOPSEPARATEATIPROC)   wglxGetProcAddress("glStencilOpSeparateATI");
		glStencilFuncSeparateATI = (PFNGLSTENCILFUNCSEPARATEATIPROC) wglxGetProcAddress("glStencilFuncSeparateATI");
#endif
	}


	GL_ATI_texture_compression_3dc_supported = isExtensionSupported("GL_ATI_texture_compression_3dc");
	GL_ATI_texture_float_supported           = isExtensionSupported("GL_ATI_texture_float");
	GL_ATI_texture_mirror_once_supported     = isExtensionSupported("GL_ATI_texture_mirror_once");

	if (GL_EXT_blend_color_supported = isExtensionSupported("GL_EXT_blend_color")){
#ifdef GL_EXT_blend_color_PROTOTYPES
		glBlendColorEXT = (PFNGLBLENDCOLOREXTPROC) wglxGetProcAddress("glBlendColorEXT");
#endif
	}

	if (GL_EXT_blend_func_separate_supported = isExtensionSupported("GL_EXT_blend_func_separate")){
#ifdef GL_EXT_blend_func_separate_PROTOTYPES
		glBlendFuncSeparateEXT = (PFNGLBLENDFUNCSEPARATEEXTPROC) wglxGetProcAddress("glBlendFuncSeparateEXT");
#endif
	}

	GL_EXT_blend_minmax_supported   = isExtensionSupported("GL_EXT_blend_minmax");
	GL_EXT_blend_subtract_supported = isExtensionSupported("GL_EXT_blend_subtract");
	if (GL_EXT_blend_minmax_supported || GL_EXT_blend_subtract_supported){
#ifdef GL_EXT_blend_minmax_PROTOTYPES
		glBlendEquationEXT = (PFNGLBLENDEQUATIONEXTPROC) wglxGetProcAddress("glBlendEquationEXT");
#endif
	}

	if (GL_EXT_draw_range_elements_supported = isExtensionSupported("GL_EXT_draw_range_elements")){
#ifdef GL_EXT_draw_range_elements_PROTOTYPES
		glDrawRangeElementsEXT = (PFNGLDRAWRANGEELEMENTSEXTPROC) wglxGetProcAddress("glDrawRangeElementsEXT");
#endif
	}

	if (GL_EXT_fog_coord_supported = isExtensionSupported("GL_EXT_fog_coord")){
#ifdef GL_EXT_fog_coord_PROTOTYPES
		glFogCoordfEXT  = (PFNGLFOGCOORDFEXTPROC)  wglxGetProcAddress("glFogCoordfEXT");
		glFogCoorddEXT  = (PFNGLFOGCOORDDEXTPROC)  wglxGetProcAddress("glFogCoorddEXT");
		glFogCoordfvEXT = (PFNGLFOGCOORDFVEXTPROC) wglxGetProcAddress("glFogCoordfvEXT");
		glFogCoorddvEXT = (PFNGLFOGCOORDDVEXTPROC) wglxGetProcAddress("glFogCoorddvEXT");
		glFogCoordPointerEXT = (PFNGLFOGCOORDPOINTEREXTPROC) wglxGetProcAddress("glFogCoordPointerEXT");
#endif
	}

	if (GL_EXT_framebuffer_object_supported = isExtensionSupported("GL_EXT_framebuffer_object")){
#ifdef GL_EXT_framebuffer_object_PROTOTYPES
		glIsRenderbufferEXT   = (PFNGLISRENDERBUFFEREXTPROC) wglxGetProcAddress("glIsRenderbufferEXT");
		glBindRenderbufferEXT = (PFNGLBINDRENDERBUFFEREXTPROC) wglxGetProcAddress("glBindRenderbufferEXT");
		glDeleteRenderbuffersEXT = (PFNGLDELETERENDERBUFFERSEXTPROC) wglxGetProcAddress("glDeleteRenderbuffersEXT");
		glGenRenderbuffersEXT    = (PFNGLGENRENDERBUFFERSEXTPROC) wglxGetProcAddress("glGenRenderbuffersEXT");
		glRenderbufferStorageEXT = (PFNGLRENDERBUFFERSTORAGEEXTPROC) wglxGetProcAddress("glRenderbufferStorageEXT");
		glGetRenderbufferParameterivEXT = (PFNGLGETRENDERBUFFERPARAMETERIVEXTPROC) wglxGetProcAddress("glGetRenderbufferParameterivEXT");
		glIsFramebufferEXT       = (PFNGLISFRAMEBUFFEREXTPROC) wglxGetProcAddress("glIsFramebufferEXT");
		glBindFramebufferEXT     = (PFNGLBINDFRAMEBUFFEREXTPROC) wglxGetProcAddress("glBindFramebufferEXT");
		glDeleteFramebuffersEXT  = (PFNGLDELETEFRAMEBUFFERSEXTPROC) wglxGetProcAddress("glDeleteFramebuffersEXT");
		glGenFramebuffersEXT     = (PFNGLGENFRAMEBUFFERSEXTPROC) wglxGetProcAddress("glGenFramebuffersEXT");
		glCheckFramebufferStatusEXT  = (PFNGLCHECKFRAMEBUFFERSTATUSEXTPROC) wglxGetProcAddress("glCheckFramebufferStatusEXT");
		glFramebufferTexture1DEXT    = (PFNGLFRAMEBUFFERTEXTURE1DEXTPROC) wglxGetProcAddress("glFramebufferTexture1DEXT");
		glFramebufferTexture2DEXT    = (PFNGLFRAMEBUFFERTEXTURE2DEXTPROC) wglxGetProcAddress("glFramebufferTexture2DEXT");
		glFramebufferTexture3DEXT    = (PFNGLFRAMEBUFFERTEXTURE3DEXTPROC) wglxGetProcAddress("glFramebufferTexture3DEXT");
		glFramebufferRenderbufferEXT = (PFNGLFRAMEBUFFERRENDERBUFFEREXTPROC) wglxGetProcAddress("glFramebufferRenderbufferEXT");
		glGetFramebufferAttachmentParameterivEXT = (PFNGLGETFRAMEBUFFERATTACHMENTPARAMETERIVEXTPROC) wglxGetProcAddress("glGetFramebufferAttachmentParameterivEXT");
		glGenerateMipmapEXT = (PFNGLGENERATEMIPMAPEXTPROC) wglxGetProcAddress("glGenerateMipmapEXT");
#endif
	}


	if (GL_EXT_multi_draw_arrays_supported = isExtensionSupported("GL_EXT_multi_draw_arrays")){
#ifdef GL_EXT_multi_draw_arrays_PROTOTYPES
		glMultiDrawArraysEXT   = (PFNGLMULTIDRAWARRAYSEXTPROC)   wglxGetProcAddress("glMultiDrawArraysEXT");
		glMultiDrawElementsEXT = (PFNGLMULTIDRAWELEMENTSEXTPROC) wglxGetProcAddress("glMultiDrawElementsEXT");
#endif
	}

	GL_EXT_packed_pixels_supported = isExtensionSupported("GL_EXT_packed_pixels");
	GL_EXT_packed_depth_stencil_supported = isExtensionSupported("GL_EXT_packed_depth_stencil");

	if (GL_EXT_secondary_color_supported = isExtensionSupported("GL_EXT_secondary_color")){
#ifdef GL_EXT_secondary_color_PROTOTYPES
		glSecondaryColor3fEXT = (PFNGLSECONDARYCOLOR3FEXTPROC) wglxGetProcAddress("glSecondaryColor3fEXT");
		glSecondaryColor3dEXT = (PFNGLSECONDARYCOLOR3DEXTPROC) wglxGetProcAddress("glSecondaryColor3dEXT");
		glSecondaryColor3bEXT = (PFNGLSECONDARYCOLOR3BEXTPROC) wglxGetProcAddress("glSecondaryColor3bEXT");
		glSecondaryColor3sEXT = (PFNGLSECONDARYCOLOR3SEXTPROC) wglxGetProcAddress("glSecondaryColor3sEXT");
		glSecondaryColor3iEXT = (PFNGLSECONDARYCOLOR3IEXTPROC) wglxGetProcAddress("glSecondaryColor3iEXT");
		glSecondaryColor3ubEXT = (PFNGLSECONDARYCOLOR3UBEXTPROC) wglxGetProcAddress("glSecondaryColor3ubEXT");
		glSecondaryColor3usEXT = (PFNGLSECONDARYCOLOR3USEXTPROC) wglxGetProcAddress("glSecondaryColor3usEXT");
		glSecondaryColor3uiEXT = (PFNGLSECONDARYCOLOR3UIEXTPROC) wglxGetProcAddress("glSecondaryColor3uiEXT");

		glSecondaryColor3fvEXT = (PFNGLSECONDARYCOLOR3FVEXTPROC) wglxGetProcAddress("glSecondaryColor3fvEXT");
		glSecondaryColor3dvEXT = (PFNGLSECONDARYCOLOR3DVEXTPROC) wglxGetProcAddress("glSecondaryColor3dvEXT");
		glSecondaryColor3bvEXT = (PFNGLSECONDARYCOLOR3BVEXTPROC) wglxGetProcAddress("glSecondaryColor3bvEXT");
		glSecondaryColor3svEXT = (PFNGLSECONDARYCOLOR3SVEXTPROC) wglxGetProcAddress("glSecondaryColor3svEXT");
		glSecondaryColor3ivEXT = (PFNGLSECONDARYCOLOR3IVEXTPROC) wglxGetProcAddress("glSecondaryColor3ivEXT");
		glSecondaryColor3ubvEXT = (PFNGLSECONDARYCOLOR3UBVEXTPROC) wglxGetProcAddress("glSecondaryColor3ubvEXT");
		glSecondaryColor3usvEXT = (PFNGLSECONDARYCOLOR3USVEXTPROC) wglxGetProcAddress("glSecondaryColor3usvEXT");
		glSecondaryColor3uivEXT = (PFNGLSECONDARYCOLOR3UIVEXTPROC) wglxGetProcAddress("glSecondaryColor3uivEXT");

		glSecondaryColorPointerEXT = (PFNGLSECONDARYCOLORPOINTEREXTPROC) wglxGetProcAddress("glSecondaryColorPointerEXT");
#endif
	}
	GL_EXT_stencil_wrap_supported = isExtensionSupported("GL_EXT_stencil_wrap");

	if (GL_EXT_texture3D_supported = isExtensionSupported("GL_EXT_texture3D")){
#ifdef GL_EXT_texture3D_PROTOTYPES
		glTexImage3DEXT = (PFNGLTEXIMAGE3DEXTPROC) wglxGetProcAddress("glTexImage3DEXT");
#endif
	}

	GL_EXT_texture_compression_s3tc_supported   = isExtensionSupported("GL_EXT_texture_compression_s3tc");
	GL_EXT_texture_edge_clamp_supported         = isExtensionSupported("GL_EXT_texture_edge_clamp") | isExtensionSupported("GL_SGIS_texture_edge_clamp");
	GL_EXT_texture_filter_anisotropic_supported = isExtensionSupported("GL_EXT_texture_filter_anisotropic");
	GL_EXT_texture_lod_bias_supported           = isExtensionSupported("GL_EXT_texture_lod_bias");

	GL_HP_occlusion_test_supported = isExtensionSupported("GL_HP_occlusion_test");

	GL_NV_blend_square_supported = isExtensionSupported("GL_NV_blend_square");

	GL_SGIS_generate_mipmap_supported = isExtensionSupported("GL_SGIS_generate_mipmap");

#if defined(_WIN32)

	if (WGL_ARB_pbuffer_supported = isWGLXExtensionSupported("WGL_ARB_pbuffer")){
		wglCreatePbufferARB    = (PFNWGLCREATEPBUFFERARBPROC)    wglxGetProcAddress("wglCreatePbufferARB");
		wglGetPbufferDCARB     = (PFNWGLGETPBUFFERDCARBPROC)     wglxGetProcAddress("wglGetPbufferDCARB");
		wglReleasePbufferDCARB = (PFNWGLRELEASEPBUFFERDCARBPROC) wglxGetProcAddress("wglReleasePbufferDCARB");
		wglDestroyPbufferARB   = (PFNWGLDESTROYPBUFFERARBPROC)   wglxGetProcAddress("wglDestroyPbufferARB");
		wglQueryPbufferARB     = (PFNWGLQUERYPBUFFERARBPROC)     wglxGetProcAddress("wglQueryPbufferARB");
	}

	if (WGL_ARB_pixel_format_supported = isWGLXExtensionSupported("WGL_ARB_pixel_format")){
		wglGetPixelFormatAttribivARB = (PFNWGLGETPIXELFORMATATTRIBIVARBPROC) wglxGetProcAddress("wglGetPixelFormatAttribivARB");
		wglGetPixelFormatAttribfvARB = (PFNWGLGETPIXELFORMATATTRIBFVARBPROC) wglxGetProcAddress("wglGetPixelFormatAttribfvARB");
		wglChoosePixelFormatARB      = (PFNWGLCHOOSEPIXELFORMATARBPROC)      wglxGetProcAddress("wglChoosePixelFormatARB");
	}

    if (WGL_ARB_make_current_read_supported  = isWGLXExtensionSupported("WGL_ARB_make_current_read")){
        wglMakeContextCurrentARB = (PFNWGLMAKECONTEXTCURRENTARBPROC) wglxGetProcAddress("wglMakeContextCurrentARB");
        wglGetCurrentReadDCARB   = (PFNWGLGETCURRENTREADDCARBPROC)   wglxGetProcAddress("wglGetCurrentReadDCARB");
    }

	WGL_ARB_multisample_supported = isWGLXExtensionSupported("WGL_ARB_multisample");

	if (WGL_ARB_render_texture_supported = isWGLXExtensionSupported("WGL_ARB_render_texture")){
		wglBindTexImageARB     = (PFNWGLBINDTEXIMAGEARBPROC)     wglxGetProcAddress("wglBindTexImageARB");
		wglReleaseTexImageARB  = (PFNWGLRELEASETEXIMAGEARBPROC)  wglxGetProcAddress("wglReleaseTexImageARB");
		wglSetPbufferAttribARB = (PFNWGLSETPBUFFERATTRIBARBPROC) wglxGetProcAddress("wglSetPbufferAttribARB");
	}

	WGL_ATI_pixel_format_float_supported = isWGLXExtensionSupported("WGL_ATI_pixel_format_float");

	if (WGL_EXT_swap_control_supported = isWGLXExtensionSupported("WGL_EXT_swap_control")){
        wglSwapIntervalEXT = (PFNWGLSWAPINTERVALEXTPROC) wglxGetProcAddress("wglSwapIntervalEXT");
        wglGetSwapIntervalEXT = (PFNWGLGETSWAPINTERVALEXTPROC) wglxGetProcAddress("wglGetSwapIntervalEXT");
	}

#elif defined(LINUX)

	GLX_ATI_pixel_format_float_supported = isWGLXExtensionSupported("GLX_ATI_pixel_format_float");

	if (GLX_ATI_render_texture_supported = isWGLXExtensionSupported("GLX_ATI_render_texture")){
		glXBindTexImageATI    = (PFNGLXBINDTEXIMAGEATIPROC)    wglxGetProcAddress("glXBindTexImageATI");
		glXReleaseTexImageATI = (PFNGLXRELEASETEXIMAGEATIPROC) wglxGetProcAddress("glXReleaseTexImageATI");
		glXDrawableAttribATI  = (PFNGLXDRAWABLEATTRIBATIPROC)  wglxGetProcAddress("glXDrawableAttribATI");
	}

#elif defined(__APPLE__)

#endif

	char *version = (char *) glGetString(GL_VERSION);
	GLMajorVersion = atoi(version);
	version = strchr(version, '.') + 1;
	GLMinorVersion = atoi(version);
	version = strchr(version, '.');
	if (version) GLReleaseVersion = atoi(version + 1);

	GL_1_1_supported = GLVER(1,1);

	if (GL_1_2_supported = GLVER(1,2)){
#ifdef GL_VERSION_1_2_PROTOTYPES
		glTexImage3D        = (PFNGLTEXIMAGE3DEXTPROC)        wglxGetProcAddress("glTexImage3D");
		glTexSubImage3D     = (PFNGLTEXSUBIMAGE3DPROC)        wglxGetProcAddress("glTexSubImage3D");
		glCopyTexSubImage3D = (PFNGLCOPYTEXSUBIMAGE3DPROC)    wglxGetProcAddress("glCopyTexSubImage3D");
		glDrawRangeElements = (PFNGLDRAWRANGEELEMENTSEXTPROC) wglxGetProcAddress("glDrawRangeElements");
		glBlendColor        = (PFNGLBLENDCOLOREXTPROC)        wglxGetProcAddress("glBlendColor");
		glBlendEquation     = (PFNGLBLENDEQUATIONEXTPROC)     wglxGetProcAddress("glBlendEquation");
#endif // GL_VERSION_1_2_PROTOTYPES
	}

	if (GL_1_3_supported = GLVER(1,3)){
#ifdef GL_VERSION_1_3_PROTOTYPES
	 	glActiveTexture = (PFNGLACTIVETEXTUREARBPROC) wglxGetProcAddress("glActiveTexture");
		glClientActiveTexture = (PFNGLCLIENTACTIVETEXTUREARBPROC) wglxGetProcAddress("glClientActiveTexture");
		glMultiTexCoord1d  = (PFNGLMULTITEXCOORD1DARBPROC)  wglxGetProcAddress("glMultiTexCoord1d");
		glMultiTexCoord1dv = (PFNGLMULTITEXCOORD1DVARBPROC) wglxGetProcAddress("glMultiTexCoord1dv");
		glMultiTexCoord1f  = (PFNGLMULTITEXCOORD1FARBPROC)  wglxGetProcAddress("glMultiTexCoord1f");
		glMultiTexCoord1fv = (PFNGLMULTITEXCOORD1FVARBPROC) wglxGetProcAddress("glMultiTexCoord1fv");
		glMultiTexCoord1i  = (PFNGLMULTITEXCOORD1IARBPROC)  wglxGetProcAddress("glMultiTexCoord1i");
		glMultiTexCoord1iv = (PFNGLMULTITEXCOORD1IVARBPROC) wglxGetProcAddress("glMultiTexCoord1iv");
		glMultiTexCoord1s  = (PFNGLMULTITEXCOORD1SARBPROC)  wglxGetProcAddress("glMultiTexCoord1s");
		glMultiTexCoord1sv = (PFNGLMULTITEXCOORD1SVARBPROC) wglxGetProcAddress("glMultiTexCoord1sv");
		glMultiTexCoord2d  = (PFNGLMULTITEXCOORD2DARBPROC)  wglxGetProcAddress("glMultiTexCoord2d");
		glMultiTexCoord2dv = (PFNGLMULTITEXCOORD2DVARBPROC) wglxGetProcAddress("glMultiTexCoord2dv");
		glMultiTexCoord2f  = (PFNGLMULTITEXCOORD2FARBPROC)  wglxGetProcAddress("glMultiTexCoord2f");
		glMultiTexCoord2fv = (PFNGLMULTITEXCOORD2FVARBPROC) wglxGetProcAddress("glMultiTexCoord2fv");
		glMultiTexCoord2i  = (PFNGLMULTITEXCOORD2IARBPROC)  wglxGetProcAddress("glMultiTexCoord2i");
		glMultiTexCoord2iv = (PFNGLMULTITEXCOORD2IVARBPROC) wglxGetProcAddress("glMultiTexCoord2iv");
		glMultiTexCoord2s  = (PFNGLMULTITEXCOORD2SARBPROC)  wglxGetProcAddress("glMultiTexCoord2s");
		glMultiTexCoord2sv = (PFNGLMULTITEXCOORD2SVARBPROC) wglxGetProcAddress("glMultiTexCoord2sv");
		glMultiTexCoord3d  = (PFNGLMULTITEXCOORD3DARBPROC)  wglxGetProcAddress("glMultiTexCoord3d");
		glMultiTexCoord3dv = (PFNGLMULTITEXCOORD3DVARBPROC) wglxGetProcAddress("glMultiTexCoord3dv");
		glMultiTexCoord3f  = (PFNGLMULTITEXCOORD3FARBPROC)  wglxGetProcAddress("glMultiTexCoord3f");
		glMultiTexCoord3fv = (PFNGLMULTITEXCOORD3FVARBPROC) wglxGetProcAddress("glMultiTexCoord3fv");
		glMultiTexCoord3i  = (PFNGLMULTITEXCOORD3IARBPROC)  wglxGetProcAddress("glMultiTexCoord3i");
		glMultiTexCoord3iv = (PFNGLMULTITEXCOORD3IVARBPROC) wglxGetProcAddress("glMultiTexCoord3iv");
		glMultiTexCoord3s  = (PFNGLMULTITEXCOORD3SARBPROC)  wglxGetProcAddress("glMultiTexCoord3s");
		glMultiTexCoord3sv = (PFNGLMULTITEXCOORD3SVARBPROC) wglxGetProcAddress("glMultiTexCoord3sv");
		glMultiTexCoord4d  = (PFNGLMULTITEXCOORD4DARBPROC)  wglxGetProcAddress("glMultiTexCoord4d");
		glMultiTexCoord4dv = (PFNGLMULTITEXCOORD4DVARBPROC) wglxGetProcAddress("glMultiTexCoord4dv");
		glMultiTexCoord4f  = (PFNGLMULTITEXCOORD4FARBPROC)  wglxGetProcAddress("glMultiTexCoord4f");
		glMultiTexCoord4fv = (PFNGLMULTITEXCOORD4FVARBPROC) wglxGetProcAddress("glMultiTexCoord4fv");
		glMultiTexCoord4i  = (PFNGLMULTITEXCOORD4IARBPROC)  wglxGetProcAddress("glMultiTexCoord4i");
		glMultiTexCoord4iv = (PFNGLMULTITEXCOORD4IVARBPROC) wglxGetProcAddress("glMultiTexCoord4iv");
		glMultiTexCoord4s  = (PFNGLMULTITEXCOORD4SARBPROC)  wglxGetProcAddress("glMultiTexCoord4s");
		glMultiTexCoord4sv = (PFNGLMULTITEXCOORD4SVARBPROC) wglxGetProcAddress("glMultiTexCoord4sv");

		glCompressedTexImage1D     = (PFNGLCOMPRESSEDTEXIMAGE1DARBPROC)    wglxGetProcAddress("glCompressedTexImage1D");
		glCompressedTexImage2D     = (PFNGLCOMPRESSEDTEXIMAGE2DARBPROC)    wglxGetProcAddress("glCompressedTexImage2D");
		glCompressedTexImage3D     = (PFNGLCOMPRESSEDTEXIMAGE3DARBPROC)    wglxGetProcAddress("glCompressedTexImage3D");
		glCompressedTexSubImage1D  = (PFNGLCOMPRESSEDTEXSUBIMAGE1DARBPROC) wglxGetProcAddress("glCompressedTexSubImage1D");
		glCompressedTexSubImage2D  = (PFNGLCOMPRESSEDTEXSUBIMAGE2DARBPROC) wglxGetProcAddress("glCompressedTexSubImage2D");
		glCompressedTexSubImage3D  = (PFNGLCOMPRESSEDTEXSUBIMAGE3DARBPROC) wglxGetProcAddress("glCompressedTexSubImage3D");
		glGetCompressedTexImage    = (PFNGLGETCOMPRESSEDTEXIMAGEARBPROC)   wglxGetProcAddress("glGetCompressedTexImage");

		glSampleCoverage = (PFNGLSAMPLECOVERAGEARBPROC) wglxGetProcAddress("glSampleCoverage");

		glLoadTransposeMatrixf = (PFNGLLOADTRANSPOSEMATRIXFARBPROC) wglxGetProcAddress("glLoadTransposeMatrixf");
		glLoadTransposeMatrixd = (PFNGLLOADTRANSPOSEMATRIXDARBPROC) wglxGetProcAddress("glLoadTransposeMatrixd");
		glMultTransposeMatrixf = (PFNGLMULTTRANSPOSEMATRIXFARBPROC) wglxGetProcAddress("glMultTransposeMatrixf");
		glMultTransposeMatrixd = (PFNGLMULTTRANSPOSEMATRIXDARBPROC) wglxGetProcAddress("glMultTransposeMatrixd");
#endif // GL_VERSION_1_3_PROTOTYPES
	}

	if (GL_1_4_supported = GLVER(1,4)){
#ifdef GL_VERSION_1_4_PROTOTYPES
		glFogCoordf  = (PFNGLFOGCOORDFEXTPROC)  wglxGetProcAddress("glFogCoordf");
		glFogCoordd  = (PFNGLFOGCOORDDEXTPROC)  wglxGetProcAddress("glFogCoordd");
		glFogCoordfv = (PFNGLFOGCOORDFVEXTPROC) wglxGetProcAddress("glFogCoordfv");
		glFogCoorddv = (PFNGLFOGCOORDDVEXTPROC) wglxGetProcAddress("glFogCoorddv");
		glFogCoordPointer = (PFNGLFOGCOORDPOINTEREXTPROC) wglxGetProcAddress("glFogCoordPointer");

		glMultiDrawArrays   = (PFNGLMULTIDRAWARRAYSEXTPROC)   wglxGetProcAddress("glMultiDrawArrays");
		glMultiDrawElements = (PFNGLMULTIDRAWELEMENTSEXTPROC) wglxGetProcAddress("glMultiDrawElements");

		glPointParameterf  = (PFNGLPOINTPARAMETERFARBPROC)  wglxGetProcAddress("glPointParameterf");
		glPointParameterfv = (PFNGLPOINTPARAMETERFVARBPROC) wglxGetProcAddress("glPointParameterfv");

		glSecondaryColor3f = (PFNGLSECONDARYCOLOR3FEXTPROC) wglxGetProcAddress("glSecondaryColor3f");
		glSecondaryColor3d = (PFNGLSECONDARYCOLOR3DEXTPROC) wglxGetProcAddress("glSecondaryColor3d");
		glSecondaryColor3b = (PFNGLSECONDARYCOLOR3BEXTPROC) wglxGetProcAddress("glSecondaryColor3b");
		glSecondaryColor3s = (PFNGLSECONDARYCOLOR3SEXTPROC) wglxGetProcAddress("glSecondaryColor3s");
		glSecondaryColor3i = (PFNGLSECONDARYCOLOR3IEXTPROC) wglxGetProcAddress("glSecondaryColor3i");
		glSecondaryColor3ub = (PFNGLSECONDARYCOLOR3UBEXTPROC) wglxGetProcAddress("glSecondaryColor3ub");
		glSecondaryColor3us = (PFNGLSECONDARYCOLOR3USEXTPROC) wglxGetProcAddress("glSecondaryColor3us");
		glSecondaryColor3ui = (PFNGLSECONDARYCOLOR3UIEXTPROC) wglxGetProcAddress("glSecondaryColor3ui");

		glSecondaryColor3fv = (PFNGLSECONDARYCOLOR3FVEXTPROC) wglxGetProcAddress("glSecondaryColor3fv");
		glSecondaryColor3dv = (PFNGLSECONDARYCOLOR3DVEXTPROC) wglxGetProcAddress("glSecondaryColor3dv");
		glSecondaryColor3bv = (PFNGLSECONDARYCOLOR3BVEXTPROC) wglxGetProcAddress("glSecondaryColor3bv");
		glSecondaryColor3sv = (PFNGLSECONDARYCOLOR3SVEXTPROC) wglxGetProcAddress("glSecondaryColor3sv");
		glSecondaryColor3iv = (PFNGLSECONDARYCOLOR3IVEXTPROC) wglxGetProcAddress("glSecondaryColor3iv");
		glSecondaryColor3ubv = (PFNGLSECONDARYCOLOR3UBVEXTPROC) wglxGetProcAddress("glSecondaryColor3ubv");
		glSecondaryColor3usv = (PFNGLSECONDARYCOLOR3USVEXTPROC) wglxGetProcAddress("glSecondaryColor3usv");
		glSecondaryColor3uiv = (PFNGLSECONDARYCOLOR3UIVEXTPROC) wglxGetProcAddress("glSecondaryColor3uiv");

		glSecondaryColorPointer = (PFNGLSECONDARYCOLORPOINTEREXTPROC) wglxGetProcAddress("glSecondaryColorPointer");

		glBlendFuncSeparate = (PFNGLBLENDFUNCSEPARATEEXTPROC) wglxGetProcAddress("glBlendFuncSeparate");

		glWindowPos2d  = (PFNGLWINDOWPOS2DARBPROC)  wglxGetProcAddress("glWindowPos2d");
		glWindowPos2f  = (PFNGLWINDOWPOS2FARBPROC)  wglxGetProcAddress("glWindowPos2f");
		glWindowPos2i  = (PFNGLWINDOWPOS2IARBPROC)  wglxGetProcAddress("glWindowPos2i");
		glWindowPos2s  = (PFNGLWINDOWPOS2SARBPROC)  wglxGetProcAddress("glWindowPos2s");
		glWindowPos2iv = (PFNGLWINDOWPOS2IVARBPROC) wglxGetProcAddress("glWindowPos2iv");
		glWindowPos2sv = (PFNGLWINDOWPOS2SVARBPROC) wglxGetProcAddress("glWindowPos2sv");
		glWindowPos2fv = (PFNGLWINDOWPOS2FVARBPROC) wglxGetProcAddress("glWindowPos2fv");
		glWindowPos2dv = (PFNGLWINDOWPOS2DVARBPROC) wglxGetProcAddress("glWindowPos2dv");
		glWindowPos3i  = (PFNGLWINDOWPOS3IARBPROC)  wglxGetProcAddress("glWindowPos3i");
		glWindowPos3s  = (PFNGLWINDOWPOS3SARBPROC)  wglxGetProcAddress("glWindowPos3s");
		glWindowPos3f  = (PFNGLWINDOWPOS3FARBPROC)  wglxGetProcAddress("glWindowPos3f");
		glWindowPos3d  = (PFNGLWINDOWPOS3DARBPROC)  wglxGetProcAddress("glWindowPos3d");
		glWindowPos3iv = (PFNGLWINDOWPOS3IVARBPROC) wglxGetProcAddress("glWindowPos3iv");
		glWindowPos3sv = (PFNGLWINDOWPOS3SVARBPROC) wglxGetProcAddress("glWindowPos3sv");
		glWindowPos3fv = (PFNGLWINDOWPOS3FVARBPROC) wglxGetProcAddress("glWindowPos3fv");
		glWindowPos3dv = (PFNGLWINDOWPOS3DVARBPROC) wglxGetProcAddress("glWindowPos3dv");
#endif // GL_VERSION_1_4_PROTOTYPES
	}

	if (GL_1_5_supported = GLVER(1,5)){
#ifdef GL_VERSION_1_5_PROTOTYPES
		glBindBuffer    = (PFNGLBINDBUFFERARBPROC)    wglxGetProcAddress("glBindBuffer");
		glDeleteBuffers = (PFNGLDELETEBUFFERSARBPROC) wglxGetProcAddress("glDeleteBuffers");
		glGenBuffers    = (PFNGLGENBUFFERSARBPROC)    wglxGetProcAddress("glGenBuffers");
		glIsBuffer      = (PFNGLISBUFFERARBPROC)      wglxGetProcAddress("glIsBuffer");
		glBufferData       = (PFNGLBUFFERDATAARBPROC)       wglxGetProcAddress("glBufferData");
		glBufferSubData    = (PFNGLBUFFERSUBDATAARBPROC)    wglxGetProcAddress("glBufferSubData");
		glGetBufferSubData = (PFNGLGETBUFFERSUBDATAARBPROC) wglxGetProcAddress("glGetBufferSubData");
		glMapBuffer   = (PFNGLMAPBUFFERARBPROC)   wglxGetProcAddress("glMapBuffer");
		glUnmapBuffer = (PFNGLUNMAPBUFFERARBPROC) wglxGetProcAddress("glUnmapBuffer");
		glGetBufferParameteriv = (PFNGLGETBUFFERPARAMETERIVARBPROC) wglxGetProcAddress("glGetBufferParameteriv");
		glGetBufferPointerv    = (PFNGLGETBUFFERPOINTERVARBPROC)    wglxGetProcAddress("glGetBufferPointerv");

		glGenQueries    = (PFNGLGENQUERIESARBPROC)    wglxGetProcAddress("glGenQueries");
		glDeleteQueries = (PFNGLDELETEQUERIESARBPROC) wglxGetProcAddress("glDeleteQueries");
		glIsQuery       = (PFNGLISQUERYARBPROC)       wglxGetProcAddress("glIsQuery");
		glBeginQuery    = (PFNGLBEGINQUERYARBPROC)    wglxGetProcAddress("glBeginQuery");
		glEndQuery      = (PFNGLENDQUERYARBPROC)      wglxGetProcAddress("glEndQuery");
		glGetQueryiv    = (PFNGLGETQUERYIVARBPROC)    wglxGetProcAddress("glGetQueryiv");
		glGetQueryObjectiv  = (PFNGLGETQUERYOBJECTIVARBPROC)  wglxGetProcAddress("glGetQueryObjectiv");
		glGetQueryObjectuiv = (PFNGLGETQUERYOBJECTUIVARBPROC) wglxGetProcAddress("glGetQueryObjectuiv");
#endif // GL_VERSION_1_5_PROTOTYPES
	}

	if (GL_2_0_supported = GLVER(2,0)){
#ifdef GL_VERSION_2_0_PROTOTYPES
		glDeleteProgram   = (PFNGLDELETEOBJECTARBPROC)        wglxGetProcAddress("glDeleteProgram");
		glDeleteShader    = (PFNGLDELETEOBJECTARBPROC)        wglxGetProcAddress("glDeleteShader");
//		glGetHandle       = (PFNGLGETHANDLEARBPROC)           wglxGetProcAddress("glGetHandle");
		glDetachShader    = (PFNGLDETACHOBJECTARBPROC)        wglxGetProcAddress("glDetachShader");
		glCreateShader    = (PFNGLCREATESHADEROBJECTARBPROC)  wglxGetProcAddress("glCreateShader");
		glShaderSource    = (PFNGLSHADERSOURCEARBPROC)        wglxGetProcAddress("glShaderSource");
		glCompileShader   = (PFNGLCOMPILESHADERARBPROC)       wglxGetProcAddress("glCompileShader");
		glCreateProgram   = (PFNGLCREATEPROGRAMOBJECTARBPROC) wglxGetProcAddress("glCreateProgram");
		glAttachShader    = (PFNGLATTACHOBJECTARBPROC)        wglxGetProcAddress("glAttachShader");
		glLinkProgram     = (PFNGLLINKPROGRAMARBPROC)         wglxGetProcAddress("glLinkProgram");
		glUseProgram      = (PFNGLUSEPROGRAMOBJECTARBPROC)    wglxGetProcAddress("glUseProgram");
		glValidateProgram = (PFNGLVALIDATEPROGRAMARBPROC)     wglxGetProcAddress("glValidateProgram");
		glUniform1f  = (PFNGLUNIFORM1FARBPROC)  wglxGetProcAddress("glUniform1f");
		glUniform2f  = (PFNGLUNIFORM2FARBPROC)  wglxGetProcAddress("glUniform2f");
		glUniform3f  = (PFNGLUNIFORM3FARBPROC)  wglxGetProcAddress("glUniform3f");
		glUniform4f  = (PFNGLUNIFORM4FARBPROC)  wglxGetProcAddress("glUniform4f");
		glUniform1i  = (PFNGLUNIFORM1IARBPROC)  wglxGetProcAddress("glUniform1i");
		glUniform2i  = (PFNGLUNIFORM2IARBPROC)  wglxGetProcAddress("glUniform2i");
		glUniform3i  = (PFNGLUNIFORM3IARBPROC)  wglxGetProcAddress("glUniform3i");
		glUniform4i  = (PFNGLUNIFORM4IARBPROC)  wglxGetProcAddress("glUniform4i");
		glUniform1fv = (PFNGLUNIFORM1FVARBPROC) wglxGetProcAddress("glUniform1fv");
		glUniform2fv = (PFNGLUNIFORM2FVARBPROC) wglxGetProcAddress("glUniform2fv");
		glUniform3fv = (PFNGLUNIFORM3FVARBPROC) wglxGetProcAddress("glUniform3fv");
		glUniform4fv = (PFNGLUNIFORM4FVARBPROC) wglxGetProcAddress("glUniform4fv");
		glUniform1iv = (PFNGLUNIFORM1IVARBPROC) wglxGetProcAddress("glUniform1iv");
		glUniform2iv = (PFNGLUNIFORM2IVARBPROC) wglxGetProcAddress("glUniform2iv");
		glUniform3iv = (PFNGLUNIFORM3IVARBPROC) wglxGetProcAddress("glUniform3iv");
		glUniform4iv = (PFNGLUNIFORM4IVARBPROC) wglxGetProcAddress("glUniform4iv");
		glUniformMatrix2fv = (PFNGLUNIFORMMATRIX2FVARBPROC) wglxGetProcAddress("glUniformMatrix2fv");
		glUniformMatrix3fv = (PFNGLUNIFORMMATRIX3FVARBPROC) wglxGetProcAddress("glUniformMatrix3fv");
		glUniformMatrix4fv = (PFNGLUNIFORMMATRIX4FVARBPROC) wglxGetProcAddress("glUniformMatrix4fv");
//		glGetObjectParameterfv = (PFNGLGETOBJECTPARAMETERFVARBPROC) wglxGetProcAddress("glGetObjectParameterfv");
//		glGetObjectParameteriv = (PFNGLGETOBJECTPARAMETERIVARBPROC) wglxGetProcAddress("glGetObjectParameteriv");
//		glGetInfoLog           = (PFNGLGETINFOLOGARBPROC)           wglxGetProcAddress("glGetInfoLog");
//		glGetAttachedObjects   = (PFNGLGETATTACHEDOBJECTSARBPROC)   wglxGetProcAddress("glGetAttachedObjects");
		glGetUniformLocation   = (PFNGLGETUNIFORMLOCATIONARBPROC)   wglxGetProcAddress("glGetUniformLocation");
		glGetActiveUniform     = (PFNGLGETACTIVEUNIFORMARBPROC)     wglxGetProcAddress("glGetActiveUniform");
		glGetUniformfv         = (PFNGLGETUNIFORMFVARBPROC)         wglxGetProcAddress("glGetUniformfv");
		glGetUniformiv         = (PFNGLGETUNIFORMIVARBPROC)         wglxGetProcAddress("glGetUniformiv");
		glGetShaderSource      = (PFNGLGETSHADERSOURCEARBPROC)      wglxGetProcAddress("glGetShaderSource");

		glBindAttribLocation = (PFNGLBINDATTRIBLOCATIONARBPROC) wglxGetProcAddress("glBindAttribLocation");
		glGetActiveAttrib    = (PFNGLGETACTIVEATTRIBARBPROC)    wglxGetProcAddress("glGetActiveAttrib");
		glGetAttribLocation  = (PFNGLGETATTRIBLOCATIONARBPROC)  wglxGetProcAddress("glGetAttribLocation");

		glVertexAttrib1s = (PFNGLVERTEXATTRIB1SARBPROC) wglxGetProcAddress("glVertexAttrib1s");
		glVertexAttrib1f = (PFNGLVERTEXATTRIB1FARBPROC) wglxGetProcAddress("glVertexAttrib1f");
		glVertexAttrib1d = (PFNGLVERTEXATTRIB1DARBPROC) wglxGetProcAddress("glVertexAttrib1d");
		glVertexAttrib2s = (PFNGLVERTEXATTRIB2SARBPROC) wglxGetProcAddress("glVertexAttrib2s");
		glVertexAttrib2f = (PFNGLVERTEXATTRIB2FARBPROC) wglxGetProcAddress("glVertexAttrib2f");
		glVertexAttrib2d = (PFNGLVERTEXATTRIB2DARBPROC) wglxGetProcAddress("glVertexAttrib2d");
		glVertexAttrib3s = (PFNGLVERTEXATTRIB3SARBPROC) wglxGetProcAddress("glVertexAttrib3s");
		glVertexAttrib3f = (PFNGLVERTEXATTRIB3FARBPROC) wglxGetProcAddress("glVertexAttrib3f");
		glVertexAttrib3d = (PFNGLVERTEXATTRIB3DARBPROC) wglxGetProcAddress("glVertexAttrib3d");
		glVertexAttrib4s = (PFNGLVERTEXATTRIB4SARBPROC) wglxGetProcAddress("glVertexAttrib4s");
		glVertexAttrib4f = (PFNGLVERTEXATTRIB4FARBPROC) wglxGetProcAddress("glVertexAttrib4f");
		glVertexAttrib4d = (PFNGLVERTEXATTRIB4DARBPROC) wglxGetProcAddress("glVertexAttrib4d");
		glVertexAttrib4Nub = (PFNGLVERTEXATTRIB4NUBARBPROC) wglxGetProcAddress("glVertexAttrib4Nub");
		glVertexAttrib1sv  = (PFNGLVERTEXATTRIB1SVARBPROC)  wglxGetProcAddress("glVertexAttrib1sv");
		glVertexAttrib1fv  = (PFNGLVERTEXATTRIB1FVARBPROC)  wglxGetProcAddress("glVertexAttrib1fv");
		glVertexAttrib1dv  = (PFNGLVERTEXATTRIB1DVARBPROC)  wglxGetProcAddress("glVertexAttrib1dv");
		glVertexAttrib2sv  = (PFNGLVERTEXATTRIB2SVARBPROC)  wglxGetProcAddress("glVertexAttrib2sv");
		glVertexAttrib2fv  = (PFNGLVERTEXATTRIB2FVARBPROC)  wglxGetProcAddress("glVertexAttrib2fv");
		glVertexAttrib2dv  = (PFNGLVERTEXATTRIB2DVARBPROC)  wglxGetProcAddress("glVertexAttrib2dv");
		glVertexAttrib3sv  = (PFNGLVERTEXATTRIB3SVARBPROC)  wglxGetProcAddress("glVertexAttrib3sv");
		glVertexAttrib3fv  = (PFNGLVERTEXATTRIB3FVARBPROC)  wglxGetProcAddress("glVertexAttrib3fv");
		glVertexAttrib3dv  = (PFNGLVERTEXATTRIB3DVARBPROC)  wglxGetProcAddress("glVertexAttrib3dv");
		glVertexAttrib4bv  = (PFNGLVERTEXATTRIB4BVARBPROC)  wglxGetProcAddress("glVertexAttrib4bv");
		glVertexAttrib4sv  = (PFNGLVERTEXATTRIB4SVARBPROC)  wglxGetProcAddress("glVertexAttrib4sv");
		glVertexAttrib4iv  = (PFNGLVERTEXATTRIB4IVARBPROC)  wglxGetProcAddress("glVertexAttrib4iv");
		glVertexAttrib4ubv = (PFNGLVERTEXATTRIB4UBVARBPROC) wglxGetProcAddress("glVertexAttrib4ubv");
		glVertexAttrib4usv = (PFNGLVERTEXATTRIB4USVARBPROC) wglxGetProcAddress("glVertexAttrib4usv");
		glVertexAttrib4uiv = (PFNGLVERTEXATTRIB4UIVARBPROC) wglxGetProcAddress("glVertexAttrib4uiv");
		glVertexAttrib4fv  = (PFNGLVERTEXATTRIB4FVARBPROC)  wglxGetProcAddress("glVertexAttrib4fv");
		glVertexAttrib4dv  = (PFNGLVERTEXATTRIB4DVARBPROC)  wglxGetProcAddress("glVertexAttrib4dv");
		glVertexAttrib4Nbv = (PFNGLVERTEXATTRIB4NBVARBPROC) wglxGetProcAddress("glVertexAttrib4Nbv");
		glVertexAttrib4Nsv = (PFNGLVERTEXATTRIB4NSVARBPROC) wglxGetProcAddress("glVertexAttrib4Nsv");
		glVertexAttrib4Niv = (PFNGLVERTEXATTRIB4NIVARBPROC) wglxGetProcAddress("glVertexAttrib4Niv");
		glVertexAttrib4Nubv = (PFNGLVERTEXATTRIB4NUBVARBPROC) wglxGetProcAddress("glVertexAttrib4Nubv");
		glVertexAttrib4Nusv = (PFNGLVERTEXATTRIB4NUSVARBPROC) wglxGetProcAddress("glVertexAttrib4Nusv");
		glVertexAttrib4Nuiv = (PFNGLVERTEXATTRIB4NUIVARBPROC) wglxGetProcAddress("glVertexAttrib4Nuiv");
		glVertexAttribPointer = (PFNGLVERTEXATTRIBPOINTERARBPROC) wglxGetProcAddress("glVertexAttribPointer");
		glEnableVertexAttribArray = (PFNGLENABLEVERTEXATTRIBARRAYARBPROC) wglxGetProcAddress("glEnableVertexAttribArray");
		glDisableVertexAttribArray = (PFNGLDISABLEVERTEXATTRIBARRAYARBPROC) wglxGetProcAddress("glDisableVertexAttribArray");
		glGetVertexAttribdv = (PFNGLGETVERTEXATTRIBDVARBPROC) wglxGetProcAddress("glGetVertexAttribdv");
		glGetVertexAttribfv = (PFNGLGETVERTEXATTRIBFVARBPROC) wglxGetProcAddress("glGetVertexAttribfv");
		glGetVertexAttribiv = (PFNGLGETVERTEXATTRIBIVARBPROC) wglxGetProcAddress("glGetVertexAttribiv");
		glGetVertexAttribPointerv = (PFNGLGETVERTEXATTRIBPOINTERVARBPROC) wglxGetProcAddress("glGetVertexAttribPointerv");

		glDrawBuffers = (PFNGLDRAWBUFFERSARBPROC) wglxGetProcAddress("glDrawBuffers");

		glStencilOpSeparate   = (PFNGLSTENCILOPSEPARATEPROC)   wglxGetProcAddress("glStencilOpSeparate");
		glStencilFuncSeparate = (PFNGLSTENCILFUNCSEPARATEPROC) wglxGetProcAddress("glStencilFuncSeparate");
		glStencilMaskSeparate = (PFNGLSTENCILMASKSEPARATEPROC) wglxGetProcAddress("glStencilMaskSeparate");

		glBlendEquationSeparate = (PFNGLBLENDEQUATIONSEPARATEPROC) wglxGetProcAddress("glBlendEquationSeparate");

#endif // GL_VERSION_2_0_PROTOTYPES
	}
}
