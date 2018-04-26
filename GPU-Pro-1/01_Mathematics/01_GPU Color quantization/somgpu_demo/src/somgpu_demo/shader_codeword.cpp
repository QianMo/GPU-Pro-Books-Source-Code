#include <stdio.h>
#include <stdlib.h>

#include <GL/glew.h>
#include <GL/glut.h>

#include <cg/cgGL.h>
#include <cg/cg.h>

#include "g_vector.h"
#include "g_pfm.h"

#include "shader_codeword.h"


void gpu_codebook2codeword( GLuint srcid, const GPfm &srcbook, GLuint codewordid )
{
  GLuint codeword_fbo;
  int srcw, srch;

  glBindTexture( GL_TEXTURE_2D, srcid );
  glGetTexLevelParameteriv( GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &srcw );
  glGetTexLevelParameteriv( GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &srch );

  {
    glGenFramebuffersEXT( 1, &codeword_fbo );
    glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, codeword_fbo );
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, codewordid, 0);
    glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, 0 );
  }

  glPushAttrib( GL_VIEWPORT_BIT | GL_COLOR_BUFFER_BIT );
  glViewport( 0,0, srcw, srch );
  glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D( 0,1,0,1 );
  glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

  glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, codeword_fbo );
  glClearColor(0,0,0,0);
  glClear( GL_COLOR_BUFFER_BIT );

  cgfl_begin_codeword( srcbook, srcid );
    glBegin(GL_QUADS);
      glTexCoord2f( 0, 0 );    glVertex2f( 0, 0 );
      glTexCoord2f( 1, 0 );    glVertex2f( 1, 0 );
      glTexCoord2f( 1, 1 );    glVertex2f( 1, 1 );
      glTexCoord2f( 0, 1 );    glVertex2f( 0, 1 );
    glEnd();
  cgfl_end_codeword();

  glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, 0 );
  glPopAttrib();

  glDeleteFramebuffersEXT( 1, &codeword_fbo );
}

void gpu_subidx_projection( 
  GLuint srcid, GLuint desid, 
  GLuint ucodewordid, GLuint vcodewordid, 
  GLuint srcbookid )
{
  GLenum mrt[2];
  GLuint codeword_subidx_fbo;
  int srcw, srch;

  glBindTexture( GL_TEXTURE_2D, srcid );
  glGetTexLevelParameteriv( GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &srcw );
  glGetTexLevelParameteriv( GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &srch );

  // multiple render target
  {
    mrt[0] = GL_COLOR_ATTACHMENT0_EXT;
    mrt[1] = GL_COLOR_ATTACHMENT1_EXT;

    glGenFramebuffersEXT( 1, &codeword_subidx_fbo );
    glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, codeword_subidx_fbo );
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, mrt[0], GL_TEXTURE_2D, vcodewordid, 0);
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, mrt[1], GL_TEXTURE_2D, desid, 0);
    glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, 0 );
  }

  glPushAttrib( GL_VIEWPORT_BIT | GL_COLOR_BUFFER_BIT );
  glViewport( 0,0, srcw, srch );
  glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D( 0,1,0,1 );
  glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

  glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, codeword_subidx_fbo );
  glDrawBuffers( 2, mrt );
  glClearColor(0,0,0,0);
  glClear( GL_COLOR_BUFFER_BIT );

  cgfl_begin_codeword_subidx( srcid, ucodewordid, srcbookid );
    glBegin(GL_QUADS);
      glTexCoord2f( 0, 0 );  glVertex2f( 0, 0 );
      glTexCoord2f( 1, 0 );  glVertex2f( 1, 0 );
      glTexCoord2f( 1, 1 );  glVertex2f( 1, 1 );
      glTexCoord2f( 0, 1 );  glVertex2f( 0, 1 );
    glEnd();
  cgfl_end_codeword_subidx();

  glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, 0 );
  glPopAttrib();

  glDeleteFramebuffersEXT( 1, &codeword_subidx_fbo );
}

CGcontext SCWD_shader = NULL;
  CGprofile SCWD_fprofile;

  CGprogram   SCWD_ucwd_program = NULL;
  CGparameter SCWD_ucwd_src;
  CGparameter *SCWD_ucwd_codebook;
  
  CGprogram   SCWD_vcwd_program = NULL;
  CGparameter SCWD_vcwd_src;
  CGparameter SCWD_vcwd_codeword;
  CGparameter SCWD_vcwd_codebook;

void cgErrorCallbackx()
{
  CGerror lastError = cgGetError();
  if(lastError)
  {
    printf("%s\n\n", cgGetErrorString(lastError));
    printf("%s\n", cgGetLastListing(SCWD_shader));
    printf("Cg error, exiting...\n");
    exit(0);
  }
}

void shader_codeword_prepare( int codebook_w, int codebook_h )
{
  char program[4096];
  {
    char *tmp;
    tmp = (char*) freadall("shader_codeword.cg");
    sprintf( program, "#define codebook_w %i\n#define codebook_h %i\n%s", codebook_w, codebook_h, tmp );
    free(tmp);
  }

  int i, j;
  char str[256];

  cgSetErrorCallback(cgErrorCallbackx);

  if( SCWD_shader )
  {
    cgDestroyProgram( SCWD_ucwd_program );
    cgDestroyContext( SCWD_shader );
  }

  SCWD_shader = cgCreateContext();
    SCWD_fprofile = cgGLGetLatestProfile( CG_GL_FRAGMENT );
    cgGLSetOptimalOptions( SCWD_fprofile );
    SCWD_ucwd_program = cgCreateProgram( SCWD_shader, CG_SOURCE, program, SCWD_fprofile, "cgfl_fp_som_codeword", 0 );
    cgGLLoadProgram( SCWD_ucwd_program );
    SCWD_ucwd_src = cgGetNamedParameter( SCWD_ucwd_program, "intex" );
    SCWD_ucwd_codebook = new CGparameter[codebook_w*codebook_h];
    for( i=0; i<codebook_h; i++ )
    {
      for ( j=0; j<codebook_w; j++ )
      {
        sprintf( str, "codebook[%i][%i]", i, j );
        SCWD_ucwd_codebook[i*codebook_w+j] = cgGetNamedParameter( SCWD_ucwd_program, str );
      }
    }

  SCWD_shader = cgCreateContext();
    SCWD_fprofile = cgGLGetLatestProfile( CG_GL_FRAGMENT );
    cgGLSetOptimalOptions( SCWD_fprofile );
    SCWD_vcwd_program = cgCreateProgram( SCWD_shader, CG_SOURCE, program, SCWD_fprofile, "cgfl_fp_som_codeword_subidx", 0 );
    cgGLLoadProgram( SCWD_vcwd_program );
    SCWD_vcwd_src = cgGetNamedParameter( SCWD_vcwd_program, "intex" );
    SCWD_vcwd_codeword = cgGetNamedParameter( SCWD_vcwd_program, "codewordtex" );
    SCWD_vcwd_codebook = cgGetNamedParameter( SCWD_vcwd_program, "codebooktex" );
}

void cgfl_begin_codeword( const GPfm &codebook, const GLuint texid )
{
  cgGLEnableProfile( SCWD_fprofile );    
  cgGLBindProgram( SCWD_ucwd_program );

  int i;
  for( i=0; i<codebook.w*codebook.h; i++ )
    cgSetParameter3fv( SCWD_ucwd_codebook[i], (float*)&codebook.fm[i] );

  cgGLSetTextureParameter( SCWD_ucwd_src, texid );
  cgGLEnableTextureParameter( SCWD_ucwd_src );
}
void cgfl_end_codeword()
{
  cgGLDisableTextureParameter( SCWD_ucwd_src );
  cgGLDisableProfile( SCWD_fprofile );
}

void cgfl_begin_codeword_subidx( const GLuint texid, 
                                 const GLuint codeword_texid, const GLuint codebook_texid )
{
  cgGLEnableProfile( SCWD_fprofile );    
  cgGLBindProgram( SCWD_vcwd_program );

  cgGLSetTextureParameter( SCWD_vcwd_src, texid );
  cgGLSetTextureParameter( SCWD_vcwd_codeword, codeword_texid );
  cgGLSetTextureParameter( SCWD_vcwd_codebook, codebook_texid );
  cgGLEnableTextureParameter( SCWD_vcwd_src );
  cgGLEnableTextureParameter( SCWD_vcwd_codeword );  
  cgGLEnableTextureParameter( SCWD_vcwd_codebook );
}
void cgfl_end_codeword_subidx()
{
  cgGLDisableTextureParameter( SCWD_vcwd_src );
  cgGLDisableTextureParameter( SCWD_vcwd_codeword );
  cgGLDisableTextureParameter( SCWD_vcwd_codebook );
  cgGLDisableProfile( SCWD_fprofile );
}


