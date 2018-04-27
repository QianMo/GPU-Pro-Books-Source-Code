#include <stdio.h>
#include <stdlib.h>

#include <GL/glew.h>
#include <cg/cgGL.h>
#include <cg/cg.h>

#include "g_pfm.h"

#include "g_vector.h"

CGcontext CG_shader;
CGprofile CG_fprofile;

CGprogram CG_cnn_program;
  CGparameter CG_cnn_utex;
  CGparameter CG_cnn_xtex;
  CGparameter CG_cnn_T_ad[25];
  CGparameter CG_cnn_cnn_step;

CGprogram CG_ledge;
  CGparameter CG_ledge_utex;
  CGparameter CG_ledge_xtex;
  CGparameter CG_ledge_T_ad[25];
  CGparameter CG_ledge_cnn_step;

CGprogram CG_redge;
  CGparameter CG_redge_utex;
  CGparameter CG_redge_xtex;
  CGparameter CG_redge_T_ad[25];
  CGparameter CG_redge_cnn_step;

CGprogram CG_unfold;
  CGparameter CG_unfold_xtex;
  CGparameter CG_unfold_xw;

void cginit( const float *a, const float *b, float cnn_step )
{
  int i;

  for( i=0; i<25; i++ )
  {
    cgGLSetParameter1f( CG_cnn_T_ad[i],   a[i] );
    cgGLSetParameter1f( CG_ledge_T_ad[i], a[i] );
    cgGLSetParameter1f( CG_redge_T_ad[i], a[i] );
  }

  cgGLSetParameter1f( CG_cnn_cnn_step, cnn_step );
  cgGLSetParameter1f( CG_ledge_cnn_step, cnn_step );
  cgGLSetParameter1f( CG_redge_cnn_step, cnn_step );
}


void cgcnn_begin( GLuint usrc, GLuint xsrc )
{
  cgGLEnableProfile( CG_fprofile );    
  cgGLBindProgram( CG_cnn_program );

  cgGLSetTextureParameter( CG_cnn_utex, usrc );
  cgGLSetTextureParameter( CG_cnn_xtex, xsrc );
  cgGLEnableTextureParameter(CG_cnn_utex);
  cgGLEnableTextureParameter(CG_cnn_xtex);
}
void cgcnn_end()
{
  cgGLDisableTextureParameter(CG_cnn_utex);
  cgGLDisableTextureParameter(CG_cnn_xtex);
  cgGLDisableProfile( CG_fprofile );
}

void cgredge_begin( GLuint usrc, GLuint xsrc )
{
  cgGLEnableProfile( CG_fprofile );    
  cgGLBindProgram( CG_redge );
  cgGLSetTextureParameter( CG_redge_utex, usrc );
  cgGLSetTextureParameter( CG_redge_xtex, xsrc );
  cgGLEnableTextureParameter(CG_redge_utex);
  cgGLEnableTextureParameter(CG_redge_xtex);
}
void cgredge_end()
{
  cgGLDisableTextureParameter(CG_redge_utex);
  cgGLDisableTextureParameter(CG_redge_xtex);
  cgGLDisableProfile( CG_fprofile );
}

void cgledge_begin( GLuint usrc, GLuint xsrc )
{
  cgGLEnableProfile( CG_fprofile );    
  cgGLBindProgram( CG_ledge );
  cgGLSetTextureParameter( CG_ledge_utex, usrc );
  cgGLSetTextureParameter( CG_ledge_xtex, xsrc );
  cgGLEnableTextureParameter(CG_ledge_utex);
  cgGLEnableTextureParameter(CG_ledge_xtex);
}
void cgledge_end()
{
  cgGLDisableTextureParameter(CG_ledge_utex);
  cgGLDisableTextureParameter(CG_ledge_xtex);
  cgGLDisableProfile( CG_fprofile );
}

void cgunfold_begin( GLuint xsrc )
{
  int xw;
  glBindTexture(GL_TEXTURE_RECTANGLE_ARB, xsrc );
  glGetTexLevelParameteriv( GL_TEXTURE_RECTANGLE_ARB, 0, GL_TEXTURE_WIDTH, &xw );

  cgGLSetParameter1f( CG_unfold_xw, float(xw-4) );
  cgGLEnableProfile( CG_fprofile );    
  cgGLBindProgram( CG_unfold );
  cgGLSetTextureParameter( CG_unfold_xtex, xsrc );
  cgGLEnableTextureParameter(CG_unfold_xtex);
}
void cgunfold_end()
{
  cgGLDisableTextureParameter(CG_unfold_xtex);
  cgGLDisableProfile( CG_fprofile );
}

void cgErrorCallback()
{
  CGerror lastError = cgGetError();
  if(lastError)
  {
    printf("%s\n\n", cgGetErrorString(lastError));
    printf("%s\n", cgGetLastListing(CG_shader));
    printf("Cg error, exiting...\n");
    exit(0);
  }
}

void cgfl_prepare( const char *exepath )
{
  char cgpath[] = "shader.cg";

  int i;
  char str[256];
  char spath[256];
  GPath gp = parse_spath(exepath);
  sprintf( spath, "%s%s", gp.dname, cgpath );
  if( !fexist(spath) )
  {
    gp = parse_spath(exepath);
    sprintf( spath, "%s../%s", gp.dname, cgpath );
    if( !fexist(spath) )
    {
      printf( "[Error] cgfl_prepare(), Cg script file %s not found.\n", cgpath );
      exit(-1);
    }
  }

  cgSetErrorCallback(cgErrorCallback);

  CG_shader = cgCreateContext();
  CG_fprofile = cgGLGetLatestProfile(CG_GL_FRAGMENT);
  cgGLSetOptimalOptions(CG_fprofile);


  CG_cnn_program = cgCreateProgramFromFile( CG_shader, CG_SOURCE, spath , CG_fprofile, NULL, 0 );
    cgGLLoadProgram(CG_cnn_program);
      CG_cnn_utex = cgGetNamedParameter( CG_cnn_program, "utex" );
      CG_cnn_xtex = cgGetNamedParameter( CG_cnn_program, "xtex" );
      for( i=0; i<25; i++ )
      {
        sprintf( str, "T[%d]", i );
        CG_cnn_T_ad[i] = cgGetNamedParameter( CG_cnn_program, str );
      }
      CG_cnn_cnn_step = cgGetNamedParameter( CG_cnn_program, "cnn_step" );


  CG_ledge = cgCreateProgramFromFile( CG_shader, CG_SOURCE, spath , CG_fprofile, "cnn_ledge", 0 );
    cgGLLoadProgram(CG_ledge);
      CG_ledge_utex = cgGetNamedParameter( CG_ledge, "utex" );
      CG_ledge_xtex = cgGetNamedParameter( CG_ledge, "xtex" );
      for ( i=0; i<25; i++ )
      {
        sprintf( str, "T[%d]", i );
        CG_ledge_T_ad[i] = cgGetNamedParameter( CG_ledge, str );
      }
      CG_ledge_cnn_step = cgGetNamedParameter( CG_ledge, "cnn_step" );


  CG_redge = cgCreateProgramFromFile( CG_shader, CG_SOURCE, spath , CG_fprofile, "cnn_redge", 0 );
    cgGLLoadProgram(CG_redge);
      CG_redge_utex = cgGetNamedParameter( CG_redge, "utex" );
      CG_redge_xtex = cgGetNamedParameter( CG_redge, "xtex" );
      for( i=0; i<25; i++ )
      {
        sprintf( str, "T[%d]", i );
        CG_redge_T_ad[i] = cgGetNamedParameter( CG_redge, str );
      }
      CG_redge_cnn_step = cgGetNamedParameter( CG_redge, "cnn_step" );

  CG_unfold = cgCreateProgramFromFile( CG_shader, CG_SOURCE, spath , CG_fprofile, "cnn_unfold", 0 );
    cgGLLoadProgram(CG_unfold);
      CG_unfold_xtex = cgGetNamedParameter( CG_unfold, "xtex" );
      CG_unfold_xw = cgGetNamedParameter( CG_unfold, "xw" );
}
