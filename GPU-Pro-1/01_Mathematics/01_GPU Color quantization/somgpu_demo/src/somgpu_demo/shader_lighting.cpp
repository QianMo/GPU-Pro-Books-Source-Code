#include <stdio.h>
#include <stdlib.h>

#include <GL/glew.h>
#include <GL/glut.h>

#include <cg/cgGL.h>
#include <cg/cg.h>

#include "g_vector.h"

CGcontext SL_shader;
  CGprofile SL_vprofile;
  CGprofile SL_fprofile;

  CGprogram   SL_flv_program;

  CGprogram   SL_flp_program;
  CGparameter SL_fl_lpos;
  CGparameter SL_fl_kd;
  CGparameter SL_fl_ka;
  CGparameter SL_fl_ks;
  CGparameter SL_fl_eye;
  CGparameter SL_fl_shininess;
  
void shader_lighting_prepare()
{
  cgSetErrorCallback(shader_lighting_prepare);

  CGerror lastError = cgGetError();
  if(lastError)
  {
    printf("%s\n\n", cgGetErrorString(lastError));
    printf("%s\n", cgGetLastListing(SL_shader));
    printf("Cg error, exiting...\n");
    exit(0);
  }

  if( SL_shader )
  {
    cgDestroyProgram( SL_flv_program );
    cgDestroyProgram( SL_flp_program );
    cgDestroyContext( SL_shader );
  }

  SL_shader = cgCreateContext();

    SL_vprofile = cgGLGetLatestProfile(CG_GL_VERTEX);
    cgGLSetOptimalOptions(SL_vprofile);

    SL_fprofile = cgGLGetLatestProfile( CG_GL_FRAGMENT );
    cgGLSetOptimalOptions( SL_fprofile );

    SL_flv_program = cgCreateProgramFromFile( SL_shader, CG_SOURCE, "shader_lighting.cg", SL_vprofile, "cgfl_vp", 0 );
    cgGLLoadProgram( SL_flv_program );

    SL_flp_program = cgCreateProgramFromFile( SL_shader, CG_SOURCE, "shader_lighting.cg", SL_fprofile, "cgfl_fp", 0 );
    cgGLLoadProgram( SL_flp_program );
    SL_fl_lpos = cgGetNamedParameter( SL_flp_program, "lpos" );
    SL_fl_kd   = cgGetNamedParameter( SL_flp_program, "kd" );
    SL_fl_ka   = cgGetNamedParameter( SL_flp_program, "ka" );
    SL_fl_ks   = cgGetNamedParameter( SL_flp_program, "ks" );
    SL_fl_eye   = cgGetNamedParameter( SL_flp_program, "eye" );
    SL_fl_shininess   = cgGetNamedParameter( SL_flp_program, "shininess" );
}

void shader_lighting_begin()
{
  float mv[16], lpos[4];
  glGetFloatv( GL_MODELVIEW_MATRIX, mv );
  glGetLightfv( GL_LIGHT0, GL_POSITION, lpos );

  FLOAT3 a, b, c, d, l;
    a = FLOAT3( mv[0], mv[1], mv[2] );
    b = FLOAT3( mv[4], mv[5], mv[6] );
    c = FLOAT3( mv[8], mv[9], mv[10] );
    d = -FLOAT3( mv[12], mv[13], mv[14] );
    l = FLOAT3( lpos[0], lpos[1], lpos[2] );

  cgGLSetParameter4f( SL_fl_lpos, vdot(a,l), vdot(b,l), vdot(c,l), lpos[3] );
  cgGLSetParameter3f( SL_fl_eye,  vdot(a,d), vdot(b,d), vdot(c,d) );
  cgGLSetParameter3fv( SL_fl_ka, (float*)&(FLOAT3(0,0,0)) );
  cgGLSetParameter3fv( SL_fl_kd, (float*)&(FLOAT3(1,1,1)) );
  cgGLSetParameter3fv( SL_fl_ks, (float*)&(FLOAT3(1,1,1)) );
  cgGLSetParameter1f ( SL_fl_shininess, 50 );

  cgGLEnableProfile( SL_vprofile );    
  cgGLEnableProfile( SL_fprofile );    
  cgGLBindProgram( SL_flv_program );
  cgGLBindProgram( SL_flp_program );
}
void shader_lighting_end()
{
  cgGLDisableProfile( SL_vprofile );
  cgGLDisableProfile( SL_fprofile );
}


