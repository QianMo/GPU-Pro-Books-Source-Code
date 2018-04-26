#include <stdio.h>
#include <stdlib.h>

#include <GL/glew.h>

#include <cg/cgGL.h>
#include <cg/cg.h>

#include "g_vector.h"
#include "g_pfm.h"

#include "shader_som.h"

CGcontext SSOM_shader = NULL;
CGprofile SSOM_vprofile;
CGprogram SSOM_flv_program;
CGparameter *SSOM_flv_codebook;
  
void cgErrorCallback()
{
  CGerror lastError = cgGetError();
  if(lastError)
  {
    printf("%s\n\n", cgGetErrorString(lastError));
    printf("%s\n", cgGetLastListing(SSOM_shader));
    printf("Cg error, exiting...\n");
    exit(0);
  }
}

void shader_som_prepare( int codebook_w, int codebook_h )
{
  int i, j;
  char str[256];
  char program[4096];

  if( SSOM_shader )
  {
    delete[] SSOM_flv_codebook;
    cgDestroyProgram( SSOM_flv_program );
    cgDestroyContext( SSOM_shader );
  }

  {
    char *tmp;
    tmp = (char*) freadall("shader_som.cg");
    sprintf( program, 
      "#define codebook_w %i\n"
      "#define codebook_h %i\n%s", 
      codebook_w, codebook_h, tmp );
    free(tmp);
  }

  cgSetErrorCallback(cgErrorCallback);

  SSOM_shader = cgCreateContext();

    SSOM_vprofile = cgGLGetLatestProfile( CG_GL_VERTEX);
    cgGLSetOptimalOptions(SSOM_vprofile);

    SSOM_flv_program = cgCreateProgram( SSOM_shader, CG_SOURCE, program, SSOM_vprofile, "cgfl_vp_som", 0 );
    cgGLLoadProgram( SSOM_flv_program );

    SSOM_flv_codebook = new CGparameter[codebook_w*codebook_h];
    for( j=0; j<codebook_h; j++ )
    for( i=0; i<codebook_w; i++ )
    {
      sprintf( str, "codebook[%i][%i]", j, i );
      SSOM_flv_codebook[j*codebook_w+i] = cgGetNamedParameter( SSOM_flv_program, str );
    }
}

void shader_som_begin( const GPfm &codebook )
{
  int i;
  cgGLEnableProfile( SSOM_vprofile );    
  cgGLBindProgram( SSOM_flv_program );
  for( i=0; i<codebook.w*codebook.h; i++ )
    cgSetParameter3fv( SSOM_flv_codebook[i], (float*)&codebook.fm[i] );
}

void shader_som_end()
{
  cgGLDisableProfile( SSOM_vprofile );
}

