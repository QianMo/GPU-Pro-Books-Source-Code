#include "shader_codeword.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <direct.h>
#include <windows.h>

#include "gpu_som.h"
#include "shader_som.h"

GPUSom::GPUSom()
{
  memset( this, 0, sizeof(GPUSom) );
}

void GPUSom::init()
{
  gpulbg_j = 0;
  gpulbg_gpubook.load( gpulbg_srcbook.w, gpulbg_srcbook.h, gpulbg_srcbook.fm );
  gpu_codebook2codeword( vis_src_id, gpulbg_srcbook, vis_ucodeword_id );
  gpu_subidx_projection( vis_src_id, vis_des_id, vis_ucodeword_id, vis_vcodeword_id, vis_codebook_id );
}

void GPUSom::set_info( int max_cycles, int cbw, int cdh )
{
  gpulbg_max_cycles = max_cycles;
  codebook_w = cbw;
  codebook_h = cdh;
}

void GPUSom::prepare_codebook()
{
  int i, *idx;
  FLOAT3 cc;

  gpulbg_srcbook.load( codebook_w, codebook_h );
  idx = (int*) malloc( codebook_w*codebook_h*sizeof(int) );
  random_index( idx, gpu_src.w*gpu_src.h, codebook_w*codebook_h );
  for( i=0; i<codebook_w*codebook_h; i++ )
  {
    vperturb( (float*)&gpu_src.fm[idx[i]], (float*)&cc, 3, .1f );
    cc = f3abs(cc);
    gpulbg_srcbook.fm[i] = cc;
  }
  free(idx);
}

void GPUSom::prepare( const char *spath )
{
  GPfm src;
  src.load( spath );
  prepare( src );
}

void GPUSom::gpulbg_iterate()
{
  //for( j=0; j<gpulbg_max_cycles; j++ )
  int &j = gpulbg_j;
  if( j<gpulbg_max_cycles )
  {
    int i;

    GPfm &desbook = gpulbg_gpubook;
    int codebook_size = desbook.w*desbook.h;

    glPushAttrib( GL_VIEWPORT_BIT );
    glViewport( 0,0, desbook.w,desbook.h );
    glMatrixMode(GL_PROJECTION);
      glLoadIdentity();
      gluOrtho2D( 0,1,0,1 );
    glMatrixMode(GL_MODELVIEW);
      glLoadIdentity();

    glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, vis_tbook_fbo );
    glClampColorARB(GL_CLAMP_VERTEX_COLOR_ARB, GL_FALSE);
    glClearColor(0,0,0,0);
    glPointSize(1);
    glEnable( GL_BLEND );
    glBlendFunc( GL_ONE, GL_ONE );

    glClear( GL_COLOR_BUFFER_BIT );

    if ( j>=(gpulbg_max_cycles-1-20) )
      glPointSize(1);
    else
    {
      if ( j<(gpulbg_max_cycles>>3) )
      {
        if ( (desbook.w>>1)>5 )
          glPointSize(5);
        else
          glPointSize( float(desbook.w>>1) );

      }
      else
      {
        if ( (desbook.w>>1)>3 )
          glPointSize(3);
        else
          glPointSize(1);
      }
    }

    shader_som_begin(desbook);
      glCallList( vis_src_list );
    shader_som_end();

    GPf4 tbook;
    tbook.load( desbook.w, desbook.h );
    glReadPixels( 0,0, desbook.w,desbook.h, GL_RGBA, GL_FLOAT, tbook.fm );

    for( i=0; i<codebook_size; i++ )
    {
      if( tbook.fm[i].w>0.5 )
        desbook.fm[i] = (FLOAT3(tbook.fm[i].x,tbook.fm[i].y,tbook.fm[i].z))/tbook.fm[i].w;
    }

    glPopAttrib();
    glDisable( GL_BLEND );
    glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, 0 );

    gpulbg_j++;
  }
}

void GPUSom::update_codeword()
{
  gpu_codebook2codeword( vis_src_id, gpulbg_gpubook, vis_ucodeword_id );
  gpu_subidx_projection( vis_src_id, vis_des_id, vis_ucodeword_id, vis_vcodeword_id, vis_codebook_id );
}


void GPUSom::clear()
{
  if( vis_src_list!=0 )
    glDeleteLists( vis_src_list, 1 );
  if( vis_tbook_id!=0 )
    glDeleteTextures( 1, &vis_tbook_id );
  if( vis_tbook_fbo!=0 )
    glDeleteFramebuffersEXT( 1, &vis_tbook_fbo );
  if( vis_codebook_id!=0 )
    glDeleteTextures( 1, &vis_codebook_id );
  if( vis_src_id!=0 )
    glDeleteTextures( 1, &vis_src_id );
  if( vis_des_id!=0 )
    glDeleteTextures( 1, &vis_des_id );
  if( vis_ucodeword_id!=0 )
    glDeleteTextures( 1, &vis_ucodeword_id );
  if( vis_vcodeword_id!=0 )
    glDeleteTextures( 1, &vis_vcodeword_id );
}

GPUSom::~GPUSom()
{
  clear();
}

void GPUSom::prepare( const GPfm &src )
{
  clear();

  int i;


  gpu_src.load( src.w, src.h, src.fm );
  prepare_codebook();
  GPfm &srcbook = gpulbg_srcbook;

  glGenTextures( 1, &vis_src_id );
  glBindTexture(GL_TEXTURE_2D, vis_src_id );
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  gpu_src.flip_vertical();
  glTexImage2D( GL_TEXTURE_2D, 0, GL_RGB32F_ARB, gpu_src.w, gpu_src.h, 0, GL_RGB, GL_FLOAT, gpu_src.fm );
  gpu_src.flip_vertical();

  glGenTextures( 1, &vis_des_id );
  glBindTexture(GL_TEXTURE_2D, vis_des_id );
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexImage2D( GL_TEXTURE_2D, 0, GL_RGB32F_ARB, gpu_src.w, gpu_src.h, 0, GL_RGB, GL_FLOAT, 0 );

  glGenTextures( 1, &vis_ucodeword_id );
  glBindTexture(GL_TEXTURE_2D, vis_ucodeword_id );
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexImage2D( GL_TEXTURE_2D, 0, GL_RGB32F_ARB, gpu_src.w, gpu_src.h, 0, GL_RGB, GL_FLOAT, 0 );
  
  glGenTextures( 1, &vis_vcodeword_id );
  glBindTexture(GL_TEXTURE_2D, vis_vcodeword_id );
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexImage2D( GL_TEXTURE_2D, 0, GL_RGB32F_ARB, gpu_src.w, gpu_src.h, 0, GL_RGB, GL_FLOAT, 0 );
  
  glGenTextures( 1, &vis_codebook_id );
  glBindTexture(GL_TEXTURE_2D, vis_codebook_id );
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexImage2D( GL_TEXTURE_2D, 0, GL_RGB32F_ARB, srcbook.w, srcbook.h, 0, GL_RGB, GL_FLOAT, srcbook.fm );
  
  glGenTextures( 1, &vis_tbook_id );
    glBindTexture(GL_TEXTURE_2D, vis_tbook_id );
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, srcbook.w, srcbook.h, 0, GL_RGBA, GL_FLOAT, 0 );
  
  glGenFramebuffersEXT( 1, &vis_tbook_fbo );
    glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, vis_tbook_fbo );
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, vis_tbook_id, 0);
    glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, 0 );

  vis_src_list = glGenLists(1);
    glNewList(vis_src_list, GL_COMPILE);
    glBegin( GL_POINTS );
    for( i=0; i<src.w*src.h; i++ )
      glVertex3fv( (float*)&src.fm[i] ); 
    glEnd();
    glEndList();

}

