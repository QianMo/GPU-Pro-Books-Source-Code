#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <GL/glew.h>

#include "g_pfm.h"
#include "shader.h"


GLuint fbo=0;
GLuint usrc, xsrc, xdes;

GLuint iterate( int n_pass );

float Float32Grey( FLOAT3 l )
{
  return (l.x+l.y+l.z)/3;
}

void prepare_src( const GPfm &src0, bool bBinary, GPf1 &src )
{
  //////////////////////////////////////////////////////////////////
  // convert to CNN data mode, value: [-1, 1],  "white -1" to "black 1"
  int i, j;
  src.load( (src0.w+3)/4*4+4, src0.h+4 );

  for( j=0; j<src0.h; j++ )
    for( i=0; i<src0.w; i++ )
      src.pm[j+2][i+2] = 2*Float32Grey(src0.pm[j][i])-1;
  if( bBinary )
  {
    for( j=0; j<src0.h; j++ )
      for( i=0; i<src0.w; i++ )
        src.pm[j+2][i+2] = src.pm[j+2][i+2]>0?1.f:-1.f;
  }
}

void finalize_src( const GPf1 &src0, GPf1 &des, int w, int h )
{
  int i;
  src0.getblk( des, 2,2, w,h );
  for( i=0; i<w*h; i++ )
  {
    des.fm[i] = g_clamp(des.fm[i],-1.f,1.f);
    des.fm[i] = des.fm[i]*.5f+.5f;
  }
}

void filteru( const GPf1 &src, GPf1 &des, const float *B, float I )
{
  int i, j, s, t, st;
  des.load( src.w, src.h );

  for( j=0; j<src.h; j++ )
  for( i=0; i<src.w; i++ )
  {
    for( t=-2, st=0; t<=2; t++ )
    for( s=-2; s<=2; s++, st++ )
      des.pm[j][i] += src.pm[g_clamp(j+t,0,src.h-1)][g_clamp(i+s,0,src.w-1)]*B[st];
    des.pm[j][i] += I;
  }
}

void foldtex( const GPf1 &src, GPf4 &des )
{
  GPf1 blk, blks[4];
  int i;
  int w0, i0;
  int w1, h1;

  w0 = (src.w+3)/4; 
  w1 = w0+4; 
  h1 = src.h; 

  for( i=0, i0=0; i<4; i++, i0+=w0 )
  {
    src.getblk( blk, i0,0, g_min(src.w-i0,w1), h1 );
    blks[i].load(w1,h1);
    blks[i].draw(blk,0,0);
  }
  des.load(w1,h1, blks[0].fm, blks[1].fm, blks[2].fm, blks[3].fm );
}

void unfoldtex( const GPf4 &src, GPf1 &des )
{
  GPf1 blk, blks[4];
  int i;
  int w0;
  w0 = src.w-4;
 
  des.load( 4*w0, src.h );
  for( i=0; i<4; i++ )
    blks[i].load(src.w,src.h);
  src.getchannel( blks[0].fm, blks[1].fm, blks[2].fm, blks[3].fm );
  for( i=0; i<4; i++ )
  {
    blks[i].getblk(blk,0,0,w0,src.h);
    des.draw( blk, i*w0, 0 );
  }
}

void cnn_prepare_all_ad_upf( 
  const GPfm &src0, bool bBinary, 
  float *A, float *B, float I, 
  bool bxinit, float xinit, float cnn_step
){
  if(fbo)
  {
    glDeleteTextures( 1, &usrc );
    glDeleteTextures( 1, &xsrc );
    glDeleteTextures( 1, &xdes );
    glDeleteFramebuffersEXT( 1, &fbo );
  }


  GPf1 src;
  prepare_src( src0, bBinary, src );

  {
    // initialize utex
    GPf1 ufiltered;
    GPf4 ufolded;
    filteru( src, ufiltered, B, I );
    foldtex( ufiltered, ufolded );

    glGenTextures( 1, &usrc );
      glBindTexture(GL_TEXTURE_RECTANGLE_ARB, usrc );
      glTexParameterf(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
      glTexParameterf(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
      glTexParameterf(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      glTexParameterf(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
      glTexImage2D( GL_TEXTURE_RECTANGLE_ARB, 0, GL_RGBA32F_ARB, ufolded.w, ufolded.h, 0, GL_RGBA, GL_FLOAT, ufolded.fm );
  }

  {
    // initialize input texture depends on Xinit
    GPf4 xfolded;
    if( bxinit )
      xfolded.load( src.w/4+4, src.h, bxinit );
    else
      foldtex( src, xfolded );

    glGenTextures( 1, &xsrc );
      glBindTexture(GL_TEXTURE_RECTANGLE_ARB, xsrc );
      glTexParameterf(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
      glTexParameterf(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
      glTexParameterf(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      glTexParameterf(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
      glTexImage2D( GL_TEXTURE_RECTANGLE_ARB, 0, GL_RGBA32F_ARB, xfolded.w, xfolded.h, 0, GL_RGBA, GL_FLOAT, xfolded.fm );
    glGenTextures( 1, &xdes );
      glBindTexture(GL_TEXTURE_RECTANGLE_ARB, xdes );
      glTexParameterf(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
      glTexParameterf(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
      glTexParameterf(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      glTexParameterf(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
      glTexImage2D( GL_TEXTURE_RECTANGLE_ARB, 0, GL_RGBA32F_ARB, xfolded.w, xfolded.h, 0, GL_RGBA, GL_FLOAT, xfolded.fm );

    glGenFramebuffersEXT( 1, &fbo );
      glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, fbo );
      glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_RECTANGLE_ARB, xdes, 0);
      glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, 0 );
  }

  cginit( A, B, cnn_step );
}

GLuint iterate( int n_pass )
{
  int w, h, i;

  glBindTexture(GL_TEXTURE_RECTANGLE_ARB, xdes );
  glGetTexLevelParameteriv( GL_TEXTURE_RECTANGLE_ARB, 0, GL_TEXTURE_WIDTH, &w );
  glGetTexLevelParameteriv( GL_TEXTURE_RECTANGLE_ARB, 0, GL_TEXTURE_HEIGHT, &h );

  glPushAttrib( GL_VIEWPORT_BIT );
  glViewport( 0,0, w,h );
  glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D( 0,w,0,h );
  glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

  for( i=0; i<n_pass; i++ )
  {
    glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, fbo );
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_RECTANGLE_ARB, xdes, 0);

    cgcnn_begin(usrc, xsrc);
    glBegin(GL_QUADS);
      glTexCoord2i(   2,   2 );  glVertex2i(   2,   2 );
      glTexCoord2i( w-2,   2 );  glVertex2i( w-2,   2 );
      glTexCoord2i( w-2, h-2 );  glVertex2i( w-2, h-2 );
      glTexCoord2i(   2, h-2 );  glVertex2i(   2, h-2 );
    glEnd();
    cgcnn_end();

    cgledge_begin(usrc, xsrc);
    glBegin(GL_QUADS);
      glTexCoord2i( w-4,   2 );  glVertex2i(   0,   2 );
      glTexCoord2i( w-2,   2 );  glVertex2i(   2,   2 );
      glTexCoord2i( w-2, h-2 );  glVertex2i(   2, h-2 );
      glTexCoord2i( w-4, h-2 );  glVertex2i(   0, h-2 );
    glEnd();
    cgledge_end();

    cgredge_begin(usrc, xsrc);
    glBegin(GL_QUADS);
      glTexCoord2i(   2,   2 );  glVertex2i( w-2,   2 );
      glTexCoord2i(   4,   2 );  glVertex2i( w  ,   2 );
      glTexCoord2i(   4, h-2 );  glVertex2i( w  , h-2 );
      glTexCoord2i(   2, h-2 );  glVertex2i( w-2, h-2 );
    glEnd();
    cgredge_end();

    swap(xsrc,xdes);
  }
  glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, 0 );
  glPopAttrib();
  glMatrixMode(GL_PROJECTION);
    glPopMatrix();
  glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

  return xsrc;
}


void save_output( GPf1 &res2, int srcw, int srch )
{
  GPf4 des;
  GPf1 res;
  int w, h;

  glBindTexture(GL_TEXTURE_RECTANGLE_ARB, xdes );
  glGetTexLevelParameteriv( GL_TEXTURE_RECTANGLE_ARB, 0, GL_TEXTURE_WIDTH, &w );
  glGetTexLevelParameteriv( GL_TEXTURE_RECTANGLE_ARB, 0, GL_TEXTURE_HEIGHT, &h );
  des.load(w,h);
  glGetTexImage( GL_TEXTURE_RECTANGLE_ARB, 0, GL_RGBA, GL_FLOAT, des.fm );
  unfoldtex( des, res );
  finalize_src( res, res2, srcw, srch );
  res2.flip_vertical();
}

void save_output( const char *spath, int srcw, int srch )
{
  GPf1 res2;
  GPfm tmp;
  save_output( res2, srcw, srch );
  tmp.load( res2.w, res2.h, res2.fm );
  tmp.save( spath, "png" );
}

void normalize_filter( float *a, int n )
{
  float ma;
  int i;
  for( i=0,ma=-1; i<n; i++ )
    if( ma<fabsf(a[i]) )
      ma = fabsf(a[i]);
  for( i=0; i<n; i++ )
      a[i] /= ma;
}

