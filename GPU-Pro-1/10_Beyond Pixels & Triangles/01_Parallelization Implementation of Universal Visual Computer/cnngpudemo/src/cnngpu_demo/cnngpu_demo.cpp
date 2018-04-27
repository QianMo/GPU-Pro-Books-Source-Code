#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <direct.h>
#include <windows.h>

#include <GL/glew.h>
#include <GL/glut.h>
#include <cg/cgGL.h>
#include <cg/cg.h>

#include "g_pfm.h"
#include "g_bell.h"

#include "shader.h"
#include "cnn_utility.h"

float a[25] = {0};
float b[25] = {0};
float I=0;
bool bxinit = false;
float xinit = 0;
float cnn_step = 0;

GPfm srcimage;
int iteration_num = 2000;
int iteration_i = 0;
bool biteration = false;
bool bBinary = false;
char filter_path[256];
char image_path[256];

const char *exepath;

float szoom = 1;
float sx = 0, sy = 0;

void display();
void keyboard( unsigned char key, int x, int y );
void special( int key, int x, int y );
void glutMenuFunc( int key );
void load_filter( const char *spath );

void main( int argc, const char **argv )
{
  if( argc<3 )
  {
    printf( "[Usage] cnngpu_demo image_path filter_path\n" );
    exit(-1);
  }

  exepath = argv[0];

  glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGB );
  glutInitWindowSize( 640, 640 );

  glutCreateWindow( "cnngpu_demo" );
  glutDisplayFunc( display );
  glutKeyboardFunc( keyboard );
  glutSpecialFunc( special );
  glutMenuFunc(0);

  glewInit(); 
  cgfl_prepare(exepath);
  
  load_filter( argv[2] );
  strcpy( image_path, argv[1] );
  srcimage.load( image_path );
  srcimage.flip_vertical();
  cnn_prepare_all_ad_upf( srcimage, bBinary, a, b, I,bxinit,xinit,cnn_step );

  glutMainLoop();
}

void load_filter( const char *spath )
{
  if( !fexist(spath) )
  {
    printf( "[Error] flie %s not found\n", spath );
    exit(-1);
  }

  int i;
  char str[256];
  
  float *pp;
  FILE *f0 = fopen( spath, "rt" );
  
  fscanf( f0, "A\n" );
  for( i=0, pp=a; i<5; i++, pp+=5 )
    fscanf( f0, "%f %f %f %f %f\n", &pp[0], &pp[1], &pp[2], &pp[3], &pp[4] );
  fgets(str,256,f0);
  
  fscanf( f0, "B\n" );
  for( i=0, pp=b; i<5; i++, pp+=5 )
    fscanf( f0, "%f %f %f %f %f\n", &pp[0], &pp[1], &pp[2], &pp[3], &pp[4] );
  fgets(str,256,f0);
  
  fscanf( f0, "I %f\n", &I );
  fscanf( f0, "BXINIT %s\n", str );
  bxinit = strcmp(str,"true")==0;
  fscanf( f0, "XINIT %f\n", &xinit );
  fscanf( f0, "CNN_STEP %f\n", &cnn_step );

  fscanf( f0, "INPUT %s\n", str );
  bBinary = strcmp(str,"binary")==0;

  strcpy( filter_path, spath );
}


void display()
{
  GLint viewport[4];
    glGetIntegerv( GL_VIEWPORT, viewport );

  glClearColor( .1, .2, .3, 1 );
  glClear( GL_COLOR_BUFFER_BIT );

  glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPickMatrix(
      viewport[2]*(.5+sx), viewport[3]*(.5+sy), 
      viewport[2]*szoom, viewport[3]*szoom, 
      viewport );
    gluOrtho2D( 0,viewport[2],0, viewport[3] );
  
  glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef( viewport[2]/2,viewport[3]/2,0 );
    glTranslatef( -srcimage.w/2,-srcimage.h/2,0 );

    int filter, xsrc;
    xsrc = iterate(0);
    glGetTexParameteriv(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, &filter);
    glTexParameterf(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    cgunfold_begin( xsrc );
    glBegin( GL_QUADS );
      glTexCoord2f(          2,2 );  glVertex2f( 0,0 );
      glTexCoord2f( srcimage.w,2 );  glVertex2f( srcimage.w,0 );
      glTexCoord2f( srcimage.w,2+srcimage.h );  glVertex2f( srcimage.w,srcimage.h );
      glTexCoord2f(          2,2+srcimage.h );  glVertex2f( 0,srcimage.h );
    glEnd();
    cgunfold_end();
    glTexParameterf(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, filter);

  glPrintf( ">>glut" );
  if( iteration_i>=iteration_num )
    glPrintf( "Progress:    DONE    [press 'space' to restart]\n" );
  else if( biteration )
  {
    glPrintf( "Progress: " );
    float s;
    s = float(clock()%1000)/1000;
    s = fabsf(2*s-1);
    FLOAT3 col;
    col = g_lerp( FLOAT3(0,1,0), FLOAT3(1,.1,.1), s );
    glColor3fv( (float*)&col );
    glPrintf( "SIMULATING" );
    glColor3f( 1,1,1 );
    glPrintf( " [press 'space' to pause]\n" );
  }else
  {
    if( iteration_i==0 )
    glPrintf( "Progress:   PAUSED   [press 'space' to begin simulation]\n" );
    else
    glPrintf( "Progress:   PAUSED   [press 'space' to resume simulation]\n" );
  }
  glPrintf( "Iteration:  %i/%i\n", iteration_i, iteration_num );
  glPrintf( "Filter:  %s\n", filter_path );
  glPrintf( "cnn_step:  %f\n", cnn_step );
  
  if( iteration_i==0 )
  {
    glColor3f( 0,1,0 );
    glPrintf( ">>glut(%i,%i)", viewport[2]/2-130, viewport[3]/2-16 );
    glPrintf( "Press 'space' to begin CNN simulation,\n" );
    glPrintf( "or see Right-Click menu for more option.\n" );
    glColor3f( 1,1,1 );
  }

  glutSwapBuffers();

  if( biteration )
  {
    iterate(100);
    iteration_i+=100;
    if( iteration_i>=iteration_num )
      biteration = false;
    glutPostRedisplay();
  }
}

void keyboard( unsigned char key, int x, int y )
{
  switch( key )
  {
    case 27:
      szoom = 1;
      sx = 0;
      sy = 0;
      glutPostRedisplay();
      break;
    case '+':
    case '=':
      szoom *= .9;
      glutPostRedisplay();
      break;
    case '_':
    case '-':
      szoom /= .9;
      glutPostRedisplay();
      break;
    case '|':
    case '\\':
      {
        cnn_prepare_all_ad_upf( srcimage, bBinary, a, b, I,bxinit,xinit,cnn_step );
        iteration_i = 0;
        biteration=false;
      }
      glutPostRedisplay();
      break;
    case ' ':
      biteration=!biteration;
      if( biteration && iteration_i>=iteration_num )
      {
        cnn_prepare_all_ad_upf( srcimage, bBinary, a, b, I,bxinit,xinit,cnn_step );
        iteration_i = 0;
      }
      glutPostRedisplay();
      break;
    case '1':
      {
        char main_ifile_filter[] =
          "all supported format (*.pfm;*.ppm;*.png;*.bmp;*.jpg;*.tga;*.gif;*.dds)\0*.pfm;*.ppm;*.png;*.bmp;*.jpg;*.tga;*.gif;*.dds\0"
          "portable float map (*.pfm)\0*.pfm\0"
          "portable pixel map (*.ppm)\0*.ppm\0"
          "portable network graphic (*.png)\0*.png\0"
          "bitmap (*.bmp)\0*.bmp\0"
          "jpeg (*.jpg)\0*.jpg\0"
          "targa (*.tga)\0*.tga\0"
          "graphics interchange format (*.gif)\0*.gif\0"
          "directdraw surface (*.dds)\0*.dds\0"
          "All files (*.*)\0*.*\0"
          "\0";
        char spath[256]="";
        GPath gp = parse_spath( image_path );
        replace_char( gp.dname, '/', '\\' );
        char curr_dir[256];
        _getcwd( curr_dir, 256 );

        OPENFILENAME ofn = { 0 };
        ofn.lpstrTitle = "Open source image ...";
        ofn.lStructSize  = sizeof(OPENFILENAME);
        ofn.lpstrFilter  = main_ifile_filter;
        ofn.nFilterIndex = 1;
        ofn.lpstrFile    = spath;
        ofn.nMaxFile     = 256;
        ofn.Flags        = OFN_FILEMUSTEXIST | OFN_HIDEREADONLY;
        ofn.lpstrInitialDir = gp.dname;
        if( GetOpenFileName(&ofn) )
        {
          srcimage.load(spath);
          srcimage.flip_vertical();
          cnn_prepare_all_ad_upf( srcimage, bBinary, a, b, I,bxinit,xinit,cnn_step );
          iteration_i = 0;
          glutPostRedisplay();
          _chdir(curr_dir);
        }
      }
      break;
    case '2':
      {
        GPath gp = parse_spath( filter_path );
        replace_char( gp.dname, '/', '\\' );
        char spath[256]="";
        OPENFILENAME ofn = { 0 };
        ofn.lpstrTitle = "Open source image ...";
        ofn.lStructSize  = sizeof(OPENFILENAME);
        ofn.lpstrFilter  = "Text documents (*.txt)\0*.txt\0";
        ofn.nFilterIndex = 1;
        ofn.lpstrFile    = spath;
        ofn.nMaxFile     = 256;
        ofn.Flags        = OFN_FILEMUSTEXIST | OFN_HIDEREADONLY;
        ofn.lpstrInitialDir = gp.dname;
        char curr_dir[256];
        _getcwd( curr_dir, 256 );
        if( GetOpenFileName(&ofn) )
        {
          load_filter( spath );
          cnn_prepare_all_ad_upf( srcimage, bBinary, a, b, I,bxinit,xinit,cnn_step );
          iteration_i = 0;
          glutPostRedisplay();
          _chdir(curr_dir);
        }
      }
      break;
    case '3':
      {
        GPath gp = parse_spath( image_path );
        replace_char( gp.dname, '/', '\\' );
        char spath[256]="cnn_output.png";
        OPENFILENAME ofn = { 0 };
        ofn.lpstrTitle = "Save output as ...";
        ofn.lStructSize  = sizeof(OPENFILENAME);
        ofn.lpstrFilter  = "portable network graphic (*.png)\0*.png\0";
        ofn.nFilterIndex = 1;
        ofn.lpstrFile    = spath;
        ofn.nMaxFile     = 256;
        ofn.Flags        = OFN_HIDEREADONLY | OFN_OVERWRITEPROMPT;
        ofn.lpstrInitialDir = gp.dname;
        char curr_dir[256];
        _getcwd( curr_dir, 256 );
        if( GetSaveFileName(&ofn) )
        {
          save_output( spath, srcimage.w, srcimage.h );
          _chdir(curr_dir);
        }
      }
      break;
  }
}

void special( int key, int x, int y )
{
  GLint viewport[4];
    glGetIntegerv( GL_VIEWPORT, viewport );

  switch( key )
  {
    case GLUT_KEY_F11:
    {
      static bool bfullscreen = false;
      bfullscreen = !bfullscreen;
      if( bfullscreen )
      {
        glutFullScreen();
      }else
      {
        glutReshapeWindow(640, 640);
        glutPositionWindow(50, 50);
      }
    }
    break;

    case GLUT_KEY_UP:
      sy -= 8*szoom/viewport[3];
      glutPostRedisplay();
      break;
    case GLUT_KEY_DOWN:
      sy += 8*szoom/viewport[3];
      glutPostRedisplay();
      break;
    case GLUT_KEY_LEFT:
      sx += 8*szoom/viewport[2];
      glutPostRedisplay();
      break;
    case GLUT_KEY_RIGHT:
      sx -= 8*szoom/viewport[2];
      glutPostRedisplay();
      break;
  }
}

void glutMenuFunc( int key )
{
  if( key==0 )
  {
    int hmenu, hmenu_sub;

    hmenu = glutCreateMenu(glutMenuFunc);
      glutAddMenuEntry( "space - toggle the simulation", ' ' );
      glutAddMenuEntry( "\\ - reset source image", '\\' );
      glutAddMenuEntry( "---------------------------", 1 );
      glutAddMenuEntry( "1 - load source image ...", '1' );
      glutAddMenuEntry( "2 - load filter ...", '2' );
      glutAddMenuEntry( "3 - save output image ...", '3' );

    hmenu_sub = glutCreateMenu(glutMenuFunc);
      glutAddMenuEntry( "+ - zoom in", '+' );
      glutAddMenuEntry( "- - zoom out", '-' );
      glutAddMenuEntry( "F11 - toggle fullscreen", -GLUT_KEY_F11 );
      glutAddMenuEntry( "up    -  model move up"    , -GLUT_KEY_UP );
      glutAddMenuEntry( "down  -  model move down"  , -GLUT_KEY_DOWN );
      glutAddMenuEntry( "left  -  model move left"  , -GLUT_KEY_LEFT );
      glutAddMenuEntry( "right -  model move right" , -GLUT_KEY_RIGHT );
      glutSetMenu( hmenu );
      glutAddSubMenu( "view", hmenu_sub );

    glutAttachMenu(GLUT_RIGHT_BUTTON);

  }else
  {
    if( key<0 )
      special( -key,0,0 );
    else
      keyboard( key,0,0 );
    return;
  }
}
