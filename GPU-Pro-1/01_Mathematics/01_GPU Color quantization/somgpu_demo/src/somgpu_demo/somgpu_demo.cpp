/////////////////////////////////////////////////////
//  Self-Organizing Map (SOM) Batch mode GPU training
//
//  Assume the codebook is having a 2D structure
//

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
#include "g_obj.h"
#include "gpu_som.h"

#include "shader_lighting.h"
#include "shader_codeword.h"
#include "shader_som.h"


int cb_dim = 16;
bool btraining = true;
float szoom = 1.235f;
float sx = 0.0214f, sy = -0.1758f;

GObj som_model;
GBell g_bell;
GPUSom som;


void display();
void keyboard( unsigned char key, int x, int y );
void special( int key, int x, int y );
void glutMenuFunc( int key );

void draw_info_images();
void draw_info_text();
void draw_info_model( float *mo );
  void init_som_model();
  void update_som_model();

void main( int argc, const char **argv )
{

  if( argc<2 )
  {
    printf( "[Usage] somgpu_demo.exe image_path\n" );
    return;
  }


  srand(0);

  glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH );
  glutInitWindowSize( 512, 512 );

  glutCreateWindow( "Training of som codeboook" );
  glewInit(); 

  glutDisplayFunc( display );
  glutKeyboardFunc( keyboard );
  glutSpecialFunc( special );
  glutMenuFunc(0);

  g_bell.create_window();
  g_bell.set_fellow( 0, GBELL_LIGHT | GBELL_OBJ | GBELL_DOWN | GBELL_DRAG | GBELL_DOCK );
  g_bell.load( "bell_save01.txt" );
  g_bell.set_active_ctrl( GBELL_OBJ );

  shader_lighting_prepare();
  shader_som_prepare( cb_dim, cb_dim );
  shader_codeword_prepare( cb_dim, cb_dim );

  som.set_info( 500, cb_dim, cb_dim );
  som.prepare( argv[1] );
  som.init();
  init_som_model();

  glutMainLoop();
}

void display()
{
  float ml[16], mo[16];
    g_bell.get_matrix( ml, GBELL_LIGHT );
    g_bell.get_matrix( mo, GBELL_OBJ );

  GLint viewport[4];
    glGetIntegerv( GL_VIEWPORT, viewport );


  glClearColor( .1, .2, .3, 1 );
  glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

  glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPickMatrix(
      viewport[2]*(.5+sx), viewport[3]*(.5+sy), 
      viewport[2]*szoom, viewport[3]*szoom, 
      viewport );
    gluPerspective( 45, double(viewport[2])/viewport[3], 0.1, 10 );
  
  glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt( 0,0,3, 0,0,0, 0,1,0 );
    glLightfv(GL_LIGHT0, GL_POSITION, &ml[8]);

    draw_info_model(mo);
    draw_info_text();
    draw_info_images();
    
  glutSwapBuffers();

  if( btraining )
  {
    som.gpulbg_iterate();
    if( som.gpulbg_j==som.gpulbg_max_cycles )
    {
      som.update_codeword();
      btraining = false;
    }

    update_som_model();
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
      g_bell.reset();
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
    case ' ':
      btraining = !btraining;
      if( btraining && som.gpulbg_j==som.gpulbg_max_cycles )
      {
        som.init();
        update_som_model();
      }
      glutPostRedisplay();
      break;
    case '\\':
      som.init();
      update_som_model();
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
        char curr_dir[256];
        OPENFILENAME ofn = { 0 };
        ofn.lpstrTitle = "Open source image for training ...";
        ofn.lStructSize  = sizeof(OPENFILENAME);
        ofn.lpstrFilter  = main_ifile_filter;
        ofn.nFilterIndex = 1;
        ofn.lpstrFile    = spath;
        ofn.nMaxFile     = 256;
        ofn.Flags        = OFN_FILEMUSTEXIST | OFN_HIDEREADONLY;
        _getcwd( curr_dir, 256 );
        if( GetOpenFileName(&ofn) )
        {
          som.prepare( spath );
          som.init();
          update_som_model();
          glutPostRedisplay();
          _chdir(curr_dir);
        }
      }
      break;
    case '2':
      {
        char curr_dir[256];
        char spath[256]="codebook.pfm";
        OPENFILENAME ofn = { 0 };
        ofn.lpstrTitle = "Save codebook as ...";
        ofn.lStructSize  = sizeof(OPENFILENAME);
        ofn.lpstrFilter  = "portable float map (*.pfm)\0*.pfm\0";
        ofn.nFilterIndex = 1;
        ofn.lpstrFile    = spath;
        ofn.nMaxFile     = 256;
        ofn.Flags        = OFN_HIDEREADONLY | OFN_OVERWRITEPROMPT;
        _getcwd( curr_dir, 256 );
        if( GetSaveFileName(&ofn) )
        {
          _chdir(curr_dir);
          som.gpulbg_gpubook.save( spath );
        }
      }
      break;
    case '3':
      {
        char spath[256]="codeword.pfm";
        char curr_dir[256];
        OPENFILENAME ofn = { 0 };
        ofn.lpstrTitle = "Save codeword as ...";
        ofn.lStructSize  = sizeof(OPENFILENAME);
        ofn.lpstrFilter  = "portable float map (*.pfm)\0*.pfm\0";
        ofn.nFilterIndex = 1;
        ofn.lpstrFile    = spath;
        ofn.nMaxFile     = 256;
        ofn.Flags        = OFN_HIDEREADONLY | OFN_OVERWRITEPROMPT;
        _getcwd( curr_dir, 256 );
        if( GetSaveFileName(&ofn) )
        {
          GPfm gpucodeword;
          gpucodeword.load( som.gpu_src.w, som.gpu_src.h );
          glBindTexture(GL_TEXTURE_2D, som.vis_vcodeword_id );
          glGetTexImage( GL_TEXTURE_2D, 0, GL_RGB, GL_FLOAT, gpucodeword.fm );
          gpucodeword.flip_vertical();
          gpucodeword.save( spath );

          _chdir(curr_dir);
        }
      }
      break;

    case '4':
      som.prepare_codebook();
      som.init();
      update_som_model();
      glutPostRedisplay();
      break;

    case '5':
      {
        char spath[256]="decode.pfm";
        char curr_dir[256];
        OPENFILENAME ofn = { 0 };
        ofn.lpstrTitle = "Save decoded image as ...";
        ofn.lStructSize  = sizeof(OPENFILENAME);
        ofn.lpstrFilter  = "portable float map (*.pfm)\0*.pfm\0";
        ofn.nFilterIndex = 1;
        ofn.lpstrFile    = spath;
        ofn.nMaxFile     = 256;
        ofn.Flags        = OFN_HIDEREADONLY | OFN_OVERWRITEPROMPT;
        _getcwd( curr_dir, 256 );
        if( GetSaveFileName(&ofn) )
        {
          GPfm decoded;
          decoded.load( som.gpu_src.w, som.gpu_src.h );
          glBindTexture(GL_TEXTURE_2D, som.vis_des_id );
          glGetTexImage( GL_TEXTURE_2D, 0, GL_RGB, GL_FLOAT, decoded.fm );
          decoded.flip_vertical();
          decoded.save( spath );

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
        glutReshapeWindow(512, 512);
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
      glutAddMenuEntry( "space - toggle the training", ' ' );
      glutAddMenuEntry( "\\ - reset codebook", '\\' );
      glutAddMenuEntry( "4 - change initial codebook", '4' );
      glutAddMenuEntry( "---------------------------", 1 );
      glutAddMenuEntry( "1 - load source image ...", '1' );
      glutAddMenuEntry( "2 - save codebook ...", '2' );
      glutAddMenuEntry( "3 - save codeword ...", '3' );
      glutAddMenuEntry( "5 - save decoded image ...", '5' );

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



void draw_info_text()
{
  GLint viewport[4];
    glGetIntegerv( GL_VIEWPORT, viewport );

  glPrintf( ">>glut" );
  if( som.gpulbg_j==som.gpulbg_max_cycles )
    glPrintf( "Progress:    DONE    [press 'space' to restart]\n" );
  else if( btraining )
  {
    glPrintf( "Progress: " );
    float s;
    s = float(clock()%1000)/1000;
    s = fabsf(2*s-1);
    FLOAT3 col;
    col = g_lerp( FLOAT3(0,1,0), FLOAT3(1,.1,.1), s );
    glColor3fv( (float*)&col );
    glPrintf( "TRAINING" );
    glColor3f( 1,1,1 );
    glPrintf( " [press 'space' to pause]\n" );
  }else
    glPrintf( "Progress:   PAUSED   [press 'space' to resume]\n" );
  glPrintf( "Iteration:  %i/%i\n", som.gpulbg_j, som.gpulbg_max_cycles );
  
  glColor3f( .7,.7,1 );
  glPrintf( "[Drag the window to rotate]\n" );
  glPrintf( "[Open right-click menu for more options]\n" );
  glColor3f( 1,1,1 );

  glColor3f( 1,1,0 );
  static bool first_training=true;
  if( som.gpulbg_j==som.gpulbg_max_cycles )
    first_training=false;
  if( first_training )
  {
  glPrintf( ">>glut(%i,%i)", viewport[2]/2-130, viewport[3]/2-16 );
  glPrintf( "The visualization of codebook topology\n" );
  }

    glPrintf( ">>glut(%i,%i)", 0, viewport[3]-24 );
    glPrintf( "[The images at the bottom are source, codebook and codewords]\n" );

  glColor3f( 1,1,1 );
}

void draw_info_images()
{
  GLint viewport[4];
    glGetIntegerv( GL_VIEWPORT, viewport );

  glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D( 0, 1, 0, double(viewport[3])/viewport[2] );
  glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

  glTranslatef( .01, 24.f/viewport[3],0 );

  glBindTexture( GL_TEXTURE_2D, som.vis_src_id );
  glEnable( GL_TEXTURE_2D );
  glBegin( GL_QUADS );
    glTexCoord2f(0,0);  glVertex2f(0,0);
    glTexCoord2f(0,1);  glVertex2f(0,.2);
    glTexCoord2f(1,1);  glVertex2f(float(som.gpu_src.w)/som.gpu_src.h*.2f,.2);
    glTexCoord2f(1,0);  glVertex2f(float(som.gpu_src.w)/som.gpu_src.h*.2f, 0);
  glEnd();
  glDisable( GL_TEXTURE_2D );

  glTranslatef( float(som.gpu_src.w)/som.gpu_src.h*.2f+.05, 0,0 );

  glBindTexture(GL_TEXTURE_2D, som.vis_codebook_id );
  glTexImage2D( GL_TEXTURE_2D, 0, GL_RGB32F_ARB, som.gpulbg_gpubook.w, som.gpulbg_gpubook.h, 0, GL_RGB, GL_FLOAT, som.gpulbg_gpubook.fm );
  glEnable( GL_TEXTURE_2D );
  glBegin( GL_QUADS );
    glTexCoord2f(0,0);  glVertex2f(0,0);
    glTexCoord2f(0,1);  glVertex2f(0,.2);
    glTexCoord2f(1,1);  glVertex2f(.2,.2);
    glTexCoord2f(1,0);  glVertex2f(.2,0);
  glEnd();
  glDisable( GL_TEXTURE_2D );

  glTranslatef( .2f+.05f, 0,0 );

  glBindTexture( GL_TEXTURE_2D, som.vis_vcodeword_id );
  glEnable( GL_TEXTURE_2D );
  glBegin( GL_QUADS );
    glTexCoord2f(0,0);  glVertex2f(0,0);
    glTexCoord2f(0,1);  glVertex2f(0,.2);
    glTexCoord2f(1,1);  glVertex2f(float(som.gpu_src.w)/som.gpu_src.h*.2f,.2);
    glTexCoord2f(1,0);  glVertex2f(float(som.gpu_src.w)/som.gpu_src.h*.2f, 0);
  glEnd();
  glDisable( GL_TEXTURE_2D );
}



void init_som_model()
{
  GObj &obj = som_model;
  GPfm codebook;
  int i, j, fi;
    
  codebook.load( som.gpulbg_gpubook.w, som.gpulbg_gpubook.h, som.gpulbg_gpubook.fm );
  codebook.resample( codebook.w*4, codebook.h*4 );
  obj.load_face( (codebook.w-1)*(codebook.h-1)*2 );
  obj.load_vertex( codebook.w*codebook.h );
  obj.default_group.type = GOBJ_VERTEX;
  for( j=0, fi=0; j<codebook.h-1; j++ )
    for( i=0; i<codebook.w-1; i++ )
    {
      obj.face_vidx[3*fi+0] = (j+0)*codebook.w+(i+0);
      obj.face_vidx[3*fi+1] = (j+0)*codebook.w+(i+1);
      obj.face_vidx[3*fi+2] = (j+1)*codebook.w+(i+1);
      obj.default_group.add(fi);
      fi++;
      obj.face_vidx[3*fi+0] = (j+0)*codebook.w+(i+0);
      obj.face_vidx[3*fi+1] = (j+1)*codebook.w+(i+1);
      obj.face_vidx[3*fi+2] = (j+1)*codebook.w+(i+0);
      obj.default_group.add(fi);
      fi++;
    }
}

void update_som_model()
{
  GObj &obj = som_model;
  GPfm codebook;
  codebook.load( som.gpulbg_gpubook.w, som.gpulbg_gpubook.h, som.gpulbg_gpubook.fm );
  codebook.resample( codebook.w*4, codebook.h*4 );

  codebook.load( som.gpulbg_gpubook.w, som.gpulbg_gpubook.h, som.gpulbg_gpubook.fm );
  codebook.resample( codebook.w*4, codebook.h*4 );
  obj.vertex.load( codebook.w*codebook.h, 1, codebook.fm );
  obj.calculate_face_normal();
  obj.calculate_vertex_normal(89);
  obj.unitize();
}

void draw_info_model( float *mo )
{
  glEnable( GL_DEPTH_TEST );
  glEnable( GL_LIGHTING );
  glEnable( GL_LIGHT0 );
    glPushMatrix();
    glMultMatrixf( mo );
    shader_lighting_begin();
      som_model.draw();
    shader_lighting_end();
    glPopMatrix();
  glDisable( GL_DEPTH_TEST );
  glDisable( GL_LIGHTING );
  glDisable( GL_LIGHT0 );
}
