#include <windows.h>
#include <stdio.h>
#include <math.h>

#include <GL/glut.h>

#include "g_common.h"
#include "g_bell.h"

#define GBELL_WORLDMAP_PATH "../../data/worldmap3.bmp"
#define GBELL_WORLDMAP_PATH2 "worldmap3.bmp"

#define GBELL_KEY_QL1 GLUT_KEY_F1
#define GBELL_KEY_QL2 GLUT_KEY_F2
#define GBELL_KEY_QL3 GLUT_KEY_F3
#define GBELL_KEY_QL4 GLUT_KEY_F4

#define GBELL_KEY_QS1 GLUT_KEY_F5
#define GBELL_KEY_QS2 GLUT_KEY_F6
#define GBELL_KEY_QS3 GLUT_KEY_F7
#define GBELL_KEY_QS4 GLUT_KEY_F8

#define GBELL_KEY_RESET 27
#define GBELL_KEY_LIGHT '1'
#define GBELL_KEY_OBJ   '2'
#define GBELL_KEY_VIEW  '3'
#define GBELL_KEY_ENV   '4'
#define GBELL_KEY_PICK   '5'
#define GBELL_KEY_FORWARD    ']'
#define GBELL_KEY_BACKWARD   '['
#define GBELL_KEY_SCN  's'
#define GBELL_KEY_ANI  ' '
#define GBELL_KEY_PAUSE  'p'
#define GBELL_KEY_PRINTT  '\r'

#define GBELL_FRAME_RATE 60
#define GBELL_DELTA_TIME .03


int GBell::N_GBELL = 0;
GBell* GBell::GBELL_LIST[16];

int GBell::N_DOCK = 0;
GBell* GBell::DOCK_LIST[16];

GBell* GBell::ani_bell = NULL;

GLuint GBell::worldmap = 0;

bool BELL_ANI_ON = false;
bool BELL_ANI_PAUSE = false;

GBell::GBell()
{
  bell_win = 0;     // handle of the arcbell window
  fellow_win = 0;   // handle of the fellow window
  fellow_state = 0;
  drag_state = -1;
  active_ctrl = 0;

  PickFunc = NULL;
  PickChangeFunc = NULL;
  active_tram = NULL;
  prev_tram = NULL;

  obj_list = 0;
  light_list = 0;
  view_list = 0;
  env_list = 0;
}

void GBell::set_active_ctrl( int bell_ctrl )
{
  if( fellow_state & bell_ctrl )
    active_ctrl = fellow_state & bell_ctrl;
  else
  {
    //printf( "[Warning] GBell::set_active_ctrl, operation discarded.\n" );
    return;
  }

  int t_win = glutGetWindow();
  if( fellow_win )
  {
    glutSetWindow( fellow_win );
    glutPostRedisplay();
    glutSetWindow( bell_win );
    glutPostRedisplay();
  }
  glutSetWindow( t_win );
}

void GBell::set_fellow( int win_id, int state )
{
  fellow_win = win_id ? win_id : glutGetWindow();
  fellow_state = state;

  int hmenu_sub, hmenu;
  int t_win = glutGetWindow();
  glutSetWindow(bell_win);

    hmenu = glutCreateMenu(bell_menu);

      hmenu_sub = glutCreateMenu(bell_menu);
        if( fellow_state & GBELL_LIGHT )
        {
          glutAddMenuEntry( "1 - control light", GBELL_KEY_LIGHT );
          active_ctrl = GBELL_LIGHT;
        }

        if( fellow_state & GBELL_OBJ )
        {
          glutAddMenuEntry( "2 - control object", GBELL_KEY_OBJ );
          active_ctrl = GBELL_OBJ;
        }

        if( fellow_state & GBELL_VIEW )
        {
          glutAddMenuEntry( "3 - control view", GBELL_KEY_VIEW );
          active_ctrl = GBELL_VIEW;
        }

        if( fellow_state & GBELL_ENV )
        {
          glutAddMenuEntry( "4 - control environment", GBELL_KEY_ENV );
          active_ctrl = GBELL_ENV;

          if( worldmap == 0 )
          {
            GPfm pfm;
              if( fexist(GBELL_WORLDMAP_PATH) )
                pfm.load( GBELL_WORLDMAP_PATH );
              else if( fexist(GBELL_WORLDMAP_PATH2) )
                pfm.load( GBELL_WORLDMAP_PATH2 );
              else
              {
                //printf( "[ERROR] GBell::set_fellow(), worldmap3.bmp is not found\n" );
                //exit(-1);
                float cb[] =
                {
                  1,1,1,1,0,
                  1,0,0,0,1,
                  1,1,1,1,0,
                  1,0,0,0,1,
                  1,0,0,0,1,
                  1,1,1,1,0,
                };
                float ce[] =
                {
                  1,1,1,1,1,
                  1,1,0,0,0,
                  1,1,1,1,0,
                  1,1,1,1,0,
                  1,1,0,0,0,
                  1,1,1,1,1,
                };
                float cl[] =
                {
                  1,1,0,0,0,
                  1,1,0,0,0,
                  1,1,0,0,0,
                  1,1,0,0,0,
                  1,1,1,1,1,
                  1,1,1,1,1,
                };

                float *ckey[4] = { cb, ce, cl, cl };
                FLOAT3 col[4] = 
                {
                  FLOAT3(   1, 0, 0 ),
                  FLOAT3( 0,   1, 0 ),
                  FLOAT3( 0, 0,   1 ),
                  FLOAT3( 0,   1,   1 ),
                };
                pfm.load( 128, 164 );

                int i, j;
                GPfm g;
                for( j=0; j<4; j++ )
                {
                  for( i=0; i<4; i++ )
                  {
                    g.load( 5,6, ckey[i] );
                    g.scale( g.w*4, g.h*4 );
                    g.mul( col[j]+.33*i );
                    pfm.draw( g, 32*j, 24 + 28*i);
                  }
                }
              }
            pfm.scale( 256, 128 );
            pfm.flip_vertical();
            glGenTextures( 1, &worldmap );
              glBindTexture(GL_TEXTURE_2D, worldmap);
              glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
              glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
              glTexImage2D( GL_TEXTURE_2D, 0, GL_RGB, pfm.w, pfm.h, 0, GL_RGB, GL_FLOAT, pfm.fm );
          }
        }

        if( fellow_state & GBELL_PICK )
        {
          glutAddMenuEntry( "5 - control pick", GBELL_KEY_PICK );
          active_ctrl = GBELL_PICK;
          if( glutGet( GLUT_WINDOW_ALPHA_SIZE )!=8 )
          {
            printf( 
              "[Error] GBell::set_fellow(), glut display mode must "
              "have GLUT_ALPHA in order to use PICK function.\n"
              );
            exit(-1);
          }
        }

        glutAddMenuEntry( "esc - reset", GBELL_KEY_RESET );
        glutAddMenuEntry( "r - rotate along axis", 'r' );
        glutAddMenuEntry( "enter - print transform", GBELL_KEY_PRINTT );
        glutSetMenu( hmenu );
        glutAddSubMenu( "control", hmenu_sub );


      hmenu_sub = glutCreateMenu(bell_menu);
        glutAddMenuEntry( "[ - backward frame", GBELL_KEY_BACKWARD );
        glutAddMenuEntry( "] - forward frame", GBELL_KEY_FORWARD );
        glutAddMenuEntry( "space - play/stop", GBELL_KEY_ANI );
        glutAddMenuEntry( "p - play/pause", GBELL_KEY_PAUSE );
        glutAddMenuEntry( "s - save frame", GBELL_KEY_SCN );
        glutSetMenu( hmenu );
        glutAddSubMenu( "frame", hmenu_sub );

      
      hmenu_sub = glutCreateMenu(bell_menu);
        glutAddMenuEntry( "F1 - load view 1", GBELL_KEY_QL1 );
        glutAddMenuEntry( "F2 - load view 2", GBELL_KEY_QL2 );
        glutAddMenuEntry( "F3 - load view 3", GBELL_KEY_QL3 );
        glutAddMenuEntry( "F4 - load view 4", GBELL_KEY_QL4 );
        glutSetMenu( hmenu );
        glutAddSubMenu( "load", hmenu_sub );

      hmenu_sub = glutCreateMenu(bell_menu);
        glutAddMenuEntry( "F5 - save view 1", GBELL_KEY_QS1 );
        glutAddMenuEntry( "F6 - save view 2", GBELL_KEY_QS2 );
        glutAddMenuEntry( "F7 - save view 3", GBELL_KEY_QS3 );
        glutAddMenuEntry( "F8 - save view 4", GBELL_KEY_QS4 );
        glutSetMenu( hmenu );
        glutAddSubMenu( "save", hmenu_sub );

    glutAttachMenu(GLUT_RIGHT_BUTTON);

    if( fellow_state & GBELL_DOCK )
    {
      set_dock( this );
      glutSetWindow(fellow_win);
      glutMouseFunc( bell_mousebutton );
      glutMotionFunc( bell_mousemove );
    }

  glutSetWindow(t_win);
}

void GBell::set_pick( void (*pick_function)(int tidx) )
{
  PickFunc = pick_function;
}

void GBell::set_pickchange( void (*pickchange_function)( const GTram *tram0, const GTram *tram1 ) )
{
  PickChangeFunc = pickchange_function;
}

void GBell::select_none()
{
  SAFE_FREE( active_tram );
  SAFE_FREE( prev_tram );
}

void GBell::select_tram( int tidx, int v0, int v1, int v2, float vw0, float vw1, float vw2 )
{
  SAFE_FREE( active_tram );
  active_tram = (GTram*)malloc( sizeof(GTram) );
  active_tram->tidx = tidx;
  active_tram->vidx[0] = v0;
  active_tram->vidx[1] = v1;
  active_tram->vidx[2] = v2;
  active_tram->vw[0] = vw0;
  active_tram->vw[1] = vw1;
  active_tram->vw[2] = vw2;
}


void GBell::process_pick( int x, int y )
{
  SAFE_FREE( active_tram );

  GLint viewport[4];
    glGetIntegerv( GL_VIEWPORT, viewport );
  if( x<0 || y<0 || x>=viewport[2] || y>=viewport[3] )
    return;

  GTram &buf_tram = *( (GTram*)malloc( sizeof(GTram) ) );

  glPushAttrib( GL_LIGHTING_BIT | GL_COLOR_BUFFER_BIT );
  glClearColor( 1,1,1,1 );
  glDisable( GL_LIGHTING );
  PickFunc(-1);
  glReadPixels( x, y, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, &buf_tram.tidx );

  if( buf_tram.tidx!=0xFFFFFFFF )
  {
    PickFunc(buf_tram.tidx);
    glReadPixels( x, y, 1, 1, GL_RGB, GL_FLOAT, buf_tram.vw );
    quick_sort( buf_tram.vw, buf_tram.vidx, 3, 0 );
    float ss = buf_tram.vw[0]+buf_tram.vw[1]+buf_tram.vw[2];
    buf_tram.vw[0] /= ss;
    buf_tram.vw[1] /= ss;
    buf_tram.vw[2] /= ss;
    active_tram = &buf_tram;
  }else
  {
    delete &buf_tram;
  }

  glPopAttrib();

  if( PickChangeFunc )
  if( 
    (!prev_tram && active_tram) ||
    (prev_tram && !active_tram) ||
    ( 
      prev_tram && active_tram && 
      ( prev_tram->tidx  != active_tram->tidx  || 
        prev_tram->vw[0] != active_tram->vw[0] || 
        prev_tram->vw[1] != active_tram->vw[1] || 
        prev_tram->vw[2] != active_tram->vw[2] )
    )
  ){
    int hwin = glutGetWindow();
    glutSetWindow( fellow_win );
    PickChangeFunc( prev_tram, active_tram );
    glutSetWindow( hwin );
  }

  SAFE_FREE( prev_tram );
  if( active_tram )
  {
    prev_tram = (GTram*)malloc( sizeof(GTram) );
    memcpy( prev_tram, active_tram, sizeof(GTram) );
  }
}

GBell::~GBell()
{
  del_bell( this );
  del_dock( this );

  if( ani_bell == this )
    ani_bell = NULL;

  SAFE_FREE( active_tram );
  SAFE_FREE( prev_tram );
}

void GBell::save( const char *spath )
{
  FILE *f0 = fopen( spath, "wt" );
    fprintf( f0, "%f %f %f %f\n", qLight.x, qLight.y, qLight.z, qLight.w );
    fprintf( f0, "%f %f %f %f\n", qObj.x, qObj.y, qObj.z, qObj.w );
    fprintf( f0, "%f %f %f %f\n", qView.x, qView.y, qView.z, qView.w );
    fprintf( f0, "%f %f %f %f\n", qEnv.x, qEnv.y, qEnv.z, qEnv.w );
    //fwrite( &qLight, sizeof(GQuat), 1, f0 );
    //fwrite( &qObj,   sizeof(GQuat), 1, f0 );
    //fwrite( &qView,  sizeof(GQuat), 1, f0 );
    //fwrite( &qEnv,   sizeof(GQuat), 1, f0 );
  fclose(f0);
}

void GBell::load_smap( const GPfm &smap )
{
  int t_win = glutGetWindow();
  glutSetWindow(bell_win);

  GPfm sm_src;
    smap.scale( sm_src, 256, 128 );
    sm_src.flip_vertical();

    {
      FLOAT3 vmean = sm_src.vmean();
      float smean = vmean.norm();
      int i,j;
      for( j=0; j<sm_src.h; j++ )
        for( i=0; i<sm_src.w; i++ )
        {
          FLOAT3 &c = sm_src.pm[j][i];
          if( c.norm() < smean )
            c = 0;
        }
    }

  glBindTexture(GL_TEXTURE_2D, worldmap);
  glTexImage2D( GL_TEXTURE_2D, 0, GL_RGB, sm_src.w, sm_src.h, 0, GL_RGB, GL_FLOAT, sm_src.fm );

  glutPostRedisplay();

  glutSetWindow(t_win);
}

void GBell::load_smap( const char *spath )
{
  GPfm smap;
    smap.load( spath );
    load_smap( smap );
}


void GBell::load( const char *spath )
{
  int t_win = glutGetWindow();
  
  FILE *f0 = fopen( spath, "rb" );
    if( f0==NULL ) return;
    fscanf( f0, "%f %f %f %f\n", &qLight.x, &qLight.y, &qLight.z, &qLight.w );
    fscanf( f0, "%f %f %f %f\n", &qObj.x, &qObj.y, &qObj.z, &qObj.w );
    fscanf( f0, "%f %f %f %f\n", &qView.x, &qView.y, &qView.z, &qView.w );
    fscanf( f0, "%f %f %f %f\n", &qEnv.x, &qEnv.y, &qEnv.z, &qEnv.w );
  fclose(f0);

    qLight0 = qLight;
    qObj0   = qObj;
    qView0  = qView;
    qEnv0   = qEnv;

  if( bell_win )
  {
    glutSetWindow(bell_win);
    glutPostRedisplay();
  }
  if( fellow_win )
  {
  glutSetWindow(fellow_win);
  glutPostRedisplay();
  }

  if( t_win )
  glutSetWindow(t_win);
}

void GBell::create_subwindow()
{
  make_window( "GBELL_SUBWINDOW" );
}

void GBell::create_window( const char *title, int width, int height )
{
  make_window( title, width, height );
}

void GBell::make_window( const char *title, int width, int height )
{
  if( bell_win )
    return;

  char bell_title[256];

  if( title==NULL )
    sprintf( bell_title, "GBell#%02i", N_GBELL );
  else
    strcpy( bell_title, title );
  bell_w = width;
  bell_h = height;

  int curr_win, curr_x, curr_y, curr_w, curr_h;
    curr_win = glutGetWindow();
    curr_x = glutGet( GLUT_WINDOW_X );
    curr_y = glutGet( GLUT_WINDOW_Y );
    curr_w = glutGet( GLUT_WINDOW_WIDTH );
    curr_h = glutGet( GLUT_WINDOW_HEIGHT );

  if( strcmp( bell_title, "GBELL_SUBWINDOW" )==0 )
  {
    bell_win = glutCreateSubWindow( curr_win, 0, 0, bell_w, bell_h ); 
    glutReshapeWindow( bell_w, bell_h );
    glutPositionWindow( 0, -bell_h+16*(N_GBELL+1));
  }else
  {
    bell_win = glutCreateWindow( bell_title );
    glutReshapeWindow( bell_w, bell_h );
    glutPositionWindow( curr_x+curr_w+8, curr_y+(bell_h+28)*N_GBELL);
  }

  glutMouseFunc(bell_mousebutton);
  glutMotionFunc(bell_mousemove);
  glutDisplayFunc(bell_display);
  glutReshapeFunc(bell_reshape);
  glutKeyboardFunc(bell_keyboard);
  glutSpecialFunc(bell_special);

  glEnable(GL_DEPTH_TEST);

  GBELL_LIST[N_GBELL] = this;
  N_GBELL++;

  if( curr_win )
    glutSetWindow( curr_win );
}


void GBell::bell_mousebutton(int button, int state, int x, int y )
{
  GBell *gb;
  int hwin = glutGetWindow();

  if( gb=get_dock(hwin) )
  {
  }else
  {
    gb = get_bell( hwin );
    if(gb==NULL) return;
    set_dock( gb );
  }

  int w,h;
    w = glutGet( GLUT_WINDOW_WIDTH );
    h = glutGet( GLUT_WINDOW_HEIGHT );

  if( button == GLUT_LEFT_BUTTON )
  {
    if( state == GLUT_DOWN )
    {

        gb->vFrom = MouseOnSphere( x,y, w,h );
        gb->vTo = gb->vFrom;
//        gb->qDrag = GQuat();
        gb->drag_state = GLUT_LEFT_BUTTON;

        if( gb->fellow_win && (gb->fellow_state & GBELL_DOWN) )
        {
          glutSetWindow(gb->fellow_win);
          if( gb->active_ctrl & GBELL_PICK )
          {
            gb->process_pick(x,h-y);
          }
          glutPostRedisplay();
        }
        bell_timer(0);
        glutSetWindow(gb->bell_win);
        glutPostRedisplay();
    }

    if( state == GLUT_UP )
    {
      if( gb->vTo.x==gb->vFrom.x && gb->vTo.y==gb->vFrom.y && gb->vTo.z==gb->vFrom.z )
      {
        BELL_ANI_PAUSE = !BELL_ANI_PAUSE;
        bell_timer(!BELL_ANI_PAUSE);
      }else
      {
        bell_timer( BELL_ANI_ON );
        BELL_ANI_PAUSE = false;

        // we normalize quat before assign to overall quat simply
        // to avoid propagation of numurical error.  
        // Nothing more than that.
        if( gb->active_ctrl & GBELL_LIGHT )
          gb->qLight0 = gb->qLight.normalize();
        
        if( gb->active_ctrl & GBELL_OBJ )
          gb->qObj0 = gb->qObj.normalize();
        
        if( gb->active_ctrl & GBELL_VIEW )
          gb->qView0 = gb->qView.normalize();
        
        if( gb->active_ctrl & GBELL_ENV )
          gb->qEnv0 = gb->qEnv.normalize();
      }


      gb->drag_state = -1;

      glutSetWindow(gb->bell_win);
      glutPostRedisplay();

      if( gb->fellow_win && (gb->fellow_state & GBELL_UP) )
      {
        glutSetWindow(gb->fellow_win);
        if( gb->active_ctrl & GBELL_PICK )
        {
          gb->process_pick(x,h-y);
        }
        glutPostRedisplay();
      }
    }
  }
  glutSetWindow(hwin);

}

void GBell::bell_mousemove(int x, int y)
{

  GBell *gb;
  int hwin = glutGetWindow();

  if( gb=get_dock(hwin) )
  {
  }else
  {
    gb = get_bell( hwin );
    if(gb==NULL) return;
    set_dock( gb );
  }

  int w,h;
    w = glutGet( GLUT_WINDOW_WIDTH );
    h = glutGet( GLUT_WINDOW_HEIGHT );

  if( gb->drag_state == GLUT_LEFT_BUTTON )
  {
      gb->vTo = MouseOnSphere( x,y, w,h );

      if( gb->vTo.x==gb->vFrom.x && gb->vTo.y==gb->vFrom.y && gb->vTo.z==gb->vFrom.z )
      {
      }else      
      {
        if(get_dock(hwin) )
        {
          float mv[16];
          gb->qView0.matrix( mv );
          gb->qDrag = GQuat( gb->vTo.rmul( mv ), gb->vFrom.rmul( mv ) );

          if( gb->active_ctrl == GBELL_VIEW )
            gb->qDrag = ~gb->qDrag;
        }else
        {
          gb->qDrag = GQuat( gb->vTo, gb->vFrom );
        }

        // ATTENTION :
        //
        // Quaternion multiplication DO NOT COMMUTE, therefore
        // the order of multiplication is IMPORTANT.
        //
        // In case of OpenGL, row vector assumed, next transform matrix
        // is concatenated on the right hand side, ie, M0 * M1 * M2 * ...
        //
        // As Quaternion is just a fancy representation of rotation matrix 
        // only, it should be concatenated in the same fashion as 
        // ordinary tranform matrix.
        //
        // For OpenGL, right handed concatenation please.  ^O^ok
        //
        if( gb->active_ctrl & GBELL_LIGHT )
          gb->qLight = gb->qLight0 * gb->qDrag;
    
        if( gb->active_ctrl & GBELL_OBJ )
          gb->qObj = gb->qObj0 * gb->qDrag;
    
        if( gb->active_ctrl & GBELL_VIEW )
          gb->qView = gb->qView0 * gb->qDrag;
    
        if( gb->active_ctrl & GBELL_ENV )
          gb->qEnv = gb->qEnv0 * gb->qDrag;
      }

    glutSetWindow(gb->bell_win);
    glutPostRedisplay();
  
    if( gb->fellow_win && (gb->fellow_state & GBELL_DRAG) )
    {
      glutSetWindow(gb->fellow_win);

      if( gb->active_ctrl & GBELL_PICK )
      {
        gb->process_pick(x,h-y);
      }

      glutPostRedisplay();
    }
  }
  glutSetWindow(hwin);
}

//
// The routine for handle animination here is messy.  Maybe
// we can fix it later on...
//
void GBell::bell_timer(int value)
{
  static int ani_on = 0;

  if( ani_on == value )
    return;

  if( value != -1 )
    ani_on = value;

  if( ani_bell && ani_on )
  {
    ani_bell->bell_frame(GBELL_DELTA_TIME);
    glutTimerFunc( 1000.f/GBELL_FRAME_RATE, bell_timer, -1 );
  }
}

void GBell::bell_keyboard( unsigned char key, int x, int y )
{
  GBell *gb = get_bell( glutGetWindow() );
  if(gb==NULL) return;

  switch( key )
  {
    case '`':
      if( gb->fellow_state & GBELL_PICK )
        gb->set_active_ctrl( GBELL_PICK );
      if( gb->fellow_state & GBELL_ENV )
        gb->set_active_ctrl( GBELL_ENV );
      if( gb->fellow_state & GBELL_VIEW )
        gb->set_active_ctrl( GBELL_VIEW );
      if( gb->fellow_state & GBELL_OBJ )
        gb->set_active_ctrl( GBELL_OBJ );
      if( gb->fellow_state & GBELL_LIGHT )
        gb->set_active_ctrl( GBELL_LIGHT );
    break;

    case '!':
      gb->set_active_ctrl( gb->active_ctrl | GBELL_LIGHT );
    break;
    case '@':
      gb->set_active_ctrl( gb->active_ctrl | GBELL_OBJ );
    break;
    case '#':
      gb->set_active_ctrl( gb->active_ctrl | GBELL_VIEW );
    break;
    case '$':
      gb->set_active_ctrl( gb->active_ctrl | GBELL_ENV );
    break;
    case '%':
      gb->set_active_ctrl( gb->active_ctrl | GBELL_PICK );
    break;

    case GBELL_KEY_LIGHT:
      gb->set_active_ctrl( GBELL_LIGHT );
    break;
    case GBELL_KEY_OBJ:
      gb->set_active_ctrl( GBELL_OBJ );
    break;
    case GBELL_KEY_VIEW:
      gb->set_active_ctrl( GBELL_VIEW );
    break;
    case GBELL_KEY_ENV:
      gb->set_active_ctrl( GBELL_ENV );
    break;
    case GBELL_KEY_PICK:
      gb->set_active_ctrl( GBELL_PICK );
    break;

    case GBELL_KEY_FORWARD:
      gb->bell_frame(GBELL_DELTA_TIME);
      break;
    case GBELL_KEY_BACKWARD:
      gb->bell_frame(-GBELL_DELTA_TIME);
      break;
    case GBELL_KEY_PRINTT:
      {
        int i;
        int all_ctrl[] = { GBELL_LIGHT, GBELL_OBJ, GBELL_VIEW, GBELL_ENV };
        char all_ctrl_str[4][16] = { "GBELL_LIGHT", "GBELL_OBJ", "GBELL_VIEW", "GBELL_ENV" };
        float m[16];

        for( i=0; i<sizeof(all_ctrl)/sizeof(all_ctrl[0]); i++ )
        {
          if( gb->active_ctrl & all_ctrl[i] )
          {
            gb->get_matrix( m, all_ctrl[i] );
            printf( "%s\n", all_ctrl_str[i] );
            printf( "%9.6f, %9.6f, %9.6f, %9.6f,\n", m[ 0], m[ 1], m[ 2], m[ 3] );
            printf( "%9.6f, %9.6f, %9.6f, %9.6f,\n", m[ 4], m[ 5], m[ 6], m[ 7] );
            printf( "%9.6f, %9.6f, %9.6f, %9.6f,\n", m[ 8], m[ 9], m[10], m[11] );
            printf( "%9.6f, %9.6f, %9.6f, %9.6f,\n", m[12], m[13], m[14], m[15] );
            printf( "\n" );
          }
        }
      }
      break;
    case GBELL_KEY_RESET:
      gb->reset();
      break;
    case GBELL_KEY_SCN:
        gb->save_screen();
      break;

    case GBELL_KEY_ANI:
      {
        BELL_ANI_PAUSE = false;
        BELL_ANI_ON = !BELL_ANI_ON;
        ani_bell = BELL_ANI_ON ? gb : NULL;
        bell_timer(BELL_ANI_ON);
        glutPostRedisplay();
      }
      break;

    case GBELL_KEY_PAUSE:
        BELL_ANI_PAUSE = !BELL_ANI_PAUSE;
        bell_timer(!BELL_ANI_PAUSE);
        glutPostRedisplay();
      break;

    case 'r':
    case 'R':
      if( gb->drag_state == -1 )
      {

        float theta;
          theta = ( key=='r'? -1 : 1  ) * 5 * G_PI/180;

        float m[16];
        GQuat qt;
  
        if( gb->active_ctrl & GBELL_LIGHT )
        {
          gb->get_matrix( m, GBELL_LIGHT );
          qt = GQuat( FLOAT3(m[8],m[9],m[10]), theta );
          gb->qLight = gb->qLight0 = gb->qLight * qt;
        }
    
        if( gb->active_ctrl & GBELL_OBJ )
        {
          gb->get_matrix( m, GBELL_OBJ );
          qt = GQuat( FLOAT3(m[4],m[5],m[6]), theta );
          gb->qObj = gb->qObj0 = gb->qObj  * qt;
        }

        if( gb->active_ctrl & GBELL_VIEW )
        {
          gb->get_matrix( m, GBELL_VIEW );
          qt = GQuat( FLOAT3(m[8],m[9],m[10]), theta );
          gb->qView = gb->qView0 = gb->qView  * qt;
        }
  
        if( gb->active_ctrl & GBELL_ENV )
        {
          gb->get_matrix( m, GBELL_ENV );
          qt = GQuat( FLOAT3(m[4],m[5],m[6]), theta );
          gb->qEnv = gb->qEnv0 = gb->qEnv  * qt;      
        }

        glutPostRedisplay();

        if( gb->fellow_win && (gb->fellow_state & GBELL_DRAG) )
        {
          glutSetWindow(gb->fellow_win);
          glutPostRedisplay();
          glutSetWindow(gb->bell_win);
        }
      }
     break;

  };
}

void GBell::bell_special( int key, int x, int y )
{
  GBell *gb = get_bell( glutGetWindow() );
  if(gb==NULL) return;

  switch( key )
  {
    case GBELL_KEY_QS1:
      gb->save( "bell_save01.txt" );
      break;
    case GBELL_KEY_QS2:
      gb->save( "bell_save02.txt" );
      break;
    case GBELL_KEY_QS3:
      gb->save( "bell_save03.txt" );
      break;
    case GBELL_KEY_QS4:
      gb->save( "bell_save04.txt" );
      break;
    case GBELL_KEY_QL1:
      gb->load( "bell_save01.txt" );
      break;
    case GBELL_KEY_QL2:
      gb->load( "bell_save02.txt" );
      break;
    case GBELL_KEY_QL3:
      gb->load( "bell_save03.txt" );
      break;
    case GBELL_KEY_QL4:
      gb->load( "bell_save04.txt" );
      break;
  };
}

void GBell::PostRedisplay()
{
  int hwin = glutGetWindow();
  glutSetWindow(bell_win);
  glutPostRedisplay();
  glutSetWindow(hwin);
}

void GBell::bell_frame( float delta_time )
{
  if( drag_state == -1 && delta_time!=0 )
  {
    int hwin = glutGetWindow();
    int i;

      GQuat q, qd, qf;
        if( delta_time > 0 )
          qd = qDrag;
        else
          qd = ~qDrag;
      delta_time = fabs(delta_time);

      qf = GQuat::slerp( GQuat(), qd, delta_time-int(delta_time) );
      for( i=0; i<int(delta_time); i++ )
        q = q * qd;
        q = q * qf;


      if( active_ctrl & GBELL_LIGHT )
      {
        qLight = qLight * q;
        qLight0 = qLight.normalize();
      }
      
      if( active_ctrl & GBELL_OBJ )
      {
        qObj = qObj * q;
        qObj0 = qObj.normalize();
      }
  
      if( active_ctrl & GBELL_VIEW )
      {
        qView = qView * q;
        qView0 = qView.normalize();
      }
  
      if( active_ctrl & GBELL_ENV )
      {
        qEnv = qEnv0 * q;
        qEnv0 = qEnv.normalize();
      }
  
      glutSetWindow(bell_win);
      glutPostRedisplay();

      glutSetWindow(fellow_win);
      glutPostRedisplay();
  
    glutSetWindow(hwin);
  }
}

void GBell::bell_menu( int item )
{
  GBell *gb = get_bell( glutGetWindow() );
  if(gb==NULL) return;

  switch( item )
  {
    case GBELL_KEY_LIGHT:
    case GBELL_KEY_OBJ:
    case GBELL_KEY_VIEW:
    case GBELL_KEY_ENV:
    case GBELL_KEY_PICK:
    case GBELL_KEY_RESET:
    case GBELL_KEY_FORWARD:
    case GBELL_KEY_BACKWARD:
    case GBELL_KEY_SCN:
    case GBELL_KEY_ANI:
    case GBELL_KEY_PAUSE:
    case 'r':
    case GBELL_KEY_PRINTT:
      bell_keyboard( item, 0,0 );
      break;

    case GBELL_KEY_QS1:
    case GBELL_KEY_QS2:
    case GBELL_KEY_QS3:
    case GBELL_KEY_QS4:
    case GBELL_KEY_QL1:
    case GBELL_KEY_QL2:
    case GBELL_KEY_QL3:
    case GBELL_KEY_QL4:
      bell_special( item, 0,0 );
      break;
  };
}

void GBell::setlight()
{
  float spec[] = { .8, .8, .8, 1 };
  float ambi[] = { .3, .3, .3, 1 };
  float diff[] = { .5, .5, .5, 1 };
  float shin[] = { 10 };
  float l1_color[4] = { .9,.9,.9,1 };
  float ml[16];

  qLight.matrix( ml );
  float v0[4] = {  ml[8],  ml[9],  ml[10], 0 };
  float v1[4] = { -ml[8], -ml[9], -ml[10], 0 };

  glLightfv(GL_LIGHT0, GL_POSITION, v0 );    
  glLightfv(GL_LIGHT1, GL_POSITION, v1 );
  glLightfv(GL_LIGHT1, GL_DIFFUSE, l1_color );
  
  glMaterialfv(GL_FRONT, GL_SPECULAR,  spec);
  glMaterialfv(GL_FRONT, GL_DIFFUSE,   diff);
  glMaterialfv(GL_FRONT, GL_AMBIENT,   ambi);
  glMaterialfv(GL_FRONT, GL_SHININESS, shin);

  glEnable(GL_LIGHT0);
  glEnable(GL_LIGHT1);
}

void GBell::bell_display()
{
  int hwin = glutGetWindow();
  GBell *gb = get_bell( hwin );
  if(gb==NULL) return;

    float ml[16], mo[16], mv[16], me[16];
      gb->qLight.matrix( ml );
      gb->qObj.matrix( mo );
      gb->qView.matrix( mv );
      gb->qEnv.matrix( me );

      gb->setlight();

    double bell_r = .75;
    double spot_r = .0666 * bell_r;
    double teapot_r = .333 * bell_r;

    GLuint &obj_list = gb->obj_list;
      if( obj_list == 0 )
      {
        obj_list = glGenLists(1);
        glNewList(obj_list, GL_COMPILE);
          glEnable(GL_LIGHTING);
          glutSolidTeapot( teapot_r );
          glDisable(GL_LIGHTING);
        glEndList();
      }

    GLuint &light_list = gb->light_list;
    if( light_list == 0 )
    {
      light_list = glGenLists(1);
      glNewList(light_list, GL_COMPILE);
        glEnable(GL_LIGHTING);
        glTranslatef(0.0, 0.0, bell_r);
        glutSolidSphere( spot_r, 20, 20);
        glDisable(GL_LIGHTING);
      glEndList();
    }

    GLuint &view_list = gb->view_list;
    if( view_list == 0 )
    {
      view_list = glGenLists(1);
      glNewList(view_list, GL_COMPILE);
        glEnable(GL_LIGHTING);
        glPushMatrix();
          glTranslatef(0.0, -0.5, 0.5);
          glRotatef( 90, 0,1,0 );
          glutSolidCone( .05, .1, 20, 20 );
        glPopMatrix();
        glDisable(GL_LIGHTING);

        glEnable( GL_BLEND );
        glEnable( GL_CULL_FACE );
        glBlendFunc( GL_ONE, GL_ONE );

        glBegin( GL_TRIANGLE_FAN );
          glColor3ub(255,255,255); 
          glVertex3f(0,0,1);
          glColor3ub(0,0,32);   
          glVertex3f( -0.5, -0.5, 0.5 );
          glVertex3f(  0.5, -0.5, 0.5 );
          glVertex3f(  0.5,  0.5, 0.5 );
          glVertex3f( -0.5,  0.5, 0.5 );
          glVertex3f( -0.5, -0.5, 0.5 );
        glEnd();

        glBegin( GL_TRIANGLE_FAN );
          glColor3ub(128,128,128); 
          glVertex3f(0,0,1);
          glColor3ub( 16,16,16 );   
          glVertex3f( -0.5, -0.5, 0.5 );
          glVertex3f( -0.5,  0.5, 0.5 );
          glVertex3f(  0.5,  0.5, 0.5 );
          glVertex3f(  0.5, -0.5, 0.5 );
          glVertex3f( -0.5, -0.5, 0.5 );
        glEnd();

        glDisable( GL_BLEND );
        glDisable( GL_CULL_FACE );

      glEndList();
    }

    GLuint &env_list = gb->env_list;
    if( env_list == 0 )
    {
      GLUquadricObj *pSphere = gluNewQuadric();
      gluQuadricDrawStyle(pSphere, GLU_FILL);
      gluQuadricTexture(pSphere, GLU_TRUE);

      env_list = glGenLists(1);
      glNewList(env_list, GL_COMPILE);

        glBlendFunc( GL_ONE, GL_ONE );
        glBindTexture(GL_TEXTURE_2D, worldmap);
        glEnable(GL_TEXTURE_2D);
        glDisable(GL_LIGHTING);
        glEnable( GL_BLEND );
        glEnable( GL_CULL_FACE );

        glPushMatrix();
          glRotatef( -90, 1,0,0 );
          glRotatef( -90, 0,0,1 );
          glColor3f( .25,.25,.25);
          glFrontFace(  GL_CW  );
          gluSphere(pSphere,1.0f,100,50);

          glColor3f( .5,.5,.5);
          glFrontFace(  GL_CCW  );
          gluSphere(pSphere,1.0f,100,50);
        glPopMatrix();

        glDisable(GL_TEXTURE_2D);
        glDisable( GL_BLEND );
        glDisable( GL_CULL_FACE );
      glEndList();

      gluDeleteQuadric(pSphere);
    }



  glClearColor( 0,0,0,1);
  // Drawing routine begin
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
      glLoadIdentity();
      float aspect = glutGet( GLUT_WINDOW_WIDTH ) / float(glutGet( GLUT_WINDOW_HEIGHT ) );
      gluPerspective( 25.0, aspect, 1.0, 20.1 );
  
    glMatrixMode(GL_MODELVIEW);
      glLoadIdentity();
        gluLookAt( 0,0,4, 0,0,0, 0,1,0 );

        glPushMatrix();
          glMultMatrixf( mo );
          glCallList( obj_list );
        glPopMatrix();

        if( gb->fellow_state & GBELL_LIGHT )
        {
          glPushMatrix();
            glMultMatrixf( ml );
            glCallList( light_list );
          glPopMatrix();
        }

        glPushMatrix();
          glScalef( bell_r, bell_r, bell_r );

          if( gb->fellow_state & GBELL_VIEW )
          {
            glPushMatrix();
              glMultMatrixf( mv );
              glCallList( view_list );
            glPopMatrix();
          }

          if( gb->fellow_state & GBELL_ENV )
          {
            glPushMatrix();
              glMultMatrixf( me );
              glCallList( env_list );
            glPopMatrix();
          }

          if( gb->drag_state != -1 )
          {
            glDisable(GL_LIGHTING);
            glDisable( GL_DEPTH_TEST );
            DrawAnyArc( gb->vFrom, gb->vTo, FLOAT3(.0,.8,.8), FLOAT3(.8,.8,.8) );
            glEnable( GL_DEPTH_TEST );
          }

        glPopMatrix();

        if( gb->active_ctrl )
        {
          glMatrixMode(GL_PROJECTION);
            glLoadIdentity();
          glMatrixMode(GL_MODELVIEW);
            glLoadIdentity();
          glDisable(GL_LIGHTING);
          glPushMatrix();
            glLoadIdentity();

            if( gb == ani_bell )
            {
              static float ani_a = 0;
              ani_a = ani_a>2? 0 : ani_a+.05;

              glPushMatrix();
                glTranslatef(-.85, .85, 0.0);
                glScalef(0.1, 0.1, 0.0);

                if( BELL_ANI_PAUSE || gb->drag_state != -1 )
                {
                  glColor3f( 1, 0, 0 );
                  glBegin( GL_QUADS );
                    glVertex2f(-.7,-.7);  
                    glVertex2f(-.2,-.7);
                    glVertex2f(-.2, .7);  
                    glVertex2f(-.7, .7);

                    glVertex2f( .2,-.7);  
                    glVertex2f( .7,-.7);
                    glVertex2f( .7, .7);  
                    glVertex2f( .2, .7);
                  glEnd();
                }else
                {
                  glColor3f( 0, fabs(ani_a-1), 0 );
                  glBegin( GL_TRIANGLES );
                    glVertex2f(-1,-1); glVertex2f(1,0); glVertex2f(-1,1);
                  glEnd();
                }
              glPopMatrix();
            }

            glColor3ub( 216, 255, 216 );
            glTranslatef(-.98, -.98, 0.0);
            glScalef(0.00085, 0.00085, 0.0);
            char str[256] = "";
              if( gb->active_ctrl & GBELL_LIGHT )
                strcat( str, "LIGHT " );

              if( gb->active_ctrl & GBELL_OBJ )
                strcat( str, "OBJECT " );

              if( gb->active_ctrl & GBELL_VIEW )
                strcat( str, "VIEW " );

              if( gb->active_ctrl & GBELL_ENV )
                strcat( str, "ENV " );
      
              if( gb->active_ctrl & GBELL_PICK )
              {
                strcat( str, "PICK " );

                char sname[64];
                if( gb->active_tram )
                {
                  sprintf( sname, "%i", gb->active_tram->tidx );
                  strcat( str, sname );
                }else
                {
                  strcat( str, "none" );
                }
              }
      
            char *ch;
            for( ch="ACTIVE: "; *ch; ch++ )
              glutStrokeCharacter(GLUT_STROKE_ROMAN, *ch);
            for( ch=str; *ch; ch++ )
              glutStrokeCharacter(GLUT_STROKE_ROMAN, *ch);
          glPopMatrix();
        }


  glutSwapBuffers();
  
}

void GBell::bell_reshape( int w, int h )
{
  glViewport(0, 0, w, h );
  bell_display();
}

GBell* GBell::get_bell( int hrd_win )
{
  for( int i=0; i<N_GBELL; i++ )
  {
    if( hrd_win==GBELL_LIST[i]->bell_win )
      return GBELL_LIST[i];
  }

  return NULL;
}

float* GBell::get_matrix( float *m, int bell_ctrl )
{
  switch( bell_ctrl )
  {
    case GBELL_LIGHT:
      qLight.matrix( m );
      break;

    case GBELL_OBJ:
      qObj.matrix( m );
      break;

    case GBELL_VIEW:
      qView.matrix( m );
      break;

    case GBELL_ENV:
      qEnv.matrix( m );
      break;
  };
  return m;
}

void GBell::reset()
{
  int hwin = glutGetWindow();
  
  qLight = qLight0 = GQuat();
  qObj   = qObj0   = GQuat();
  qView  = qView0  = GQuat();
  qEnv   = qEnv0   = GQuat();

  glutSetWindow(bell_win);
  glutReshapeWindow(bell_w,bell_h);
  glutPostRedisplay();
  glutSetWindow(fellow_win);
  glutPostRedisplay();
  
  glutSetWindow(hwin);
}

void GBell::set_dock( GBell *bell )
{
  int i;
  for( i=0; i<N_DOCK; i++ )
  {
    if( DOCK_LIST[i]->fellow_win == bell->fellow_win )
    {
      DOCK_LIST[i] = bell;
      return;
    }
  }
  DOCK_LIST[N_DOCK++] = bell;
}

void GBell::del_dock( GBell *bell )
{
  int i,j;
  for( i=0; i<N_DOCK; i++ )
  {
    if( DOCK_LIST[i] == bell )
    {
      N_DOCK--;
      for( j=i; j<N_DOCK; j++ )
        DOCK_LIST[j] = DOCK_LIST[j+1];
      return;
    }
  }
}

GBell *GBell::get_dock( int win_id )
{
  int i;
  for( i=0; i<N_DOCK; i++ )
    if( DOCK_LIST[i]->fellow_win == win_id )
      return DOCK_LIST[i];
  return NULL;
}

void GBell::del_bell( GBell *bell )
{
  int i,j;
  if( bell->bell_win )
  {
    for( i=0; i<N_GBELL; i++ )
      if( GBELL_LIST[i]->bell_win == bell->bell_win )
      {
        N_GBELL--;
        for( j=i; j<N_GBELL; j++ )
          GBELL_LIST[j] = GBELL_LIST[j+1];
      }
    glutDestroyWindow( bell->bell_win );

    bell->obj_list=0;
    bell->light_list=0;
    bell->view_list=0;
    bell->env_list=0;
    bell->bell_win=0;

  }
}

// Draw an arc defined by its ends.
#define LG_NSEGS 4
#define NSEGS (1<<LG_NSEGS)
void GBell::DrawAnyArc( const FLOAT3 &vFrom, const FLOAT3 &vTo )
{
  int i;
  FLOAT3 pts[NSEGS+1];
  double dot;

  pts[0] = vFrom;
  pts[1] = pts[NSEGS] = vTo;

  for (i=0; i<LG_NSEGS; i++) 
    pts[1] = vbisect(pts[0], pts[1]);

  dot = 2.0*vdot(&pts[0], &pts[1]);

  for (i=2; i<NSEGS; i++) 
  {
    pts[i] = pts[i-1] * dot  -  pts[i-2] ;
  }

  // OGLXXX for multiple, independent line segments: use GL_LINES
  glBegin(GL_LINE_STRIP);
  for (i=0; i<=NSEGS; i++)
    glVertex3fv((float *)&pts[i]);
  glEnd();
}

void GBell::DrawAnyArc( const FLOAT3 &vFrom, const FLOAT3 &vTo, const FLOAT3 &cForm, const FLOAT3 &cTo )
{
  int i;
  FLOAT3 pts[NSEGS+1];
  double dot;

  pts[0] = vFrom;
  pts[1] = pts[NSEGS] = vTo;

  for (i=0; i<LG_NSEGS; i++) 
    pts[1] = vbisect(pts[0], pts[1]);

  dot = 2.0*vdot( pts[0], pts[1]);

  for (i=2; i<NSEGS; i++) 
  {
    pts[i] = pts[i-1] * dot  -  pts[i-2] ;
  }

  float r;
  glBegin(GL_LINE_STRIP);
    for (i=0; i<=NSEGS; i++)
    {
      r = float(i)/NSEGS;
      glColor3fv( (float*)&(cForm*(1-r) + cTo*r) );
      glVertex3fv((float *)&pts[i]);
    }
  glEnd();
}

void GBell::save_screen( const char *_spath )
{
  char spath[256];
  {
    if( _spath )
    {
      strcpy( spath, _spath );
    }else
    {
      int i = 0;
      do{
        sprintf( spath, "screen%04i.png", i++ );
      }while( fexist(spath) );
    }
  }
  
  int hwin = glutGetWindow();
  glutSetWindow(fellow_win);

  int w,h;
    w = glutGet( GLUT_WINDOW_WIDTH );
    h = glutGet( GLUT_WINDOW_HEIGHT );

  GPng png;
    png.load( w, h );
    glReadBuffer( GL_BACK );
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels( 0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, png.bm );
    png.flip_vertical();
    png.save( spath );

  glutSetWindow(hwin);
}

void GBell::set_title( const char *title )
{
  int hwin = glutGetWindow();
  glutSetWindow(bell_win);
  glutSetWindowTitle( title );
  glutSetWindow(hwin);
}

int glPrintf( const char *format, ... )
{
  char buffer[256], *p;
  va_list args;
  va_start(args, format);
  vsprintf(buffer, format, args);
  va_end(args);

  GLint viewport[4];
    glGetIntegerv( GL_VIEWPORT, viewport );

  glPushAttrib( GL_LIGHTING_BIT | GL_DEPTH_BUFFER_BIT );
    glDisable( GL_LIGHTING );

  glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D( 0,viewport[2],0,viewport[3]); 

  glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

  FLOAT4 p0, p1, c0;

  // we need a call to glRasterPos in advance,
  // in order to enforce OpenGL updating
  // GL_CURRENT_COLOR to GL_CURRENT_RASTER_COLOR .
  // Strange... bugs of OpenGL?
  glGetFloatv( GL_CURRENT_RASTER_POSITION, (float*)&p0 );
  glRasterPos2f( p0.x,p0.y );

  glGetFloatv( GL_CURRENT_RASTER_COLOR, (float*)&c0 );


  // static int mode = 0;
  // static unsigned int base = 0;
  static GStack<int> gmode;
  static GStack<unsigned int> gbase;
  static GStack<HGLRC> gctx;
  static GStack<int> gx, gy;
  HGLRC ctx = wglGetCurrentContext();
  int i, j, offset;
  for( i=0; i<gctx.ns; i++ )
    if(gctx.buf[i] == ctx )
      break;
  if( i==gctx.ns )
  {
    gctx.push(ctx);
    gmode.push(0);
    gbase.push(0);
    gx.push(0);
    gy.push(0);
  }
  int &mode = gmode.buf[i];
  unsigned int &base = gbase.buf[i];
  int &x0=gx.buf[i];
  int &y0=gy.buf[i];

  if( memcmp( buffer, ">>glut", 6 )==0 )
  {
    if( base==0 )
    {
      base = glGenLists(256);
      for( i=0; i<256; i++ )
      {
        glNewList( base+i, GL_COMPILE);
          glutBitmapCharacter( GLUT_BITMAP_8_BY_13, i );
        glEndList();
      }
    }

    mode = 1;
    if( sscanf( buffer, ">>glut(%i,%i)", &x0, &y0 ) != 2 )
    {
      x0=0;
      y0=0;
    }
    glTranslatef( x0, -y0, 0 );
    glRasterPos2f( 4, viewport[3]-16 );
  }else if( memcmp( buffer, ">>win32", 7 )==0 )
  {
    if( base==0 )
    {
      HFONT hFont;
      HDC hdc;

      base = glGenLists(256);
      hFont = CreateFont( 18, 0, 0, 0, FW_BOLD, FALSE, FALSE, FALSE, ANSI_CHARSET, 
          OUT_TT_PRECIS, CLIP_DEFAULT_PRECIS, ANTIALIASED_QUALITY,
          FF_DONTCARE | DEFAULT_PITCH, "Courier New" );
      hdc = wglGetCurrentDC();
      SelectObject( hdc, hFont );
      wglUseFontBitmaps( hdc, 0, 256, base );
    }
    mode = 2;
    if( sscanf( buffer, ">>glut(%i,%i)", &x0, &y0 ) != 2 )
    {
      x0=0;
      y0=0;
    }
    glTranslatef( x0, -y0, 0 );
    glRasterPos2f( 4, viewport[3]-16 );
  }else
  {
    switch( mode )
    {
      case 0:
        printf( "%s", buffer );
        break;
      case 1:
      case 2:
        glListBase( base );
        glGetFloatv( GL_CURRENT_RASTER_POSITION, (float*)&p1 );
        glDisable( GL_DEPTH_TEST );
        for( j=0; j<3; j++ )
        {
          if( j==0 )
          {
            offset = 1;
            glColor4f( 1,1,1,1 );
            continue;  // turn off this case
          }
          if( j==1 )
          {
            offset = -1;
            glColor4f( 0,0,0,1 );
          }
          if( j==2 )
          {
            offset = 0;
            glColor4fv( (float*)&c0 );
          }

          glRasterPos2f( p1.x-offset, p1.y+offset );

          for( p=buffer; *p; p++ )
          {
            if( *p=='\n' )
            {
              glGetFloatv( GL_CURRENT_RASTER_POSITION, (float*)&p0 );
              glRasterPos2f( 4+x0-offset, p0.y-16 );
            }else
            {
              glCallLists( 1, GL_UNSIGNED_BYTE, p );
            }
          }
        }
        break;
    }
  }

  glMatrixMode(GL_PROJECTION);
    glPopMatrix();
  glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
  glPopAttrib();

  return 0;
}

void glOblique( GLdouble alpha, GLdouble phi, GLdouble aspect, GLdouble zNear, GLdouble zFar )
{
  double x, z, a, b, c, m[16];
    x  = 1/aspect;
    a  = cos(phi*G_PI/180)/tan(alpha*G_PI/180);
    b  = sin(phi*G_PI/180)/tan(alpha*G_PI/180);
    z  = -2/(zFar-zNear);
    c  = -(zFar+zNear)/(zFar-zNear);
    m[0]=x;  m[4]=0;  m[ 8]=-a;  m[12]=a;
    m[1]=0;  m[5]=1;  m[ 9]=-b;  m[13]=b;
    m[2]=0;  m[6]=0;  m[10]= z;  m[14]=c;
    m[3]=0;  m[7]=0;  m[11]= 0;  m[15]=1;
  glMultMatrixd(m);
}

























