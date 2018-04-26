#ifndef G_BELL_H
#define G_BELL_H

#include "g_vector.h"
#include "g_pfm.h"

#define GBELL_UP     1<<8
#define GBELL_DOWN   1<<9
#define GBELL_DRAG   1<<10
#define GBELL_LIGHT  1<<11
#define GBELL_OBJ    1<<12
#define GBELL_VIEW   1<<13
#define GBELL_ENV    1<<14
#define GBELL_DOCK   1<<15
#define GBELL_PICK   1<<16

typedef struct _GTram
{
  unsigned int tidx;
  int vidx[3];
  float vw[3];
}GTram;


class GBell
{
  public:

    GQuat qLight0, qLight;
    GQuat qObj0,   qObj;
    GQuat qView0,  qView;
    GQuat qEnv0,   qEnv;
    GQuat qDrag;

    GBell();
    ~GBell();
    void create_window( const char *title=0, int bell_w=200, int bell_h=200 );
    void create_subwindow();

    void set_title( const char *title );
    void set_fellow( int win_id = 0, int state = GBELL_DOWN | GBELL_UP | GBELL_DRAG | GBELL_DOCK | GBELL_LIGHT );
    void set_active_ctrl( int bell_ctrl );
    void reset();
    void save( const char *spath );
    void load( const char *spath );
    void save_screen( const char *spath=NULL );

    void load_smap( const char *spath );
    void load_smap( const GPfm &smap );

    // delta_time>0 for forward frame
    // delta_time<0 for backward frame
    void bell_frame( float delta_time );

    void PostRedisplay();

    // Filling parameter m with 
    // 4x4 rotation matrix (opengl version), ie assumed the following :
    //  - row vector
    //  - right handed orientation
    //  - y-axis point upward, x-axis point rightward (ie, z-axis point out of the screen )
    // bell_ctrl = { GBELL_OBJ, GBELL_VIEW, GBELL_ENV, GBELL_LIGHT }

    float* get_matrix( float *m, int bell_ctrl );

    static void bell_mousebutton(int button, int state, int x, int y);
    static void bell_mousemove(int x, int y);

    // state value of actively controled object
    int active_ctrl;
    int drag_state;

    GTram *active_tram;
    GTram *prev_tram;

    void (*PickFunc)(int tidx);
    void (*PickChangeFunc)( const GTram *tram0, const GTram *tram1 );
    void set_pick( void (*pick_function)(int tidx) );
    void set_pickchange( void (*pickchange_function)( const GTram *tram0, const GTram *tram1 ) );
    void select_none();
    void select_tram( int tidx, int v0, int v1, int v2, float vw0, float vw1, float vw2 );

  private:

    void process_pick( int x, int y );
    void make_window( const char *title=0, int bell_w=200, int bell_h=200 );
    void setlight();

    // handle of the arcbell window
    int bell_win;
    int bell_w, bell_h;

    unsigned int obj_list;
    unsigned int light_list;
    unsigned int view_list;
    unsigned int env_list;
    
    // state value of events registered
    int fellow_state;

    // handle of the fellow window
    int fellow_win;

    FLOAT3 vFrom, vTo;
    
    static GBell* ani_bell;
    static GLuint worldmap;

    static int N_GBELL;
    static GBell* GBELL_LIST[16];
    static void del_bell( GBell *bell );
    static GBell* get_bell( int hrd_win );

    static int N_DOCK;
    static GBell* DOCK_LIST[16];
    static void set_dock( GBell *bell );
    static void del_dock( GBell *bell );
    static GBell *get_dock( int win_id );

    static void bell_display();
    static void bell_reshape( int w, int h );
    static void bell_keyboard( unsigned char key, int x, int y );
    static void bell_special( int key, int x, int y );
    static void bell_menu( int item );
    static void bell_timer( int value );
    static void DrawAnyArc( const FLOAT3 &vFrom, const FLOAT3 &vTo );
    static void DrawAnyArc( const FLOAT3 &vFrom, const FLOAT3 &vTo, const FLOAT3 &cForm, const FLOAT3 &cTo );
};

int glPrintf( const char *format, ... );
void glOblique( GLdouble alpha, GLdouble phi, GLdouble aspect, GLdouble zNear, GLdouble zFar );

#endif
