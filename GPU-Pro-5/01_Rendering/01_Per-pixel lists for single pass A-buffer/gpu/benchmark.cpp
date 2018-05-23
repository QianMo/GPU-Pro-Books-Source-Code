// SL 2013-01-06

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include <Windows.h>

#include <LibSL/LibSL.h>
#include <LibSL/LibSL_gl4.h>

#include <sqlite3.h>
#include <tclap/CmdLine.h>

using namespace std;

// --------------------------

#include "abuffer.h"

#include "sh_count.h"
AutoBindShader::sh_count g_ShCount;
AutoBindShader::sh_count g_ShCountClear;
GLBuffer                 g_Counts;

string                   g_DBName;

// --------------------------

const float ZN = 1.0f;
const float ZF = 40.0f;

// --------------------------

AutoPtr<Shapes::Box> g_Box;

void draw_box()
{
  g_Box->render();
}

void draw_square()
{
  glBegin(GL_QUADS);
  glVertex2i(0,0);
  glVertex2i(1,0);
  glVertex2i(1,1);
  glVertex2i(0,1);
  glEnd();
}

// --------------------------

void mainKeypressed(uchar key)
{
  if (key == ' ') {
  // display stats
    int bytesz = 0;
    float loadFactor = 0;
    abuffer_print_stats(&bytesz,&loadFactor);
    cerr << Console::white << "Memory: " << printByteSize(bytesz) << Console::gray << endl;
    SimpleUI::exit();
  }
}

// --------------------------

void transpose(float *m)
{
  for (int j=0;j<4;j++) {
    for (int i=0;i<4;i++) {
      if (i<j) { continue; }
      int k  = i+j*4;
      int t  = j+i*4;
      std::swap( m[k],m[t] );
    }
  }
}

// --------------------------

static int callback(void *NotUsed, int argc, char **argv, char **azColName)
{
  return 0;
}

// --------------------------

string      g_DLLName;
int         g_NumBoxes = 32;
int         g_Gutter   = 0;
float       g_BoxScale = 1.0f;
float       g_Opacity  = 0.7f;

void render()
{
  // use GL to produce matrices

  float persp[16];
  float view [16];

  /// Setup GL matrices (read them back to send to library)
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  // gluPerspective( 60.0f, 1.0f, ZN/*near*/, ZF /*far*/); // (carefull to properly match znear-zfar with abuffer_init
  glOrtho(0,1024,0,1024,0,1024);
  glGetFloatv(GL_PROJECTION_MATRIX, persp);
  transpose( persp );

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  //gluLookAt( 10,10,10, 0,0,0, 0,0,1 );
  glGetFloatv(GL_MODELVIEW_MATRIX, view);
  transpose( view );

  glClearColor(0.5f,0.5f,0.5f,0.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glEnable   ( GL_DEPTH_TEST );
  glDepthMask( GL_TRUE );

  float lpos[3] = {0,0,1};

  // setup abuffer

  abuffer_set_perspective( persp ); // perspective matrix
  abuffer_set_view       ( view );  // view (camera) matrix 
  abuffer_set_lightpos   ( lpos );  // light direction

  abuffer_frame_begin(0.5f,0.5f,0.5f); // start a frame, background color should match glClearColor

  // draw objects

  static float agl = 0;
  
  agl += 1.0f;
  
  // generate random squares
  vector< v3f > squares;
  // srand(42); //// NOTE: uncomment to get the same set of squares each frame
  ForIndex(n,g_NumBoxes) {
      // -> matrix
      glMatrixMode(GL_MODELVIEW);
      glLoadIdentity();
      float i = g_Gutter + rnd()*(1024.0f-g_BoxScale-2*g_Gutter);
      float j = g_Gutter + rnd()*(1024.0f-g_BoxScale-2*g_Gutter);
      float k = rnd()*1024.0f;
      squares.push_back(V3F(i,j,-k));
  }

  // draw primitives (nothing is drawns on screen, the A-buffer is built instead)
  ForIndex(n,g_NumBoxes) {
    // -> setup
    m4x4f model = translationMatrix(squares[n]) * scaleMatrix(V3F(g_BoxScale,g_BoxScale,1));
    abuffer_set_model_matrix( &model[0] );
    v3f rnd = randomColorFromIndex( n );
    abuffer_begin(rnd[0]/*r*/,rnd[1]/*g*/,rnd[2]/*b*/, g_Opacity /*opacity*/);
    // -> draw
    draw_square();
    abuffer_end();
  }

  // done with this frame (this call draws the result on screen, accumulating contributions from fragments)
  t_abuffer_frame_status fstat;
  abuffer_frame_end(&fstat);
  if (1) cerr << sprint("[clear:%6.4f | build:%6.4f | render:%6.4f ms] ",fstat.tm_clear,fstat.tm_build,fstat.tm_render);

  // record performance
  static Timeout warmup(3000);

  static float tmavg = 0;
  static float tmavg_clear  = 0;
  static float tmavg_build  = 0;
  static float tmavg_render = 0;
  static int   frame_cnt = 0;
  if (warmup.expired()) {
     tmavg        += fstat.tm_clear+fstat.tm_build+fstat.tm_render;
     tmavg_clear  += fstat.tm_clear;
     tmavg_build  += fstat.tm_build;
     tmavg_render += fstat.tm_render;
     frame_cnt ++;
   }

  static float avgdepthavg = 0;
  if (1) {
    glMemoryBarrier( GL_ALL_BARRIER_BITS );

    // compute depth complexity
    glColorMask(GL_FALSE,GL_FALSE,GL_FALSE,GL_FALSE);
    g_ShCountClear.begin();
    g_ShCountClear.u_Model.set( scaleMatrix(V3F(1024,1024,1)) );
    draw_square();
    g_ShCountClear.end();

    glMemoryBarrier( GL_ALL_BARRIER_BITS );

    g_ShCount.begin();
    ForIndex(n,g_NumBoxes) {
      g_ShCount.u_Model.set( translationMatrix(squares[n]) * scaleMatrix(V3F(g_BoxScale,g_BoxScale,1)) );
      draw_square();
    }
    g_ShCount.end();
    glColorMask(GL_TRUE,GL_TRUE,GL_TRUE,GL_TRUE);

    glMemoryBarrier( GL_ALL_BARRIER_BITS );

    Array<int> counts;
    counts.allocate( g_Counts.size()/sizeof(uint) );
    g_Counts.readBack( counts );
    int avgdepth = 0;
    int num = 0;
    ForIndex(c,counts.size()-1) {
      if (counts[c]>0) {
        num ++;
        avgdepth += counts[c];
      }
    }
    cerr << " <avg:" << avgdepth/(float)num << " max: " << counts[counts.size()-1] << '>'<< endl;
    if (num > 0 && warmup.expired()) {
      avgdepthavg += avgdepth/(float)num;
    }
  }

  if (1) {
    static Timeout measure(6000);
    if (measure.expired()) {
      int bytesz = 0;
      float loadFactor = 0;
      abuffer_print_stats( &bytesz,&loadFactor );
      cerr << Console::white << "Memory: " << printByteSize(bytesz) << Console::gray << endl;
      cerr << Console::white << "Time  : " << tmavg / (float)frame_cnt << " ms." << Console::gray << endl;
      sqlite3 *db  = NULL;
      char    *err = NULL;
      sqlite3_open("bench.db", &db);
      // create table  
      sqlite3_exec(db,sprint("CREATE TABLE %s (Id INTEGER PRIMARY KEY,name TEXT,NumBoxes INT,Time REAL,"
                             "ByteSize INT,Gutter INT,Opacity REAL,AvgDepth REAL,NumFrags INT,LoadFactor REAL,"
                             "TimeClear REAL,TimeBuild REAL,TimeRender REAL)",g_DBName.c_str()),callback,NULL,&err);
      // insert result
      sqlite3_exec(db,sprint("INSERT INTO %s VALUES (NULL,'%s',%d,%f,%d,%d,%f,%f,%d,%f,%f,%f,%f)",
        g_DBName.c_str(),
        removeExtensionFromFileName(extractFileName( g_DLLName )).c_str(),
        g_NumBoxes,
        tmavg / (float)frame_cnt,
        bytesz,
        g_Gutter,
        g_Opacity,
        (float)avgdepthavg / (float)frame_cnt,
        int(g_NumBoxes*g_BoxScale*g_BoxScale),
        loadFactor,
        tmavg_clear / (float)frame_cnt,
        tmavg_build / (float)frame_cnt,
        tmavg_render / (float)frame_cnt
        ),
        callback,NULL,&err);
      // close
      sqlite3_close( db );
      SimpleUI::exit();
    }
  }

}

// --------------------------

int main(int argc,char **argv)
{
  TCLAP::CmdLine cmd("", ' ', "1.0");

  TCLAP::UnlabeledValueArg<string> dllArg("dll","DLL to use",true,"prelin.dll","string");
  TCLAP::ValueArg<int>    nSqArg    ("n","numsquares", "Number of squares",false,100,"int"   );
  TCLAP::ValueArg<int>    sqSzArg   ("r","squareres", "Resolution of squares (pixels)",false,128,"int"   );
  TCLAP::ValueArg<int>    gutterArg ("g","gutter", "Gutter (pixels)",false,0,"int"   );
  TCLAP::ValueArg<float>  opacityArg("o","opacity", "Opacity",false,0.5,"float"   );
  TCLAP::ValueArg<float>  loadArg   ("l","loadfactor", "Load factor",false,0.5,"float"   );
  TCLAP::ValueArg<string> dbArg     ("d","dbarg", "DB name",false,"results","string"   );

  cmd.add( dllArg );
  cmd.add( nSqArg );
  cmd.add( sqSzArg );
  cmd.add( gutterArg );
  cmd.add( opacityArg );
  cmd.add( loadArg );
  cmd.add( dbArg );

  cmd.parse( argc, argv );
  
  srand(42);

  SimpleUI::onRender = render;
  SimpleUI::onKeyPressed = mainKeypressed;
  
  int scrsz   = 1024;
  SimpleUI::init(scrsz,scrsz);

  g_Box = new Shapes::Box();

  abuffer_load_dll(dllArg.getValue().c_str());
  g_DLLName  = dllArg.getValue();
  g_NumBoxes = nSqArg.getValue();
  g_BoxScale = (float)sqSzArg.getValue();
  g_Gutter   = gutterArg.getValue();
  g_Opacity  = opacityArg.getValue();
  g_DBName   = dbArg.getValue();

  // compute required records
  int num_rec = (int)ceil( (float)g_NumBoxes * (float)(g_BoxScale*g_BoxScale) / loadArg.getValue() );
  cerr << "Number of required records           : " << num_rec << endl;
  // adjust depending on a-buffer scheme
  num_rec = abuffer_compute_num_records( scrsz, num_rec);
  cerr << "Number of required records (adjusted): " << num_rec << endl;

  abuffer_init(
    scrsz,   /*resolution - screen has to be square */
    num_rec, /* number of records in tables */
    ZN, /* znear */
    ZF, /* zfar */
    false,
    true);

  g_ShCount     .Clear = false;
  g_ShCountClear.Clear = true;
  g_ShCount     .init();
  g_ShCountClear.init();

  g_Counts.init( (scrsz*scrsz+1)*sizeof(uint) );

  g_ShCountClear.begin();
  g_ShCountClear.u_ScreenSz  .set(uint(1024));
  g_ShCountClear.u_Counts    .set(g_Counts);
  g_ShCountClear.u_Projection.set( orthoMatrixGL<float>(0,1024,0,1024, 0,1024 ) );
  g_ShCountClear.end();

  g_ShCount.begin();
  g_ShCount.u_ScreenSz  .set(uint(1024));
  g_ShCount.u_Counts    .set(g_Counts);
  g_ShCount.u_Projection.set( orthoMatrixGL<float>(0,1024,0,1024, 0,1024 ) );
  g_ShCount.end();

  SimpleUI::loop();

  abuffer_terminate();
  g_Counts        .terminate();
  g_ShCount       .terminate();
  g_ShCountClear  .terminate();

  SimpleUI::shutdown();
}

// --------------------------
