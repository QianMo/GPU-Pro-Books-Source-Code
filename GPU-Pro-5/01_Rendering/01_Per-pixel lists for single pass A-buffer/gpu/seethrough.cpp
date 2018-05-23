// SL 2013-01-06

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include <Windows.h>

#include <LibSL/LibSL.h>
#include <LibSL/LibSL_gl4.h>
LIBSL_WIN32_FIX;

#include <sqlite3.h>
#include <tclap/CmdLine.h>

using namespace std;

// --------------------------

#include "abuffer.h"

// --------------------------

const uint  g_ScreenSz = 1024;
const float ZN =  0.01f;
const float ZF = 10.0f;

bool   g_ShowDepthComplexity = false;
bool   g_MeasurePerf         = false;
string g_DBTable;

// --------------------------
// Mesh loader/renderer

typedef struct
{
  LibSL::Math::v3f pos;
  LibSL::Math::v3f nrm;
  LibSL::Math::v2f uv;
} t_VertexData;

typedef MVF3(mvf_position_3f,mvf_normal_3f,mvf_texcoord0_2f) t_VertexFormat;

TriangleMesh_Ptr                               g_Mesh;
AutoPtr<TexturedMeshRenderer<t_VertexFormat> > g_TexturedRenderer;

// --------------------------
// Count depth complexity

#include "sh_count.h"
AutoBindShader::sh_count g_ShCount;
AutoBindShader::sh_count g_ShCountClear;
AutoBindShader::sh_count g_ShCountDisplay;
GLBuffer                 g_Counts;

// --------------------------

static int callback(void *NotUsed, int argc, char **argv, char **azColName)
{
  return 0;
}

// --------------------------

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
    g_ShowDepthComplexity = !g_ShowDepthComplexity;
  } else if (key == '*') {
    ImageRGBA img(g_ScreenSz,g_ScreenSz);
    glReadPixels(0,0,g_ScreenSz,g_ScreenSz,GL_RGBA,GL_UNSIGNED_BYTE,img.pixels().raw());
    img.flipH();
    static int cnt = 0;
    saveImage(sprint("shot%04d.png",cnt++),&img);
  }
}

// --------------------------

string      g_DLLName;
float       g_Opacity;

void render()
{
  static Timeout warmup(3000);
  static float tmavg = 0;
  static float tmavg_clear  = 0;
  static float tmavg_build  = 0;
  static float tmavg_render = 0;
  static float avgdepthavg  = 0;
  static long long avgnumfrags  = 0;
  static int   frame_cnt    = 0;

  glViewport (0,0,g_ScreenSz,g_ScreenSz);
  glEnable   ( GL_DEPTH_TEST );
  glDepthMask( GL_TRUE );
  glClearColor(1.0f,1.0f,1.0f,0.0f);

  // view
  m4x4f persp = perspectiveMatrixGL<float>( M_PI/4.0f,1.0f,ZN,ZF);
  m4x4f view  = TrackballUI::matrix();
  // setup abuffer uniforms
  float lpos[3] = {0,0,1};
  abuffer_set_perspective( &persp[0] ); // perspective matrix
  abuffer_set_view       ( &view[0] );  // view (camera) matrix 
  abuffer_set_lightpos   ( lpos );      // light direction

  bool overflow = false;
  do {

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // start frame, background color should match glClearColor
    abuffer_frame_begin(1.0f,1.0f,1.0f);  

    // draw scene
    m4x4f model = m4x4f::identity();
    abuffer_set_model_matrix( &model[0] );

    ForIndex(s,g_Mesh->numSurfaces()) {
      v3f clr = randomColorFromIndex(s);
      abuffer_begin(clr[0],clr[1],clr[2],g_Opacity);
      g_TexturedRenderer->renderSurface(s);
      abuffer_end();
    }

    // done with this frame (this call draws the result on screen, accumulating contributions from fragments)
    t_abuffer_frame_status fstat;
    abuffer_frame_end(&fstat);
    if (fstat.overflow) {
      abuffer_change_size( 2.0f );
    }
    cerr << sprint("[clear:%6.4f | build:%6.4f | render:%6.4f ms] ",fstat.tm_clear,fstat.tm_build,fstat.tm_render);

    // record performance
    if (warmup.expired()) {
      tmavg        += fstat.tm_clear+fstat.tm_build+fstat.tm_render;
      tmavg_clear  += fstat.tm_clear;
      tmavg_build  += fstat.tm_build;
      tmavg_render += fstat.tm_render;
      frame_cnt ++;
    }

  } while (overflow);

  // compute depth counters
  if (1) {
    glMemoryBarrier( GL_ALL_BARRIER_BITS );

    // compute depth complexity
    glColorMask(GL_FALSE,GL_FALSE,GL_FALSE,GL_FALSE);
    g_ShCountClear.begin();
    g_ShCountClear.u_Model.set( scaleMatrix(V3F(g_ScreenSz,g_ScreenSz,1)) );
    draw_square();
    g_ShCountClear.end();

    glMemoryBarrier( GL_ALL_BARRIER_BITS );

    g_ShCount.begin();
    g_ShCount.u_Projection.set(persp);
    g_ShCount.u_Model.set( view );
    ForIndex(s,g_Mesh->numSurfaces()) {
      g_TexturedRenderer->renderSurface(s);
    }
    g_ShCount.end();
    glColorMask(GL_TRUE,GL_TRUE,GL_TRUE,GL_TRUE);

    glMemoryBarrier( GL_ALL_BARRIER_BITS );

    //uint maxdepth = 0;
    //g_Counts.readBackSub( &maxdepth, sizeof(uint), g_Counts.size() - sizeof(uint) );
    //cerr << "max depth: " << maxdepth << endl;

    Array<int> counts;
    counts.allocate( g_Counts.size()/sizeof(uint) );
    g_Counts.readBack( counts );
    int   avgdepth = 0;
    int   num = 0;
    ForIndex(c,counts.size()-1) {
      if (counts[c]>0) {
        num ++;
        avgdepth += counts[c];
      }
    }
    cerr << " <avg:" << avgdepth/(float)num << " max: " << counts[counts.size()-1] << '>'<< endl;
    if (num > 0 && warmup.expired()) {
      avgdepthavg += avgdepth/(float)num;
      avgnumfrags += avgdepth;
    }

    if (g_ShowDepthComplexity) {
      // display on screen
      glDisable(GL_DEPTH_TEST);
      g_ShCountDisplay.begin();
      g_ShCountDisplay.u_Model.set( scaleMatrix(V3F(g_ScreenSz,g_ScreenSz,1)) );
      draw_square();
      g_ShCountDisplay.end();
      glEnable(GL_DEPTH_TEST);
    }
  }

  if (g_MeasurePerf) {
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
      sqlite3_exec(db,sprint("CREATE TABLE %s (Id INTEGER PRIMARY KEY,name TEXT,Time REAL,"
                             "ByteSize INT,Opacity REAL,AvgDepth REAL,NumFrags INT,LoadFactor REAL,"
                             "TimeClear REAL,TimeBuild REAL,TimeRender REAL)",g_DBTable.c_str()),callback,NULL,&err);
      // insert result
      sqlite3_exec(db,sprint("INSERT INTO %s VALUES (NULL,'%s',%f,%d,%f,%f,%d,%f,%f,%f,%f)",
        g_DBTable.c_str(),
        removeExtensionFromFileName(extractFileName( g_DLLName )).c_str(),
        tmavg / (float)frame_cnt,
        bytesz,
        g_Opacity,
        (float)avgdepthavg / (float)frame_cnt,
        (int)(avgnumfrags / frame_cnt),
        loadFactor,
        tmavg_clear / (float)frame_cnt,
        tmavg_build / (float)frame_cnt,
        tmavg_render / (float)frame_cnt
        ),
        callback,NULL,&err);
      // close
      sqlite3_close( db );

      TrackballUI::exit();
    }
  }
}

// --------------------------

int main(int argc,char **argv)
{
  try {
  TCLAP::CmdLine cmd("", ' ', "1.0");

  TCLAP::UnlabeledValueArg<string> dllArg("dll","DLL to use",true,"prelin.dll","string");
  TCLAP::ValueArg<float>  opacityArg("o","opacity", "Opacity",false,0.5,"float"   );
  TCLAP::ValueArg<string> meshArg   ("m","mesh", "Mesh name",false,"foo.obj","string");
  TCLAP::ValueArg<string> viewArg   ("t","trackball", "Trackball filename",false,"trackball.F01","string"   );
  TCLAP::ValueArg<string> dbArg     ("d","db", "DB table",false,"seethrough","string");
  TCLAP::SwitchArg        measureArg("p","perf", "Measure performance",false);

  cmd.add( opacityArg );
  cmd.add( meshArg );
  cmd.add( viewArg );
  cmd.add( measureArg );
  cmd.add( dbArg );
  cmd.add( dllArg );
  cmd.parse( argc, argv );
  
  g_MeasurePerf = measureArg.isSet();
  g_DBTable     = dbArg.getValue();
  cerr << g_DBTable << endl;

  srand(42);

  TrackballUI::onRender     = render;
  TrackballUI::onKeyPressed = mainKeypressed;

  int scrsz = g_ScreenSz;
  TrackballUI::init(scrsz,scrsz);

  g_ShCountClear  .Clear   = true;
  g_ShCountDisplay.Display = true;
  g_ShCount       .init();
  g_ShCountClear  .init();
  g_ShCountDisplay.init();

  g_Counts.init( (scrsz*scrsz+1)*sizeof(uint) );

  g_ShCount       .begin();
  g_ShCount       .u_ScreenSz  .set(uint(g_ScreenSz));
  g_ShCount       .u_Counts    .set(g_Counts);
  g_ShCount       .end();
  g_ShCountClear  .begin();
  g_ShCountClear  .u_ScreenSz  .set(uint(g_ScreenSz));
  g_ShCountClear  .u_Counts    .set(g_Counts);
  g_ShCountClear  .u_Projection.set( orthoMatrixGL<float>(0,g_ScreenSz,0,g_ScreenSz, 0,g_ScreenSz ) );
  g_ShCountClear  .end();
  g_ShCountDisplay.begin();
  g_ShCountDisplay.u_ScreenSz  .set(uint(g_ScreenSz));
  g_ShCountDisplay.u_Counts    .set(g_Counts);
  g_ShCountDisplay.u_Projection.set( orthoMatrixGL<float>(0,g_ScreenSz,0,g_ScreenSz, 0,g_ScreenSz ) );
  g_ShCountDisplay.end();

  cerr << "Loading mesh      ";
  g_Mesh     = loadTriangleMesh<t_VertexData,t_VertexFormat>( meshArg.getValue().c_str() );
  cerr << "[OK]" << endl;
  cerr << "  mesh bbox : " << g_Mesh->bbox().minCorner() << " - " << g_Mesh->bbox().maxCorner() << endl;
	g_Mesh->scaleToUnitCube();
  g_Mesh->centerOn(0);
  TrackballUI::trackball().translation() = g_Mesh->bbox().center()[0];
  cerr << sprint("  mesh contains %d vertices, %d triangles, %d surfaces\n",g_Mesh->numVertices(),g_Mesh->numTriangles(),g_Mesh->numSurfaces());

  cerr << "Creating renderer ";
  if (g_Mesh->numSurfaces() == 0) {
    cerr << " <<not a textured model>>\n";
    exit (-1);
  } else {
     g_TexturedRenderer = new TexturedMeshRenderer<t_VertexFormat>( g_Mesh,
       AutoPtr<TextureProvider>(new DefaultTextureProvider("", NULL, ".png", vector<string>() )) );
  }
  cerr << "[OK]" << endl;

  g_DLLName  = dllArg.getValue();
  g_Opacity  = opacityArg.getValue();

  int num_rec = scrsz*scrsz*8;

  abuffer_load_dll( g_DLLName.c_str() );
  if (!g_MeasurePerf) abuffer_set_custom_fragment_code( "clr = texture(Tex,uv);" );
  abuffer_init(
    scrsz,   /*resolution - screen has to be square */
    num_rec, /* number of records in tables */
    ZN, /* znear */
    ZF, /* zfar */
    true,
    true);

  TrackballUI::trackballLoad(viewArg.getValue().c_str());

  TrackballUI::loop();

  abuffer_terminate();
  
  g_TexturedRenderer = NULL;
  g_Counts        .terminate();
  g_ShCount       .terminate();
  g_ShCountClear  .terminate();
  g_ShCountDisplay.terminate();

  TrackballUI::shutdown();

  } catch (Fatal& f) {
    cerr << f.message() << endl;
  }
}

// --------------------------
