/*

Sylvain Lefebvre, Samuel Hornus - 2013

This file implements all techniques/variants of A-buffer
described in our GPUPro paper.

The techniques/variants are selected through pre-processor
defines, so this fill is compiled several times to generate
all possible combinations.
Each variant is compiled into a DLL implementing the base
functions of a A-buffer library. The functions are 
described in "abuffer.h"

*/
// ------------------------------------------------

#include <LibSL/LibSL.h>
#include <LibSL/LibSL_gl4.h>

#define ABUFFER_NO_EXTERN
#include "../abuffer.h"

// ------------------------------------------------

#include "asppll_build.h"
#include "asppll_render.h"
#include "asppll_clear.h"

// --------------------------

#include "GL_EXT_gpu_shader4.h"
GLUX_REQUIRE( GL_EXT_gpu_shader4 );
#include "GL_NV_shader_buffer_store.h"

// ------------------------------------------------

using namespace std;

// ------------------------------------------------

// Shaders -- these classes are automatically generated during
// pre-compilation by a Python script (AutoBindShader).
AutoBindShader::asppll_build      g_ShBuild;
AutoBindShader::asppll_render     g_ShRender;
AutoBindShader::asppll_clear      g_ShClear;

// Screen size
uint                              g_ScreenSz;

// Data tables, GPU side
uint                              g_TableSz;
uint                              g_NumRecords;
GLBuffer                          g_Records;
GLBuffer                          g_Counts;
GLBuffer                          g_RGBA;
GLuint                            g_Atomics;

// View parameters
m4x4f                             g_Proj;
m4x4f                             g_View;
m4x4f                             g_Mdl;
v3f                               g_LPos;
float                             g_ZNEAR;
float                             g_ZFAR;
v3f                               g_BkgColor = 0;

// Additional GLSL parameters for custom rendering code
GLParameter                       g_Color;
GLParameter                       g_Opacity;
GLParameter                       g_LightPos;
GLParameter                       g_Tex;

// OpenGL timer objects
bool                              g_UseTimers;
GLTimer                           g_TmClear;
GLTimer                           g_TmBuild;
GLTimer                           g_TmRender;

typedef unsigned long long  t_uint64;
typedef unsigned int        t_uint32;

// ------------------------------------------------

extern "C" _declspec(dllexport) 
  void abuffer_init(int screen_sz,int num_records,float znear,float zfar,bool interruptable,bool use_timers)
{
  try {

    gluxInit();

    g_ScreenSz  = screen_sz;
    g_ZNEAR     = znear;
    g_ZFAR      = zfar;
    g_UseTimers = use_timers;

    // The following string describes the GLSL rendering code computing the final color of each (transparent) fragment.
    // -> it may be customized with 'abuffer_set_custom_fragment_code' /prior/ to calling init
    if (g_ShBuild.ComputeData.length() == 0) {
      string str = "\
                   uniform vec3  Color;\n\
                   uniform float Opacity;\n\
                   uniform vec3  LightPos; \n\
                   uniform sampler2D Tex; \n\
                   uint32_t computeData() { \n\
                   vec3  nrm     = v_Normal; \n\
                   nrm           = normalize( nrm ); \n\
                   vec3 view     = normalize( v_View ); \n\
                   /*vec3 light    = normalize( LightPos - v_View );*/ \n\
                   vec3 clr      = /*(max(0.2,dot(nrm,light))) * */ Color;\n\
                   return (uint32_t(clr.x*255.0) << 24u) + (uint32_t(clr.y*255.0) << 16u) + (uint32_t(clr.z*255.0) << 8u) + (uint32_t(Opacity*255.0)); \n\
                   } \n\
                   ";
      g_ShBuild.ComputeData = str;
    }

    cerr << Console::white;

    // Selects different shader paths depending on defines.
    // These booleans change the GLSL code sent to the GLSL compiler, so they incur no overhead.

#ifdef EARLYCULLING
    g_ShBuild .EarlyCulling  = true; // only implemented for HA-buffer
#ifdef BUBBLE_SORT
    cerr << "Cannot perform early-culling on post-sort techniques" << endl;
    throw Fatal("Cannot perform early-culling on post-sort techniques");
#endif
#endif
    g_ShBuild.Interruptible = interruptable; 
    g_ShClear.Interruptible = interruptable; 

#ifdef HABUFFER

    g_ShClear .HABuffer = true;
    g_ShBuild .HABuffer = true;
    g_ShRender.HABuffer = true;

#ifdef BUBBLE_SORT
    cerr << "<Hashed lists with bubble sort>" << endl;
    g_ShBuild .BubbleSort = true;
    g_ShRender.BubbleSort = true;
#else
    cerr << "<HA-buffer>" << endl;
    g_ShBuild .BubbleSort = false;
    g_ShRender.BubbleSort = false;
#endif

#else // not HABUFFER (=> linked lists)

    g_ShClear .HABuffer = false;
    g_ShBuild .HABuffer = false;
    g_ShRender.HABuffer = false;

#ifdef BUBBLE_SORT
    cerr << "<Per-pixel linked lists with bubble sort>" << endl;
    g_ShBuild  .BubbleSort = true;
    g_ShRender.BubbleSort = true;
#else
    g_ShBuild .BubbleSort = false;
    g_ShRender.BubbleSort = false;
#ifdef ASPPLL_CAS32 
    cerr << "<Always sorted per-pixel linked lists (cas32)>" << endl;
    g_ShBuild  .AsppllCas32 = true;
    g_ShRender.AsppllCas32 = true;
#else
    cerr << "<Always sorted per-pixel linked lists (max64)>" << endl;
    g_ShBuild .AsppllCas32 = false;
    g_ShRender.AsppllCas32 = false;
#endif
#endif

#ifdef ALLOC_NAIVE
    cerr << "[Alloc naive]" << endl;
    g_ShBuild .AllocNaive = true;
    g_ShClear .AllocNaive = true;
#else
    cerr << "[Alloc paged]" << endl;
    g_ShBuild .AllocNaive = false;
    g_ShClear .AllocNaive = false;
#endif

#endif

    // Due to an issue creating race-conditions when atomics are used in loops, 
    // we have to detect whether the code runs on a Kepler or on a Fermi.
    // In the future, this should become unecessary
    if ( strstr((char*)glGetString(GL_RENDERER),"680")
      || strstr((char*)glGetString(GL_RENDERER),"TITAN")) {
        cerr << Console::yellow;
        cerr << "<Running Kepler>" << endl;
        // Activate Kepler 'hacks' to circumvent the race-condition.
        g_ShBuild.Kepler = true;
    }
    cerr << Console::gray;

    g_ShBuild  .init();
    g_ShRender.init();
    g_ShClear .init();

    g_TmClear .init();
    g_TmBuild .init();
    g_TmRender.init();

    // Initialize custom parameters (see GLSL rendering code above)
    g_Color   .init( g_ShBuild.shader(), "Color");
    g_Opacity .init( g_ShBuild.shader(), "Opacity");
    g_LightPos.init( g_ShBuild.shader(), "LightPos");
    g_Tex     .init( g_ShBuild.shader(), "Tex");

    // Compute sizes for the tables, depending on requested number of records (fragments).
    g_TableSz    = max( g_ScreenSz, ceil(sqrt((float)num_records)) ); // the table is at least the screen size (required for hashed lists)
    g_NumRecords = g_TableSz * g_TableSz;
    cerr << Console::white << 
      sprint("<=== Screen: %d, Num Records: %d (%dx%d) ===>\n",g_ScreenSz,g_NumRecords,g_TableSz,g_TableSz)
      << Console::gray;

    // Allocate all tables
    g_Records.init( g_NumRecords*sizeof(t_uint32)*2 );
    g_RGBA   .init( g_NumRecords*sizeof(t_uint32)   );
    g_Counts .init( (screen_sz*screen_sz+1)*sizeof(t_uint32) );

    // initialize counts to non-zero
    // - required for some race-condition avoidance tricks
    // - last entry is used for overflow detection 
    Array<uint> ones(screen_sz*screen_sz+1);
    ones.fill(1);
    g_Counts.writeTo(ones);

    // global allocation counter
    glGenBuffersARB(1, &g_Atomics);
    glBindBufferARB(GL_ATOMIC_COUNTER_BUFFER, g_Atomics);
    glBufferDataARB(GL_ATOMIC_COUNTER_BUFFER, sizeof(GLuint), NULL, GL_DYNAMIC_DRAW);
    glBindBufferARB(GL_ATOMIC_COUNTER_BUFFER, 0);

    glMemoryBarrier( GL_ALL_BARRIER_BITS );

    // LIBSL_GL_CHECK_ERROR;

  } catch (Fatal& f) {
    cerr << f.message() << endl;
  }
}

// ------------------------------------------------

extern "C" _declspec(dllexport) 
  void abuffer_terminate()
{
  g_Records .terminate();
  g_RGBA    .terminate();
  g_Counts  .terminate();

  g_ShBuild  .terminate();
  g_ShRender.terminate();
  g_ShClear .terminate();

  g_TmClear .terminate();
  g_TmBuild .terminate();
  g_TmRender.terminate();
}

// ------------------------------------------------

extern "C" _declspec(dllexport) 
  void abuffer_change_size(float factor)
{
  // Adapt sizes
  g_TableSz    = (int)ceil( (double)g_TableSz * sqrt((double)factor) );
  g_NumRecords = g_TableSz * g_TableSz;
  cerr << Console::white << 
    sprint("<=== Screen: %d, Num Records: %d (%dx%d) ===>\n",g_ScreenSz,g_NumRecords,g_TableSz,g_TableSz)
    << Console::gray;

  // Resize all tables
  g_Records.resize( g_NumRecords*sizeof(t_uint32)*2 );
  g_RGBA   .resize( g_NumRecords*sizeof(t_uint32)   );

  glMemoryBarrier( GL_ALL_BARRIER_BITS );
}

// ------------------------------------------------

extern "C" _declspec(dllexport) 
  void abuffer_set_perspective(const float *m)
{
  memcpy( &g_Proj[0],m,sizeof(float)*16 );
}

extern "C" _declspec(dllexport) 
  void abuffer_set_view(const float *m)
{
  memcpy( &g_View[0],m,sizeof(float)*16 );
}

extern "C" _declspec(dllexport) 
  void abuffer_set_lightpos(const float *d)
{
  memcpy( &g_LPos[0],d,sizeof(float)*3 );
}

// ------------------------------------------------

// Clears all buffers for new frame rendering (varies depending on selected technique)
void asppll_clear()
{
  LIBSL_GL_CHECK_ERROR;

  if (g_UseTimers) g_TmClear.start();

  GPUHelpers::Renderer::setViewport(0,0,g_ScreenSz,g_ScreenSz);
  glColorMask(GL_FALSE,GL_FALSE,GL_FALSE,GL_FALSE); // no effect on screen
  g_ShClear.begin();
  // setup parameters
  g_ShClear.u_Projection      .set( orthoMatrixGL<float>(0,1,0,1,-1,1) );
  g_ShClear.u_Records         .set( g_Records );
  g_ShClear.u_RGBA            .set( g_RGBA );
  g_ShClear.u_Counts          .set( g_Counts );
  g_ShClear.u_ScreenSz        .set( g_ScreenSz );
  g_ShClear.u_HashSz          .set( g_TableSz );
  g_ShClear.u_NumRecords      .set( g_NumRecords );
  // execute
  glDisable(GL_DEPTH_TEST);
  glDisable(GL_CULL_FACE);
  glBegin(GL_QUADS);
  glVertex3i(0,0,0);
  glVertex3i(1,0,0);
  glVertex3i(1,1,0);
  glVertex3i(0,1,0);
  glEnd();
  g_ShClear.end();
  glColorMask(GL_TRUE,GL_TRUE,GL_TRUE,GL_TRUE);
  LIBSL_GL_CHECK_ERROR;

  glMemoryBarrier( GL_SHADER_GLOBAL_ACCESS_BARRIER_BIT_NV );

  // reinint global allocation counter
  glBindBufferARB(GL_ATOMIC_COUNTER_BUFFER, g_Atomics);
  GLuint z = 0;
  glBufferSubDataARB(GL_ATOMIC_COUNTER_BUFFER, 0 , sizeof(GLuint), &z);
  glBindBufferARB(GL_ATOMIC_COUNTER_BUFFER, 0);
  glMemoryBarrier( GL_ALL_BARRIER_BITS );
  glBindBufferBaseNV(GL_ATOMIC_COUNTER_BUFFER, 0, g_Atomics);

#ifdef HABUFFER
  // Generate random offsets at each frame. 
  static Array<v2u> offsets(256);
  ForIndex(i,256) {
    offsets[i]    = V2U( rand() ^ (rand()<<8) ^ (rand()<<16) , rand() ^ (rand()<<8) ^ (rand()<<16));
    offsets[i][0] = offsets[i][0] % g_TableSz;
    offsets[i][1] = offsets[i][1] % g_TableSz;
  }
  g_ShBuild  .begin();
  g_ShBuild  .u_Offsets.setArray(offsets.raw(),256);
  g_ShBuild  .end();
  g_ShRender.begin();
  g_ShRender.u_Offsets.setArray(offsets.raw(),256);
  g_ShRender.end();
#endif

  if (g_UseTimers) g_TmClear.stop();
}

// ------------------------------------------------

extern "C" _declspec(dllexport) 
  void abuffer_frame_begin(float r,float g,float b)
{
  try {
    LIBSL_GL_CHECK_ERROR;

    // -> clear buffers
    asppll_clear();

    if (g_UseTimers) g_TmBuild.start();
    GPUHelpers::Renderer::setViewport(0,0,g_ScreenSz,g_ScreenSz);
    // begin voxelizing
    glDisable(GL_CULL_FACE);
    glDisable(GL_DEPTH_TEST);
    // background color
    g_BkgColor = V3F(r,g,b);
    // -> staxelize: start
    g_ShBuild.begin();
    // setup view
    float eps = 1e-6f;
    g_ShBuild.u_Projection .set( g_Proj  );
    g_ShBuild.u_View       .set( g_View );
    g_ShBuild.u_Model      .set( m4x4f::identity() );
    g_ShBuild.u_ZNear      .set( g_ZNEAR );
    g_ShBuild.u_ZFar       .set( g_ZFAR );
    // setup asppll
    g_ShBuild.u_Records    .set( g_Records );
    g_ShBuild.u_Rec32      .set( g_Records );
    g_ShBuild.u_RGBA       .set( g_RGBA );
    g_ShBuild.u_Counts     .set( g_Counts );
    g_ShBuild.u_ScreenSz   .set( g_ScreenSz );
    g_ShBuild.u_HashSz     .set( g_TableSz );
    g_ShBuild.u_NumRecords .set( g_NumRecords );
    LIBSL_GL_CHECK_ERROR;
  } catch (Fatal& f) {
    cerr << f.message() << endl;
  }
}

// ------------------------------------------------

extern "C" _declspec(dllexport) 
  void abuffer_begin(float r,float g,float b,float a)
{
  g_LightPos  .set( g_View.mulPoint(g_LPos) );
  g_Opacity   .set( a );
  g_Color     .set( V3F(r,g,b) );
  g_Tex       .set( 0 );
}

// ------------------------------------------------

extern "C" _declspec(dllexport) 
  void abuffer_end()
{

}

// ------------------------------------------------

extern "C" _declspec(dllexport) 
  void abuffer_set_model_matrix( const float *modelMatrix )
{
  memcpy( &g_Mdl[0],modelMatrix,sizeof(float)*16 );
  g_ShBuild.u_Model.set( g_Mdl );
}

// ------------------------------------------------

extern "C" _declspec(dllexport) 
  void abuffer_frame_end(t_abuffer_frame_status *status)
{
  try {
    LIBSL_GL_CHECK_ERROR;

    bool overflow = false;
    memset(status,0x00,sizeof(t_abuffer_frame_status));

    // -> staxelize: stop
    g_ShBuild.end();

    glMemoryBarrier( GL_SHADER_GLOBAL_ACCESS_BARRIER_BIT_NV );

    if (g_ShBuild .Interruptible) {
      // check for interrupted rendering
      glMemoryBarrier( GL_BUFFER_UPDATE_BARRIER_BIT );
      uint numInserted = 0;
      g_Counts.readBackSub(&numInserted,sizeof(uint),g_ScreenSz*g_ScreenSz*sizeof(uint));
      if (numInserted >= ((g_NumRecords * 10)>>4)) {
        cerr << Console::red << "[frame was interrupted]" << Console::gray << endl;
        overflow = true;
      }
      // cerr << "load: " << numInserted/(float)g_NumRecords << endl;
    }

    if (g_UseTimers) g_TmBuild.stop();

    // render final
    if ( !overflow && 1 ) {
      if (g_UseTimers) g_TmRender.start();
      // setup screen
      GPUHelpers::Renderer::setViewport(0,0,g_ScreenSz,g_ScreenSz);
      glPushAttrib( GL_ENABLE_BIT );
      glDepthMask ( GL_FALSE );
      glDisable   ( GL_DEPTH_TEST );
      // render
      g_ShRender.begin();
      g_ShRender.BkgColor.set( g_BkgColor );
      g_ShRender.u_Projection .set( orthoMatrixGL<float>( 0,1, 0,1, -1,1 ) );
      g_ShRender.u_View       .set( m4x4f::identity() );
      g_ShRender.u_Model      .set( m4x4f::identity() );
      g_ShRender.ZNear        .set( g_ZNEAR );
      g_ShRender.ZFar         .set( g_ZFAR );
      // setup asppll
      g_ShRender.u_Records    .set( g_Records );
      g_ShRender.u_RGBA       .set( g_RGBA );
      g_ShRender.u_Counts     .set( g_Counts );
      g_ShRender.u_ScreenSz   .set( g_ScreenSz );
      g_ShRender.u_HashSz     .set( g_TableSz );
      g_ShRender.u_NumRecords .set( g_NumRecords );
      glBegin(GL_QUADS);
      glVertex2i(0,0);
      glVertex2i(1,0);
      glVertex2i(1,1);
      glVertex2i(0,1);
      glEnd();
      g_ShRender.end();
      glDepthMask( GL_TRUE );
      glPopAttrib();
      if (g_UseTimers) g_TmRender.stop();
    }

    LIBSL_GL_CHECK_ERROR;

    if (g_UseTimers) {
      while (1) { 
        GLint64 tm = g_TmClear.done();
        if (tm >= 0) { status->tm_clear = ((double)tm)/(1e6); break; }
      }
      while (1) { 
        GLint64 tm = g_TmBuild.done();
        if (tm >= 0) { status->tm_build = ((double)tm)/(1e6); break; }
      }
      if (!overflow) {
        while (1) { 
          GLint64 tm = g_TmRender.done();
        if (tm >= 0) { status->tm_render = ((double)tm)/(1e6); break; }
        }
      }
    }

    status->overflow = overflow;

  } catch (Fatal& f) {
    cerr << f.message() << endl;
  }

}

// ------------------------------------------------

static uint numBits(uint v)
{
  // NOTE: this is a naive way to do this
  uint n = 0;
  ForIndex(i,32) {
    if (v & (1<<i)) n++;
  }
  return n;
}

// ------------------------------------------------

extern "C" _declspec(dllexport) 
  void abuffer_print_stats(int *byteSize,float *loadFactor)
{
  cerr << "abuffer_print_stats" << endl;

  glMemoryBarrier( GL_ALL_BARRIER_BITS );

  try {
    LIBSL_GL_CHECK_ERROR;

#ifdef HABUFFER

    /// Hashed lists
    cerr << "[hashed list]" << endl;
    Array<long long> records;
    records.allocate( g_NumRecords );
    g_Records.readBack(records.raw(), g_NumRecords*sizeof(long long));
    uint totused = 0;
    ForIndex(i,g_NumRecords) {
      if (records[i]>0) { totused ++; }
    }
    cerr << "used       :" << totused << " elements, ";
    cerr << printByteSize(totused*sizeof(uint)) << endl;
    *loadFactor = totused/(float)g_NumRecords;
    cerr << "load factor: " << (*loadFactor) << endl;
    *byteSize   = 
        g_NumRecords*sizeof(uint)*2        /*records:age,depth,rgba on 64 bits*/
      + g_ScreenSz*g_ScreenSz*sizeof(uint) /*maxage table*/;

#else

    /// Linked lists 
    // Head records are removed from computations since a slightly
    // different implementation could be made to avoid the wasted 32 bits per list.
    // We kept the implementation uniform across all techniques for simplicity.

#ifdef ALLOC_NAIVE
    // naive
    cerr << "[naive]" << endl;
    uint totused = 0;
    glBindBufferARB(GL_ATOMIC_COUNTER_BUFFER, g_Atomics);
    glGetBufferSubDataARB(GL_ATOMIC_COUNTER_BUFFER, 0, sizeof(GLuint), &totused);
    glBindBufferARB(GL_ATOMIC_COUNTER_BUFFER, 0);
    cerr << "used       :" << totused << " elements, ";
    *loadFactor = totused/(float)(g_NumRecords-g_ScreenSz*g_ScreenSz);
    cerr << "load factor: " << (*loadFactor) << endl;
    // best possible size
    *byteSize = 
        (g_NumRecords-g_ScreenSz*g_ScreenSz)*sizeof(uint)*3 /*depth,rgba,ptr as three 32 bits ints*/
      + g_ScreenSz*g_ScreenSz*sizeof(uint) /*heads, as 32 bits pointers*/
      + sizeof(uint) /*counter, single 32 bit integer*/;
#else
    // paged
    cerr << "[paged]" << endl;
    uint numpages = 0;
    glBindBufferARB(GL_ATOMIC_COUNTER_BUFFER, g_Atomics);
    glGetBufferSubDataARB(GL_ATOMIC_COUNTER_BUFFER, 0, sizeof(GLuint), &numpages);
    glBindBufferARB(GL_ATOMIC_COUNTER_BUFFER, 0);
    Array<uint> counts(g_ScreenSz*g_ScreenSz+1);
    g_Counts.readBack( counts.raw(), (g_ScreenSz*g_ScreenSz+1)*sizeof(uint) );
    const uint PG_SIZE = 4;
    uint wasted = 0;
    ForIndex(c,counts.size()-1) {
      if (counts[c] > 0) {
        wasted += 3 - ((counts[c] - 1) % PG_SIZE) /*page size*/;
      }
    }
    uint totused = PG_SIZE*numpages/*atomic counter counts pages*/ - wasted;
    cerr << "used       :" << totused << " elements, ";
    *loadFactor = totused/(float)(g_NumRecords-g_ScreenSz*g_ScreenSz);
    cerr << "load factor: " << (*loadFactor) << endl;
    // number of allocated but unused elements:
    cerr << "wasted     :" << wasted << " elements" << endl;
    *byteSize = 
        (g_NumRecords-g_ScreenSz*g_ScreenSz)*sizeof(uint)*3  /*depth,rgba,ptr as three 32 bits ints*/
      + g_ScreenSz*g_ScreenSz*sizeof(uint) /*heads, as 32 bits pointers*/
#ifdef BUBBLE_SORT
      + (g_ScreenSz*g_ScreenSz+1)*sizeof(uint) /*counters, as 32 bits integers*/
#endif
      ;
    cerr << "actual load:" << numpages*PG_SIZE / (float)g_NumRecords << endl;
#endif

#endif

  } catch (Fatal& f) {
    cerr << f.message() << endl;
  }
}

// ------------------------------------------------

/*
This function is used to adjust the number of required records
desired by the user, depending on the selected technique.
In particular, linked lists require the addition of head pointers.
*/
extern "C" _declspec(dllexport) 
  int abuffer_compute_num_records(int screen_sz,int num_required_records)
{
#ifndef HABUFFER

  // linked lists: add heads records
#ifdef ALLOC_NAIVE
  // naive
  return num_required_records + screen_sz*screen_sz;
#else
#ifdef ALLOC_HASH
  // halloc
  return num_required_records + screen_sz*screen_sz;
#else
  // paged
  return max(num_required_records,screen_sz*screen_sz*4) + screen_sz*screen_sz;
#endif
#endif

#else

  // hashed lists
  return num_required_records;

#endif
}

// ------------------------------------------------

extern "C" _declspec(dllexport) 
  void abuffer_set_custom_fragment_code(const char *glsl_code)
{
  string str = "\
                 uniform vec3  Color;\n\
                 uniform float Opacity;\n\
                 uniform vec3  LightPos; \n\
                 uniform sampler2D Tex; \n\
                 uint32_t computeData() { \n\
                 vec3  nrm     = v_Normal; \n\
                 vec2  uv      = v_Tex; \n\
                 vec4  clr;\n\
                 " + std::string(glsl_code) +"\n\
                 return (uint32_t(clr.x*255.0) << 24u) + (uint32_t(clr.y*255.0) << 16u) + (uint32_t(clr.z*255.0) << 8u) + (uint32_t(Opacity*255.0)); \n\
                 } \n\
                 ";
  cerr << str << endl;
  g_ShBuild.ComputeData = str;
}

// ------------------------------------------------

BOOL WINAPI DllMain(
  HINSTANCE hinstDLL,  // handle to DLL module
  DWORD fdwReason,     // reason for calling function
  LPVOID lpReserved )  // reserved
{
  // Perform actions based on the reason for calling.
  switch( fdwReason ) 
  { 
  case DLL_PROCESS_ATTACH:
    // Initialize once for each new process.
    // Return FALSE to fail DLL load.
    break;

  case DLL_THREAD_ATTACH:
    // Do thread-specific initialization.
    break;

  case DLL_THREAD_DETACH:
    // Do thread-specific cleanup.
    break;

  case DLL_PROCESS_DETACH:
    // Perform any necessary cleanup.
    break;
  }
  return TRUE;  // Successful DLL_PROCESS_ATTACH.
}

// ------------------------------------------------
