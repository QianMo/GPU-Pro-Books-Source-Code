//#define REDIRECT
#include "framework.h"

#include "octree.h"
#include "intersection.h"

#include "debug_draw.h"

#include "browser.h"

#include <sstream>
#include <string>
#include <locale>
#include <codecvt>

#define WIN32_LEAN_AND_MEAN
#ifdef _WIN32
#include <Windows.h>
#endif

#include "fixed_pos.h"

///////////////////////////////////////////////////////////////////////////////////////
// data structures etc.
///////////////////////////////////////////////////////////////////////////////////////

enum light_type
{
  lightPOINT = 0, lightSPOT, lightDIRECTIONAL, lightLAST
};

enum attenuation_type
{
  FULL = 0, LINEAR, attLAST
};

struct light_data
{
  vec4 diffuse_color;
  vec4 specular_color; //w is light_size
  vec4 vs_position;
  vec4 ms_position;
  float attenuation_end;
  float attenuation_cutoff; // ]0...1], 1 (need low values for nice attenuation)
  float radius;
  float spot_exponent;
  int attenuation_type; // 0 full (constant, linear, quadratic), 1 linear
  int lighting_type; // 0 point, 1 spot, 2 directional
  int lighting_model; //unused
  int layer;
  vec4 spot_direction; //w is spot_cutoff ([0...90], 180)
  mat4 spot_shadow_mat;
  mat4 point_shadow_mats[6];
}; //560 bytes, 35 vec4s

//uniform data containers
vector< vec4 > diffuse_color_data;
vector< vec4 > specular_color_data;
vector< float > light_size_data;
vector< vec4 > vs_position_data;
vector< vec4 > ms_position_data;
vector< float > attenuation_end_data;
vector< float > attenuation_cutoff_data;
vector< float > radius_data;
vector< float > spot_exponent_data;
vector< int > attenuation_type_data;
vector< int > lighting_type_data;
vector< int > layer_data;
vector< vec4 > spot_direction_data;
vector< float > spot_cutoff_data;
vector< mat4 > spot_shadow_mat_data;
vector< mat4 > point_shadow_mat_data;

struct light
{
  camera<float> light_cam;
  vec3 diffuse_color, specular_color;
  float attenuation_coeff, radius, spot_exponent, spot_cutoff;
  attenuation_type att_type;
  light_type type;
  shape* bv;
  int layer;
  float light_size;
  camera<float> point_shadow_cams[6];
  frustum point_shadow_frustums[6];
  mat4 spot_shadow_mat;
  mat4 point_shadow_mats[6];
};

typedef vector< int > layer; //each layer contains lights that don't intersect
typedef vector< int > intersections;

///////////////////////////////////////////////////////////////////////////////////////
// globals
///////////////////////////////////////////////////////////////////////////////////////

map<string, string> args;

uvec2 screen( 0 );
bool fullscreen = false;
bool silent = false;
string title = "Practical Screen Space Soft Shadows";

framework frm;

pipeline<float> ppl;
matrix_stack<float> mvm;
camera<float> cam;
frame<float> the_frame;

float cam_fov = 58.7f;
float cam_near = 2.5f;
float cam_far = 1000.0f;

octree<unsigned>* o = new octree<unsigned>(aabb(vec3(0), vec3(1024)));

GLuint quad;
GLuint ss_quad;

int nummeshes = 469;

vector<mesh> scene;

//shadow scale/bias matrix
mat4 bias_matrix( 0.5f, 0, 0, 0,
                  0, 0.5f, 0, 0,
                  0, 0, 0.5f, 0,
                  0.5f, 0.5f, 0.5f, 1 );

//pos x
mat4 posx = mat4(  0,  0, -1, 0,
                    0, -1,  0, 0,
                  -1,  0,  0, 0,
                    0,  0,  0, 1 );
//neg x
mat4 negx = mat4( 0,  0, 1, 0,
                  0, -1, 0, 0,
                  1,  0, 0, 0,
                  0,  0, 0, 1 );
//pos y
mat4 posy = mat4( 1, 0,  0, 0,
                  0, 0, -1, 0,
                  0, 1,  0, 0,
                  0, 0,  0, 1 );
//neg y
mat4 negy = mat4( 1,  0, 0, 0,
                  0,  0, 1, 0,
                  0, -1, 0, 0,
                  0,  0, 0, 1 );
//pos z
mat4 posz = mat4( 1,  0,  0, 0,
                  0, -1,  0, 0,
                  0,  0, -1, 0,
                  0,  0,  0, 1 );
//neg z
mat4 negz = mat4( -1,  0, 0, 0,
                    0, -1, 0, 0,
                    0,  0, 1, 0,
                    0,  0, 0, 1 );

vector< light_data > lights_data; //gpu side generated data
vector< light > lights; //cpu side data for manipulating lights

GLuint the_box;

vector< layer > layers;
vector< pair< int, intersections > > light_intersections;

vec2 gws, lws, dispatch_size;

GLuint depth_texture;
GLuint normal_texture;
GLuint result_texture;
GLuint layered_penumbra_texture;
GLuint layered_shadow_texture;
//GLuint layered_translucency_texture;
GLuint gauss_texture0;
GLuint gauss_texture1;
/*GLuint gauss_translucency_texture0;
GLuint gauss_translucency_texture1;*/

GLuint lighting_shader = 0;
GLuint lighting_shader_hard_shadow = 0;
GLuint lighting_shader_hard_shadow_exponential = 0;
GLuint lighting_shader_pcf = 0;
GLuint lighting_shader_pcf_exponential = 0;
GLuint lighting_shader_pcss = 0;
GLuint lighting_shader_pcss_exponential = 0;
GLuint lighting_shader_ssss = 0;
GLuint lighting_shader_ssss_exponential = 0;
GLuint gbuffer_shader = 0;
GLuint display_shader = 0;
GLuint spot_shadow_gen_shader = 0;
GLuint point_shadow_gen_shader = 0;
GLuint cubemap_debug_shader = 0;
GLuint debug_shader = 0;
GLuint penumbra_shader = 0;
GLuint penumbra_shader_exponential = 0;
GLuint gauss_shader0 = 0;
GLuint gauss_shader0_exponential = 0;
GLuint gauss_shader0_supersampling = 0;
GLuint gauss_shader0_exponential_supersampling = 0;
GLuint gauss_shader1 = 0;
GLuint gauss_shader1_supersampling = 0;
GLuint minfilter_point_shader = 0;
GLuint penumbra_minfilter_shader = 0;
GLuint penumbra_minfilter_exponential_shader = 0;
/*GLuint translucent_lighting_shader = 0;
GLuint spot_shadow_translucent_gen_shader = 0;*/
GLuint browser_shader = 0;

#define SHADOW_CULL

bool gauss_blur_half_res = true;
bool penumbra_half_res = false;
bool exponential_shadows = false;
bool gauss_supersampling = true;

enum technique
{
  techLIGHTING = 0, techHARD_SHADOWS, techPCF, techPCSS, techSSSSMINFILTER, techSSSSBLOCKER, techLAST
};

technique active_technique = techSSSSMINFILTER;

int active_light = 0;

DebugDrawManager ddman;

///////////////////////////////////////////////////////////////////////////////////////
// functions
///////////////////////////////////////////////////////////////////////////////////////

void set_up_screen_textures()
{
  glActiveTexture( GL_TEXTURE0 );

  //set up deferred shading
  glBindTexture( GL_TEXTURE_2D, depth_texture );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
  glTexImage2D( GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32, screen.x, screen.y, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, 0 );

  glBindTexture( GL_TEXTURE_2D, normal_texture );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
  //glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
  //glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
  glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, screen.x, screen.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0 );

  glBindTexture( GL_TEXTURE_2D, result_texture );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
  glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, screen.x, screen.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0 );

  glBindTexture( GL_TEXTURE_2D, layered_penumbra_texture );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
  
  if( penumbra_half_res )
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA16F, screen.x/2, screen.y/2, 0, GL_RGBA, GL_FLOAT, 0 );
  else
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA16F, screen.x, screen.y, 0, GL_RGBA, GL_FLOAT, 0 );

  glBindTexture( GL_TEXTURE_2D, layered_shadow_texture );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );

  if( penumbra_half_res )
  {
    if( exponential_shadows )
      glTexImage2D( GL_TEXTURE_2D, 0, GL_R32F, screen.x/2, screen.y/2, 0, GL_RED, GL_FLOAT, 0 );
    else
      glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, screen.x/2, screen.y/2, 0, GL_RGBA, GL_FLOAT, 0 );
  }
  else
  {
    if( exponential_shadows )
      glTexImage2D( GL_TEXTURE_2D, 0, GL_R32F, screen.x, screen.y, 0, GL_RED, GL_FLOAT, 0 );
    else
      glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, screen.x, screen.y, 0, GL_RGBA, GL_FLOAT, 0 );
  }

  /*glBindTexture( GL_TEXTURE_2D, layered_translucency_texture );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
  
  if( penumbra_half_res )
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32UI, screen.x/2, screen.y/2, 0, GL_RGBA_INTEGER, GL_UNSIGNED_INT, 0 );
  else
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32UI, screen.x, screen.y, 0, GL_RGBA_INTEGER, GL_UNSIGNED_INT, 0 );*/

  glBindTexture( GL_TEXTURE_2D, gauss_texture0 );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
  
  if( gauss_blur_half_res )
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, screen.x/2, screen.y/2, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0 );
  else
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, screen.x, screen.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0 );

  glBindTexture( GL_TEXTURE_2D, gauss_texture1 );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
  
  if( gauss_blur_half_res )
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, screen.x/2, screen.y/2, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0 );
  else
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, screen.x, screen.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0 );

  /*glBindTexture( GL_TEXTURE_2D, gauss_translucency_texture0 );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );

  if( gauss_blur_half_res )
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32UI, screen.x/2, screen.y/2, 0, GL_RGBA_INTEGER, GL_UNSIGNED_INT, 0 );
  else
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32UI, screen.x, screen.y, 0, GL_RGBA_INTEGER, GL_UNSIGNED_INT, 0 );

  glBindTexture( GL_TEXTURE_2D, gauss_translucency_texture1 );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );

  if( gauss_blur_half_res )
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32UI, screen.x/2, screen.y/2, 0, GL_RGBA_INTEGER, GL_UNSIGNED_INT, 0 );
  else
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32UI, screen.x, screen.y, 0, GL_RGBA_INTEGER, GL_UNSIGNED_INT, 0 );*/
}

void set_workgroup_size()
{
  //set up work group sizes
  unsigned local_ws[2] = {16, 16};
  unsigned global_ws[2];
  unsigned gw = 0, gh = 0, count = 1;
  
  while( gw < screen.x )
  {
    gw = local_ws[0] * count;
    count++;
  }
  
  count = 1;
  
  while( gh < screen.y )
  {
    gh = local_ws[1] * count;
    count++;
  }

  global_ws[0] = gw;
  global_ws[1] = gh;

  gws = vec2( global_ws[0], global_ws[1] );
  lws = vec2( local_ws[0], local_ws[1] );
  dispatch_size = gws / lws;
}

int set_up_layers()
{
  layers.clear();
  light_intersections.clear();

  /**/
  //set up layers

  //TODO optimize this: O( n^2 )
  //add octree maybe?
  for( int c = 0; c < lights.size(); ++c )
  {
    light_intersections.push_back( make_pair( c, intersections() ) );
    for( int d = 0; d < lights.size(); ++d )
    {
      //if two lights intersect
      if( c != d && lights[c].bv->intersects( lights[d].bv ) )
      {  
        light_intersections[c].second.push_back( d );
      }
    }
  }

  //order vertices by their vertex degree
  sort( light_intersections.begin(), light_intersections.end(), 
  []( const pair< int, intersections >& a, const pair< int, intersections >& b ) -> bool
  {
    return a.second.size() < b.second.size();  
  } );

  auto is_conflicting_layer = []( const intersections& i, const layer& l ) -> bool
  {
    for( int c = 0; c < l.size(); ++c )
    {
      for( int d = 0; d < i.size(); ++d )
      {
        if( l[c] == i[d] )
          return true;
      }
    }

    return false;
  };

  auto get_intersections = [&]( int i )
  {
    for( int c = 0; c < light_intersections.size(); ++c )
    {
      if( light_intersections[c].first == i )
      {
        return c;
      }
    }

    return -1;
  };

  auto add_to_layers = [&]( int li ) -> bool
  {
    for( int c = 0; c < layers.size(); ++c )
    {
      if( !is_conflicting_layer( light_intersections[get_intersections(li)].second, layers[c] ) )
      {
        layers[c].push_back( li );
        return true;
      }
    }

    return false;
  };

  if( light_intersections.size() > 0 )
  {
    layers.resize( light_intersections.back().second.size() + 1 );

    for( int c = 0; c < lights.size(); ++c )
    {
      if( !add_to_layers( c ) )
      {
        std::cerr << "Couldn't add light to layers: " << c << std::endl;
      }
    }
  }

  vec3 colors[] = 
  {
    vec3(1, 0, 0),
    vec3(0, 1, 0),
    vec3(0, 0, 1),
    vec3(1, 1, 0),
    vec3(1, 0, 1),
    vec3(0, 1, 1),
    vec3(1, 1, 1),
  };

  int counter = 0;
  for( int c = 0; c < layers.size(); ++c )
  {
    if( !layers[c].empty() )
      ++counter;
  }

  //std::cout << "Number of layers used: " << counter << std::endl;

  assert( counter <= 7 ); //coloring

  for( int c = 0; c < layers.size(); ++c )
  {
    vec3 color = vec3(1);
    color = colors[c];
    for( int d = 0; d < layers[c].size(); ++d )
    {
      //lights[layers[c][d]].diffuse_color = color;
      //lights[layers[c][d]].specular_color = lights[layers[c][d]].diffuse_color;
      lights[layers[c][d]].layer = c;
    }
  }

  /**/

  return counter;
}

float get_random_num( float min, float max )
{
  return min + ( max - min ) * ( float )rand() / ( float )RAND_MAX; //min...max
}


light_data create_light( mat4 modelview, const light& l )
{
  float spot_cos_cutoff = 0.0f;
  mm::vec3 light_spot_dir_buf;
  
  mm::vec4 light_pos_buf;
  
  if( l.type == light_type::lightPOINT )
  {
    light_pos_buf = modelview * mm::vec4( l.light_cam.pos.xyz, 1.0f );
  }
  else if( l.type == light_type::lightSPOT )
  {
    spot_cos_cutoff = std::cos( mm::radians( l.spot_cutoff ) );
    light_spot_dir_buf = mm::normalize( ( modelview * mm::vec4( l.light_cam.view_dir.xyz, 0.0f ) ).xyz );
    light_pos_buf = modelview * mm::vec4( l.light_cam.pos.xyz, 1.0f );
  }
  else if( l.type == light_type::lightDIRECTIONAL )
  {
    if( mm::notEqual( l.light_cam.pos, mm::vec3( 0 ) ) )
    {
      light_pos_buf.xyz = mm::normalize( ( modelview * mm::vec4( l.light_cam.pos.xyz, 0.0f ) ).xyz );
    }
  }
  
  float att_end = 0.0f;
  
  if( l.att_type == attenuation_type::FULL )
  {
    att_end = l.radius / l.attenuation_coeff;
  }
  else if( l.att_type == attenuation_type::LINEAR )
  {
    att_end = l.radius;
  }
  
  light_data cll;
  cll.diffuse_color.xyz = l.diffuse_color;
  cll.specular_color.xyz = l.specular_color;
  cll.vs_position = light_pos_buf;
  cll.ms_position = mm::vec4( l.light_cam.pos.xyz, 1.0f );
  cll.attenuation_end = att_end;
  cll.attenuation_cutoff = l.attenuation_coeff;
  cll.attenuation_type = l.att_type;
  cll.lighting_type = l.type;
  cll.radius = l.radius;
  cll.spot_direction.xyz = light_spot_dir_buf;
  cll.spot_exponent = l.spot_exponent;
  cll.spot_direction.w = spot_cos_cutoff;
  cll.layer = l.layer;
  cll.specular_color.w = l.light_size;
  cll.spot_shadow_mat = l.spot_shadow_mat;
  memcpy( &cll.point_shadow_mats, &l.point_shadow_mats, sizeof(mat4) * 6 );

  return cll;
}

//cpp to js functions that can be called from here
namespace js
{
  //example cpp to js function call
  void cpp_to_js( const browser_instance& w, const std::wstring& str )
  {
    std::wstringstream ws;
    ws << "cpp_to_js('" << str << "');";

    browser::get().execute_javascript( w, ws.str() );
  }

  void set_user_pos( const browser_instance& w, vec3 pos )
  {
    std::wstringstream ws;
    ws << "set_user_pos(" << pos.x << ", " << pos.y << ", " << pos.z << ");";
    browser::get().execute_javascript( w, ws.str() );
  }

  void set_user_view( const browser_instance& w, vec3 view )
  {
    std::wstringstream ws;
    ws << "set_user_view(" << view.x << ", " << view.y << ", " << view.z << ");";
    browser::get().execute_javascript( w, ws.str() );
  }

  void set_user_up( const browser_instance& w, vec3 up )
  {
    std::wstringstream ws;
    ws << "set_user_up(" << up.x << ", " << up.y << ", " << up.z << ");";
    browser::get().execute_javascript( w, ws.str() );
  }

  void set_light_pos( const browser_instance& w, vec3 pos )
  {
    std::wstringstream ws;
    ws << "set_light_pos(" << pos.x << ", " << pos.y << ", " << pos.z << ");";
    browser::get().execute_javascript( w, ws.str() );
  }

  void set_light_view( const browser_instance& w, vec3 view )
  {
    std::wstringstream ws;
    ws << "set_light_view(" << view.x << ", " << view.y << ", " << view.z << ");";
    browser::get().execute_javascript( w, ws.str() );
  }

  void set_light_up( const browser_instance& w, vec3 up )
  {
    std::wstringstream ws;
    ws << "set_light_up(" << up.x << ", " << up.y << ", " << up.z << ");";
    browser::get().execute_javascript( w, ws.str() );
  }

  void set_light_color( const browser_instance& w, vec3 color )
  {
    std::wstringstream ws;
    ws << "set_light_color_cpp(" << int(color.x*255) << ", " << int(color.y*255) << ", " << int(color.z*255) << ");";
    browser::get().execute_javascript( w, ws.str() );
  }

  void set_light_radius( const browser_instance& w, float radius )
  {
    std::wstringstream ws;
    ws << "set_light_radius_cpp(" << radius << ");";
    browser::get().execute_javascript( w, ws.str() );
  }

  void set_light_size( const browser_instance& w, float size )
  {
    std::wstringstream ws;
    ws << "set_light_size_cpp(" << size << ");";
    browser::get().execute_javascript( w, ws.str() );
  }

  //this function gets called when a page is loaded
  //ie. you'd want to do any js initializing AFTER this point
  void bindings_complete( const browser_instance& w )
  {
    //override window.open function, so that creating a new berkelium window actually works
    //the berkelium api is broken... :D
    //this means that opening a window calls our function (see below)
    browser::get().execute_javascript( w, 
    L"\
    window.open = function (open) \
    { \
      return function (url, name, features) \
      { \
        open_window(url); \
        return open.call ?  open.call(window, url, name, features):open( url, name, features); \
      }; \
    }(window.open);" );

    //for manually loaded webpages (loaded from the cpp code)
    //we add an onclick function call to the <input type="file" />
    //tags, so that file selection, opening etc. works
    //see below
    browser::get().execute_javascript( w, 
    L"\
    var list = document.getElementsByTagName('input'); \
    for( var i = 0; i < list.length; ++i ) \
    { \
      var att = list[i].getAttribute('type'); \
      if(att && att == 'file') \
      { \
        list[i].onclick = function(){ choose_file('Open file of'); }; \
      } \
    } \
    " );

    std::wstringstream ws;
    ws << "bindings_complete();";

    browser::get().execute_javascript( w, ws.str() );

    ws.str(L"");
  
    std::vector<sf::VideoMode> video_modes = sf::VideoMode::getFullscreenModes();
  
    ws << L"set_resolutions_autocomplete([";
  
    std::for_each( video_modes.begin(), video_modes.end(),
                   [&]( sf::VideoMode & v )
    {
      if( v.bitsPerPixel == 32 )
        ws << L"\"" << v.width << L"x" << v.height << L"\", ";
    }
                 );
  
    ws << L"]);";

    browser::get().execute_javascript( w, ws.str() );

    js::set_light_pos( w, lights[active_light].light_cam.pos );
    js::set_light_view( w, lights[active_light].light_cam.view_dir );
    js::set_light_up( w, lights[active_light].light_cam.up_vector );
    js::set_light_color( w, lights[active_light].diffuse_color.xyz );
    js::set_light_radius( w, lights[active_light].radius );
    js::set_light_size( w, lights[active_light].light_size );
  }
}

//NOTE broken, don't use
/*void browser::onResizeRequested( Berkelium::Window* win, 
                        int x, 
                        int y, 
                        int newWidth, 
                        int newHeight )
{
  std::cout << "resize requested" << std::endl;

  frm.resize( x, y, newWidth, newHeight );
}*/

//NOTE you can do whatever with the page title, I decided to print in to the top of the window
void browser::onTitleChanged( Berkelium::Window* win,
                              Berkelium::WideString title )
{
  std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> conv;
  std::wstring str( title.mData, title.mLength );
  frm.set_title( conv.to_bytes(str) );
}

void load_shaders()
{
  if( lighting_shader )
  {
    glDeleteProgram( lighting_shader );
    lighting_shader = 0;
  }
  frm.load_shader( lighting_shader, GL_COMPUTE_SHADER, "../shaders/ssss/glsl_compute_light.cs", false, 
  "#version 430\n" );

  if( lighting_shader_hard_shadow )
  {
    glDeleteProgram( lighting_shader_hard_shadow );
    lighting_shader_hard_shadow = 0;
  }
  frm.load_shader( lighting_shader_hard_shadow, GL_COMPUTE_SHADER, "../shaders/ssss/glsl_compute_light.cs", false, 
  "#version 430\n"
  "#define HARD_SHADOW\n" );
  
  if( lighting_shader_hard_shadow_exponential )
  {
    glDeleteProgram( lighting_shader_hard_shadow_exponential );
    lighting_shader_hard_shadow_exponential = 0;
  }
  frm.load_shader( lighting_shader_hard_shadow_exponential, GL_COMPUTE_SHADER, "../shaders/ssss/glsl_compute_light.cs", false, 
  "#version 430\n"
  "#define HARD_SHADOW\n"
  "#define EXPONENTIAL_SHADOWS\n" );
  
  if( lighting_shader_pcf )
  {
    glDeleteProgram( lighting_shader_pcf );
    lighting_shader_pcf = 0;
  }
  frm.load_shader( lighting_shader_pcf, GL_COMPUTE_SHADER, "../shaders/ssss/glsl_compute_light.cs", false, 
  "#version 430\n"
  "#define PCF\n" );
  
  if( lighting_shader_pcf_exponential )
  {
    glDeleteProgram( lighting_shader_pcf_exponential );
    lighting_shader_pcf_exponential = 0;
  }
  frm.load_shader( lighting_shader_pcf_exponential, GL_COMPUTE_SHADER, "../shaders/ssss/glsl_compute_light.cs", false, 
  "#version 430\n"
  "#define PCF\n"
  "#define EXPONENTIAL_SHADOWS\n" );
  
  if( lighting_shader_pcss )
  {
    glDeleteProgram( lighting_shader_pcss );
    lighting_shader_pcss = 0;
  }
  frm.load_shader( lighting_shader_pcss, GL_COMPUTE_SHADER, "../shaders/ssss/glsl_compute_light.cs", false, 
  "#version 430\n"
  "#define PCSS\n" );
  
  if( lighting_shader_pcss_exponential )
  {
    glDeleteProgram( lighting_shader_pcss_exponential );
    lighting_shader_pcss_exponential = 0;
  }
  frm.load_shader( lighting_shader_pcss_exponential, GL_COMPUTE_SHADER, "../shaders/ssss/glsl_compute_light.cs", false, 
  "#version 430\n"
  "#define PCSS\n"
  "#define EXPONENTIAL_SHADOWS\n" );
  
  if( lighting_shader_ssss )
  {
    glDeleteProgram( lighting_shader_ssss );
    lighting_shader_ssss = 0;
  }
  frm.load_shader( lighting_shader_ssss, GL_COMPUTE_SHADER, "../shaders/ssss/glsl_compute_light.cs", false, 
  "#version 430\n"
  "#define SSSS\n" );
  
  if( lighting_shader_ssss_exponential )
  {
    glDeleteProgram( lighting_shader_ssss_exponential );
    lighting_shader_ssss_exponential = 0;
  }
  frm.load_shader( lighting_shader_ssss_exponential, GL_COMPUTE_SHADER, "../shaders/ssss/glsl_compute_light.cs", false, 
  "#version 430\n"
  "#define SSSS\n"
  "#define EXPONENTIAL_SHADOWS\n" );
  
  if( gbuffer_shader )
  {
    glDeleteProgram( gbuffer_shader );
    gbuffer_shader = 0;
  }
  frm.load_shader( gbuffer_shader, GL_VERTEX_SHADER, "../shaders/ssss/gbuffer.vs" );
  frm.load_shader( gbuffer_shader, GL_FRAGMENT_SHADER, "../shaders/ssss/gbuffer.ps" );
  
  if( display_shader )
  {
    glDeleteProgram( display_shader );
    display_shader = 0;
  }
  frm.load_shader( display_shader, GL_VERTEX_SHADER, "../shaders/ssss/display.vs" );
  frm.load_shader( display_shader, GL_FRAGMENT_SHADER, "../shaders/ssss/display.ps" );
  
  if( spot_shadow_gen_shader )
  {
    glDeleteProgram( spot_shadow_gen_shader );
    spot_shadow_gen_shader = 0;
  }
  frm.load_shader( spot_shadow_gen_shader, GL_VERTEX_SHADER, "../shaders/ssss/spot_shadow_gen.vs" );
  frm.load_shader( spot_shadow_gen_shader, GL_FRAGMENT_SHADER, "../shaders/ssss/spot_shadow_gen.ps" );
  
  if( point_shadow_gen_shader )
  {
    glDeleteProgram( point_shadow_gen_shader );
    point_shadow_gen_shader = 0;
  }
  frm.load_shader( point_shadow_gen_shader, GL_VERTEX_SHADER, "../shaders/ssss/point_shadow_gen.vs" );
  frm.load_shader( point_shadow_gen_shader, GL_GEOMETRY_SHADER, "../shaders/ssss/point_shadow_gen.gs" );
  frm.load_shader( point_shadow_gen_shader, GL_FRAGMENT_SHADER, "../shaders/ssss/point_shadow_gen.ps" );
  
  if( debug_shader )
  {
    glDeleteProgram( debug_shader );
    debug_shader = 0;
  }
  frm.load_shader( debug_shader, GL_VERTEX_SHADER, "../shaders/ssss/debug.vs" );
  frm.load_shader( debug_shader, GL_FRAGMENT_SHADER, "../shaders/ssss/debug.ps" );

  if( cubemap_debug_shader )
  {
    glDeleteProgram( cubemap_debug_shader );
    cubemap_debug_shader = 0;
  }
  frm.load_shader( cubemap_debug_shader, GL_VERTEX_SHADER, "../shaders/ssss/cubemap_debug.vs" );
  frm.load_shader( cubemap_debug_shader, GL_FRAGMENT_SHADER, "../shaders/ssss/cubemap_debug.ps" );
  
  if( penumbra_shader )
  {
    glDeleteProgram( penumbra_shader );
    penumbra_shader = 0;
  }
  frm.load_shader( penumbra_shader, GL_VERTEX_SHADER, "../shaders/ssss/penumbra.vs" );
  frm.load_shader( penumbra_shader, GL_FRAGMENT_SHADER, "../shaders/ssss/penumbra.ps", false,
  "#version 430\n" );

  if( penumbra_shader_exponential )
  {
    glDeleteProgram( penumbra_shader_exponential );
    penumbra_shader_exponential = 0;
  }
  frm.load_shader( penumbra_shader_exponential, GL_VERTEX_SHADER, "../shaders/ssss/penumbra.vs" );
  frm.load_shader( penumbra_shader_exponential, GL_FRAGMENT_SHADER, "../shaders/ssss/penumbra.ps", false, 
  "#version 430\n"
  "#define EXPONENTIAL_SHADOWS\n" );

  if( gauss_shader0 )
  {
    glDeleteProgram( gauss_shader0 );
    gauss_shader0 = 0;
  }
  frm.load_shader( gauss_shader0, GL_VERTEX_SHADER, "../shaders/ssss/gauss_blur.vs" );
  frm.load_shader( gauss_shader0, GL_FRAGMENT_SHADER, "../shaders/ssss/gauss_blur.ps", false,
  "#version 430\n" );

  if( gauss_shader0_exponential )
  {
    glDeleteProgram( gauss_shader0_exponential );
    gauss_shader0_exponential = 0;
  }
  frm.load_shader( gauss_shader0_exponential, GL_VERTEX_SHADER, "../shaders/ssss/gauss_blur.vs" );
  frm.load_shader( gauss_shader0_exponential, GL_FRAGMENT_SHADER, "../shaders/ssss/gauss_blur.ps", false,
  "#version 430\n"
  "#define EXPONENTIAL_SHADOWS\n" );

  if( gauss_shader0_supersampling )
  {
    glDeleteProgram( gauss_shader0_supersampling );
    gauss_shader0_supersampling = 0;
  }
  frm.load_shader( gauss_shader0_supersampling, GL_VERTEX_SHADER, "../shaders/ssss/gauss_blur.vs" );
  frm.load_shader( gauss_shader0_supersampling, GL_FRAGMENT_SHADER, "../shaders/ssss/gauss_blur.ps", false,
  "#version 430\n"
  "#define GAUSS_SUPERSAMPLE\n" );

  if( gauss_shader0_exponential_supersampling )
  {
    glDeleteProgram( gauss_shader0_exponential_supersampling );
    gauss_shader0_exponential_supersampling = 0;
  }
  frm.load_shader( gauss_shader0_exponential_supersampling, GL_VERTEX_SHADER, "../shaders/ssss/gauss_blur.vs" );
  frm.load_shader( gauss_shader0_exponential_supersampling, GL_FRAGMENT_SHADER, "../shaders/ssss/gauss_blur.ps", false,
  "#version 430\n"
  "#define GAUSS_SUPERSAMPLE\n"
  "#define EXPONENTIAL_SHADOWS\n" );

  if( gauss_shader1 )
  {
    glDeleteProgram( gauss_shader1 );
    gauss_shader1 = 0;
  }
  frm.load_shader( gauss_shader1, GL_VERTEX_SHADER, "../shaders/ssss/gauss_blur.vs" );
  frm.load_shader( gauss_shader1, GL_FRAGMENT_SHADER, "../shaders/ssss/gauss_blur2.ps", false,
  "#version 430\n" );

  if( gauss_shader1_supersampling )
  {
    glDeleteProgram( gauss_shader1_supersampling );
    gauss_shader1_supersampling = 0;
  }
  frm.load_shader( gauss_shader1_supersampling, GL_VERTEX_SHADER, "../shaders/ssss/gauss_blur.vs" );
  frm.load_shader( gauss_shader1_supersampling, GL_FRAGMENT_SHADER, "../shaders/ssss/gauss_blur2.ps", false,
  "#version 430\n"
  "#define GAUSS2_SUPERSAMPLE\n" );

  if( minfilter_point_shader )
  {
    glDeleteProgram( minfilter_point_shader );
    minfilter_point_shader = 0;
  }
  frm.load_shader( minfilter_point_shader, GL_VERTEX_SHADER, "../shaders/ssss/minfilter.vs" );
  frm.load_shader( minfilter_point_shader, GL_FRAGMENT_SHADER, "../shaders/ssss/minfilter_point.ps" );

  if( penumbra_minfilter_shader )
  {
    glDeleteProgram( penumbra_minfilter_shader );
    penumbra_minfilter_shader = 0;
  }
  frm.load_shader( penumbra_minfilter_shader, GL_VERTEX_SHADER, "../shaders/ssss/penumbra.vs" );
  frm.load_shader( penumbra_minfilter_shader, GL_FRAGMENT_SHADER, "../shaders/ssss/penumbra_minfilter.ps", false,
  "#version 430\n" );

  if( penumbra_minfilter_exponential_shader )
  {
    glDeleteProgram( penumbra_minfilter_exponential_shader );
    penumbra_minfilter_exponential_shader = 0;
  }
  frm.load_shader( penumbra_minfilter_exponential_shader, GL_VERTEX_SHADER, "../shaders/ssss/penumbra.vs" );
  frm.load_shader( penumbra_minfilter_exponential_shader, GL_FRAGMENT_SHADER, "../shaders/ssss/penumbra_minfilter.ps", false,
  "#version 430\n"
  "#define EXPONENTIAL_SHADOWS\n" );

  /*if( translucent_lighting_shader )
  {
    glDeleteProgram( translucent_lighting_shader );
    translucent_lighting_shader = 0;
  }
  frm.load_shader( translucent_lighting_shader, GL_VERTEX_SHADER, "../shaders/ssss/translucent_lighting.vs" );
  frm.load_shader( translucent_lighting_shader, GL_FRAGMENT_SHADER, "../shaders/ssss/translucent_lighting.ps" );

  if( spot_shadow_translucent_gen_shader )
  {
    glDeleteProgram( spot_shadow_translucent_gen_shader );
    spot_shadow_translucent_gen_shader = 0;
  }
  frm.load_shader( spot_shadow_translucent_gen_shader, GL_VERTEX_SHADER, "../shaders/ssss/spot_shadow_translucent_gen.vs" );
  frm.load_shader( spot_shadow_translucent_gen_shader, GL_FRAGMENT_SHADER, "../shaders/ssss/spot_shadow_translucent_gen.ps" );*/

  //NOTE: you should handle the rendering, so this should be custom code
  if( browser_shader )
  {
    glDeleteProgram( browser_shader );
    browser_shader = 0;
  }
  frm.load_shader( browser_shader, GL_VERTEX_SHADER, "../shaders/browser/browser.vs" );
  frm.load_shader( browser_shader, GL_FRAGMENT_SHADER, "../shaders/browser/browser.ps" );
}

void add_light()
{
  lights.resize( lights.size()+1 ); //we'll have 64 lights
  lights_data.resize( lights.size() );
  
  diffuse_color_data.resize( lights.size() );
  specular_color_data.resize( lights.size() );
  light_size_data.resize( lights.size() );
  vs_position_data.resize( lights.size() );
  ms_position_data.resize( lights.size() );
  attenuation_end_data.resize( lights.size() );
  attenuation_cutoff_data.resize( lights.size() );
  radius_data.resize( lights.size() );
  spot_exponent_data.resize( lights.size() );
  attenuation_type_data.resize( lights.size() );
  lighting_type_data.resize( lights.size() );
  layer_data.resize( lights.size() );
  spot_direction_data.resize( lights.size() );
  spot_cutoff_data.resize( lights.size() );
  spot_shadow_mat_data.resize( lights.size() );
  point_shadow_mat_data.resize( lights.size() * 6 );

  lights.back().type = light_type::lightPOINT;
  lights.back().att_type = attenuation_type::LINEAR;
  lights.back().attenuation_coeff = 0.25;
  lights.back().light_size = 0.15f*1.5f;
  lights.back().diffuse_color = vec3( get_random_num( 0.0, 1 ), get_random_num( 0.0, 1 ), get_random_num( 0.0, 1 ) );
  lights.back().specular_color = lights.back().diffuse_color;
  lights.back().radius = 30;//get_random_num( 30, 75 );
  lights.back().spot_cutoff = get_random_num( 20, 50 );
  lights.back().spot_exponent = get_random_num( 20, 40 );
  lights.back().light_cam = camera<float>();
  lights.back().light_cam.pos = cam.pos + cam.view_dir * 5;
  lights.back().light_cam.set_frame(new frame<float>());
  //const_cast<frame<float>*>(lights.back().light_cam.get_frame())->set_perspective(radians(lights.back().spot_cutoff * 2), 1, 0.1f, lights.back().radius);
  const_cast<frame<float>*>(lights.back().light_cam.get_frame())->set_perspective( radians(90.0f), 1, 1, lights.back().radius );
  lights.back().bv = new sphere(lights.back().light_cam.pos, lights.back().radius);

  lights.back().point_shadow_mats[0] = lights.back().light_cam.get_frame()->projection_matrix * posx;
  lights.back().point_shadow_mats[1] = lights.back().light_cam.get_frame()->projection_matrix * negx;
  lights.back().point_shadow_mats[2] = lights.back().light_cam.get_frame()->projection_matrix * posy;
  lights.back().point_shadow_mats[3] = lights.back().light_cam.get_frame()->projection_matrix * negy;
  lights.back().point_shadow_mats[4] = lights.back().light_cam.get_frame()->projection_matrix * posz;
  lights.back().point_shadow_mats[5] = lights.back().light_cam.get_frame()->projection_matrix * negz;

  for( int d = 0; d < 6; ++d )
  {
    lights.back().point_shadow_cams[d] = camera<float>();
    lights.back().point_shadow_cams[d].move_forward( lights.back().point_shadow_cams[d].pos.z );
    lights.back().point_shadow_cams[d].move_up( lights.back().point_shadow_cams[d].pos.y );
    lights.back().point_shadow_cams[d].move_right( lights.back().point_shadow_cams[d].pos.x );
    lights.back().point_shadow_cams[d].set_frame( const_cast<frame<float>*>(lights.back().light_cam.get_frame()) );
  }

  lights.back().point_shadow_cams[0].rotate_y( radians( 90 ) );  //posx
  lights.back().point_shadow_cams[1].rotate_y( radians( -90 ) ); //negx
  lights.back().point_shadow_cams[2].rotate_x( radians( 90 ) );  //posy
  lights.back().point_shadow_cams[3].rotate_x( radians( 90 ) );  //negy
  lights.back().point_shadow_cams[4].rotate_y( radians( 180 ) ); //posz
  //negz

  for( int d = 0; d < 6; ++d )
  {
    lights.back().point_shadow_frustums[d].set_up( lights.back().point_shadow_cams[d] );
  }
}

void remove_light()
{
  auto it = lights.begin();
  int counter = 0;
  for( ; counter != active_light; ++counter, ++it );
  lights.erase( it );
  active_light = 0;
}

int main( int argc, char** argv )
{
  shape::set_up_intersection();
  srand( time( 0 ) );

  for( int c = 1; c < argc; ++c )
  {
    args[argv[c]] = c + 1 < argc ? argv[c + 1] : "";
    ++c;
  }

  cout << "Arguments: " << endl;
  for_each( args.begin(), args.end(), []( pair<string, string> p )
  {
    cout << p.first << " " << p.second << endl;
  } );

  /*
   * Process program arguments
   */

  stringstream ss;
  ss.str( args["--screenx"] );
  ss >> screen.x;
  ss.clear();
  ss.str( args["--screeny"] );
  ss >> screen.y;
  ss.clear();

  if( screen.x == 0 )
  {
    screen.x = 1280;
  }

  if( screen.y == 0 )
  {
    screen.y = 720;
  }

  try
  {
    args.at( "--fullscreen" );
    fullscreen = true;
  }
  catch( ... ) {}

  try
  {
    args.at( "--help" );
    cout << title << ", written by Marton Tamas and Viktor Heisenberger." << endl <<
         "Usage: --silent      //don't display FPS info in the terminal" << endl <<
         "       --screenx num //set screen width (default:1280)" << endl <<
         "       --screeny num //set screen height (default:720)" << endl <<
         "       --fullscreen  //set fullscreen, windowed by default" << endl <<
         "       --help        //display this information" << endl;
    return 0;
  }
  catch( ... ) {}

  try
  {
    args.at( "--silent" );
    silent = true;
  }
  catch( ... ) {}

  /*
   * Initialize the OpenGL context
   */
  
  //screen.x = 1920;
  //screen.y = 1080;
  //fullscreen = true;

  frm.init( screen, title, fullscreen );
  frm.set_vsync( false );

  //set opengl settings
  glEnable( GL_DEPTH_TEST );
  glDepthFunc( GL_LEQUAL );
  glFrontFace( GL_CCW );
  glEnable( GL_CULL_FACE );
  glClearColor( 0.5f, 0.5f, 0.8f, 0.0f ); //sky color
  glClearDepth( 1.0f );
  glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS); //we want pretty please!

  glViewport( 0, 0, screen.x, screen.y );

  frm.get_opengl_error();

  /*
   * Set up mymath
   */

  the_frame.set_perspective( radians( cam_fov ), ( float )screen.x / ( float )screen.y, cam_near, cam_far );
  ppl.set_model_view_matrix_stack( &mvm );

  cam.set_frame( &the_frame );

  /*
   * Set up the scene
   */

  //scene octree
  o->set_up_octree(&o);

  quad = frm.create_quad( the_frame.far_ll.xyz, the_frame.far_lr.xyz, the_frame.far_ul.xyz, the_frame.far_ur.xyz );
  ss_quad = frm.create_quad( vec3(-1, -1, 0), vec3(1, -1, 0), vec3(-1, 1, 0), vec3(1, 1, 0) );

  float move_amount = 10;

  //nummeshes = 25;
  scene.resize( nummeshes );

  for( int c = 0; c < nummeshes; ++c )
  {
    stringstream ss;
    ss << "../resources/mesh/mesh" << c << ".mesh";
    //ss << "../resources/mesh/mesh" << 24 << ".mesh";
    scene[c].read_mesh( ss.str() );

    //create aabb for this
    scene[c].bv = new aabb();
    for( auto d : scene[c].vertices )
    {
      static_cast<aabb*>(scene[c].bv)->expand( d );
    }

    //ddman.CreateAABoxMinMax( static_cast<aabb*>(scene[c].bv)->min, static_cast<aabb*>(scene[c].bv)->max, -1 );

    o->insert(c, scene[c].bv);
  }

  //mesh::save_meshes( "out.obj", scene );

  //vector< mesh > meshes;
  //mesh::load_into_meshes( "../resources/plane.obj", meshes );

  //mesh::save_meshes( "out.obj", meshes );

  for( auto & c : scene )
    c.upload();

  //cleanup
  for( auto& c : scene )
  {
    c.indices.resize(0);
    c.indices.reserve(0);
    c.vertices.resize(0);
    c.vertices.reserve(0);
    c.normals.resize(0);
    c.normals.reserve(0);
    c.tex_coords.resize(0);
    c.tex_coords.reserve(0);
    c.tangents.resize(0);
    c.tangents.reserve(0);
  }

  the_box = frm.create_box();

  /*
   * Set up lights
   */

  /**/
  //NOTE: set to 16 to have the default light setup
  lights.resize( 1 ); //we'll have 64 lights
  lights_data.resize( lights.size() );
  
  diffuse_color_data.resize( lights.size() );
  specular_color_data.resize( lights.size() );
  light_size_data.resize( lights.size() );
  vs_position_data.resize( lights.size() );
  ms_position_data.resize( lights.size() );
  attenuation_end_data.resize( lights.size() );
  attenuation_cutoff_data.resize( lights.size() );
  radius_data.resize( lights.size() );
  spot_exponent_data.resize( lights.size() );
  attenuation_type_data.resize( lights.size() );
  lighting_type_data.resize( lights.size() );
  layer_data.resize( lights.size() );
  spot_direction_data.resize( lights.size() );
  spot_cutoff_data.resize( lights.size() );
  spot_shadow_mat_data.resize( lights.size() );
  point_shadow_mat_data.resize( lights.size() * 6 );

  for( int c = 0; c < lights.size(); ++c )
  {
    lights[c].type = light_type::lightPOINT;
    lights[c].att_type = attenuation_type::LINEAR;
    lights[c].attenuation_coeff = 0.25;
    lights[c].light_size = 0.15f*1.5f;
    lights[c].diffuse_color = vec3( get_random_num( 0.0, 1 ), get_random_num( 0.0, 1 ), get_random_num( 0.0, 1 ) );
    lights[c].specular_color = lights[c].diffuse_color;
    lights[c].radius = 30;//get_random_num( 30, 75 );
    lights[c].spot_cutoff = get_random_num( 20, 50 );
    lights[c].spot_exponent = get_random_num( 20, 40 );
    lights[c].light_cam = camera<float>();
    lights[c].light_cam.move_forward( c%4 == 0 || c%4 == 1 ? -5 : 15 );
    //lights[c].light_cam.move_forward( get_random_num( -50, 50 ) );
    //lights[c].light_cam.move_right( get_random_num( -100, 100 ) );
    lights[c].light_cam.move_right( -90 + c/4 * 50 );
    //lights[c].light_cam.move_up( get_random_num( 5, 100 ) );
    lights[c].light_cam.move_up( c%4 == 0 || c%4 == 2 ? 10 : 70);
    //lights[c].light_cam.rotate_y( radians( get_random_num( 0, 360 ) ) );
    lights[c].light_cam.set_frame(new frame<float>());
    //const_cast<frame<float>*>(lights[c].light_cam.get_frame())->set_perspective(radians(lights[c].spot_cutoff * 2), 1, 0.1f, lights[c].radius);
    const_cast<frame<float>*>(lights[c].light_cam.get_frame())->set_perspective( radians(90.0f), 1, 1, lights[c].radius );
    lights[c].bv = new sphere(lights[c].light_cam.pos, lights[c].radius);

    lights[c].point_shadow_mats[0] = lights[c].light_cam.get_frame()->projection_matrix * posx;
    lights[c].point_shadow_mats[1] = lights[c].light_cam.get_frame()->projection_matrix * negx;
    lights[c].point_shadow_mats[2] = lights[c].light_cam.get_frame()->projection_matrix * posy;
    lights[c].point_shadow_mats[3] = lights[c].light_cam.get_frame()->projection_matrix * negy;
    lights[c].point_shadow_mats[4] = lights[c].light_cam.get_frame()->projection_matrix * posz;
    lights[c].point_shadow_mats[5] = lights[c].light_cam.get_frame()->projection_matrix * negz;

    for( int d = 0; d < 6; ++d )
    {
      lights[c].point_shadow_cams[d] = camera<float>();
      lights[c].point_shadow_cams[d].move_forward( lights[c].point_shadow_cams[d].pos.z );
      lights[c].point_shadow_cams[d].move_up( lights[c].point_shadow_cams[d].pos.y );
      lights[c].point_shadow_cams[d].move_right( lights[c].point_shadow_cams[d].pos.x );
      lights[c].point_shadow_cams[d].set_frame( const_cast<frame<float>*>(lights[c].light_cam.get_frame()) );
    }

    lights[c].point_shadow_cams[0].rotate_y( radians( 90 ) );  //posx
    lights[c].point_shadow_cams[1].rotate_y( radians( -90 ) ); //negx
    lights[c].point_shadow_cams[2].rotate_x( radians( 90 ) );  //posy
    lights[c].point_shadow_cams[3].rotate_x( radians( 90 ) );  //negy
    lights[c].point_shadow_cams[4].rotate_y( radians( 180 ) ); //posz
    //negz

    for( int d = 0; d < 6; ++d )
    {
      lights[c].point_shadow_frustums[d].set_up( lights[c].point_shadow_cams[d] );
    }

    //ddman.CreateSphere( lights[c].light_cam.pos, lights[c].radius, -1 );
  }
  /**/

  set_up_layers();

  /*
   * Set up the browser
   */

  //init with the path where the berkelium resources can be found
  //this only works with berkelium >11
  //for berkelium 8 (on linux) these need to be in the same folder as the exe
  browser::get().init( L"../resources/berkelium/win32" );

  //this is a browser window
  //you can have multiple, and render them whereever you'd like to
  //like, display a youtube video
  //run a webgl game on one of the walls etc.
  //you get a texture, and you can render it whereever or however you'd like to
  browser_instance b; 
  browser::get().create( b, screen ); //automatically navigates to google.com
  //uncomment this to see the demo ui
  std::string path = frm.get_app_path();
  browser::get().navigate( b, "file:///"+path+"resources/ui/ui.html" ); //user our local GUI file

  //NOTE these are important bits here
  //use callbacks like these to handle js to cpp function calls
  //these will be called when you call the browser::update function

  //example js to cpp function call
  browser::get().register_callback( L"js_to_cpp", fun( []( std::wstring str )
  {
    wcout << str << endl;
    js::cpp_to_js( browser::get().get_last_callback_window(), L"cpp to js function call" );
  }, std::wstring() ) ); //setting the type here will make sure a functor with an argument will be created (omit this to create a functor wo/ an argument)

  browser::get().register_callback( L"go_to_fixed_position", fun( []( int val )
  {
    cam.pos = fixed_positions[val*3+0];
    cam.view_dir = fixed_positions[val*3+1];
    cam.up_vector = fixed_positions[val*3+2];
  }, int() ) );

  browser::get().register_callback( L"set_resolution", fun( []( std::wstring str )
  {
    int w, h;
    if( str != L"" )
    {
      w = stoi(str.substr( 0, str.find(L"x") ));
      h = stoi(str.substr( str.find(L"x")+1, std::wstring::npos ));
    
      frm.resize( frm.get_window_pos().x, frm.get_window_pos().y, w, h );
    }
  }, std::wstring() ) );

  browser::get().register_callback( L"set_technique", fun( []( std::wstring str )
  {
    if( str == L"SSSS minfilter" )
    {
      active_technique = techSSSSMINFILTER;
    }
    else if( str == L"SSSS blocker search" )
    {
      active_technique = techSSSSBLOCKER;
    }
    else if( str == L"PCSS" )
    {
      active_technique = techPCSS;
    }
    else if( str == L"PCF" )
    {
      active_technique = techPCF;
    }
    else if( str == L"Hard Shadows" )
    {
      active_technique = techHARD_SHADOWS;
    }
    else
      active_technique = techLIGHTING;

  }, std::wstring() ) );

  browser::get().register_callback( L"set_gauss_blur_res", fun( []( bool val )
  {
    gauss_blur_half_res = !val;
    set_up_screen_textures();
  }, bool() ) );

  browser::get().register_callback( L"set_penumbra_res", fun( []( bool val )
  {
    penumbra_half_res = !val;
    set_up_screen_textures();
  }, bool() ) );

  browser::get().register_callback( L"set_supersampling", fun( []( bool val )
  {
    gauss_supersampling = val;
  }, bool() ) );

  browser::get().register_callback( L"set_exponential", fun( []( bool val )
  {
    exponential_shadows = val;
  }, bool() ) );

  browser::get().register_callback( L"set_light_color", fun( []( std::wstring val )
  {
    float r, g, b;
    std::wstring rs, gs, bs;
    rs = val.substr( 0, val.find( L" " ) );
    gs = val.substr( val.find( L" " )+1, val.rfind(L" ") - (val.find( L" " )+1) );
    bs = val.substr( val.rfind( L" " )+1, std::wstring::npos );
    r = stod( rs );
    g = stod( gs );
    b = stod( bs );
    lights[active_light].diffuse_color = vec3( r, g, b );
    lights[active_light].specular_color = lights[active_light].diffuse_color;
  }, std::wstring() ) );

  browser::get().register_callback( L"set_light_radius", fun( []( float val )
  {
    lights[active_light].radius = val;
    delete lights[active_light].bv;
    lights[active_light].bv = new sphere(lights[active_light].light_cam.pos, lights[active_light].radius);
    const_cast<frame<float>*>(lights[active_light].light_cam.get_frame())->set_perspective( radians(90.0f), 1, 1, lights[active_light].radius );
    lights[active_light].point_shadow_mats[0] = lights[active_light].light_cam.get_frame()->projection_matrix * posx;
    lights[active_light].point_shadow_mats[1] = lights[active_light].light_cam.get_frame()->projection_matrix * negx;
    lights[active_light].point_shadow_mats[2] = lights[active_light].light_cam.get_frame()->projection_matrix * posy;
    lights[active_light].point_shadow_mats[3] = lights[active_light].light_cam.get_frame()->projection_matrix * negy;
    lights[active_light].point_shadow_mats[4] = lights[active_light].light_cam.get_frame()->projection_matrix * posz;
    lights[active_light].point_shadow_mats[5] = lights[active_light].light_cam.get_frame()->projection_matrix * negz;

    for( int d = 0; d < 6; ++d )
    {
      lights[active_light].point_shadow_frustums[d].set_up( lights[active_light].point_shadow_cams[d] );
    }
  }, float() ) );

  browser::get().register_callback( L"set_light_size", fun( []( float val )
  {
    lights[active_light].light_size = val;
  }, float() ) );

  browser::get().register_callback( L"reload_shaders", fun( []()
  {
    load_shaders();
  } ) );

  browser::get().register_callback( L"add_light", fun( []()
  {
    if( lights.size() < 16 )
    {
      add_light();
    }
  } ) );

  browser::get().register_callback( L"remove_light", fun( []()
  {
    if( lights.size() > 1 )
    {
      remove_light();
    }
  } ) );

  /*
   * Set up the shaders
   */

  load_shaders();

  /*
   * Set up fbos / textures
   */

  //set up shadow textures
  uvec2 shadow_texture_size( 1024 );

  //set up deferred shading
  depth_texture = 0;
  glGenTextures( 1, &depth_texture );

  normal_texture = 0;
  glGenTextures( 1, &normal_texture );

  //set up lighting result texture
  result_texture = 0;
  glGenTextures( 1, &result_texture );

  //set up penumbra
  layered_penumbra_texture = 0;
  glGenTextures( 1, &layered_penumbra_texture );

  layered_shadow_texture = 0;
  glGenTextures( 1, &layered_shadow_texture );

  /*layered_translucency_texture = 0;
  glGenTextures( 1, &layered_translucency_texture );*/

  //set up gauss blur textures
  gauss_texture0 = 0;
  glGenTextures( 1, &gauss_texture0 );

  gauss_texture1 = 0;
  glGenTextures( 1, &gauss_texture1 );

  /*gauss_translucency_texture0 = 0;
  glGenTextures( 1, &gauss_translucency_texture0 );

  gauss_translucency_texture1 = 0;
  glGenTextures( 1, &gauss_translucency_texture1 );*/

  set_up_screen_textures();

  set_workgroup_size();

  //set up deferred shading
  GLuint gbuffer_fbo = 0;
  glGenFramebuffers( 1, &gbuffer_fbo );
  glBindFramebuffer( GL_FRAMEBUFFER, gbuffer_fbo );
  glDrawBuffer( GL_COLOR_ATTACHMENT0 );

  glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, normal_texture, 0 );
  glFramebufferTexture2D( GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth_texture, 0 );

  frm.check_fbo_status();

  //set up local light result texture
  /*GLuint local_light_result_texture;
  glGenTextures( 1, &local_light_result_texture );
  glBindTexture( GL_TEXTURE_BUFFER, local_light_result_texture );

  GLuint local_light_result_buffer; //TBO
  glGenBuffers( 1, &local_light_result_buffer );
  glBindBuffer( GL_TEXTURE_BUFFER, local_light_result_buffer );
  glBufferData( GL_TEXTURE_BUFFER, 1024 * dispatch_size.x * dispatch_size.y, 0, GL_DYNAMIC_DRAW );
  glTexBuffer( GL_TEXTURE_BUFFER, GL_R32F, local_light_result_buffer );*/

  GLuint spot_shadow_texture;
  glGenTextures(1, &spot_shadow_texture);
  glBindTexture(GL_TEXTURE_2D_ARRAY, spot_shadow_texture);

  glTexParameteri( GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
  glTexParameteri( GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
  glTexParameteri( GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
  glTexParameteri( GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
  glTexImage3D( GL_TEXTURE_2D_ARRAY, 0, GL_DEPTH_COMPONENT32, shadow_texture_size.x, shadow_texture_size.y, 16, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, 0 );

  vector<GLuint> spot_shadow_fbos;
  spot_shadow_fbos.resize( 16 );
  glGenFramebuffers( 16, &spot_shadow_fbos[0] );

  for( int c = 0; c < 16; ++c )
  {
    glBindFramebuffer( GL_FRAMEBUFFER, spot_shadow_fbos[c] );
    glDrawBuffer( GL_NONE );
    glFramebufferTextureLayer( GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, spot_shadow_texture, 0, c );
    frm.check_fbo_status();
  }

  //point light shadows
  GLuint point_shadow_texture;
  glGenTextures(1, &point_shadow_texture);
  glBindTexture(GL_TEXTURE_CUBE_MAP_ARRAY, point_shadow_texture);

  glTexParameteri( GL_TEXTURE_CUBE_MAP_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
  glTexParameteri( GL_TEXTURE_CUBE_MAP_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
  glTexParameteri( GL_TEXTURE_CUBE_MAP_ARRAY, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE );
  glTexParameteri( GL_TEXTURE_CUBE_MAP_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
  glTexParameteri( GL_TEXTURE_CUBE_MAP_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
  glTexStorage3D( GL_TEXTURE_CUBE_MAP_ARRAY, 1, GL_DEPTH_COMPONENT32, shadow_texture_size.x, shadow_texture_size.y, 16 * 6 );

  vector< GLuint > point_shadow_texture_views;
  point_shadow_texture_views.resize(16*6);
  glGenTextures(16*6, &point_shadow_texture_views[0]);
  
  for( int c = 0; c < 16; ++c )
  {
    for( int d = 0; d < 6; ++d )
    {
      glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
      glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
      glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
      glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
      glTextureView( point_shadow_texture_views[c*6+d], GL_TEXTURE_2D, point_shadow_texture, GL_DEPTH_COMPONENT32, 0, 1, c*6+d, 1 );
    }
  }

  /*GLuint point_shadow_translucent_texture;
  glGenTextures(1, &point_shadow_translucent_texture);
  glBindTexture(GL_TEXTURE_CUBE_MAP_ARRAY, point_shadow_translucent_texture);

  glTexParameteri( GL_TEXTURE_CUBE_MAP_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
  glTexParameteri( GL_TEXTURE_CUBE_MAP_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
  glTexParameteri( GL_TEXTURE_CUBE_MAP_ARRAY, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE );
  glTexParameteri( GL_TEXTURE_CUBE_MAP_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
  glTexParameteri( GL_TEXTURE_CUBE_MAP_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
  glTexImage3D( GL_TEXTURE_CUBE_MAP_ARRAY, 0, GL_RGBA8, shadow_texture_size.x, shadow_texture_size.y, 1 * 6, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0 );*/

  GLuint point_shadow_minfilter_texture0;
  glGenTextures(1, &point_shadow_minfilter_texture0);
  glBindTexture(GL_TEXTURE_CUBE_MAP_ARRAY, point_shadow_minfilter_texture0);

  glTexParameteri( GL_TEXTURE_CUBE_MAP_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
  glTexParameteri( GL_TEXTURE_CUBE_MAP_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
  glTexParameteri( GL_TEXTURE_CUBE_MAP_ARRAY, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE );
  glTexParameteri( GL_TEXTURE_CUBE_MAP_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
  glTexParameteri( GL_TEXTURE_CUBE_MAP_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
  //glTexParameteri( GL_TEXTURE_CUBE_MAP_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
  //glTexParameteri( GL_TEXTURE_CUBE_MAP_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
  //glTexImage3D( GL_TEXTURE_CUBE_MAP_ARRAY, 0, GL_R32F, shadow_texture_size.x, shadow_texture_size.y, 16 * 6, 0, GL_RED, GL_FLOAT, 0 );
  //glTexImage3D( GL_TEXTURE_CUBE_MAP_ARRAY, 0, GL_R32F, 64, 64, 16 * 6, 0, GL_RED, GL_FLOAT, 0 );
  glTexStorage3D( GL_TEXTURE_CUBE_MAP_ARRAY, 1, GL_R32F, 64, 64, 16 * 6 );

  vector< GLuint > point_shadow_minfilter_texture0_views;
  point_shadow_minfilter_texture0_views.resize(16*6);
  glGenTextures(16*6, &point_shadow_minfilter_texture0_views[0]);
  
  for( int c = 0; c < 16; ++c )
  {
    for( int d = 0; d < 6; ++d )
    {
      glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
      glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
      glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
      glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
      glTextureView( point_shadow_minfilter_texture0_views[c*6+d], GL_TEXTURE_2D, point_shadow_minfilter_texture0, GL_R32F, 0, 1, c*6+d, 1 );
    }
  }

  GLuint point_shadow_minfilter_texture1;
  glGenTextures(1, &point_shadow_minfilter_texture1);
  glBindTexture(GL_TEXTURE_CUBE_MAP_ARRAY, point_shadow_minfilter_texture1);

  glTexParameteri( GL_TEXTURE_CUBE_MAP_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
  glTexParameteri( GL_TEXTURE_CUBE_MAP_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
  glTexParameteri( GL_TEXTURE_CUBE_MAP_ARRAY, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE );
  //glTexParameteri( GL_TEXTURE_CUBE_MAP_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
  //glTexParameteri( GL_TEXTURE_CUBE_MAP_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
  glTexParameteri( GL_TEXTURE_CUBE_MAP_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
  glTexParameteri( GL_TEXTURE_CUBE_MAP_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
  glTexImage3D( GL_TEXTURE_CUBE_MAP_ARRAY, 0, GL_R32F, 64, 64, 16 * 6, 0, GL_RED, GL_FLOAT, 0 );

  vector<GLuint> point_shadow_fbos;
  vector<GLuint> point_shadow_minfilter_fbos0;
  vector<GLuint> point_shadow_minfilter_fbos1;
  point_shadow_fbos.resize( 16 * 6 );
  point_shadow_minfilter_fbos0.resize( 16 * 6 );
  point_shadow_minfilter_fbos1.resize( 16 * 6 );
  //we'll render each cubemap face separately, so we'll generate an fbo for each of the faces
  glGenFramebuffers( 16 * 6, &point_shadow_fbos[0] );
  glGenFramebuffers( 16 * 6, &point_shadow_minfilter_fbos0[0] );
  glGenFramebuffers( 16 * 6, &point_shadow_minfilter_fbos1[0] );

  for( int c = 0; c < 16; ++c )
  {
    for( int d = 0; d < 6; ++d )
    {
      glBindFramebuffer( GL_FRAMEBUFFER, point_shadow_fbos[c * 6 + d] );
      
      /*if( c == 13 )
      {
        glDrawBuffer( GL_COLOR_ATTACHMENT0 );
      }
      else*/
      {
        glDrawBuffer( GL_NONE );
      }

      //glFramebufferTexture( GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, point_shadow_texture, 0 );
      glFramebufferTextureLayer( GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, point_shadow_texture, 0, c * 6 + d );
      //glFramebufferTextureLayer( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, point_shadow_translucent_texture, 0, 0 * 6 + d );
      frm.check_fbo_status();

      glBindFramebuffer( GL_FRAMEBUFFER, point_shadow_minfilter_fbos0[c * 6 + d] );
      glDrawBuffer( GL_COLOR_ATTACHMENT0 );
      //glFramebufferTexture( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, point_shadow_minfilter_texture0, 0 );
      glFramebufferTextureLayer( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, point_shadow_minfilter_texture0, 0, c * 6 + d );
      frm.check_fbo_status();

      glBindFramebuffer( GL_FRAMEBUFFER, point_shadow_minfilter_fbos1[c * 6 + d] );
      glDrawBuffer( GL_COLOR_ATTACHMENT0 );
      //glFramebufferTexture( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, point_shadow_minfilter_texture1, 0 );
      glFramebufferTextureLayer( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, point_shadow_minfilter_texture1, 0, c * 6 + d );
      frm.check_fbo_status();
    }
  }

  GLuint penumbra_fbo = 0;
  glGenFramebuffers( 1, &penumbra_fbo );
  glBindFramebuffer( GL_FRAMEBUFFER, penumbra_fbo );
  GLenum modes[2] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
  glDrawBuffers( 2, modes );
  //GLenum modes[3] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2 };
  //glDrawBuffers( 3, modes );

  glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, layered_shadow_texture, 0 );
  glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, layered_penumbra_texture, 0 );
  //glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, layered_translucency_texture, 0 );

  frm.check_fbo_status();

  GLuint gauss_fbo0 = 0;
  glGenFramebuffers( 1, &gauss_fbo0 );
  glBindFramebuffer( GL_FRAMEBUFFER, gauss_fbo0 );
  GLenum gauss_modes[] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
  glDrawBuffer( GL_COLOR_ATTACHMENT0 );
  //glDrawBuffers( 2, gauss_modes );

  glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, gauss_texture0, 0 );
  //glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, gauss_translucency_texture0, 0 );

  frm.check_fbo_status();

  GLuint gauss_fbo1 = 0;
  glGenFramebuffers( 1, &gauss_fbo1 );
  glBindFramebuffer( GL_FRAMEBUFFER, gauss_fbo1 );
  glDrawBuffer( GL_COLOR_ATTACHMENT0 );
  //glDrawBuffers( 2, gauss_modes );

  glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, gauss_texture1, 0 );
  //glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, gauss_translucency_texture1, 0 );

  frm.check_fbo_status();

  //set up translucent stuff
  GLuint translucent_fbo = 0;
  glGenFramebuffers( 1, &translucent_fbo );
  glBindFramebuffer( GL_FRAMEBUFFER, translucent_fbo );
  glDrawBuffer( GL_COLOR_ATTACHMENT0 );

  glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, result_texture, 0 );
  glFramebufferTexture2D( GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth_texture, 0 );

  frm.check_fbo_status();

  glBindFramebuffer( GL_FRAMEBUFFER, 0 );

  frm.get_opengl_error();

  /*
   * Handle events
   */

  //resizing works here, but doesn't work from the browser
  //maybe driver bug?
  //frm.resize( frm.get_window_pos().x, frm.get_window_pos().y, 1920, 1080 );

  //go to fixed pos
  cam.pos = fixed_positions[8*3+0];
  cam.view_dir = fixed_positions[8*3+1];
  cam.up_vector = fixed_positions[8*3+2];

  bool warped = false, ignore = true;
  bool move_cam = false;
  vec2 last_mpos = vec2(0.5);
  vec2 movement_speed;

  vec2 mouse_pos = vec2(0.5);
  bool clicked = false;

  bool do_translate = false;

  auto event_handler = [&]( const sf::Event & ev )
  {
    switch( ev.type )
    {
      case sf::Event::MouseMoved:
      {
        vec2 mpos( ev.mouseMove.x / float( screen.x ), ev.mouseMove.y / float( screen.y ) );
        mouse_pos = mpos;
        mouse_pos.y = 1 - mouse_pos.y;

        if( move_cam )
        {
          if( warped )
          {
            ignore = false;
          }
          else
          {
            frm.set_mouse_pos( ivec2( screen.x / 2.0f, screen.y / 2.0f ) );
            warped = true;
            ignore = true;
          }

          if( !ignore && notEqual( mpos, vec2( 0.5 ) ) )
          {
            cam.rotate_world( radians( -180.0f * ( mpos.x - 0.5f ) ), vec3( 0.0f, 1.0f, 0.0f ) );
            cam.rotate_x( radians( 180.0f * ( mpos.y - 0.5f ) ) );
            frm.set_mouse_pos( ivec2( screen.x / 2.0f, screen.y / 2.0f ) );
            warped = true;

            js::set_user_view( b, cam.view_dir );
            js::set_user_up( b, cam.up_vector );
          }
        }

        if( !move_cam )
        {
          last_mpos = mpos;
        }

        browser::get().mouse_moved( b, mpos );

        break;
      }
      case sf::Event::KeyPressed:
      {
        if( ev.key.code == sf::Keyboard::Space )
        {
          fstream f;
          f.open("out.txt", ios::out);
          f << " Pos: " << cam.pos << " View: " << cam.view_dir << " Up: " << cam.up_vector << endl;
          f.close();
        }

        if( ev.key.code == sf::Keyboard::BackSpace )
        {
          //debug
          //frm.resize(0, 0, 1920, 1080 );
        }

        if( ev.key.code == sf::Keyboard::T && !do_translate )
        {
          frm.set_mouse_pos( ivec2( screen.x / 2.0f, screen.y / 2.0f ) );
          mouse_pos = vec2(0.5);

          do_translate = true;
        }

        if( ev.key.code == sf::Keyboard::Add )
        {
          if( lights.size() < 16 )
          {
            add_light();
          }
        }

        if( ev.key.code == sf::Keyboard::Subtract )
        {
          if( lights.size() > 1 )
          {
            remove_light();
          }
        }

        break;
      }
      case sf::Event::KeyReleased:
      {
        if( ev.key.code == sf::Keyboard::T )
        {
          do_translate = false;
        }

        break;
      }
      case sf::Event::TextEntered:
      {
        wchar_t txt[2];
        txt[0] = ev.text.unicode;
        txt[1] = '\0';
        browser::get().text_entered( b, txt );

        break;
      }
      case sf::Event::MouseButtonPressed:
      {
        if( ev.mouseButton.button == sf::Mouse::Left )
        {
          browser::get().mouse_button_event( b, sf::Mouse::Left, true );
          clicked = true;
        }
        else
        {
          browser::get().mouse_button_event( b, sf::Mouse::Right, true );
          move_cam = true;
          frm.set_mouse_visibility( false );
          frm.set_mouse_pos( ivec2( screen.x / 2.0f, screen.y / 2.0f ) );
        }

        break;
      }
      case sf::Event::MouseButtonReleased:
      {
        if( ev.mouseButton.button == sf::Mouse::Left )
        {
          browser::get().mouse_button_event( b, sf::Mouse::Left, false );
          clicked = false;
        }
        else
        {
          browser::get().mouse_button_event( b, sf::Mouse::Right, false );
          move_cam = false;
          frm.set_mouse_visibility( true );
          frm.set_mouse_pos( ivec2( last_mpos.x * screen.x, last_mpos.y * screen.y ) );
        }

        break;
      }
      case sf::Event::MouseWheelMoved:
      {
        browser::get().mouse_wheel_moved( b, ev.mouseWheel.delta * 100.0f );

        break;
      }
      case sf::Event::Resized:
      {
        screen = uvec2( ev.size.width, ev.size.height );

        glViewport( 0, 0, screen.x, screen.y );

        browser::get().resize( b, screen );
        set_workgroup_size();
        set_up_screen_textures();
        the_frame.set_perspective( radians( cam_fov ), ( float )screen.x / ( float )screen.y, cam_near, cam_far );

        cerr << "Screen resize: " << screen << endl;

        break;
      }
      default:
        break;
    }
  };

  /*
   * Render
   */

  static vector<unsigned> culled_objs;
  culled_objs.clear();

  unsigned draw_calls = 0;

  int frame_count = 0;

  sf::Clock timer;
  timer.restart();

  sf::Clock movement_timer;
  movement_timer.restart();

  float orig_mov_amount = move_amount;

  int num_layers_used;

  frm.display( [&]
  {
    frm.handle_events( event_handler );

    browser::get().update();

    //display debug stuff
    static stringstream ss;

    ++frame_count;

    if( timer.getElapsedTime().asMilliseconds() > 1000 )
    {
      int timepassed = timer.getElapsedTime().asMilliseconds();
      int fps = 1000.0f / ( (float)timepassed / (float)frame_count );

      ss << "FPS: " << fps << " --- Time: " << (float)timepassed / (float) frame_count << " ms ";
      ss << " --- Draw calls: " << draw_calls / (float)frame_count << " --- Lights: " << culled_objs.size();
      ss << " --- Number of layers used: " << num_layers_used;

      draw_calls = 0;
      frame_count = 0;
      timer.restart();

      frm.set_title( ss.str() );
      ss.str("");

      //debug
      //cout << screen;
      //cout << dispatch_size;
    }

    float seconds = movement_timer.getElapsedTime().asMilliseconds() / 1000.0f;

    if( sf::Keyboard::isKeyPressed( sf::Keyboard::LShift ) || sf::Keyboard::isKeyPressed( sf::Keyboard::RShift ) )
    {
      move_amount = orig_mov_amount * 3.0f;
    }
    else
    {
      move_amount = orig_mov_amount;
    }

    if( seconds > 0.01667 )
    {
      //move camera
      if( sf::Keyboard::isKeyPressed( sf::Keyboard::A ) )
      {
        movement_speed.x -= move_amount;
      }

      if( sf::Keyboard::isKeyPressed( sf::Keyboard::D ) )
      {
        movement_speed.x += move_amount;
      }

      if( sf::Keyboard::isKeyPressed( sf::Keyboard::W ) )
      {
        movement_speed.y += move_amount;
      }

      if( sf::Keyboard::isKeyPressed( sf::Keyboard::S ) )
      {
        movement_speed.y -= move_amount;
      }

      if( move_cam )
      {
        cam.move_forward( movement_speed.y * seconds );
        cam.move_right( movement_speed.x * seconds );
      }

      movement_speed *= 0.5;

      js::set_user_pos( b, cam.pos );

      movement_timer.restart();
    }

    //lights[1].light_cam.pos.x = sin( radians(timer.getElapsedTime().asMilliseconds() * 0.001f * 10.0f) ) * 30.0f;
    //lights[1].light_cam.pos.z = cos( radians(timer.getElapsedTime().asMilliseconds() * 0.001f * 10.0f) ) * 17.0f;

    //-----------------------------
    //set up matrices
    //-----------------------------

    mvm.push_matrix( cam );
    mat4 mv = ppl.get_model_view_matrix();
    mat4 inv_mv = inverse( mv );
    mat4 inv_view = inverse( cam.get_camera_matrix( true ) );
    mat3 normal_mat = ppl.get_normal_matrix();
    vec4 vs_eye_pos = mv * vec4( cam.pos, 1 );
    mvm.pop_matrix();

    frustum f;
    f.set_up( cam );

    //cull objects
    culled_objs = o->get_culled_objects(&f);

    //-----------------------------
    //gbuffer rendering
    //-----------------------------

    glViewport( 0, 0, screen.x, screen.y );

    glEnable( GL_DEPTH_TEST );

    glBindFramebuffer( GL_FRAMEBUFFER, gbuffer_fbo );
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

    glUseProgram( gbuffer_shader );

    mvm.push_matrix( cam.get_camera_matrix( false ) );
    glUniformMatrix4fv( glGetUniformLocation( gbuffer_shader, "mvp" ), 1, false, &ppl.get_model_view_projection_matrix( cam )[0][0] );
    glUniformMatrix3fv( glGetUniformLocation( gbuffer_shader, "normal_mat" ), 1, false, &ppl.get_normal_matrix()[0][0] );
    mvm.pop_matrix();

    for( auto& c : culled_objs )
    {
      if( f.intersects( scene[c].bv )/* && c != 66*/ )
      {
        scene[c].render();
        ++draw_calls;
      }
    }

    glDisable( GL_DEPTH_TEST );

    glBindFramebuffer( GL_FRAMEBUFFER, 0 );

    //-----------------------------
    //render the shadows
    //-----------------------------

    //cull lights
    culled_objs.clear();
    for( int c = 0; c < lights.size(); ++c )
    {
      if( f.intersects( lights[c].bv ) )
      {
        culled_objs.push_back(c);

        //ddman.CreateSphere( lights_data[c].ms_position.xyz, lights_data[c].attenuation_end, 0 );
      }
    }

    num_layers_used = set_up_layers();

    int counter = 0;
    for( auto& c : culled_objs )
    {
      lights_data[counter] = create_light( mv, lights[c] );

      diffuse_color_data[counter].xyz = lights_data[counter].diffuse_color.xyz;
      specular_color_data[counter].xyz = lights_data[counter].specular_color.xyz;
      light_size_data[counter] = lights_data[counter].specular_color.w;
      vs_position_data[counter] = lights_data[counter].vs_position;
      ms_position_data[counter] = lights_data[counter].ms_position;
      attenuation_end_data[counter] = lights_data[counter].attenuation_end;
      attenuation_cutoff_data[counter] = lights_data[counter].attenuation_cutoff;
      radius_data[counter] = lights_data[counter].radius;
      spot_exponent_data[counter] = lights_data[counter].spot_exponent;
      lighting_type_data[counter] = lights_data[counter].lighting_type;
      attenuation_type_data[counter] = lights_data[counter].attenuation_type;
      layer_data[counter] = lights_data[counter].layer;
      spot_direction_data[counter].xyz = lights_data[counter].spot_direction.xyz;
      spot_cutoff_data[counter] = lights_data[counter].spot_direction.w;
      spot_shadow_mat_data[counter] = lights_data[counter].spot_shadow_mat;
      memcpy( &point_shadow_mat_data[counter*6][0], &lights_data[counter].point_shadow_mats[0], 6 * sizeof(mat4) );

      //ddman.CreateSphere( lights_data[counter].ms_position.xyz, lights_data[counter].attenuation_end, 0 );

      counter++;
    }

    if( active_technique != techLIGHTING )
    {
      /**/
      glEnable( GL_DEPTH_TEST );
      glDepthMask( GL_TRUE );
      glColorMask( GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE );
      glViewport( 0, 0, shadow_texture_size.x, shadow_texture_size.y );

      glUseProgram( spot_shadow_gen_shader );

      for( auto& c : culled_objs )
      {
        if( lights[c].type == light_type::lightSPOT )
        {
          glBindFramebuffer( GL_FRAMEBUFFER, spot_shadow_fbos[c] );
          glClear( GL_DEPTH_BUFFER_BIT );
          static vector<unsigned> culled_shadow_objs;
          culled_shadow_objs.clear();
          culled_shadow_objs = o->get_culled_objects(lights[c].bv);

          mat4 light_mvp = lights[c].light_cam.get_frame()->projection_matrix * lights[c].light_cam.get_camera_matrix( false );
          lights[c].spot_shadow_mat = bias_matrix * light_mvp * inv_mv;

          glUniformMatrix4fv( glGetUniformLocation( spot_shadow_gen_shader, "mvp" ), 1, false, &light_mvp[0][0] );

          //for( auto& e : culled_shadow_objs )
          for( int e = 0; e < scene.size(); ++e )
          {
            //if( lights[c].bv->intersects( scene[e].bv ) )
            {
              scene[e].render();
              ++draw_calls;
            }
          }
        }
      }

      //point lights
      glUseProgram( spot_shadow_gen_shader );

      for( auto& c : culled_objs )
      {
        if( lights[c].type == light_type::lightPOINT )
        {
          static vector<unsigned> culled_shadow_objs;
          culled_shadow_objs.clear();
          culled_shadow_objs = o->get_culled_objects(lights[c].bv);

          mat4 light_model_mat = create_translation( -lights[c].light_cam.pos );

          for( int d = 0; d < 6; ++d )
          {
            //if( lights[c].point_shadow_frustums[d].intersects( &f ) )
            {
              //culled_shadow_objs.clear(); 
              //culled_shadow_objs = o->get_culled_objects(&lights[c].point_shadow_frustums[d]);

              mat4 light_mvp = lights[c].point_shadow_mats[d] * light_model_mat;
              glUniformMatrix4fv( glGetUniformLocation( spot_shadow_gen_shader, "mvp" ), 1, false, &light_mvp[0][0] );

              glBindFramebuffer( GL_FRAMEBUFFER, point_shadow_fbos[c * 6 + d] );
              glClear( GL_DEPTH_BUFFER_BIT );

              //if( c == 13 )
              //  continue;

              for( auto& e : culled_shadow_objs )
              {
                if( lights[c].bv->intersects( scene[e].bv ) )
                {
                  scene[e].render();
                  //ddman.CreateAABoxMinMax( static_cast<aabb*>(scene[e].bv)->min, static_cast<aabb*>(scene[e].bv)->max, 0 );
                  ++draw_calls;
                }
              }
            }
          }
        }
      }

      //point lights w/ translucent shadow
      /*glUseProgram( spot_shadow_translucent_gen_shader );

      glColorMask( GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE );

      for( auto& c : culled_objs )
      {
        if( lights[c].type == light_type::lightPOINT && c == 13 )
        {
          static vector<unsigned> culled_shadow_objs;
          culled_shadow_objs.clear();
          culled_shadow_objs = o->get_culled_objects(lights[c].bv);

          //for( int e = 0; e < scene.size(); ++e )
          //{
          // if( lights[c].bv->intersects( scene[e].bv ) )
          //    culled_shadow_objs.push_back(e);
          //}

          mat4 light_model_mat = create_translation( -lights[c].light_cam.pos );

          for( int d = 0; d < 6; ++d )
          {
            mat4 light_mvp = lights[c].point_shadow_mats[d] * light_model_mat;
            glUniformMatrix4fv( glGetUniformLocation( spot_shadow_translucent_gen_shader, "mvp" ), 1, false, &light_mvp[0][0] );
          
            //culled_shadow_objs.clear();
            //culled_shadow_objs = o->get_culled_objects(&light_frustums[c * 6 + d]);

            glBindFramebuffer( GL_FRAMEBUFFER, point_shadow_fbos[c * 6 + d] );
            glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

            for( int e = 0; e < scene.size(); ++e )
            {
              //if( light_frustums[c * 6 + d].intersects( scene[e].bv ) )
              if( lights[c].bv->intersects( scene[e].bv ) )
              {
                if( e == 66 )
                  glUniform4f( glGetUniformLocation( spot_shadow_translucent_gen_shader, "incolor" ), 1, 0, 0, 0.5 );
                else
                  glUniform4f( glGetUniformLocation( spot_shadow_translucent_gen_shader, "incolor" ), 0, 0, 0, 0 );

                scene[e].render();
                ++draw_calls;
              }
            }
          }
        }
      }*/

      glViewport( 0, 0, screen.x, screen.y );
      glColorMask( GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE );
      glDisable( GL_DEPTH_TEST );
      /**/
    }

    //-----------------------------
    //calculate minfilter data
    //-----------------------------

    if( active_technique == techSSSSMINFILTER ) 
    {
      /**/
      glUseProgram( minfilter_point_shader );

      glUniformMatrix4fv( glGetUniformLocation( minfilter_point_shader, "mvp" ), 1, false, &ppl.get_model_view_projection_matrix( cam )[0][0] );

      glActiveTexture( GL_TEXTURE0 );
      glBindVertexArray( quad );

      //glViewport( 0, 0, shadow_texture_size.x, shadow_texture_size.y );
      glViewport( 0, 0, 64, 64 );
      glUniform2f( glGetUniformLocation( minfilter_point_shader, "dir" ), 0, 1 );
      glActiveTexture( GL_TEXTURE0 );

      glUniform1fv( glGetUniformLocation( minfilter_point_shader, "light_size_data" ), culled_objs.size(), (float*)&light_size_data[0] );

      for( auto& c : culled_objs )
      {
        if( lights[c].type == light_type::lightPOINT )
        {
          glUniform1i( glGetUniformLocation( minfilter_point_shader, "light_index" ), c );

          for( int d = 0; d < 6; ++d )
          {
            glUniform1i( glGetUniformLocation( minfilter_point_shader, "cubemap_face" ), d );

            glBindTexture( GL_TEXTURE_2D, point_shadow_texture_views[c*6+d] );
            glBindFramebuffer( GL_FRAMEBUFFER, point_shadow_minfilter_fbos0[c * 6 + d] );
            glDrawElements( GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0 );
            ++draw_calls; 
          }
        }
      }

      glViewport( 0, 0, 64, 64 );
      glUniform2f( glGetUniformLocation( minfilter_point_shader, "dir" ), 1, 0 );

      for( auto& c : culled_objs )
      {
        if( lights[c].type == light_type::lightPOINT )
        {
          glUniform1i( glGetUniformLocation( minfilter_point_shader, "light_index" ), c );

          for( int d = 0; d < 6; ++d )
          {
            glUniform1i( glGetUniformLocation( minfilter_point_shader, "cubemap_face" ), d );
    
            glBindTexture( GL_TEXTURE_2D, point_shadow_minfilter_texture0_views[c*6+d] );
            glBindFramebuffer( GL_FRAMEBUFFER, point_shadow_minfilter_fbos1[c * 6 + d] );
            glUniform2f( glGetUniformLocation( minfilter_point_shader, "dir" ), 1, 0 );
            glDrawElements( GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0 );
            ++draw_calls;
          }
        }
      }

      glViewport( 0, 0, screen.x, screen.y );

      /**/
    }

    //-----------------------------
    //calculate layered penumbra data w/ minfilter
    //-----------------------------

    if( !penumbra_half_res )
      glViewport( 0, 0, screen.x, screen.y );
    else
      glViewport( 0, 0, screen.x / 2, screen.y / 2 );

    if( active_technique == techSSSSMINFILTER )
    {
      /**/
      glBindFramebuffer( GL_FRAMEBUFFER, penumbra_fbo );
      //glBindFramebuffer( GL_FRAMEBUFFER, 0 );

      GLuint the_shader = penumbra_minfilter_shader;
      
      if( exponential_shadows )
        the_shader = penumbra_minfilter_exponential_shader;

      glUseProgram( the_shader );

      glUniformMatrix4fv( glGetUniformLocation( the_shader, "mvp" ), 1, false, &ppl.get_model_view_projection_matrix( cam )[0][0] );
      glUniformMatrix4fv( glGetUniformLocation( the_shader, "mv" ), 1, false, &ppl.get_model_view_matrix()[0][0] );
      glUniform2f( glGetUniformLocation( the_shader, "nearfar" ), cam_near, cam_far );
      glUniformMatrix4fv( glGetUniformLocation( the_shader, "inv_view" ), 1, false, &inv_view[0][0] );
      glUniformMatrix4fv( glGetUniformLocation( the_shader, "inv_mv" ), 1, false, &inv_mv[0][0] );
      glUniform1i( glGetUniformLocation( the_shader, "num_lights" ), culled_objs.size() );
    
      glUniform1fv( glGetUniformLocation( the_shader, "light_size_data" ), culled_objs.size(), (float*)&light_size_data[0] );
      glUniform4fv( glGetUniformLocation( the_shader, "vs_position_data" ), culled_objs.size(), (float*)&vs_position_data[0] );
      glUniform4fv( glGetUniformLocation( the_shader, "ms_position_data" ), culled_objs.size(), (float*)&ms_position_data[0] );
      glUniform1fv( glGetUniformLocation( the_shader, "radius_data" ), culled_objs.size(), (float*)&radius_data[0] );
      glUniform1iv( glGetUniformLocation( the_shader, "lighting_type_data" ), culled_objs.size(), (int*)&lighting_type_data[0] );
      glUniform1iv( glGetUniformLocation( the_shader, "layer_data" ), culled_objs.size(), (int*)&layer_data[0] );
      glUniformMatrix4fv( glGetUniformLocation( the_shader, "spot_shadow_mat_data" ), culled_objs.size(), false, (float*)&spot_shadow_mat_data[0] );
      glUniformMatrix4fv( glGetUniformLocation( the_shader, "point_shadow_mat_data" ), culled_objs.size() * 6, false, (float*)&point_shadow_mat_data[0] );
      glUniform1uiv( glGetUniformLocation( the_shader, "light_indices" ), culled_objs.size(), (unsigned*)&culled_objs[0] );

      glActiveTexture( GL_TEXTURE0 );
      glBindTexture( GL_TEXTURE_2D, depth_texture );
      glActiveTexture( GL_TEXTURE1 );
      //glBindTexture( GL_TEXTURE_2D_ARRAY, spot_shadow_texture );
      glActiveTexture( GL_TEXTURE2 );
      glBindTexture( GL_TEXTURE_CUBE_MAP_ARRAY, point_shadow_minfilter_texture1 );
      glActiveTexture( GL_TEXTURE3 );
      glBindTexture( GL_TEXTURE_2D_ARRAY, spot_shadow_texture );
      glActiveTexture( GL_TEXTURE4 );
      glBindTexture( GL_TEXTURE_CUBE_MAP_ARRAY, point_shadow_texture );
      glActiveTexture( GL_TEXTURE0 );

      glBindVertexArray( quad );
      glDrawElements( GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0 );
      ++draw_calls;
      /**/
    }

    //-----------------------------
    //calculate layered penumbra data
    //-----------------------------

    if( active_technique == techSSSSBLOCKER )
    {
      /**/
      glBindFramebuffer( GL_FRAMEBUFFER, penumbra_fbo );
      //glBindFramebuffer( GL_FRAMEBUFFER, 0 );

      GLuint the_shader = penumbra_shader;

      if( exponential_shadows )
        the_shader = penumbra_shader_exponential;

      glUseProgram( the_shader );

      glUniformMatrix4fv( glGetUniformLocation( the_shader, "mvp" ), 1, false, &ppl.get_model_view_projection_matrix( cam )[0][0] );
      glUniformMatrix4fv( glGetUniformLocation( the_shader, "mv" ), 1, false, &ppl.get_model_view_matrix()[0][0] );
      glUniform2f( glGetUniformLocation( the_shader, "nearfar" ), cam_near, cam_far );
      glUniformMatrix4fv( glGetUniformLocation( the_shader, "inv_view" ), 1, false, &inv_view[0][0] );
      glUniformMatrix4fv( glGetUniformLocation( the_shader, "inv_mv" ), 1, false, &inv_mv[0][0] );
      glUniform1i( glGetUniformLocation( the_shader, "num_lights" ), culled_objs.size() );

      glUniform1fv( glGetUniformLocation( the_shader, "light_size_data" ), culled_objs.size(), (float*)&light_size_data[0] );
      glUniform4fv( glGetUniformLocation( the_shader, "vs_position_data" ), culled_objs.size(), (float*)&vs_position_data[0] );
      glUniform4fv( glGetUniformLocation( the_shader, "ms_position_data" ), culled_objs.size(), (float*)&ms_position_data[0] );
      glUniform1fv( glGetUniformLocation( the_shader, "radius_data" ), culled_objs.size(), (float*)&radius_data[0] );
      glUniform1iv( glGetUniformLocation( the_shader, "lighting_type_data" ), culled_objs.size(), (int*)&lighting_type_data[0] );
      glUniform1iv( glGetUniformLocation( the_shader, "layer_data" ), culled_objs.size(), (int*)&layer_data[0] );
      glUniformMatrix4fv( glGetUniformLocation( the_shader, "spot_shadow_mat_data" ), culled_objs.size(), false, (float*)&spot_shadow_mat_data[0] );
      glUniformMatrix4fv( glGetUniformLocation( the_shader, "point_shadow_mat_data" ), culled_objs.size() * 6, false, (float*)&point_shadow_mat_data[0] );
      glUniform1uiv( glGetUniformLocation( the_shader, "light_indices" ), culled_objs.size(), (unsigned*)&culled_objs[0] );
    
      glActiveTexture( GL_TEXTURE0 );
      glBindTexture( GL_TEXTURE_2D, depth_texture );
      glActiveTexture( GL_TEXTURE1 );
      glBindTexture( GL_TEXTURE_2D_ARRAY, spot_shadow_texture );
      glActiveTexture( GL_TEXTURE2 );
      glBindTexture( GL_TEXTURE_CUBE_MAP_ARRAY, point_shadow_texture );
      glActiveTexture( GL_TEXTURE3 );
      //glBindTexture( GL_TEXTURE_CUBE_MAP_ARRAY, point_shadow_translucent_texture );
      glActiveTexture( GL_TEXTURE0 );

      glBindVertexArray( quad );
      glDrawElements( GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0 );
      ++draw_calls;
      /**/
    }

    if( penumbra_half_res )
      glViewport( 0, 0, screen.x, screen.y );

    //-----------------------------
    //anisotropic gauss blurring
    //-----------------------------

    if( active_technique == techSSSSMINFILTER || active_technique == techSSSSBLOCKER )
    {
      /**/
      float err_depth = 0.03f;

      if( gauss_blur_half_res )
        glViewport( 0, 0, screen.x / 2, screen.y / 2 );
      else
        glViewport( 0, 0, screen.x, screen.y );

      //vertical pass
      glBindFramebuffer( GL_FRAMEBUFFER, gauss_fbo0 );
      //glBindFramebuffer( GL_FRAMEBUFFER, 0 );

      GLuint the_shader = gauss_shader0;

      if( exponential_shadows )
        if( gauss_supersampling )
          the_shader = gauss_shader0_exponential_supersampling;
        else
          the_shader = gauss_shader0_exponential;
      else
        the_shader = gauss_shader0_supersampling;

      glUseProgram( the_shader );

      glUniformMatrix4fv( glGetUniformLocation( the_shader, "mvp" ), 1, false, &ppl.get_model_view_projection_matrix( cam )[0][0] );
      glUniform1f( glGetUniformLocation( the_shader, "err_depth" ), err_depth );
      glUniform2f( glGetUniformLocation( the_shader, "nearfar" ), -cam_near, -cam_far );
      glUniformMatrix4fv( glGetUniformLocation( the_shader, "mv" ), 1, false, &ppl.get_model_view_matrix()[0][0] );
      glUniformMatrix4fv( glGetUniformLocation( the_shader, "inv_view" ), 1, false, &inv_view[0][0] );
      glUniformMatrix4fv( glGetUniformLocation( the_shader, "inv_mv" ), 1, false, &inv_mv[0][0] );

      glActiveTexture( GL_TEXTURE0 );
      glBindTexture( GL_TEXTURE_2D, normal_texture );
      glActiveTexture( GL_TEXTURE1 );
      glBindTexture( GL_TEXTURE_2D, depth_texture );
      glActiveTexture( GL_TEXTURE2 );
      glBindTexture( GL_TEXTURE_2D, layered_penumbra_texture );
      glActiveTexture( GL_TEXTURE3 );
      glBindTexture( GL_TEXTURE_2D, layered_shadow_texture );
      glActiveTexture( GL_TEXTURE4 );
      //glBindTexture( GL_TEXTURE_2D_ARRAY, spot_shadow_texture );
      glActiveTexture( GL_TEXTURE5 );
      //glBindTexture( GL_TEXTURE_CUBE_MAP_ARRAY, point_shadow_minfilter_texture1 );
      glActiveTexture( GL_TEXTURE6 );
      //glBindTexture( GL_TEXTURE_2D, layered_translucency_texture );
      glActiveTexture( GL_TEXTURE0 );

      glBindVertexArray( quad );
      glDrawElements( GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0 );
      ++draw_calls;

      //horizontal pass
      glBindFramebuffer( GL_FRAMEBUFFER, gauss_fbo1 );
      //glBindFramebuffer( GL_FRAMEBUFFER, 0 );

      the_shader = gauss_shader1;

      if( gauss_supersampling )
        the_shader = gauss_shader1_supersampling;

      glUseProgram( the_shader );

      glUniformMatrix4fv( glGetUniformLocation( the_shader, "mvp" ), 1, false, &ppl.get_model_view_projection_matrix( cam )[0][0] );
      glUniform1f( glGetUniformLocation( the_shader, "err_depth" ), err_depth );
      glUniform2f( glGetUniformLocation( the_shader, "nearfar" ), -cam_near, -cam_far );
      glUniformMatrix4fv( glGetUniformLocation( the_shader, "mv" ), 1, false, &ppl.get_model_view_matrix()[0][0] );
      glUniformMatrix4fv( glGetUniformLocation( the_shader, "inv_view" ), 1, false, &inv_view[0][0] );
      glUniformMatrix4fv( glGetUniformLocation( the_shader, "inv_mv" ), 1, false, &inv_mv[0][0] );

      glActiveTexture( GL_TEXTURE0 );
      glBindTexture( GL_TEXTURE_2D, normal_texture );
      glActiveTexture( GL_TEXTURE1 );
      glBindTexture( GL_TEXTURE_2D, depth_texture );
      glActiveTexture( GL_TEXTURE2 );
      glBindTexture( GL_TEXTURE_2D, layered_penumbra_texture );
      glActiveTexture( GL_TEXTURE3 );
      glBindTexture( GL_TEXTURE_2D, gauss_texture0 );
      glActiveTexture( GL_TEXTURE4 );
      //glBindTexture( GL_TEXTURE_2D_ARRAY, spot_shadow_texture );
      glActiveTexture( GL_TEXTURE5 );
      //glBindTexture( GL_TEXTURE_CUBE_MAP_ARRAY, point_shadow_minfilter_texture1 );
      glActiveTexture( GL_TEXTURE6 );
      //glBindTexture( GL_TEXTURE_2D, gauss_translucency_texture0 );
      glActiveTexture( GL_TEXTURE0 );

      glBindVertexArray( quad );
      glDrawElements( GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0 );
      ++draw_calls;
      /**/
    }

    if( gauss_blur_half_res )
      glViewport( 0, 0, screen.x, screen.y );

    //-----------------------------
    //render the lights
    //-----------------------------

    /**/
    GLuint the_shader = lighting_shader;

    switch( active_technique )
    {
      case techHARD_SHADOWS:
      {
        if( exponential_shadows )
          the_shader = lighting_shader_hard_shadow_exponential;
        else
          the_shader = lighting_shader_hard_shadow;

        break;
      }
      case techPCF:
      {
        if( exponential_shadows )
          the_shader = lighting_shader_pcf_exponential;
        else
          the_shader = lighting_shader_pcf;

        break;
      }
      case techPCSS:
      {
        if( exponential_shadows )
          the_shader = lighting_shader_pcss_exponential;
        else
          the_shader = lighting_shader_pcss;

        break;
      }
      case techSSSSBLOCKER:
      case techSSSSMINFILTER:
      {
        if( exponential_shadows )
          the_shader = lighting_shader_ssss_exponential;
        else
          the_shader = lighting_shader_ssss;

        break;
      }
    }

    glUseProgram( the_shader );

    glBindImageTexture( 0, result_texture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8 );
    //glBindImageTexture( 1, local_light_result_texture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F );

    glActiveTexture( GL_TEXTURE0 );
    glBindTexture( GL_TEXTURE_2D, normal_texture );
    glActiveTexture( GL_TEXTURE1 );
    glBindTexture( GL_TEXTURE_2D, depth_texture );
    glActiveTexture( GL_TEXTURE2 );
    glBindTexture( GL_TEXTURE_2D_ARRAY, spot_shadow_texture );
    glActiveTexture( GL_TEXTURE3 );
    glBindTexture( GL_TEXTURE_CUBE_MAP_ARRAY, point_shadow_texture );
    glActiveTexture( GL_TEXTURE4 );
    glBindTexture( GL_TEXTURE_2D, gauss_texture1 );
    glActiveTexture( GL_TEXTURE5 );
    //glBindTexture( GL_TEXTURE_2D, gauss_translucency_texture1 );
    glActiveTexture( GL_TEXTURE0 );

    glUniform2f( glGetUniformLocation( the_shader, "nearfar" ), -cam_near, -cam_far );
    glUniform1i( glGetUniformLocation( the_shader, "num_lights" ), culled_objs.size() );
    vec4 tmp_far_plane0 = vec4( the_frame.far_ll.xyz, the_frame.far_ur.x );
    vec2 tmp_far_plane1 = vec2( the_frame.far_ur.yz );
    glUniform4fv( glGetUniformLocation( the_shader, "far_plane0" ), 1, &tmp_far_plane0.x );
    glUniform2fv( glGetUniformLocation( the_shader, "far_plane1" ), 1, &tmp_far_plane1.x );
    glUniformMatrix4fv( glGetUniformLocation( the_shader, "proj_mat" ), 1, false, &the_frame.projection_matrix[0][0] );
    glUniformMatrix4fv( glGetUniformLocation( the_shader, "inv_view" ), 1, false, &inv_view[0][0] );
    glUniformMatrix4fv( glGetUniformLocation( the_shader, "inv_mv" ), 1, false, &inv_mv[0][0] );

    glUniform4fv( glGetUniformLocation( the_shader, "diffuse_color_data" ), culled_objs.size(), (float*)&diffuse_color_data[0] );
    glUniform4fv( glGetUniformLocation( the_shader, "specular_color_data" ), culled_objs.size(), (float*)&specular_color_data[0] );
    glUniform1fv( glGetUniformLocation( the_shader, "light_size_data" ), culled_objs.size(), (float*)&light_size_data[0] );
    glUniform4fv( glGetUniformLocation( the_shader, "vs_position_data" ), culled_objs.size(), (float*)&vs_position_data[0] );
    glUniform4fv( glGetUniformLocation( the_shader, "ms_position_data" ), culled_objs.size(), (float*)&ms_position_data[0] );
    glUniform1fv( glGetUniformLocation( the_shader, "attenuation_end_data" ), culled_objs.size(), (float*)&attenuation_end_data[0] );
    glUniform1fv( glGetUniformLocation( the_shader, "attenuation_cutoff_data" ), culled_objs.size(), (float*)&attenuation_cutoff_data[0] );
    glUniform1fv( glGetUniformLocation( the_shader, "radius_data" ), culled_objs.size(), (float*)&radius_data[0] );
    glUniform1fv( glGetUniformLocation( the_shader, "spot_exponent_data" ), culled_objs.size(), (float*)&spot_exponent_data[0] );
    glUniform1iv( glGetUniformLocation( the_shader, "attenuation_type_data" ), culled_objs.size(), (int*)&attenuation_type_data[0] );
    glUniform1iv( glGetUniformLocation( the_shader, "lighting_type_data" ), culled_objs.size(), (int*)&lighting_type_data[0] );
    glUniform1iv( glGetUniformLocation( the_shader, "layer_data" ), culled_objs.size(), (int*)&layer_data[0] );
    glUniform4fv( glGetUniformLocation( the_shader, "spot_direction_data" ), culled_objs.size(), (float*)&spot_direction_data[0] );
    glUniform1fv( glGetUniformLocation( the_shader, "spot_cutoff_data" ), culled_objs.size(), (float*)&spot_cutoff_data[0] );
    glUniformMatrix4fv( glGetUniformLocation( the_shader, "spot_shadow_mat_data" ), culled_objs.size(), false, (float*)&spot_shadow_mat_data[0] );
    glUniformMatrix4fv( glGetUniformLocation( the_shader, "point_shadow_mat_data" ), culled_objs.size() * 6, false, (float*)&point_shadow_mat_data[0] );
    glUniform1uiv( glGetUniformLocation( the_shader, "light_indices" ), culled_objs.size(), (unsigned*)&culled_objs[0] );

    glDispatchCompute( dispatch_size.x, dispatch_size.y, 1 );

    glMemoryBarrier( GL_SHADER_IMAGE_ACCESS_BARRIER_BIT );
    /**/

    //--------------------------------------
    //render translucent objects with lights
    //--------------------------------------

    //vector<int> texdata;
    //texdata.resize(1024 * dispatch_size.x * dispatch_size.y);
    //glBindTexture( GL_TEXTURE_1D, local_light_result_texture );
    //glGetTexImage( GL_TEXTURE_1D, 0, GL_R32F, GL_FLOAT, &texdata[0] );

    /**
    glEnable( GL_BLEND );
    glEnable( GL_DEPTH_TEST );
    glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );

    glBindFramebuffer( GL_FRAMEBUFFER, translucent_fbo );

    glUseProgram( translucent_lighting_shader );

    mvm.push_matrix( cam.get_camera_matrix( false ) );
    glUniformMatrix4fv( glGetUniformLocation( translucent_lighting_shader, "mvp" ), 1, false, &ppl.get_model_view_projection_matrix( cam )[0][0] );
    glUniformMatrix3fv( glGetUniformLocation( translucent_lighting_shader, "normal_mat" ), 1, false, &ppl.get_normal_matrix()[0][0] );
    mvm.pop_matrix();

    scene[66].render();

    glDisable( GL_BLEND );
    glDisable( GL_DEPTH_TEST );
    /**/

    //--------------------------------------
    //render the results
    //--------------------------------------

    //display the results
    /**/
    glDisable( GL_DEPTH_TEST );
    glBindFramebuffer( GL_FRAMEBUFFER, 0 );

    glViewport( 0, 0, screen.x, screen.y );

    glUseProgram( display_shader );

    glUniformMatrix4fv( glGetUniformLocation( display_shader, "mvp" ), 1, false, &ppl.get_model_view_projection_matrix( cam )[0][0] );

    glActiveTexture( GL_TEXTURE0 );
    glBindTexture( GL_TEXTURE_2D, result_texture );
    //glBindTexture( GL_TEXTURE_CUBE_MAP_ARRAY, point_shadow_translucent_texture );

    glBindVertexArray( quad );
    glDrawElements( GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0 );
    ++draw_calls;
    /**/

    //display debug cubes
    /**/
    glUseProgram( cubemap_debug_shader );

    glActiveTexture( GL_TEXTURE0 );
    glBindTexture( GL_TEXTURE_CUBE_MAP_ARRAY, point_shadow_texture );
    //glBindTexture( GL_TEXTURE_CUBE_MAP_ARRAY, point_shadow_minfilter_texture1 );
    //glBindTexture( GL_TEXTURE_CUBE_MAP_ARRAY, point_shadow_minfilter_texture0 );

    mvm.push_matrix( cam );

    if( do_translate )
    {
      vec2 delta = mouse_pos - 0.5;
      float cam_aspect = float(screen.x) / screen.y;
      float top = length( lights[active_light].light_cam.pos - cam.pos ) * std::tan( cam_fov * 0.5f );
      float right = top * cam_aspect;
      vec3 right_vec = normalize( cross( cam.view_dir, cam.up_vector ) );
      vec3 up_vec = normalize( cam.up_vector );

      lights[active_light].light_cam.pos += delta.x * right_vec * 2 * top;
      lights[active_light].light_cam.pos += delta.y * up_vec * 2 * right;

      //warp
      frm.set_mouse_pos( ivec2( screen.x / 2.0f, screen.y / 2.0f ) );
    }

    if( clicked )
    {
      vec2 mouse_pos_ndc = mouse_pos * 2 - 1;

      aabb light_obj_space_aabb(0, 1);

      for( int i = 0; i < lights.size(); ++i )
      {
        mvm.push_matrix();
        mvm.translate_matrix( lights[i].light_cam.pos.xyz );
        mat4 mvp = ppl.get_model_view_projection_matrix( cam );
        mvm.pop_matrix();

        mat4 inv_mvp = inverse( mvp );

        vec3 mouse_ray_start = unproject( vec3( mouse_pos_ndc, 0 ), inv_mvp );
        vec3 mouse_ray_end = unproject( vec3( mouse_pos_ndc, 1 ), inv_mvp );
        vec3 ori = mouse_ray_start;
        vec3 dir = normalize( mouse_ray_end - mouse_ray_start );

        ray obj_space_ray = ray( ori, dir );

        if( light_obj_space_aabb.intersects( &obj_space_ray ) )
        {
          active_light = i;

          js::set_light_pos( b, lights[active_light].light_cam.pos );
          js::set_light_view( b, lights[active_light].light_cam.view_dir );
          js::set_light_up( b, lights[active_light].light_cam.up_vector );
          js::set_light_color( b, lights[active_light].diffuse_color.xyz );
          js::set_light_radius( b, lights[active_light].radius );
          js::set_light_size( b, lights[active_light].light_size );

          break;
        }
      }
    }

    for( int i = 0; i < lights.size(); ++i )
    {
      if( lights[i].type == light_type::lightPOINT )
      {
        mvm.push_matrix();
        mvm.translate_matrix( lights[i].light_cam.pos.xyz );
        glUniformMatrix4fv( glGetUniformLocation( cubemap_debug_shader, "mvp" ), 1, false, &ppl.get_model_view_projection_matrix( cam )[0][0] );
        mvm.pop_matrix();

        glUniform1i( glGetUniformLocation( cubemap_debug_shader, "layer" ), i );
        
        if( i != active_light )
          glUniform3f( glGetUniformLocation( cubemap_debug_shader, "tint" ), 1, 1, 1 );
        else
        {
          glUniform3f( glGetUniformLocation( cubemap_debug_shader, "tint" ), 0, 1, 0 );
        }

        glBindVertexArray( the_box );
        glDrawElements( GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0 );
        ++draw_calls;
      }
    }

    mvm.pop_matrix();
    /**/

    //-----------------------------
    //render the debug objects
    //-----------------------------

    glUseProgram( debug_shader );

    mvm.push_matrix( cam );
    glUniformMatrix4fv( glGetUniformLocation( debug_shader, "mvp" ), 1, false, &ppl.get_model_view_projection_matrix( cam )[0][0] );
    mvm.pop_matrix();

    ddman.DrawAndUpdate( 16 );

    //-----------------------------
    //render the UI
    //-----------------------------

    /**/

    //NOTE you decide how you'd like to render the browsers
    glDisable( GL_DEPTH_TEST );
    glEnable( GL_BLEND );
    glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
    glUseProgram( browser_shader );

    glActiveTexture( GL_TEXTURE0 );
    glBindTexture( GL_TEXTURE_2D, b.browser_texture );
    
    glBindVertexArray( ss_quad );
    glDrawElements( GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0 );

    glDisable( GL_BLEND );
    glEnable( GL_DEPTH_TEST );

    /**/

    //frm.get_opengl_error();
  }, silent );

  browser::get().destroy( b );
  browser::get().shutdown();

  return 0;
}
