#ifndef framework_h
#define framework_h

#include <GL/glew.h>
#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>
#include <assimp/scene.h>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <mymath/mymath.h>

#ifdef USE_CL

#include <SFML/OpenGL.hpp>
#include <CL/cl.h>
#include <CL/cl_gl.h>

#if defined (__APPLE__) || defined(MACOSX)
#define GL_SHARING_EXTENSION "cl_APPLE_gl_sharing"
#else
#define GL_SHARING_EXTENSION "cl_khr_gl_sharing"
#endif

#endif

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <streambuf>
#include <list>
#include <vector>
#include <map>

#ifdef _WIN32
#include <Windows.h>
#endif
#undef NEAR

using namespace mymath;
using namespace std;

#define MAX_SOURCE_SIZE (0x100000)
#define INFOLOG_SIZE 4096
#define GET_INFOLOG_SIZE INFOLOG_SIZE - 1
#define STRINGIFY(s) #s

#include "intersection.h"

class framework
{
    sf::Window the_window;
    sf::Event the_event;
		
#ifdef USE_CL
    cl_device_id device;
    cl_uint uiNumDevsUsed;
#endif

    bool run;

    static void shader_include( std::string& text, const std::string& path )
    {
      size_t start_pos = 0;
      std::string include_dir = "#include ";
  
      while( ( start_pos = text.find( include_dir, start_pos ) ) != std::string::npos )
      {
        int pos = start_pos + include_dir.length() + 1;
        int length = text.find( "\"", pos );
        std::string filename = text.substr( pos, length - pos );
        std::string content = "";
  
        std::ifstream f;
        f.open( (path + filename).c_str() );
  
        if( f.is_open() )
        {
          content = std::string( ( std::istreambuf_iterator<char>( f ) ),
                                 std::istreambuf_iterator<char>() );
        }
        else
        {
          cerr << "Couldn't include shader file: " << filename << endl;
          return;
        }
  
        text.replace( start_pos, ( length + 1 ) - start_pos, content );
        start_pos += content.length();
      }
    }


    void compile_shader( const char* text, const GLuint& program, const GLenum& type, const std::string& additional_str ) const
    {
      GLchar infolog[INFOLOG_SIZE];

      GLuint id = glCreateShader( type );
      std::string str = text;
      str = additional_str + str;
      const char* c = str.c_str();
      glShaderSource( id, 1, &c, 0 );
      glCompileShader( id );

      GLint success;
      glGetShaderiv( id, GL_COMPILE_STATUS, &success );

      if( !success )
      {
        glGetShaderInfoLog( id, GET_INFOLOG_SIZE, 0, infolog );
        cerr << infolog << endl;
      }
      else
      {
        glAttachShader( program, id );
        glDeleteShader( id );
      }
    }

    void link_shader( const GLuint& shader_program ) const
    {
      glLinkProgram( shader_program );

      GLint success;
      glGetProgramiv( shader_program, GL_LINK_STATUS, &success );

      if( !success )
      {
        GLchar infolog[INFOLOG_SIZE];
        glGetProgramInfoLog( shader_program, GET_INFOLOG_SIZE, 0, infolog );
        cout << infolog << endl;
      }

      glValidateProgram( shader_program );

      glGetProgramiv( shader_program, GL_VALIDATE_STATUS, &success );

      if( !success )
      {
        GLchar infolog[INFOLOG_SIZE];
        glGetProgramInfoLog( shader_program, GET_INFOLOG_SIZE, 0, infolog );
        cout << infolog << endl;
      }
    }

#ifdef USE_CL
	int is_extension_supported( const char* support_str, const char* ext_string, size_t ext_buffer_size)
	{
		size_t offset = 0;
		const char* space_substr = strstr(ext_string + offset, " "/*, ext_buffer_size - offset*/);
		size_t space_pos = space_substr ? space_substr - ext_string : 0;
		while (space_pos < ext_buffer_size)
		{
			if( strncmp(support_str, ext_string + offset, space_pos) == 0 ) 
			{
				// Device supports requested extension!
				cout << "Info: Found extension support " << support_str << endl;
				return 1;
			}
			// Keep searching -- skip to next token string
			offset = space_pos + 1;
			space_substr = strstr(ext_string + offset, " "/*, ext_buffer_size - offset*/);
			space_pos = space_substr ? space_substr - ext_string : 0;
		}
		cerr << "Warning: Extension not supported " << support_str << endl;
		return 0;
	}
#endif

#ifdef _WIN32
  char* realpath( const char* path, char** ret )
  {
    char* the_ret = 0;

    if( !ret )
    {
      the_ret = new char[MAX_PATH];
    }
    else
    {
      if( !*ret )
      {
        *ret = new char[MAX_PATH];
      }
      else
      {
        unsigned long s = strlen( *ret );

        if( s < MAX_PATH )
        {
          delete [] *ret;
          *ret = new char[MAX_PATH];

        }

        the_ret = *ret;
      }
    }

    unsigned long size = GetFullPathNameA( path, MAX_PATH, the_ret, 0 );

    if( size > MAX_PATH )
    {
      //too long path
      cerr << "Path too long, truncated." << endl;
      delete [] the_ret;
      return "";
    }

    if( ret )
    {
      *ret = the_ret;
    }

    return the_ret;
  }
#endif

  public:

#ifdef USE_CL
    cl_command_queue cl_cq;
    cl_context cl_GPU_context;
    cl_int cl_err_num;
#endif

    void set_mouse_visibility( bool vis )
    {
      the_window.setMouseCursorVisible( vis );
    }

    uvec2 get_window_size()
    {
      return uvec2( the_window.getSize().x, the_window.getSize().y );
    }

    ivec2 get_window_pos()
    {
      return ivec2( the_window.getPosition().x, the_window.getPosition().y );
    }

    void show_cursor( bool show )
    {
      the_window.setMouseCursorVisible( show );
    }

    void* get_window_handle()
    {
      return the_window.getSystemHandle();
    }

    void resize( int x, int y, unsigned w, unsigned h )
    {
      the_window.setPosition( sf::Vector2<int>( x, y ) );
      the_window.setSize( sf::Vector2<unsigned>( w, h ) );
    }

    void set_title( const string& str )
    {
      the_window.setTitle( str );
    }

    void set_mouse_pos( ivec2 xy )
    {
      sf::Mouse::setPosition( sf::Vector2i( xy.x, xy.y ), the_window );
    }

    void init( const uvec2& screen = uvec2( 1280, 720 ), const string& title = "", const bool& fullscreen = false )
    {
#ifdef REDIRECT
      //Redirect the STD output
      FILE* file_stream = 0;

      file_stream = freopen( "stdout.txt", "w", stdout );

      if( !file_stream )
      {
        cerr << "Error rerouting the standard output (std::cout)!" << endl;
      }

      file_stream = freopen( "stderr.txt", "w", stderr );

      if( !file_stream )
      {
        cerr << "Error rerouting the standard output (std::cerr)!" << endl;
      }

#endif

      run = true;

      srand( time( 0 ) );

      the_window.create( sf::VideoMode( screen.x > 0 ? screen.x : 1280, screen.y > 0 ? screen.y : 720, 32 ), title, fullscreen ? sf::Style::Fullscreen : sf::Style::Default );	  

      if( !the_window.isOpen() )
      {
        cerr << "Couldn't initialize SFML." << endl;
        the_window.close();
        return;
      }

      the_window.setPosition( sf::Vector2i( 0, 0 ) );

      GLenum glew_error = glewInit();

      glGetError(); //ignore glew errors

      cout << "Vendor: " << glGetString( GL_VENDOR ) << endl;
      cout << "Renderer: " << glGetString( GL_RENDERER ) << endl;
      cout << "OpenGL version: " << glGetString( GL_VERSION ) << endl;
      cout << "GLSL version: " << glGetString( GL_SHADING_LANGUAGE_VERSION ) << endl;

      if( glew_error != GLEW_OK )
      {
        cerr << "Error initializing GLEW: " << glewGetErrorString( glew_error ) << endl;
        the_window.close();
        return;
      }

      if( !GLEW_VERSION_4_3 )
      {
        cerr << "Error: " << STRINGIFY( GLEW_VERSION_4_3 ) << " is required" << endl;
        the_window.close();
        exit(0);
      }

      glEnable( GL_DEBUG_OUTPUT );
    }

#ifdef USE_CL
	void initCL()
	{
		cl_platform_id cp_platform;

		cl_uint num_platforms = 0;
		cl_err_num = clGetPlatformIDs (0, NULL, &num_platforms);
		if (cl_err_num != CL_SUCCESS)
		{
			cout << " Error in clGetPlatformIDs call!" << endl;
			get_opencl_error(cl_err_num);
			return;
		}
		if(num_platforms == 0)
		{
			cout << "No OpenCL platform found!" << endl;
			return;
		}

		cl_platform_id *cl_platforms = new cl_platform_id[num_platforms];

		// get platform info for each platform
		cl_err_num = clGetPlatformIDs (num_platforms, cl_platforms, NULL);
		char ch_buffer[1024];
		bool found_dev = false;
		for(cl_uint i = 0; i < num_platforms; ++i)
		{
			cl_err_num = clGetPlatformInfo (cl_platforms[i], CL_PLATFORM_PROFILE, 1024, &ch_buffer, NULL);
			if(cl_err_num == CL_SUCCESS )
			{
				if(strstr(ch_buffer, "FULL_PROFILE") != NULL)
				{
					cp_platform = cl_platforms[i];
					cl_err_num = clGetPlatformInfo (cl_platforms[i], CL_PLATFORM_NAME, 1024, &ch_buffer, NULL);

					// Get the number of GPU devices available to the platform
					cl_uint dev_count = 0;
					cl_err_num = clGetDeviceIDs(cp_platform, CL_DEVICE_TYPE_GPU, 0, NULL, &dev_count);
					if (cl_err_num == CL_SUCCESS)
					{
						if(dev_count > 0)
						{
							// Create the device list
							cl_device_id *cl_devices = new cl_device_id[dev_count];
							cl_err_num = clGetDeviceIDs(cp_platform, CL_DEVICE_TYPE_GPU, dev_count, cl_devices, NULL);
							for (int j=0; j<dev_count; j++)
							{
								// Get string containing supported device extensions
								size_t ext_size = 1024;
								char ext_string[1024];
								cl_err_num = clGetDeviceInfo(cl_devices[j], CL_DEVICE_EXTENSIONS, ext_size, ext_string, &ext_size);
								// Search for GL support in extension string (space delimited)
								int supported = is_extension_supported(GL_SHARING_EXTENSION, ext_string, ext_size);
								if( cl_err_num == CL_SUCCESS && supported ) 
								{
									// Device supports context sharing with OpenGL
									cl_err_num = clGetDeviceInfo(cl_devices[j], CL_DEVICE_NAME, ext_size, ext_string, &ext_size);
									cout << "Found OpenCL platform: " << ch_buffer << endl;
									cout << "Found GL Sharing Support: " <<  ext_string << endl;
									device = cl_devices[j];
									found_dev = true;
									break;
								}
							}
							delete[] cl_devices;		
						}
					}

					if(found_dev == true)							
						break;
				}
			}
		}
		delete[] cl_platforms;

		if(found_dev != true)	
		{			
			cout << "Not found GL Sharing Support!"  << endl;
			return;
		}
		// Create CL context properties, add WGL context & handle to DC 
		cl_context_properties properties[] = { 
			CL_GL_CONTEXT_KHR, (cl_context_properties)wglGetCurrentContext(), // WGL Context 
			CL_WGL_HDC_KHR, (cl_context_properties)wglGetCurrentDC(), // WGL HDC
			CL_CONTEXT_PLATFORM, (cl_context_properties)cp_platform, // OpenCL platform
			0
		};
		// Find CL capable devices in the current GL context 
		cl_GPU_context = clCreateContext(properties, 1, &device, NULL, 0, 0);

		// create a command-queue
		cl_cq = clCreateCommandQueue(cl_GPU_context, device, 0, &cl_err_num);
	}
#endif

#ifdef USE_CL
	void cl_clean_up()
	{
		//release openCL
		if(cl_cq)clReleaseCommandQueue(cl_cq);
		if(cl_GPU_context)clReleaseContext(cl_GPU_context);
	}
#endif

    void set_vsync( bool vsync )
    {
      the_window.setVerticalSyncEnabled( vsync );
    }

    template< class t >
    void handle_events( const t& f )
    {
      while( the_window.pollEvent( the_event ) )
      {
        if( the_event.type == sf::Event::Closed ||
            (
              the_event.type == sf::Event::KeyPressed &&
              the_event.key.code == sf::Keyboard::Escape
            )
          )
        {
          run = false;
        }

        f( the_event );
      }
    }

    template< class t >
    void display( const t& f, const bool& silent = false )
    {
      while( run )
      {
        f();

        the_window.display();
      }
    }

	void create_depth_texture(GLuint *tex, const uvec2 &size)
	{
		glGenTextures( 1, tex );

		glBindTexture( GL_TEXTURE_2D, *tex );
		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
		glTexImage2D( GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32, size.x, size.y, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, 0 );
	}

	void create_shmap_texture(GLuint *tex, const uvec2 &size)
	{
		glGenTextures( 1, tex );

		glBindTexture( GL_TEXTURE_2D, *tex );
		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_R_TO_TEXTURE );
		glTexImage2D( GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32, size.x, size.y, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, 0 );
	}

	void create_color_texture(GLuint *tex, const uvec2 &size)
	{
		glGenTextures( 1, tex );

		glBindTexture( GL_TEXTURE_2D, *tex );
		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
		glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F, size.x, size.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0 );
	}

    void load_shader( GLuint& program, const GLenum& type, const string& filename, bool print = false, const std::string& additional_str = "" ) const
    {
      ifstream f( filename );

      if( !f.is_open() )
      {
        cerr << "Couldn't load shader: " << filename << endl;
        return;
      }

      string str( ( istreambuf_iterator<char>( f ) ),
                  istreambuf_iterator<char>() );

      shader_include( str, filename.substr( 0, filename.rfind("/")+1 ) );

      if( print )
        cout << str << endl;

      if( !program ) program = glCreateProgram();

      compile_shader( str.c_str(), program, type, additional_str );
      link_shader( program );
    }

#ifdef USE_CL
	void load_cl_program( cl_program &program, const string& filename)
	{
		// Load the kernel source code into the array source_str
		ifstream f( filename );
		if( !f.is_open() )
		{
			cerr << "Couldn't load cl program: " << filename << endl;
			return;
		}

		string str( ( istreambuf_iterator<char>( f ) ),
			istreambuf_iterator<char>() );

		size_t size = str.size();
		const char *source = str.c_str();
		if( !program )
			program = clCreateProgramWithSource(cl_GPU_context, 1,(const char **) &source, &size, &cl_err_num);		

		// build the program
		cl_err_num = clBuildProgram(program, 0, NULL, "-cl-single-precision-constant -cl-mad-enable -cl-no-signed-zeros -cl-finite-math-only -cl-fast-relaxed-math", NULL, NULL);
		char c_build_log[10240];
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 
			sizeof(c_build_log), c_build_log, NULL );
		get_opencl_error(cl_err_num);
		cout << c_build_log << endl;

	}
#endif


#ifdef USE_CL
	void cl_aquire_buffer(cl_mem *buff, cl_uint count)
	{
		glFlush();
		cl_err_num = clEnqueueAcquireGLObjects(cl_cq,count,buff,0,NULL,NULL);	
		get_opencl_error(cl_err_num);
	}
#endif

#ifdef USE_CL
	void cl_release_buffer(cl_mem *buff, cl_uint count)
	{
		clEnqueueReleaseGLObjects(cl_cq,count,buff,0,NULL,NULL);
		cl_err_num = clFinish(cl_cq);
		get_opencl_error(cl_err_num);
	}
#endif

#ifdef USE_CL
	void execute_kernel(cl_kernel kernel, size_t global_work_size[2], size_t local_work_size[2], int dim)
	{
		cl_err_num = clEnqueueNDRangeKernel(cl_cq, kernel, dim, NULL, global_work_size, local_work_size, 0, NULL, NULL);
		get_opencl_error(cl_err_num);
	}
#endif

    GLuint create_quad( const vec3& ll, const vec3& lr, const vec3& ul, const vec3& ur ) const
    {
      vector<vec3> vertices;
      vector<vec2> tex_coords;
      vector<unsigned int> indices;
      GLuint vao = 0;
      GLuint vertex_vbo = 0, tex_coord_vbo = 0, index_vbo = 0;

      indices.push_back( 0 );
      indices.push_back( 1 );
      indices.push_back( 2 );

      indices.push_back( 0 );
      indices.push_back( 2 );
      indices.push_back( 3 );

      /*vertices.push_back( vec3( -1, -1, 0 ) );
      vertices.push_back( vec3( 1, -1, 0 ) );
      vertices.push_back( vec3( 1, 1, 0 ) );
      vertices.push_back( vec3( -1, 1, 0 ) );*/
      vertices.push_back( ll );
      vertices.push_back( lr );
      vertices.push_back( ur );
      vertices.push_back( ul );

      tex_coords.push_back( vec2( 0, 0 ) );
      tex_coords.push_back( vec2( 1, 0 ) );
      tex_coords.push_back( vec2( 1, 1 ) );
      tex_coords.push_back( vec2( 0, 1 ) );

      glGenVertexArrays( 1, &vao );
      glBindVertexArray( vao );

      glGenBuffers( 1, &vertex_vbo );
      glBindBuffer( GL_ARRAY_BUFFER, vertex_vbo );
      glBufferData( GL_ARRAY_BUFFER, sizeof( float ) * vertices.size() * 3, &vertices[0][0], GL_STATIC_DRAW );
      glEnableVertexAttribArray( 0 );
      glVertexAttribPointer( 0, 3, GL_FLOAT, 0, 0, 0 );

      glGenBuffers( 1, &tex_coord_vbo );
      glBindBuffer( GL_ARRAY_BUFFER, tex_coord_vbo );
      glBufferData( GL_ARRAY_BUFFER, sizeof( float ) * tex_coords.size() * 2, &tex_coords[0][0], GL_STATIC_DRAW );
      glEnableVertexAttribArray( 1 );
      glVertexAttribPointer( 1, 2, GL_FLOAT, 0, 0, 0 );

      glGenBuffers( 1, &index_vbo );
      glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, index_vbo );
      glBufferData( GL_ELEMENT_ARRAY_BUFFER, sizeof( unsigned int ) * indices.size(), &indices[0], GL_STATIC_DRAW );

      glBindVertexArray( 0 );

      return vao;
    }
	
    inline GLuint create_box() const
    {
      vector<vec3> vertices;
      vector<vec3> normals;
      vector<unsigned int> indices;
      GLuint vao = 0;
      GLuint vertex_vbo = 0, normal_vbo = 0, index_vbo = 0;

      //front
      vertices.push_back( vec3( -1, -1, 1 ) );
      vertices.push_back( vec3( 1, -1, 1 ) );
      vertices.push_back( vec3( 1, 1, 1 ) );
      vertices.push_back( vec3( -1, 1, 1 ) );

      indices.push_back( 0 );
      indices.push_back( 1 );
      indices.push_back( 2 );

      indices.push_back( 0 );
      indices.push_back( 2 );
      indices.push_back( 3 );

      normals.push_back( vec3( 0, 0, 1 ) );
      normals.push_back( vec3( 0, 0, 1 ) );
      normals.push_back( vec3( 0, 0, 1 ) );
      normals.push_back( vec3( 0, 0, 1 ) );

      //back
      vertices.push_back( vec3( -1, -1, -1 ) );
      vertices.push_back( vec3( 1, -1, -1 ) );
      vertices.push_back( vec3( 1, 1, -1 ) );
      vertices.push_back( vec3( -1, 1, -1 ) );

      indices.push_back( 6 );
      indices.push_back( 5 );
      indices.push_back( 4 );

      indices.push_back( 7 );
      indices.push_back( 6 );
      indices.push_back( 4 );

      normals.push_back( vec3( 0, 0, -1 ) );
      normals.push_back( vec3( 0, 0, -1 ) );
      normals.push_back( vec3( 0, 0, -1 ) );
      normals.push_back( vec3( 0, 0, -1 ) );

      //left
      vertices.push_back( vec3( -1, -1, 1 ) );
      vertices.push_back( vec3( -1, -1, -1 ) );
      vertices.push_back( vec3( -1, 1, 1 ) );
      vertices.push_back( vec3( -1, 1, -1 ) );

      indices.push_back( 10 );
      indices.push_back( 9 );
      indices.push_back( 8 );

      indices.push_back( 11 );
      indices.push_back( 9 );
      indices.push_back( 10 );

      normals.push_back( vec3( -1, 0, 0 ) );
      normals.push_back( vec3( -1, 0, 0 ) );
      normals.push_back( vec3( -1, 0, 0 ) );
      normals.push_back( vec3( -1, 0, 0 ) );

      //right
      vertices.push_back( vec3( 1, -1, 1 ) );
      vertices.push_back( vec3( 1, -1, -1 ) );
      vertices.push_back( vec3( 1, 1, 1 ) );
      vertices.push_back( vec3( 1, 1, -1 ) );

      indices.push_back( 12 );
      indices.push_back( 13 );
      indices.push_back( 14 );

      indices.push_back( 14 );
      indices.push_back( 13 );
      indices.push_back( 15 );

      normals.push_back( vec3( 1, 0, 0 ) );
      normals.push_back( vec3( 1, 0, 0 ) );
      normals.push_back( vec3( 1, 0, 0 ) );
      normals.push_back( vec3( 1, 0, 0 ) );

      //up
      vertices.push_back( vec3( -1, 1, 1 ) );
      vertices.push_back( vec3( 1, 1, 1 ) );
      vertices.push_back( vec3( 1, 1, -1 ) );
      vertices.push_back( vec3( -1, 1, -1 ) );

      indices.push_back( 16 );
      indices.push_back( 17 );
      indices.push_back( 18 );

      indices.push_back( 16 );
      indices.push_back( 18 );
      indices.push_back( 19 );

      normals.push_back( vec3( 0, 1, 0 ) );
      normals.push_back( vec3( 0, 1, 0 ) );
      normals.push_back( vec3( 0, 1, 0 ) );
      normals.push_back( vec3( 0, 1, 0 ) );

      //down
      vertices.push_back( vec3( -1, -1, 1 ) );
      vertices.push_back( vec3( 1, -1, 1 ) );
      vertices.push_back( vec3( 1, -1, -1 ) );
      vertices.push_back( vec3( -1, -1, -1 ) );

      indices.push_back( 22 );
      indices.push_back( 21 );
      indices.push_back( 20 );

      indices.push_back( 23 );
      indices.push_back( 22 );
      indices.push_back( 20 );

      normals.push_back( vec3( 0, -1, 0 ) );
      normals.push_back( vec3( 0, -1, 0 ) );
      normals.push_back( vec3( 0, -1, 0 ) );
      normals.push_back( vec3( 0, -1, 0 ) );

      glGenVertexArrays( 1, &vao );
      glBindVertexArray( vao );

      glGenBuffers( 1, &vertex_vbo );
      glBindBuffer( GL_ARRAY_BUFFER, vertex_vbo );
      glBufferData( GL_ARRAY_BUFFER, sizeof( float ) * vertices.size() * 3, &vertices[0][0], GL_STATIC_DRAW );
      glEnableVertexAttribArray( 0 );
      glVertexAttribPointer( 0, 3, GL_FLOAT, 0, 0, 0 );

      glGenBuffers( 1, &normal_vbo );
      glBindBuffer( GL_ARRAY_BUFFER, normal_vbo );
      glBufferData( GL_ARRAY_BUFFER, sizeof( float ) * normals.size() * 3, &normals[0][0], GL_STATIC_DRAW );
      glEnableVertexAttribArray( 2 );
      glVertexAttribPointer( 2, 3, GL_FLOAT, 0, 0, 0 );

      glGenBuffers( 1, &index_vbo );
      glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, index_vbo );
      glBufferData( GL_ELEMENT_ARRAY_BUFFER, sizeof( unsigned int ) * indices.size(), &indices[0], GL_STATIC_DRAW );

      glBindVertexArray( 0 );

      return vao;
    }

    GLuint load_image( const std::string& str )
    {
      sf::Image im;
      im.loadFromFile( str );
      unsigned w = im.getSize().x;
      unsigned h = im.getSize().y;

      GLuint tex = 0;
      glGenTextures( 1, &tex );
      glBindTexture( GL_TEXTURE_2D, tex );
      glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT );
      glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT );
      glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
      glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
      glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, im.getPixelsPtr() );

      return tex;
    }

    std::string get_app_path()
    {

      char fullpath[4096];
      std::string app_path;

      /* /proc/self is a symbolic link to the process-ID subdir
       * of /proc, e.g. /proc/4323 when the pid of the process
       * of this program is 4323.
       *
       * Inside /proc/<pid> there is a symbolic link to the
       * executable that is running as this <pid>.  This symbolic
       * link is called "exe".
       *
       * So if we read the path where the symlink /proc/self/exe
       * points to we have the full path of the executable.
       */

#ifdef __unix__
      int length;
      length = readlink( "/proc/self/exe", fullpath, sizeof( fullpath ) );

      /* Catch some errors: */

      if( length < 0 )
      {
        my::log << my::lock << "Couldnt read app path. Error resolving symlink /proc/self/exe." << my::endl << my::unlock;
        loop::get().shutdown();
      }

      if( length >= 4096 )
      {
        my::log << my::lock << "Couldnt read app path. Path too long. Truncated." << my::endl << my::unlock;
        loop::get().shutdown();
      }

      /* I don't know why, but the string this readlink() function
       * returns is appended with a '@'.
       */
      fullpath[length] = '\0';       /* Strip '@' off the end. */

#endif

#ifdef _WIN32

      if( GetModuleFileName( 0, ( char* )&fullpath, sizeof( fullpath ) - 1 ) == 0 )
      {
        cerr << "Couldn't get the app path." << endl;
        return "";
      }

#endif

      app_path = fullpath;

#ifdef _WIN32
      app_path = app_path.substr( 0, app_path.rfind( "\\" ) + 1 );
#endif

#ifdef __unix__
      config::get().app_path = config::get().app_path.substr( 0, config::get().app_path.rfind( "/" ) + 1 );
#endif

      app_path += "../";

      char* res = 0;

      res = realpath( app_path.c_str(), 0 );

      if( res )
      {
#if _WIN32
        app_path = std::string( res );
        delete [] res;
#endif

#if __unix__
        config::get().app_path = std::string( res ) + "/";
        free( res ); //the original linux version of realpath uses malloc
#endif
      }

      std::replace( app_path.begin(), app_path.end(), '\\', '/' );

      return app_path;
    }

    void get_opengl_error( const bool& ignore = false ) const
    {
      bool got_error = false;
      GLenum error = 0;
      error = glGetError();
      string errorstring = "";

      while( error != GL_NO_ERROR )
      {
        if( error == GL_INVALID_ENUM )
        {
          //An unacceptable value is specified for an enumerated argument. The offending command is ignored and has no other side effect than to set the error flag.
          errorstring += "OpenGL error: invalid enum...\n";
          got_error = true;
        }

        if( error == GL_INVALID_VALUE )
        {
          //A numeric argument is out of range. The offending command is ignored and has no other side effect than to set the error flag.
          errorstring += "OpenGL error: invalid value...\n";
          got_error = true;
        }

        if( error == GL_INVALID_OPERATION )
        {
          //The specified operation is not allowed in the current state. The offending command is ignored and has no other side effect than to set the error flag.
          errorstring += "OpenGL error: invalid operation...\n";
          got_error = true;
        }

        if( error == GL_STACK_OVERFLOW )
        {
          //This command would cause a stack overflow. The offending command is ignored and has no other side effect than to set the error flag.
          errorstring += "OpenGL error: stack overflow...\n";
          got_error = true;
        }

        if( error == GL_STACK_UNDERFLOW )
        {
          //This command would cause a stack underflow. The offending command is ignored and has no other side effect than to set the error flag.
          errorstring += "OpenGL error: stack underflow...\n";
          got_error = true;
        }

        if( error == GL_OUT_OF_MEMORY )
        {
          //There is not enough memory left to execute the command. The state of the GL is undefined, except for the state of the error flags, after this error is recorded.
          errorstring += "OpenGL error: out of memory...\n";
          got_error = true;
        }

        if( error == GL_TABLE_TOO_LARGE )
        {
          //The specified table exceeds the implementation's maximum supported table size.  The offending command is ignored and has no other side effect than to set the error flag.
          errorstring += "OpenGL error: table too large...\n";
          got_error = true;
        }

        error = glGetError();
      }

      if( got_error && !ignore )
      {
        cerr << errorstring;
        return;
      }
    }

    void check_fbo_status( const GLenum& target = GL_FRAMEBUFFER )
    {
      GLenum error_code = glCheckFramebufferStatus( target );

      switch( error_code )
      {
        case GL_FRAMEBUFFER_COMPLETE:
          break;
        case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
          cerr << "GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT." << endl;
          break;
        case GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS_EXT:
          cerr << "GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS." << endl;
          break;
        case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
          cerr << "GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT." << endl;
          break;
        case GL_FRAMEBUFFER_UNSUPPORTED:
          cerr << "GL_FRAMEBUFFER_UNSUPPORTED." << endl;
          break;
        case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER:
          cerr << "GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER." << endl;
          break;
        case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER:
          cerr << "GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER." << endl;
          break;
        case GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE:
          cerr << "GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE." << endl;
          break;
        case GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS_ARB:
          cerr << "GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS." << endl;
          break;
        case GL_FRAMEBUFFER_INCOMPLETE_LAYER_COUNT_ARB:
          cerr << "GL_FRAMEBUFFER_INCOMPLETE_LAYER_COUNT." << endl;
          break;
        case GL_FRAMEBUFFER_INCOMPLETE_FORMATS_EXT:
          cerr << "GL_FRAMEBUFFER_INCOMPLETE_FORMATS." << endl;
          break;
        default:
          cerr << "Unknown Frame Buffer error cause: " << error_code << endl;
          break;
      }
    }

#ifdef USE_CL
	// Helper function to get OpenCL error string from constant
	// *********************************************************************
	void get_opencl_error(cl_int error)
	{
		if (error == 0)
			return;
		static const char* errorString[] = {
			"CL_SUCCESS",
			"CL_DEVICE_NOT_FOUND",
			"CL_DEVICE_NOT_AVAILABLE",
			"CL_COMPILER_NOT_AVAILABLE",
			"CL_MEM_OBJECT_ALLOCATION_FAILURE",
			"CL_OUT_OF_RESOURCES",
			"CL_OUT_OF_HOST_MEMORY",
			"CL_PROFILING_INFO_NOT_AVAILABLE",
			"CL_MEM_COPY_OVERLAP",
			"CL_IMAGE_FORMAT_MISMATCH",
			"CL_IMAGE_FORMAT_NOT_SUPPORTED",
			"CL_BUILD_PROGRAM_FAILURE",
			"CL_MAP_FAILURE",
			"",
			"",
			"",
			"",
			"",
			"",
			"",
			"",
			"",
			"",
			"",
			"",
			"",
			"",
			"",
			"",
			"",
			"CL_INVALID_VALUE",
			"CL_INVALID_DEVICE_TYPE",
			"CL_INVALID_PLATFORM",
			"CL_INVALID_DEVICE",
			"CL_INVALID_CONTEXT",
			"CL_INVALID_QUEUE_PROPERTIES",
			"CL_INVALID_COMMAND_QUEUE",
			"CL_INVALID_HOST_PTR",
			"CL_INVALID_MEM_OBJECT",
			"CL_INVALID_IMAGE_FORMAT_DESCRIPTOR",
			"CL_INVALID_IMAGE_SIZE",
			"CL_INVALID_SAMPLER",
			"CL_INVALID_BINARY",
			"CL_INVALID_BUILD_OPTIONS",
			"CL_INVALID_PROGRAM",
			"CL_INVALID_PROGRAM_EXECUTABLE",
			"CL_INVALID_KERNEL_NAME",
			"CL_INVALID_KERNEL_DEFINITION",
			"CL_INVALID_KERNEL",
			"CL_INVALID_ARG_INDEX",
			"CL_INVALID_ARG_VALUE",
			"CL_INVALID_ARG_SIZE",
			"CL_INVALID_KERNEL_ARGS",
			"CL_INVALID_WORK_DIMENSION",
			"CL_INVALID_WORK_GROUP_SIZE",
			"CL_INVALID_WORK_ITEM_SIZE",
			"CL_INVALID_GLOBAL_OFFSET",
			"CL_INVALID_EVENT_WAIT_LIST",
			"CL_INVALID_EVENT",
			"CL_INVALID_OPERATION",
			"CL_INVALID_GL_OBJECT",
			"CL_INVALID_BUFFER_SIZE",
			"CL_INVALID_MIP_LEVEL",
			"CL_INVALID_GLOBAL_WORK_SIZE",
		};

		const int errorCount = sizeof(errorString) / sizeof(errorString[0]);

		const int index = -error;

		cout << ((index >= 0 && index < errorCount) ? errorString[index] : "Unspecified Error") << endl;
	}
#endif
};

// Round Up Division function
inline size_t round_up(int group_size, int global_size) 
{
	int r = global_size % group_size;
	if(r == 0) 
		return global_size;
	else 
		return global_size + group_size - r;
}

class mesh
{
  public:
    std::vector< uvec3 > indices;
    std::vector< vec3 > vertices;
    std::vector< vec3 > normals;
    std::vector< vec3 > tangents;
    std::vector< vec2 > tex_coords;

    unsigned rendersize;

    GLuint vao;
    GLuint vbos[6];

    shape* bv;

    enum vbo_type
    {
      VERTEX = 0, TEX_COORD, NORMAL, TANGENT, INDEX
    };

    static void save_meshes( const std::string& path, vector<mesh>& meshes )
    {
      fstream f;
      f.open( path.c_str(), ios::out );
      
      if( !f.is_open() )
      {
        cerr << "Couldn't save meshes into file: " << path << endl;
      }

      int counter = 0;
      int global_max_index = 0;
      for( auto& c : meshes )
      {
        f << "o object" << counter++ << endl;

        for( auto& v : c.vertices )
        {
          f << "v " << v.x << " " << v.y << " " << v.z << endl;
        }

        for( auto& t : c.tex_coords )
        {
          f << "vt " << t.x << " " << t.y << endl;
        }

        for( auto& n : c.normals )
        {
          f << "vn " << n.x << " " << n.y << " " << n.z << endl;
        }

        int max_index = global_max_index;
        for( int i = 0; i < c.indices.size(); ++i )
        {
          f << "f " << max_index + c.indices[i].x+1 << "/" 
                    << max_index + c.indices[i].x+1 << "/" 
                    << max_index + c.indices[i].x+1 << " ";
            f << max_index + c.indices[i].y+1 << "/" 
              << max_index + c.indices[i].y+1 << "/" 
              << max_index + c.indices[i].y+1 << " ";
            f << max_index + c.indices[i].z+1 << "/" 
              << max_index + c.indices[i].z+1 << "/" 
              << max_index + c.indices[i].z+1 << endl;

          global_max_index = max( global_max_index, max_index + c.indices[i].x+1 );
          global_max_index = max( global_max_index, max_index + c.indices[i].y+1 );
          global_max_index = max( global_max_index, max_index + c.indices[i].z+1 );
        }

        f << endl;
      }

      f.close();
    }

	  static void load_into_meshes( const std::string& path, vector<mesh>& meshes, const bool& flip = false )
    {
      Assimp::Importer the_importer;

      const aiScene* the_scene = the_importer.ReadFile( path.c_str(), aiProcess_JoinIdenticalVertices |
                                 aiProcess_ImproveCacheLocality |
                                 aiProcess_LimitBoneWeights |
                                 aiProcess_RemoveRedundantMaterials |
                                 aiProcess_SplitLargeMeshes |
                                 aiProcess_FindDegenerates |
                                 aiProcess_FindInvalidData |
                                 aiProcess_FindInstances |
                                 aiProcess_ValidateDataStructure |
                                 aiProcess_OptimizeMeshes |
                                 aiProcess_CalcTangentSpace |
                                 ( flip ? aiProcess_FlipUVs : 0 ) );

      if( !the_scene )
      {
        std::cerr << the_importer.GetErrorString() << std::endl;
        return;
      }

      for( unsigned int c = 0; c < the_scene->mNumMeshes; ++c )
      {
		meshes.resize(meshes.size()+1);

        //write out face indices
        for( unsigned int d = 0; d < the_scene->mMeshes[c]->mNumFaces; ++d )
        {
          const aiFace* faces = &the_scene->mMeshes[c]->mFaces[d];
          meshes[c].indices.push_back( uvec3( faces->mIndices[0], faces->mIndices[1], faces->mIndices[2] ) );
        }

        //write out vertices
		meshes[c].vertices.resize(the_scene->mMeshes[c]->mNumVertices);
		memcpy(&meshes[c].vertices[0].x, &the_scene->mMeshes[c]->mVertices[0], the_scene->mMeshes[c]->mNumVertices * sizeof(vec3));
        /*for( unsigned int d = 0; d < the_scene->mMeshes[c]->mNumVertices; ++d )
        {
          const aiVector3D* vertex = &the_scene->mMeshes[c]->mVertices[d];
          meshes[c].vertices.push_back( vec3( vertex->x, vertex->y, vertex->z ) );
        }*/

        //write out normals
        if( the_scene->mMeshes[c]->mNormals )
        {
		  meshes[c].normals.resize(the_scene->mMeshes[c]->mNumVertices);
		  memcpy(&meshes[c].normals[0].x, &the_scene->mMeshes[c]->mNormals[0], the_scene->mMeshes[c]->mNumVertices * sizeof(vec3));
          /*for( unsigned int d = 0; d < the_scene->mMeshes[c]->mNumVertices; ++d )
          {
            const aiVector3D* normal = &the_scene->mMeshes[c]->mNormals[d];
            meshes[c].normals.push_back( vec3( normal->x, normal->y, normal->z ) );
          }*/
        }

        //write out tex coords
        if( the_scene->mMeshes[c]->mTextureCoords[0] )
        {
          for( unsigned int d = 0; d < the_scene->mMeshes[c]->mNumVertices; ++d )
          {
            const aiVector3D* tex_coord = &the_scene->mMeshes[c]->mTextureCoords[0][d];
            meshes[c].tex_coords.push_back( vec2( tex_coord->x, tex_coord->y ) );
          }
        }

        if( the_scene->mMeshes[c]->mTangents )
        {
		  meshes[c].tangents.resize(the_scene->mMeshes[c]->mNumVertices);
		  memcpy(&meshes[c].tangents[0].x, &the_scene->mMeshes[c]->mTangents[0], the_scene->mMeshes[c]->mNumVertices * sizeof(vec3));
          /*for( unsigned int d = 0; d < the_scene->mMeshes[c]->mNumVertices; ++d )
          {
            const aiVector3D* tangent = &the_scene->mMeshes[c]->mTangents[d];
            meshes[c].tangents.push_back( vec3( tangent->x, tangent->y, tangent->z ) );
          }*/
        }
      }
    }

    void load( const std::string& path, const bool& flip = false )
    {
      Assimp::Importer the_importer;

      const aiScene* the_scene = the_importer.ReadFile( path.c_str(), aiProcess_JoinIdenticalVertices |
                                 aiProcess_ImproveCacheLocality |
                                 aiProcess_LimitBoneWeights |
                                 aiProcess_RemoveRedundantMaterials |
                                 aiProcess_SplitLargeMeshes |
                                 aiProcess_FindDegenerates |
                                 aiProcess_FindInvalidData |
                                 aiProcess_FindInstances |
                                 aiProcess_ValidateDataStructure |
                                 aiProcess_OptimizeMeshes |
                                 aiProcess_CalcTangentSpace |
                                 ( flip ? aiProcess_FlipUVs : 0 ) );

      if( !the_scene )
      {
        std::cerr << the_importer.GetErrorString() << std::endl;
        return;
      }

      for( unsigned int c = 0; c < the_scene->mNumMeshes; ++c )
      {

        //write out face indices
        for( unsigned int d = 0; d < the_scene->mMeshes[c]->mNumFaces; ++d )
        {
          const aiFace* faces = &the_scene->mMeshes[c]->mFaces[d];
          indices.push_back( uvec3( faces->mIndices[0], faces->mIndices[1], faces->mIndices[2] ) );
        }

        //write out vertices
		vertices.resize(the_scene->mMeshes[c]->mNumVertices);
		memcpy(&vertices[0].x, &the_scene->mMeshes[c]->mVertices[0], the_scene->mMeshes[c]->mNumVertices * sizeof(vec3));
        /*for( unsigned int d = 0; d < the_scene->mMeshes[c]->mNumVertices; ++d )
        {
          const aiVector3D* vertex = &the_scene->mMeshes[c]->mVertices[d];
          vertices.push_back( vec3( vertex->x, vertex->y, vertex->z ) );
        }*/

        //write out normals
        if( the_scene->mMeshes[c]->mNormals )
        {
		  normals.resize(the_scene->mMeshes[c]->mNumVertices);
		  memcpy(&normals[0].x, &the_scene->mMeshes[c]->mNormals[0], the_scene->mMeshes[c]->mNumVertices * sizeof(vec3));
          /*for( unsigned int d = 0; d < the_scene->mMeshes[c]->mNumVertices; ++d )
          {
            const aiVector3D* normal = &the_scene->mMeshes[c]->mNormals[d];
            normals.push_back( vec3( normal->x, normal->y, normal->z ) );
          }*/
        }

        //write out tex coords
        if( the_scene->mMeshes[c]->mTextureCoords[0] )
        {
          for( unsigned int d = 0; d < the_scene->mMeshes[c]->mNumVertices; ++d )
          {
            const aiVector3D* tex_coord = &the_scene->mMeshes[c]->mTextureCoords[0][d];
            tex_coords.push_back( vec2( tex_coord->x, tex_coord->y ) );
          }
        }

        if( the_scene->mMeshes[c]->mTangents )
        {
		  tangents.resize(the_scene->mMeshes[c]->mNumVertices);
		  memcpy(&tangents[0].x, &the_scene->mMeshes[c]->mTangents[0], the_scene->mMeshes[c]->mNumVertices * sizeof(vec3));
          /*for( unsigned int d = 0; d < the_scene->mMeshes[c]->mNumVertices; ++d )
          {
            const aiVector3D* tangent = &the_scene->mMeshes[c]->mTangents[d];
            tangents.push_back( vec3( tangent->x, tangent->y, tangent->z ) );
          }*/
        }
      }
    }

	void write_mesh( const std::string& path )
    {
      std::fstream f;
      f.open( path, std::ios::out | std::ios::binary );

      if( !f.is_open() )
      {
        std::cerr << "Couldn't open file." << std::endl;
        return;
      }

	  unsigned size = indices.size();
	  f.write((const char*)&size, sizeof(unsigned)); //write out num of faces
	  size = vertices.size();
	  f.write((const char*)&size, sizeof(unsigned)); //write out num of vertices
	  size = !normals.empty();
	  f.write((const char*)&size, sizeof(unsigned)); //does it have normals?
	  size = !tex_coords.empty();
	  f.write((const char*)&size, sizeof(unsigned)); //does it have tex coords?

	  f.write((const char*)&indices[0].x, sizeof(uvec3) * indices.size());
	  /*for( int c = 0; c < indices.size(); ++c )
      {
		  f.write((const char*)&indices[c].x, sizeof(uvec3));
      }*/

	  f.write((const char*)&vertices[0].x, sizeof(vec3) * vertices.size());
	  /*for( int c = 0; c < vertices.size(); ++c )
      {
		  f.write((const char*)&vertices[c].x, sizeof(vec3));
      }*/

	  if( !normals.empty() )
      {
		f.write((const char*)&normals[0].x, sizeof(vec3) * normals.size());
		/*for( int c = 0; c < normals.size(); ++c )
        {
          f.write((const char*)&normals[c].x, sizeof(vec3));
        }*/
      }

      if( !tex_coords.empty() )
      {
		f.write((const char*)&tex_coords[0].x, sizeof(vec2) * tex_coords.size());
		/*for( int c = 0; c < tex_coords.size(); ++c )
        {
          f.write((const char*)&tex_coords[c].x, sizeof(vec2));
        }*/
      }

      f.close();
    }

    void read_mesh( const std::string& path )
    {
      std::fstream f;
      f.open( path, std::ios::in | std::ios::binary );

      if( !f.is_open() )
      {
        std::cerr << "Couldn't open file." << std::endl;
        return;
      }

      unsigned int num_faces = 0, num_vertices = 0, has_normals = 0, has_tex_coords = 0;
      char* buffer = new char[sizeof( unsigned int )];

      f.read( buffer, sizeof( unsigned int ) );
      num_faces = *( unsigned int* )buffer;
      f.read( buffer, sizeof( unsigned int ) );
      num_vertices = *( unsigned int* )buffer;
      f.read( buffer, sizeof( unsigned int ) );
      has_normals = *( unsigned int* )buffer;
      f.read( buffer, sizeof( unsigned int ) );
      has_tex_coords = *( unsigned int* )buffer;

#ifdef WRITESTATS
      std::cout << "Num faces: " << num_faces << std::endl;
      std::cout << "Num vertices: " << num_vertices << std::endl;
      std::cout << "Has normals: " << ( has_normals ? "yes" : "no" ) << std::endl;
      std::cout << "Has tex coords: " << ( has_tex_coords ? "yes" : "no" ) << std::endl;
#endif

      delete [] buffer;
      //buffer = new char[sizeof( unsigned int ) * 3];

#ifdef WRITECOMPONENTS
      std::cout << "Indices: " << std::endl;
#endif
	  indices.resize(num_faces);
	  f.read( (char*)&indices[0].x, sizeof( uvec3 ) * num_faces );
	  /*
      for( int c = 0; c < num_faces; ++c )
      {
        f.read( buffer, sizeof( unsigned int ) * 3 );
        unsigned int x = *( unsigned int* )( buffer + 0 * sizeof( unsigned int ) );
        unsigned int y = *( unsigned int* )( buffer + 1 * sizeof( unsigned int ) );
        unsigned int z = *( unsigned int* )( buffer + 2 * sizeof( unsigned int ) );
        indices.push_back( uvec3( x, y, z ) );
#ifdef WRITECOMPONENTS
        std::cout << x << " " << y << " " << z << std::endl;
#endif
      }*/

      //delete [] buffer;
      //buffer = new char[sizeof( float ) * 3];

#ifdef WRITECOMPONENTS
      std::cout << std::endl << "Vertices: " << std::endl;
#endif
	  vertices.resize(num_vertices);
	  f.read( (char*)&vertices[0].x, sizeof( vec3 ) * num_vertices );
      /*for( int c = 0; c < num_vertices; ++c )
      {
        f.read( buffer, sizeof( float ) * 3 );
        float x = *( float* )( buffer + 0 * sizeof( float ) );
        float y = *( float* )( buffer + 1 * sizeof( float ) );
        float z = *( float* )( buffer + 2 * sizeof( float ) );
        vertices.push_back( vec3( x, y, z ) );
#ifdef WRITECOMPONENTS
        std::cout << x << " " << y << " " << z << std::endl;
#endif
      }*/

      if( has_normals )
      {
#ifdef WRITECOMPONENTS
        std::cout << std::endl << "Normals: " << std::endl;
#endif
		normals.resize(num_vertices);
		f.read( (char*)&normals[0].x, sizeof( vec3 ) * num_vertices );
        /*for( int c = 0; c < num_vertices; ++c )
        {
          f.read( buffer, sizeof( float ) * 3 );
          float x = *( float* )( buffer + 0 * sizeof( float ) );
          float y = *( float* )( buffer + 1 * sizeof( float ) );
          float z = *( float* )( buffer + 2 * sizeof( float ) );
          normals.push_back( vec3( x, y, z ) );
#ifdef WRITECOMPONENTS
          std::cout << x << " " << y << " " << z << std::endl;
#endif
        }*/
      }

      //delete [] buffer;
      //buffer = new char[sizeof( float ) * 2];

      if( has_tex_coords )
      {
#ifdef WRITECOMPONENTS
        std::cout << std::endl << "Tex coords: " << std::endl;
#endif
		tex_coords.resize(num_vertices);
		f.read( (char*)&tex_coords[0].x, sizeof( vec2 ) * num_vertices );
        /*for( int c = 0; c < num_vertices; ++c )
        {
          f.read( buffer, sizeof( float ) * 2 );
          float x = *( float* )( buffer + 0 * sizeof( float ) );
          float y = *( float* )( buffer + 1 * sizeof( float ) );
          tex_coords.push_back( vec2( x, y ) );
#ifdef WRITECOMPONENTS
          std::cout << x << " " << y << std::endl;
#endif
        }*/
      }

      //delete [] buffer;

      f.close();
    }

    void upload()
    {
      glGenVertexArrays( 1, &vao );
      glBindVertexArray( vao );

      glGenBuffers( 1, &vbos[VERTEX] );
      glBindBuffer( GL_ARRAY_BUFFER, vbos[VERTEX] );
      glBufferData( GL_ARRAY_BUFFER, sizeof( vec3 ) * vertices.size(), &vertices[0], GL_STATIC_DRAW );
      glEnableVertexAttribArray( VERTEX );
      glVertexAttribPointer( VERTEX, 3, GL_FLOAT, 0, 0, 0 );

      if( normals.size() > 0 )
      {
        glGenBuffers( 1, &vbos[NORMAL] );
        glBindBuffer( GL_ARRAY_BUFFER, vbos[NORMAL] );
        glBufferData( GL_ARRAY_BUFFER, sizeof( vec3 ) * normals.size(), &normals[0], GL_STATIC_DRAW );
        glEnableVertexAttribArray( NORMAL );
        glVertexAttribPointer( NORMAL, 3, GL_FLOAT, 0, 0, 0 );
      }

      if( tex_coords.size() > 0 )
      {
        glGenBuffers( 1, &vbos[TEX_COORD] );
        glBindBuffer( GL_ARRAY_BUFFER, vbos[TEX_COORD] );
        glBufferData( GL_ARRAY_BUFFER, sizeof( vec2 ) * tex_coords.size(), &tex_coords[0], GL_STATIC_DRAW );
        glEnableVertexAttribArray( TEX_COORD );
        glVertexAttribPointer( TEX_COORD, 2, GL_FLOAT, 0, 0, 0 );
      }

      if( tangents.size() > 0 )
      {
        glGenBuffers( 1, &vbos[TANGENT] );
        glBindBuffer( GL_ARRAY_BUFFER, vbos[TANGENT] );
        glBufferData( GL_ARRAY_BUFFER, sizeof( vec3 ) * tangents.size(), &tangents[0], GL_STATIC_DRAW );
        glEnableVertexAttribArray( TANGENT );
        glVertexAttribPointer( TANGENT, 3, GL_FLOAT, 0, 0, 0 );
      }

      glGenBuffers( 1, &vbos[INDEX] );
      glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, vbos[INDEX] );
      glBufferData( GL_ELEMENT_ARRAY_BUFFER, sizeof( vec3 ) * indices.size(), &indices[0], GL_STATIC_DRAW );

      glBindVertexArray( 0 );
      //glBindBuffer( GL_ARRAY_BUFFER, 0 );
      //glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, 0 );

      rendersize = indices.size() * 3;
    }

    void render()
    {
      glBindVertexArray( vao );
      glDrawElements( GL_TRIANGLES, rendersize, GL_UNSIGNED_INT, 0 );
    }
};

#endif
