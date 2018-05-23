// terrain.c - public domain GPU quadtree renderers
// author: Jonathan Dupuy (jdupuy@liris.cnrs.fr)
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#if LINUX
#include <unistd.h>
#endif
#include "buf.h"
#include "glload.h"     /* opengl functions loader */
#include "GLFW/glfw3.h"  /* window manager */
#include "vec.h"      /* generic vector math library */
#include "affine.h"   /* affine transformation */
#include "frustum.h"  /* frustum */
#include "ft.h"        /* font renderer */
#include "image.h"    /* image utility */
#include "program.h"  /* program loader */
#include "timer.h"    /* opengl timers */

// ---------------------------------------------------------
// Constants
enum {
	BUFFER_PATCH_DATA1 = 0,
	BUFFER_PATCH_DATA2,
	BUFFER_INDIRECT,
	BUFFER_FRUSTUM,
	BUFFER_GRID_VERTICES,
	BUFFER_GRID_INDEXES,
	BUFFER_COUNT,

#if TERRAIN_RENDERER
	TEXTURE_PUGET = 0,
	TEXTURE_COUNT,
#else
	TEXTURE_DUMMY = 0, // not used
	TEXTURE_COUNT,
#endif

	VERTEX_ARRAY_EMPTY = 0,
	VERTEX_ARRAY_QUADTREE1,
	VERTEX_ARRAY_QUADTREE2,
	VERTEX_ARRAY_TERRAIN1,
	VERTEX_ARRAY_TERRAIN2,
	VERTEX_ARRAY_COUNT,

	FEEDBACK_QUADTREE1 = 0,
	FEEDBACK_QUADTREE2,
	FEEDBACK_COUNT,

	PROGRAM_LOD = 0,
	PROGRAM_CULL,
	PROGRAM_RENDER,
	PROGRAM_COUNT,

	LOCATION_LOD_SCENE_SIZE = 0,
	LOCATION_LOD_EYE_POS,
	LOCATION_CULL_SCENE_SIZE,
	LOCATION_RENDER_SCENE_SIZE,
	LOCATION_RENDER_MVP,
	LOCATION_RENDER_EYE_POS,
	LOCATION_RENDER_GPU_TESS_FACTOR,
	LOCATION_RENDER_GRID_TESS_FACTOR,
#if TERRAIN_RENDERER
	LOCATION_LOD_PUGET_SAMPLER,
	LOCATION_RENDER_PUGET_SAMPLER,
#endif
	LOCATION_COUNT,

	TIMER_LOD = 0,
	TIMER_CULL,
	TIMER_RENDER,
	TIMER_COUNT,

	WINDOW_WIDTH = 1024,
	WINDOW_HEIGHT = 1024
};

// ---------------------------------------------------------
// Global variables
const float pi = 3.14159f;
const float screen_ratio = (float)WINDOW_WIDTH / WINDOW_HEIGHT;
const float fovy = 3.14159f * 0.25f;
const float camera_velocity = 15.f/3.6f; // 15km/h

// opengl objects
GLuint g_buffers[BUFFER_COUNT];
GLuint g_vertex_arrays[VERTEX_ARRAY_COUNT];
GLuint g_feedbacks[FEEDBACK_COUNT];
GLuint g_programs[PROGRAM_COUNT];
GLuint g_textures[TEXTURE_COUNT];
GLuint g_query = 0;
// uniform locations
GLint g_uniform_locations[LOCATION_COUNT];
// flags
GLboolean g_init_ok = GL_FALSE;
GLboolean g_wireframe = GL_TRUE;
GLboolean g_eye_lock = GL_FALSE;
GLboolean g_eye_freeze = GL_FALSE;
GLboolean g_eye_boost = GL_FALSE;
GLboolean g_show_gui = GL_TRUE;
// misc
vec3_t g_eye_velocity = {0, 0, 0};
struct affine_t *g_eye;
struct timer_t *g_timers[TIMER_COUNT];
// terrain
GLint g_pingpong  = 1; // ping pong variable
#if TERRAIN_RENDERER
GLfloat g_scene_size = 1e5; // terrain size
#else
GLfloat g_scene_size = 5e3; // terrain size
#endif
GLint g_grid_tess_factor = 8; // path tessellation [2, 256]
GLint g_gpu_tess_factor   = 0;  // GPU tessellation factor [0,5]
GLint g_patch_index_count = 0;

// ---------------------------------------------------------
// utilities
float maxf(float a, float b) {
	return a > b ? a : b;
}
float sqrf(float x) {
	return x * x;
}

// ---------------------------------------------------------
// lod program
#define SHADER_PATH(x) "../../src/shaders/" x
static GLboolean load_lod_program() {
	const GLchar *varyings[] = {"o_data"};
	struct program_args_t *args = program_args_create();
	GLuint program = 0;

	program_args_set_version(args, 420, false);
#if TERRAIN_RENDERER
	program_args_push_string(args, "#define TERRAIN_RENDERER 1\n");
	program_args_push_file(args, SHADER_PATH("terrain.glsl"));
#elif PARAMETRIC_RENDERER
	program_args_push_string(args, "#define PARAMETRIC_RENDERER 1\n");
	program_args_push_file(args, SHADER_PATH("parametrics.glsl"));
#endif
	program_args_push_file(args, SHADER_PATH("ltree.glsl"));
	program_args_push_file(args, SHADER_PATH("quadtree_lod.glsl"));
	program = program_create(args, false);
	if(!program) {
		fprintf(stderr, "log: %s\n", program_get_log());
		return GL_FALSE;
	}
	glTransformFeedbackVaryings(program, 1, varyings, GL_SEPARATE_ATTRIBS);
	glLinkProgram(program);
	g_programs[PROGRAM_LOD] = program;
	return GL_TRUE;
}

// ---------------------------------------------------------
// cull program
static GLboolean load_cull_program() {
	const GLchar *varyings[] = {"o_data"};
	struct program_args_t *args = program_args_create();
	GLuint program = 0;

	program_args_set_version(args, 420, false);
#if TERRAIN_RENDERER
	program_args_push_string(args, "#define TERRAIN_RENDERER 1\n");
#elif PARAMETRIC_RENDERER
	program_args_push_string(args, "#define PARAMETRIC_RENDERER 1\n");
	program_args_push_file(args, SHADER_PATH("parametrics.glsl"));
#endif
	program_args_push_file(args, SHADER_PATH("ltree.glsl"));
	program_args_push_file(args, SHADER_PATH("quadtree_cull.glsl"));
	program = program_create(args, false);
	if(!program) {
		fprintf(stderr, "log: %s\n", program_get_log());
		return GL_FALSE;
	}
	glTransformFeedbackVaryings(program, 1, varyings, GL_SEPARATE_ATTRIBS);
	glLinkProgram(program);
	g_programs[PROGRAM_CULL] = program;
	return GL_TRUE;
}

// ---------------------------------------------------------
// terrain program
static GLuint load_render_program() {
	struct program_args_t *args = program_args_create();
	GLuint program = 0;

	if(!g_wireframe)
		program_args_forbid_stage(args, GL_GEOMETRY_SHADER);
	program_args_set_version(args, 420, false);
	program_args_push_file(args, SHADER_PATH("ltree.glsl"));
#if TERRAIN_RENDERER
	program_args_push_string(args, "#define TERRAIN_RENDERER 1\n");
	program_args_push_file(args, SHADER_PATH("terrain.glsl"));
#elif PARAMETRIC_RENDERER
	program_args_push_string(args, "#define PARAMETRIC_RENDERER 1\n");
	program_args_push_file(args, SHADER_PATH("parametrics.glsl"));
#endif
	program_args_push_file(args, SHADER_PATH("quadtree_render.glsl"));
	program = program_create(args, true);
	if(!program) {
		fprintf(stderr, "log: %s\n", program_get_log());
		return GL_FALSE;
	}
	g_programs[PROGRAM_RENDER] = program;
	return GL_TRUE;
}

#if TERRAIN_RENDERER
// ---------------------------------------------------------
// heightfield texture
#define IMAGE_PATH(x) "../../resources/" x
static GLboolean load_puget_texture() {
	const char filename[] = IMAGE_PATH("puget.png");
	struct image_t *image = image_load_png (filename);
	GLenum iformat;

	if (!image) return GL_FALSE;

	iformat = image_get_gl_type (image) == GL_UNSIGNED_BYTE ? GL_R8 : GL_R16;
	glActiveTexture(GL_TEXTURE0 + TEXTURE_PUGET);
	glBindTexture(GL_TEXTURE_2D, g_textures[TEXTURE_PUGET]);
		image_glTexImage2D (GL_TEXTURE_2D, 0, iformat, 1, 1, image);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, 16.f);

	return GL_TRUE;
}
#endif // TERRAIN_RENDERER

// ---------------------------------------------------------
// grid mesh
static GLboolean load_grid_mesh() {
#if 1
	size_t vertices_byte_size = sizeof(vec2_t) * sqrf(g_grid_tess_factor);
	size_t indexes_byte_size = sizeof(uint16_t) * sqrf(g_grid_tess_factor - 1) * 4;
	vec2_t *vertices = malloc(vertices_byte_size);
	uint16_t *indexes = malloc(indexes_byte_size);
	int i, j;

	assert(vertices && indexes);
	for(i = 0; i < g_grid_tess_factor; ++i)
		for(j = 0; j < g_grid_tess_factor; ++j) {
			vec2_t *vertex = vertices + i * g_grid_tess_factor + j;

			(*vertex)[0] = (float)i / (g_grid_tess_factor - 1) - 0.5f;
			(*vertex)[1] = (float)j / (g_grid_tess_factor - 1) - 0.5f;
		}
	for(i = 0; i < g_grid_tess_factor - 1; ++i)
		for(j = 0; j < g_grid_tess_factor - 1; ++j) {
			uint16_t *index = indexes + 4 * i * (g_grid_tess_factor - 1) + 4 * j;

			index[0] = i     + g_grid_tess_factor *  j;
			index[1] = i + 1 + g_grid_tess_factor *  j;
			index[2] = i + 1 + g_grid_tess_factor * (j + 1);
			index[3] = i     + g_grid_tess_factor * (j + 1);
		}
#else
	int w = 1 * g_grid_tess_factor;
	int h = 4 * g_grid_tess_factor;
	size_t vertices_byte_size = sizeof(vec2_t) * w * h;
	size_t indexes_byte_size = sizeof(uint16_t) * (w-1) * (h-1) * 4;
	vec2_t *vertices = malloc(vertices_byte_size);
	uint16_t *indexes = malloc(indexes_byte_size);
	int i, j;

	assert(vertices && indexes);
	for(i = 0; i < w; ++i)
		for(j = 0; j < h; ++j) {
			vec2_t *vertex = vertices + i * h + j;

			(*vertex)[0] = (float)i / (w - 1) - 0.5f;
			(*vertex)[1] = (float)j / (h - 1) - 0.5f;
		}
	for(i = 0; i < h - 1; ++i)
		for(j = 0; j < w - 1; ++j) {
			uint16_t *index = indexes + 4 * i * (w - 1) + 4 * j;

			index[0] = i     + h *  j;
			index[1] = i + 1 + h *  j;
			index[2] = i + 1 + h * (j + 1);
			index[3] = i     + h * (j + 1);
		}
#endif

	// upload to GPU
	glBindBuffer(GL_ARRAY_BUFFER, g_buffers[BUFFER_GRID_VERTICES]);
		glBufferData(GL_ARRAY_BUFFER, vertices_byte_size, vertices, GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_buffers[BUFFER_GRID_INDEXES]);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, indexes_byte_size, indexes, GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	g_patch_index_count = indexes_byte_size / sizeof(uint16_t);

	free(vertices);
	free(indexes);
	return GL_TRUE;
}

// ---------------------------------------------------------
// load textures
static GLboolean load_textures() {
	GLboolean load_ok = GL_TRUE;

#if TERRAIN_RENDERER
	load_ok &= load_puget_texture();
#endif

	return load_ok;
}

// ---------------------------------------------------------
// load programs
static GLboolean load_programs() {
	static GLboolean first = GL_TRUE;
	GLboolean load_ok = GL_TRUE;
	GLint i = 0;

	for(;i<PROGRAM_COUNT && !first; ++i)
		glDeleteProgram(g_programs[i]);
	first = GL_FALSE;

	// create programs
	load_ok &= load_lod_program();
	load_ok &= load_cull_program();
	load_ok &= load_render_program();
	if(!load_ok)
		return GL_FALSE;

	// get locations
	g_uniform_locations[LOCATION_LOD_SCENE_SIZE] = 
		glGetUniformLocation(g_programs[PROGRAM_LOD],
		                     "u_scene_size");
	g_uniform_locations[LOCATION_CULL_SCENE_SIZE] = 
		glGetUniformLocation(g_programs[PROGRAM_CULL],
		                     "u_scene_size");
	g_uniform_locations[LOCATION_RENDER_SCENE_SIZE] = 
		glGetUniformLocation(g_programs[PROGRAM_RENDER],
		                     "u_scene_size");
	g_uniform_locations[LOCATION_RENDER_MVP] = 
		glGetUniformLocation(g_programs[PROGRAM_RENDER],
		                     "u_mvp");
	g_uniform_locations[LOCATION_RENDER_EYE_POS] = 
		glGetUniformLocation(g_programs[PROGRAM_RENDER],
		                     "u_eye_pos");
	g_uniform_locations[LOCATION_LOD_EYE_POS] = 
		glGetUniformLocation(g_programs[PROGRAM_LOD],
		                     "u_eye_pos");
	g_uniform_locations[LOCATION_RENDER_GPU_TESS_FACTOR] = 
		glGetUniformLocation(g_programs[PROGRAM_RENDER],
		                     "u_gpu_tess_factor");
	g_uniform_locations[LOCATION_RENDER_GRID_TESS_FACTOR] = 
		glGetUniformLocation(g_programs[PROGRAM_RENDER],
		                     "u_grid_tess_factor");
#if TERRAIN_RENDERER
	g_uniform_locations[LOCATION_LOD_PUGET_SAMPLER] = 
		glGetUniformLocation(g_programs[PROGRAM_LOD],
		                     "u_puget_sampler");
	g_uniform_locations[LOCATION_RENDER_PUGET_SAMPLER] = 
		glGetUniformLocation(g_programs[PROGRAM_RENDER],
		                     "u_puget_sampler");
#endif

	// set uniforms
	glProgramUniform1f(g_programs[PROGRAM_LOD],
	                   g_uniform_locations[LOCATION_LOD_SCENE_SIZE],
	                   g_scene_size);
	glProgramUniform1f(g_programs[PROGRAM_CULL],
	                   g_uniform_locations[LOCATION_CULL_SCENE_SIZE],
	                   g_scene_size);
	glProgramUniform1f(g_programs[PROGRAM_RENDER],
	                   g_uniform_locations[LOCATION_RENDER_SCENE_SIZE],
	                   g_scene_size);
	glProgramUniform1f(g_programs[PROGRAM_RENDER],
	                   g_uniform_locations[LOCATION_RENDER_GPU_TESS_FACTOR],
	                   g_gpu_tess_factor);
	glProgramUniform2f(g_programs[PROGRAM_RENDER],
	                   g_uniform_locations[LOCATION_RENDER_GRID_TESS_FACTOR],
	                   g_grid_tess_factor, g_grid_tess_factor);
#if TERRAIN_RENDERER
	glProgramUniform1i(g_programs[PROGRAM_RENDER],
	                   g_uniform_locations[LOCATION_RENDER_PUGET_SAMPLER],
	                   TEXTURE_PUGET);
	glProgramUniform1i(g_programs[PROGRAM_LOD],
	                   g_uniform_locations[LOCATION_LOD_PUGET_SAMPLER],
	                   TEXTURE_PUGET);
#endif

	return GL_TRUE;
}

// ---------------------------------------------------------
// load vertex arrays
static GLboolean load_vertex_arrays() {
	glBindVertexArray(g_vertex_arrays[VERTEX_ARRAY_QUADTREE1]);
		glEnableVertexAttribArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, g_buffers[BUFFER_PATCH_DATA1]);
		glVertexAttribIPointer(0, 4, GL_UNSIGNED_INT, 0, 0);
	glBindVertexArray(g_vertex_arrays[VERTEX_ARRAY_QUADTREE2]);
		glEnableVertexAttribArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, g_buffers[BUFFER_PATCH_DATA2]);
		glVertexAttribIPointer(0, 4, GL_UNSIGNED_INT, 0, 0);
	glBindVertexArray(g_vertex_arrays[VERTEX_ARRAY_TERRAIN1]);
		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);
		glBindBuffer(GL_ARRAY_BUFFER, g_buffers[BUFFER_PATCH_DATA1]);
		glVertexAttribPointer(0, 4, GL_FLOAT, 0, 0, 0);
		glVertexAttribDivisor(0, 1);
		glBindBuffer(GL_ARRAY_BUFFER, g_buffers[BUFFER_GRID_VERTICES]);
		glVertexAttribPointer(1, 2, GL_FLOAT, 0, 0, 0);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_buffers[BUFFER_GRID_INDEXES]);
	glBindVertexArray(g_vertex_arrays[VERTEX_ARRAY_TERRAIN2]);
		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);
		glBindBuffer(GL_ARRAY_BUFFER, g_buffers[BUFFER_PATCH_DATA2]);
		glVertexAttribPointer(0, 4, GL_FLOAT, 0, 0, 0);
		glVertexAttribDivisor(0, 1);
		glBindBuffer(GL_ARRAY_BUFFER, g_buffers[BUFFER_GRID_VERTICES]);
		glVertexAttribPointer(1, 2, GL_FLOAT, 0, 0, 0);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_buffers[BUFFER_GRID_INDEXES]);
	glBindVertexArray(g_vertex_arrays[VERTEX_ARRAY_EMPTY]);
	glBindVertexArray(0);
	return GL_TRUE;
}

// ---------------------------------------------------------
// load transform feedbacks
static GLboolean load_feedbacks() {
	glBindTransformFeedback(GL_TRANSFORM_FEEDBACK,
	                        g_feedbacks[FEEDBACK_QUADTREE1]);
		glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER,
		                 0u,
		                 g_buffers[BUFFER_PATCH_DATA2]);
	glBindTransformFeedback(GL_TRANSFORM_FEEDBACK,
	                        g_feedbacks[FEEDBACK_QUADTREE2]);
		glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER,
		                 0u,
		                 g_buffers[BUFFER_PATCH_DATA1]);
	glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);
	return GL_TRUE;
}

// ---------------------------------------------------------
// load buffers
static GLboolean load_buffers() {
	const GLint dummy[] = {0,0,0,0};
	glBindBuffer(GL_ARRAY_BUFFER, g_buffers[BUFFER_PATCH_DATA1]);
		glBufferData(GL_ARRAY_BUFFER, 
		             sizeof(GLuint) << 16,
		             NULL,
		             GL_STATIC_DRAW);
		glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(dummy), dummy);
	glBindBuffer(GL_ARRAY_BUFFER, g_buffers[BUFFER_PATCH_DATA2]);
		glBufferData(GL_ARRAY_BUFFER,
		             sizeof(GLuint) << 16,
		             NULL,
		             GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_UNIFORM_BUFFER, g_buffers[BUFFER_FRUSTUM]);
		glBufferData(GL_UNIFORM_BUFFER,
		             sizeof(vec4_t) * 6,
		             NULL,
		             GL_STATIC_DRAW);
/*	glBindBuffer(GL_UNIFORM_BUFFER, 0);*/

	// bind primcount to atomic in shaders
#if 0
	glBindBufferRange(GL_ATOMIC_COUNTER_BUFFER,
	                  0,
	                  g_buffers[BUFFER_INDIRECT],
	                  0,
	                  sizeof(GLint)*5);
#endif
	load_grid_mesh();
	glBindBufferBase(GL_UNIFORM_BUFFER,
	                 0,
	                 g_buffers[BUFFER_FRUSTUM]);
	return GL_TRUE;
}

// ---------------------------------------------------------
static GLvoid lt_first_pass() {
	// pre process quadtree
	glEnable(GL_RASTERIZER_DISCARD);
	glUseProgram(g_programs[PROGRAM_LOD]);
	glUniform3f(g_uniform_locations[LOCATION_LOD_EYE_POS],
	            0, 0, 0);
	glBindTransformFeedback(GL_TRANSFORM_FEEDBACK,
	                        g_feedbacks[FEEDBACK_QUADTREE1]);
	glBeginTransformFeedback(GL_POINTS);
		glBindVertexArray(g_vertex_arrays[VERTEX_ARRAY_QUADTREE1]);
		glDrawArrays(GL_POINTS, 0, 1);
	glEndTransformFeedback();
	glDisable(GL_RASTERIZER_DISCARD);

	glFinish(); // for AMD only ?
}

#ifndef NDEBUG
// ---------------------------------------------------------
// debug output callback
static void gl_debug_message_callback(GLenum source,
                                      GLenum type,
                                      GLuint id,
                                      GLenum severity,
                                      GLsizei length,
                                      const GLchar* message,
                                      GLvoid* userParam ) {
	fprintf(stderr, "GL_ARB_debug_output:\n%s\n",message);
}
#endif // NDEBUG

// ---------------------------------------------------------
// mouse motion
static void on_mouse_motion(GLFWwindow* window, GLdouble x, GLdouble y) {
	static GLdouble xlast = 0;
	static GLdouble ylast = 0;
	GLdouble xrel = x - xlast;
	GLdouble yrel = y - ylast;
	if(g_eye_lock) { // camera
		affine_rotatex_local(g_eye, -yrel * 0.01f);
		affine_rotatey_world(g_eye, -xrel * 0.01f);
	}
	xlast = x;
	ylast = y;
}

// ---------------------------------------------------------
// mouse button
static void on_mouse_button(GLFWwindow* window, int button, int action, int mods) {
	g_eye_lock = (button == GLFW_MOUSE_BUTTON_LEFT);
	g_eye_lock&= (action == GLFW_PRESS);
}

// ---------------------------------------------------------
// set scene
static void set_eye() {
#if PARAMETRIC_RENDERER
	vec3_t t = {0, 0, g_scene_size*2.f};
#else
	vec3_t t = {0, 4800.5f, 0};
#endif

	affine_identity(g_eye);
	affine_translate_local(g_eye, t);
#if !defined(PARAMETRIC_RENDERER) && ! defined(TERRAIN_RENDERER)
	affine_rotatex_local(g_eye, -pi * 0.5f);
#endif
}


// ---------------------------------------------------------
// init
static void init() {
	GLint i;

#ifndef NDEBUG
	// setup debug output
	glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS_ARB);
	glDebugMessageCallbackARB(
		(GLDEBUGPROCARB) &gl_debug_message_callback, NULL);
#endif

	// set up camera
	g_eye = affine_create();
	set_eye();

	// gen objects
	glGenBuffers(BUFFER_COUNT, g_buffers);
	g_init_ok = load_buffers();

	// vertex arrays
	glGenVertexArrays(VERTEX_ARRAY_COUNT, g_vertex_arrays);
	g_init_ok &= load_vertex_arrays();

	// transform feedbacks
	glGenTransformFeedbacks(FEEDBACK_COUNT, g_feedbacks);
	g_init_ok &= load_feedbacks();

	// load textures
	glGenTextures(TEXTURE_COUNT, g_textures);
	g_init_ok &= load_textures();

	// load programs
	g_init_ok &= load_programs();

	// set up font renderer
	g_init_ok &= ft_init(GL_TEXTURE0+TEXTURE_COUNT) == FT_SUCCESS;

	// OpenGL context flags
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glPatchParameteri(GL_PATCH_VERTICES, 4);

	// gen timers
	for(i=0; i<TIMER_COUNT; ++i)
		g_timers[i] = timer_create();

	// prepare quadtree
	lt_first_pass();

	glGenQueries(1, &g_query);
}

// ---------------------------------------------------------
// load programs
static GLvoid handle_key(GLFWwindow* window, int key, int scancode, int action, int mods) {
	switch (key) {
	case GLFW_KEY_R:
		if (action == GLFW_PRESS) {
			fprintf(stderr, "reloading programs..\n");
			g_init_ok = load_programs();
		}
		break;
	case GLFW_KEY_F:
		if (action == GLFW_PRESS)
			g_eye_freeze = !g_eye_freeze;
		break;
	case GLFW_KEY_G:
		if (action == GLFW_PRESS)
			g_show_gui = !g_show_gui;
		break;
	case GLFW_KEY_E:
		if (action == GLFW_PRESS) {
			g_wireframe = !g_wireframe;
			g_init_ok = load_programs();
		}
		break;
	case GLFW_KEY_W:
		if (action == GLFW_PRESS)
			g_eye_velocity[2]-= camera_velocity * (g_eye_boost ? 1e3 : 1.f);
		else if (action == GLFW_RELEASE)
			g_eye_velocity[2]+= camera_velocity * (g_eye_boost ? 1e3 : 1.f);
		break;
	case GLFW_KEY_S:
		if (action == GLFW_PRESS)
			g_eye_velocity[2]+= camera_velocity * (g_eye_boost ? 1e3 : 1.f);
		else if (action == GLFW_RELEASE)
			g_eye_velocity[2]-= camera_velocity * (g_eye_boost ? 1e3 : 1.f);
		break;
	case GLFW_KEY_A:
		if (action==GLFW_PRESS)
			g_eye_velocity[0]-= camera_velocity * (g_eye_boost ? 1e3 : 1.f);
		else if (action == GLFW_RELEASE)
			g_eye_velocity[0]+= camera_velocity * (g_eye_boost ? 1e3 : 1.f);
		break;
	case GLFW_KEY_D:
		if (action == GLFW_PRESS)
			g_eye_velocity[0]+= camera_velocity * (g_eye_boost ? 1e3 : 1.f);
		else if (action == GLFW_RELEASE)
			g_eye_velocity[0]-= camera_velocity * (g_eye_boost ? 1e3 : 1.f);
		break;
	case GLFW_KEY_P:
		if (action == GLFW_PRESS)
			image_save_gl_front_buffer();
		break;
	case GLFW_KEY_LEFT_SHIFT:
		if (action == GLFW_PRESS) {
			vec3_t boost_val = {1e3, 1e3, 1e3};
			vec3_mul (g_eye_velocity, g_eye_velocity, boost_val);
		} // key press
		else
		if (action == GLFW_RELEASE) {
			vec3_t boost_val = {1e-3, 1e-3, 1e-3};
			vec3_mul (g_eye_velocity, g_eye_velocity, boost_val);
		} // key release
		g_eye_boost = (action != GLFW_RELEASE);
		break;
	case GLFW_KEY_LEFT:
		if (action == GLFW_PRESS) {
			--g_grid_tess_factor;
			if (g_grid_tess_factor < 2) g_grid_tess_factor = 2;
			g_init_ok = load_grid_mesh();
			g_init_ok&= load_vertex_arrays();
			glProgramUniform2f (g_programs[PROGRAM_RENDER],
			                    g_uniform_locations[LOCATION_RENDER_GRID_TESS_FACTOR],
			                    g_grid_tess_factor, g_grid_tess_factor);
		}
		break;
	case GLFW_KEY_RIGHT:
		if (action == GLFW_PRESS) {
			++g_grid_tess_factor;
			if (g_grid_tess_factor > 256) g_grid_tess_factor = 256;
			g_init_ok = load_grid_mesh();
			g_init_ok&= load_vertex_arrays();
			glProgramUniform2f (g_programs[PROGRAM_RENDER],
			                    g_uniform_locations[LOCATION_RENDER_GRID_TESS_FACTOR],
			                    g_grid_tess_factor, g_grid_tess_factor);
		}
		break;
	case GLFW_KEY_UP:
		if (action == GLFW_PRESS) {
			++g_gpu_tess_factor;
			if (g_gpu_tess_factor > 5) g_gpu_tess_factor = 5;
			glProgramUniform1f (g_programs[PROGRAM_RENDER],
			                    g_uniform_locations[LOCATION_RENDER_GPU_TESS_FACTOR],
			                    g_gpu_tess_factor);
		}
		break;
	case GLFW_KEY_DOWN:
		if (action == GLFW_PRESS) {
			--g_gpu_tess_factor;
			if (g_gpu_tess_factor < 0) g_gpu_tess_factor = 0;
			glProgramUniform1f (g_programs[PROGRAM_RENDER],
			                    g_uniform_locations[LOCATION_RENDER_GPU_TESS_FACTOR],
			                    g_gpu_tess_factor);
		}
		break;
	case GLFW_KEY_ESCAPE:
		glfwSetWindowShouldClose (window, GL_TRUE);
		break;
	default:
		break;
	}
}


// ---------------------------------------------------------
// update framebuffer
static void draw() {
	mat4_t mvp, view;
	vec3_t translation;
	vec3_t eye_pos;

	if(!g_init_ok) {
		glClearColor(1,0,0,1);
		glClear(GL_COLOR_BUFFER_BIT);
		return;
	}

	affine_get_position(g_eye, eye_pos);
#if PARAMETRIC_RENDERER
	mat4_perspective(mvp, fovy, screen_ratio, 
	                 0.01f * (1.f+vec3_length(eye_pos)),
	                 200.f * (1.f+vec3_length(eye_pos)));
#else
	mat4_perspective(mvp, fovy, screen_ratio, 
	                 0.01f * (1.f+abs(eye_pos[1])),
	                 200.f * (1.f+abs(eye_pos[1])));
#endif

	affine_inverse_matrix(g_eye, view);
	mat4_mul(mvp, mvp, view);

	// update uniforms
	{
		vec4_t frustum[6];
		glProgramUniformMatrix4fv(g_programs[PROGRAM_RENDER],
		                          g_uniform_locations[LOCATION_RENDER_MVP],
		                          1, GL_FALSE, mvp);
		frustum_extract_mvp(frustum, mvp);
		glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(vec4_t)*6, frustum);
	}

	glClearColor(0.0, 0.0, 0.0, 0.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// update quadtree
	static GLuint queryResult = 0;
	if(!g_eye_freeze) {
		timer_start(g_timers[TIMER_LOD]);
		glProgramUniform3fv(g_programs[PROGRAM_LOD],
		                    g_uniform_locations[LOCATION_LOD_EYE_POS],
		                    1, eye_pos);
		glProgramUniform3fv(g_programs[PROGRAM_RENDER],
		                    g_uniform_locations[LOCATION_RENDER_EYE_POS],
		                    1, eye_pos);

		glEnable(GL_RASTERIZER_DISCARD);

		glUseProgram(g_programs[PROGRAM_LOD]);
		glBindTransformFeedback(GL_TRANSFORM_FEEDBACK,
		                        g_feedbacks[FEEDBACK_QUADTREE1 + g_pingpong]);
		glBeginTransformFeedback(GL_POINTS);
			glBindVertexArray(g_vertex_arrays[VERTEX_ARRAY_QUADTREE1 + g_pingpong]);
			glDrawTransformFeedback(GL_POINTS,
			                        g_feedbacks[FEEDBACK_QUADTREE2 - g_pingpong]);
		glEndTransformFeedback();
		timer_stop(g_timers[TIMER_LOD]);

		timer_start(g_timers[TIMER_CULL]);
		glUseProgram(g_programs[PROGRAM_CULL]);
		glBindTransformFeedback(GL_TRANSFORM_FEEDBACK,
		                        g_feedbacks[FEEDBACK_QUADTREE2 - g_pingpong]);
		glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, g_query);
		glBeginTransformFeedback(GL_POINTS);
			glBindVertexArray(g_vertex_arrays[VERTEX_ARRAY_QUADTREE2 - g_pingpong]);
			glDrawTransformFeedback(GL_POINTS,
			                        g_feedbacks[FEEDBACK_QUADTREE1 + g_pingpong]);
		glEndTransformFeedback();
		glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);
		glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);
		timer_stop(g_timers[TIMER_CULL]);

		glDisable(GL_RASTERIZER_DISCARD);

		glGetQueryObjectuiv(g_query, GL_QUERY_RESULT, &queryResult);
		g_pingpong = 1 - g_pingpong;
	}

#if 1
	// temporary stuff, will be removed for drawIndirect
	if(queryResult > 0) {
		timer_start(g_timers[TIMER_RENDER]);
		glUseProgram(g_programs[PROGRAM_RENDER]);
		glBindVertexArray(g_vertex_arrays[VERTEX_ARRAY_TERRAIN2 - g_pingpong]);
			glDrawElementsInstanced(GL_PATCHES, g_patch_index_count, GL_UNSIGNED_SHORT, NULL, queryResult);
		timer_stop(g_timers[TIMER_RENDER]);
#if 0
		glBindVertexArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, g_buffers[BUFFER_PATCH_DATA1+g_pingpong]);
		GLuint *p = glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);
		if(p) {
			int i=0;
			for(;i<queryResult;++i)
				printf("%i ", (p[i*4]>>4u));
			printf("\n");
			glUnmapBuffer(GL_ARRAY_BUFFER);
		}
#endif
	}
#endif

	if(g_show_gui) {
		GLdouble lod_cpu_ticks, lod_gpu_ticks;
		GLdouble cull_cpu_ticks, cull_gpu_ticks;
		GLdouble render_cpu_ticks, render_gpu_ticks;

		timer_ticks(g_timers[TIMER_LOD], &lod_cpu_ticks, &lod_gpu_ticks);
		timer_ticks(g_timers[TIMER_CULL], &cull_cpu_ticks, &cull_gpu_ticks);
		timer_ticks(g_timers[TIMER_RENDER], &render_cpu_ticks, &render_gpu_ticks);
/*		glColorMask(1,0,0,1);*/
		ft_print(FT_FONT_SIZE_SMALL, 10, 10,
		         "LOD :   CPU %.3fms GPU %.3fms\n"
		         "CULL:   CPU %.3fms GPU %.3fms\n"
		         "RENDER: CPU %.3fms GPU %.3fms\n"
		         "ALL:    CPU %.3fms GPU %.3fms\n\n"
		         "terrain size (km) %.1f\n"
		         "eye altitude (m)  %.3f\n"
		         "znear %.3f zfar %.3f\n\n"
		         "{f} freeze viewpoint: %s\n"
		         "{e} wireframe: %s\n"
		         "{left/right} patch tess: %i\n"
		         "{ up /down } GPU tess  : %i (x%i)\n"
		         "{lshift} boost",
		         lod_cpu_ticks * 1e3, lod_gpu_ticks * 1e3,
		         cull_cpu_ticks * 1e3, cull_gpu_ticks * 1e3,
		         render_cpu_ticks * 1e3, render_gpu_ticks * 1e3,
		         (lod_cpu_ticks + cull_cpu_ticks + render_cpu_ticks) * 1e3,
		         (lod_gpu_ticks + cull_gpu_ticks + render_gpu_ticks) * 1e3,
		         g_scene_size*1e-3f,
		         eye_pos[1],
		         0.01f * (1.f+abs(eye_pos[1])),
		         1000.f * (1.f+abs(eye_pos[1])),
		         g_eye_freeze ? "ON":"OFF",
		         g_wireframe ? "ON":"OFF",
		         g_grid_tess_factor,
		         g_gpu_tess_factor,
		         1 << g_gpu_tess_factor);
/*		glColorMask(1,1,1,1);*/
	}
#if 0 // benchmarking
	{
		const GLdouble frame_bench_count = 1e3;
		static GLdouble frame_count = 0;
		static GLdouble lod_cpu_ticks = 0.0, lod_gpu_ticks = 0.0;
		static GLdouble cull_cpu_ticks = 0.0, cull_gpu_ticks = 0.0;
		static GLdouble render_cpu_ticks = 0.0, render_gpu_ticks = 0.0;
		static GLdouble lod_cpu_ticks2 = 0.0, lod_gpu_ticks2 = 0.0;
		static GLdouble cull_cpu_ticks2 = 0.0, cull_gpu_ticks2 = 0.0;
		static GLdouble render_cpu_ticks2 = 0.0, render_gpu_ticks2 = 0.0;
		GLdouble cpu_ticks, gpu_ticks;

		timer_ticks(g_timers[TIMER_LOD], &cpu_ticks, &gpu_ticks);
		lod_cpu_ticks+= cpu_ticks * 1e3;
		lod_gpu_ticks+= gpu_ticks * 1e3;
		lod_cpu_ticks2+= cpu_ticks * cpu_ticks * 1e6;
		lod_gpu_ticks2+= gpu_ticks * gpu_ticks * 1e6;

		timer_ticks(g_timers[TIMER_CULL], &cpu_ticks, &gpu_ticks);
		cull_cpu_ticks+= cpu_ticks * 1e3;
		cull_gpu_ticks+= gpu_ticks * 1e3;
		cull_cpu_ticks2+= cpu_ticks * cpu_ticks * 1e6;
		cull_gpu_ticks2+= gpu_ticks * gpu_ticks * 1e6;

		timer_ticks(g_timers[TIMER_RENDER], &cpu_ticks, &gpu_ticks);
		render_cpu_ticks+= cpu_ticks * 1e3;
		render_gpu_ticks+= gpu_ticks * 1e3;
		render_cpu_ticks2+= cpu_ticks * cpu_ticks * 1e6;
		render_gpu_ticks2+= gpu_ticks * gpu_ticks * 1e6;

		++frame_count;
		if(frame_count >= frame_bench_count) {
			// average timings
			lod_cpu_ticks/= frame_count;
			lod_gpu_ticks/= frame_count;
			lod_cpu_ticks2/= frame_count;
			lod_gpu_ticks2/= frame_count;
			cull_cpu_ticks/= frame_count;
			cull_gpu_ticks/= frame_count;
			cull_cpu_ticks2/= frame_count;
			cull_gpu_ticks2/= frame_count;
			render_cpu_ticks/= frame_count;
			render_gpu_ticks/= frame_count;
			render_cpu_ticks2/= frame_count;
			render_gpu_ticks2/= frame_count;

			// print mean and stdev timings for each kernel
			fprintf(stderr, "lod_cpu:  %.3fms; stdev: %.3f\n"
			                "lod_gpu:  %.3fms; stdev: %.3f\n"
			                "cull_cpu: %.3fms; stdev: %.3f\n"
			                "cull_gpu: %.3fms; stdev: %.3f\n"
			                "render_cpu: %.3fms; stdev: %.3f\n"
			                "render_gpu: %.3fms; stdev: %.3f\n",
			         lod_cpu_ticks, sqrt(lod_cpu_ticks2 - lod_cpu_ticks*lod_cpu_ticks),
			         lod_gpu_ticks, sqrt(lod_gpu_ticks2 - lod_gpu_ticks*lod_gpu_ticks),
			         cull_cpu_ticks, sqrt(cull_cpu_ticks2 - cull_cpu_ticks*cull_cpu_ticks),
			         cull_gpu_ticks, sqrt(cull_gpu_ticks2 - cull_gpu_ticks*cull_gpu_ticks),
			         render_cpu_ticks, sqrt(render_cpu_ticks2 - render_cpu_ticks*render_cpu_ticks),
			         render_gpu_ticks, sqrt(render_gpu_ticks2 - render_gpu_ticks*render_gpu_ticks)
			         );
			
			// reset counters
			frame_count = 
			lod_cpu_ticks = 
			lod_gpu_ticks =
			lod_cpu_ticks2 = 
			lod_gpu_ticks2 = 
			cull_cpu_ticks = 
			cull_gpu_ticks = 
			cull_cpu_ticks2 = 
			cull_gpu_ticks2 = 
			render_cpu_ticks = 
			render_gpu_ticks = 
			render_cpu_ticks2 = 
			render_gpu_ticks2 = 0.0;
		}
	}
#endif
	// go back to default vertex array
	glBindVertexArray(0);

	// move camera
	vec3_cpy(translation, g_eye_velocity);
	translation[0]*= 0.01f;
	translation[1]*= 0.01f;
	translation[2]*= 0.01f;
	affine_translate_local(g_eye, translation);
}


// ---------------------------------------------------------
// cleanup 
static void cleanup(GLFWwindow *window) {
	GLint i=0;
	for(;i<PROGRAM_COUNT;++i)
		glDeleteProgram(g_programs[i]);
	for(i=0; i<TIMER_COUNT; ++i)
		timer_release(g_timers[i]);
	glDeleteVertexArrays(VERTEX_ARRAY_COUNT, g_vertex_arrays);
	glDeleteTransformFeedbacks(FEEDBACK_COUNT, g_feedbacks);
	glDeleteBuffers(BUFFER_COUNT, g_buffers);
	glDeleteBuffers(TEXTURE_COUNT, g_textures);
	glDeleteQueries(1, &g_query);

	ft_shutdown();

	affine_release(g_eye);
}


#ifdef _WIN32
// ---------------------------------------------------------
// ask user to press the enter/return key
static void press_enter() {
	fprintf(stderr, "Press enter to continue...");
	getchar();
	fprintf(stderr, "Done.\n");
}
#endif


// ---------------------------------------------------------
// main
int main (int argc, char **argv) {
	GLFWwindow* window;

#ifdef _WIN32
	atexit (&press_enter); // prevents immediate console termination
#endif
	if(!glfwInit ()) {
		fprintf (stderr, "glfwInit failed\n");
		exit (EXIT_FAILURE);
	}

	glfwWindowHint (GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint (GLFW_CONTEXT_VERSION_MINOR, 2);
	glfwWindowHint (GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifndef NDEBUG
	glfwWindowHint (GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);
#endif
	window = glfwCreateWindow (WINDOW_WIDTH, WINDOW_HEIGHT, "OpenGL", NULL, NULL);

	if(!window) {
		fprintf (stderr, "glfwCreateWindow failed\n");
		glfwTerminate ();
		exit (EXIT_FAILURE);
	}

	glfwSetKeyCallback (window, &handle_key);
	glfwSetCursorPosCallback (window, &on_mouse_motion);
	glfwSetMouseButtonCallback (window, &on_mouse_button);
	glfwSetWindowCloseCallback (window, &cleanup);

	glfwMakeContextCurrent (window);
	if(ogl_LoadFunctions() != ogl_LOAD_SUCCEEDED) {
		fprintf(stderr, "ogl_LoadFunctions failed\n");
		glfwTerminate();
		exit(EXIT_FAILURE);
	}
	init ();
	while (!glfwWindowShouldClose (window)) {
		draw ();
#ifdef _WIN32
		Sleep (1);
#else
		usleep (1000);
#endif
/*		glfwSleep (1e-2);*/
		glfwSwapBuffers (window);
		glfwPollEvents ();
	}
	glfwTerminate ();
	return EXIT_SUCCESS;
}

