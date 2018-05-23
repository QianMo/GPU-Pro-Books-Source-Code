/*

Sylvain Lefebvre, Samuel Hornus - 2013

This file describes all base functions of a typical A-buffer technique.
It is used as a virtual interface to create different DLLs implementing
an A-buffer.

*/

#pragma once

typedef struct {
  bool  overflow;
  float tm_clear;
  float tm_build;
  float tm_render;
} t_abuffer_frame_status;

// Initialize the A-buffer
typedef void (*f_abuffer_init)(int screen_sz,int num_records,float znear,float zfar,bool interruptable,bool use_timers);
// Terminate the A-buffer
typedef void (*f_abuffer_terminate)();
// Increase the A-buffer storage by some factor
typedef void (*f_abuffer_change_size)(float factor);
// Setup the perspective matrix
typedef void (*f_abuffer_set_perspective)(const float *m);
// Setup the view matrix
typedef void (*f_abuffer_set_view)(const float *m);
// Setup the light position
typedef void (*f_abuffer_set_lightpos)(const float *d);
// Start a new frame
typedef void (*f_abuffer_frame_begin)(float r,float g,float b);
// Start drawing a new object with color RGB and transparency A (all in [0,1])
typedef void (*f_abuffer_begin)(float r,float g,float b,float a);
// Stop drawing the object
typedef void (*f_abuffer_end)();
// Setup object model matrix
typedef void (*f_abuffer_set_model_matrix)( const float *modelMatrix );
// Stop the frame (triggers final rendering pass)
typedef void (*f_abuffer_frame_end)(t_abuffer_frame_status *status);
// Printout implementation-dependent statistics
typedef void (*f_abuffer_print_stats)(int *byteSize,float *loadFactor);
// Adjust the number of required records depending on implementation
typedef int  (*f_abuffer_compute_num_records)(int screen_sz,int num_required_records);
// Provide custom rendering code for each fragments color
// available inputs are 'nrm' (vec3,normal), 'uv' (vec2,texcoord0) and output must be in 'clr' (vec4,rgba)
typedef void (*f_abuffer_set_custom_fragment_code)(const char *glsl_code);

#ifndef ABUFFER_NO_EXTERN
extern f_abuffer_init abuffer_init;
extern f_abuffer_terminate abuffer_terminate;
extern f_abuffer_change_size abuffer_change_size;
extern f_abuffer_set_perspective abuffer_set_perspective;
extern f_abuffer_set_view abuffer_set_view;
extern f_abuffer_set_lightpos abuffer_set_lightpos;
extern f_abuffer_frame_begin abuffer_frame_begin;
extern f_abuffer_begin abuffer_begin;
extern f_abuffer_end abuffer_end;
extern f_abuffer_set_model_matrix abuffer_set_model_matrix;
extern f_abuffer_frame_end abuffer_frame_end;
extern f_abuffer_print_stats abuffer_print_stats;
extern f_abuffer_compute_num_records abuffer_compute_num_records;
extern f_abuffer_set_custom_fragment_code abuffer_set_custom_fragment_code;
#endif

// Loads a DLL implementing all functions. 
// - 'fname' is the filename of the DLL
void abuffer_load_dll(const char *fname);
