// --------------------------------------------------------------------

#include <Windows.h>
#include <LibSL/LibSL.h>

// --------------------------------------------------------------------

#define ABUFFER_NO_EXTERN
#include "abuffer.h"

// --------------------------------------------------------------------

f_abuffer_init abuffer_init                                         = NULL;
f_abuffer_terminate abuffer_terminate                               = NULL;
f_abuffer_change_size abuffer_change_size                           = NULL;
f_abuffer_set_perspective abuffer_set_perspective                   = NULL;
f_abuffer_set_view abuffer_set_view                                 = NULL;
f_abuffer_set_lightpos abuffer_set_lightpos                         = NULL;
f_abuffer_frame_begin abuffer_frame_begin                           = NULL;
f_abuffer_begin abuffer_begin                                       = NULL;
f_abuffer_end abuffer_end                                           = NULL;
f_abuffer_set_model_matrix abuffer_set_model_matrix                 = NULL;
f_abuffer_frame_end abuffer_frame_end                               = NULL;
f_abuffer_print_stats abuffer_print_stats                           = NULL;
f_abuffer_compute_num_records abuffer_compute_num_records           = NULL;
f_abuffer_set_custom_fragment_code abuffer_set_custom_fragment_code = NULL;    

// --------------------------------------------------------------------

void abuffer_load_dll(const char *fname)
{
  HMODULE h = LoadLibraryA(fname);
  sl_assert( h != NULL );
  abuffer_init                     = (f_abuffer_init)GetProcAddress(h, "abuffer_init"); 
  abuffer_terminate                = (f_abuffer_terminate)GetProcAddress(h, "abuffer_terminate"); 
  abuffer_change_size              = (f_abuffer_change_size)GetProcAddress(h, "abuffer_change_size"); 
  abuffer_set_perspective          = (f_abuffer_set_perspective)GetProcAddress(h, "abuffer_set_perspective"); 
  abuffer_set_view                 = (f_abuffer_set_view)GetProcAddress(h, "abuffer_set_view"); 
  abuffer_set_lightpos             = (f_abuffer_set_lightpos)GetProcAddress(h, "abuffer_set_lightpos"); 
  abuffer_frame_begin              = (f_abuffer_frame_begin)GetProcAddress(h, "abuffer_frame_begin"); 
  abuffer_begin                    = (f_abuffer_begin)GetProcAddress(h, "abuffer_begin"); 
  abuffer_end                      = (f_abuffer_end)GetProcAddress(h, "abuffer_end"); 
  abuffer_set_model_matrix         = (f_abuffer_set_model_matrix)GetProcAddress(h, "abuffer_set_model_matrix"); 
  abuffer_frame_end                = (f_abuffer_frame_end)GetProcAddress(h, "abuffer_frame_end"); 
  abuffer_print_stats              = (f_abuffer_print_stats)GetProcAddress(h, "abuffer_print_stats"); 
  abuffer_compute_num_records      = (f_abuffer_compute_num_records)GetProcAddress(h, "abuffer_compute_num_records"); 
  abuffer_set_custom_fragment_code = (f_abuffer_set_custom_fragment_code)GetProcAddress(h, "abuffer_set_custom_fragment_code"); 
}

// --------------------------------------------------------------------
