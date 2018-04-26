#ifndef SHADER_SOM_H
#define SHADER_SOM_H

#include <GL/glew.h>

#include "g_pfm.h"

void shader_som_prepare( int codebook_w, int codebook_h );
void shader_som_begin( const GPfm &codebook );
void shader_som_end();

#endif
