#ifndef SHADER_CODEWORD_H
#define SHADER_CODEWORD_H

#include <GL/glew.h>
#include "g_pfm.h"

void gpu_codebook2codeword( GLuint srcid, const GPfm &srcbook, GLuint codewordid );
void gpu_subidx_projection( GLuint srcid, GLuint desid, GLuint ucodewordid, GLuint vcodewordid, GLuint srcbookid );

void shader_codeword_prepare( int codebook_w, int codebook_h );
void cgfl_begin_codeword( const GPfm &codebook, const GLuint texid );
void cgfl_end_codeword();

void cgfl_begin_codeword_subidx( const GLuint texid, const GLuint codeword_texid, const GLuint codebook_texid );
void cgfl_end_codeword_subidx();

#endif
