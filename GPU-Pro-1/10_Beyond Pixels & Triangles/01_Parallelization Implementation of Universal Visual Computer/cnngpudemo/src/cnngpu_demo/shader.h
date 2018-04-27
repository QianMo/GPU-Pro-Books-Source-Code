#ifndef SHADER_H
#define SHADER_H

void cgfl_prepare( const char *exepath );
void cgcnn_begin( GLuint usrc, GLuint xsrc );
void cgcnn_end();
void cgredge_begin( GLuint usrc, GLuint xsrc );
void cgredge_end();
void cgledge_begin( GLuint usrc, GLuint xsrc );
void cgledge_end();
void cgunfold_begin( GLuint xsrc );
void cgunfold_end();

void cginit( const float *a, const float *b, float cnn_step );

#endif
