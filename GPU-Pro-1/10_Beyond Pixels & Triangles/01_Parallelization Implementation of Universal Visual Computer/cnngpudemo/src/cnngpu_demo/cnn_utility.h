#ifndef CNN_UTILITY_H
#define CNN_UTILITY_H

void cnn_prepare_all_ad_upf( 
  const GPfm &src0, bool bBinary,
  float *A, float *B, float I, 
  bool bxinit, float xinit, float cnn_step
);
GLuint iterate( int n_pass );
void save_output( GPf1 &res2, int srcw, int srch );
void save_output( const char *spath, int srcw, int srch );


#endif