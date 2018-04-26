#ifndef GPU_SOM_H
#define GPU_SOH_H

#include <GL/glew.h>

#include "g_pfm.h"

class GPUSom
{
  public:
    int gpulbg_max_cycles;
    int codebook_w, codebook_h;
    int gpulbg_j;

    GPfm gpu_src;
    GPfm gpulbg_srcbook;
    GPfm gpulbg_gpubook;

    GLuint vis_src_id;
    GLuint vis_src_list;
    GLuint vis_ucodeword_id;
    GLuint vis_vcodeword_id;
    GLuint vis_des_id;
    GLuint vis_tbook_id;
    GLuint vis_tbook_fbo;
    GLuint vis_codebook_id;

    GPUSom();
    ~GPUSom();
    void set_info( int max_cycles, int codebook_w, int codebook_h );
    void prepare( const char *spath );
    void prepare_codebook();

    void init();
    void gpulbg_iterate();
    void update_codeword();
  private:
    void prepare( const GPfm &src );
    void clear();
    
};

#endif