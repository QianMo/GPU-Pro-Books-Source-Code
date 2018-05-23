#ifndef FRUSTUM_H
#define FRUSTUM_H

// extract frustum planes from mvp col-major matrix
void frustum_extract_mvp(vec4_t out[6], const mat4_t mvp);

#endif //FRUSTUM_H

