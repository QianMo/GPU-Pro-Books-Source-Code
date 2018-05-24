#ifndef mymath_h
#define mymath_h

#include "mm_common.h"

#ifdef MYMATH_USE_SSE2
#include "mm_fvec2_impl.h"
#include "mm_fvec3_impl.h"
#include "mm_fvec4_impl.h"
#include "mm_fvec_swizzle_out.h"
#include "mm_fvec_func.h"
#include "mm_fmat_func.h"
#endif

#include "mm_vec2_impl.h"
#include "mm_vec3_impl.h"
#include "mm_vec4_impl.h"
#include "mm_vec_swizzle_out.h"
#include "mm_vec_func.h"
#include "mm_mat2_impl.h"
#include "mm_mat3_impl.h"
#include "mm_mat4_impl.h"
#include "mm_mat_func.h"

#include "mm_quat_impl.h"
#include "mm_quat_func.h"

#include "mm_util.h"
#include "mm_frame.h"
#include "mm_camera.h"
#include "mm_matrix_stack.h"
#include "mm_pipeline.h"

namespace mymath
{
#if MYMATH_DOUBLE_PRECISION == 1
  typedef impl::vec2i<double> dvec2;
  typedef impl::vec3i<double> dvec3;
  typedef impl::vec4i<double> dvec4;
  typedef impl::mat2i<double> dmat2;
  typedef impl::mat3i<double> dmat3;
  typedef impl::mat4i<double> dmat4;
#endif

  typedef impl::vec2i<float> vec2;
  typedef impl::vec2i<bool> bvec2;
  typedef impl::vec2i<int> ivec2;
  typedef impl::vec2i<unsigned int> uvec2;

  typedef impl::vec3i<float> vec3;
  typedef impl::vec3i<bool> bvec3;
  typedef impl::vec3i<int> ivec3;
  typedef impl::vec3i<unsigned int> uvec3;

  typedef impl::vec4i<float> vec4;
  typedef impl::vec4i<bool> bvec4;
  typedef impl::vec4i<int> ivec4;
  typedef impl::vec4i<unsigned int> uvec4;

  typedef impl::mat2i<float> mat2;

  typedef impl::mat3i<float> mat3;

  typedef impl::mat4i<float> mat4;

  typedef impl::quati<float> quat;
}

#endif
