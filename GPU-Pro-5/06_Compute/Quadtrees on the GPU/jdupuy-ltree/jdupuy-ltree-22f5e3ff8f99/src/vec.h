// vec.h - public domain C vector math library
#ifndef VEC_H
#define VEC_H

#include <math.h>
#ifndef assert // custom assertions
#	include <assert.h>
#endif

typedef float vec2_t[2];
typedef float vec3_t[3];
typedef float vec4_t[4];
typedef float mat2_t[4];
typedef float mat3_t[9];
typedef float mat4_t[16];
typedef float quat_t[4];

// access matrix elements
#define mat2_00(m) (m)[0]
#define mat2_01(m) (m)[1]
#define mat2_10(m) (m)[2]
#define mat2_11(m) (m)[3]

#define mat3_00(m) (m)[0]
#define mat3_01(m) (m)[1]
#define mat3_02(m) (m)[2]
#define mat3_10(m) (m)[3]
#define mat3_11(m) (m)[4]
#define mat3_12(m) (m)[5]
#define mat3_20(m) (m)[6]
#define mat3_21(m) (m)[7]
#define mat3_22(m) (m)[8]

#define mat4_00(m) (m)[0]
#define mat4_01(m) (m)[1]
#define mat4_02(m) (m)[2]
#define mat4_03(m) (m)[3]
#define mat4_10(m) (m)[4]
#define mat4_11(m) (m)[5]
#define mat4_12(m) (m)[6]
#define mat4_13(m) (m)[7]
#define mat4_20(m) (m)[8]
#define mat4_21(m) (m)[9]
#define mat4_22(m) (m)[10]
#define mat4_23(m) (m)[11]
#define mat4_30(m) (m)[12]
#define mat4_31(m) (m)[13]
#define mat4_32(m) (m)[14]
#define mat4_33(m) (m)[15]

// read / write matrix columns as vec_t
#define mat2_c0(m) (*((vec2_t *) &(m)[0]))
#define mat2_c1(m) (*((vec2_t *) &(m)[2]))

#define mat3_c0(m) (*((vec3_t *) &(m)[0]))
#define mat3_c1(m) (*((vec3_t *) &(m)[3]))
#define mat3_c2(m) (*((vec3_t *) &(m)[6]))

#define mat4_c0(m) (*((vec4_t *) &(m)[0]))
#define mat4_c1(m) (*((vec4_t *) &(m)[4]))
#define mat4_c2(m) (*((vec4_t *) &(m)[8]))
#define mat4_c3(m) (*((vec4_t *) &(m)[12]))

// read matrix columns as vec_t
#define mat2_c0_readonly(m) (*((const vec2_t *) &(m)[0]))
#define mat2_c1_readonly(m) (*((const vec2_t *) &(m)[2]))

#define mat3_c0_readonly(m) (*((const vec3_t *) &(m)[0]))
#define mat3_c1_readonly(m) (*((const vec3_t *) &(m)[3]))
#define mat3_c2_readonly(m) (*((const vec3_t *) &(m)[6]))

#define mat4_c0_readonly(m) (*((const vec4_t *) &(m)[0]))
#define mat4_c1_readonly(m) (*((const vec4_t *) &(m)[4]))
#define mat4_c2_readonly(m) (*((const vec4_t *) &(m)[8]))
#define mat4_c3_readonly(m) (*((const vec4_t *) &(m)[12]))

// generic declaration macros
#define VEC_CAT(n,x) n##x
#define VEC_DECL_FUNC1(name,x,...) \
	_Generic(x, \
	         vec2_t:VEC_CAT(vec2_,name),\
	         vec3_t:VEC_CAT(vec3_,name),\
	         vec4_t:VEC_CAT(vec4_,name))\
	         (x,##__VA_ARGS__)

/* fill vector */
static inline void
vec2_set(vec2_t dst, float value) {
	dst[0] = value;
	dst[1] = value;
}
static inline void
vec3_set(vec3_t dst, float value) {
	dst[0] = value;
	dst[1] = value;
	dst[2] = value;
}
static inline void
vec4_set(vec4_t dst, float value) {
	dst[0] = value;
	dst[1] = value;
	dst[2] = value;
	dst[3] = value;
}
#define vec_set(dst, value) VEC_DECL_FUNC1(set,(dst),(value))


/* copy vector */
static inline void
vec2_cpy(vec2_t dst, const vec2_t src) {
	dst[0] = src[0];
	dst[1] = src[1];
}
static inline void
vec3_cpy(vec3_t dst, const vec3_t src) {
	dst[0] = src[0];
	dst[1] = src[1];
	dst[2] = src[2];
}
static inline void
vec4_cpy(vec4_t dst, const vec4_t src) {
	dst[0] = src[0];
	dst[1] = src[1];
	dst[2] = src[2];
	dst[3] = src[3];
}
#define vec_cpy(dst,src) VEC_DECL_FUNC1(cpy,dst,src)


/* comp wise add */
static inline void
vec2_add(vec2_t out, const vec2_t x, const vec2_t y) {
	out[0] = x[0] + y[0];
	out[1] = x[1] + y[1];
}
static inline void
vec3_add(vec3_t out, const vec3_t x, const vec3_t y) {
	out[0] = x[0] + y[0];
	out[1] = x[1] + y[1];
	out[2] = x[2] + y[2];
}
static inline void
vec4_add(vec4_t out, const vec4_t x, const vec4_t y) {
	out[0] = x[0] + y[0];
	out[1] = x[1] + y[1];
	out[2] = x[2] + y[2];
	out[3] = x[3] + y[3];
}
#define vec_add(out,x,y) VEC_DECL_FUNC1(add,(out),(x),(y))


/* comp wise sub */
static inline void
vec2_sub(vec2_t out, const vec2_t x, const vec2_t y) {
	out[0] = x[0] - y[0];
	out[1] = x[1] - y[1];
}
static inline void
vec3_sub(vec3_t out, const vec3_t x, const vec3_t y) {
	out[0] = x[0] - y[0];
	out[1] = x[1] - y[1];
	out[2] = x[2] - y[2];
}
static inline void
vec4_sub(vec4_t out, const vec4_t x, const vec4_t y) {
	out[0] = x[0] - y[0];
	out[1] = x[1] - y[1];
	out[2] = x[2] - y[2];
	out[3] = x[3] - y[3];
}
#define vec_sub(out,x,y) VEC_DECL_FUNC1(sub,(out),(x),(y))


/* com wise mul */
static inline void
vec2_mul(vec2_t out, const vec2_t x, const vec2_t y) {
	out[0] = x[0] * y[0];
	out[1] = x[1] * y[1];
}
static inline void
vec3_mul(vec3_t out, const vec3_t x, const vec3_t y) {
	out[0] = x[0] * y[0];
	out[1] = x[1] * y[1];
	out[2] = x[2] * y[2];
}
static inline void
vec4_mul(vec4_t out, const vec4_t x, const vec4_t y) {
	out[0] = x[0] * y[0];
	out[1] = x[1] * y[1];
	out[2] = x[2] * y[2];
	out[3] = x[3] * y[3];
}
#define vec_mul(out,x,y) VEC_DECL_FUNC1(vmul,(out),(x),(y))


/* com wise div */
static inline void
vec2_div(vec2_t out, const vec2_t x, const vec2_t y) {
	out[0] = x[0] / y[0];
	out[1] = x[1] / y[1];
}
static inline void
vec3_div(vec3_t out, const vec3_t x, const vec3_t y) {
	out[0] = x[0] / y[0];
	out[1] = x[1] / y[1];
	out[2] = x[2] / y[2];
}
static inline void
vec4_div(vec4_t out, const vec4_t x, const vec4_t y) {
	out[0] = x[0] / y[0];
	out[1] = x[1] / y[1];
	out[2] = x[2] / y[2];
	out[3] = x[3] / y[3];
}
#define vec_div(out,x,y) VEC_DECL_FUNC1(div,(out),(x),(y))


/* comp wise min */
static inline void
vec2_min(vec2_t out, const vec2_t x, const vec2_t y) {
	out[0] = x[0] > y[0] ? y[0] : x[0];
	out[1] = x[1] > y[1] ? y[1] : x[1];
}
static inline void
vec3_min(vec3_t out, const vec3_t x, const vec3_t y) {
	out[0] = x[0] > y[0] ? y[0] : x[0];
	out[1] = x[1] > y[1] ? y[1] : x[1];
	out[2] = x[2] > y[2] ? y[2] : x[2];
}
static inline void
vec4_min(vec4_t out, const vec4_t x, const vec4_t y) {
	out[0] = x[0] > y[0] ? y[0] : x[0];
	out[1] = x[1] > y[1] ? y[1] : x[1];
	out[2] = x[2] > y[2] ? y[2] : x[2];
	out[3] = x[3] > y[3] ? y[3] : x[3];
}
#define vec_min(out,x,y) VEC_DECL_FUNC1(min,(out),(x),(y))


/* comp wise max */
static inline void
vec2_max(vec2_t out, const vec2_t x, const vec2_t y) {
	out[0] = x[0] > y[0] ? x[0] : y[0];
	out[1] = x[1] > y[1] ? x[1] : y[1];
}
static inline void
vec3_max(vec3_t out, const vec3_t x, const vec3_t y) {
	out[0] = x[0] > y[0] ? x[0] : y[0];
	out[1] = x[1] > y[1] ? x[1] : y[1];
	out[2] = x[2] > y[2] ? x[2] : y[2];
}
static inline void
vec4_max(vec4_t out, const vec4_t x, const vec4_t y) {
	out[0] = x[0] > y[0] ? x[0] : y[0];
	out[1] = x[1] > y[1] ? x[1] : y[1];
	out[2] = x[2] > y[2] ? x[2] : y[2];
	out[3] = x[3] > y[3] ? x[3] : y[3];
}
#define vec_max(out,x,y) VEC_DECL_FUNC1(max,(out),(x),(y))


/* comp wise clamp */
static inline void
vec2_clamp2(vec2_t out,
            const vec2_t x,
            const vec2_t min,
            const vec2_t max) {
	vec2_t temp;

	vec2_min(temp, x, max);
	vec2_max(out, temp, min);
}
static inline void
vec3_clamp(vec3_t out,
           const vec3_t x,
           const vec3_t min,
           const vec3_t max
          ) {
	vec3_t temp;

	vec3_min(temp, x, max);
	vec3_max(out, temp, min);
}
static inline void
vec4_clamp(vec4_t out,
           const vec4_t x,
           const vec4_t min,
           const vec4_t max) {
	vec4_t temp;

	vec4_min(temp, x, max);
	vec4_max(out, temp, min);
}
#define vec_clamp(out,x,min,max) \
	VEC_DECL_FUNC1(clamp,(out),(x),(min),(max))


/* cross product */
static inline void
vec3_cross(vec3_t out, const vec3_t x, const vec3_t y) {
	out[0] = x[1] * y[2] - x[2] * y[1];
	out[1] = x[2] * y[0] - x[0] * y[2];
	out[2] = x[0] * y[1] - x[1] * y[0];
}
#define vec_cross(out,x,y) vec3_cross((out),(x),(y))


/* dot product */
static inline float
vec2_dot(const vec2_t x, const vec2_t y) {
	return x[0] * y[0] + x[1] * y[1];
}
static inline float
vec3_dot(const vec3_t x, const vec3_t y) {
	return x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
}
static inline float
vec4_dot(const vec4_t x, const vec4_t y) {
	return x[0] * y[0] + x[1] * y[1] + x[2] * y[2] + x[3] * y[3];
}
#define vec_dot(x,y) VEC_DECL_FUNC1(dot,(x),(y))


/* vector squared mag */
static inline float
vec2_length2(const vec2_t x) {
	return vec2_dot(x, x);
}
static inline float
vec3_length2(const vec3_t x) {
	return vec3_dot(x, x);
}
static inline float
vec4_length2(const vec4_t x) {
	return vec4_dot(x, x);
}
#define vec_length2(x) VEC_DECL_FUNC1(length2,(x))


/* vector mag */
static inline float
vec2_length(const vec2_t x) {
	return sqrt(vec2_length2(x));
}
static inline float
vec3_length(const vec3_t x) {
	return sqrt(vec3_length2(x));
}
static inline float
vec4_length(const vec4_t x) {
	return sqrt(vec4_length2(x));
}
#define vec_length(x) VEC_DECL_FUNC1(length,x)


/* normalize vector */
static inline void
vec2_normalize(vec2_t out, const vec2_t x) {
	assert(vec2_length2(x) > 0.f);
	float l = 1.f / vec2_length(x);
	vec2_t lv = {l, l};

	vec2_mul(out, lv, x);
}
static inline void
vec3_normalize(vec3_t out, const vec3_t x) {
	assert(vec3_length2(x) > 0.f);
	float l = 1.f / vec3_length(x);
	vec3_t lv = {l, l, l};

	vec3_mul(out, lv, x);
}
static inline void
vec4_normalize(vec4_t out, const vec4_t x) {
	assert(vec4_length2(x) > 0.f);
	float l = 1.f / vec4_length(x);
	vec4_t lv = {l, l, l, l};

	vec4_mul(out, lv, x);
}
#define vec_normalize(out,x) VEC_DECL_FUNC1(normalize,(out),(x))


/* squared distance */
static inline float
vec2_distance2(const vec2_t x, const vec2_t y) {
	vec2_t xy;

	vec2_sub(xy, x, y);
	return vec2_length2(xy);
}
static inline float
vec3_distance2(const vec3_t x, const vec3_t y) {
	vec3_t xy;

	vec3_sub(xy, x, y);
	return vec3_length2(xy);
}
static inline float
vec4_distance2(const vec4_t x, const vec4_t y) {
	vec4_t xy;

	vec4_sub(xy, x, y);
	return vec4_length2(xy);
}
#define vec_distance2(x,y) VEC_DECL_FUNC1(distance2,(x),(y))


/* distance */
static inline float
vec2_distance(const vec2_t x, const vec2_t y) {
	return sqrt(vec2_distance2(x, y));
}
static inline float
vec3_distance(const vec3_t x, const vec3_t y) {
	return sqrt(vec3_distance2(x, y));
}
static inline float
vec4_distance(const vec4_t x, const vec4_t y) {
	return sqrt(vec4_distance2(x, y));
}
#define vec_distance(x,y) VEC_DECL_FUNC1(distance,(x),(y))


/* linear interpolation */
static inline void
vec2_mix(vec2_t out, const vec2_t x, const vec2_t y, float a) {
	out[0] = x[0] + a * (y[0] - x[0]);
	out[1] = x[1] + a * (y[1] - x[1]);
}
static inline void
vec3_mix(vec3_t out, const vec3_t x, const vec3_t y, float a) {
	out[0] = x[0] + a * (y[0] - x[0]);
	out[1] = x[1] + a * (y[1] - x[1]);
	out[2] = x[2] + a * (y[2] - x[2]);
}
static inline void
vec4_mix(vec4_t out, const vec4_t x, const vec4_t y, float a) {
	out[0] = x[0] + a * (y[0] - x[0]);
	out[1] = x[1] + a * (y[1] - x[1]);
	out[2] = x[2] + a * (y[2] - x[2]);
	out[3] = x[3] + a * (y[3] - x[3]);
}
#define vec_mix(out,x,y,a) VEC_DECL_FUNC1(mix,(out),(x),(y),(a))


/* reflect vector */
static inline void
vec2_reflect(vec2_t out, const vec2_t i, const vec2_t n) {
	float d = 2.f * vec2_dot(i,n);
	vec2_t dv = {d,d};
	vec2_t mul;

	vec2_mul(mul, n, dv);
	vec2_sub(out, i, mul);
}
static inline void
vec3_reflect(vec3_t out, const vec3_t i, const vec3_t n) {
	float d = 2.f * vec3_dot(i,n);
	vec3_t dv = {d,d,d};
	vec3_t mul;

	vec3_mul(mul, n, dv);
	vec3_sub(out, i, mul);
}
static inline void
vec4_reflect(vec4_t out, const vec4_t i, const vec4_t n) {
	float d = 2.f * vec4_dot(i,n);
	vec4_t dv = {d,d,d,d};
	vec4_t mul;

	vec4_mul(mul, n, dv);
	vec4_sub(out, i, mul);
}
#define vec_reflect(out,i,n) VEC_DECL_FUNC1(reflect,(out),(i),(n))


/* refract vector */
static inline void
vec2_refract(vec2_t out, const vec2_t i, const vec2_t n, float eta) {
	float d = 2.f * vec2_dot(i,n);
	float k = 1.f - eta * eta * (1.f - d * d);

	if(k<0.f) {
		vec2_set(out, 0.f);
	} else {
		float k2 = eta * d + sqrt(k);
		vec2_t k2v = {k2,k2};
		vec2_t etav = {eta,eta};
		vec2_t r1, r2;

		vec2_mul(r1, i, etav);
		vec2_mul(r2, n, k2v);
		vec2_sub(out, r2, r1);
	}
}
static inline void
vec3_refract(vec3_t out, const vec3_t i, const vec3_t n, float eta) {
	float d = 2.f * vec3_dot(i,n);
	float k = 1.f - eta * eta * (1.f - d * d);
	if(k<0.f) {
		vec3_set(out,0.f);
	} else {
		float k2 = eta * d + sqrt(k);
		vec3_t k2v = {k2,k2,k2};
		vec3_t etav = {eta,eta,eta};
		vec3_t r1, r2;

		vec3_mul(r1, i, etav);
		vec3_mul(r2, n, k2v);
		vec3_sub(out, r2, r1);
	}
}
static inline void
vec4_refract(vec4_t out, const vec4_t i, const vec4_t n, float eta) {
	float d = 2.f * vec4_dot(i,n);
	float k = 1.f - eta * eta * (1.f - d * d);

	if(k<0.f) {
		vec4_set(out,0.f);
	} else {
		float k2 = eta * d + sqrt(k);
		vec4_t k2v = {k2,k2,k2,k2};
		vec4_t etav = {eta,eta,eta};
		vec4_t r1, r2;

		vec4_mul(r1, i, etav);
		vec4_mul(r2, n, k2v);
		vec4_sub(out, r2, r1);
	}
}
#define vec_refract(out,i,n,eta) \
	VEC_DECL_FUNC1(refract,(out),(i),(n),(eta))

/* outer product */
static inline void
vec2_outer(mat2_t out, const vec2_t x, const vec2_t y) {
	mat2_00(out) = x[0] * y[0];
	mat2_01(out) = x[1] * y[0];
		mat2_10(out) = x[0] * y[1];
		mat2_11(out) = x[1] * y[1];
}
static inline void
vec3_outer(mat3_t out, const vec3_t x, const vec3_t y) {
	mat3_00(out) = x[0] * y[0];
	mat3_01(out) = x[1] * y[0];
	mat3_02(out) = x[2] * y[0];
		mat3_10(out) = x[0] * y[1];
		mat3_11(out) = x[1] * y[1];
		mat3_12(out) = x[2] * y[1];
			mat3_20(out) = x[0] * y[2];
			mat3_21(out) = x[1] * y[2];
			mat3_22(out) = x[2] * y[2];
}
static inline void
vec4_outer(mat4_t out, const vec4_t x, const vec4_t y) {
	mat4_00(out) = x[0] * y[0];
	mat4_01(out) = x[1] * y[0];
	mat4_02(out) = x[2] * y[0];
	mat4_03(out) = x[3] * y[0];
		mat4_10(out) = x[0] * y[1];
		mat4_11(out) = x[1] * y[1];
		mat4_12(out) = x[2] * y[1];
		mat4_13(out) = x[3] * y[1];
			mat4_20(out) = x[0] * y[2];
			mat4_21(out) = x[1] * y[2];
			mat4_22(out) = x[2] * y[2];
			mat4_23(out) = x[3] * y[2];
				mat4_30(out) = x[0] * y[3];
				mat4_31(out) = x[1] * y[3];
				mat4_32(out) = x[2] * y[3];
				mat4_33(out) = x[3] * y[3];
}
#define vec_outer(out,x,y) VEC_DECL_FUNC1(outer,(out),(x),(y))


/* matrices */
#define VEC_DECL_FUNC2(name,x,...) \
	_Generic((x), \
	          mat2_t:VEC_CAT(mat2_,name),\
	          mat3_t:VEC_CAT(mat3_,name),\
	          mat4_t:VEC_CAT(mat4_,name))\
	          (x,##__VA_ARGS__)


/* matrix copy */
static inline void
mat2_cpy(mat2_t dst, const mat2_t src) {
	vec2_cpy(mat2_c0(dst), mat2_c0_readonly(src));
	vec2_cpy(mat2_c1(dst), mat2_c1_readonly(src));
}

static inline void
mat3_cpy(mat3_t dst, const mat3_t src) {
	vec3_cpy(mat3_c0(dst), mat3_c0_readonly(src));
	vec3_cpy(mat3_c1(dst), mat3_c1_readonly(src));
	vec3_cpy(mat3_c2(dst), mat3_c2_readonly(src));
}

static inline void
mat4_cpy(mat4_t dst, const mat4_t src) {
	vec4_cpy(mat4_c0(dst), mat4_c0_readonly(src));
	vec4_cpy(mat4_c1(dst), mat4_c1_readonly(src));
	vec4_cpy(mat4_c2(dst), mat4_c2_readonly(src));
	vec4_cpy(mat4_c3(dst), mat4_c3_readonly(src));
}
#define mat_cpy(dst,src) VEC_DECL_FUNC2(cpy,(dst),(src))


/* set matrix to identity */
static inline void
mat2_identity(mat2_t m) {
	mat2_00(m) = 1.f; mat2_10(m) = 0.f;
	mat2_01(m) = 0.f; mat2_11(m) = 1.f;
}
static inline void
mat3_identity(mat3_t m) {
	mat3_00(m) = 1.f; mat3_10(m) = 0.f; mat3_20(m) = 0.f;
	mat3_01(m) = 0.f; mat3_11(m) = 1.f; mat3_21(m) = 0.f;
	mat3_02(m) = 0.f; mat3_12(m) = 0.f; mat3_22(m) = 1.f;
}
static inline void
mat4_identity(mat4_t m) {
	mat4_00(m) = 1.f; mat4_10(m) = 0.f; mat4_20(m) = 0.f; mat4_30(m) = 0.f;
	mat4_01(m) = 0.f; mat4_11(m) = 1.f; mat4_21(m) = 0.f; mat4_31(m) = 0.f;
	mat4_02(m) = 0.f; mat4_12(m) = 0.f; mat4_22(m) = 1.f; mat4_32(m) = 0.f;
	mat4_03(m) = 0.f; mat4_13(m) = 0.f; mat4_23(m) = 0.f; mat4_33(m) = 1.f;
}
#define mat_identity(m) VEC_DECL_FUNC2(identity,(m))


/* matrix determinant */
static inline float
mat2_determinant(const mat2_t m) {
	return mat2_00(m) * mat2_11(m) - mat2_01(m) * mat2_10(m);
}
static inline float
mat3_determinant(const mat3_t m) {
	float d1 = mat3_11(m) * mat3_22(m) - mat3_21(m) * mat3_12(m);
	float d2 = mat3_21(m) * mat3_02(m) - mat3_01(m) * mat3_22(m);
	float d3 = mat3_01(m) * mat3_12(m) - mat3_11(m) * mat3_02(m);
	return mat3_00(m) * d1 - mat3_10(m) * d2 + mat3_20(m) * d3;
}
static inline float
mat4_determinant(const mat4_t m) {
	float s01 = mat4_00(m) * mat4_11(m) - mat4_10(m) * mat4_01(m);
	float s02 = mat4_22(m) * mat4_33(m) - mat4_32(m) * mat4_23(m);
	float s03 = mat4_00(m) * mat4_21(m) - mat4_20(m) * mat4_10(m);
	float s04 = mat4_12(m) * mat4_33(m) - mat4_32(m) * mat4_13(m);
	float s05 = mat4_00(m) * mat4_31(m) - mat4_30(m) * mat4_01(m);
	float s06 = mat4_12(m) * mat4_23(m) - mat4_22(m) * mat4_13(m);
	float s07 = mat4_10(m) * mat4_21(m) - mat4_20(m) * mat4_11(m);
	float s08 = mat4_02(m) * mat4_33(m) - mat4_32(m) * mat4_03(m);
	float s09 = mat4_10(m) * mat4_31(m) - mat4_30(m) * mat4_11(m);
	float s10 = mat4_02(m) * mat4_23(m) - mat4_22(m) * mat4_03(m);
	float s11 = mat4_20(m) * mat4_31(m) - mat4_30(m) * mat4_21(m);
	float s12 = mat4_02(m) * mat4_13(m) - mat4_12(m) * mat4_03(m);
	return s01*s02 - s03*s04 + s05*s06 + s07*s08 - s09*s10 + s11*s12;
}
#define mat_determinant(m) VEC_DECL_FUNC2(determinant,(m))


/* matrix transpose */
static inline void
mat2_transpose(mat2_t out, const mat2_t m) {
	mat2_t t;

	mat2_00(t) = mat2_00(m);
	mat2_01(t) = mat2_10(m);
		mat2_10(t) = mat2_01(m);
		mat2_11(t) = mat2_11(m);
	mat2_cpy(out, t);
}
static inline void
mat3_transpose(mat3_t out, const mat3_t m) {
	mat3_t t;

	mat3_00(t) = mat3_00(m);
	mat3_01(t) = mat3_10(m);
	mat3_02(t) = mat3_20(m);
		mat3_10(t) = mat3_01(m);
		mat3_11(t) = mat3_11(m);
		mat3_12(t) = mat3_21(m);
			mat3_20(t) = mat3_02(m);
			mat3_21(t) = mat3_12(m);
			mat3_22(t) = mat3_22(m);
	mat3_cpy(out, t);
}
static inline void
mat4_transpose(mat4_t out, const mat4_t m) {
	mat4_t t;

	mat4_00(t) = mat4_00(m);
	mat4_01(t) = mat4_10(m);
	mat4_02(t) = mat4_20(m);
	mat4_03(t) = mat4_30(m);
		mat4_10(t) = mat4_01(m);
		mat4_11(t) = mat4_11(m);
		mat4_12(t) = mat4_21(m);
		mat4_13(t) = mat4_31(m);
			mat4_20(t) = mat4_02(m);
			mat4_21(t) = mat4_12(m);
			mat4_22(t) = mat4_22(m);
			mat4_23(t) = mat4_32(m);
				mat4_30(t) = mat4_03(m);
				mat4_31(t) = mat4_13(m);
				mat4_32(t) = mat4_23(m);
				mat4_33(t) = mat4_33(m);
	mat4_cpy(out, t);
}
#define mat_transpose(out,m) VEC_DECL_FUNC2(transpose,(out),(m))


/* matrix adjugate */
static inline void
mat2_adjugate(mat2_t out, const mat2_t m) {
	mat2_t a;

	mat2_00(a) = mat2_11(m);
	mat2_01(a) =-mat2_10(m);

	mat2_10(a) =-mat2_01(m);
	mat2_11(a) = mat2_00(m);
	mat2_cpy(out, a);
}
static inline void
mat3_adjugate(mat3_t out, const mat3_t m) {
	mat3_t a;

	mat3_00(a) = mat3_11(m) * mat3_22(m) - mat3_12(m) * mat3_21(m);
	mat3_01(a) = mat3_02(m) * mat3_22(m) - mat3_01(m) * mat3_22(m);
	mat3_02(a) = mat3_01(m) * mat3_12(m) - mat3_11(m) * mat3_02(m);

	mat3_10(a) = mat3_12(m) * mat3_20(m) - mat3_10(m) * mat3_22(m);
	mat3_11(a) = mat3_00(m) * mat3_22(m) - mat3_02(m) * mat3_20(m);
	mat3_12(a) = mat3_10(m) * mat3_02(m) - mat3_00(m) * mat3_12(m);

	mat3_20(a) = mat3_10(m) * mat3_21(m) - mat3_11(m) * mat3_20(m);
	mat3_21(a) = mat3_20(m) * mat3_01(m) - mat3_00(m) * mat3_21(m);
	mat3_22(a) = mat3_00(m) * mat3_11(m) - mat3_01(m) * mat3_10(m);
	mat3_cpy(out, a);
}
static inline void
mat4_adjugate(mat4_t out, const mat4_t m) {
	// use laplace expansion theorem
	mat4_t a;
	float s0 = mat4_00(m) * mat4_11(m) - mat4_10(m) * mat4_01(m);
	float s1 = mat4_00(m) * mat4_21(m) - mat4_20(m) * mat4_01(m);
	float s2 = mat4_00(m) * mat4_31(m) - mat4_30(m) * mat4_01(m);
	float s3 = mat4_10(m) * mat4_21(m) - mat4_20(m) * mat4_11(m);
	float s4 = mat4_10(m) * mat4_31(m) - mat4_30(m) * mat4_11(m);
	float s5 = mat4_20(m) * mat4_31(m) - mat4_30(m) * mat4_21(m);
	float c5 = mat4_22(m) * mat4_33(m) - mat4_32(m) * mat4_23(m);
	float c4 = mat4_12(m) * mat4_33(m) - mat4_32(m) * mat4_13(m);
	float c3 = mat4_12(m) * mat4_23(m) - mat4_22(m) * mat4_13(m);
	float c2 = mat4_02(m) * mat4_33(m) - mat4_32(m) * mat4_03(m);
	float c1 = mat4_02(m) * mat4_23(m) - mat4_22(m) * mat4_03(m);
	float c0 = mat4_02(m) * mat4_13(m) - mat4_12(m) * mat4_03(m);

	mat4_00(a) = mat4_11(m)*c5 - mat4_21(m)*c4 + mat4_31(m)*c3;
	mat4_01(a) =-mat4_01(m)*c5 + mat4_21(m)*c2 - mat4_31(m)*c1;
	mat4_02(a) = mat4_01(m)*c4 - mat4_11(m)*c2 + mat4_31(m)*c0;
	mat4_03(a) =-mat4_01(m)*c3 + mat4_11(m)*c1 - mat4_21(m)*c0;

	mat4_10(a) =-mat4_10(m)*c5 + mat4_20(m)*c4 - mat4_30(m)*c3;
	mat4_11(a) = mat4_00(m)*c5 - mat4_20(m)*c2 + mat4_30(m)*c1;
	mat4_12(a) =-mat4_00(m)*c4 + mat4_10(m)*c2 - mat4_30(m)*c0;
	mat4_13(a) = mat4_00(m)*c3 - mat4_10(m)*c1 + mat4_20(m)*c0;

	mat4_20(a) = mat4_13(m)*s5 - mat4_23(m)*s4 + mat4_33(m)*s3;
	mat4_21(a) =-mat4_03(m)*s5 + mat4_23(m)*s2 - mat4_33(m)*s1;
	mat4_22(a) = mat4_03(m)*s4 - mat4_13(m)*s2 + mat4_33(m)*s0;
	mat4_23(a) =-mat4_03(m)*s3 + mat4_13(m)*s1 - mat4_23(m)*s0;

	mat4_30(a) =-mat4_12(m)*s5 + mat4_22(m)*s4 - mat4_32(m)*s3;
	mat4_31(a) = mat4_02(m)*s5 - mat4_22(m)*s2 + mat4_32(m)*s1;
	mat4_32(a) =-mat4_02(m)*s4 + mat4_12(m)*s2 - mat4_32(m)*s0;
	mat4_33(a) = mat4_02(m)*s3 - mat4_12(m)*s1 + mat4_22(m)*s0;
	mat4_cpy(out, a);
}
#define ADJUGATE(out,m)   VEC_DECL_FUNC2(adjugate,(out),(m))


/* inverse matrix */
static inline void
mat2_inverse(mat2_t out, const mat2_t m) {
	assert(mat2_determinant(m) != 0.f);
	mat2_t i;
	float d = 1.0f / mat2_determinant(m);
	vec2_t dv = {d,d};

	mat2_adjugate(i, m);
	vec2_mul(mat2_c0(out), dv, mat2_c0_readonly(i));
	vec2_mul(mat2_c1(out), dv, mat2_c1_readonly(i));
}
static inline void
mat3_inverse(mat3_t out, const mat3_t m) {
	assert(mat3_determinant(m) != 0.f);
	mat3_t i;
	float d = 1.0f / mat3_determinant(m);
	vec3_t dv = {d,d,d};

	mat3_adjugate(i, m);
	vec3_mul(mat3_c0(out), dv, mat3_c0_readonly(i));
	vec3_mul(mat3_c1(out), dv, mat3_c1_readonly(i));
	vec3_mul(mat3_c2(out), dv, mat3_c2_readonly(i));
}
static inline void
mat4_inverse(mat4_t out, const mat4_t m) {
	// use laplace expansion theorem
	mat4_t a;
	float s0 = mat4_00(m) * mat4_11(m) - mat4_10(m) * mat4_01(m);
	float s1 = mat4_00(m) * mat4_21(m) - mat4_20(m) * mat4_01(m);
	float s2 = mat4_00(m) * mat4_31(m) - mat4_30(m) * mat4_01(m);
	float s3 = mat4_10(m) * mat4_21(m) - mat4_20(m) * mat4_11(m);
	float s4 = mat4_10(m) * mat4_31(m) - mat4_30(m) * mat4_11(m);
	float s5 = mat4_20(m) * mat4_31(m) - mat4_30(m) * mat4_21(m);
	float c5 = mat4_22(m) * mat4_33(m) - mat4_32(m) * mat4_23(m);
	float c4 = mat4_12(m) * mat4_33(m) - mat4_32(m) * mat4_13(m);
	float c3 = mat4_12(m) * mat4_23(m) - mat4_22(m) * mat4_13(m);
	float c2 = mat4_02(m) * mat4_33(m) - mat4_32(m) * mat4_03(m);
	float c1 = mat4_02(m) * mat4_23(m) - mat4_22(m) * mat4_03(m);
	float c0 = mat4_02(m) * mat4_13(m) - mat4_12(m) * mat4_03(m);
	float d = s0*c5 - s1*c4 + s2*c3 + s3*c2 - s4*c1 + s5*c0;
	assert(d!=0.f);
	d = 1.f / d;

	mat4_00(a) = d*( mat4_11(m)*c5 - mat4_21(m)*c4 + mat4_31(m)*c3);
	mat4_01(a) = d*(-mat4_01(m)*c5 + mat4_21(m)*c2 - mat4_31(m)*c1);
	mat4_02(a) = d*( mat4_01(m)*c4 - mat4_11(m)*c2 + mat4_31(m)*c0);
	mat4_03(a) = d*(-mat4_01(m)*c3 + mat4_11(m)*c1 - mat4_21(m)*c0);

	mat4_10(a) = d*(-mat4_10(m)*c5 + mat4_20(m)*c4 - mat4_30(m)*c3);
	mat4_11(a) = d*( mat4_00(m)*c5 - mat4_20(m)*c2 + mat4_30(m)*c1);
	mat4_12(a) = d*(-mat4_00(m)*c4 + mat4_10(m)*c2 - mat4_30(m)*c0);
	mat4_13(a) = d*( mat4_00(m)*c3 - mat4_10(m)*c1 + mat4_20(m)*c0);

	mat4_20(a) = d*( mat4_13(m)*s5 - mat4_23(m)*s4 + mat4_33(m)*s3);
	mat4_21(a) = d*(-mat4_03(m)*s5 + mat4_23(m)*s2 - mat4_33(m)*s1);
	mat4_22(a) = d*( mat4_03(m)*s4 - mat4_13(m)*s2 + mat4_33(m)*s0);
	mat4_23(a) = d*(-mat4_03(m)*s3 + mat4_13(m)*s1 - mat4_23(m)*s0);

	mat4_30(a) = d*(-mat4_12(m)*s5 + mat4_22(m)*s4 - mat4_32(m)*s3);
	mat4_31(a) = d*( mat4_02(m)*s5 - mat4_22(m)*s2 + mat4_32(m)*s1);
	mat4_32(a) = d*(-mat4_02(m)*s4 + mat4_12(m)*s2 - mat4_32(m)*s0);
	mat4_33(a) = d*( mat4_02(m)*s3 - mat4_12(m)*s1 + mat4_22(m)*s0);
	mat4_cpy(out, a);
}
#define mat_inverse(out,m) VEC_DECL_FUNC2(inverse,(out),(m))


/* matrix multiplication */
static inline void
mat2_vec2_mul(vec2_t out, const mat2_t m, const vec2_t v) {
	vec2_t vv = {v[0], v[1]};
	vec2_t r1 = {mat2_00(m), mat2_10(m)};
	vec2_t r2 = {mat2_01(m), mat2_11(m)};

	out[0] = vec2_dot(vv,r1); 
	out[1] = vec2_dot(vv,r2);
}
static inline void
mat3_vec3_mul(vec3_t out, const mat3_t m, const vec3_t v) {
	vec3_t vv = {v[0], v[1], v[2]};
	vec3_t r1 = {mat3_00(m), mat3_10(m), mat3_20(m)};
	vec3_t r2 = {mat3_01(m), mat3_11(m), mat3_21(m)};
	vec3_t r3 = {mat3_02(m), mat3_12(m), mat3_22(m)};

	out[0] = vec3_dot(vv,r1);
	out[1] = vec3_dot(vv,r2);
	out[2] = vec3_dot(vv,r3);
}
static inline void
mat4_vec4_mul(vec4_t out, const mat4_t m, const vec4_t v) {
	vec4_t vv = {v[0], v[1], v[2], v[3]};
	vec4_t r1 = {mat4_00(m), mat4_10(m), mat4_20(m), mat4_30(m)};
	vec4_t r2 = {mat4_01(m), mat4_11(m), mat4_21(m), mat4_31(m)};
	vec4_t r3 = {mat4_02(m), mat4_12(m), mat4_22(m), mat4_32(m)};
	vec4_t r4 = {mat4_03(m), mat4_13(m), mat4_23(m), mat4_33(m)};

	out[0] = vec4_dot(vv,r1);
	out[1] = vec4_dot(vv,r2);
	out[2] = vec4_dot(vv,r3);
	out[3] = vec4_dot(vv,r4);
}
static inline void
mat2_mul(mat2_t out, const mat2_t x, const mat2_t y) {
	mat2_t xt, r;

	mat2_transpose(xt, x);
	mat2_00(r) = vec2_dot(mat2_c0_readonly(xt), mat2_c0_readonly(y));
	mat2_01(r) = vec2_dot(mat2_c1_readonly(xt), mat2_c0_readonly(y));

	mat2_10(r) = vec2_dot(mat2_c0_readonly(xt), mat2_c1_readonly(y));
	mat2_11(r) = vec2_dot(mat2_c1_readonly(xt), mat2_c1_readonly(y));
	mat2_cpy(out, r);
}
static inline void
mat3_mul(mat3_t out, const mat3_t x, const mat3_t y) {
	mat3_t xt, r;

	mat3_transpose(xt, x);
	mat3_00(r) = vec3_dot(mat3_c0_readonly(xt), mat3_c0_readonly(y));
	mat3_01(r) = vec3_dot(mat3_c1_readonly(xt), mat3_c0_readonly(y));
	mat3_02(r) = vec3_dot(mat3_c2_readonly(xt), mat3_c0_readonly(y));

	mat3_10(r) = vec3_dot(mat3_c0_readonly(xt), mat3_c1_readonly(y));
	mat3_11(r) = vec3_dot(mat3_c1_readonly(xt), mat3_c1_readonly(y));
	mat3_12(r) = vec3_dot(mat3_c2_readonly(xt), mat3_c1_readonly(y));

	mat3_20(r) = vec3_dot(mat3_c0_readonly(xt), mat3_c2_readonly(y));
	mat3_21(r) = vec3_dot(mat3_c1_readonly(xt), mat3_c2_readonly(y));
	mat3_22(r) = vec3_dot(mat3_c2_readonly(xt), mat3_c2_readonly(y));
	mat3_cpy(out, r);
}
static inline void
mat4_mul(mat4_t out, const mat4_t x, const mat4_t y) {
	mat4_t xt, r;

	mat4_transpose(xt, x);
	mat4_00(r) = vec4_dot(mat4_c0_readonly(xt), mat4_c0_readonly(y));
	mat4_01(r) = vec4_dot(mat4_c1_readonly(xt), mat4_c0_readonly(y));
	mat4_02(r) = vec4_dot(mat4_c2_readonly(xt), mat4_c0_readonly(y));
	mat4_03(r) = vec4_dot(mat4_c3_readonly(xt), mat4_c0_readonly(y));

	mat4_10(r) = vec4_dot(mat4_c0_readonly(xt), mat4_c1_readonly(y));
	mat4_11(r) = vec4_dot(mat4_c1_readonly(xt), mat4_c1_readonly(y));
	mat4_12(r) = vec4_dot(mat4_c2_readonly(xt), mat4_c1_readonly(y));
	mat4_13(r) = vec4_dot(mat4_c3_readonly(xt), mat4_c1_readonly(y));

	mat4_20(r) = vec4_dot(mat4_c0_readonly(xt), mat4_c2_readonly(y));
	mat4_21(r) = vec4_dot(mat4_c1_readonly(xt), mat4_c2_readonly(y));
	mat4_22(r) = vec4_dot(mat4_c2_readonly(xt), mat4_c2_readonly(y));
	mat4_23(r) = vec4_dot(mat4_c3_readonly(xt), mat4_c2_readonly(y));

	mat4_30(r) = vec4_dot(mat4_c0_readonly(xt), mat4_c3_readonly(y));
	mat4_31(r) = vec4_dot(mat4_c1_readonly(xt), mat4_c3_readonly(y));
	mat4_32(r) = vec4_dot(mat4_c2_readonly(xt), mat4_c3_readonly(y));
	mat4_33(r) = vec4_dot(mat4_c3_readonly(xt), mat4_c3_readonly(y));
	mat4_cpy(out, r);
}
#define mat_mul(out,t,x) \
	_Generic((out), vec2_t:mat2_vec2_mul,\
	                vec3_t:mat3_vec3_mul,\
	                vec4_t:mat4_vec4_mul,\
	                mat2_t:mat2_mul,\
	                mat3_t:mat3_mul,\
	                mat4_t:mat4_mul)\
	                ((out),(t),(x));


/* 2x2 rotate */
static inline void
mat2_rotate(mat2_t m, float angle_radians) {
	float c = cos(angle_radians);
	float s = sin(angle_radians);

	mat2_00(m) = c;
	mat2_01(m) = s;
		mat2_10(m) = -s;
		mat2_11(m) =  c;
}


/* rotatex */
static inline void
mat3_rotate_x(mat3_t m, float angle_radians) {
	float c = cos(angle_radians);
	float s = sin(angle_radians);

	mat3_00(m) = 1.f;
	mat3_01(m) = 0.f;
	mat3_02(m) = 0.f;
		mat3_10(m) = 0.f;
		mat3_11(m) = c;
		mat3_12(m) = s;
			mat3_20(m) = 0.f;
			mat3_21(m) = -s;
			mat3_22(m) =  c;
}
static inline void
mat4_rotate_x(mat4_t m, float angle_radians) {
	float c = cos(angle_radians);
	float s = sin(angle_radians);

	mat4_00(m) = 1.f;
	mat4_01(m) = 0.f;
	mat4_02(m) = 0.f;
	mat4_03(m) = 0.f;
		mat4_10(m) = 0.f;
		mat4_11(m) = c;
		mat4_12(m) = s;
		mat4_13(m) = 0.f;
			mat4_20(m) = 0.f;
			mat4_21(m) = -s;
			mat4_22(m) =  c;
			mat4_23(m) = 0.f;
				mat4_30(m) = 0.f;
				mat4_31(m) = 0.f;
				mat4_32(m) = 0.f;
				mat4_00(m) = 1.f;
}
#define mat_rotate_x(m,angle_radians) \
	VEC_DECL_FUNC2(rotate_x,(m),(angle_radians))


/* rotatey */
static inline void
mat3_rotate_y(mat3_t m, float angle_radians) {
	float c = cos(angle_radians);
	float s = sin(angle_radians);

	mat3_00(m) = c;
	mat3_01(m) = 0.f;
	mat3_02(m) = -s;
		mat3_10(m) = 0.f;
		mat3_11(m) = 1.f;
		mat3_12(m) = 0.f;
			mat3_20(m) = s;
			mat3_21(m) = 0.f;
			mat3_22(m) = c;
}
static inline void
mat4_rotate_y(mat4_t m, float angle_radians) {
	float c = cos(angle_radians);
	float s = sin(angle_radians);

	mat4_00(m) = c;
	mat4_01(m) = 0.f;
	mat4_02(m) = -s;
	mat4_03(m) = 0.f;
		mat4_10(m) = 0.f;
		mat4_11(m) = 1.f;
		mat4_12(m) = 0.f;
		mat4_13(m) = 0.f;
			mat4_20(m) = s;
			mat4_21(m) = 0.f;
			mat4_22(m) = c;
			mat4_23(m) = 0.f;
				mat4_30(m) = 0.f;
				mat4_31(m) = 0.f;
				mat4_32(m) = 0.f;
				mat4_00(m) = 1.f;
}
#define mat_rotate_y(m,angle_radians)  \
	VEC_DECL_FUNC2(rotate_y,(m),(angle_radians))


/* rotatez */
static inline void
mat3_rotate_z(mat3_t m, float angle_radians) {
	float c = cos(angle_radians);
	float s = sin(angle_radians);

	mat3_00(m) = c;
	mat3_01(m) = s;
	mat3_02(m) = 0.f;
		mat3_10(m) = -s;
		mat3_11(m) =  c;
		mat3_12(m) = 0.f;
			mat3_20(m) = 0.f;
			mat3_21(m) = 0.f;
			mat3_22(m) = 1.f;
}
static inline void
mat4_rotate_z(mat4_t m, float angle_radians) {
	float c = cos(angle_radians);
	float s = sin(angle_radians);

	mat4_00(m) = c;
	mat4_01(m) = s;
	mat4_02(m) = 0.f;
	mat4_03(m) = 0.f;
		mat4_10(m) = -s;
		mat4_11(m) =  c;
		mat4_12(m) = 0.f;
		mat4_13(m) = 0.f;
			mat4_20(m) = 0.f;
			mat4_21(m) = 0.f;
			mat4_22(m) = 1.f;
			mat4_23(m) = 0.f;
				mat4_30(m) = 0.f;
				mat4_31(m) = 0.f;
				mat4_32(m) = 0.f;
				mat4_00(m) = 1.f;
}
#define mat_rotate_z(m,angle_radians)  \
	VEC_DECL_FUNC2(rotate_z,(m),(angle_radians))


/* quaternion extraction */
static inline void
mat3_from_quat(mat3_t out, const quat_t q) {
	float X2,Y2,Z2;      //2*QX, 2*QY, 2*QZ
	float XX2,YY2,ZZ2;   //2*QX*QX, 2*QY*QY, 2*QZ*QZ
	float XY2,XZ2,XW2;   //2*QX*QY, 2*QX*QZ, 2*QX*QW
	float YZ2,YW2,ZW2;   // ...

	X2  = 2.0f * q[0];
	XX2 = X2   * q[0];
	XY2 = X2   * q[1];
	XZ2 = X2   * q[2];
	XW2 = X2   * q[3];

	Y2  = 2.0f * q[1];
	YY2 = Y2   * q[1];
	YZ2 = Y2   * q[2];
	YW2 = Y2   * q[3];

	Z2  = 2.0f * q[2];
	ZZ2 = Z2   * q[2];
	ZW2 = Z2   * q[3];

	mat3_00(out) = 1.0f - YY2 - ZZ2;
	mat3_01(out) = XY2  - ZW2;
	mat3_02(out) = XZ2  + YW2;
		mat3_10(out) = XY2  + ZW2;
		mat3_11(out) = 1.0f - XX2 - ZZ2;
		mat3_12(out) = YZ2  - XW2;
			mat3_20(out) = XZ2  - YW2;
			mat3_21(out) = YZ2  + XW2;
			mat3_22(out) = 1.0f - XX2 - YY2;
}

static inline void
mat4_from_quat(mat4_t out, const quat_t q) {
	mat3_t m3;

	mat3_from_quat(m3, q);
	mat4_00(out) = mat3_00(m3);
	mat4_01(out) = mat3_01(m3);
	mat4_02(out) = mat3_02(m3);
	mat4_03(out) = 0.f;
		mat4_10(out) = mat3_10(m3);
		mat4_11(out) = mat3_11(m3);
		mat4_12(out) = mat3_12(m3);
		mat4_13(out) = 0.f;
			mat4_20(out) = mat3_20(m3);
			mat4_21(out) = mat3_21(m3);
			mat4_22(out) = mat3_22(m3);
			mat4_23(out) = 0.f;
				mat4_30(out) = 0.f;
				mat4_31(out) = 0.f;
				mat4_32(out) = 0.f;
				mat4_33(out) = 1.f;
}


/* ortho */
static inline void
mat4_ortho(mat4_t m,
           float left, float right, float bottom,
           float top, float znear, float zfar) {
	assert(left!=right && bottom!=top && znear!=zfar);
	float c1 = 1.f/(right-left);
	float c2 = 1.f/(top-bottom);
	float c3 = 1.f/(zfar-znear);

	mat4_00(m) = 2.f * c1;
	mat4_01(m) = 0.f;
	mat4_02(m) = 0.f;
	mat4_03(m) = 0.f;
		mat4_10(m) = 0.f;
		mat4_11(m) = 2.f * c2;
		mat4_12(m) = 0.f;
		mat4_13(m) = 0.f;
			mat4_20(m) = 0.f;
			mat4_21(m) = 0.f;
			mat4_22(m) = -2.f*c3;
			mat4_23(m) = 0.f;
				mat4_30(m) = -(right+left) * c1;
				mat4_31(m) = -(top+bottom) * c2;
				mat4_32(m) = -(zfar+znear) * c3;
				mat4_33(m) = 1.f;
}


/* ortho from vector */
static inline void
mat4_orthov(mat4_t m, const float *fv) {
	mat4_ortho(m, fv[0], fv[1], fv[2], fv[3], fv[4], fv[5]);
}


/* frustum */
static inline void
mat4_frustum(mat4_t m, float left, float right, float bottom,
                       float top, float znear, float zfar) {
	assert(left!=right && bottom!=top && znear<zfar && znear>0.f);
	float c1 = 1.f / (right - left);
	float c2 = 1.f / (top - bottom);
	float c3 = 1.f / (zfar - znear);
	float c4 = 2.f * znear;

	mat4_00(m) = c4 * c1;
	mat4_01(m) = 0.f;
	mat4_02(m) = 0.f;
	mat4_03(m) = 0.f;
		mat4_10(m) = 0.f;
		mat4_11(m) = c4 * c2;
		mat4_12(m) = 0.f;
		mat4_13(m) = 0.f;
			mat4_20(m) = (right + left) * c1;
			mat4_21(m) = (top + bottom) * c2;
			mat4_22(m) = -(zfar + znear) * c3;
			mat4_23(m) = -1.f;
				mat4_30(m) = 0.f;
				mat4_31(m) = 0.f;
				mat4_32(m) = -c4 * c3 * zfar;
				mat4_33(m) = 0.f;
}


/* frustum from vector */
static inline void
mat4_frustumv(mat4_t m, const float *fv) {
	mat4_frustum(m, fv[0], fv[1], fv[2], fv[3], fv[4], fv[5]);
}


/* perspective */
static inline void
mat4_perspective(mat4_t m, float fovy, float aspect, 
                           float znear, float zfar) {
	assert(fovy>0.f && aspect>0.f && znear<zfar && znear>0.f);
	float f = 1.f / tan(fovy*0.5f);
	float c = 1.f / (znear-zfar);

	mat4_00(m) = f/aspect;
	mat4_01(m) = 0.f;
	mat4_02(m) = 0.f;
	mat4_03(m) = 0.f;
		mat4_10(m) = 0.f;
		mat4_11(m) = f;
		mat4_12(m) = 0.f;
		mat4_13(m) = 0.f;
			mat4_20(m) = 0.f;
			mat4_21(m) = 0.f;
			mat4_22(m) = (zfar+znear)*c;
			mat4_23(m) = -1.f;
				mat4_30(m) = 0.f;
				mat4_31(m) = 0.f;
				mat4_32(m) = 2.f*znear*zfar*c;
				mat4_33(m) = 0.f;
}


/* perspective from vector */
static inline void
mat4_perspectivev(mat4_t m, const float *fv) {
	mat4_perspective(m, fv[0], fv[1], fv[2], fv[3]);
}

/* quaternion identity */
static inline void
quat_identity(quat_t q) {
	q[0] = q[1] = q[2] = 0.f; q[3] = 1.f;
}

/* quaternion copy */
static inline void
quat_cpy(quat_t dst, const quat_t src) {
	vec4_cpy(dst, src);
}

/* quaternion normalization */
static inline void
quat_normalize(quat_t out, const quat_t q) {
	vec4_normalize(out, q);
}

/* quaternion inverse */
static inline void
quat_inverse(quat_t out, const quat_t q) {
	out[0] = -q[0];
	out[1] = -q[1];
	out[2] = -q[2];
	out[3] =  q[3];
}

/* quaternion multiplication */
static inline void
quat_mul(quat_t out, const quat_t x, const quat_t y) {
	quat_t r;

	r[0] = x[3]*y[0] + x[0]*y[3] + x[1]*y[2] - x[2]*y[1];
	r[1] = x[3]*y[1] - x[0]*y[2] + x[1]*y[3] + x[2]*y[0];
	r[2] = x[3]*y[2] + x[0]*y[1] - x[1]*y[0] + x[2]*y[3];
	r[3] = x[3]*y[3] - x[0]*y[0] - x[1]*y[1] - x[2]*y[2];
	quat_cpy(out, r);
}
static inline void
quat_vec3_mul(vec3_t out, const quat_t q, const vec3_t v) {
	quat_t qvec = { v[0], v[1], v[2], 0 };
	quat_t qinv = { -q[0], -q[1], -q[2], q[3] };
	quat_t temp;

	quat_mul(temp, q, qvec);
	quat_mul(temp, qinv, temp);
	out[0] = temp[0];
	out[1] = temp[1];
	out[2] = temp[2];
}

/* quaternion linear interpolation */
static inline void
quat_mix(quat_t out, quat_t x, quat_t y, float a) {
	vec4_mix(out, x, y, a);
}
static inline void
quat_mix_normalize(quat_t out, quat_t x, quat_t y, float a) {
	quat_mix(out, x, y, a);
	quat_normalize(out, out);
}

/* quaternion rotations */
static inline void
quat_rotate_x(quat_t out, float angle_radians) {
	out[0] = sin(angle_radians*0.5f);
	out[1] = out[2] = 0.f;
	out[3] = cos(angle_radians*0.5f);
}
static inline void
quat_rotate_y(quat_t out, float angle_radians) {
	out[1] = sin(angle_radians*0.5f);
	out[0] = out[2] = 0.f;
	out[3] = cos(angle_radians*0.5f);
}
static inline void
quat_rotate_z(quat_t out, float angle_radians) {
	out[2] = sin(angle_radians*0.5f);
	out[0] = out[1] = 0.f;
	out[3] = cos(angle_radians*0.5f);
}

#endif //VEC_H

