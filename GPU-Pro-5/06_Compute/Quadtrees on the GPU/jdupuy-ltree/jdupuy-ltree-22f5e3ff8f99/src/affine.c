#include <stdio.h>
#include <stdlib.h>
#include "vec.h"
#include "affine.h"

//
struct affine_t {
	mat3_t axis;
	vec3_t pos;
	float scale;
};

/*
normalize axis
*/
static void affine_normalize_axis(struct affine_t *affine) {
	vec3_normalize(mat3_c0(affine->axis), mat3_c0_readonly(affine->axis));
	vec3_normalize(mat3_c1(affine->axis), mat3_c1_readonly(affine->axis));
	vec3_normalize(mat3_c2(affine->axis), mat3_c2_readonly(affine->axis));
}

/*
construction
*/
struct affine_t *affine_create() {
	struct affine_t *affine = malloc(sizeof(*affine));

	if(!affine) {
		fprintf(stderr, "affine: malloc failed\n");
		return NULL;
	}
	affine_identity(affine);
	return affine;
}

/*
destruction
*/
void affine_release(struct affine_t *affine) {
	free(affine);
}

/*
identity
*/
void affine_identity(struct affine_t *affine) {
	mat3_identity(affine->axis);
	vec3_set(affine->pos, 0.f);
	affine->scale = 1.f;
}

/*
translations
*/
void affine_translate_world(struct affine_t *affine, const vec3_t dir) {
	vec3_add(affine->pos, affine->pos, dir);
}

void affine_translate_local(struct affine_t *affine, const vec3_t dir) {
	vec3_t t;

	mat3_vec3_mul(t, affine->axis, dir);
	vec3_add(affine->pos, affine->pos, t);
}

/*
rotations
*/
void affine_rotatex_world(struct affine_t *affine, float radians) {
	mat3_t r;

	mat3_rotate_x(r, radians);
	mat3_mul(affine->axis, r, affine->axis);
	affine_normalize_axis(affine);
}

void affine_rotatey_world(struct affine_t *affine, float radians)  {
	mat3_t r;

	mat3_rotate_y(r,radians);
	mat3_mul(affine->axis, r, affine->axis);
	affine_normalize_axis(affine);
}

void affine_rotatez_world(struct affine_t *affine, float radians)  {
	mat3_t r;

	mat3_rotate_z(r,radians);
	mat3_mul(affine->axis, r, affine->axis);
	affine_normalize_axis(affine);
}

void affine_rotatex_local(struct affine_t *affine, float radians)  {
	mat3_t r;

	mat3_rotate_x(r, radians);
	mat3_mul(affine->axis, affine->axis, r);
	affine_normalize_axis(affine);
}

void affine_rotatey_local(struct affine_t *affine, float radians)  {
	mat3_t r;

	mat3_rotate_y(r, radians);
	mat3_mul(affine->axis, affine->axis, r);
	affine_normalize_axis(affine);
}

void affine_rotatez_local(struct affine_t *affine, float radians)  {
	mat3_t r;

	mat3_rotate_z(r, radians);
	mat3_mul(affine->axis, affine->axis, r);
	affine_normalize_axis(affine);
}

/*
scale
*/
void affine_scale(struct affine_t *affine, float factor) {
	if(affine->scale > 0.f)
		affine->scale*= factor;
	assert(affine->scale > 0.f);
}

/*
lookats
*/
void
affine_xlookat(struct affine_t *affine, const vec3_t target, const vec3_t up) {
	vec3_t x, y, z;

	vec3_sub(x, target, affine->pos);
	vec3_normalize(x, x);
	vec3_cross(y, x, up);
	vec3_cross(z, y, x);
	vec3_cpy(mat3_c0(affine->axis), x);
	vec3_cpy(mat3_c0(affine->axis), y);
	vec3_cpy(mat3_c2(affine->axis), z);
	affine_normalize_axis(affine);
}

void
affine_ylookat(struct affine_t *affine, const vec3_t target, const vec3_t up) {
	vec3_t x, y, z;

	vec3_sub(y, target, affine->pos);
	vec3_normalize(y, y);
	vec3_cross(x, y, up);
	vec3_cross(z, x, y);
	vec3_cpy(mat3_c0(affine->axis), x);
	vec3_cpy(mat3_c1(affine->axis), y);
	vec3_cpy(mat3_c2(affine->axis), z);
	affine_normalize_axis(affine);
}

void
affine_zlookat(struct affine_t *affine, const vec3_t target, const vec3_t up) {
	vec3_t x, y, z;

	vec3_sub(z, target, affine->pos);
	vec3_normalize(z, z);
	vec3_cross(x, z, up);
	vec3_cross(y, x, z);
	vec3_cpy(mat3_c0(affine->axis), x);
	vec3_cpy(mat3_c1(affine->axis), y);
	vec3_cpy(mat3_c2(affine->axis), z);
	affine_normalize_axis(affine);
}

/*
queries
*/
void affine_get_position(const struct affine_t *affine, vec3_t pos) {
	vec3_cpy(pos, affine->pos);
}

void affine_get_axis(const struct affine_t *affine, mat3_t axis) {
	mat3_cpy(axis, affine->axis);
}

void affine_get_scale(const struct affine_t *affine, float *scale) {
	*scale = affine->scale;
}

/*
matrix extract
*/
void affine_matrix(const struct affine_t *affine, mat4_t out) {
	vec3_cpy(mat4_c0(out), mat3_c0_readonly(affine->axis));
	vec3_cpy(mat4_c1(out), mat3_c1_readonly(affine->axis));
	vec3_cpy(mat4_c2(out), mat3_c2_readonly(affine->axis));
	vec3_cpy(mat4_c3(out), affine->pos);
	mat4_03(out) = 0.f;
	mat4_13(out) = 0.f;
	mat4_23(out) = 0.f;
	mat4_33(out) = 1.f;
}
void affine_inverse_matrix(const struct affine_t *affine, mat4_t out) {
	affine_matrix(affine, out);
	mat4_inverse(out, out);
}

/*
load
*/
int affine_load(struct affine_t *affine, const char *filename) {
	FILE *file = fopen(filename, "r");
	float *data = (float *) affine;
	int i = 0;

	if(!file) {
		perror("affine: fopen failed: ");
		return AFFINE_ERR;
	}
	for(;i<13;++i)
		fscanf(file, "%f\n", data+i);
	fclose(file);
	return AFFINE_OK;
}

/*
save
*/
int affine_save(const struct affine_t *affine, const char *filename) {
	FILE *file = fopen(filename, "w");
	const float *data = (const float *) affine;
	int i = 0;

	if(!file) {
		perror("affine: fopen failed: ");
		return AFFINE_ERR;
	}
	for(;i<13;++i)
		fprintf(file, "%f\n", data[i]);
	fclose(file);
	return AFFINE_OK;
}

