// affine.h - public domain C affine library
#ifndef AFFINE_H
#define AFFINE_H

/* construction / destruction */
struct affine_t *affine_create(void);
void affine_release(struct affine_t *affine);

/* identity */
void affine_identity(struct affine_t *affine);

/* translations */
void affine_translate_world(struct affine_t *affine, const vec3_t dir);
void affine_translate_local(struct affine_t *affine, const vec3_t dir);

/* rotations */
void affine_rotatex_world(struct affine_t *affine, float radians);
void affine_rotatey_world(struct affine_t *affine, float radians);
void affine_rotatez_world(struct affine_t *affine, float radians);
void affine_rotatex_local(struct affine_t *affine, float radians);
void affine_rotatey_local(struct affine_t *affine, float radians);
void affine_rotatez_local(struct affine_t *affine, float radians);

/* scale */
void affine_scale(struct affine_t *affine, float factor);

/* look ats */
void affine_xlookat(struct affine_t *affine,
                    const vec3_t target_pos,
                    const vec3_t unit_up);
void affine_ylookat(struct affine_t *affine,
                    const vec3_t target_pos,
                    const vec3_t unit_up);
void affine_zlookat(struct affine_t *affine,
                    const vec3_t target_pos,
                    const vec3_t unit_up);

/* queries */
void affine_get_position(const struct affine_t *affine, vec3_t pos);
void affine_get_axis(const struct affine_t *affine, mat3_t axis);
void affine_get_scale(const struct affine_t *affine, float *scale);

/* matrix extraction */
void affine_matrix(const struct affine_t *affine, mat4_t out);
void affine_inverse_matrix(const struct affine_t *affine, mat4_t out);

/* load / save */
#define AFFINE_ERR (-1)
#define AFFINE_OK (0)
int affine_load(struct affine_t *affine, const char *filename);
int affine_save(const struct affine_t *affine, const char *filename);

#endif //AFFINE_H

