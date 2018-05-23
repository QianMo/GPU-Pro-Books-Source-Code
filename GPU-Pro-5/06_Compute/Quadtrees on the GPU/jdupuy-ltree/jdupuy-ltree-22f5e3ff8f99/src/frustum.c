#include "vec.h"
#include "frustum.h"

static void normalize_plane(vec4_t plane) {
	float l = vec3_length(plane);
	vec4_t m = {l, l, l, l};

	vec4_mul(plane, plane, m);
}

void frustum_extract_mvp(vec4_t frustum[6], const mat4_t mvp) {
	// left
	frustum[0][0] = mat4_03(mvp) + mat4_00(mvp);
	frustum[0][1] = mat4_13(mvp) + mat4_10(mvp);
	frustum[0][2] = mat4_23(mvp) + mat4_20(mvp);
	frustum[0][3] = mat4_33(mvp) + mat4_30(mvp);
	normalize_plane(frustum[0]);

	// right
	frustum[1][0] = mat4_03(mvp) - mat4_00(mvp);
	frustum[1][1] = mat4_13(mvp) - mat4_10(mvp);
	frustum[1][2] = mat4_23(mvp) - mat4_20(mvp);
	frustum[1][3] = mat4_33(mvp) - mat4_30(mvp);
	normalize_plane(frustum[1]);

	// top
	frustum[2][0] = mat4_03(mvp) + mat4_01(mvp);
	frustum[2][1] = mat4_13(mvp) + mat4_11(mvp);
	frustum[2][2] = mat4_23(mvp) + mat4_21(mvp);
	frustum[2][3] = mat4_33(mvp) + mat4_31(mvp);
	normalize_plane(frustum[2]);

	// bottom
	frustum[3][0] = mat4_03(mvp) - mat4_01(mvp);
	frustum[3][1] = mat4_13(mvp) - mat4_11(mvp);
	frustum[3][2] = mat4_23(mvp) - mat4_21(mvp);
	frustum[3][3] = mat4_33(mvp) - mat4_31(mvp);
	normalize_plane(frustum[3]);

	// near
	frustum[4][0] = mat4_03(mvp) + mat4_02(mvp);
	frustum[4][1] = mat4_13(mvp) + mat4_12(mvp);
	frustum[4][2] = mat4_23(mvp) + mat4_22(mvp);
	frustum[4][3] = mat4_33(mvp) + mat4_32(mvp);
	normalize_plane(frustum[4]);

	// far
	frustum[5][0] = mat4_03(mvp) - mat4_02(mvp);
	frustum[5][1] = mat4_13(mvp) - mat4_12(mvp);
	frustum[5][2] = mat4_23(mvp) - mat4_22(mvp);
	frustum[5][3] = mat4_33(mvp) - mat4_32(mvp);
	normalize_plane(frustum[5]);
}

