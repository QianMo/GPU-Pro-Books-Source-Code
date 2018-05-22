
/* * * * * * * * * * * * * Author's note * * * * * * * * * * * *\
*   _       _   _       _   _       _   _       _     _ _ _ _   *
*  |_|     |_| |_|     |_| |_|_   _|_| |_|     |_|  _|_|_|_|_|  *
*  |_|_ _ _|_| |_|     |_| |_|_|_|_|_| |_|     |_| |_|_ _ _     *
*  |_|_|_|_|_| |_|     |_| |_| |_| |_| |_|     |_|   |_|_|_|_   *
*  |_|     |_| |_|_ _ _|_| |_|     |_| |_|_ _ _|_|  _ _ _ _|_|  *
*  |_|     |_|   |_|_|_|   |_|     |_|   |_|_|_|   |_|_|_|_|    *
*                                                               *
*                     http://www.humus.name                     *
*                                                                *
* This file is a part of the work done by Humus. You are free to   *
* use the code in any way you like, modified, unmodified or copied   *
* into your own work. However, I expect you to respect these points:  *
*  - If you use this file and its contents unmodified, or use a major *
*    part of this file, please credit the author and leave this note. *
*  - For use in anything commercial, please request my approval.     *
*  - Share your work and ideas too as much as you can.             *
*                                                                *
\* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include "Scissor.h"
/*
bool getScissorRectangle(const mat4 &projection, const mat4 &modelview, const vec3 &camPos, const vec3 &lightPos, const float radius, const int width, const int height, int *x, int *y, int *w, int *h){

	float d = distance(camPos, lightPos);
	
	float p = d * radius / sqrtf(d * d - radius * radius);

	vec3 dx = modelview.rows[0].xyz();
	vec3 dy = modelview.rows[1].xyz();

//	vec3 dz = normalize(lightPos - camPos);
//	dx = cross(dy, dz);
//	dy = cross(dz, dx);

	vec3 dz = normalize(lightPos - camPos);
	vec3 dx = normalize(vec3(dz.z, 0, -dz.x));
	vec3 dy = normalize(cross(dz, dx));

	vec4 leftPos  = vec4(lightPos - p * dx, 1.0f);
	vec4 rightPos = vec4(lightPos + p * dx, 1.0f);

	mat4 mvp = projection * modelview;

	leftPos  = mvp * leftPos;
	rightPos = mvp * rightPos;

	int left  = int(width * (leftPos.x  / leftPos.z  * 0.5f + 0.5f));
	int right = int(width * (rightPos.x / rightPos.z * 0.5f + 0.5f));

	*x = left;
	*w = right - left;

	*y = 0;
	*h = height;

	return true;
}
*/

#define EPSILON 0.0005f

bool getScissorRectangle(const mat4 &modelview, const vec3 &pos, const float radius, const float fov, const int width, const int height, int *x, int *y, int *w, int *h){
	vec4 lightPos = modelview * vec4(pos, 1.0f);

	float ex = tanf(fov / 2);
	float ey = ex * height / width;

	float Lxz = (lightPos.x * lightPos.x + lightPos.z * lightPos.z);
	float a = -radius * lightPos.x / Lxz;
	float b = (radius * radius - lightPos.z * lightPos.z) / Lxz;
	float f = -b + a * a;

	float lp = 0;
	float rp = 1;
	float bp = 0;
	float tp = 1;

//	if (f > EPSILON){
	if (f > 0){
		float Nx0 = -a + sqrtf(f);
		float Nx1 = -a - sqrtf(f);
		float Nz0 = (radius - Nx0 * lightPos.x) / lightPos.z;
		float Nz1 = (radius - Nx1 * lightPos.x) / lightPos.z;

		float x0 = 0.5f * (1 - Nz0 / (Nx0 * ex));
		float x1 = 0.5f * (1 - Nz1 / (Nx1 * ex));

		float Pz0 = (Lxz - radius * radius) / (lightPos.z - lightPos.x * Nz0 / Nx0);
		float Pz1 = (Lxz - radius * radius) / (lightPos.z - lightPos.x * Nz1 / Nx1);

		float Px0 = -(Pz0 * Nz0) / Nx0;
		float Px1 = -(Pz1 * Nz1) / Nx1;

		if (Px0 > lightPos.x) rp = x0;
		if (Px0 < lightPos.x) lp = x0;
		if (Px1 > lightPos.x && x1 < rp) rp = x1;
		if (Px1 < lightPos.x && x1 > lp) lp = x1;
	}

	float Lyz = (lightPos.y * lightPos.y + lightPos.z * lightPos.z);
	a = -radius * lightPos.y / Lyz;
	b = (radius * radius - lightPos.z * lightPos.z) / Lyz;
	f = -b + a * a;

//	if (f > EPSILON){
	if (f > 0){
		float Ny0 = -a + sqrtf(f);
		float Ny1 = -a - sqrtf(f);
		float Nz0 = (radius - Ny0 * lightPos.y) / lightPos.z;
		float Nz1 = (radius - Ny1 * lightPos.y) / lightPos.z;

		float y0 = 0.5f * (1 - Nz0 / (Ny0 * ey));
		float y1 = 0.5f * (1 - Nz1 / (Ny1 * ey));

		float Pz0 = (Lyz - radius * radius) / (lightPos.z - lightPos.y * Nz0 / Ny0);
		float Pz1 = (Lyz - radius * radius) / (lightPos.z - lightPos.y * Nz1 / Ny1);

		float Py0 = -(Pz0 * Nz0) / Ny0;
		float Py1 = -(Pz1 * Nz1) / Ny1;

		if (Py0 > lightPos.y) tp = y0;
		if (Py0 < lightPos.y) bp = y0;
		if (Py1 > lightPos.y && y1 < tp) tp = y1;
		if (Py1 < lightPos.y && y1 > bp) bp = y1;
	}

	lp *= width;
	rp *= width;
	tp *= height;
	bp *= height;

	int left   = int(lp);
	int right  = int(rp);
	int top    = int(tp);
	int bottom = int(bp);

	if (right <= left || top <= bottom) return false;

	*x = min(max(int(left),   0), width  - 1);
	*y = min(max(int(bottom), 0), height - 1);
	*w = min(int(right) - *x, width  - *x);
	*h = min(int(top)   - *y, height - *y);

	return (*w > 0 && *h > 0);
}
