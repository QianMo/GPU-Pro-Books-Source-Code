
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

#include "Frustum.h"

void Frustum::loadFrustum(const mat4 &mvp){
	planes[FRUSTUM_LEFT  ] = Plane(mvp[12] - mvp[0], mvp[13] - mvp[1], mvp[14] - mvp[2],  mvp[15] - mvp[3]);
	planes[FRUSTUM_RIGHT ] = Plane(mvp[12] + mvp[0], mvp[13] + mvp[1], mvp[14] + mvp[2],  mvp[15] + mvp[3]);

	planes[FRUSTUM_TOP   ] = Plane(mvp[12] - mvp[4], mvp[13] - mvp[5], mvp[14] - mvp[6],  mvp[15] - mvp[7]);
	planes[FRUSTUM_BOTTOM] = Plane(mvp[12] + mvp[4], mvp[13] + mvp[5], mvp[14] + mvp[6],  mvp[15] + mvp[7]);

	planes[FRUSTUM_FAR   ] = Plane(mvp[12] - mvp[8], mvp[13] - mvp[9], mvp[14] - mvp[10], mvp[15] - mvp[11]);
	planes[FRUSTUM_NEAR  ] = Plane(mvp[12] + mvp[8], mvp[13] + mvp[9], mvp[14] + mvp[10], mvp[15] + mvp[11]);
}

bool Frustum::pointInFrustum(const vec3 &pos) const {
	for (int i = 0; i < 6; i++){
		if (planes[i].dist(pos) <= 0) return false;
	}
    return true;
}

bool Frustum::sphereInFrustum(const vec3 &pos, const float radius) const {
	for (int i = 0; i < 6; i++){
		if (planes[i].dist(pos) <= -radius) return false;
	}
    return true;
}

bool Frustum::cubeInFrustum(const float minX, const float maxX, const float minY, const float maxY, const float minZ, const float maxZ) const {
    for (int i = 0; i < 6; i++){
		if (planes[i].dist(vec3(minX, minY, minZ)) > 0) continue;
		if (planes[i].dist(vec3(minX, minY, maxZ)) > 0) continue;
		if (planes[i].dist(vec3(minX, maxY, minZ)) > 0) continue;
		if (planes[i].dist(vec3(minX, maxY, maxZ)) > 0) continue;
		if (planes[i].dist(vec3(maxX, minY, minZ)) > 0) continue;
		if (planes[i].dist(vec3(maxX, minY, maxZ)) > 0) continue;
		if (planes[i].dist(vec3(maxX, maxY, minZ)) > 0) continue;
		if (planes[i].dist(vec3(maxX, maxY, maxZ)) > 0) continue;
        return false;
    }
    return true;
}
