
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

#ifndef _FRUSTUM_H_
#define _FRUSTUM_H_

#include "Vector.h"

#define FRUSTUM_LEFT   0
#define FRUSTUM_RIGHT  1
#define FRUSTUM_TOP    2
#define FRUSTUM_BOTTOM 3
#define FRUSTUM_FAR    4
#define FRUSTUM_NEAR   5

struct Plane {
	Plane(){}
	Plane(const float x, const float y, const float z, const float o){
		normal = vec3(x, y, z);
		float invLen = 1.0f / length(normal);
		normal *= invLen;
		offset = o * invLen;
	}

	float dist(const vec3 &pos) const {
		return dot(normal, pos) + offset;
	}

	vec3 normal;
	float offset;
};

class Frustum {
public:
    void loadFrustum(const mat4 &mvp);

    bool pointInFrustum(const vec3 &pos) const;
    bool sphereInFrustum(const vec3 &pos, const float radius) const;
    bool cubeInFrustum(const float minX, const float maxX, const float minY, const float maxY, const float minZ, const float maxZ) const;

	const Plane &getPlane(const int plane) const { return planes[plane]; }

protected:
	Plane planes[6];
};


#endif // _FRUSTUM_H_
