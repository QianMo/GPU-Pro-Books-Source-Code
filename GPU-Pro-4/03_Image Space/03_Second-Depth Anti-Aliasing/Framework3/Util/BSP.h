
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

#ifndef _BSP_H_
#define _BSP_H_

#ifdef USE_SIMD
#include "../Math/SIMD.h"
#endif

#include "../Platform.h"
#include "../Math/Vector.h"
#include "Array.h"
#include <stdio.h>


struct BTri {
	void split(BTri *dest, int &nPos, int &nNeg, const vec4 &plane, const float epsilon) const;
	void finalize();

	bool intersects(const vec3 &v0, const vec3 &v1) const;

	bool isAbove(const vec3 &pos) const;
#ifdef USE_SIMD
	bool isAbove3DNow(v2sf v0XY, v2sf v0Z1) const;
#endif
	float getDistance(const vec3 &pos) const;

	vec4 plane;
	vec4 edgePlanes[3];

	vec3 v[3];
/*
	vec3 edgeNormals[3];
	float edgeOffsets[3];
	vec3 normal;
	float offset;
*/
	void *data;
};

struct BNode {
	~BNode();

	bool intersects(const vec3 &v0, const vec3 &v1, const vec3 &dir, vec3 *point, const BTri **triangle) const;
	BTri *intersectsCached(const vec3 &v0, const vec3 &v1, const vec3 &dir) const;
#ifdef USE_SIMD
	bool intersects3DNow(const vec4 &v0, const vec4 &v1, const vec4 &dir) const;
#endif

	bool pushSphere(vec3 &pos, const float radius) const;
	void getDistance(const vec3 &pos, float &minDist) const;

	void build(Array <BTri> &tris, const int splitCost, const int balCost, const float epsilon);
	//void build(Array <BTri> &tris);

	void read(FILE *file);
	void write(FILE *file) const;

	
	BNode *back;
	BNode *front;

	BTri tri;
};

#ifdef USE_SIMD
struct SSETri {
    v4sf plane;
	v4sf edgePlanes[3];
};

align(16) struct SSENode {
	SSETri tri;

	SSENode *back;
	SSENode *front;

    void build(const BNode *node, SSENode *&sseDest);
};
#endif

class BSP {
public:
	BSP(){
		top = NULL;
#ifdef USE_SIMD
		sseTop = NULL;
		sseDest = NULL;
#endif
		cache = NULL;
	}
	~BSP(){
#ifdef USE_SIMD
		delete sseDest;
#endif
		delete top;
	}

	void addTriangle(const vec3 &v0, const vec3 &v1, const vec3 &v2, void *data = NULL);
	void build(const int splitCost = 3, const int balCost = 1, const float epsilon = 0.001f);

	bool intersects(const vec3 &v0, const vec3 &v1, vec3 *point = NULL, const BTri **triangle = NULL) const;
	bool intersectsCached(const vec3 &v0, const vec3 &v1);
#ifdef USE_SIMD
	bool intersects3DNow(const vec3 &v0, const vec3 &v1) const;
#endif

	bool pushSphere(vec3 &pos, const float radius) const;
	float getDistance(const vec3 &pos) const;

	bool isInOpenSpace(const vec3 &pos) const;
#ifdef USE_SIMD
	bool isInOpenSpace3DNow(const vec3 &pos) const;
	bool isInOpenSpaceSSE(const vec3 &pos) const;
#endif

	bool loadFile(const char *fileName);
	bool saveFile(const char *fileName) const;

protected:
	Array <BTri> tris;
	BNode *top;
	BTri *cache;
#ifdef USE_SIMD
	SSENode *sseTop;
	SSENode *sseDest;
#endif
};

#endif // _BSP_H_
