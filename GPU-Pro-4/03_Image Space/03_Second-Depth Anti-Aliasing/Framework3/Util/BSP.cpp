
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

#include "BSP.h"

#ifdef _WIN32
#pragma warning(push, 1)
#pragma warning(disable: 4799)
#endif

no_alias vec3 planeHit(const vec3 &v0, const vec3 &v1, const vec4 &plane){
	vec3 dir = v1 - v0;
	float d = planeDistance(plane, v0);
	vec3 pos = v0 - (d / dot((vec3 &) plane, dir)) * dir;

	return pos;
}

void BTri::split(BTri *dest, int &nPos, int &nNeg, const vec4 &plane, const float epsilon) const {
	float d[3];
	for (int i = 0; i < 3; i++){
		d[i] = planeDistance(plane, v[i]);
	}

	int first  = 2;
	int second = 0;
	while (!(d[second] > epsilon && d[first] <= epsilon)){
		first = second;
		second++;
	}

	// Positive triangles
	nPos = 0;
	vec3 h = planeHit(v[first], v[second], plane);
	do {
		first = second;
		second++;
		if (second >= 3) second = 0;

		dest->v[0] = h;
		dest->v[1] = v[first];
		if (d[second] > epsilon){
			dest->v[2] = v[second];
		} else {
			dest->v[2] = h = planeHit(v[first], v[second], plane);
		}

		dest->data = data;
		dest->finalize();
		dest++;
		nPos++;
	} while (d[second] > epsilon);

	// Skip zero area triangle
	if (fabsf(d[second]) <= epsilon){
		first = second;
		second++;
		if (second >= 3) second = 0;
	}

	// Negative triangles
	nNeg = 0;
	do {
		first = second;
		second++;
		if (second >= 3) second = 0;

		dest->v[0] = h;
		dest->v[1] = v[first];
		if (d[second] < -epsilon){
			dest->v[2] = v[second];
		} else {
			dest->v[2] = planeHit(v[first], v[second], plane);
		}

		dest->data = data;
		dest->finalize();
		dest++;
		nNeg++;
	} while (d[second] < -epsilon);
}

void BTri::finalize(){
	vec3 normal = normalize(cross(v[1] - v[0], v[2] - v[0]));
	float offset = -dot(v[0], normal);

    vec3 edgeNormals[3];
	edgeNormals[0] = cross(normal, v[0] - v[2]);
	edgeNormals[1] = cross(normal, v[1] - v[0]);
	edgeNormals[2] = cross(normal, v[2] - v[1]);

	float edgeOffsets[3];
	edgeOffsets[0] = dot(edgeNormals[0], v[0]);
	edgeOffsets[1] = dot(edgeNormals[1], v[1]);
	edgeOffsets[2] = dot(edgeNormals[2], v[2]);

	plane = vec4(normal, offset);
	edgePlanes[0] = vec4(edgeNormals[0], -edgeOffsets[0]);
	edgePlanes[1] = vec4(edgeNormals[1], -edgeOffsets[1]);
	edgePlanes[2] = vec4(edgeNormals[2], -edgeOffsets[2]);
}

no_alias bool BTri::intersects(const vec3 &v0, const vec3 &v1) const {
	vec3 dir = v0 - v1;
//	float k = (dot(normal, v0) + offset) / dot(normal, dir);
	float k = planeDistance(plane, v0) / dot(plane.xyz(), dir);

	if (k < 0 || k > 1) return false;

	vec3 pos = v0 - k * dir;

	for (unsigned int i = 0; i < 3; i++){
		if (planeDistance(edgePlanes[i], pos) < 0){
//		if (dot(edgeNormals[i], pos) < edgeOffsets[i]){
			return false;
		}
	}
	return true;
}

no_alias bool BTri::isAbove(const vec3 &pos) const {
/*
	return (edgeNormals[0].x * pos.x + edgeNormals[0].y * pos.y + edgeNormals[0].z * pos.z >= edgeOffsets[0] &&
			edgeNormals[1].x * pos.x + edgeNormals[1].y * pos.y + edgeNormals[1].z * pos.z >= edgeOffsets[1] &&
			edgeNormals[2].x * pos.x + edgeNormals[2].y * pos.y + edgeNormals[2].z * pos.z >= edgeOffsets[2]);
*/
/*
	return (edgePlanes[0].x * pos.x + edgePlanes[0].y * pos.y + edgePlanes[0].z * pos.z >= -edgePlanes[0].w &&
			edgePlanes[1].x * pos.x + edgePlanes[1].y * pos.y + edgePlanes[1].z * pos.z >= -edgePlanes[1].w &&
			edgePlanes[2].x * pos.x + edgePlanes[2].y * pos.y + edgePlanes[2].z * pos.z >= -edgePlanes[2].w);
*/
	return (planeDistance(edgePlanes[0], pos) >= 0 && planeDistance(edgePlanes[1], pos) >= 0 && planeDistance(edgePlanes[2], pos) >= 0);
}

no_alias float BTri::getDistance(const vec3 &pos) const {
	int k = 2;
	for (int i = 0; i < 3; i++){
		float d = planeDistance(edgePlanes[i], pos);
		if (d < 0){
			// Project onto the line between the points
			vec3 dir = v[i] - v[k];
			float c = dot(dir, pos - v[k]) / dot(dir, dir);

			vec3 d;
			if (c >= 1){
				d = v[i];
			} else {
				d = v[k];
				if (c > 0) d += c * dir;
			}

			return length(pos - d);
		}

		k = i;
	}

	return fabsf(planeDistance(plane, pos));
}


#ifdef USE_SIMD
bool BTri::isAbove3DNow(v2sf v0XY, v2sf v0Z1) const {
	for (int i = 0; i < 3; i++){
		v2sf planeXY = ((v2sf *) &edgePlanes[i])[0];
		v2sf planeZD = ((v2sf *) &edgePlanes[i])[1];

		v2sf dotXY = pfmul(planeXY, v0XY);
		v2sf dotZD = pfmul(planeZD, v0Z1);
		v2sf dot = pfacc(dotXY, dotZD);
		dot = pfacc(dot, dot);

		int d = _m_to_int(dot);
		if (d < 0) return false;
	}

	return true;
}
#endif

BNode::~BNode(){
    delete back;
	delete front;
}

no_alias bool BNode::intersects(const vec3 &v0, const vec3 &v1, const vec3 &dir, vec3 *point, const BTri **triangle) const {
#if 0
	float d0 = planeDistance(tri.plane, v0);
	float d1 = planeDistance(tri.plane, v1);

	vec3 pos;
	if (d0 > 0){
		if (d1 <= 0){
			pos = v0 - (d0 / dot(tri.plane.xyz(), dir)) * dir;
		}

		if (front != NULL && front->intersects(v0, (d1 <= 0)? pos : v1, dir, point, triangle)) return true;

		if (d1 <= 0){
			if (tri.isAbove(pos)){
				if (point) *point = pos;
				if (triangle) *triangle = &tri;
				return true;
			}
			if (back != NULL && back->intersects(pos, v1, dir, point, triangle)) return true;
		}
	} else {
		if (d1 > 0){
			pos = v0 - (d0 / dot(tri.plane.xyz(), dir)) * dir;
		}

		if (back != NULL && back->intersects(v0, (d1 > 0)? pos : v1, dir, point, triangle)) return true;

		if (d1 > 0){
			if (tri.isAbove(pos)){
				if (point) *point = pos;
				if (triangle) *triangle = &tri;
				return true;
			}
			if (front != NULL && front->intersects(pos, v1, dir, point, triangle)) return true;
		}
	}

#else
	float d = planeDistance(tri.plane, v0);

	if (d > 0){
		if (front != NULL && front->intersects(v0, v1, dir, point, triangle)) return true;
		if (planeDistance(tri.plane, v1) < 0){
			vec3 pos = v0 - (d / dot(tri.plane.xyz(), dir)) * dir;
			if (tri.isAbove(pos)){
				if (point) *point = pos;
				if (triangle) *triangle = &tri;
				return true;
			}
			if (back != NULL && back->intersects(v0, v1, dir, point, triangle)) return true;
		}
	} else {
		if (back != NULL && back->intersects(v0, v1, dir, point, triangle)) return true;
		if (planeDistance(tri.plane, v1) > 0){
			vec3 pos = v0 - (d / dot(tri.plane.xyz(), dir)) * dir;
			if (tri.isAbove(pos)){
				if (point) *point = pos;
				if (triangle) *triangle = &tri;
				return true;
			}
			if (front != NULL && front->intersects(v0, v1, dir, point, triangle)) return true;
		}
	}
#endif

	return false;
}

no_alias BTri *BNode::intersectsCached(const vec3 &v0, const vec3 &v1, const vec3 &dir) const {
#if 0
	float d0 = planeDistance(tri.plane, v0);
	float d1 = planeDistance(tri.plane, v1);

	vec3 pos;

	if (d0 > 0){
		if (d1 <= 0){
			pos = v0 - (d0 / dot((vec3 &) tri.plane, dir)) * dir;
		}

		if (front != NULL){
			BTri *tri;
			if (d1 <= 0){
				tri = front->intersectsCached(v0, pos, dir);
			} else {
				tri = front->intersectsCached(v0, v1, dir);
			}
			if (tri) return tri;
		}

		if (d1 <= 0){
			if (tri.isAbove(pos)) return (BTri *) &tri;
			if (back != NULL){
				BTri *tri = back->intersectsCached(pos, v1, dir);
				if (tri) return tri;
			}
		}
	} else {
		if (d1 > 0){
			pos = v0 - (d0 / dot((vec3 &) tri.plane, dir)) * dir;
		}
		if (back != NULL){
			BTri *tri;
			if (d1 > 0){
				tri = back->intersectsCached(v0, pos, dir);
			} else {
				tri = back->intersectsCached(v0, v1, dir);
			}
			if (tri) return tri;
		}
		if (d1 > 0){
			if (tri.isAbove(pos)) return (BTri *) &tri;
			if (front != NULL){
				BTri *tri = front->intersectsCached(pos, v1, dir);
				if (tri) return tri;
			}
		}
	}

#else

	float d = planeDistance(tri.plane, v0);

	if (d > 0){
		if (front != NULL){
			BTri *tri = front->intersectsCached(v0, v1, dir);
			if (tri) return tri;
		}
		if (planeDistance(tri.plane, v1) < 0){
			vec3 pos = v0 - (d / dot(tri.plane.xyz(), dir)) * dir;
			if (tri.isAbove(pos)) return (BTri *) &tri;
			if (back != NULL){
				BTri *tri = back->intersectsCached(v0, v1, dir);
				if (tri) return tri;
			}
		}
	} else {
		if (back != NULL){
			BTri *tri = back->intersectsCached(v0, v1, dir);
			if (tri) return tri;
		}
		if (planeDistance(tri.plane, v1) > 0){
			vec3 pos = v0 - (d / dot(tri.plane.xyz(), dir)) * dir;
			if (tri.isAbove(pos)) return (BTri *) &tri;
			if (front != NULL){
				BTri *tri = front->intersectsCached(v0, v1, dir);
				if (tri) return tri;
			}
		}
	}
#endif

	return NULL;
}

#ifdef USE_SIMD
bool BNode::intersects3DNow(const vec4 &v0, const vec4 &v1, const vec4 &dir) const {
	v2sf planeXY = ((v2sf *) &tri.plane)[0];
	v2sf planeZD = ((v2sf *) &tri.plane)[1];

	v2sf v0XY = ((v2sf *) &v0)[0];
	v2sf v0Z1 = ((v2sf *) &v0)[1];

	v2sf dotXY = pfmul(planeXY, v0XY);
	v2sf dotZD = pfmul(planeZD, v0Z1);
	v2sf dotD = pfacc(dotXY, dotZD);
	dotD = pfacc(dotD, dotD);

	int d = _m_to_int(dotD);

	if (d > 0){
		if (front != NULL && front->intersects3DNow(v0, v1, dir)) return true;

		v2sf dotXY = pfmul(planeXY, ((v2sf *) &v1)[0]);
		v2sf dotZD = pfmul(planeZD, ((v2sf *) &v1)[1]);
		v2sf dot = pfacc(dotXY, dotZD);
		dot = pfacc(dot, dot);

		int d = _m_to_int(dot);
		if (d < 0){
			v2sf dirXY = ((v2sf *) &dir)[0];
			v2sf dirZ0 = ((v2sf *) &dir)[1];

			v2sf dotXY = pfmul(planeXY, dirXY);
			v2sf dotZ0 = pfmul(planeZD, dirZ0);
			v2sf dot = pfacc(dotXY, dotZ0);
			dot = pfacc(dot, dot);
			dot = pfrcp(dot);
			dot = pfmul(dot, dotD);

			dirXY = pfmul(dirXY, dot);
			dirZ0 = pfmul(dirZ0, dot);

			v0XY = pfsub(v0XY, dirXY);
			v0Z1 = pfsub(v0Z1, dirZ0);

			if (tri.isAbove3DNow(v0XY, v0Z1)){
				return true;
			}

			if (back != NULL && back->intersects3DNow(v0, v1, dir)) return true;
		}

	} else {
		if (back != NULL && back->intersects3DNow(v0, v1, dir)) return true;

		v2sf dotXY = pfmul(planeXY, ((v2sf *) &v1)[0]);
		v2sf dotZD = pfmul(planeZD, ((v2sf *) &v1)[1]);
		v2sf dot = pfacc(dotXY, dotZD);
		dot = pfacc(dot, dot);
		
		int d = _m_to_int(dot);
		if (d > 0){
			v2sf dirXY = ((v2sf *) &dir)[0];
			v2sf dirZ0 = ((v2sf *) &dir)[1];

			v2sf dotXY = pfmul(planeXY, dirXY);
			v2sf dotZ0 = pfmul(planeZD, dirZ0);
			v2sf dot = pfacc(dotXY, dotZ0);
			dot = pfacc(dot, dot);
			dot = pfrcp(dot);
			dot = pfmul(dot, dotD);

			dirXY = pfmul(dirXY, dot);
			dirZ0 = pfmul(dirZ0, dot);

			v0XY = pfsub(v0XY, dirXY);
			v0Z1 = pfsub(v0Z1, dirZ0);

			if (tri.isAbove3DNow(v0XY, v0Z1)){
				return true;
			}
			if (front != NULL && front->intersects3DNow(v0, v1, dir)) return true;
		}
	}
	return false;
}
#endif

no_alias bool BNode::pushSphere(vec3 &pos, const float radius) const {
	float d = planeDistance(tri.plane, pos);

	bool pushed = false;
	if (fabsf(d) < radius){
		if (tri.isAbove(pos)){
//			pos += (radius - d) * tri.normal; 
			pos += (radius - d) * tri.plane.xyz();
			pushed = true;
		}
	}

	if (front != NULL && d > -radius) pushed |= front->pushSphere(pos, radius);
	if (back  != NULL && d <  radius) pushed |= back ->pushSphere(pos, radius);

	return pushed;
}

no_alias void BNode::getDistance(const vec3 &pos, float &minDist) const {
	float d = planeDistance(tri.plane, pos);

	float dist = tri.getDistance(pos);
	if (dist < minDist){
		minDist = dist;
	}
	
	if (back && d < minDist){
		back->getDistance(pos, minDist);		
	}

	if (front && -d < minDist){
		front->getDistance(pos, minDist);
	}
}

void BNode::read(FILE *file){
	fread(&tri.v, sizeof(tri.v), 1, file);
	tri.finalize();

	int flags = 0;
	fread(&flags, sizeof(int), 1, file);
	if (flags & 1){
		back = new BNode;
		back->read(file);
	} else back = NULL;
	if (flags & 2){
		front = new BNode;
		front->read(file);
	} else front = NULL;
}

void BNode::write(FILE *file) const {
	fwrite(&tri.v, sizeof(tri.v), 1, file);
	int flags = 0;
	if (back) flags |= 1;
	if (front) flags |= 2;
	fwrite(&flags, sizeof(int), 1, file);
	if (back) back->write(file);
	if (front) front->write(file);
}
/*
void BNode::build(Array <BTri> &tris){
	uint index = 0;
	int minScore = 0x7FFFFFFF;
	for (uint i = 0; i < tris.getCount(); i++){
		int score = 0;
		int diff = 0;
		for (uint k = 0; k < tris.getCount(); k++){
			uint neg = 0, pos = 0;
			for (uint j = 0; j < 3; j++){
//				float dist = dot(tris[k].v[j], tris[i].normal) + tris[i].offset;
				float dist = planeDistance(tris[i].plane, tris[k].v[j]);
				if (dist < 0) neg++; else pos++;
			}
			if (pos){
				if (neg) score += 3; else diff++;
			} else diff--;
		}
		score += abs(diff);
		if (score < minScore){
			minScore = score;
			index = i;
		}
	}
	tri = tris[index];
	tris.fastRemove(index);

	Array <BTri> backTris;
	Array <BTri> frontTris;
	for (uint i = 0; i < tris.getCount(); i++){
		uint neg = 0, pos = 0;
		for (uint j = 0; j < 3; j++){
//			float dist = dot(tris[i].v[j], tri.normal) + tri.offset;
			float dist = planeDistance(tri.plane, tris[i].v[j]);
            if (dist < 0) neg++; else pos++;
		}
		if (neg) backTris.add(tris[i]);
		if (pos) frontTris.add(tris[i]);
	}
	//tris.clear();

	if (backTris.getCount() > 0){
		back = new BNode;
		back->build(backTris);
	} else back = NULL;

	if (frontTris.getCount() > 0){
		front = new BNode;
		front->build(frontTris);
	} else front = NULL;
}
*/

void BNode::build(Array <BTri> &tris, const int splitCost, const int balCost, const float epsilon){
	uint index = 0;
	int minScore = 0x7FFFFFFF;

	for (uint i = 0; i < tris.getCount(); i++){
		int score = 0;
		int diff = 0;
		for (uint k = 0; k < tris.getCount(); k++){
			uint neg = 0, pos = 0;
			for (uint j = 0; j < 3; j++){
				float dist = planeDistance(tris[i].plane, tris[k].v[j]);
				if (dist < -epsilon) neg++; else
				if (dist >  epsilon) pos++;
			}
			if (pos){
				if (neg) score += splitCost; else diff++;
			} else {
				if (neg) diff--; else diff++;
			}
		}
		score += balCost * abs(diff);
		if (score < minScore){
			minScore = score;
			index = i;
		}
	}

	tri = tris[index];
	tris.fastRemove(index);

	Array <BTri> backTris;
	Array <BTri> frontTris;
	for (uint i = 0; i < tris.getCount(); i++){

		uint neg = 0, pos = 0;
		for (uint j = 0; j < 3; j++){
			float dist = planeDistance(tri.plane, tris[i].v[j]);
            if (dist < -epsilon) neg++; else
			if (dist >  epsilon) pos++;
		}

		if (neg){
			if (pos){
				BTri newTris[3];
				int nPos, nNeg;
				tris[i].split(newTris, nPos, nNeg, tri.plane, epsilon);
				for (int i = 0; i < nPos; i++){
					frontTris.add(newTris[i]);
				}
				for (int i = 0; i < nNeg; i++){
					backTris.add(newTris[nPos + i]);
				}
			} else {
				backTris.add(tris[i]);
			}
		} else {
			frontTris.add(tris[i]);
		}
	}
	tris.reset();

	if (backTris.getCount() > 0){
		back = new BNode;
		back->build(backTris, splitCost, balCost, epsilon);
	} else back = NULL;

	if (frontTris.getCount() > 0){
		front = new BNode;
		front->build(frontTris, splitCost, balCost, epsilon);
	} else front = NULL;
}

#ifdef USE_SIMD
void SSENode::build(const BNode *node, SSENode *&sseDest){
	tri.plane = loadups((const float *) &node->tri.plane);
	for (int i = 0; i < 3; i++){
		tri.edgePlanes[i] = loadups((const float *) &node->tri.edgePlanes[i]);
	}
	if (node->front){
        front = sseDest++;
		front->build(node->front, sseDest);
	} else front = NULL;
	if (node->back){
        back = sseDest++;
		back->build(node->back, sseDest);
	} else back = NULL;
}
#endif

void BSP::addTriangle(const vec3 &v0, const vec3 &v1, const vec3 &v2, void *data){
	BTri tri;

	tri.v[0] = v0;
	tri.v[1] = v1;
	tri.v[2] = v2;
	tri.data = data;

	tri.finalize();

	tris.add(tri);
}

void BSP::build(const int splitCost, const int balCost, const float epsilon){
//	int nTris = tris.getCount();

	top = new BNode;
//	top->build(tris);
	top->build(tris, splitCost, balCost, epsilon);
/*
	SSENode *mem = new SSENode[nTris * 4];

	sseTop = (SSENode *) ((intptr(mem) + 15) & ~intptr(0xF));
	sseDest = sseTop + 1;
	sseTop->build(top, sseDest);

	sseDest = mem;
*/
}

no_alias bool BSP::intersects(const vec3 &v0, const vec3 &v1, vec3 *point, const BTri **triangle) const {
	if (top != NULL) return top->intersects(v0, v1, v1 - v0, point, triangle);

	return false;
}

bool BSP::intersectsCached(const vec3 &v0, const vec3 &v1){
	if (top != NULL){
		if (cache){
			if (cache->intersects(v0, v1)) return true;
		}
		cache = top->intersectsCached(v0, v1, v1 - v0);
		return (cache != NULL);
	}

	return false;
}

#ifdef USE_SIMD
bool BSP::intersects3DNow(const vec3 &v0, const vec3 &v1) const {
	if (top != NULL){
		femms();

		vec4 v04 = vec4(v0, 1);
		vec4 v14 = vec4(v1, 1);

		bool result = top->intersects3DNow(v04, v14, v14 - v04);

		femms();
		return result;
	}

	return false;
}
#endif

bool BSP::pushSphere(vec3 &pos, const float radius) const {
	if (top != NULL) return top->pushSphere(pos, radius);

	return false;
}

no_alias float BSP::getDistance(const vec3 &pos) const {
	float dist = FLT_MAX;

	if (top != NULL) top->getDistance(pos, dist);

	return dist;
}


no_alias bool BSP::isInOpenSpace(const vec3 &pos) const {
	if (top != NULL){

		BNode *node = top;
		while (true){
			float d = planeDistance(node->tri.plane, pos);

			if (d > 0){
				if (node->front){
					node = node->front;
				} else return true;
			} else {
				if (node->back){
					node = node->back;
				} else return false;
			}
		}
	}

	return false;
}

#ifdef USE_SIMD
bool BSP::isInOpenSpace3DNow(const vec3 &pos) const {
	if (top != NULL){
		SSENode *node = sseTop;
		femms();

		v2sf posXY, posZ1;
		posXY = *(v2sf *) &pos.x;
		posZ1.m64_f32[0] = pos.z;
		posZ1.m64_f32[1] = 1.0f;

		while (true){
			v2sf planeXY = ((v2sf *) &node->tri.plane)[0];
			v2sf planeZD = ((v2sf *) &node->tri.plane)[1];

			v2sf dotXY = pfmul(planeXY, posXY);
			v2sf dotZD = pfmul(planeZD, posZ1);
			v2sf dot = pfacc(dotXY, dotZD);
			dot = pfacc(dot, dot);

			int d = _m_to_int(dot);

			if (d > 0){
				if (node->front){
					node = node->front;
				} else {
					femms();
					return true;
				}
			} else {
				if (node->back){
					node = node->back;
				} else {
					femms();
					return false;
				}
			}
		}
	}

	return false;
}

bool BSP::isInOpenSpaceSSE(const vec3 &pos) const {
	if (top != NULL){
		v4sf zero = setzerops();
		v4sf pos4 = loadups((const float *) &vec4(pos, 1));

		SSENode *node = sseTop;
		while (true){
			v4sf d = dot4(node->tri.plane, pos4);

			if (comigt(d, zero)){
				if (node->front){
					node = node->front;
				} else return true;
			} else {
				if (node->back){
					node = node->back;
				} else return false;
			}
		}
	}

	return false;
}
#endif

bool BSP::loadFile(const char *fileName){
	FILE *file = fopen(fileName, "rb");
	if (file == NULL) return false;

	delete top;

	top = new BNode;
	top->read(file);
	fclose(file);

	return true;
}

bool BSP::saveFile(const char *fileName) const {
	if (top == NULL) return false;

	FILE *file = fopen(fileName, "wb");
	if (file == NULL) return false;

	top->write(file);
	fclose(file);

	return true;
}

#ifdef _WIN32
#pragma warning(pop)
#endif
