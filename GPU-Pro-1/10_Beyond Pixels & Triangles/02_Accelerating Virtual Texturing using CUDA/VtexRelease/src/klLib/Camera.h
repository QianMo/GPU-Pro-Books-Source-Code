/**
 *
 *  This software module was originally developed for research purposes,
 *  by Multimedia Lab at Ghent University (Belgium).
 *  Its performance may not be optimized for specific applications.
 *
 *  Those intending to use this software module in hardware or software products
 *  are advized that its use may infringe existing patents. The developers of 
 *  this software module, their companies, Ghent Universtity, nor Multimedia Lab 
 *  have any liability for use of this software module or modifications thereof.
 *
 *  Ghent University and Multimedia Lab (Belgium) retain full right to modify and
 *  use the code for their own purpose, assign or donate the code to a third
 *  party, and to inhibit third parties from using the code for their products. 
 *
 *  This copyright notice must be included in all copies or derivative works.
 *
 *  For information on its use, applications and associated permission for use,
 *  please contact Prof. Rik Van de Walle (rik.vandewalle@ugent.be). 
 *
 *  Detailed information on the activities of
 *  Ghent University Multimedia Lab can be found at
 *  http://multimedialab.elis.ugent.be/.
 *
 *  Copyright (c) Ghent University 2004-2009.
 *
 **/

#ifndef __KLCAMERA_H
#define __KLCAMERA_H

/**
Copyright (c) 2003-2008 Charles Hollemeersch

-Not for commercial use without written permission
-This code should not be redistributed or made public
-This code is distributed without any warranty

*/

#include "Matrix.h"
#include "Plane.h"

/**
	A camera class, renderer independent
*/
class klCamera {
private:
	klVec3 origin;
	klMatrix3x3 rotation; // X = forward, Y = up, Z = side

	float fov;
	float nearp;
	float farp;
	float aspect;

	klVec3 orthoMin, orthoMax;

	void extractFrustumPlanes (void) const;

	bool perspective;
public:

	klCamera(void) {

		perspective = true;
        origin.zero();
		rotation.setIdentity();
		aspect = 1.0f;
		nearp = 5.0f;
		farp = 40000.0f;

        setFov( 60.0f );
	}

    void setTransform(const klMatrix4x4 &transform) {
        origin = transform.getTranslation();
        rotation = transform.getRotation();
    }

	void setOrigin(const klVec3 &o) { origin = o; }
	klVec3 getOrigin(void) const { return origin;}

    void setRotation(const klMatrix3x3 &rot) { rotation = rot; }
    const klMatrix3x3 &getRotation(void) { return rotation; }

	void setDirection(const klVec3 &d, const klVec3 &up = klVec3(0.0,1.0,0.0) ) {
        klVec3 s = d.cross(up);
        s.normalize();
        klVec3 u = s.cross(d);
        u.normalize();
        rotation = klAxisMatrix(d,u,s);
    }

	klVec3 getDirection(void) const {
        return rotation.getXAxis();
    }

	klVec3 getUp(void) const {
        return rotation.getYAxis();
    }

	void setFov(float f);
	float getFov(void) const { return fov; }

	void setNear(float f) { nearp = f;}
	float getNear(void) const { return nearp; }

	void setFar(float f) { farp = f;}
	float getFar(void) const { return farp; }

	void setAspect(float f) { aspect = f; }
	float getAspect(void) const { return aspect; }

	void setOrtho(klVec3 min, klVec3 max) {
		perspective = false;
		orthoMin = min;
		orthoMax = max;
	}

	klMatrix4x4 getViewMatrix(void) const;
	klMatrix4x4 getProjMatrix(void) const;

	ClipResult clipBox(const klVec3 &mins,const klVec3 &maxs) const;

	/**
		Returns the distance between the nearest point on the box and the near camera plane
		Algorithm is based on the box/plane side test. (see AutumnPlane.h)
		Returns 0 if the box is behind or interects the near plane.
		(Also note that the near plane normal as we calculate it is pointing outside of the frustum
		so we actually want to calculate the distance the box is behind the plane...)
	*/
	float getBoxDist(const klVec3 &mins,const klVec3 &maxs);
	
};

#endif //__KLCAMERA_H