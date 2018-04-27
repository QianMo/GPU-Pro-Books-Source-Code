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

/**
Copyright (c) 2003-2004 Charles Hollemeersch

-Not for commercial use without written permission
-This code should not be redistributed or made public
-This code is distributed without any warranty

*/

#include "Maths.h"
#include "Camera.h"

void klCamera::setFov(float f) { 
	fov = f;
}

void klCamera::extractFrustumPlanes(void) const { 
}

klMatrix4x4 klCamera::getViewMatrix(void) const {
    klVec3 normal = rotation.getXAxis();
    klVec3 up = rotation.getYAxis();
    klVec3 side = rotation.getZAxis();

    klMatrix4x4 rot
      (   side[0],   side[1],   side[2],  0,
            up[0],     up[1],     up[2],  0,
       -normal[0],-normal[1],-normal[2],  0,
                0,         0,         0,  1);

    klMatrix4x4 trans = klTranslationMatrix(-origin);

   klMatrix4x4 matView = rot*trans;
    return matView;
}

klMatrix4x4 klCamera::getProjMatrix(void) const {
    klMatrix4x4 matProj;
	if (perspective) {
        float f = cotan((float)(fov / 180.0f * M_PI)/2.0f);

        float t1 = (farp+nearp)/(nearp-farp);
        float t2 = (2.0f*farp*nearp)/(nearp-farp);

        matProj = klMatrix4x4(
            f/aspect,    0,    0,    0,
                   0,    f,    0,    0,
                   0,    0,   t1,   t2, 
                   0,    0,   -1,    0);
	} else {
        float tx = (orthoMin[0] + orthoMax[0]) / (orthoMax[0] - orthoMin[0]);
        float ty = (orthoMin[1] + orthoMax[1]) / (orthoMax[1] - orthoMin[1]);
        float tz = (orthoMin[2] + orthoMax[2]) / (orthoMax[2] - orthoMin[2]);

        matProj = klMatrix4x4(
             2.0f / (orthoMax[0] - orthoMin[0]), 0, 0, tx,
            0,  2.0f / (orthoMax[1] - orthoMin[1]), 0, ty,
            0, 0, -2.0f / (orthoMax[2] - orthoMin[2]), tz,
            0, 0, 0, 1);                               
	}
	return matProj;
}

ClipResult klCamera::clipBox(const klVec3 &mins,const klVec3 &maxs) const {
    return CLIP_IN;
}

float klCamera::getBoxDist(const klVec3 &mins,const klVec3 &maxs) {
    return 0;
}
