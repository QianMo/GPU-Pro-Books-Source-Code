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

#ifndef KL_MODELLOADUTIL_H
#define KL_MODELLOADUTIL_H

/**
Copyright (c) 2003-2008 Charles Hollemeersch

-Not for commercial use without written permission.
-This code should not be redistributed or made public.
-This code is distributed without any warranty.

------------------------------------------------------

Utilities used by various model loaders, this is not 
exposed when you include "Klubnika.h".
*/

#include <vector>
#include "vectors.h"
#include "Model.h"

struct DiscSurface {
    std::vector<klVec3>   *xyz;
    std::vector<klVec2>   *uv;
    std::vector<klVec3>   *normal;
    std::vector<klVec3>   *tangent;
    std::vector<klVec3>   *binormal;
    std::vector<klVec3>   *color;
    std::vector<int>      *xyzFaces;
    std::vector<int>      *uvFaces;
    std::vector<int>      *normalFaces;
    std::vector<int>      *tangentFaces;
    std::vector<int>      *binormalFaces;
    std::vector<int>      *colorFaces;
    char                  material[64];

    DiscSurface(void) :
        xyz(NULL), uv(NULL), normal(NULL), tangent(NULL), binormal(NULL), color(NULL), xyzFaces(NULL), uvFaces(NULL),
        normalFaces(NULL), tangentFaces(NULL), binormalFaces(NULL), colorFaces(NULL) {
            strcpy(material,"default");
        }   
};

typedef std::vector<DiscSurface> DiscSurfaceList;

// Converts the raw disc model in an optimized model for rendering.
// missing elements will be generated. (Like normals, tangent spaces,..)
void ProcessDiscModel( DiscSurfaceList &dsl, klModel * dstModel );

#endif //KL_MODELLOADUTIL_H