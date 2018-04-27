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

#ifndef KLMODEL_H
#define KLMODEL_H

#include <vector>
#include "GlBuffer.h"
#include "Material.h"

struct klSurface {
    klVertexBuffer *vertices;
    klIndexBuffer  *indices;
    klMaterial     *material;
};

typedef std::vector<klSurface> klSurfaceList;

class klModel {
    klSurfaceList surfaces;
public:

    void addSurface(const klSurface &surface) { surfaces.push_back(surface); }
    int numSurfaces(void) const { return (int)surfaces.size(); }
    klSurface &getSurface(int index) { return surfaces[index]; }
    const klSurface &getSurface(int index) const { return surfaces[index]; }

    // This just dumps the model to opengl for easy rendering
    void render();

    // This enques the model to the backend
    void backendRender() {}
};

class klModelManager : public klManager<klModel> {
protected:
	virtual klModel *getInstance(const char *name);
};

extern klModelManager modelManager;

#endif KLMODEL_H