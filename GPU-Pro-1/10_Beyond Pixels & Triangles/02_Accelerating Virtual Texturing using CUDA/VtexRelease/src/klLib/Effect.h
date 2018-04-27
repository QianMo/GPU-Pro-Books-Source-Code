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

#ifndef KLEFFECT_H
#define KLEFFECT_H

#include "Vectors.h"
#include "Matrix.h"
#include "Texture.h"
#include <Cg/cg.h>
#include <Cg/cgGL.h>

struct klInstanceParameters {
    size_t      instanceId;
    klMatrix4x4 modelViewProjection;
    klMatrix4x4 modelToWorld;
    klMatrix4x4 worldToModel;
    klVec3      modelSpaceCamera;
    klVec3      modelSpaceLight;

    static const int NUM_PARAMETERS = 16;
    float userParams[NUM_PARAMETERS];
};

class klEffect {
    CGeffect handle;
    CGtechnique tech;
    size_t   currentInstance;

    CGparameter parmModelViewProjection;
    CGparameter parmModelToWorld;
    CGparameter parmWorldToModel;
    CGparameter parmModelCameraOrigin;
    CGparameter parmModelLightOrigin;
    CGparameter parmInvViewProjection;
    CGparameter parmInvProjection;
    CGparameter parmTime;
    CGparameter parmUserParms;

    CGparameter getBuildInSemanticParameter(const char *name);
    std::vector<CGprogram> instanceDependentPrograms;

    void loadResources( const char *effectName);
    void freeResources(void);

public:
    typedef CGparameter ParamHandle;

    klEffect(const char *name);

    inline ParamHandle getParameter(const char *name) {
        return cgGetNamedEffectParameter(handle,name);
    }

    inline void setParameter(ParamHandle handle, klTexture *texture) {
        cgGLSetTextureParameter(handle, texture->getHandle() );
    }

    inline void setParameter(ParamHandle handle, const klMatrix4x4 &mat) {
        cgGLSetMatrixParameterfc(handle, mat.toCPtr());
    }

    inline void setParameter(ParamHandle handle, const klVec4 &vec) {
        cgGLSetParameter4fv(handle, vec.toCPtr());
    }

    inline void setParameter(ParamHandle handle, const klVec3 &vec) {
        cgGLSetParameter3fv(handle, vec.toCPtr());
    }

    inline void setParameter(ParamHandle handle, const klVec2 &vec) {
        cgGLSetParameter2fv(handle, vec.toCPtr());
    }

    inline void setParameter(ParamHandle handle, float f) {
        cgGLSetParameter1f(handle, f);
    }

    void setup(void);
    void reset(void);
    void setupInstance(klInstanceParameters &inst);

    // These should only be used during development
    void reload(const char *effectName);
    void dumpCompiledPrograms(void);
};

class klEffectManager : public klManager<klEffect> {
    CGparameter parmInvProjection;
    CGparameter parmInvViewProjection;
    CGparameter parmTime;
protected:
	virtual klEffect *getInstance(const char *name);
public:
    void init(void);
    friend class klEffect;

    enum GlobalParamKind {
        PK_TIME,
        PK_INVPROJECTION,
        PK_INVVIEWPROJECTION
    };

    void setGlobalParameter(GlobalParamKind pk, const float *val);
};

extern klEffectManager effectManager;

#endif //KLEFFECT_H