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

#ifndef KLMATERIAL_H
#define KLMATERIAL_H

#include <vector>
#include "Effect.h"

enum klMaterialSort {
    MS_FIRST,
    MS_EARLY,
    MS_NORMAL,
    MS_LATE,
    MS_LAST,
    MS_POST
};

class klMaterial {
protected:
    klEffect *effect;
    klMaterial *shadowMaterial;
    bool defaulted;
    klMaterialSort sort;
    
    struct TextureBinding {
        std::string name;
        klEffect::ParamHandle param;
        klTexture *tex;
    };

    std::vector<TextureBinding> textures;

    struct ValueBinding {
        int offset;
        float value[4];
        int len;
    };

    std::vector<ValueBinding> values;

public:

    klMaterial( const char *effectName );

    void setTextureParam(const char *paramName, klTexture *tex);   
    void setParam(const char *paramName, float *vec, int numVec); 
    void updateParamHandles(void);

    bool isDefault(void) {
        return defaulted;
    }

    klMaterial *getShadowMaterial(void) {
        return shadowMaterial;
    }

    klMaterialSort getSort(void) {
        return sort;
    }

    // Sets up the gl state for rendering the given pass
    // All these passes need to be rendered during the ambient/fill rendering stage
    //PassIterator setupFirstPass(void);
    //PassIterator setupNextPass(PassIterator iter);
    //void resetPass(PassIterator iter);
    
    // For backend
    void setup(void);
    void reset(void);
    void setupInstance(klInstanceParameters &instance);
    

    // All these passes need to be rendered for all lights.
    // returns null if no per light passes.
   // PassIterator setupFirstLightPass(void);
   // PassIterator setupLightPass(PassIterator iter);

    friend class klMaterialManager;
};


//////////////////////////////////////////////////////////////


class klMaterialManager : public klManager<klMaterial> {
protected:
	virtual klMaterial *getInstance(const char *name);
    klMaterial *default;
public:
    void parseMaterialFile(std::istream &stream);
    
    // Gets all the mtr files and parses them
    void loadMaterials(void);

    // After the effects got reloaded call this to
    // request the parameter handles again.
    void updateParamHandles(void);
};

extern klMaterialManager materialManager;

#endif //KLMATERIAL_H