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

#include "shared.h"
#include "Material.h"
#include "FileSystem.h"
#include "Console.h"
#include "cgfx_dump.h"


klMaterial::klMaterial( const char *effectName ) {
    effect = effectManager.getForName(effectName);
    defaulted = false;
    shadowMaterial = this;
    sort = MS_NORMAL;
}

void klMaterial::updateParamHandles(void) {
    for ( size_t i=0; i<textures.size(); i++ ) {
        textures[i].param = effect->getParameter(textures[i].name.c_str());
    }
}

void klMaterial::setTextureParam(const char *paramName, klTexture *tex) {
    CGparameter param = effect->getParameter(paramName);

    if ( param == NULL ) return;

    for ( size_t i=0; i<textures.size(); i++ ) {
        if ( textures[i].param == param ) {
            textures[i].tex = tex;
        }
    }

    TextureBinding binding;
    binding.param = param;
    binding.tex = tex;
    binding.name = paramName;

    textures.push_back(binding);
}

void klMaterial::setParam(const char * /*paramName*/, float * /*vec*/, int /*numVec*/) {
    /* nop */
}

void klMaterial::setup(void) {
    for ( size_t i=0; i<textures.size(); i++ ) {
        effect->setParameter(textures[i].param, textures[i].tex);
    }

    effect->setup();
}

void klMaterial::setupInstance(klInstanceParameters &instance) {
    effect->setupInstance(instance);
}

void klMaterial::reset(void) {
    effect->reset();
}

///////////////////////////////////////////////////////////////////////////////////////////

klMaterial *klMaterialManager::getInstance(const char *name) {
    // We cannot load these thy are parsed from material files
    if ( !default ) {
        klFatalError("Material '%s' not found and no default material was defined.",name);
        return NULL;
    } else {
        return new klMaterial(*default);
    }
}

void klMaterialManager::parseMaterialFile(std::istream &stream) {
    klStringStream ss(stream);
    char token[128];

    while ( ss.getToken(token) ) {
        if ( strcmp(token,"material") == 0 ) {

            // Material name
            char name[128];
            char effectName[128] = "default";
            ss.getToken(name);

            // Optional effect specifier
            ss.getToken(token);
            if ( strcmp(token,":") == 0 ) {
                ss.getToken(effectName); 
                if (!ss.expectToken("{")) return;
            } else  if ( strcmp(token,"{") != 0 ) {
                klFatalError("line %i: Expected { but found %s.", ss.getLineNo(), token);
            }

            klMaterial *material = new klMaterial(effectName);

            // Now parameters can be set these depend on the effect and others...
            while ( ss.getToken(token) && strcmp(token,"}") ) {
                char paramName[128];
                strcpy(paramName,token);
                
                if ( strcmp(paramName,"sort") == 0 ) {
                    ss.getToken(token);
                    if ( strcmp(token,"first") == 0 ) {
                        material->sort = MS_FIRST;
                    } else if ( strcmp(token,"early") == 0 ) {
                        material->sort = MS_EARLY;
                    } else if ( strcmp(token,"normal") == 0 ) {
                        material->sort = MS_NORMAL;
                    } else if ( strcmp(token,"late") == 0 ) {
                        material->sort = MS_LATE;
                    } else if ( strcmp(token,"last") == 0 ) {
                        material->sort = MS_LAST;
                    } else if ( strcmp(token,"post") == 0 ) {
                        material->sort = MS_POST;
                    }
                } else if ( strcmp(paramName,"shadowMaterial") == 0 ) {
                    ss.getToken(token);
                    if ( strcmp(token,"null") == 0 ) {
                        material->shadowMaterial = NULL;
                    } else {
                        material->shadowMaterial = materialManager.getForName(token);
                        if (material->shadowMaterial->isDefault()) {
                            klFatalError("Shadow material not found '%s', shadow materials need to be defined before actual material\n",token);
                        }
                    }
                } else {
                    // These are generic parameters set on the effect...
                    // the engine doesn't interpret them at al

                    ss.getToken(token);

                    // Texture params can be identified because they are followed by a map command
                    if ( strcmp(token,"map") == 0 ) {
                        // Load the texture program
                        if (!ss.expectToken("(")) return; 
                        std::string textureLoadString;
                        while ( ss.getToken(token) && strcmp(token,")") ) {
                            textureLoadString += token;
                        }
                        klTexture *tex = textureManager.getForName(textureLoadString.c_str());
                        if ( !tex ) {
                            klFatalError("line %i: Texture map('%s') not found.", ss.getLineNo(), textureLoadString.c_str());
                        }

                        // Set it as a parameter
                        material->setTextureParam(paramName, tex);                        
                    } else {
                        float vec[4];
                        int numVec = 0;

                        if ( strcmp(token,"(") == 0 ) {
                            while ( ss.getToken(token) && strcmp(token,")") ) {
                                vec[numVec] = (float)atof(token);
                                numVec++;
                                if (!ss.expectToken(",")) return; 
                            }
                        } else {
                            vec[0] = (float)atof(token);
                            numVec = 1;
                        }

                        material->setParam(paramName, vec, numVec);    
                    }
                }
            }

            objects[name] = material;
        } else {
            klFatalError("line %i: Unknown resource %s.", ss.getLineNo(), token);
        }
    }
}

void klMaterialManager::loadMaterials(void) {
    effectManager.init();

    //std::string  str = fileSystem.findFirst("*.mat");
    std::string  str = fileSystem.findFirst("base/materials/*.mat");
    while ( !str.empty() ) {
        std::istream *stream = fileSystem.openFile(str.c_str(),false);
        if (stream) {
            klLog("Parsing %s...",str.c_str());
            parseMaterialFile(*stream);
            delete stream;
        } else {
            klError("Error loading material file %s",str.c_str());
        }
        str = fileSystem.findNext();
    }

    default = getForName("default");
    default->defaulted = true;
}

void klMaterialManager::updateParamHandles(void) {
    for (std::map<std::string, klMaterial*>::iterator i = objects.begin(); i != objects.end(); i++) {
        if ( i->second ) {
            i->second->updateParamHandles();
        }
	}
}


klMaterialManager materialManager;