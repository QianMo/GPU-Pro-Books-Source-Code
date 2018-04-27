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
#include "Model.h"
#include "FileSystem.h"
#include "ModelLoadUtil.h"

/////////////////////////// OBJ file loader /////////////////////////////////

bool LoadObj(std::istream &stream, DiscSurfaceList &list, const char *materialName ) {
    klStringStream str(stream);
    char token[128];

    klVec3 temp3;
    klVec2 temp2;
    int tempInd;
    DiscSurface surface;
    surface.xyz         = new std::vector<klVec3>();
    surface.uv          = new std::vector<klVec2>();
    surface.normal      = new std::vector<klVec3>();
    surface.color       = new std::vector<klVec3>();
    surface.xyzFaces    = new std::vector<int>();
    surface.uvFaces     = new std::vector<int>();
    surface.normalFaces = new std::vector<int>();
    surface.colorFaces  = new std::vector<int>();

    int uvIndexOffset = 1;
    int vertexIndexOffset = 1;
    int normalIndexOffset = 1;
    
    bool seenFirstSurface = false;

    strcpy(surface.material,materialName);

    while ( str.getToken(token) ) {
        if ( strcmp(token,"#") == 0 ) {
            str.skipLine();
        } else if ( strcmp(token,"v") == 0 ) {
            for ( int i=0; i<3; i++ ) {
                str.getToken(token);
                temp3[i] = (float)atof(token);
            }
            surface.xyz->push_back(temp3);
        } else if ( strcmp(token,"vt") == 0 ) {
            for ( int i=0; i<2; i++ ) {
                str.getToken(token);
                temp2[i] = (float)atof(token);
            }
            str.getToken(token); // Discard 'w' component
            surface.uv->push_back(temp2);
        } else if ( strcmp(token,"vn") == 0 ) {
            for ( int i=0; i<3; i++ ) {
                str.getToken(token);
                temp3[i] = (float)atof(token);
            }
            temp3.normalize();
            surface.normal->push_back(temp3);
        } else if ( strcmp(token,"vc") == 0 ) { 
            for ( int i=0; i<3; i++ ) {
                str.getToken(token);
                temp3[i] = (float)atof(token);
            }
            surface.color->push_back(temp3);
        } else if ( strcmp(token,"f") == 0 ) {
            for ( int i=0; i<3; i++ ) {

                str.getToken(token);
                tempInd = atoi(token);
                surface.xyzFaces->push_back(tempInd-vertexIndexOffset);

                if ( !str.expectToken("/") ) {
                    return false;
                }

                str.getToken(token);

                if (strcmp(token,"/") != 0 ) {
                    tempInd = atoi(token);
                    surface.uvFaces->push_back(tempInd-uvIndexOffset);

                    if ( !str.expectToken("/") ) {
                        return false;
                    }
                } else {
                    // without texture coordinates
                    // Of the form xxx//xxx 
                    surface.uvFaces->push_back(0);
                }
               
                str.getToken(token);
                tempInd = atoi(token);
                surface.normalFaces->push_back(tempInd-normalIndexOffset);
            }

            surface.colorFaces->push_back(0);
            surface.colorFaces->push_back(0);
            surface.colorFaces->push_back(0);

        } else if ( strcmp(token,"nf") == 0 ) { 
            // New face
            int inds[12];
            for ( int i=0;i<12; i++) {
                str.getToken(token);
                inds[i] = atoi(token)-1;
            }

            for ( int i=0;i<12; i+=4) {
                surface.xyzFaces->push_back   (inds[i]);
                surface.uvFaces->push_back    (inds[i+1]);
                surface.normalFaces->push_back(inds[i+2]);
                surface.colorFaces->push_back (inds[i+3]);
            }
        } else if ( strcmp(token,"surface") == 0 ) { 
            
            str.getToken(token);

            if ( seenFirstSurface ) {
                // Save the surface we were working on
                list.push_back(surface);

                // Start a new one
                surface.xyz         = new std::vector<klVec3>();
                surface.uv          = new std::vector<klVec2>();
                surface.normal      = new std::vector<klVec3>();
                surface.color       = new std::vector<klVec3>();
                surface.xyzFaces    = new std::vector<int>();
                surface.uvFaces     = new std::vector<int>();
                surface.normalFaces = new std::vector<int>();
                surface.colorFaces  = new std::vector<int>();
                strcpy(surface.material,token);
            } else {
                seenFirstSurface = true;
                strcpy(surface.material,token);
            }
        }
    }

    if ( surface.color->size() == 0 ) {
        surface.color->push_back(klVec3(1.0f,1.0f,1.0f));
    }

    // and push the last surface
    list.push_back(surface);
    return true;
}

/////////////////////////// BOB file loader /////////////////////////////////

bool LoadBob(std::istream &stream, DiscSurfaceList &list, const char *materialName ) {
    DiscSurface surface;    
    size_t length;

    strcpy(surface.material,materialName);

    // Positions
    stream.read((char *)&length,sizeof(length));
    surface.xyz = new std::vector<klVec3>(length);
    for ( size_t i=0; i<length; i++ ) {
        stream.read((char *)&(*surface.xyz)[i],sizeof(klVec3));
    }

    stream.read((char *)&length,sizeof(length));
    surface.xyzFaces = new std::vector<int>(length*3);
    for ( size_t i=0; i<length*3; i++ ) {
        stream.read((char *)&(*surface.xyzFaces)[i],sizeof(int));
    }

    // UV's
    stream.read((char *)&length,sizeof(length));
    surface.uv = new std::vector<klVec2>(length);
    for ( size_t i=0; i<length; i++ ) {
        stream.read((char *)&(*surface.uv)[i],sizeof(klVec2));
    }

    stream.read((char *)&length,sizeof(length));
    surface.uvFaces = new std::vector<int>(length*3);
    for ( size_t i=0; i<length*3; i++ ) {
        stream.read((char *)&(*surface.uvFaces)[i],sizeof(int));
    }

    // Normals
    stream.read((char *)&length,sizeof(length));
    surface.normal = new std::vector<klVec3>(length);
    for ( size_t i=0; i<length; i++ ) {
        stream.read((char *)&(*surface.normal)[i],sizeof(klVec3));
    }

    stream.read((char *)&length,sizeof(length));
    if ( length == 0 ) {
        surface.normalFaces = new std::vector<int>(*surface.xyzFaces);
    } else {
        surface.normalFaces = new std::vector<int>(length*3);
        for ( size_t i=0; i<length*3; i++ ) {
            stream.read((char *)&(*surface.normalFaces)[i],sizeof(int));
        }
    }

    // Tangents
    stream.read((char *)&length,sizeof(length));
    surface.tangent = new std::vector<klVec3>(length);
    for ( size_t i=0; i<length; i++ ) {
        stream.read((char *)&(*surface.tangent)[i],sizeof(klVec3));
    }

    stream.read((char *)&length,sizeof(length));
    if ( length == 0 ) {
        surface.normalFaces = new std::vector<int>(*surface.xyzFaces);
    } else {
        surface.tangentFaces = new std::vector<int>(length*3);
        for ( size_t i=0; i<length*3; i++ ) {
            stream.read((char *)&(*surface.tangentFaces)[i],sizeof(int));
        }
    }

    // Binormals
    stream.read((char *)&length,sizeof(length));
    surface.binormal = new std::vector<klVec3>(length);
    for ( size_t i=0; i<length; i++ ) {
        stream.read((char *)&(*surface.binormal)[i],sizeof(klVec3));
    }

    stream.read((char *)&length,sizeof(length));
    if ( length == 0 ) {
        surface.normalFaces = new std::vector<int>(*surface.xyzFaces);
    } else {
        surface.binormalFaces = new std::vector<int>(length*3);
        for ( size_t i=0; i<length*3; i++ ) {
            stream.read((char *)&(*surface.binormalFaces)[i],sizeof(int));
        }
    }

    // Color
    surface.color = NULL;
    surface.colorFaces = NULL;

    list.push_back(surface);
 
    return true;
}

/////////////////////////// Manager stuff /////////////////////////////////


klModel *klModelManager::getInstance(const char *name) {
    DiscSurfaceList list;
    klModel *model = new klModel();

    if ( name[0] == '_' && name[1] == '_' ) {
        if ( !strcmp(name,"__quad") ) {
            klSurface surf;

            klVertex vertices[4] = {
            {klVec3(-1.0,-1.0,0.0),klVec2(0.0,0.0),0xFFFFFFFF,klVec3(0.0,0.0,1.0),klVec3(1.0,0.0,0.0),klVec3(0.0,1.0,0.0)},
            {klVec3( 1.0,-1.0,0.0),klVec2(1.0,0.0),0xFFFFFFFF,klVec3(0.0,0.0,1.0),klVec3(1.0,0.0,0.0),klVec3(0.0,1.0,0.0)},
            {klVec3( 1.0, 1.0,0.0),klVec2(1.0,1.0),0xFFFFFFFF,klVec3(0.0,0.0,1.0),klVec3(1.0,0.0,0.0),klVec3(0.0,1.0,0.0)},
            {klVec3(-1.0, 1.0,0.0),klVec2(0.0,1.0),0xFFFFFFFF,klVec3(0.0,0.0,1.0),klVec3(1.0,0.0,0.0),klVec3(0.0,1.0,0.0)}};
            surf.vertices = new klVertexBuffer(vertices,4);

            unsigned int indices[6] = {0,1,2,0,2,3};
            surf.indices = new klIndexBuffer(indices,6);

            surf.material = materialManager.getForName("default");

            model->addSurface(surf);
        } else if ( !strcmp(name,"__cube") ) {
            klSurface surf;

            klVertex vertices[24] = {
            // Top
                {klVec3(-1.0f,  1.0f, -1.0f), klVec2(1.0f, 1.0f),0xFFFFFFFF,klVec3(0.0,0.0,1.0),klVec3(1.0,0.0,0.0),klVec3(0.0,1.0,0.0)}, 
                {klVec3(-1.0f,  1.0f,  1.0f), klVec2(0.0f, 1.0f),0xFFFFFFFF,klVec3(0.0,0.0,1.0),klVec3(1.0,0.0,0.0),klVec3(0.0,1.0,0.0)},
                {klVec3( 1.0f,  1.0f,  1.0f), klVec2(0.0f, 0.0f),0xFFFFFFFF,klVec3(0.0,0.0,1.0),klVec3(1.0,0.0,0.0),klVec3(0.0,1.0,0.0)},
                {klVec3( 1.0f,  1.0f, -1.0f), klVec2(1.0f, 0.0f),0xFFFFFFFF,klVec3(0.0,0.0,1.0),klVec3(1.0,0.0,0.0),klVec3(0.0,1.0,0.0)},
            // Bottom
                {klVec3(-1.0f, -1.0f, -1.0f), klVec2(1.0f, 1.0f),0xFFFFFFFF,klVec3(0.0,0.0,1.0),klVec3(1.0,0.0,0.0),klVec3(0.0,1.0,0.0)}, 
                {klVec3( 1.0f, -1.0f, -1.0f), klVec2(0.0f, 1.0f),0xFFFFFFFF,klVec3(0.0,0.0,1.0),klVec3(1.0,0.0,0.0),klVec3(0.0,1.0,0.0)},
                {klVec3( 1.0f, -1.0f,  1.0f), klVec2(0.0f, 0.0f),0xFFFFFFFF,klVec3(0.0,0.0,1.0),klVec3(1.0,0.0,0.0),klVec3(0.0,1.0,0.0)},
                {klVec3(-1.0f, -1.0f,  1.0f), klVec2(1.0f, 0.0f),0xFFFFFFFF,klVec3(0.0,0.0,1.0),klVec3(1.0,0.0,0.0),klVec3(0.0,1.0,0.0)},
			// Front Face
                {klVec3(-1.0f, -1.0f,  1.0f), klVec2(0.0f, 0.0f),0xFFFFFFFF,klVec3(0.0,0.0,1.0),klVec3(1.0,0.0,0.0),klVec3(0.0,1.0,0.0)},
                {klVec3( 1.0f, -1.0f,  1.0f), klVec2(1.0f, 0.0f),0xFFFFFFFF,klVec3(0.0,0.0,1.0),klVec3(1.0,0.0,0.0),klVec3(0.0,1.0,0.0)},
                {klVec3( 1.0f,  1.0f,  1.0f), klVec2(1.0f, 1.0f),0xFFFFFFFF,klVec3(0.0,0.0,1.0),klVec3(1.0,0.0,0.0),klVec3(0.0,1.0,0.0)},
                {klVec3(-1.0f,  1.0f,  1.0f), klVec2(0.0f, 1.0f),0xFFFFFFFF,klVec3(0.0,0.0,1.0),klVec3(1.0,0.0,0.0),klVec3(0.0,1.0,0.0)},
			// Back Face
                {klVec3(-1.0f, -1.0f, -1.0f), klVec2(1.0f, 0.0f),0xFFFFFFFF,klVec3(0.0,0.0,1.0),klVec3(1.0,0.0,0.0),klVec3(0.0,1.0,0.0)}, 
                {klVec3(-1.0f,  1.0f, -1.0f), klVec2(1.0f, 1.0f),0xFFFFFFFF,klVec3(0.0,0.0,1.0),klVec3(1.0,0.0,0.0),klVec3(0.0,1.0,0.0)}, 
                {klVec3( 1.0f,  1.0f, -1.0f), klVec2(0.0f, 1.0f),0xFFFFFFFF,klVec3(0.0,0.0,1.0),klVec3(1.0,0.0,0.0),klVec3(0.0,1.0,0.0)}, 
                {klVec3( 1.0f, -1.0f, -1.0f), klVec2(0.0f, 0.0f),0xFFFFFFFF,klVec3(0.0,0.0,1.0),klVec3(1.0,0.0,0.0),klVec3(0.0,1.0,0.0)}, 
			// Right face
                {klVec3( 1.0f, -1.0f, -1.0f), klVec2(1.0f, 0.0f),0xFFFFFFFF,klVec3(0.0,0.0,1.0),klVec3(1.0,0.0,0.0),klVec3(0.0,1.0,0.0)}, 
                {klVec3( 1.0f,  1.0f, -1.0f), klVec2(1.0f, 1.0f),0xFFFFFFFF,klVec3(0.0,0.0,1.0),klVec3(1.0,0.0,0.0),klVec3(0.0,1.0,0.0)}, 
                {klVec3( 1.0f,  1.0f,  1.0f), klVec2(0.0f, 1.0f),0xFFFFFFFF,klVec3(0.0,0.0,1.0),klVec3(1.0,0.0,0.0),klVec3(0.0,1.0,0.0)}, 
                {klVec3( 1.0f, -1.0f,  1.0f), klVec2(0.0f, 0.0f),0xFFFFFFFF,klVec3(0.0,0.0,1.0),klVec3(1.0,0.0,0.0),klVec3(0.0,1.0,0.0)}, 
			// Left Face
                {klVec3(-1.0f, -1.0f, -1.0f), klVec2(0.0f, 0.0f),0xFFFFFFFF,klVec3(0.0,0.0,1.0),klVec3(1.0,0.0,0.0),klVec3(0.0,1.0,0.0)}, 
                {klVec3(-1.0f, -1.0f,  1.0f), klVec2(1.0f, 0.0f),0xFFFFFFFF,klVec3(0.0,0.0,1.0),klVec3(1.0,0.0,0.0),klVec3(0.0,1.0,0.0)}, 
                {klVec3(-1.0f,  1.0f,  1.0f), klVec2(1.0f, 1.0f),0xFFFFFFFF,klVec3(0.0,0.0,1.0),klVec3(1.0,0.0,0.0),klVec3(0.0,1.0,0.0)}, 
                {klVec3(-1.0f,  1.0f, -1.0f), klVec2(0.0f, 1.0f),0xFFFFFFFF,klVec3(0.0,0.0,1.0),klVec3(1.0,0.0,0.0),klVec3(0.0,1.0,0.0)}
            }; 
            surf.vertices = new klVertexBuffer(vertices,24);

            unsigned int indices[36] = { 0, 1, 2, 0, 2, 3,
                                         4, 5, 6, 4, 6, 7,
                                         8, 9,10, 8,10,11,
                                        12,13,14,12,14,15,
                                        16,17,18,16,18,19,
                                        20,21,22,20,22,23};
            surf.indices = new klIndexBuffer(indices,36);

            surf.material = materialManager.getForName("default");

            model->addSurface(surf);
        }
    } else if ( name[0] == '_' ) {
        return model;
    } else {
        klFileName fname(name);
        std::istream *str = fileSystem.openFile(name);

        if ( str == NULL ) {
            klFatalError("Model: '%s' not found\n",name);
        }

        if ( fname.getExt() == "obj" || fname.getExt() == "pbj" ) {
            LoadObj(*str, list, fname.getName().c_str());
        } else {
            LoadBob(*str, list, fname.getName().c_str() );
        }

        ProcessDiscModel(list, model);  
        delete str;
    }
    return model;
}

klModelManager modelManager;

void klModel::render() {

}
