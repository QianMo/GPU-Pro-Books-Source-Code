//
// nvModelObj.cpp - Model support class
//
// The nvModel class implements an interface for a multipurpose model
// object. This class is useful for loading and formatting meshes
// for use by OpenGL. It can compute face normals, tangents, and
// adjacency information. The class supports the obj file format.
//
// This file implements the obj file parser and translator.
//
// Author: Evan Hart
// Email: sdkfeedback@nvidia.com
//
// Copyright (c) NVIDIA Corporation. All rights reserved.
////////////////////////////////////////////////////////////////////////////////
/*
#define _HAS_ITERATOR_DEBUGGING 0
#define _SECURE_SCL 0
*/
#include "nvModel.h"

#include <stdio.h>
#include <map>
#include <string>

#define BUF_SIZE 256

using std::vector;

namespace nv {

bool Model::loadObjFromFile( const char *file, Model &m) {
    FILE *fp;

    fp = fopen( file, "r");
    if (!fp) {
        return false;
    }

    char buf[BUF_SIZE];
    float val[4];
    int idx[3][3];
    int match;
    bool vtx4Comp = false;
    bool tex3Comp = false;
    bool hasTC = false;
    bool hasNormals = false;
    std::map<std::string, int> matIdx;
    int currMat = -1;

    while ( fscanf( fp, "%s", buf) != EOF ) {

        switch (buf[0]) {
            case '#':
                //comment line, eat the remainder
                fgets( buf, BUF_SIZE, fp);
                break;

            case 'v':
                switch (buf[1]) {
                
                    case '\0':
                        //vertex, 3 or 4 components
                        val[3] = 1.0f;  //default w coordinate
                        match = fscanf( fp, "%f %f %f %f", &val[0], &val[1], &val[2], &val[3]);
                        m._positions.push_back( val[0]);
                        m._positions.push_back( val[1]);
                        m._positions.push_back( val[2]);
                        m._positions.push_back( val[3]);
                        vtx4Comp |= ( match == 4);
                        assert( match > 2 && match < 5);
                        break;

                    case 'n':
                        //normal, 3 components
                        match = fscanf( fp, "%f %f %f", &val[0], &val[1], &val[2]);
                        m._normals.push_back( val[0]);
                        m._normals.push_back( val[1]);
                        m._normals.push_back( val[2]);
                        assert( match == 3);
                        break;

                    case 't':
                        //texcoord, 2 or 3 components
                        val[2] = 0.0f;  //default r coordinate
                        match = fscanf( fp, "%f %f %f %f", &val[0], &val[1], &val[2]);
                        m._texCoords.push_back( val[0]);
                        m._texCoords.push_back( val[1]);
                        m._texCoords.push_back( val[2]);
                        tex3Comp |= ( match == 3);
                        assert( match > 1 && match < 4);
                        break;
                }
                break;

            case 'f':
                //face
                fscanf( fp, "%s", buf);

                //determine the type, and read the initial vertex, all entries in a face must have the same format
                if ( sscanf( buf, "%d//%d", &idx[0][0], &idx[0][1]) == 2) {
                    //This face has vertex and normal indices

                    //remap them to the right spot
                    idx[0][0] = (idx[0][0] > 0) ? (idx[0][0] - 1) : ((int)m._positions.size() - idx[0][0]);
                    idx[0][1] = (idx[0][1] > 0) ? (idx[0][1] - 1) : ((int)m._normals.size() - idx[0][1]);

                    //grab the second vertex to prime
                    fscanf( fp, "%d//%d", &idx[1][0], &idx[1][1]);

                    //remap them to the right spot
                    idx[1][0] = (idx[1][0] > 0) ? (idx[1][0] - 1) : ((int)m._positions.size() - idx[1][0]);
                    idx[1][1] = (idx[1][1] > 0) ? (idx[1][1] - 1) : ((int)m._normals.size() - idx[1][1]);

                    //create the fan
                    while ( fscanf( fp, "%d//%d", &idx[2][0], &idx[2][1]) == 2) {
                        //remap them to the right spot
                        idx[2][0] = (idx[2][0] > 0) ? (idx[2][0] - 1) : ((int)m._positions.size() - idx[2][0]);
                        idx[2][1] = (idx[2][1] > 0) ? (idx[2][1] - 1) : ((int)m._normals.size() - idx[2][1]);

                        //add the indices
                        for (int ii = 0; ii < 3; ii++) {
                            m._pIndex.push_back( idx[ii][0]);
                            m._nIndex.push_back( idx[ii][1]);
                            m._tIndex.push_back(0); // dummy index, to ensure that the buffers are of identical size
                            m._mIndex.push_back(currMat);
                        }
                        
                        //prepare for the next iteration
                        idx[1][0] = idx[2][0];
                        idx[1][1] = idx[2][1];
                    }
                    hasNormals = true;
                }
                else if ( sscanf( buf, "%d/%d/%d", &idx[0][0], &idx[0][1], &idx[0][2]) == 3) {
                    //This face has vertex, texture coordinate, and normal indices

                    //remap them to the right spot
                    idx[0][0] = (idx[0][0] > 0) ? (idx[0][0] - 1) : ((int)m._positions.size() - idx[0][0]);
                    idx[0][1] = (idx[0][1] > 0) ? (idx[0][1] - 1) : ((int)m._texCoords.size() - idx[0][1]);
                    idx[0][2] = (idx[0][2] > 0) ? (idx[0][2] - 1) : ((int)m._normals.size() - idx[0][2]);

                    //grab the second vertex to prime
                    fscanf( fp, "%d/%d/%d", &idx[1][0], &idx[1][1], &idx[1][2]);

                    //remap them to the right spot
                    idx[1][0] = (idx[1][0] > 0) ? (idx[1][0] - 1) : ((int)m._positions.size() - idx[1][0]);
                    idx[1][1] = (idx[1][1] > 0) ? (idx[1][1] - 1) : ((int)m._texCoords.size() - idx[1][1]);
                    idx[1][2] = (idx[1][2] > 0) ? (idx[1][2] - 1) : ((int)m._normals.size() - idx[1][2]);

                    //create the fan
                    while ( fscanf( fp, "%d/%d/%d", &idx[2][0], &idx[2][1], &idx[2][2]) == 3) {
                        //remap them to the right spot
                        idx[2][0] = (idx[2][0] > 0) ? (idx[2][0] - 1) : ((int)m._positions.size() - idx[2][0]);
                        idx[2][1] = (idx[2][1] > 0) ? (idx[2][1] - 1) : ((int)m._texCoords.size() - idx[2][1]);
                        idx[2][2] = (idx[2][2] > 0) ? (idx[2][2] - 1) : ((int)m._normals.size() - idx[2][2]);

                        //add the indices
                        for (int ii = 0; ii < 3; ii++) {
                            m._pIndex.push_back( idx[ii][0]);
                            m._tIndex.push_back( idx[ii][1]);
                            m._nIndex.push_back( idx[ii][2]);
                            m._mIndex.push_back(currMat);
                        }
                        
                        //prepare for the next iteration
                        idx[1][0] = idx[2][0];
                        idx[1][1] = idx[2][1];
                        idx[1][2] = idx[2][2];
                    }

                    hasTC = true;
                    hasNormals = true;
                }
                else if ( sscanf( buf, "%d/%d", &idx[0][0], &idx[0][1]) == 2) {
                    //This face has vertex and texture coordinate indices

                    //remap them to the right spot
                    idx[0][0] = (idx[0][0] > 0) ? (idx[0][0] - 1) : ((int)m._positions.size() - idx[0][0]);
                    idx[0][1] = (idx[0][1] > 0) ? (idx[0][1] - 1) : ((int)m._texCoords.size() - idx[0][1]);

                    //grab the second vertex to prime
                    fscanf( fp, "%d/%d", &idx[1][0], &idx[1][1]);

                    //remap them to the right spot
                    idx[1][0] = (idx[1][0] > 0) ? (idx[1][0] - 1) : ((int)m._positions.size() - idx[1][0]);
                    idx[1][1] = (idx[1][1] > 0) ? (idx[1][1] - 1) : ((int)m._texCoords.size() - idx[1][1]);

                    //create the fan
                    while ( fscanf( fp, "%d/%d", &idx[2][0], &idx[2][1]) == 2) {
                        //remap them to the right spot
                        idx[2][0] = (idx[2][0] > 0) ? (idx[2][0] - 1) : ((int)m._positions.size() - idx[2][0]);
                        idx[2][1] = (idx[2][1] > 0) ? (idx[2][1] - 1) : ((int)m._texCoords.size() - idx[2][1]);

                        //add the indices
                        for (int ii = 0; ii < 3; ii++) {
                            m._pIndex.push_back( idx[ii][0]);
                            m._tIndex.push_back( idx[ii][1]);
                            m._nIndex.push_back( 0); //dummy normal index to keep everything in synch
                            m._mIndex.push_back(currMat);
                        }
                        
                        //prepare for the next iteration
                        idx[1][0] = idx[2][0];
                        idx[1][1] = idx[2][1];
                    }
                    hasTC = true;
                }
                else if ( sscanf( buf, "%d", &idx[0][0]) == 1) {
                    //This face has only vertex indices

                    //remap them to the right spot
                    idx[0][0] = (idx[0][0] > 0) ? (idx[0][0] - 1) : ((int)m._positions.size() - idx[0][0]);

                    //grab the second vertex to prime
                    fscanf( fp, "%d", &idx[1][0]);

                    //remap them to the right spot
                    idx[1][0] = (idx[1][0] > 0) ? (idx[1][0] - 1) : ((int)m._positions.size() - idx[1][0]);

                    //create the fan
                    while ( fscanf( fp, "%d", &idx[2][0]) == 1) {
                        //remap them to the right spot
                        idx[2][0] = (idx[2][0] > 0) ? (idx[2][0] - 1) : ((int)m._positions.size() - idx[2][0]);

                        //add the indices
                        for (int ii = 0; ii < 3; ii++) {
                            m._pIndex.push_back( idx[ii][0]);
                            m._tIndex.push_back( 0); //dummy index to keep things in synch
                            m._nIndex.push_back( 0); //dummy normal index to keep everything in synch
                            m._mIndex.push_back(currMat);
                        }
                        
                        //prepare for the next iteration
                        idx[1][0] = idx[2][0];
                    }
                }
                else {
                    //bad format
                    assert(0);
                    fgets( buf, BUF_SIZE, fp);
                }
                break;

            case 'u': {
                // read material name
                fscanf( fp, "%s", buf);
                // look for it in database
                std::string matName(buf);
                std::map<std::string, int>::iterator iter = matIdx.find(matName);
                if (iter==matIdx.end()) {
                    // insert new material if not found
                    currMat = matIdx.size();
                    matIdx.insert(std::pair<std::string, int>(matName, currMat));
                } else {
                    // use existing material
                    currMat = iter->second;
                }
                break; }
            case 's':
            case 'g':
                //all presently ignored
            default:
                fgets( buf, BUF_SIZE, fp);
        };
    }

    fclose(fp);

    //post-process data

    //free anything that ended up being unused
    if (!hasNormals) {
        m._normals.clear();
        m._nIndex.clear();
    }

    if (!hasTC) {
        m._texCoords.clear();
        m._tIndex.clear();
    }

    //set the defaults as the worst-case for an obj file
    m._posSize = 4;
    m._tcSize = 3;

    //compact to 3 component vertices if possible
    if (!vtx4Comp) {
        vector<float>::iterator src = m._positions.begin();
        vector<float>::iterator dst = m._positions.begin();

        for ( ; src < m._positions.end(); ) {
            *(dst++) = *(src++);
            *(dst++) = *(src++);
            *(dst++) = *(src++);
            src++;
        }

        m._positions.resize( (m._positions.size() / 4) * 3);

        m._posSize = 3;
    }

    //compact to 2 component tex coords if possible
    if (!tex3Comp) {
        vector<float>::iterator src = m._texCoords.begin();
        vector<float>::iterator dst = m._texCoords.begin();

        for ( ; src < m._texCoords.end(); ) {
            *(dst++) = *(src++);
            *(dst++) = *(src++);
            src++;
        }

        m._texCoords.resize( (m._texCoords.size() / 3) * 2);

        m._tcSize = 2; 
    }

    // no materials
    if (currMat==-1) return true;

    // read materials
    // replace extension by ".mtl";
    char *mtlFile = new char[strlen(file)+5];
    strcpy(mtlFile, file);
    char *ext = strrchr(mtlFile, '.');

    if (ext==0) ext = mtlFile + strlen(mtlFile);

    *(ext++) = '.';
    *(ext++) = 'm';
    *(ext++) = 't';
    *(ext++) = 'l';
    *(ext++) = '\0';


    // open file
    fp = fopen( mtlFile, "r");
    if (!fp) {
        return true;
    }
    free(mtlFile);

    m._materials.clear();
    m._materials.resize(matIdx.size());
    Material *mat = 0;

    while ( fscanf( fp, "%s", buf) != EOF ) {

        bool eatLine = true;
        if (strcmp(buf, "newmtl")==0) {
            fscanf( fp, "%s", buf);
            std::string matName(buf);
            std::map<std::string, int>::iterator iter = matIdx.find(matName);
            if (iter==matIdx.end()) {
                // unknown material => skip
                mat = 0;
            } else {
                // use existing material
                mat = &(m._materials[iter->second]);
            }
            eatLine = false;
        } else if (mat!=0) {
            // read material properties only if material is known
            if (strcmp(buf, "Ka")==0) {
                if (fscanf( fp, "%f %f %f", &(mat->ka[0]), &(mat->ka[1]), &(mat->ka[2])) != 3) {
                    // on error, use black
                    mat->ka[0] = 0.0f;
                    mat->ka[1] = 0.0f;
                    mat->ka[2] = 0.0f;
                }
                eatLine = false;
            } else if (strcmp(buf, "Kd")==0) {
                if (fscanf( fp, "%f %f %f", &(mat->kd[0]), &(mat->kd[1]), &(mat->kd[2])) != 3) {
                    // on error, use black
                    mat->kd[0] = 0.0f;
                    mat->kd[1] = 0.0f;
                    mat->kd[2] = 0.0f;
                }
                eatLine = false;
            } else if (strcmp(buf, "Ks")==0) {
                if (fscanf( fp, "%f %f %f", &(mat->ks[0]), &(mat->ks[1]), &(mat->ks[2])) != 3) {
                    // on error, use black
                    mat->ks[0] = 0.0f;
                    mat->ks[1] = 0.0f;
                    mat->ks[2] = 0.0f;
                }
                eatLine = false;
            } else if (strcmp(buf, "Ns")==0) {
                if (fscanf( fp, "%f", &(mat->ns)) != 1) {
                    // on error, use black
                    mat->ns = 0.0f;
                }
                eatLine = false;
            } else if (strcmp(buf, "map_Kd")==0) {
                if (fscanf( fp, "%s", buf) != 1) {
                    // on error, use no tex
                    mat->map_kd = "";
                } else {
                    mat->map_kd = buf;
                }
                eatLine = false;
            } else if (strcmp(buf, "map_Ks")==0) {
                if (fscanf( fp, "%s", buf) != 1) {
                    // on error, use no tex
                    mat->map_ks = "";
                } else {
                    mat->map_ks = buf;
                }
                eatLine = false;
            }
        }

        // eat the rest of the line if not processed otherwise
        if (eatLine) {
            fgets( buf, BUF_SIZE, fp);
        }
    }

    fclose(fp);


    return true;
}


};

