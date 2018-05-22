//
// nvModel.cpp - Model support class
//
// The nvModel class implements an interface for a multipurpose model
// object. This class is useful for loading and formatting meshes
// for use by OpenGL. It can compute face normals, tangents, and
// adjacency information. The class supports the obj file format.
//
// This file implements the fomat independent part of the code.
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

#include <stdio.h>

#include <set>
#include <map>
#include <algorithm>
#include <string.h>
//#include <iostream>

#include "nvModel.h"

#include <nvMath.h>

#undef min
#undef max

//fix for non-standard naming
#ifdef WIN32
#define strcasecmp _stricmp
#endif

using std::vector;
using std::set;
using std::map;
using std::min;
using std::max;

namespace nv {

//////////////////////////////////////////////////////////////////////
//
// Local data structures
//
//////////////////////////////////////////////////////////////////////

//
//  Index gathering and ordering structure
////////////////////////////////////////////////////////////
struct IdxSet {
    GLuint pIndex;
    GLuint nIndex;
    GLuint tIndex;
    GLuint tanIndex;
    GLuint cIndex;

    bool operator< ( const IdxSet &rhs) const {
        if (pIndex < rhs.pIndex)
            return true;
        else if (pIndex == rhs.pIndex) {
            if (nIndex < rhs.nIndex)
                return true;
            else if (nIndex == rhs.nIndex) {
            
                if ( tIndex < rhs.tIndex)
                    return true;
                else if ( tIndex == rhs.tIndex) {
                    if (tanIndex < rhs.tanIndex)
                        return true;
                    else if (tanIndex == rhs.tanIndex)
                        return (cIndex < rhs.cIndex);
                }
            }
        }

        return false;
    }
};

//
//  Edge connectivity structure 
////////////////////////////////////////////////////////////
struct Edge {
    GLuint pIndex[2]; //position indices

    bool operator< (const Edge &rhs) const {
        return ( pIndex[0] == rhs.pIndex[0]) ? ( pIndex[1] < rhs.pIndex[1]) : pIndex[0] < rhs.pIndex[0];
    }

    Edge( GLuint v0, GLuint v1) {
        pIndex[0] = std::min( v0, v1);
        pIndex[1] = std::max( v0, v1);
    }

private:
    Edge() {} // disallow the default constructor
};

//////////////////////////////////////////////////////////////////////
//
//  Static data
//
//////////////////////////////////////////////////////////////////////
Model::FormatInfo Model::formatTable[] = {
    { "obj", Model::loadObjFromFile }
};

//
//
////////////////////////////////////////////////////////////
Model::Model() : _posSize(0), _tcSize(0), _cSize(0), _pOffset(-1), _nOffset(-1), _tcOffset(-1), _sTanOffset(-1), _cOffset(-1), _vtxSize(0), _openEdges(0) {
    nv::vec2<float> val;
}

//
//
//////////////////////////////////////////////////////////////////////
Model::~Model() {
    //dynamic allocations presently all handled via stl
}

//
//initialize a model from a file
//////////////////////////////////////////////////////////////////////
bool Model::loadModelFromFile( const char* file) {
    const char* extension;
    extension = strrchr( file, '.');

    if (extension)
        extension++; //start looking after the .
    else
        return false; //don't try to place guess the file type

    int formatCount = sizeof(Model::formatTable) / sizeof(Model::FormatInfo);
    
    //try to match by format first
    for ( int ii = 0; ii < formatCount; ii++) {
        if ( ! strcasecmp( formatTable[ii].extension, extension)) {
            //extension matches, load it
            return formatTable[ii].reader( file, *this);
        }
    }


    return false;
}



//
// compile the model to an acceptable format
//////////////////////////////////////////////////////////////////////
void Model::compileModel( Model::PrimType prim) {
    bool needsTriangles = false;
    bool needsTrianglesWithAdj = false;
    bool needsEdges = false;
    bool needsPoints = false;

    if ( (prim & Model::eptPoints) == Model::eptPoints)
        needsPoints = true;

    if ( (prim & Model::eptTriangles) == Model::eptTriangles)
        needsTriangles = true;

    if ( (prim & Model::eptTrianglesWithAdjacency) == Model::eptTrianglesWithAdjacency) {
        needsTriangles = true;
        needsTrianglesWithAdj = true;
    }

    if ( (prim & Model::eptEdges) == Model::eptEdges) {
        needsTriangles = true;
        needsEdges = true;
    }


    //merge the points
    map<IdxSet, GLuint> pts;

    //find whether a position is unique
    set<GLuint> ptSet;
    
    {
        vector<GLuint>::iterator pit = _pIndex.begin();
        vector<GLuint>::iterator nit = _nIndex.begin();
        vector<GLuint>::iterator tit = _tIndex.begin();
        vector<GLuint>::iterator tanit = _tanIndex.begin();
        vector<GLuint>::iterator cit = _cIndex.begin();

        while ( pit < _pIndex.end()) {
            IdxSet idx;
            idx.pIndex = *pit;

            if ( _normals.size() > 0)
                idx.nIndex = *nit;
            else
                idx.nIndex = 0;

            if ( _tIndex.size() > 0)
                idx.tIndex = *tit;
            else
                idx.tIndex = 0;

            if ( _tanIndex.size() > 0)
                idx.tanIndex = *tanit;
            else
                idx.tanIndex = 0;

            if ( _cIndex.size() > 0)
                idx.cIndex = *cit;
            else
                idx.cIndex = 0;

            map<IdxSet,GLuint>::iterator mit = pts.find(idx);

            if (mit == pts.end()) {

                if (needsTriangles)
                    _indices[2].push_back( (GLuint)pts.size());

                //since this one is a new vertex, check to see if this position is already referenced
                if (needsPoints && ptSet.find(idx.pIndex) != ptSet.end()) {
                    ptSet.insert( idx.pIndex);
                }

                pts.insert( map<IdxSet,GLuint>::value_type(idx, (GLuint)pts.size()));

                //position
                _vertices.push_back( _positions[idx.pIndex*_posSize]);
                _vertices.push_back( _positions[idx.pIndex*_posSize + 1]);
                _vertices.push_back( _positions[idx.pIndex*_posSize + 2]);
                if (_posSize == 4)
                    _vertices.push_back( _positions[idx.pIndex*_posSize + 3]);

                //normal
                if (_normals.size() > 0) {
                    _vertices.push_back( _normals[idx.nIndex*3]);
                    _vertices.push_back( _normals[idx.nIndex*3 + 1]);
                    _vertices.push_back( _normals[idx.nIndex*3 + 2]);
                }

                //texture coordinate
                if (_texCoords.size() > 0) {
                    _vertices.push_back( _texCoords[idx.tIndex*_tcSize]);
                    _vertices.push_back( _texCoords[idx.tIndex*_tcSize + 1]);
                    if (_tcSize == 3)
                        _vertices.push_back( _texCoords[idx.tIndex*_tcSize + 2]);
                }

                //tangents
                if (_sTangents.size() > 0) {
                    _vertices.push_back( _sTangents[idx.tanIndex*3]);
                    _vertices.push_back( _sTangents[idx.tanIndex*3 + 1]);
                    _vertices.push_back( _sTangents[idx.tanIndex*3 + 2]);
                }

                //colors
                if (_colors.size() > 0) {
                    _vertices.push_back( _colors[idx.cIndex*_cSize]);
                    _vertices.push_back( _colors[idx.cIndex*_cSize + 1]);
                    _vertices.push_back( _colors[idx.cIndex*_cSize + 2]);
                    if (_cSize == 4)
                        _vertices.push_back( _colors[idx.cIndex*_cSize + 3]);
                }
            }
            else {
                if (needsTriangles)
                    _indices[2].push_back( mit->second);
            }

            pit++;
            if ( _normals.size() > 0) nit++;
            if ( _tIndex.size() > 0) tit++;
            if ( _tanIndex.size() > 0) tanit++;
            if ( _cIndex.size() > 0) cit++;
        }
    }

    //create an edge list, if necessary
    if (needsEdges || needsTrianglesWithAdj) {
        std::multimap<Edge, GLuint> edges;


        //edges are only based on positions only
        for (int ii = 0; ii < (int)_pIndex.size(); ii += 3) {
            for (int jj = 0; jj < 3; jj++) {
                Edge w( _pIndex[ii + jj], _pIndex[ii + (jj +1) % 3]);
                std::multimap<Edge, GLuint>::iterator it = edges.find(w);

                //if we are storing edges, make sure we store only one copy
                if (needsEdges && it == edges.end()) {
                    _indices[1].push_back( _indices[2][ii+jj]);
                    _indices[1].push_back( _indices[2][ii + (jj +1) % 3]);
                }
                edges.insert( std::multimap<Edge, GLuint>::value_type( w, ii / 3));
            }
        }


        //now handle triangles with adjacency
        if (needsTrianglesWithAdj) {
            for (int ii = 0; ii < (int)_pIndex.size(); ii += 3) {
                for (int jj = 0; jj < 3; jj++) {
                    Edge w( _pIndex[ii + jj], _pIndex[ii + (jj + 1) % 3]);
                    std::multimap<Edge, GLuint>::iterator it = edges.lower_bound(w);
                    std::multimap<Edge, GLuint>::iterator limit = edges.upper_bound(w);
                    GLuint adjVertex = 0;

                    while ( it != edges.end() && it->second == ii /3 && it != limit)
                        it++;

                    if ( it == edges.end() || it == limit || it->first.pIndex[0] != w.pIndex[0] || it->first.pIndex[1] != w.pIndex[1] ) {
                        //no adjacent triangle found, duplicate the vertex
                        adjVertex = _indices[2][ii + jj];
                        _openEdges++;
                        
                    }
                    else {
                        GLuint triOffset = it->second * 3; //compute the starting index of the triangle
                        adjVertex = _indices[2][triOffset]; //set the vertex to a default, in case the adjacent triangle it a degenerate

                        //find the unshared vertex
                        for ( int kk=0; kk<3; kk++) {
                            if ( _pIndex[triOffset + kk] != w.pIndex[0] && _pIndex[triOffset + kk] != w.pIndex[1] ) {
                                adjVertex = _indices[2][triOffset + kk];
                                break;
                            }
                        }
                    }

                    //store the vertices for this edge
                    _indices[3].push_back( _indices[2][ii + jj]);
                    _indices[3].push_back( adjVertex);
                }
            }
        }

    }

    //create selected prim

    //set the offsets and vertex size
    _pOffset = 0; //always first
    _vtxSize = _posSize;
    if ( hasNormals()) {
        _nOffset = _vtxSize;
        _vtxSize += 3;
    }
    else {
        _nOffset = -1;
    }
    if ( hasTexCoords()) {
        _tcOffset = _vtxSize;
        _vtxSize += _tcSize;
    }
    else {
        _tcOffset = -1;
    }
    if ( hasTangents()) {
        _sTanOffset = _vtxSize;
        _vtxSize += 3;
    }
    else {
        _sTanOffset = -1;
    }
    if ( hasColors()) {
        _cOffset = _vtxSize;
        _vtxSize += _cSize;
    }
    else {
        _cOffset = -1;
    }

    
}

//
// compute tangents in the S direction
//
//////////////////////////////////////////////////////////////////////
void Model::computeTangents() {

    //make sure tangents don't already exist
    if ( hasTangents()) 
        return;

    //make sure that the model has texcoords
    if ( !hasTexCoords())
        return;

    //alloc memory and initialize to 0
    _tanIndex.reserve( _pIndex.size());
    _sTangents.resize( (_texCoords.size() / _tcSize) * 3, 0.0f);

    // the collision map records any alternate locations for the tangents
    std::multimap< GLuint, GLuint> collisionMap;

    //process each face, compute the tangent and try to add it
    for (int ii = 0; ii < (int)_pIndex.size(); ii += 3) {
        vec3f p0(&_positions[_pIndex[ii]*_posSize]);
        vec3f p1(&_positions[_pIndex[ii+1]*_posSize]);
        vec3f p2(&_positions[_pIndex[ii+2]*_posSize]);
        vec2f st0(&_texCoords[_tIndex[ii]*_tcSize]);
        vec2f st1(&_texCoords[_tIndex[ii+1]*_tcSize]);
        vec2f st2(&_texCoords[_tIndex[ii+2]*_tcSize]);

        //compute the edge and tc differentials
        vec3f dp0 = p1 - p0;
        vec3f dp1 = p2 - p0;
        vec2f dst0 = st1 - st0;
        vec2f dst1 = st2 - st0;

        float factor = 1.0f / (dst0[0] * dst1[1] - dst1[0] * dst0[1]);

        //compute sTangent
        vec3f sTan;
        sTan[0] = dp0[0] * dst1[1] - dp1[0] * dst0[1];
        sTan[1] = dp0[1] * dst1[1] - dp1[1] * dst0[1];
        sTan[2] = dp0[2] * dst1[1] - dp1[2] * dst0[1];
        sTan *= factor;

        //should this really renormalize?
        sTan =normalize( sTan);

        //loop over the vertices, to update the tangents
        for (int jj = 0; jj < 3; jj++) {
            //get the present accumulated tangnet
            vec3f curTan(&_sTangents[_tIndex[ii + jj]*3]);

            //check to see if it is uninitialized, if so, insert it
            if (curTan[0] == 0.0f && curTan[1] == 0.0f && curTan[2] == 0.0f) {
                _sTangents[_tIndex[ii + jj]*3] = sTan[0];
                _sTangents[_tIndex[ii + jj]*3+1] = sTan[1];
                _sTangents[_tIndex[ii + jj]*3+2] = sTan[2];
                _tanIndex.push_back(_tIndex[ii + jj]);
            }
            else {
                //check for agreement
                curTan = normalize( curTan);

                if ( dot( curTan, sTan) >= cosf( 3.1415926f * 0.333333f)) {
                    //tangents are in agreement
                    _sTangents[_tIndex[ii + jj]*3] += sTan[0];
                    _sTangents[_tIndex[ii + jj]*3+1] += sTan[1];
                    _sTangents[_tIndex[ii + jj]*3+2] += sTan[2];
                    _tanIndex.push_back(_tIndex[ii + jj]);
                }
                else {
                    //tangents disagree, this vertex must be split in tangent space 
                    std::multimap< GLuint, GLuint>::iterator it = collisionMap.find( _tIndex[ii + jj]);

                    //loop through all hits on this index, until one agrees
                    while ( it != collisionMap.end() && it->first == _tIndex[ii + jj]) {
                        curTan = vec3f( &_sTangents[it->second*3]);

                        curTan = normalize(curTan);
                        if ( dot( curTan, sTan) >= cosf( 3.1415926f * 0.333333f))
                            break;

                        it++;
                    }

                    //check for agreement with an earlier collision
                    if ( it != collisionMap.end() && it->first == _tIndex[ii + jj]) {
                        //found agreement with an earlier collision, use that one
                        _sTangents[it->second*3] += sTan[0];
                        _sTangents[it->second*3+1] += sTan[1];
                        _sTangents[it->second*3+2] += sTan[2];
                        _tanIndex.push_back(it->second);
                    }
                    else {
                        //we have a new collision, create a new tangent
                        GLuint target = (GLuint)_sTangents.size() / 3;
                        _sTangents.push_back( sTan[0]);
                        _sTangents.push_back( sTan[1]);
                        _sTangents.push_back( sTan[2]);
                        _tanIndex.push_back( target);
                        collisionMap.insert( std::multimap< GLuint, GLuint>::value_type( _tIndex[ii + jj], target));
                    }
                } // else ( if tangent agrees)
            } // else ( if tangent is uninitialized )
        } // for jj = 0 to 2 ( iteration of triangle verts)
    } // for ii = 0 to numFaces *3 ( iterations over triangle faces

    //normalize all the tangents
    for (int ii = 0; ii < (int)_sTangents.size(); ii += 3) {
        vec3f tan(&_sTangents[ii]);
        tan = normalize(tan);
        _sTangents[ii] = tan[0];
        _sTangents[ii+1] = tan[1];
        _sTangents[ii+2] = tan[2];
    }
}
//
//compute vertex normals
//////////////////////////////////////////////////////////////////////
void Model::computeNormals() {

    // don't recompute normals
    if (hasNormals())
        return;

    //allocate and initialize the normal values
    _normals.resize( (_positions.size() / _posSize) * 3, 0.0f);
    _nIndex.reserve( _pIndex.size());

    // the collision map records any alternate locations for the normals
    std::multimap< GLuint, GLuint> collisionMap;

    //iterate over the faces, computing the face normal and summing it them
    for ( int ii = 0; ii < (int)_pIndex.size(); ii += 3) {
        vec3f p0(&_positions[_pIndex[ii]*_posSize]);
        vec3f p1(&_positions[_pIndex[ii+1]*_posSize]);
        vec3f p2(&_positions[_pIndex[ii+2]*_posSize]);

        //compute the edge vectors
        vec3f dp0 = p1 - p0;
        vec3f dp1 = p2 - p0;

        vec3f fNormal = cross( dp0, dp1); // compute the face normal
        vec3f nNormal = normalize(fNormal);  // compute a normalized normal

        //iterate over the vertices, adding the face normal influence to each
        for ( int jj = 0; jj < 3; jj++) {
            // get the current normal from the default location (index shared with position) 
            vec3f cNormal( &_normals[_pIndex[ii + jj]*3]);

            // check to see if this normal has not yet been touched 
            if ( cNormal[0] == 0.0f && cNormal[1] == 0.0f && cNormal[2] == 0.0f) {
                // first instance of this index, just store it as is
                _normals[_pIndex[ii + jj]*3] = fNormal[0];
                _normals[_pIndex[ii + jj]*3 + 1] = fNormal[1];
                _normals[_pIndex[ii + jj]*3 + 2] = fNormal[2];
                _nIndex.push_back( _pIndex[ii + jj]); 
            }
            else {
                // check for agreement
                cNormal = normalize( cNormal);

                if ( dot( cNormal, nNormal) >= cosf( 3.1415926f * 0.333333f)) {
                    //normal agrees, so add it
                    _normals[_pIndex[ii + jj]*3] += fNormal[0];
                    _normals[_pIndex[ii + jj]*3 + 1] += fNormal[1];
                    _normals[_pIndex[ii + jj]*3 + 2] += fNormal[2];
                    _nIndex.push_back( _pIndex[ii + jj]);
                }
                else {
                    //normals disagree, this vertex must be along a facet edge 
                    std::multimap< GLuint, GLuint>::iterator it = collisionMap.find( _pIndex[ii + jj]);

                    //loop through all hits on this index, until one agrees
                    while ( it != collisionMap.end() && it->first == _pIndex[ii + jj]) {
                        cNormal = normalize(vec3f( &_normals[it->second*3]));

                        if ( dot( cNormal, nNormal) >= cosf( 3.1415926f * 0.333333f))
                            break;

                        it++;
                    }

                    //check for agreement with an earlier collision
                    if ( it != collisionMap.end() && it->first == _pIndex[ii + jj]) {
                        //found agreement with an earlier collision, use that one
                        _normals[it->second*3] += fNormal[0];
                        _normals[it->second*3+1] += fNormal[1];
                        _normals[it->second*3+2] += fNormal[2];
                        _nIndex.push_back(it->second);
                    }
                    else {
                        //we have a new collision, create a new normal
                        GLuint target = (GLuint)_normals.size() / 3;
                        _normals.push_back( fNormal[0]);
                        _normals.push_back( fNormal[1]);
                        _normals.push_back( fNormal[2]);
                        _nIndex.push_back( target);
                        collisionMap.insert( std::multimap< GLuint, GLuint>::value_type( _pIndex[ii + jj], target));
                    }
                } // else ( if normal agrees)
            } // else (if normal is uninitialized)
        } // for each vertex in triangle
    } // for each face

    //now normalize all the normals
    for ( int ii = 0; ii < (int)_normals.size(); ii += 3) {
        vec3f norm(&_normals[ii]);
        norm =normalize(norm);
        _normals[ii] = norm[0];
        _normals[ii+1] = norm[1];
        _normals[ii+2] = norm[2];
    }

}

//
//
//////////////////////////////////////////////////////////////////////
void Model::computeBoundingBox( vec3f &minVal, vec3f &maxVal) {

    if ( _positions.empty())
        return;

    minVal = vec3f( _positions[0], _positions[1], _positions[2]);
    maxVal = minVal;

    for ( vector<float>::iterator pit = _positions.begin() + _posSize; pit < _positions.end(); pit += _posSize) {
        minVal = cwmin( minVal, vec3f( &pit[0]));
        maxVal = cwmax( maxVal, vec3f( &pit[0]));
    }
}

//
//
//////////////////////////////////////////////////////////////////////
void Model::rescale( float radius) {

    if ( _positions.empty())
        return;

    vec3f minVal, maxVal;
    computeBoundingBox(minVal, maxVal);

    vec3f r = 0.5f*(maxVal - minVal);
    vec3f center = minVal + r;
    float oldRadius = length(r);
    float scale = radius / oldRadius;

    for ( vector<float>::iterator pit = _positions.begin(); pit < _positions.end(); pit += _posSize) {
        vec3f np = scale*(vec3f(&pit[0]) - center);
        pit[0] = np.x;
        pit[1] = np.y;
        pit[2] = np.z;
    }
}


//
//
//////////////////////////////////////////////////////////////////////
void Model::clearNormals(){
    _normals.clear();
    _nIndex.clear();
}

//
//
//////////////////////////////////////////////////////////////////////
void Model::clearTexCoords(){
    _texCoords.clear();
    _tIndex.clear();
}

//
//
//////////////////////////////////////////////////////////////////////
void Model::clearTangents(){
    _sTangents.clear();
    _tanIndex.clear();
}

//
//
//////////////////////////////////////////////////////////////////////
void Model::clearColors(){
    _colors.clear();
    _cIndex.clear();
}

//
//
//////////////////////////////////////////////////////////////////////
void Model::removeDegeneratePrims() {
    GLuint *pSrc = 0, *pDst = 0, *tSrc = 0, *tDst = 0, *nSrc = 0, *nDst = 0, *cSrc = 0, *cDst = 0;
    int degen = 0;

    pSrc = &_pIndex[0];
    pDst = pSrc;

    if (hasTexCoords()) {
        tSrc = &_tIndex[0];
        tDst = tSrc;
    }

    if (hasNormals()) {
        nSrc = &_nIndex[0];
        nDst = nSrc;
    }

    if (hasColors()) {
        cSrc =&_cIndex[0];
        cDst = cSrc;
    }

    for (int ii = 0; ii < (int)_pIndex.size(); ii += 3, pSrc += 3, tSrc += 3, nSrc += 3, cSrc += 3) {
        if ( pSrc[0] == pSrc[1] || pSrc[0] == pSrc[2] || pSrc[1] == pSrc[2]) {
            degen++;
            continue; //skip updating the dest
        }

        for (int jj = 0; jj < 3; jj++) {
            *pDst++ = pSrc[jj];

            if (hasTexCoords())
                *tDst++ = tSrc[jj];

            if (hasNormals())
                *nDst++ = nSrc[jj];

            if (hasColors())
                *cDst++ = cSrc[jj];
        }
    }

    _pIndex.resize( _pIndex.size() - degen * 3);

    if (hasTexCoords())
        _tIndex.resize( _tIndex.size() - degen * 3);

    if (hasNormals())
        _nIndex.resize( _nIndex.size() - degen * 3);

    if (hasColors())
        _cIndex.resize( _cIndex.size() - degen * 3);

}

};