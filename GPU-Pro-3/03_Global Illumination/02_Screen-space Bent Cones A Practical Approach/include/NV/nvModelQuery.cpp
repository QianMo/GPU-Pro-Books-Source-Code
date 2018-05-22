//
// nvModelQuery.h - Model support class
//
// The nvModel class implements an interface for a multipurpose model
// object. This class is useful for loading and formatting meshes
// for use by OpenGL. It can compute face normals, tangents, and
// adjacency information. The class supports the obj file format.
//
// This function implements the query functions. (number of vertices, etc)
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

//fix for non-standard naming
#ifdef WIN32
#define strcasecmp _stricmp
#endif

using std::vector;

namespace nv {

//
//
////////////////////////////////////////////////////////////
bool Model::hasNormals() const {
    return _normals.size() > 0;
}

//
//
////////////////////////////////////////////////////////////
bool Model::hasTexCoords() const {
    return _texCoords.size() > 0;
}

//
//
////////////////////////////////////////////////////////////
bool Model::hasTangents() const {
    return _sTangents.size() > 0;
}

//
//
////////////////////////////////////////////////////////////
bool Model::hasColors() const {
    return _colors.size() > 0;
}

//
//
////////////////////////////////////////////////////////////
int Model::getPositionSize() const {
    return _posSize;
}

//
//
////////////////////////////////////////////////////////////
int Model::getNormalSize() const {
    return 3;
}

//
//
////////////////////////////////////////////////////////////
int Model::getTexCoordSize() const {
    return _tcSize;
}

//
//
////////////////////////////////////////////////////////////
int Model::getTangentSize() const {
    return 3;
}

//
//
////////////////////////////////////////////////////////////
int Model::getColorSize() const {
    return _cSize;
}


//raw data access functions
//  These are to be used to get the raw array data from the file, each array has its own index

//
//
////////////////////////////////////////////////////////////
const float* Model::getPositions() const {
    return ( _positions.size() > 0) ? &(_positions[0]) : 0;
}

//
//
////////////////////////////////////////////////////////////
const float* Model::getNormals() const {
    return ( _normals.size() > 0) ? &(_normals[0]) : 0;
}

//
//
////////////////////////////////////////////////////////////
const float* Model::getTexCoords() const {
    return ( _texCoords.size() > 0) ? &(_texCoords[0]) : 0;
}

//
//
////////////////////////////////////////////////////////////
const float* Model::getTangents() const {
    return ( _sTangents.size() > 0) ? &(_sTangents[0]) : 0;
}

//
//
////////////////////////////////////////////////////////////
const float* Model::getColors() const {
    return ( _colors.size() > 0) ? &(_colors[0]) : 0;
}

//
//
////////////////////////////////////////////////////////////
const GLuint* Model::getPositionIndices() const {
    return ( _pIndex.size() > 0) ? &(_pIndex[0]) : 0;
}

//
//
////////////////////////////////////////////////////////////
const GLuint* Model::getNormalIndices() const {
    return ( _nIndex.size() > 0) ? &(_nIndex[0]) : 0;
}

//
//
////////////////////////////////////////////////////////////
const GLuint* Model::getTexCoordIndices() const {
    return ( _tIndex.size() > 0) ? &(_tIndex[0]) : 0;
}

//
//
////////////////////////////////////////////////////////////
const GLuint* Model::getTangentIndices() const {
    return ( _tanIndex.size() > 0) ? &(_tanIndex[0]) : 0;
}

//
//
////////////////////////////////////////////////////////////
const GLuint* Model::getColorIndices() const {
    return ( _cIndex.size() > 0) ? &(_cIndex[0]) : 0;
}

const int* Model::getMaterialIndices() const
{
    return ( _mIndex.size() > 0) ? &(_mIndex[0]) : 0;
}

//
//
////////////////////////////////////////////////////////////
int Model::getPositionCount() const {
    return (_posSize > 0) ? (int)_positions.size() / _posSize : 0;
}

//
//
////////////////////////////////////////////////////////////
int Model::getNormalCount() const {
    return (int)_normals.size() / 3;
}

//
//
////////////////////////////////////////////////////////////
int Model::getTexCoordCount() const {
    return (_tcSize > 0) ? (int)_texCoords.size() / _tcSize : 0;
}

//
//
////////////////////////////////////////////////////////////
int Model::getTangentCount() const {
    return (int)_sTangents.size() / 3;
}

//
//
////////////////////////////////////////////////////////////
int Model::getColorCount() const {
    return (_cSize > 0) ? (int)_colors.size() / _cSize : 0;
}

//
//
////////////////////////////////////////////////////////////
int Model::getIndexCount() const {
    return (int)_pIndex.size();
}

//
//
////////////////////////////////////////////////////////////
int Model::getMaterialCount() const {
    return (int)_materials.size();
}

//
//
////////////////////////////////////////////////////////////
const Model::Material *Model::getMaterial(int id) const
{
    if (id<0 || id>=_materials.size()) return 0;
    return &(_materials[id]);
}

//compiled data access functions

//
//
////////////////////////////////////////////////////////////
const float* Model::getCompiledVertices() const {
    return (_vertices.size() > 0) ? &_vertices[0] : 0;
}

//
//
////////////////////////////////////////////////////////////
const GLuint* Model::getCompiledIndices( Model::PrimType prim) const {
    switch (prim) {
        case Model::eptPoints:
            return (_indices[0].size() > 0) ? &_indices[0][0] : 0;
        case Model::eptEdges:
            return (_indices[1].size() > 0) ? &_indices[1][0] : 0;
        case Model::eptTriangles:
            return (_indices[2].size() > 0) ? &_indices[2][0] : 0;
        case Model::eptTrianglesWithAdjacency:
            return (_indices[3].size() > 0) ? &_indices[3][0] : 0;
    }

    return 0; 
}

//
//
////////////////////////////////////////////////////////////
int Model::getCompiledPositionOffset() const {
    return _pOffset;
}

//
//
////////////////////////////////////////////////////////////
int Model::getCompiledNormalOffset() const {
    return _nOffset;
}

//
//
////////////////////////////////////////////////////////////
int Model::getCompiledTexCoordOffset() const {
    return _tcOffset;
}

//
//
////////////////////////////////////////////////////////////
int Model::getCompiledTangentOffset() const {
    return _sTanOffset;
}

//
//
////////////////////////////////////////////////////////////
int Model::getCompiledColorOffset() const {
    return _cOffset;
}

// returns the size of the merged vertex in # of floats
//
//
////////////////////////////////////////////////////////////
int Model::getCompiledVertexSize() const {
    return _vtxSize;
}

//
//
////////////////////////////////////////////////////////////
int Model::getCompiledVertexCount() const {
    return (_vtxSize > 0) ? (int)_vertices.size() / _vtxSize : 0;
}

//
//
////////////////////////////////////////////////////////////
int Model::getCompiledIndexCount( Model::PrimType prim) const {
    switch (prim) {
        case Model::eptPoints:
            return (int)_indices[0].size();
        case Model::eptEdges:
            return (int)_indices[1].size();
        case Model::eptTriangles:
            return (int)_indices[2].size();
        case Model::eptTrianglesWithAdjacency:
            return (int)_indices[3].size();
    }

    return 0;
}

//
//
////////////////////////////////////////////////////////////
int Model::getOpenEdgeCount() const {
    return _openEdges;
}

};