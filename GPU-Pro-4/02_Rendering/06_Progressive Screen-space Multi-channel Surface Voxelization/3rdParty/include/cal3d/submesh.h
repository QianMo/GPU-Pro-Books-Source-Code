//****************************************************************************//
// submesh.h                                                                  //
// Copyright (C) 2001, 2002 Bruno 'Beosil' Heidelberger                       //
//****************************************************************************//
// This library is free software; you can redistribute it and/or modify it    //
// under the terms of the GNU Lesser General Public License as published by   //
// the Free Software Foundation; either version 2.1 of the License, or (at    //
// your option) any later version.                                            //
//****************************************************************************//

#ifndef CAL_SUBMESH_H
#define CAL_SUBMESH_H


#include "cal3d/global.h"
#include "cal3d/vector.h"


class CalCoreSubmesh;


class CAL3D_API CalSubmesh
{
public:
  struct PhysicalProperty
  {
    CalVector position;
    CalVector positionOld;
    CalVector force;
  };

  struct TangentSpace
  {
    CalVector tangent;
    float crossFactor;
  };

  struct Face
  {
    CalIndex vertexId[3];
  };

public:
  CalSubmesh(CalCoreSubmesh* coreSubmesh);
  ~CalSubmesh() { }

  CalCoreSubmesh *getCoreSubmesh();
  int getCoreMaterialId();
  int getFaceCount();
  int getFaces(CalIndex *pFaceBuffer);
  std::vector<CalVector>& getVectorNormal();
  std::vector<std::vector<TangentSpace> >& getVectorVectorTangentSpace();
  std::vector<PhysicalProperty>& getVectorPhysicalProperty();
  std::vector<CalVector>& getVectorVertex();
  int getVertexCount();
  bool hasInternalData();
  void disableInternalData();
  void setCoreMaterialId(int coreMaterialId);
  void setLodLevel(float lodLevel);
  bool isTangentsEnabled(int mapId);
  bool enableTangents(int mapId, bool enabled);
  std::vector<float>& getVectorWeight();
  void setMorphTargetWeight(int blendId,float weight);
  float getMorphTargetWeight(int blendId);
  float getBaseWeight();
  int getMorphTargetWeightCount();
  std::vector<float>& getVectorMorphTargetWeight();

private:
  CalCoreSubmesh *m_pCoreSubmesh;
  std::vector<float> m_vectorMorphTargetWeight;
  std::vector<CalVector> m_vectorVertex;
  std::vector<CalVector> m_vectorNormal;
  std::vector<std::vector<TangentSpace> > m_vectorvectorTangentSpace;
  std::vector<Face> m_vectorFace;
  std::vector<PhysicalProperty> m_vectorPhysicalProperty;
  int m_vertexCount;
  int m_faceCount;
  int m_coreMaterialId;
  bool m_bInternalData;
};

#endif
