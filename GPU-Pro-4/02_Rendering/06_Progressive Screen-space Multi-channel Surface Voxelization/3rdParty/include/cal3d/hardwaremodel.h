//****************************************************************************//
// hardwaremodel.h                                                         //
// Copyright (C) 2004 Desmecht Laurent                                        //
//****************************************************************************//
// This library is free software; you can redistribute it and/or modify it    //
// under the terms of the GNU Lesser General Public License as published by   //
// the Free Software Foundation; either version 2.1 of the License, or (at    //
// your option) any later version.                                            //
//****************************************************************************//

#ifndef CAL_HARDWAREMODEL_H
#define CAL_HARDWAREMODEL_H


#include "cal3d/global.h"
#include "cal3d/coresubmesh.h"


class CalCoreModel;
class CalSkeleton;
class CalCoreMaterial;


class CAL3D_API CalHardwareModel
{
public:
  struct CalHardwareMesh
  {
    std::vector<int> m_vectorBonesIndices;
    
    int baseVertexIndex;
    int vertexCount;
    int startIndex;
    int faceCount;
    CalCoreMaterial *pCoreMaterial;

    int meshId,submeshId;
  };

public:
  CalHardwareModel(CalCoreModel* pCoreModel);
  ~CalHardwareModel() { }
  
  void setVertexBuffer( char * pVertexBuffer, int stride); 
  void setIndexBuffer( CalIndex * pIndexBuffer); 
  void setNormalBuffer( char * pNormalBuffer, int stride); 
  void setWeightBuffer( char * pWeightBuffer, int stride); 
  void setMatrixIndexBuffer( char * pMatrixIndexBuffer, int stride); 
  void setTextureCoordNum(int textureCoordNum);
  void setTextureCoordBuffer(int mapId, char * pTextureCoordBuffer, int stride);
  void setTangentSpaceBuffer(int mapId, char * pTangentSpaceBuffer, int stride);
  void setCoreMeshIds(const std::vector<int>& coreMeshIds);

  bool load(int baseVertexIndex, int startIndex,int maxBonesPerMesh);
      
  std::vector<CalHardwareMesh> & getVectorHardwareMesh();
  void getAmbientColor(unsigned char *pColorBuffer);
  void getDiffuseColor(unsigned char *pColorBuffer);
  void getSpecularColor(unsigned char *pColorBuffer);
  const CalQuaternion & getRotationBoneSpace(int boneId, CalSkeleton *pSkeleton);
  const CalVector & getTranslationBoneSpace(int boneId, CalSkeleton *pSkeleton);

  float getShininess();
  
  int getHardwareMeshCount();
  int getFaceCount();
  int getVertexCount();
  int getBoneCount();

  int getBaseVertexIndex();
  int getStartIndex();

  int getTotalFaceCount();
  int getTotalVertexCount();    

  Cal::UserData getMapUserData(int mapId);
  
  bool selectHardwareMesh(size_t meshId);
  
private:
  bool canAddFace(CalHardwareMesh &hardwareMesh, CalCoreSubmesh::Face & face,std::vector<CalCoreSubmesh::Vertex>& vectorVertex, int maxBonesPerMesh);
  int  addVertex(CalHardwareMesh &hardwareMesh, int indice , CalCoreSubmesh *pCoreSubmesh, int maxBonesPerMesh);
  int  addBoneIndice(CalHardwareMesh &hardwareMesh, int Indice, int maxBonesPerMesh);  
    

private:
  
  std::vector<CalHardwareMesh> m_vectorHardwareMesh;
  std::vector<CalIndex> m_vectorVertexIndiceUsed;
  int m_selectedHardwareMesh;
  std::vector<int> m_coreMeshIds;
  CalCoreModel *m_pCoreModel;
  
  
  char * m_pVertexBuffer;
  int m_vertexStride;
  char * m_pNormalBuffer;
  int m_normalStride;
  char * m_pWeightBuffer;
  int m_weightStride;
  char * m_pMatrixIndexBuffer;
  int m_matrixIndexStride;
  char * m_pTextureCoordBuffer[8];
  int m_textureCoordStride[8];
  int m_textureCoordNum;  
  char * m_pTangentSpaceBuffer[8];
  int m_tangentSpaceStride[8];
  
  CalIndex * m_pIndexBuffer;

  int m_totalVertexCount;
  int m_totalFaceCount;
};

#endif
