//****************************************************************************//
// renderer.h                                                                 //
// Copyright (C) 2001, 2002 Bruno 'Beosil' Heidelberger                       //
//****************************************************************************//
// This library is free software; you can redistribute it and/or modify it    //
// under the terms of the GNU Lesser General Public License as published by   //
// the Free Software Foundation; either version 2.1 of the License, or (at    //
// your option) any later version.                                            //
//****************************************************************************//

#ifndef CAL_RENDERER_H
#define CAL_RENDERER_H


#include "cal3d/global.h"


class CalModel;
class CalSubmesh;


class CAL3D_API CalRenderer
{
public:
  CalRenderer(CalModel* pModel);
  CalRenderer(CalRenderer* pRenderer); 
  ~CalRenderer() { }

  bool beginRendering();
  void endRendering();
  void getAmbientColor(unsigned char *pColorBuffer);
  void getDiffuseColor(unsigned char *pColorBuffer);
  int getFaceCount();
  int getFaces(CalIndex *pFaceBuffer);
  int getMapCount();
  Cal::UserData getMapUserData(int mapId);
  int getMeshCount();
  int getNormals(float *pNormalBuffer, int stride=0);
  float getShininess();
  void getSpecularColor(unsigned char *pColorBuffer);
  int getSubmeshCount(int meshId);
  int getTextureCoordinates(int mapId, float *pTextureCoordinateBuffer, int stride=0);
  int getVertexCount();
  int getVertices(float *pVertexBuffer, int stride=0);
  int getTangentSpaces(int mapId, float *pTangentSpaceBuffer, int stride=0);
  int getVerticesAndNormals(float *pVertexBuffer, int stride=0);
  int getVerticesNormalsAndTexCoords(float *pVertexBuffer,int NumTexCoords=1);
  bool isTangentsEnabled(int mapId);
  bool selectMeshSubmesh(int meshId, int submeshId);
  void setNormalization(bool normalize);

private:
  CalModel *m_pModel;
  CalSubmesh *m_pSelectedSubmesh;
};

#endif

//****************************************************************************//
