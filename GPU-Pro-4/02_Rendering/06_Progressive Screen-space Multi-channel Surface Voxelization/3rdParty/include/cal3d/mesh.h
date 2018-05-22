//****************************************************************************//
// mesh.h                                                                     //
// Copyright (C) 2001, 2002 Bruno 'Beosil' Heidelberger                       //
//****************************************************************************//
// This library is free software; you can redistribute it and/or modify it    //
// under the terms of the GNU Lesser General Public License as published by   //
// the Free Software Foundation; either version 2.1 of the License, or (at    //
// your option) any later version.                                            //
//****************************************************************************//

#ifndef CAL_MESH_H
#define CAL_MESH_H


#include "cal3d/global.h"


class CalModel;
class CalCoreMesh;
class CalSubmesh;


class CAL3D_API CalMesh
{
// constructors/destructor
public:
  CalMesh(CalCoreMesh *pCoreMesh);
  ~CalMesh();

  CalCoreMesh *getCoreMesh();
  CalSubmesh *getSubmesh(int id);
  int getSubmeshCount();
  std::vector<CalSubmesh *>& getVectorSubmesh();
  void setLodLevel(float lodLevel);
  void setMaterialSet(int setId);
  void setModel(CalModel *pModel);
  void disableInternalData();

private:
  CalModel *m_pModel;
  CalCoreMesh *m_pCoreMesh;
  std::vector<CalSubmesh *> m_vectorSubmesh;
};

#endif

//****************************************************************************//
