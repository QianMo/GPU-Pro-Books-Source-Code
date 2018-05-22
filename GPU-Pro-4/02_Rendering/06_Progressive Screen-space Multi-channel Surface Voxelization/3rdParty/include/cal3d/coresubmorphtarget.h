//****************************************************************************//
// coresubmorphtarget.h                                                       //
// Copyright (C) 2001, 2002 Bruno 'Beosil' Heidelberger                       //
//****************************************************************************//
// This library is free software; you can redistribute it and/or modify it    //
// under the terms of the GNU Lesser General Public License as published by   //
// the Free Software Foundation; either version 2.1 of the License, or (at    //
// your option) any later version.                                            //
//****************************************************************************//

#ifndef CAL_CORESUBMORPHTARGET_H
#define CAL_CORESUBMORPHTARGET_H


#include "cal3d/global.h"
#include "cal3d/vector.h"


class CAL3D_API CalCoreSubMorphTarget
{
public:
  struct BlendVertex
  {
     CalVector position;
     CalVector normal;
  };
  
public:
  CalCoreSubMorphTarget()  { }
  ~CalCoreSubMorphTarget() { }

  int getBlendVertexCount();
  std::vector<BlendVertex>& getVectorBlendVertex();
  bool reserve(int blendVertexCount);
  bool setBlendVertex(int vertexId, const BlendVertex& vertex);

private:
  std::vector<BlendVertex> m_vectorBlendVertex;
};
#endif
//****************************************************************************//
