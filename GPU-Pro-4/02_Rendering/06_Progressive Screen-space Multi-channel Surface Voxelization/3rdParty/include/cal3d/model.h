//****************************************************************************//
// model.h                                                                    //
// Copyright (C) 2001, 2002 Bruno 'Beosil' Heidelberger                       //
//****************************************************************************//
// This library is free software; you can redistribute it and/or modify it    //
// under the terms of the GNU Lesser General Public License as published by   //
// the Free Software Foundation; either version 2.1 of the License, or (at    //
// your option) any later version.                                            //
//****************************************************************************//

#ifndef CAL_MODEL_H
#define CAL_MODEL_H


#include "cal3d/global.h"
#include "cal3d/vector.h"


class CalCoreModel;
class CalSkeleton;
class CalAbstractMixer;
class CalMixer;
class CalMorphTargetMixer;
class CalPhysique;
class CalSpringSystem;
class CalRenderer;
class CalMesh;


class CAL3D_API CalModel : cal3d::noncopyable
{
public: 
  CalModel(CalCoreModel* pCoreModel);
  ~CalModel();

  bool attachMesh(int coreMeshId);
  bool detachMesh(int coreMeshId);
  CalCoreModel *getCoreModel() const;
  CalMesh *getMesh(int coreMeshId) const;
  CalMixer *getMixer() const;
  CalAbstractMixer *getAbstractMixer() const;
  void setAbstractMixer(CalAbstractMixer* pMixer);
  CalMorphTargetMixer *getMorphTargetMixer() const;
  CalPhysique *getPhysique() const;
  CalRenderer *getRenderer() const;
  CalSkeleton *getSkeleton() const;
  CalSpringSystem *getSpringSystem() const;
  CalBoundingBox & getBoundingBox(bool precision = false);
  Cal::UserData getUserData() const;
  std::vector<CalMesh *>& getVectorMesh();
  void setLodLevel(float lodLevel);
  void setMaterialSet(int setId);
  void setUserData(Cal::UserData userData);
  void update(float deltaTime);
  void disableInternalData();

private:
  CalCoreModel *m_pCoreModel;
  CalSkeleton *m_pSkeleton;
  CalAbstractMixer *m_pMixer;
  CalMorphTargetMixer *m_pMorphTargetMixer;
  CalPhysique *m_pPhysique;
  CalSpringSystem *m_pSpringSystem;
  CalRenderer *m_pRenderer;
  Cal::UserData m_userData;
  std::vector<CalMesh *> m_vectorMesh;
  CalBoundingBox m_boundingBox;
};

#endif

//****************************************************************************//
