//****************************************************************************//
// corematerial.h                                                             //
// Copyright (C) 2001, 2002 Bruno 'Beosil' Heidelberger                       //
//****************************************************************************//
// This library is free software; you can redistribute it and/or modify it    //
// under the terms of the GNU Lesser General Public License as published by   //
// the Free Software Foundation; either version 2.1 of the License, or (at    //
// your option) any later version.                                            //
//****************************************************************************//

#ifndef CAL_COREMATERIAL_H
#define CAL_COREMATERIAL_H


#include "cal3d/global.h"
#include "cal3d/refcounted.h"
#include "cal3d/refptr.h"


class CAL3D_API CalCoreMaterial : public cal3d::RefCounted
{
public:
  struct Color
  {
    unsigned char red;
    unsigned char green;
    unsigned char blue;
    unsigned char alpha;
  };

  struct Map
  {
    std::string strFilename;
    Cal::UserData userData;
  };

  CalCoreMaterial();

protected:
  ~CalCoreMaterial() { }

public:
  Color& getAmbientColor();
  Color& getDiffuseColor();
  int getMapCount();
  const std::string& getMapFilename(int mapId);
  Cal::UserData getMapUserData(int mapId);
  float getShininess();
  Color& getSpecularColor();
  Cal::UserData getUserData();
  std::vector<Map>& getVectorMap();
  bool reserve(int mapCount);
  void setAmbientColor(const Color& ambientColor);
  void setDiffuseColor(const Color& diffuseColor);
  bool setMap(int mapId, const Map& map);
  bool setMapUserData(int mapId, Cal::UserData userData);
  void setShininess(float shininess);
  void setSpecularColor(const Color& specularColor);
  void setFilename(const std::string& filename);
  const std::string& getFilename(void);
  void setName(const std::string& name);
  const std::string& getName(void);
  void setUserData(Cal::UserData userData);

private:
  Color m_ambientColor;
  Color m_diffuseColor;
  Color m_specularColor;
  float m_shininess;
  std::vector<Map> m_vectorMap;
  Cal::UserData m_userData;
  std::string m_name;
  std::string m_filename;
};
typedef cal3d::RefPtr<CalCoreMaterial> CalCoreMaterialPtr;

#endif

//****************************************************************************//
