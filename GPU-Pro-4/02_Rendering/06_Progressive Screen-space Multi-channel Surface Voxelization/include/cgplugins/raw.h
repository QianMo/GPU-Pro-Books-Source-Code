#pragma once

#include "Mesh3D.h"

// NOTE: the class can not have local data as
// it is not directly alloced

class rawMesh3D : public Mesh3D
{
  public:
    void readFormat (const char *filename);
    void writeFormat (const char *filename);
};

