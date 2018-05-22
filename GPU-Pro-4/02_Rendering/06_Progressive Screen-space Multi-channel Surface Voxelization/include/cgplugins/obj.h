#pragma once

#include "Mesh3D.h"

// NOTE: the class can not have local data as
// it is not directly allocated

class objMesh3D : public Mesh3D
{
  public:
    void readFormat (const char *filename);
    void writeFormat (const char *filename);

  private:
    // internal query methods
    int findGroup(char *name);

    // internal geometry structuring methods
    unsigned int addGroup( char *name);

    void readMTL(char *filename);
};

