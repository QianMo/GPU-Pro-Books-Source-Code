///////////////////////////////////////////////////////////////////////
// Real-time Near-Field Global Illumination based on a Voxel Model
// GPU Pro 3
///////////////////////////////////////////////////////////////////////
// Authors : Sinje Thiedemann, Niklas Henrich, 
//           Thorsten Grosch, Stefan Mueller
// Created:  August 2011
///////////////////////////////////////////////////////////////////////

#ifndef BITMASK_H
#define BITMASK_H

#include "OpenGL.h"

#include <vector>

using std::vector;

// Creates and manages voxel/ray intersection bitmasks

class Bitmask
{
public:

   static void createBitmasks();

private:
   Bitmask() {}
   ~Bitmask() {}

   static void create1DBitmaskOR(); 
   static void create1DBitmaskXOR(); 
   static void create2DBitmaskXORRays();
};

#endif