#ifndef FULLSCREENQUAD_H
#define FULLSCREENQUAD_H

#include "OpenGL.h"


class FullScreenQuad
{

public:

   // STATIC METHODS
   static void setupQuadDisplayList();

   static void drawComplete();
   static void setupRendering();
   static void resetRendering();
   static void drawOnly();
private:
   static GLuint displayListQuad;

};


#endif
