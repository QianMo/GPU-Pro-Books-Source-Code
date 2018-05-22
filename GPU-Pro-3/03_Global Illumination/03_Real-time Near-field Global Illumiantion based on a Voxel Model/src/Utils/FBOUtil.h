#ifndef FBOUTIL_H
#define FBOUTIL_H

#include "OpenGL.h"


class FBOUtil
{
public:
   // 2 attachments
   static GLenum buffers01[2];
   static GLenum buffers12[2];
   static GLenum buffers13[2];
   static GLenum buffers04[2];
   static GLenum buffers14[2];
   static GLenum buffers23[2];
   static GLenum buffers34[2];
   static GLenum buffers56[2];

   // 3 attachments
   static GLenum buffers012[3];
   static GLenum buffers123[3];
   static GLenum buffers134[3];
   static GLenum buffers567[3];

   // 4 attachments
   static GLenum buffers0123[4];
   static GLenum buffers0125[4];

   // 5 attachments
   static GLenum buffers01234[5];

   // 6 attachments
   static GLenum buffers012345[6];


   // 8 attachments (all)
   static GLenum buffersAll[8];

private:
   FBOUtil() {};


};

#endif
