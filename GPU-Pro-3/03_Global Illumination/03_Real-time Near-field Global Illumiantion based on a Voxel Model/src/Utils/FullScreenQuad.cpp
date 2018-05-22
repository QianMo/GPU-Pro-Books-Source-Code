#include "FullScreenQuad.h"

GLuint FullScreenQuad::displayListQuad = 0;

/////////////////////////////////// STATIC METHODS

void FullScreenQuad::setupQuadDisplayList()
{
   if(!glIsList(displayListQuad))
   {
      displayListQuad =  glGenLists(1);
      glNewList(displayListQuad, GL_COMPILE);
      
      glBegin(GL_QUADS);
      glTexCoord2f(0,0);		glVertex2f(-1,-1);
      glTexCoord2f(1,0);		glVertex2f( 1,-1);
      glTexCoord2f(1,1);		glVertex2f( 1, 1);
      glTexCoord2f(0,1);		glVertex2f(-1, 1);
      glEnd();

      glEndList();

   }
}


void FullScreenQuad::setupRendering()
{
   glMatrixMode(GL_MODELVIEW); 
   glPushMatrix(); 
   glLoadIdentity(); 

   glMatrixMode(GL_PROJECTION); 
   glPushMatrix(); 
   glLoadIdentity();

   glDepthMask(GL_FALSE); // do not write this quad to the depth buffer

}

void FullScreenQuad::drawOnly()
{
   glCallList(displayListQuad);
}


void FullScreenQuad::resetRendering()
{

   glDepthMask(GL_TRUE);

   glPopMatrix();  // PROJECTION
   glMatrixMode(GL_MODELVIEW);
   glPopMatrix(); // MODELVIEW

}

void FullScreenQuad::drawComplete()
{
   glMatrixMode(GL_MODELVIEW); 
   glPushMatrix(); 
   glLoadIdentity(); 

   glMatrixMode(GL_PROJECTION); 
   glPushMatrix(); 
   glLoadIdentity();

   glDepthMask(GL_FALSE); // do not write this quad to the depth buffer

   //glCallList(displayListQuad);
      glBegin(GL_QUADS);
      glTexCoord2f(0,0);		glVertex2f(-1,-1);
      glTexCoord2f(1,0);		glVertex2f( 1,-1);
      glTexCoord2f(1,1);		glVertex2f( 1, 1);
      glTexCoord2f(0,1);		glVertex2f(-1, 1);
      glEnd();

   glDepthMask(GL_TRUE);

   glPopMatrix();  // PROJECTION
   glMatrixMode(GL_MODELVIEW);
   glPopMatrix(); // MODELVIEW
}
