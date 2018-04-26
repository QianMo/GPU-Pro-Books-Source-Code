
#include <stdlib.h>

#if defined(__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include "mouse.h"

MOUSE& getMouse()
{
   static MOUSE mouse;
   return mouse;
}

void mouseButtonHandler(int button, int state, int x, int y)
{
 
   getMouse().updateMouseState( x, y,  state == GLUT_DOWN ? MOUSE::REFRESH_BUTTONS : MOUSE::RESET_BUTTONS , 1<<button);

 
   glutPostRedisplay();
}

void mouseMotionHandler(int x, int y)
{
   getMouse().updateMouseState(x, y, MOUSE::REFRESH_COORDS);
   glutPostRedisplay();
}

bool MOUSE::init(float *newRotation, float *newTranslation)
{
   glutMouseFunc(mouseButtonHandler);
   glutMotionFunc(mouseMotionHandler);

   return setCallbackVars(newRotation, newTranslation);
}

bool MOUSE::setCallbackVars(float *newRotation, float *newTranslation)
{ 
   m_rotationPtr = newRotation; 
   m_translationPtr = newTranslation; 
   return m_rotationPtr != 0 && m_translationPtr != 0; 
}

void MOUSE::updateMouseState(int x, int y, MOUSE::UPDATE_TYPE updateType, int state)
{
   if (updateType == REFRESH_BUTTONS) {

      m_buttonsState |= state;

   } else if (updateType == RESET_BUTTONS) {

      m_buttonsState = 0;

   } else if (updateType == REFRESH_COORDS) {

      if (m_rotationPtr && m_translationPtr) {

         float dx = x - m_previousX;
         float dy = y - m_previousY;

         if (m_buttonsState == 1) {

            m_rotationPtr[0] += dy * m_sensitivity;
            m_rotationPtr[1] += dx * m_sensitivity;

         } else if (m_buttonsState == 2) {

            m_translationPtr[0] += dx * m_sensitivity * 0.1f;
            m_translationPtr[1] -= dy * m_sensitivity * 0.1f;

         } else if (m_buttonsState == 3) {

            m_translationPtr[2] += dy * m_sensitivity * 0.1f;
         }     
      }
   }

   m_previousX = x;
   m_previousY = y;   
}