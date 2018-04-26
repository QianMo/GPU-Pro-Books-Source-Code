#ifndef _GLUT_MOUSE_H_
#define _GLUT_MOUSE_H_

// simple class for mouse handling based on GLUT 

class MOUSE
{
public:

   enum UPDATE_TYPE {
      REFRESH_BUTTONS,
      RESET_BUTTONS,
      REFRESH_COORDS
   };

   MOUSE() : m_previousX(0), m_previousY(0), m_buttonsState(0), m_rotationPtr(0), m_translationPtr(0), m_sensitivity(0.3f) { }

   // variables to be changed when some mouse-related events happen
   bool setCallbackVars(float *newRotation, float *newTranslation);

   bool init(float *newRotation, float *newTranslation);

   void updateMouseState(int x, int y, UPDATE_TYPE updateType = REFRESH_BUTTONS, int state = 0);

private:

   int   m_previousX, m_previousY;
   int   m_buttonsState;
   float m_sensitivity;

   // these are the expected values modified by the callback function (not so nice,
   // but not bad for simple programs)
   float *m_rotationPtr, *m_translationPtr;   
   
};

MOUSE& getMouse();

#endif //_GLUT_MOUSE