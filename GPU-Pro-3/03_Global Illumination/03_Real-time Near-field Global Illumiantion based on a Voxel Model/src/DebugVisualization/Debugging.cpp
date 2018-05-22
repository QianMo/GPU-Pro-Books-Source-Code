#include "Debugging.h"
#include "DebugVisualization/VoxelVisualization.h"

#include "Qt/Settings.h"
#include "Lighting/Sampling.h"

#include "Scene/Camera.h"
#include "Scene/Scene.h"

#include "Utils/GLError.h"
#include "Utils/ShaderProgram.h"

bool Debugging::showLambertRays = true;

Debugging::Debugging()
{
   quadric = gluNewQuadric();

   createPixelDisplayList();

}

void Debugging::createPixelDisplayList()
{
   pixelDisplayList = glGenLists(1);
   glNewList(pixelDisplayList, GL_COMPILE);
   glBegin(GL_POINTS);
   for(int y = 0; y < SCENE->getWindowHeight(); y+=40)
   {
      for(int x = 0; x < SCENE->getWindowWidth(); x+=40)
      {
         glVertex2i(x, y);
      }
   }
   glEnd();
   glEndList();
}


void Debugging::drawWorldSpaceAxes()
{
   glLineWidth(4.0);
   glBegin(GL_LINES);
      // x axis
      glColor3f(1, 0, 0);
      glVertex3f(0, 0, 0);
      glVertex3f(1, 0, 0);

      // y axis
      glColor3f(0, 1, 0);
      glVertex3f(0, 0, 0);
      glVertex3f(0, 1, 0);

      // z axis
      glColor3f(0, 0, 1);
      glVertex3f(0, 0, 0);
      glVertex3f(0, 0, 1);
   glEnd();

}



void Debugging::drawOrthoFrustum(const Camera* const cam,
                             bool withDiagonals,
                             bool highlightFrustum,
                             bool overlapped,
                             bool withOrthogonalFrustums)
{
   withOrthogonalFrustums = withOrthogonalFrustums;
   Frustum f = cam->getFrustum();
   const GLfloat* eye = &cam->getEye()[0];
   //glm::vec3 viewDir = cam->getViewDirection();

   V(glEnable(GL_DEPTH_TEST));
   glDisable(GL_LIGHTING);
   glUseProgram(0);

   V(glPolygonMode(GL_FRONT_AND_BACK, GL_LINE));


   glColor4d(0.2, 0.2, 0.2 ,1);

   glPointSize(10.0);
   glBegin(GL_POINTS);
   // Eye
   glVertex3fv(&cam->getEye()[0]);
   if(withDiagonals)
   {
      // Target
      glColor3d(1,1,0);
      glVertex3fv(&cam->getTarget()[0]);
   }
   glEnd();

   V(glLineWidth(3.0));

   glm::vec3 to = cam->getEye() - 0.5f * cam->getViewDirection();
   glm::vec3 up = cam->getEye() + 0.5f * cam->getUpVector();
   glm::vec3 right = cam->getEye() + 0.5f * cam->getRightVector();

   glBegin(GL_LINES);
      
   // Viewing direction
      glColor3f(0,0,1);
      glVertex3fv(eye);
      glVertex3fv(&to[0]);
      
      // UpVector
      glColor3f(0,1,0);
      glVertex3fv(eye);
      glVertex3fv(&up[0]);
      
      //// Right Vector
      glColor3f(1,0,0);
      glVertex3fv(eye);
      glVertex3fv(&right[0]);

   V(glEnd()); 

   if(withOrthogonalFrustums)
   {
      V(glPushMatrix());
         glMultMatrixf(&cam->getInverseViewMatrixOrthogonalX()[0][0]);
         glBegin(GL_POINTS);
            glColor3f(0, 0, 0);
            glVertex3f(0, 0, 0);
         glEnd();
         glBegin(GL_LINES);
            glColor3f(0, 0, 1);
            glVertex3f(0, 0, 0);
            glVertex3f(0, 0, 0.5);
            glColor3f(0, 1, 0);
            glVertex3f(0, 0, 0);
            glVertex3f(0, 0.5, 0);
            glColor3f(1, 0, 0);
            glVertex3f(0, 0, 0);
            glVertex3f(0.5, 0, 0);
         glEnd();
         glColor3d(0,0,0);
         // Frustum
         glLineWidth(1.0);
         VoxelVisualization::drawBox(f.left, f.bottom, -f.zFar, f.right, f.top, -f.zNear); // looking along negative z axis

      glPopMatrix();

      V(glPushMatrix());
         glMultMatrixf(&cam->getInverseViewMatrixOrthogonalY()[0][0]);
         glBegin(GL_POINTS);
            glColor3f(0, 0, 0);
            glVertex3f(0, 0, 0);
         glEnd();
         glLineWidth(3.0);
         glBegin(GL_LINES);
            glColor3f(0, 0, 1);
            glVertex3f(0, 0, 0);
            glVertex3f(0, 0, 0.5);
            glColor3f(0, 1, 0);
            glVertex3f(0, 0, 0);
            glVertex3f(0, 0.5, 0);
            glColor3f(1, 0, 0);
            glVertex3f(0, 0, 0);
            glVertex3f(0.5, 0, 0);
         glEnd();
         V(glColor3d(0,0,0));
         // Frustum
         V(glLineWidth(1.0));
         VoxelVisualization::drawBox(f.left, f.bottom, -f.zFar, f.right, f.top, -f.zNear); // looking along negative z axis

      glPopMatrix();
   }
   V(glLineWidth(2.0));
   V(glColor3d(0.2,0.2,0.2));

   V(glPushMatrix());
      V(glMultMatrixf(&cam->getInverseViewMatrix()[0][0]));

      if(highlightFrustum)
      {
         V(glColor3d(0,1,0));
         glLineWidth(4.0);
      }
      if(overlapped)
      {
         glLineWidth(6.0);
         glColor3d(1,0,1);
      }
      // Frustum
      VoxelVisualization::drawBox(f.left, f.bottom, -f.zFar, f.right, f.top, -f.zNear); // looking along negative z axis

      V(glColor3d(0,0,0));
      glLineWidth(1.0);

      // 4 box diagonals
      if(withDiagonals)
      {
         glBegin(GL_LINES);

         glVertex3f(f.left, f.bottom, -f.zNear);
         glVertex3f(f.right, f.top, -f.zFar);

         glVertex3f(f.left, f.top, -f.zNear);
         glVertex3f(f.right, f.bottom, -f.zFar);

         glEnd();
      }
      
   glPopMatrix();

   glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
   V(glDisable(GL_DEPTH_TEST));


}

void Debugging::drawPerspectiveFrustum(Camera* camera)
{
   glm::vec3 eye = camera->getEye();
   Frustum f = camera->getFrustum();

   //cout << f.left << endl;
   //cout << f.right << endl;
   //cout << f.bottom << endl;
   //cout << f.top << endl;
   //cout << f.zNear << endl;
   //cout << f.zFar << endl;

   glEnable(GL_DEPTH_TEST);
   glColor3d(0, 0, 0);

   glPushMatrix();
   glMultMatrixf(&camera->getInverseViewMatrix()[0][0]);

   // Eye
   glPointSize(15.0);
   glBegin(GL_POINTS);
   glVertex3f(0, 0, 0);
   glEnd();

   // Viewing direction
   glLineWidth(2.0);
   glBegin(GL_LINES);
   glVertex3f(0, 0, 0);
   glVertex3f(0, 0, -1);
   glEnd();

   // Frustum
   glLineWidth(1.0);

   float t;

   // left bottom
   glm::vec3 leftBottomNear, leftBottomFar;
   glm::vec3 dirLeftBottom;
   leftBottomNear = glm::vec3(f.left, f.bottom, -f.zNear);
   dirLeftBottom = glm::normalize(leftBottomNear);
   t = (-f.zFar - (-f.zNear)) / dirLeftBottom.z;
   leftBottomFar = leftBottomNear + t * dirLeftBottom;

   // right bottom
   glm::vec3 rightBottomNear, rightBottomFar;
   glm::vec3 dirRightBottom;
   rightBottomNear = glm::vec3(f.right, f.bottom, -f.zNear);
   dirRightBottom = glm::normalize(rightBottomNear);
   t = (-f.zFar - (-f.zNear)) / dirRightBottom.z;
   rightBottomFar = rightBottomNear + t * dirRightBottom;

   // left top
   glm::vec3 leftTopNear, leftTopFar;
   glm::vec3 dirLeftTop;
   leftTopNear = glm::vec3(f.left, f.top, -f.zNear);
   dirLeftTop = glm::normalize(leftTopNear);
   t = (-f.zFar - (-f.zNear)) / dirLeftTop.z;
   leftTopFar = leftTopNear + t * dirLeftTop;

   // right top
   glm::vec3 rightTopNear, rightTopFar;
   glm::vec3 dirRightTop;
   rightTopNear = glm::vec3(f.right, f.top, -f.zNear);
   dirRightTop = glm::normalize(rightTopNear);
   t = (-f.zFar - (-f.zNear)) / dirRightTop.z;
   rightTopFar = rightTopNear + t * dirRightTop;

   
   glBegin(GL_LINES);

   glVertex3fv(&leftBottomNear[0]);  glVertex3fv(&leftBottomFar[0]);
   glVertex3fv(&rightBottomNear[0]); glVertex3fv(&rightBottomFar[0]);
   glVertex3fv(&leftTopNear[0]);     glVertex3fv(&leftTopFar[0]);
   glVertex3fv(&rightTopNear[0]);    glVertex3fv(&rightTopFar[0]);

   // Planes
   glVertex3fv(&leftBottomNear[0]);   glVertex3fv(&rightBottomNear[0]);
   glVertex3fv(&leftTopNear[0]);      glVertex3fv(&rightTopNear[0]);
   glVertex3fv(&leftBottomNear[0]);   glVertex3fv(&leftTopNear[0]);
   glVertex3fv(&rightBottomNear[0]);   glVertex3fv(&rightTopNear[0]);

   glVertex3fv(&leftBottomFar[0]);   glVertex3fv(&rightBottomFar[0]);
   glVertex3fv(&leftTopFar[0]);      glVertex3fv(&rightTopFar[0]);
   glVertex3fv(&leftBottomFar[0]);   glVertex3fv(&leftTopFar[0]);
   glVertex3fv(&rightBottomFar[0]);   glVertex3fv(&rightTopFar[0]);

   glEnd();
   

   glPopMatrix();

   glDisable(GL_DEPTH_TEST);

}

void initSampleDrawing()
{
   // draw all pairs of the same bounce
   glDisable(GL_LIGHTING);
   glUseProgram(0);
   glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
   glClearColor(1, 1, 0.9f, 1);
   glClear(GL_COLOR_BUFFER_BIT);

   glPushAttrib(GL_VIEWPORT_BIT);
   glViewport(0, 0, 512, 512);

   glMatrixMode(GL_PROJECTION);
   glPushMatrix();
   glLoadIdentity();
   glOrtho(0, 1, 0, 1, -10 ,10);

   glMatrixMode(GL_MODELVIEW);
   glPushMatrix();
   glLoadIdentity();

   glPointSize(5.0);

}

void endSampleDrawing()
{
   glPopAttrib();

   glPopMatrix();
   glMatrixMode(GL_PROJECTION);
   glPopMatrix();

   glMatrixMode(GL_MODELVIEW);

}

void Debugging::renderPixelDisplayList(GLuint list, int width, int height)
{
   glDisable(GL_LIGHTING);
   glUseProgram(0);

   glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);

   glPointSize(1.0);

   //glClearColor(0,0,1,1);
   //glClear(GL_COLOR_BUFFER_BIT);

   glMatrixMode(GL_PROJECTION);
   glPushMatrix();
   glLoadIdentity();
   glOrtho(0, width, 0, height, -10 ,10);
   
   glMatrixMode(GL_MODELVIEW);
   glPushMatrix();
      glLoadIdentity();

      glColor4f(1, 1, 0, 1);

      glCallList(list);

   glPopMatrix();

   glMatrixMode(GL_PROJECTION);
   glPopMatrix();

   glMatrixMode(GL_MODELVIEW);

}
