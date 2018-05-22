#ifndef DEBUGGING_H
#define DEBUGGING_H

class Camera;
class Scene;
class ShaderProgram;

#include "OpenGL.h"

#include "glm/glm.hpp" // OpenGL Mathematics Library

#include <vector>

using namespace std;

class Debugging
{
public:
   Debugging();

   static void renderPixelDisplayList(GLuint list, int width, int height);

   static void drawWorldSpaceAxes();

   /// Visualizes a camera (perspective camera) by a point at the eye position and
   /// a line pointing in the viewing direction. The frustum is drawn as a wireframe.
   static void drawPerspectiveFrustum(Camera* camera); 

   /// Visualizes a camera (orthographic camera) by a point at the eye position and
   /// a line pointing in the viewing direction. The frustum is drawn as a wireframe.
   static void drawOrthoFrustum(const Camera* const camera,
      bool withDiagonals = false,
      bool highlightFrustum = false,
      bool overlapped = false,
      bool withOrthogonalFrustums = false);

   static bool showLambertRays;

private:
   void createPixelDisplayList();

   GLUquadric* quadric;
   GLuint pixelDisplayList;

};

#endif
