#include "CoutMethods.h"


void coutMatrix(const GLfloat* matrix) 
{
	for (int y= 0; y<4; y++)
	{
		for(int x = 0; x<4; x++)
			cout << matrix[x*4+y] << " \t\t";
		cout << endl;
	}
}

void coutMatrix(const glm::mat4 matrix)
{
   cout << endl;
   cout << matrix[0].x << "\t " << matrix[1].x << "\t " << matrix[2].x << "\t " << matrix[3].x << endl;
   cout << matrix[0].y << "\t " << matrix[1].y << "\t " << matrix[2].y << "\t " << matrix[3].y << endl;
   cout << matrix[0].z << "\t " << matrix[1].z << "\t " << matrix[2].z << "\t " << matrix[3].z << endl;
   cout << matrix[0].w << "\t " << matrix[1].w << "\t " << matrix[2].w << "\t " << matrix[3].w << endl;
   cout << endl;
}

void coutVec(const glm::vec2& vec)
{
   cout << vec.x << " " << vec.y;
}

void coutVec(const glm::vec3& vec)
{
   cout << vec.x << " " << vec.y << " " << vec.z;
}

void coutVec(const glm::vec4& vec)
{
   cout << vec.x << " " << vec.y << " " << vec.z << " " << vec.w;
}


void coutSceneData(const SceneData& data)
{
   cout << endl << endl << "Scene Name: " << data.name << endl << endl;

   cout << "== Camera == " << endl << endl;
   cout << "Fovh: " << data.cameraData.fovh << endl;
   cout << "Aspect: " << data.cameraData.aspect << endl;
   cout << "zNear: " << data.cameraData.zNear << endl;
   cout << "zFar: " << data.cameraData.zFar << endl;
   for(unsigned int i = 0; i < data.cameraData.poses.size(); i++)
   {
      cout << "# Pose: " << endl;
      Pose p = data.cameraData.poses[i];
      cout << "   Eye:      " << p.userPosition.x << " " << p.userPosition.y << " " << p.userPosition.z << endl;
      cout << "   ViewDir:  " << p.viewDirection.x << " " << p.viewDirection.y << " " << p.viewDirection.z << endl;
      cout << "     AngleX: " << p.angleX << endl;
      cout << "     AngleY: " << p.angleY << endl;
      cout << "     AngleZ: " << p.angleZ << endl;
      cout << "   UpVector: " << p.upVector.x << " " << p.upVector.y << " " << p.upVector.z << endl;

   }
   cout << endl;

   cout << "== Lights == " << endl << endl;
   if(data.spotLights.empty())
   {
      cout << "# No Spot Light defined." << endl;
   }
   else
   {
      for(unsigned int i = 0; i < data.spotLights.size(); i++)
      {
         SpotLightData spot = data.spotLights.at(i);
         cout << "# Spot Light" << endl;
         cout << "constantAttenuation:  " << spot.constantAttenuation << endl;
         cout << "quadraticAttenuation: " << spot.quadraticAttenuation << endl;
         cout << "cutoffAngle:  " << spot.cutoffAngle << endl;
         cout << "spotExponent: " << spot.spotExponent << endl;
         cout << "Position: " << spot.position.x << " " << spot.position.y << " " << spot.position.z << endl;
         cout << "I:        " << spot.I.x << " " << spot.I.y << " " << spot.I.z << endl;  
         cout << "AngleX: " << spot.angleX << endl;
         cout << "AngleY: " << spot.angleY << endl;
         cout << "AngleZ: " << spot.angleZ << endl;
         cout << endl;
      }
   }
   cout << endl;
   if(data.dynamicElements.empty())
   {
      cout << "== No Dynamic Elements." << endl << endl;
   }
   else
   {
      cout << "== Dynamic Elements == " << endl << endl;

      for(unsigned int e = 0; e < data.dynamicElements.size(); e++)
      {
         DynamicElementData dyn = data.dynamicElements.at(e);
         cout << "# Dynamic Element [ " << dyn.name << " ]" << endl;
         coutCommonElementData(dyn);
         cout << "Anim. File Start Index: " << dyn.animFileStartIndex << endl;
         cout << "Anim. File End Index:   " << dyn.animFileEndIndex << endl;
         cout << "Number of Anim. Files:  " << (dyn.animFileEndIndex - dyn.animFileStartIndex + 1) << endl;
         cout << endl;

         for(unsigned int i = 0; i < dyn.instances.size(); i++)
         {
            cout << "  # Instance " << i << endl;
            DynamicInstanceData inst = dyn.instances.at(i);
            cout << "Position: " << inst.position.x << " " << inst.position.y << " " << inst.position.z << endl;
            cout << "Rotation: " << inst.rotation.x << " " << inst.rotation.y << " " << inst.rotation.z << endl;
            cout << "Scale:    " << inst.scaleFactor << endl;
            cout << "Is movable by User: " << inst.isUserMovable << endl;
            cout << "Looping?  " << inst.looping << endl;
            cout << "Forwards? " << inst.forwards << endl;
            cout << "startAtFrame " << inst.startAtFrame << endl;
            cout << "stepInterval " << inst.stepInterval << endl;
            cout << endl;
         }

         cout << endl;

      }      
   }
   if(data.staticElements.empty())
   {
      cout << "== No Static Elements." << endl << endl;
   }
   else
   {
      cout << "== Static Elements == " << endl << endl;
      for(unsigned int i = 0; i < data.staticElements.size(); i++)
      {
         StaticElementData stat = data.staticElements.at(i);
         cout << "# Static Element [ " << stat.name << " ]" << endl;
         cout << "Position: " << stat.position.x << " " << stat.position.y << " " << stat.position.z << endl;
         cout << "Rotation: " << stat.rotation.x << " " << stat.rotation.y << " " << stat.rotation.z << endl;

         coutCommonElementData(stat);
         cout << endl;
      }      
   }


   cout << endl << endl; 

}

void coutCommonElementData(const CommonElementData& elem)
{
   cout << "LoaderSettings: " << endl;
   cout << "  path of model: " << elem.pathModel << endl;
   cout << "  path of atlas: " << elem.pathAtlas << endl;
   cout << "          atlas width:  " << elem.atlasWidth << endl;
   cout << "          atlas height: " << elem.atlasHeight << endl;
   cout << "  name: " << elem.name << endl;
   cout << "  defaultDrawMode: " << ObjModel::drawModeToString(elem.defaultDrawMode) << endl;
   cout << "  unitized: " << elem.unitized << endl;
   cout << "  centered: " << elem.centered << endl;
   cout << "  fixedScaleFactor: " << elem.fixedScaleFactor << endl;

   cout << " computedVertexNormals: " << elem.computedVertexNormals << endl;
   if(elem.computedVertexNormals)
   {
      cout << "    vertexNormalsAngle: " << elem.vertexNormalsAngle << endl;
      cout << "    vertexNormalsSmoothingGroups: " << elem.vertexNormalsSmoothingGroups<< endl;
   }
}