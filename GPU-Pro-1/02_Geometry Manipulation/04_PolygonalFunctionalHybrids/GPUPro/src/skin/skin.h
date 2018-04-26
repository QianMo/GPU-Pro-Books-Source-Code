#ifndef __SKIN_H__
#define __SKIN_H__

#include <vector>
#include <assert.h>

#include "../defines.h"

// simple storage for the mesh, weights and bind pose matrices
struct SKIN_DATA {

	static const int MAX_JOINTS_PER_VERTEX = 4;


	struct VERTEX {
		float x, y, z;
	};

	struct TRIANGLE {
		int  i[3];
	};

   struct UV {
		float u, v;
	};

	struct JOINT_INDICES_DATA {
      // in fact this is just an integer index converted to float, which is converted back in the shader
		float i[MAX_JOINTS_PER_VERTEX];
	};

	struct VERTEX_WEIGHT_DATA {
		float w[MAX_JOINTS_PER_VERTEX];

	};

	typedef	std::vector<VERTEX>		VERTICES;
   typedef	std::vector<UV>	      UVS;	
	typedef	std::vector<JOINT_INDICES_DATA>	JOINT_INDICES;
	typedef	std::vector<VERTEX_WEIGHT_DATA>	VERTICES_WEIGHTS;

   typedef	std::vector<TRIANGLE>	TRIANGLES;
	
   typedef  std::vector<MATRIX_STORE>   MATRICES;
	
   enum UPDATE_MODE {
      UPDATE_RESERVE,
      UPDATE_RESIZE
   };

   // updates all appropriate buffers accordingly (to use operator []/push_back to access allocated data afterwards)   
   void updateNumbers(int vertexNumber, int triangleNumber, UPDATE_MODE updateMode = UPDATE_RESIZE)
   {
      vertexNumber   += (int)meshVertices.size();
      triangleNumber += (int)meshTriangles.size();

      updateBuffer(meshVertices, vertexNumber, updateMode);
      updateBuffer(meshUVs, vertexNumber, updateMode);
      updateBuffer(meshNormals, vertexNumber, updateMode);
      updateBuffer(jointIndices, vertexNumber, updateMode);
      updateBuffer(verticesWeights, vertexNumber, updateMode);

      // this is just for export:
      uvIndices.resize(vertexNumber, -1);

      VERTEX_WEIGHT_DATA color;
      color.w[0] = color.w[1] = color.w[2] = color.w[3] = 1.f;
      colors.resize(vertexNumber, color);
      
      updateBuffer(meshTriangles, triangleNumber, updateMode);      
   }

   template <typename T>
   void updateBuffer(T& buffer, int newSize, UPDATE_MODE updateMode) 
   {
      if (updateMode == UPDATE_RESIZE) {

         buffer.resize(newSize);
    
      } else if (updateMode == UPDATE_RESERVE) {

         buffer.reserve(newSize);          

      } else {

         assert(0);

      }
   }
	
	VERTICES			   meshVertices;
   UVS			      meshUVs;
   VERTICES			   meshNormals;
	JOINT_INDICES		jointIndices;
	VERTICES_WEIGHTS	verticesWeights;

   // this is only needed at the export stage:
   std::vector<int>  uvIndices;
   VERTICES_WEIGHTS	colors;

   TRIANGLES		   meshTriangles;

	MATRICES			   bindPoseJoints;

};


#endif // __SKIN_H__