
#include <fstream>
#include <strstream>
#include <assert.h>

#include "../skin/skin.h"
#include "../skin/animationData.h"

#include "loadUtils.h"

#include "skinLoader.h"

SKIN_LOADER::RESULT SKIN_LOADER::loadData(const std::string& fileName, SKIN_DATA* skinData, ANIMATION_DATA* animationData)
{
   if (!skinData || !animationData) {
      assert(0);
      return WRONG_PARAMS;
   } 
   

   std::ifstream	fileStream;
	fileStream.open(fileName.c_str());

   if (!fileStream.is_open()) {
      animationData->framesNum = 0;
      animationData->matrices.resize(0);

      return NO_FILE;
   }

   if (!fileStream.good()) {
		assert(0);
		return ERROR_LOADING;
	}   
  
   std::istreambuf_iterator<char> fileStreamIterator( fileStream );
   std::istreambuf_iterator<char> endOfStream;

   std::string str( fileStreamIterator, endOfStream );     
   std::istrstream inputStream(str.c_str());

	int numTriangles = 0, numVertices = 0;
	inputStream >> numTriangles >> numVertices;	   
	
   skinData->updateNumbers(numVertices, numTriangles);

   int maxIndex = 0;

	for (int i = 0; i < numTriangles; i++) {
		SKIN_DATA::TRIANGLE& triangle = skinData->meshTriangles[i];

		inputStream >> triangle.i[0] >> triangle.i[1] >> triangle.i[2];

      maxIndex = std::max(triangle.i[2], std::max(triangle.i[0], triangle.i[1]));
	}
	
 
	for (int i=0; i < numVertices; i++) {

		SKIN_DATA::VERTEX&   vertex   =  skinData->meshVertices[i];
      SKIN_DATA::VERTEX&   normal   =  skinData->meshNormals[i];
      SKIN_DATA::UV&       uv       =  skinData->meshUVs[i];     
      SKIN_DATA::VERTEX_WEIGHT_DATA& color =  skinData->colors[i];
		inputStream >> vertex.x >> vertex.y >> vertex.z;
      inputStream >> normal.x >> normal.y >> normal.z;
      inputStream >> uv.u >> uv.v;

      inputStream >> color.w[0] >> color.w[1] >> color.w[2] >> color.w[3];
   }

	int numWeightedVertices = 0;
	inputStream >> numWeightedVertices;

	assert(numWeightedVertices == numVertices);

	for (int i = 0; i < numWeightedVertices; i++) {
		
		SKIN_DATA::JOINT_INDICES_DATA&	jointIndices	=	skinData->jointIndices[i];	
		SKIN_DATA::VERTEX_WEIGHT_DATA&	vertexWeights	=	skinData->verticesWeights[i];

      // for conformity all vertices have the same number of influences (no clustering here for the sake of simplicity) 
      for (int j = 0; j < SKIN_DATA::MAX_JOINTS_PER_VERTEX; j++) {			

			inputStream >> jointIndices.i[j];
			inputStream >> vertexWeights.w[j];
		}
	}
   
  	std::string tmp;

	int jointsNum = 0;
	inputStream >> jointsNum;
   
	skinData->bindPoseJoints.resize(jointsNum);
	
	for (int i = 0; i < jointsNum; i++) {
		int parentIdx;

		inputStream >> tmp;
		inputStream >> tmp >>parentIdx;      

		MATRIX_STORE m;
		// bind matrix 
		inputStream >> tmp;
      loadMatrix(inputStream, &m);
         
      inputStream >> tmp;
      loadMatrix(inputStream, &skinData->bindPoseJoints[i]);
	}

   inputStream >> tmp >> animationData->hasAttachmentInfo >> animationData->attachmentIndex;

   animationData->framesNum = 0;
	inputStream >> animationData->framesNum;

	animationData->matrices.resize(animationData->framesNum  * jointsNum);

   if (animationData->hasAttachmentInfo) {
      animationData->absoluteMatrices.resize(animationData->framesNum  * jointsNum);
   }

	animationData->matricesPerSample  = jointsNum;   

	for (int frame = 0; frame < animationData->framesNum ; frame++) {

		int frameNum;
		inputStream >> tmp >> frameNum;

		for (int i = 0; i < jointsNum; i++) {	

			inputStream >> tmp;// name

         int currentIndex = i + frame * jointsNum;
			
         loadMatrix(inputStream, &animationData->matrices[currentIndex]);
			
         if (animationData->hasAttachmentInfo && i == animationData->attachmentIndex) {

            inputStream >> tmp;// name

            loadMatrix(inputStream, &animationData->absoluteMatrices[currentIndex]);
         }
		}
	}

   fileStream.close();

   return LOADED;
}
