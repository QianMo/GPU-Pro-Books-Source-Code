#ifndef __ANIMATION_DATA_H__
#define __ANIMATION_DATA_H__

#include <vector>
#include <string>

#include "../defines.h"

//#include <math/matrix.h>


// simple storage for a sequence of sampled transformation matrices
struct ANIMATION_DATA {
public:

   // either matrix with absolute transform or 
   enum MATRIX_TYPE {      
      MATRIX_PREMULTIPLIED,   // premultiplied by the inverse bind pose matrix
      MATRIX_ABSOLUTE,        // matrix with absolute transform
   };

   typedef  std::vector<MATRIX_STORE>  MATRICES;
   
   const MATRIX_STORE& getMatrixAt(int frameNum, int matrixNum, MATRIX_TYPE matrixType = MATRIX_PREMULTIPLIED) const 
   { 
      // clamp to maximum frame
      frameNum = frameNum % framesNum;

      int index = matrixNum + frameNum * matricesPerSample;
      return matrixType == MATRIX_ABSOLUTE ? absoluteMatrices[index] : matrices[index]; 
   }

   // retrieve the data required for attachment (use one the matrices as world parent transform)
   const MATRIX_STORE* getAttachmentMatrix(int frameNum)
   {
      if (!hasAttachmentInfo) {
         // no attachment available
         return NULL;
      }
      
      //int curFrame = (currentFrame) % animationData.framesNum;
      return &getMatrixAt( frameNum, attachmentIndex, ANIMATION_DATA::MATRIX_ABSOLUTE);      
   }

   int      framesNum, matricesPerSample;

   // basic information to attach objects to the joints (yeah, it should be in another system, not here)
   bool     hasAttachmentInfo;
   int      attachmentIndex;

   MATRICES matrices;
   MATRICES absoluteMatrices;

};


#endif // __ANIMATION_DATA_H__