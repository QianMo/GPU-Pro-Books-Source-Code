
#include <assert.h>

#include "loadUtils.h"


bool loadMatrix(std::istrstream& inputStream, MATRIX_STORE* matrixPtr)
{
   if (!matrixPtr) {
      assert(0);
      return false;
   }

   MATRIX_STORE& m =  *matrixPtr;
   
   // yeah, strange order of loading here
   
	inputStream >> m.x[0] >> m.x[4] >> m.x[8] >> m.x[12];
	inputStream >> m.x[1] >> m.x[5] >> m.x[9] >> m.x[13];
   inputStream >> m.x[2] >> m.x[6] >> m.x[10] >> m.x[14];
	inputStream >> m.x[3] >> m.x[7] >> m.x[11] >> m.x[15];

   return true;
}
