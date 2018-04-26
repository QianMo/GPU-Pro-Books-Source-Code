#ifndef __SKIN_LOADER_H__
#define __SKIN_LOADER_H__

#include <vector>
#include <string>

struct SKIN_DATA;
struct ANIMATION_DATA;

struct SKIN_LOADER {

   enum RESULT {
      WRONG_PARAMS,
      NO_FILE,
      ERROR_LOADING,
      LOADED,
   };

   // loads mesh, skinning information and appropriate animation
	static RESULT loadData (const std::string& fileName, SKIN_DATA* skinData,	ANIMATION_DATA* animationData);

};


#endif // __SKIN_LOADER_H__