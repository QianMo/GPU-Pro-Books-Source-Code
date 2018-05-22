#ifndef COOKING_H

#define COOKING_H

#include "NxCooking.h"

class NxPMap;
class NxTriangleMesh;
class NxUserOutputStream;

bool HasCookingLibrary(); // check to see if the cooking library is available or not!
bool InitCooking(NxUserAllocator* allocator = NULL, NxUserOutputStream* outputStream = NULL);
void CloseCooking();
bool CookConvexMesh(const NxConvexMeshDesc& desc, NxStream& stream);
bool CookTriangleMesh(const NxTriangleMeshDesc& desc, NxStream& stream);
//bool CookSoftBodyMesh(const NxSoftBodyMeshDesc& desc, NxStream& stream);
bool CreatePMap(NxPMap& pmap, const NxTriangleMesh& mesh, NxU32 density, NxUserOutputStream* outputStream = NULL);
bool ReleasePMap(NxPMap& pmap);


#endif
