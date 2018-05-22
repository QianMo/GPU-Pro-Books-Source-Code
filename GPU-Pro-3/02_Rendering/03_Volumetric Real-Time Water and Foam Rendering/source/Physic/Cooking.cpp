#include <stdio.h>
//#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "Cooking.h"
#include "NxPhysicsSDK.h"
#include "NxPMap.h"
#include "PhysXLoader.h"

// if on the Windows, Linux or PS3 platform and 2.5.0 or higher, use the versioned Cooking
// interface via PhysXLoader
#if defined(WIN32) || defined(LINUX) || defined(__CELLOS_LV2__)
#if NX_SDK_VERSION_NUMBER >= 250
#define COOKING_INTERFACE 1
#endif
#endif

#ifdef COOKING_INTERFACE
NxCookingInterface *gCooking=0;
#endif

// -----------------------------------------------------------------------------
// ----------------------------- HasCookingLibrary -----------------------------
// -----------------------------------------------------------------------------
bool HasCookingLibrary(void) // check to see if the cooking library is available or not!
{
	bool ret = true;

#ifdef COOKING_INTERFACE
	if (gCooking == 0)
	{
		gCooking = NxGetCookingLib(NX_PHYSICS_SDK_VERSION);
		if (gCooking == 0)
			ret = false;
	}
#endif

	return ret;
}


// -----------------------------------------------------------------------------
// ------------------------------- CookConvexMesh ------------------------------
// -----------------------------------------------------------------------------
bool CookConvexMesh(const NxConvexMeshDesc& desc, NxStream& stream)
{
#ifdef COOKING_INTERFACE
	HasCookingLibrary();
	if (!gCooking)
		return false;
	return gCooking->NxCookConvexMesh(desc,stream);
#else
	return NxCookConvexMesh(desc,stream);
#endif
}


// -----------------------------------------------------------------------------
// ------------------------------ CookTriangleMesh -----------------------------
// -----------------------------------------------------------------------------
bool CookTriangleMesh(const NxTriangleMeshDesc& desc, NxStream& stream)
{
#ifdef COOKING_INTERFACE
	HasCookingLibrary();
	if ( !gCooking )
		return false;
	return gCooking->NxCookTriangleMesh(desc,stream);
#else
	return NxCookTriangleMesh(desc,stream);
#endif
}


// -----------------------------------------------------------------------------
// -------------------------------- InitCooking --------------------------------
// -----------------------------------------------------------------------------
bool InitCooking(NxUserAllocator* allocator, NxUserOutputStream* outputStream)
{
#ifdef COOKING_INTERFACE
	HasCookingLibrary();
	if (!gCooking)
		return false;
	return gCooking->NxInitCooking(allocator, outputStream);
#else
	return NxInitCooking(allocator, outputStream);
#endif
}


// -----------------------------------------------------------------------------
// -------------------------------- CloseCooking -------------------------------
// -----------------------------------------------------------------------------
void CloseCooking()
{
#ifdef COOKING_INTERFACE
	if (!gCooking)
		return;
	gCooking->NxCloseCooking();
#else
	return NxCloseCooking();
#endif
}


// -----------------------------------------------------------------------------
// --------------------------------- CreatePMap --------------------------------
// -----------------------------------------------------------------------------
bool CreatePMap(NxPMap& pmap, const NxTriangleMesh& mesh, NxU32 density, NxUserOutputStream* outputStream)
{
#ifdef COOKING_INTERFACE
	HasCookingLibrary();
	if (!gCooking)
		return false;
	return gCooking->NxCreatePMap(pmap,mesh,density,outputStream);
#else
	return NxCreatePMap(pmap,mesh,density,outputStream);
#endif
}


// -----------------------------------------------------------------------------
// -------------------------------- ReleasePMap --------------------------------
// -----------------------------------------------------------------------------
bool ReleasePMap(NxPMap& pmap)
{
#ifdef COOKING_INTERFACE
	HasCookingLibrary();
	if (!gCooking)
		return false;
	return gCooking->NxReleasePMap(pmap);
#else
	return NxReleasePMap(pmap);
#endif
}
