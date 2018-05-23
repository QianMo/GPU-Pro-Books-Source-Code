/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#pragma once
#ifndef BE_SCENE_INLINE_MESH_SERIALIZATION
#define BE_SCENE_INLINE_MESH_SERIALIZATION

#include "beScene.h"
#include <beCore/beParameterSet.h>
#include <beCore/beSerializationJobs.h>

#include <beGraphics/beMaterial.h>

#include <lean/smart/resource_ptr.h>

namespace beScene
{

class AssembledMesh;
class RenderableMesh;
class RenderableMaterial;
class MeshCache;
class RenderableMaterialCache;

/// Schedules the given mesh for inline serialization.
BE_SCENE_API void SaveMesh(const AssembledMesh *mesh,
	beCore::ParameterSet &parameters, beCore::SerializationQueue<beCore::SaveJob> &queue);
/// Schedules the given mesh for inline serialization.
BE_SCENE_API void SaveMesh(const RenderableMesh *mesh,
	beCore::ParameterSet &parameters, beCore::SerializationQueue<beCore::SaveJob> &queue);

/// Saves the given mesh to the given XML node.
BE_SCENE_API void SaveMesh(const RenderableMesh &mesh, rapidxml::xml_node<lean::utf8_t> &node);

/// Load the given mesh from the given XML node.
BE_SCENE_API void LoadMesh(RenderableMesh &mesh, const rapidxml::xml_node<lean::utf8_t> &node,
						   MeshCache &meshes, beg::MaterialCache &materials, RenderableMaterialCache &renderableMaterials,
						   const RenderableMaterial *pDefaultMaterial);
/// Creates a mesh from the given XML node.
BE_SCENE_API lean::resource_ptr<RenderableMesh, lean::critical_ref> LoadMesh(const rapidxml::xml_node<lean::utf8_t> &node,
																			 MeshCache &meshes, beg::MaterialCache &materials,
																			 RenderableMaterialCache &renderableMaterials,
																			 const RenderableMaterial *pDefaultMaterial);

} // namespace

#endif
