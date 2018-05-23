/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#include "beSceneInternal/stdafx.h"
#include "beScene/beInlineMeshSerialization.h"
#include "beScene/beInlineMaterialSerialization.h"

#include <beEntitySystem/beSerializationParameters.h>
#include <beEntitySystem/beSerializationTasks.h>

#include "beScene/beSerializationParameters.h"
#include "beScene/beResourceManager.h"
#include "beScene/beEffectDrivenRenderer.h"

#include "beScene/beMesh.h"
#include "beScene/beAssembledMesh.h"
#include "beScene/beMeshCache.h"
#include "beScene/beRenderableMesh.h"
#include "beScene/beRenderableMeshCache.h"

#include "beScene/beRenderableMaterialCache.h"
#include "beScene/beInlineMaterialSerialization.h"

#include <lean/xml/utility.h>
#include <lean/xml/numeric.h>

#include <set>

#include <lean/logging/errors.h>

namespace beScene
{

// Saves the given mesh to the given XML node.
void SaveMesh(const RenderableMesh &mesh, rapidxml::xml_node<lean::utf8_t> &node)
{
	rapidxml::xml_document<utf8_t> &document = *node.document();

	const AssembledMesh *pSourceMesh = mesh.GetSource();
	utf8_ntr sourceName = bec::GetCachedName<utf8_ntr>(pSourceMesh);

	// Tied to one source mesh?
	if (pSourceMesh)
	{
		if (!sourceName.empty())
			lean::append_attribute(document, node, "source", sourceName);
		else
			LEAN_LOG_ERROR_MSG("Could not identify source mesh, will be lost");
	}

	RenderableMesh::MeshRange meshes = mesh.GetMeshes();
	RenderableMesh::MaterialRange materials = mesh.GetMaterials();

	// LODs
	for (RenderableMesh::LODRange lods = mesh.GetLODs(); lods; ++lods)
	{
		rapidxml::xml_node<utf8_t> &lodNode = *lean::allocate_node<utf8_t>(document, "lod");
		lean::append_float_attribute(document, lodNode, "distance", lods->Distance);
		node.append_node(&lodNode);

		// Subsets
		for (uint4 subsetIdx = lods->Subsets.Begin; subsetIdx < lods->Subsets.End; ++subsetIdx)
		{
			const Mesh *subsetMesh = meshes[subsetIdx];
			const RenderableMaterial *subsetMaterial = materials[subsetIdx];

			if (subsetMesh)
			{
				rapidxml::xml_node<utf8_t> &subsetNode = *lean::allocate_node<utf8_t>(document, "s");
				lean::append_attribute(document, subsetNode, "n", subsetMesh->GetName());
				lodNode.append_node(&subsetNode);

				utf8_ntr materialName = bec::GetCachedName<utf8_ntr>(subsetMaterial ? subsetMaterial->GetMaterial() : nullptr);
				if (!materialName.empty())
					lean::append_attribute(document, subsetNode, "material", materialName);
				else
					LEAN_LOG_ERROR_CTX("Could not identify material, will be lost", subsetMesh->GetName());

				if (!pSourceMesh)
				{
					utf8_ntr meshName = bec::GetCachedName<utf8_ntr>(subsetMesh->GetCompound());
					if (!meshName.empty())
						lean::append_attribute(document, subsetNode, "mesh", meshName);
					else
						LEAN_LOG_ERROR_CTX("Could not identify mesh, will be lost", subsetMesh->GetName());
				}
			}
			else
				LEAN_LOG_ERROR_CTX("Invalid mesh, ignoring", sourceName);
		}
	}
}

// Load the given mesh from the given XML node.
void LoadMesh(RenderableMesh &mesh, const rapidxml::xml_node<lean::utf8_t> &node,
			  MeshCache &meshes, beg::MaterialCache &materials, RenderableMaterialCache &renderableMaterials)
{
	// Mesh (single or distributed)
	utf8_ntr sourceName = lean::get_attribute(node, "source");

	if (!sourceName.empty())
	{
		const AssembledMesh *sourceMesh = meshes.GetByName(sourceName, true);

		RenderableMesh::LODRange sourceLODs = sourceMesh->GetLODs();
		RenderableMesh::MeshRange sourceMeshes = sourceMesh->GetMeshes();

		// Copy subset meshes
		for (uint4 lodIdx = 0, lodCount = Size4(sourceLODs); lodIdx < lodCount; ++lodIdx)
		{
			const RenderableMesh::LOD &lod = sourceLODs[lodIdx];
			for (uint4 subsetIdx = lod.Subsets.Begin; subsetIdx < lod.Subsets.End; ++subsetIdx)
				mesh.AddMeshWithMaterial(sourceMeshes[subsetIdx], nullptr, lodIdx);
		}
		mesh.SetSource(sourceMesh);
		
		RenderableMesh::LODRange lods = mesh.GetLODs();
		RenderableMesh::MeshRange meshes = mesh.GetMeshes();
		
		// LODs
		uint4 lodIdx = 0, lodCount = Size4(lods);
		for (const rapidxml::xml_node<utf8_t> *lodNode = node.first_node("lod");
			lodNode; lodNode = lodNode->next_sibling("lod"), ++lodIdx)
		{
			if (lodIdx < lodCount)
			{
				// Load LOD information
				mesh.SetLODDistance(
						lodIdx, 
						lean::get_float_attribute(*lodNode, "distance", sourceLODs[lodIdx].Distance)
					);

				const RenderableMesh::LOD &lod = lods[lodIdx];

				// Subset materials
				for (const rapidxml::xml_node<utf8_t> *subsetNode = lodNode->first_node();
					subsetNode; subsetNode = subsetNode->next_sibling())
				{
					utf8_ntr subsetName = lean::get_attribute(*subsetNode, "n"); 
					bool bSubsetIdentified = false;

					for (uint4 subsetIdx = lod.Subsets.Begin; subsetIdx < lod.Subsets.End; ++subsetIdx)
						// Find first matching subset with unset material
						if (meshes[subsetIdx]->GetName() == subsetName && !mesh.GetMaterials()[subsetIdx])
						{
							mesh.SetMeshWithMaterial( subsetIdx,
									nullptr,
									renderableMaterials.GetMaterial( materials.GetByName( lean::get_attribute(*subsetNode, "material"), true ) )
								);
							bSubsetIdentified = true;
							break;
						}

					if (!bSubsetIdentified)
						LEAN_LOG_ERROR_CTX("Subset could not be matched, information will be lost", subsetName);
				}
			}
			else
				LEAN_LOG_ERROR_CTX("LOD count changed, trailing LOD information will be lost", sourceName);
		}
	}
	else
	{
		// LODs
		uint4 lodIdx = 0;
		for (const rapidxml::xml_node<utf8_t> *lodNode = node.first_node("lod");
			lodNode; lodNode = lodNode->next_sibling("lod"), ++lodIdx)
		{
			// Load LOD information
			mesh.SetLODDistance( lodIdx,  lean::get_float_attribute(*lodNode, "distance", 0.0f) );

			// Subsets
			for (const rapidxml::xml_node<utf8_t> *subsetNode = lodNode->first_node();
				subsetNode; subsetNode = subsetNode->next_sibling())
			{
				const AssembledMesh *sourceMesh = meshes.GetByName( lean::get_attribute(*subsetNode, "mesh"), true );
				utf8_ntr subsetName = lean::get_attribute(*subsetNode, "n"); 
				bool bSubsetIdentified = false;
				
				for (RenderableMesh::MeshRange sourceMeshes = sourceMesh->GetMeshes(); sourceMeshes; ++sourceMeshes)
					if (sourceMeshes[0]->GetName() == subsetName)
					{
						mesh.AddMeshWithMaterial(
								sourceMeshes[0],
								renderableMaterials.GetMaterial( materials.GetByName( lean::get_attribute(*subsetNode, "material"), true ) ),
								lodIdx
							);
						bSubsetIdentified = true;
						break;
					}

				if (!bSubsetIdentified)
					LEAN_LOG_ERROR_CTX("Subset could not be matched, information will be lost", subsetName);
			}
		}
	}
}

// Creates a mesh from the given XML node.
lean::resource_ptr<RenderableMesh, lean::critical_ref> LoadMesh(const rapidxml::xml_node<lean::utf8_t> &node,
																MeshCache &meshes, beg::MaterialCache &materials,
																RenderableMaterialCache &renderableMaterials)
{
	lean::resource_ptr<RenderableMesh> mesh = new_resource RenderableMesh();
	LoadMesh(*mesh, node, meshes, materials, renderableMaterials);
	return mesh.transfer();
}

namespace
{


/// Serializes a list of materials.
class MeshImportSerializer : public beCore::SaveJob
{
private:
	typedef std::set<const AssembledMesh*> mesh_set;
	mesh_set m_meshes;

public:
	/// Adds the given material for serialization.
	void Add(const AssembledMesh *mesh)
	{
		LEAN_ASSERT_NOT_NULL( mesh );
		LEAN_ASSERT_NOT_NULL( mesh->GetCache() ); // TODO: Exceptions + check source & source cache
		m_meshes.insert(mesh);
	}

	/// Saves anything, e.g. to the given XML root node.
	void Save(rapidxml::xml_node<lean::utf8_t> &root, beCore::ParameterSet &parameters, beCore::SerializationQueue<beCore::SaveJob> &queue) const
	{
		rapidxml::xml_document<utf8_t> &document = *root.document();

		rapidxml::xml_node<utf8_t> &meshesNode = *lean::allocate_node<utf8_t>(document, "meshes");
		// ORDER: Append FIRST, otherwise parent document == nullptrs
		root.append_node(&meshesNode);

		for (mesh_set::const_iterator itMesh = m_meshes.begin(); itMesh != m_meshes.end(); itMesh++)
		{
			const AssembledMesh *mesh = *itMesh;
			const MeshCache *cache = mesh->GetCache();
			utf8_ntr name = cache->GetName(mesh);
			utf8_ntr file = cache->GetFile(mesh);
			
			if (name.empty())
				LEAN_LOG_ERROR_CTX("Imported mesh missing name, will be lost", file);
			else if (file.empty())
				LEAN_LOG_ERROR_CTX("Imported mesh missing file, will be lost", name);
			else
			{
				rapidxml::xml_node<utf8_t> &meshNode = *lean::allocate_node<utf8_t>(document, "m");
				lean::append_attribute( document, meshNode, "name", name );
				lean::append_attribute( document, meshNode, "file", cache->GetPathResolver().Shorten(file) );
				meshesNode.append_node(&meshNode);
			}
		}
	}
};

/// Serializes a list of materials.
class MeshSerializer : public beCore::SaveJob
{
private:
	typedef std::set<const RenderableMesh*> mesh_set;
	mesh_set m_meshes;

public:
	/// Adds the given material for serialization.
	void Add(const RenderableMesh *mesh)
	{
		LEAN_ASSERT_NOT_NULL( mesh );
		LEAN_ASSERT_NOT_NULL( mesh->GetCache() ); // TODO: Exceptions + check source & source cache
		m_meshes.insert(mesh);
	}

	/// Saves anything, e.g. to the given XML root node.
	void Save(rapidxml::xml_node<lean::utf8_t> &root, beCore::ParameterSet &parameters, beCore::SerializationQueue<beCore::SaveJob> &queue) const
	{
		rapidxml::xml_document<utf8_t> &document = *root.document();

		rapidxml::xml_node<utf8_t> &meshesNode = *lean::allocate_node<utf8_t>(document, "renderablemeshes");
		// ORDER: Append FIRST, otherwise parent document == nullptrs
		root.append_node(&meshesNode);

		for (mesh_set::const_iterator itMesh = m_meshes.begin(); itMesh != m_meshes.end(); itMesh++)
		{
			const RenderableMesh *mesh = *itMesh;
			const RenderableMeshCache *cache = mesh->GetCache();
			utf8_ntr name = cache->GetName(mesh);

			if (name.empty())
				LEAN_LOG_ERROR_MSG("Renderable mesh missing name, will be lost");
			else
			{
				rapidxml::xml_node<utf8_t> &meshNode = *lean::allocate_node<utf8_t>(document, "m");
				lean::append_attribute( document, meshNode, "name", name );
				// ORDER: Append FIRST, otherwise parent document == nullptr
				meshesNode.append_node(&meshNode);

				SaveMesh(*mesh, meshNode);
			}
		}
	}
};

struct InlineSerializationToken;

} // namespace

// Schedules the given mesh for inline serialization.
void SaveMesh(const AssembledMesh *mesh, beCore::ParameterSet &parameters, beCore::SerializationQueue<beCore::SaveJob> &queue)
{
	// Schedule mesh for serialization
	bees::GetOrMakeSaveJob<MeshImportSerializer, InlineSerializationToken>(
			parameters, "beScene.MeshImportSerializer", queue
		).Add(mesh);
}

// Schedules the given material for inline serialization.
void SaveMesh(const RenderableMesh *mesh, beCore::ParameterSet &parameters, beCore::SerializationQueue<beCore::SaveJob> &queue)
{
	// Schedule mesh for serialization
	bees::GetOrMakeSaveJob<MeshSerializer, InlineSerializationToken>(
			parameters, "beScene.MeshSerializer", queue
		).Add(mesh);

	// Schedule source meshes for serialization
	if (const AssembledMesh *sourceMesh = mesh->GetSource())
		SaveMesh(sourceMesh, parameters, queue);
	else
	{
		for (RenderableMesh::MeshRange meshes = mesh->GetMeshes(); meshes.Begin < meshes.End; ++meshes.Begin)
			if (const AssembledMesh *sourceMesh = meshes[0]->GetCompound())
				SaveMesh(sourceMesh, parameters, queue);
			else
				LEAN_LOG_ERROR_MSG("Cannot identify subset mesh compound, will be lost");
	}

	// Schedule materials for serialization
	for (RenderableMesh::MaterialRange materials = mesh->GetMaterials(); materials.Begin < materials.End; ++materials.Begin)
	{
		if (materials[0])
		{
			const beg::Material *material = materials[0]->GetMaterial();
			SaveMaterial(material, parameters, queue);

			// TODO: Special owned material config treatment?
		}
	}
}

namespace
{

/// Imports a list of meshes.
class MeshImportLoader : public beCore::LoadJob
{
public:
	/// Loads anything, e.g. to the given XML root node.
	void Load(const rapidxml::xml_node<lean::utf8_t> &root, beCore::ParameterSet &parameters, beCore::SerializationQueue<beCore::LoadJob> &queue) const
	{
		SceneParameters sceneParameters = GetSceneParameters(parameters);
		MeshCache &meshCache = *LEAN_ASSERT_NOT_NULL(sceneParameters.ResourceManager)->MeshCache();
		
		bool bNoOverwrite = beEntitySystem::GetNoOverwriteParameter(parameters);

		for (const rapidxml::xml_node<utf8_t> *meshesNode = root.first_node("meshes");
			meshesNode; meshesNode = meshesNode->next_sibling("meshes"))
			for (const rapidxml::xml_node<utf8_t> *meshNode = meshesNode->first_node();
				meshNode; meshNode = meshNode->next_sibling())
			{
				utf8_ntr name = lean::get_attribute(*meshNode, "name");
				utf8_ntr file = lean::get_attribute(*meshNode, "file");

				// Do not overwrite meshes, if not permitted
				if (!bNoOverwrite || !meshCache.GetByName(name))
				{
					lean::resource_ptr<AssembledMesh> mesh = meshCache.GetByFile(file);
					try {
						meshCache.SetName(mesh, name);
					}
					catch (const bec::ResourceCollision<AssembledMesh> &e)
					{
						LEAN_ASSERT(!bNoOverwrite);
						meshCache.Replace(e.Resource, mesh);
					}
				}
			}
	}
};


/// Loads a list of meshes.
class MeshLoader : public beCore::LoadJob
{
public:
	/// Loads anything, e.g. to the given XML root node.
	void Load(const rapidxml::xml_node<lean::utf8_t> &root, beCore::ParameterSet &parameters, beCore::SerializationQueue<beCore::LoadJob> &queue) const
	{
		SceneParameters sceneParameters = GetSceneParameters(parameters);
		RenderableMeshCache &renderableMeshCache = *LEAN_ASSERT_NOT_NULL(sceneParameters.Renderer)->RenderableMeshes();
		MeshCache &meshCache = *LEAN_ASSERT_NOT_NULL(sceneParameters.ResourceManager)->MeshCache();
		beg::MaterialCache &materialCache = *LEAN_ASSERT_NOT_NULL(sceneParameters.ResourceManager)->MaterialCache();
		RenderableMaterialCache &renderableMatCache = *LEAN_ASSERT_NOT_NULL(sceneParameters.Renderer)->RenderableMaterials();

		bool bNoOverwrite = beEntitySystem::GetNoOverwriteParameter(parameters);

		for (const rapidxml::xml_node<utf8_t> *meshesNode = root.first_node("renderablemeshes");
			meshesNode; meshesNode = meshesNode->next_sibling("renderablemeshes"))
			for (const rapidxml::xml_node<utf8_t> *meshNode = meshesNode->first_node();
				meshNode; meshNode = meshNode->next_sibling())
			{
				utf8_ntr name = lean::get_attribute(*meshNode, "name");

				// Do not overwrite meshes, if not permitted
				if (!bNoOverwrite || !renderableMeshCache.GetByName(name))
				{
					lean::resource_ptr<RenderableMesh> mesh = LoadMesh(*meshNode, meshCache, materialCache, renderableMatCache);
					try {
						renderableMeshCache.SetName(mesh, name);
					}
					catch (const bec::ResourceCollision<RenderableMesh> &e)
					{
						LEAN_ASSERT(!bNoOverwrite);
						renderableMeshCache.Replace(e.Resource, mesh);
					}
				}
			}
	}
};

} // namespace

const bec::LoadJob *CreateMeshImportLoader() { return new MeshImportLoader(); }
const bec::LoadJob *CreateMeshLoader() { return new MeshLoader(); }

} // namespace
