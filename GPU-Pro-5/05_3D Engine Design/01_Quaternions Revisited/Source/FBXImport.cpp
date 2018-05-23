/****************************************************************************

  GPU Pro 5 : Quaternions revisited - sample code
  All sample code written from scratch by Sergey Makeev specially for article.

  Copyright (c) 2013, Sergey Makeev

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software.

  2. If you use this software in a non-commercial product, an acknowledgment
     in the product documentation would be appreciated but is not required.

  3. If you use this software in a commercial product you requried to credit
     the author.

  4. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.

  5. This notice may not be removed or altered from any source distribution.
    

   Please let me know if you use the code in your products or have any questions or suggestions.

   e-mail: sergeymakeev@inbox.ru
   http://linkedin.com/in/sergeymakeev/

****************************************************************************/
#include <fbxsdk.h>
#include <vector>
#include <set>
#include "Assert.h"
#include "FBXImport.h"
#include "MeshIndexer.h"
#include "D3DMesh.h"
#include "D3DAnimation.h"
#include "Convertors.h"
#include "AdjacencyTreeBuilder.h"


template <typename T, typename LAYERTYPE>
T GetLayerData(LAYERTYPE* layer, int controlPointIndex, int vertexId)
{
	FbxLayerElement::EMappingMode mappingMode = layer->GetMappingMode();
	if (mappingMode == FbxGeometryElement::eByControlPoint)
	{
		switch (layer->GetReferenceMode())
		{
		case FbxGeometryElement::eDirect:
			return layer->GetDirectArray().GetAt(controlPointIndex);
		case FbxGeometryElement::eIndexToDirect:
			return layer->GetDirectArray().GetAt( layer->GetIndexArray().GetAt(controlPointIndex) );
		}
	} else
	{
		if (mappingMode == FbxGeometryElement::eByPolygonVertex)
		{
			switch (layer->GetReferenceMode())
			{
			case FbxGeometryElement::eDirect:
				return layer->GetDirectArray().GetAt(vertexId);
			case FbxGeometryElement::eIndexToDirect:
				return layer->GetDirectArray().GetAt( layer->GetIndexArray().GetAt(vertexId) );
			}
		}
	}
	ASSERT(false, "Unsupported layer mapping mode");
	return T();
}


template <typename T>
T GetUVLayerData(FbxMesh* mesh, int triangleIndex, int vertexIndex, FbxGeometryElementUV* layer, int controlPointIndex)
{
	FbxLayerElement::EMappingMode mappingMode = layer->GetMappingMode();
	if (mappingMode == FbxGeometryElement::eByControlPoint)
	{
		switch (layer->GetReferenceMode())
		{
		case FbxGeometryElement::eDirect:
			return layer->GetDirectArray().GetAt(controlPointIndex);
		case FbxGeometryElement::eIndexToDirect:
			return layer->GetDirectArray().GetAt( layer->GetIndexArray().GetAt(controlPointIndex) );
		}
	} else
	{
		if (mappingMode == FbxGeometryElement::eByPolygonVertex)
		{
			int textureUVIndex = mesh->GetTextureUVIndex(triangleIndex, vertexIndex);
			switch (layer->GetReferenceMode())
			{
			case FbxGeometryElement::eDirect:
			case FbxGeometryElement::eIndexToDirect:
				return layer->GetDirectArray().GetAt(textureUVIndex);
			}
		}
	}
	ASSERT(false, "Unsupported UV layer mapping mode");
	return T();
}


void CalculateQuaternionFromTBN( const Vector3& t, const Vector3& b, const Vector3& n, Quaternion* result, bool* invertedHandedness)
{
	ASSERT(invertedHandedness, "Output param 'invertedHandedness' can't be NULL");
	ASSERT(result, "Output param 'result' can't be NULL");

	*result = Quaternion::Identity();

	//Do not construct quaternions from wrong basis
	unsigned long res = Quaternion::CheckQuaternionSource(t, b, n);

	const float dotCross = dot( t, cross( b, n ) );
	float handedness = dotCross > 0.0f ? 1.0f : -1.0f;

	Matrix4x3 tbn = Matrix4x3::Identity();
	tbn.SetX(t);
	tbn.SetY(b * Vector3(handedness));
	tbn.SetZ(n);

	//Basis have scale, try to renormalize
	if ((res & Quaternion::SOURCE_BASIS_HAVE_SCALE ) != 0)
	{
		Vector3 normalizedN = n;
		normalizedN.Normalize();

		Vector3 normalizedT = t;
		normalizedT.Normalize();

		tbn.SetX(normalizedT);
		tbn.SetY(cross(normalizedT, normalizedN));
		tbn.SetZ(normalizedN);
	}

	*result = Quaternion( tbn );

	if (handedness < 0.0)
		*invertedHandedness = true;
	else
		*invertedHandedness = false;
}




FBXImporter::FBXImporter()
{
	fbxImporter = NULL;
	fbxScene = NULL;
	fbxManager = FbxManager::Create();
	ASSERT(fbxManager, "Error: Unable to create FBX Manager!");
}

FBXImporter::~FBXImporter()
{
	fbxManager->Destroy();
	fbxManager = NULL;
}

void FBXImporter::Cleanup()
{
	if (fbxScene)
	{
		fbxScene->Destroy();
		fbxScene = NULL;
	}

	if (fbxImporter)
	{
		fbxImporter->Destroy();
		fbxImporter = NULL;
	}
}


Matrix4x3 FBXImporter::GetFinalWorldTransform(FbxNode* node)
{
	FbxVector4 translation = node->GetGeometricTranslation(FbxNode::eSourcePivot);
	FbxVector4 rotation = node->GetGeometricRotation(FbxNode::eSourcePivot);
	FbxVector4 scale = node->GetGeometricScaling(FbxNode::eSourcePivot);

	FbxAMatrix geometryTransform;
	geometryTransform.SetTRS(translation, rotation, scale);

	FbxAMatrix globalTransform = node->GetScene()->GetEvaluator()->GetNodeGlobalTransform(node);

	FbxAMatrix finalTransform;
	finalTransform = globalTransform * geometryTransform;

	return Matrix4x3(finalTransform);
}


void FBXImporter::AddMesh(FbxNode* node, FbxNode* parent)
{
	FbxMesh* mesh = node->GetMesh();

	if (!mesh)
		return;

	if (!mesh->IsTriangleMesh())
		return;

	Matrix4x3 mtxWorld = GetFinalWorldTransform(node);

	mesh->RemoveBadPolygons();
	mesh->ComputeBBox();

	int trianglesCount = mesh->GetPolygonCount();

	if (sourceMesh.empty())
		sourceMesh.reserve(trianglesCount);

	int controlPointsCount = mesh->GetControlPointsCount();

	std::vector<FBXImporter::JointInfluences> meshVerticesJointsInfluences;
	meshVerticesJointsInfluences.resize(controlPointsCount);

	int skinCount = mesh->GetDeformerCount(FbxDeformer::eSkin);
	for(int skinIndex = 0; skinIndex < skinCount; skinIndex++)
	{
		FbxSkin* skin = (FbxSkin *) mesh->GetDeformer(skinIndex, FbxDeformer::eSkin);
		FbxCluster::ELinkMode clusterMode0 = skin->GetCluster(0)->GetLinkMode();

		int jointsCount = skin->GetClusterCount();
		for (int jointIndex = 0; jointIndex < jointsCount; jointIndex++)
		{
			FbxCluster* joint = skin->GetCluster(jointIndex);

			FbxCluster::ELinkMode clusterMode = joint->GetLinkMode();
			ASSERT(clusterMode == clusterMode0, "Different cluster modes in different joints?");      

			FbxAMatrix lMatrix;
			lMatrix = joint->GetTransformMatrix(lMatrix);

			int influencedCount = joint->GetControlPointIndicesCount();
			
			int* influenceIndices = joint->GetControlPointIndices();
			double* influenceWeights = joint->GetControlPointWeights();

			for (int influenceIndex = 0; influenceIndex < influencedCount; influenceIndex++)
			{
				int controlPointIndex = influenceIndices[influenceIndex];
				ASSERT(controlPointIndex < (int)meshVerticesJointsInfluences.size(), "Invalid skin control point index");
				meshVerticesJointsInfluences[controlPointIndex].influences.push_back( FBXImporter::JointInfluence(joint, (float)influenceWeights[influenceIndex]) );
			}
		}
		ASSERT((clusterMode0 == FbxCluster::eNormalize || clusterMode0 == FbxCluster::eTotalOne), "Unsupported cluster mode");
	}


	FbxVector4* meshControlPoints = mesh->GetControlPoints(); 
	for (int triangleIndex = 0; triangleIndex < trianglesCount; triangleIndex++)
	{
		int polygonVertexCount = mesh->GetPolygonSize(triangleIndex);
		ASSERT(polygonVertexCount == 3, "Invalid mesh triangulation");

		FatTriangle tri;
		for (int vertexIndex = 0; vertexIndex < 3; vertexIndex++)
		{
			FatVertex & vertex = tri.v[vertexIndex];

			vertex.quaternionTBN = Quaternion::Identity();
			vertex.invertedHandednessTBN = false;

			int controlPointIndex = mesh->GetPolygonVertex(triangleIndex, vertexIndex);
			int vertexID = triangleIndex * 3 + vertexIndex;

			const FBXImporter::JointInfluences & jointInfluences = meshVerticesJointsInfluences[controlPointIndex];
			vertex.jointInfluences = jointInfluences;

			const FbxVector4 & pos = meshControlPoints[controlPointIndex];
			vertex.pos = mtxWorld.TrasnformVertex( Vector3(pos) );

			bboxMin.Min(vertex.pos);
			bboxMax.Max(vertex.pos);

			FbxVector4 normal;
			mesh->GetPolygonVertexNormal(triangleIndex, vertexIndex, normal);
			vertex.normal = mtxWorld.TrasnformNormal( Vector3(normal) );
			vertex.normal.Normalize();

/*
			for( int normalIndex = 0; normalIndex < mesh->GetElementNormalCount(); normalIndex++ )
			{
				FbxLayerElementNormal * meshNormals = mesh->GetElementNormal( normalIndex );
				FbxVector4 normal = GetLayerData<FbxVector4>(meshNormals, controlPointIndex, vertexID);
				vertex.normal = mtxWorld.TrasnformNormal( Convert::ConvertVector3( Vector3(normal) ) );
				vertex.normal.Normalize();
			}
*/
			
			vertex.tangent = Vector3(0.0f, 0.0f, 0.0f);
			ASSERT(mesh->GetElementTangentCount() > 0, "No Tangents found in FBX file");
			for( int tangentIndex = 0; tangentIndex < mesh->GetElementTangentCount(); tangentIndex++)
			{
				FbxGeometryElementTangent* meshTangents = mesh->GetElementTangent( tangentIndex );
				FbxVector4 tangent = GetLayerData<FbxVector4>(meshTangents, controlPointIndex, vertexID);
				vertex.tangent = mtxWorld.TrasnformNormal( Vector3(tangent) );
				vertex.tangent.Normalize();
			}

			vertex.binormal = Vector3(0.0f, 0.0f, 0.0f);
			ASSERT(mesh->GetElementBinormalCount() > 0, "No Binormals found in FBX file");
			for( int binormalIndex = 0; binormalIndex < mesh->GetElementBinormalCount(); binormalIndex++)
			{
				FbxGeometryElementBinormal* meshBinormals = mesh->GetElementBinormal( binormalIndex );
				FbxVector4 binormal = GetLayerData<FbxVector4>(meshBinormals, controlPointIndex, vertexID);
				vertex.binormal = mtxWorld.TrasnformNormal( Vector3(binormal) );
				vertex.binormal.Normalize();
			}

			for (int uvIndex = 0; uvIndex < mesh->GetElementUVCount(); uvIndex++)
			{
				FbxGeometryElementUV* meshUVs = mesh->GetElementUV( uvIndex );
				FbxVector2 uv = GetUVLayerData<FbxVector2>(mesh, triangleIndex, vertexIndex, meshUVs, controlPointIndex);
				vertex.uv = Vector2(uv);
				vertex.uv.y = (-vertex.uv.y + 1.0f);
			}

			for ( int vertexColorIndex = 0; vertexColorIndex < mesh->GetElementVertexColorCount(); vertexColorIndex++)
			{
				FbxGeometryElementVertexColor* meshVertexColors = mesh->GetElementVertexColor( vertexColorIndex );
				FbxColor vertexColor = GetLayerData<FbxColor>(meshVertexColors, controlPointIndex, vertexID);
			}
		} // vertexIndex

		int triangleMaterialsCount = mesh->GetElementMaterialCount();
		ASSERT(triangleMaterialsCount <= 1, "Multiple materials per triangle is not supported");
		for( int materialIndex = 0; materialIndex < triangleMaterialsCount; materialIndex++)
		{
			FbxGeometryElementMaterial* meshMaterials = mesh->GetElementMaterial(materialIndex);
			int materialID = -1;
			if( meshMaterials->GetMappingMode() == FbxGeometryElement::eAllSame) 
			{
				materialID = meshMaterials->GetIndexArray().GetAt(0);
			} else
			{
				materialID = meshMaterials->GetIndexArray().GetAt(triangleIndex);
			}

			if (materialID >= 0)
			{
				tri.material = node->GetMaterial(materialID);
				sceneMaterials.insert( tri.material );
			}
		}


		/*
		In situations where the UV winding order of a face is reversed (that is, counter-clockwise), a straightforward left or right-handed tangent calculation results in a flipped binormal.

		URL for details
		http://download.autodesk.com/global/docs/maya2013/en_us/index.html?url=files/Polygons_nodes_Tangent_Space.htm,topicNumber=d30e172891

		*/

		Vector3 uvEdge0 = Vector3(tri.v[0].uv - tri.v[1].uv, 0.0f);
		Vector3 uvEdge1 = Vector3(tri.v[0].uv - tri.v[2].uv, 0.0f);
		Vector3 uvNormal = cross(uvEdge0, uvEdge1);
		if ( dot(uvNormal, Vector3(0.0f, 0.0f, 1.0f)) > 0.0f)
		{
			tri.v[0].binormal = -tri.v[0].binormal;
			tri.v[1].binormal = -tri.v[1].binormal;
			tri.v[2].binormal = -tri.v[2].binormal;
		}
		sourceMesh.push_back(tri);
	} // triangleIndex
}

void FBXImporter::AddJoint(FbxNode* node, FbxNode* parent)
{
	sceneJointToIndex[node] = (int)sceneJoints.size();
	sceneJoints.push_back( Joint(node) );
}

int FBXImporter::GetSkinJointIndex(const FbxCluster* jointNode) const
{
	auto it = skinJointToIndex.find(jointNode);
	if (it != skinJointToIndex.end())
	{
		return it->second;
	}
	return -1;
}

int FBXImporter::GetSceneJointIndex(const FbxNode* jointNode) const
{
	auto it = sceneJointToIndex.find(jointNode);
	if (it != sceneJointToIndex.end())
	{
		return it->second;
	}
	return -1;
}


void FBXImporter::ProcessNode(FbxNode* node, FbxNode* parent)
{
	FbxNodeAttribute* att = node->GetNodeAttribute();
	if(att != NULL)
	{
		FbxNodeAttribute::EType type = att->GetAttributeType();
		switch(type)
		{
		case FbxNodeAttribute::eMesh:
			AddMesh(node, parent);
			break;
		case FbxNodeAttribute::eSkeleton:
			AddJoint(node, parent);
			break;
		}
	}

	for(int i = 0; i < node->GetChildCount(); i++)
	{
		if (FbxNode* child = node->GetChild(i))
		{
			ProcessNode(child, node);
		}
	}
}


bool FBXImporter::CreateIndexedMesh(D3DMesh & mesh)
{
	ASSERT(!sourceMesh.empty(), "No triangles found");

	CreateJoints(mesh);
	CreateMaterials(mesh);

	Vector3 bboxExtents = bboxMax - bboxMin;

	float maxExtent = Utils::Max( Utils::Max( bboxExtents.x, bboxExtents.y ), bboxExtents.z );
	float indexerGridStep = maxExtent / (float)MeshIndexer<FatVertex>::GRID_SIZE;

	MeshIndexer<FatVertex> indexer( indexerGridStep );
	indexer.Reserve( sourceMesh.size() );

	int cullMode[3] = {0, 2, 1};

	D3DMesh::Material* materials = mesh.MapMaterials();
	int materialIndex = 0;
	for each (FbxSurfaceMaterial* material in sceneMaterials)
	{
		int minimumIndex = INT_MAX;

		materials[materialIndex].startIndex = (int)indexer.GetIndexBuffer().size();
		
		for (auto iter = sourceMesh.begin(); iter != sourceMesh.end(); ++iter)
		{
			FatTriangle & tri = *iter;

			if (tri.material != material)
				continue;

			for (int i = 0; i < 3; i++)
			{
				FatVertex & v = tri.v[ cullMode[i] ];
				int indice = indexer.AddVertex( v );

				minimumIndex = Utils::Min(minimumIndex, indice);
			}
		} // raw triangles iterator


		materials[materialIndex].trianglesCount = (indexer.GetIndexBuffer().size() - materials[materialIndex].startIndex) / 3;
		materialIndex++;

	} // materials loop

	mesh.UnmapMaterials();



	//calculate quaternions from tbn
	//-------------------------------------------------------------------------
	std::vector<MeshIndexer<FatVertex>::Vertex> & indexedVertices = indexer.GetMutableVertexBuffer();
	for (int i = 0; i < (int)indexedVertices.size(); i++)
	{
		FatVertex & vertex = indexedVertices[i].v;
		CalculateQuaternionFromTBN(vertex.tangent, vertex.binormal, vertex.normal, &vertex.quaternionTBN, &vertex.invertedHandednessTBN);
		vertex.quaternionTBN.Normalize();
	}

	//-------------------------------------------------------------------------

	AdjacencyTreeBuilder<int> adjacencyBuilder;
	adjacencyBuilder.Build(indexer.GetIndexBuffer(), (int)indexedVertices.size());

	const std::vector<AdjacencyTreeBuilder<int>::Edge> & adjacencies = adjacencyBuilder.GetAdjacencies();

	//Traverse the tree, align quaternions so their dot product becomes positive
	for ( size_t i = 0; i < adjacencies.size(); ++i )
	{
		Quaternion & child = indexedVertices[adjacencies[i].first].v.quaternionTBN;
		const Quaternion & parent = indexedVertices[adjacencies[i].second].v.quaternionTBN;

		//If the interpolation would go along the long arch, flip child
		if ( dot(child, parent ) < 0.0f )
		{
			child = -child;
		}
	}

	//-------------------------------------------------------------------------


	const std::vector<MeshIndexer<FatVertex>::Vertex> & sourceVertices = indexer.GetVertexBuffer();

	int verticesCount = (int)sourceVertices.size();
	OutputDebugStringA(Utils::StringFormat("Vertex count : %d\n", verticesCount));
	mesh.CreateVertexBuffer( verticesCount );

	D3DMesh::VertexUnpackedTBN76* verticesUnpackedTBN = (D3DMesh::VertexUnpackedTBN76*)mesh.MapVertexBuffer(D3DMesh::VFMT_UNPACKED_TBN);
	D3DMesh::VertexPackedTBN40* verticesPackedTBN = (D3DMesh::VertexPackedTBN40*)mesh.MapVertexBuffer(D3DMesh::VFMT_PACKED_TBN);
	D3DMesh::VertexUnpackedQuat44* verticesUnpackedQuat = (D3DMesh::VertexUnpackedQuat44*)mesh.MapVertexBuffer(D3DMesh::VFMT_UNPACKED_QUAT);
	D3DMesh::VertexPackedQuat32* verticesPackedQuat = (D3DMesh::VertexPackedQuat32*)mesh.MapVertexBuffer(D3DMesh::VFMT_PACKED_QUAT);

	D3DMesh::VertexNoSkinUnpackedTBN56* verticesNoSkinUnpackedTBN = (D3DMesh::VertexNoSkinUnpackedTBN56*)mesh.MapVertexBuffer(D3DMesh::VFMT_UNPACKED_TBN_NOSKIN);
	D3DMesh::VertexNoSkinPackedTBN28* verticesNoSkinPackedTBN = (D3DMesh::VertexNoSkinPackedTBN28*)mesh.MapVertexBuffer(D3DMesh::VFMT_PACKED_TBN_NOSKIN);
	D3DMesh::VertexNoSkinUnpackedQuat32* verticesNoSkinUnpackedQuat = (D3DMesh::VertexNoSkinUnpackedQuat32*)mesh.MapVertexBuffer(D3DMesh::VFMT_UNPACKED_QUAT_NOSKIN);
	D3DMesh::VertexNoSkinPackedQuat20* verticesNoSkinPackedQuat = (D3DMesh::VertexNoSkinPackedQuat20*)mesh.MapVertexBuffer(D3DMesh::VFMT_PACKED_QUAT_NOSKIN);

	JointInfluences vertexInfluences;
	for (int i = 0; i < verticesCount; i++)
	{
		const FatVertex & src = sourceVertices[i].v;

		ASSERT(src.uv.x >= -8.0f && src.uv.x <= 8.0f, "UV must be in -8..8 range");
		ASSERT(src.uv.y >= -8.0f && src.uv.y <= 8.0f, "UV must be in -8..8 range");

		D3DMesh::VertexUnpackedTBN76 & dstUnpackedTBN = verticesUnpackedTBN[i];
		D3DMesh::VertexPackedTBN40 & dstPackedTBN = verticesPackedTBN[i];
		D3DMesh::VertexUnpackedQuat44 & dstUnpackedQuat = verticesUnpackedQuat[i];
		D3DMesh::VertexPackedQuat32 & dstPackedQuat = verticesPackedQuat[i];
		D3DMesh::VertexNoSkinUnpackedTBN56 & dstNoSkinUnpackedTBN = verticesNoSkinUnpackedTBN[i];
		D3DMesh::VertexNoSkinPackedTBN28 & dstNoSkinPackedTBN = verticesNoSkinPackedTBN[i];
		D3DMesh::VertexNoSkinUnpackedQuat32 & dstNoSkinUnpackedQuat = verticesNoSkinUnpackedQuat[i];
		D3DMesh::VertexNoSkinPackedQuat20 & dstNoSkinPackedQuat = verticesNoSkinPackedQuat[i];


		dstUnpackedTBN.pos = src.pos;
		dstUnpackedTBN.normal = src.normal;
		dstUnpackedTBN.tangent = src.tangent;
		dstUnpackedTBN.binormal = src.binormal;
		dstUnpackedTBN.uv = src.uv;

		dstNoSkinUnpackedTBN.pos = dstUnpackedTBN.pos;
		dstNoSkinUnpackedTBN.normal = dstUnpackedTBN.normal;
		dstNoSkinUnpackedTBN.tangent = dstUnpackedTBN.tangent;
		dstNoSkinUnpackedTBN.binormal = dstUnpackedTBN.binormal;
		dstNoSkinUnpackedTBN.uv = dstUnpackedTBN.uv;

		dstPackedTBN.pos = src.pos;
		dstPackedTBN.packedNormal = Convert::QuantizeNormalizedVector(src.normal);
		dstPackedTBN.packedTangent = Convert::QuantizeNormalizedVector(src.tangent);
		dstPackedTBN.packedBinormal = Convert::QuantizeNormalizedVector(src.binormal);
		dstPackedTBN.uv[0] = Convert::QuantizeTexCoord(src.uv.x);
		dstPackedTBN.uv[1] = Convert::QuantizeTexCoord(src.uv.y);

		dstNoSkinPackedTBN.pos = dstPackedTBN.pos;
		dstNoSkinPackedTBN.packedNormal = dstPackedTBN.packedNormal;
		dstNoSkinPackedTBN.packedTangent = dstPackedTBN.packedTangent;
		dstNoSkinPackedTBN.packedBinormal = dstPackedTBN.packedBinormal;
		dstNoSkinPackedTBN.uv[0] = dstPackedTBN.uv[0];
		dstNoSkinPackedTBN.uv[1] = dstPackedTBN.uv[1];

		dstUnpackedQuat.pos = src.pos;
		dstUnpackedQuat.quaternionTBN[0] = src.quaternionTBN.x;
		dstUnpackedQuat.quaternionTBN[1] = src.quaternionTBN.y;
		dstUnpackedQuat.quaternionTBN[2] = src.quaternionTBN.z;
		dstUnpackedQuat.quaternionTBN[3] = ((src.invertedHandednessTBN ? 0.0f : 128.0f) + (127.0f * (src.quaternionTBN.w * 0.5f + 0.5f))) / 255.0f;
		dstUnpackedQuat.uv[0] = Convert::QuantizeTexCoord(src.uv.x);
		dstUnpackedQuat.uv[1] = Convert::QuantizeTexCoord(src.uv.y);

		dstNoSkinUnpackedQuat.pos = dstUnpackedQuat.pos;
		dstNoSkinUnpackedQuat.quaternionTBN[0] = dstUnpackedQuat.quaternionTBN[0];
		dstNoSkinUnpackedQuat.quaternionTBN[1] = dstUnpackedQuat.quaternionTBN[1];
		dstNoSkinUnpackedQuat.quaternionTBN[2] = dstUnpackedQuat.quaternionTBN[2];
		dstNoSkinUnpackedQuat.quaternionTBN[3] = dstUnpackedQuat.quaternionTBN[3];
		dstNoSkinUnpackedQuat.uv[0] = dstUnpackedQuat.uv[0];
		dstNoSkinUnpackedQuat.uv[1] = dstUnpackedQuat.uv[1];


		dstPackedQuat.pos = src.pos;
		dstPackedQuat.packedQuaternionTBN = Convert::QuantizeNormalizedQuaternionWithHandedness(src.quaternionTBN, src.invertedHandednessTBN);
		dstPackedQuat.uv[0] = Convert::QuantizeTexCoord(src.uv.x);
		dstPackedQuat.uv[1] = Convert::QuantizeTexCoord(src.uv.y);

		dstNoSkinPackedQuat.pos = dstPackedQuat.pos;
		dstNoSkinPackedQuat.packedQuaternionTBN = dstPackedQuat.packedQuaternionTBN;
		dstNoSkinPackedQuat.uv[0] = dstPackedQuat.uv[0];
		dstNoSkinPackedQuat.uv[1] = dstPackedQuat.uv[1];

		vertexInfluences = src.jointInfluences;
		vertexInfluences.LimitMaxInfluences(4);
		vertexInfluences.NormalizeWeights();

		for (int j = 0; j < 4; j++)
		{
			dstUnpackedTBN.blendWeights[j] = 0.0f;
			dstUnpackedTBN.blendIndices[j] = 0xFF;

			dstPackedTBN.blendWeights[j] = Convert::QuantizeSkinWeight(0.0f);
			dstPackedTBN.blendIndices[j] = 0xFF;

			dstUnpackedQuat.blendWeights[j] = Convert::QuantizeSkinWeight(0.0f);
			dstUnpackedQuat.blendIndices[j] = 0xFF;

			dstPackedQuat.blendWeights[j] = Convert::QuantizeSkinWeight(0.0f);
			dstPackedQuat.blendIndices[j] = 0xFF;
		}

		for (int j = 0; j < vertexInfluences.influences.size(); j++)
		{
			JointInfluence & influence = vertexInfluences.influences[j];

			int jointIndex = GetSkinJointIndex(influence.joint);
			ASSERT(jointIndex >= 0 && jointIndex <= 254, "Too many joints. No more than 254 joints supported.");

			dstUnpackedTBN.blendIndices[j] = (unsigned char)jointIndex;
			dstUnpackedTBN.blendWeights[j] = influence.weight;

			dstPackedTBN.blendIndices[j] = (unsigned char)jointIndex;
			dstPackedTBN.blendWeights[j] = Convert::QuantizeSkinWeight(influence.weight);

			dstUnpackedQuat.blendIndices[j] = (unsigned char)jointIndex;
			dstUnpackedQuat.blendWeights[j] = Convert::QuantizeSkinWeight(influence.weight);

			dstPackedQuat.blendIndices[j] = (unsigned char)jointIndex;
			dstPackedQuat.blendWeights[j] = Convert::QuantizeSkinWeight(influence.weight);
		}
	}

	mesh.UnmapVertexBuffer(D3DMesh::VFMT_UNPACKED_TBN);
	mesh.UnmapVertexBuffer(D3DMesh::VFMT_PACKED_TBN);
	mesh.UnmapVertexBuffer(D3DMesh::VFMT_UNPACKED_QUAT);
	mesh.UnmapVertexBuffer(D3DMesh::VFMT_PACKED_QUAT);
	mesh.UnmapVertexBuffer(D3DMesh::VFMT_UNPACKED_TBN_NOSKIN);
	mesh.UnmapVertexBuffer(D3DMesh::VFMT_PACKED_TBN_NOSKIN);
	mesh.UnmapVertexBuffer(D3DMesh::VFMT_UNPACKED_QUAT_NOSKIN);
	mesh.UnmapVertexBuffer(D3DMesh::VFMT_PACKED_QUAT_NOSKIN);


	const std::vector<int> & sourceIndices = indexer.GetIndexBuffer();
	int indicesCount = (int)sourceIndices.size();
	OutputDebugStringA(Utils::StringFormat("Triangles count : %d\n", indicesCount/3));

	mesh.CreateIndexBuffer(indicesCount);
	WORD* indices = mesh.MapIndexBuffer();
	for (int i = 0; i < indicesCount; i++)
	{
		ASSERT(sourceIndices[i] < 0xFFFF, "Too many indices");
		ASSERT(sourceIndices[i] < verticesCount, "Invalid indice");
		indices[i] = (WORD)sourceIndices[i];
	}
	mesh.UnmapIndexBuffer();

	return true;
}


void FBXImporter::CreateJoints(D3DMesh & mesh)
{
	std::set<FbxCluster*> skinJoints;
	for (auto iter = sourceMesh.begin(); iter != sourceMesh.end(); ++iter)
	{
		FatTriangle & tri = *iter;
		for (int i = 0; i < 3; i++)
		{
			FatVertex & v = tri.v[ i ];
			for (int influenceIndex = 0; influenceIndex < v.jointInfluences.influences.size(); influenceIndex++)
			{
				skinJoints.insert(v.jointInfluences.influences[influenceIndex].joint);
			}
		}
	}

	OutputDebugStringA(Utils::StringFormat("%d joints used for skinning, %d joints total\n\n", skinJoints.size(), sceneJoints.size()));

	ASSERT(!skinJoints.empty(), "No bones found");
	mesh.CreateBones( (int)skinJoints.size() );
	D3DMesh::SkinBone* dstBones = mesh.MapBones();

	int skinJointIndex = 0;
	for (auto it = skinJoints.begin(); it != skinJoints.end(); ++it, skinJointIndex++)
	{
		FbxCluster* skinJoint = *it;
		FbxNode* skinJointNode = skinJoint->GetLink();

		if (!skinJointNode)
			continue;

		const char* boneName = skinJointNode->GetName();

		int sceneJointIndex = GetSceneJointIndex(skinJointNode);
		ASSERT(sceneJointIndex >= 0 && sceneJointIndex < (int)sceneJoints.size(), "Invalid scene joint index");

		sceneJoints[sceneJointIndex].usedInSkinning = true;

		FbxAMatrix boneReferenceTransform;
		skinJoint->GetTransformLinkMatrix(boneReferenceTransform);

		Matrix4x3 bindPose = Matrix4x3(boneReferenceTransform);

		skinJointToIndex[skinJoint] = skinJointIndex;
		dstBones[skinJointIndex].indexInAnimation = D3DAnimation::BONE_INVALID_INDEX;
		dstBones[skinJointIndex].name = boneName;

		Quaternion invBindPoseRotate = Quaternion(bindPose).Conjugate();
		dstBones[skinJointIndex].invBindPoseRotate = invBindPoseRotate;
		dstBones[skinJointIndex].invBindPoseTranslate = invBindPoseRotate * -bindPose.GetTranslate();
	}

	mesh.UnmapBones();
}


void FBXImporter::ProcessAnimationNode(FbxNode* node, FbxNode* parent, const FbxTimeSpan &animInterval, D3DAnimation & animation)
{
	FbxNodeAttribute* att = node->GetNodeAttribute();
	if(att != NULL)
	{
		FbxNodeAttribute::EType type = att->GetAttributeType();
		if (type == FbxNodeAttribute::eSkeleton)
		{
			bool parentIsBone = false;
			if (parent)
			{
				FbxNodeAttribute* parentAtt = parent->GetNodeAttribute();
				if(parentAtt != NULL)
				{
					FbxNodeAttribute::EType parentType = parentAtt->GetAttributeType();
					if (parentType == FbxNodeAttribute::eSkeleton)
					{
						parentIsBone = true;
					}
				}
			}

			const char* parentName = parent ? parent->GetName() : "";
			const char* boneName = node->GetName();

			D3DAnimation::AnimationKey* trackData = animation.CreateBone(boneName, parentIsBone ? parentName : NULL);

			FbxLongLong startFrame = animInterval.GetStart().GetFrameCount();
			FbxLongLong stopFrame = animInterval.GetStop().GetFrameCount();

			int frameIndex = 0;
			FbxTime evalTime;
			for (FbxLongLong frame = startFrame; frame <= stopFrame; frame++, frameIndex++)
			{
				evalTime.SetFrame(frame);

				FbxAMatrix nodeTransform = node->EvaluateGlobalTransform(evalTime);
				Matrix4x3 nodeWorld = Matrix4x3(nodeTransform);

				//We don't need to blend between animations in this sample, so we can store world bones transforms as animation keys for simplicity.
				trackData[frameIndex].translate = nodeWorld.GetTranslate();
				trackData[frameIndex].rotate = Quaternion(nodeWorld);
			}
		}
	}

	for(int i = 0; i < node->GetChildCount(); i++)
	{
		if (FbxNode* child = node->GetChild(i))
		{
			ProcessAnimationNode(child, node, animInterval, animation);
		}
	}
}


void FBXImporter::CreateAnimation(D3DAnimation & animation)
{
	FbxTimeSpan animInterval;
	fbxScene->GetRootNode()->GetAnimationInterval(animInterval);
	
	FbxLongLong startFrame = animInterval.GetStart().GetFrameCount();
	FbxLongLong stopFrame = animInterval.GetStop().GetFrameCount();
	OutputDebugStringA(Utils::StringFormat("animation: %I64d <->  %I64d\n", startFrame, stopFrame));

	ASSERT(stopFrame >= startFrame, "Invalid animation range");

	FbxTime zeroTime;
	zeroTime.SetFrame(0);

	if (startFrame < 0)
		animInterval.SetStart(zeroTime);

	if (stopFrame < 0)
		animInterval.SetStop(zeroTime);

	FbxLongLong duration = animInterval.GetDuration().GetFrameCount();
	animation.SetDuration( (int)(duration+1) );

	FbxGlobalSettings& globalSettings = fbxScene->GetGlobalSettings();
	FbxTime::EMode timeMode = globalSettings.GetTimeMode(); 
	float fFrameRate = (float)FbxTime::GetFrameRate( timeMode );
	animation.SetFrameRate(fFrameRate);

	FbxNode* rootNode = fbxScene->GetRootNode();
	ProcessAnimationNode(rootNode, NULL, animInterval, animation);

	int bonesCount = animation.GetBonesCount();

	//Build root bones list
	std::vector<int> rootBones;
	rootBones.reserve(16);
	for (int i = 0; i < bonesCount; i++)
	{
		int boneParentID = animation.GetBoneParentIndex(i);
		if (boneParentID == D3DAnimation::BONE_INVALID_INDEX)
		{
			rootBones.push_back(i);
		}
	}

	//Already processed bones list
	std::vector<Bool> visitedBones;
	visitedBones.resize(bonesCount, False);
	visitedBones[rootBones[0]] = True;

	//Align root bones
	int frameCount = animation.GetDuration();
	for (int i = 1; i < ((int)rootBones.size()-1); i++)
	{
		int parentBoneIndex = rootBones[i-1];
		int boneIndex = rootBones[i];

		const D3DAnimation::AnimationKey* parentKeys = animation.GetBoneAnimationTrack(parentBoneIndex);
		D3DAnimation::AnimationKey* childKeys = animation.GetBoneAnimationTrack(boneIndex);

		visitedBones[boneIndex] = True;
		for (int j = 0; j < frameCount; j++)
		{
			const Quaternion & parent = parentKeys[j].rotate;
			Quaternion &child = childKeys[j].rotate;

			//If the interpolation would go along the long arch, flip child
			if ( dot(child, parent ) < 0.0f )
			{
				child = -child;
			}
		}
	}


	//Align child bones
	for (int boneIndex = 0; boneIndex < (int)visitedBones.size(); boneIndex++)
	{
		if (visitedBones[boneIndex] == True)
		{
			continue;
		}

		int parentBoneIndex = animation.GetBoneParentIndex(boneIndex);
		ASSERT(visitedBones[parentBoneIndex] == True, "Bones should be stored in the order, from parents to children");
		
		const D3DAnimation::AnimationKey* parentKeys = animation.GetBoneAnimationTrack(parentBoneIndex);
		D3DAnimation::AnimationKey* childKeys = animation.GetBoneAnimationTrack(boneIndex);

		visitedBones[boneIndex] = True;
		for (int j = 0; j < frameCount; j++)
		{
			const Quaternion & parent = parentKeys[j].rotate;
			Quaternion &child = childKeys[j].rotate;

			//If the interpolation would go along the long arch, flip child
			if ( dot(child, parent ) < 0.0f )
			{
				child = -child;
			}
		}

		//restart loop
		boneIndex = 0;
	}
}


void FBXImporter::CreateMaterials(D3DMesh & mesh)
{
	int materialsCount = (int)sceneMaterials.size();
	ASSERT(materialsCount > 0, "No materials found");

	char textureFileName[MAX_PATH] = { '\0' };

	mesh.CreateMaterials(materialsCount);
	D3DMesh::Material* materials = mesh.MapMaterials();

	int materialIndex = 0;
	for each (const FbxSurfaceMaterial* material in sceneMaterials)
	{
		const char* materialName = material->GetName();

		OutputDebugStringA(Utils::StringFormat("Material '%s'\n", materialName));

		FbxProperty diffuseTextures = material->FindProperty(FbxSurfaceMaterial::sDiffuse);
		int texturesCount = diffuseTextures.GetSrcObjectCount<FbxFileTexture>();
		for(int textureIndex = 0; textureIndex < texturesCount; textureIndex++)
		{
			FbxFileTexture* pTexture = diffuseTextures.GetSrcObject<FbxFileTexture>(textureIndex);
			if(pTexture)
			{
				const char* textureName = pTexture->GetName();
				const char* fileName = pTexture->GetFileName();
				_splitpath_s(fileName, 0, 0, 0, 0, &textureFileName[0], MAX_PATH, 0, 0);

				OutputDebugStringA(Utils::StringFormat("Texture name '%s', File name '%s', Name only: '%s'\n", textureName, fileName, textureFileName));

				materials[materialIndex].fileName = textureFileName;
				materials[materialIndex].texAlbedoIndex = -1;
				materials[materialIndex].texNormalsIndex = -1;
				materials[materialIndex].trianglesCount = 0;
				materials[materialIndex].startIndex = 0;

			}
		}

		materialIndex++;
	}

	mesh.UnmapMaterials();
}

bool FBXImporter::Import(const char* fbxFileName, D3DMesh* mesh, D3DAnimation* animation)
{
	ASSERT(mesh, "You must specify destination mesh");
	ASSERT(animation, "You must specify destination animation");

	bboxMin = Vector3(FLT_MAX);
	bboxMax = Vector3(-FLT_MAX);

	sourceMesh.clear();
	sceneMaterials.clear();
	sceneJoints.clear();
	sceneJointToIndex.clear();
	skinJointToIndex.clear();

	fbxScene = FbxScene::Create(fbxManager, "");
	if( !fbxScene )
	{
		OutputDebugStringA("Error: Unable to create FBX Scene!\n");
		return false;
	}

	fbxImporter = FbxImporter::Create(fbxManager,"");
	if( !fbxImporter )
	{
		Cleanup();
		OutputDebugStringA("Error: Unable to create FBX Importer!\n");
		return false;
	}

	bool importStatus = fbxImporter->Initialize(fbxFileName, -1, fbxManager->GetIOSettings());
	if (!importStatus)
	{
		Cleanup();
		OutputDebugStringA("Error: Can't open FBX file!\n");
		return false;
	}

	importStatus = fbxImporter->Import(fbxScene);
	if (!importStatus)
	{
		Cleanup();
		OutputDebugStringA("Error: Can't open FBX file!\n");
		return false;
	}

	//
	//This code is not work, due to left/right handed axis system conversion issues in FBX SDK
	//
	//FbxAxisSystem::DirectX.ConvertScene(fbxScene);

	FbxAxisSystem SceneAxisSystem = fbxScene->GetGlobalSettings().GetAxisSystem();
	FbxAxisSystem OurAxisSystem(FbxAxisSystem::eYAxis, FbxAxisSystem::eParityOdd, FbxAxisSystem::eRightHanded);
	if ( SceneAxisSystem != OurAxisSystem )
	{
		OurAxisSystem.ConvertScene( fbxScene );
	}

	FbxSystemUnit SceneSystemUnit = fbxScene->GetGlobalSettings().GetSystemUnit();
	if ( fabs(SceneSystemUnit.GetScaleFactor() - 1.0) > 0.00001 )
	{
		FbxSystemUnit OurSystemUnit(1.0);
		OurSystemUnit.ConvertScene( fbxScene );
	}

	FbxGeometryConverter geometryConverter(fbxManager);
	geometryConverter.Triangulate(fbxScene, true);

	CreateAnimation(*animation);

	FbxNode* rootNode = fbxScene->GetRootNode();
	ProcessNode(rootNode, NULL);
	CreateIndexedMesh(*mesh);

	Cleanup();

	mesh->SetAnimation(animation);
	mesh->CreateTextures();

	return true;
}

