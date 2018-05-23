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
#pragma once

#include "Math.h"
#include <hash_map>

class D3DMesh;
class D3DAnimation;

class FBXImporter
{
	//////////////////////////////////////////////////////////////////////////
	struct JointInfluence
	{
		FbxCluster* joint;
		float weight;

		JointInfluence(FbxCluster* _joint, float _weight)
		{
			joint = _joint;
			weight = _weight;
		}
	};

	//////////////////////////////////////////////////////////////////////////
	struct JointInfluences
	{
		Utils::FixedArray<JointInfluence, 8> influences;

		void NormalizeWeights()
		{
			float totalWeight = 0.0f;
			for (int i = 0; i < influences.size(); i++)
			{
				totalWeight += influences[i].weight;
			}

			float invTotalWeight = 1.0f / totalWeight;
			for (int i = 0; i < influences.size(); i++)
			{
				influences[i].weight *= invTotalWeight;
			}
		}

		bool LimitMaxInfluences(int maxInfluenceCount)
		{
			int influencesCount = influences.size();

			if (influencesCount <= 1)
				return false;

			//simple descending bubble sort...
			for (int i = 0; i < (influencesCount - 1); i++)
			{
				// ...optimized for branch predictor
				// http://blog.gamedeff.com/?p=268
				//
				for (int j = i; j >= 0; j--)
				{
					if (influences[j].weight < influences[j + 1].weight)
					{
						Utils::Swap(influences[j], influences[j + 1]);
					}
				}
			}

			influences.trim(maxInfluenceCount);
			return true;
		}


	};

	//////////////////////////////////////////////////////////////////////////
	struct FatVertex
	{
		Vector3 pos;
		Vector3 normal;
		Vector3 tangent;
		Vector3 binormal;

		Quaternion quaternionTBN;
		bool invertedHandednessTBN;

		Vector2 uv;
		JointInfluences jointInfluences;

		bool CanWeld(const FatVertex & v) const
		{
			const float weldThreshold = 0.00001f;

			if (fabs(pos.x - v.pos.x) > weldThreshold)
				return false;

			if (fabs(pos.y - v.pos.y) > weldThreshold)
				return false;

			if (fabs(pos.z - v.pos.z) > weldThreshold)
				return false;

			const float normalWeldThreshold = 0.001f;

			if (fabs(normal.x - v.normal.x) > normalWeldThreshold)
				return false;

			if (fabs(normal.y - v.normal.y) > normalWeldThreshold)
				return false;

			if (fabs(normal.z - v.normal.z) > normalWeldThreshold)
				return false;

			if (fabs(tangent.x - v.tangent.x) > normalWeldThreshold)
				return false;

			if (fabs(tangent.y - v.tangent.y) > normalWeldThreshold)
				return false;

			if (fabs(tangent.z - v.tangent.z) > normalWeldThreshold)
				return false;

			if (fabs(binormal.x - v.binormal.x) > normalWeldThreshold)
				return false;

			if (fabs(binormal.y - v.binormal.y) > normalWeldThreshold)
				return false;

			if (fabs(binormal.z - v.binormal.z) > normalWeldThreshold)
				return false;

			const float uvWeldThreshold = (1.0f / 8192.0f);

			if (fabs(uv.x - v.uv.x) > uvWeldThreshold)
				return false;

			if (fabs(uv.y - v.uv.y) > uvWeldThreshold)
				return false;

			if (jointInfluences.influences.size() != v.jointInfluences.influences.size())
				return false;

			const float jointWieghtThreshold = 0.0001f;

			for (int i = 0; i < jointInfluences.influences.size(); i++)
			{
				if (jointInfluences.influences[i].joint != v.jointInfluences.influences[i].joint)
					return false;

				if (fabs(jointInfluences.influences[i].weight - v.jointInfluences.influences[i].weight) > jointWieghtThreshold)
					return false;
			}

			return true;
		}
	};

	//////////////////////////////////////////////////////////////////////////
	struct FatTriangle
	{
		FatVertex v[3];
		FbxSurfaceMaterial *material;

		FatTriangle()
		{
			material = NULL;
		}
	};


	//////////////////////////////////////////////////////////////////////////
	struct Joint
	{
		FbxNode* joint;
		bool usedInSkinning;

		Joint(FbxNode* _joint)
		{
			joint = _joint;
			usedInSkinning = false;
		}
	};

	FbxManager* fbxManager;
	FbxScene* fbxScene;
	FbxImporter* fbxImporter;

	Vector3 bboxMin;
	Vector3 bboxMax;

	std::vector<FatTriangle> sourceMesh;
	std::set<FbxSurfaceMaterial*> sceneMaterials;

	std::vector<Joint> sceneJoints;
	stdext::hash_map<const FbxNode*, int> sceneJointToIndex;

	stdext::hash_map<const FbxCluster*, int> skinJointToIndex;

	int GetSceneJointIndex(const FbxNode* jointNode) const;
	int GetSkinJointIndex(const FbxCluster* jointNode) const;

	void Cleanup();

	Matrix4x3 GetFinalWorldTransform(FbxNode* node);
	
	void AddMesh(FbxNode* node, FbxNode* parent);
	void AddJoint(FbxNode* node, FbxNode* parent);

	void ProcessNode(FbxNode* node, FbxNode* parent);

	bool CreateIndexedMesh(D3DMesh & mesh);
	void CreateJoints(D3DMesh & mesh);
	void CreateMaterials(D3DMesh & mesh);


	void ProcessAnimationNode(FbxNode* node, FbxNode* parent, const FbxTimeSpan &animInterval, D3DAnimation & animation);
	void CreateAnimation(D3DAnimation & animation);


public:

	FBXImporter();
	~FBXImporter();

	bool Import(const char* fbxFileName, D3DMesh* mesh, D3DAnimation* animation);

};

