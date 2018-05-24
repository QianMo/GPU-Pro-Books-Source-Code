#pragma once


#include <essentials/stl.h>
#include <math/types.h>


namespace NMesh
{
	struct Vertex
	{
		NMath::Vector3 position;
		NMath::Vector3 normal;
		NMath::Vector3 tangent;
		NMath::Vector3 bitangent;
		NMath::Vector2 texCoord;
		NMath::Vector3 color;
	};

	struct Material
	{
		string name;
		string textureFileName;
	};

	struct Node
	{
		Node* parent;
		string name;

		NMath::Vector3 geometricTranslation;
		NMath::Vector3 geometricRotation;
		NMath::Vector3 geometricScaling;

		vector<NMath::Vector3> localTranslations;
		vector<NMath::Vector3> localRotations;
		vector<NMath::Vector3> localScales;
		vector<NMath::Matrix> localTransforms;
		vector<NMath::Matrix> globalTransforms;
	};

	struct Mesh
	{
		int nodeIndex;
		int materialIndex;

		bool hasNormals;
		bool hasTangents;
		bool hasTexCoords;
		bool hasColors;

		vector<Vertex> vertices;
		vector<int> indices;
	};

	enum class MirrorType { NoMirror, AxisX, AxisY, AxisZ };
}
