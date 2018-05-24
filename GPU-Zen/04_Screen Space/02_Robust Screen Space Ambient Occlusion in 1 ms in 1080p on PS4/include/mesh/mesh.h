#pragma once


#include "types.h"
#include <essentials/stl.h>


namespace NMesh
{
	bool ImportASE(const string& path, vector<Mesh>& meshes);
	bool ImportOBJ(const string& path, vector<Mesh>& meshes, vector<NMesh::Material>& materials);
	void ToIndexed(const vector<Vertex>& vertices, vector<Vertex>& indexedVertices, vector<int>& indices);
	void ToIndexed(Mesh& mesh);
	void FlipVectors(vector<Vertex>& vertices, bool normals, bool tangents, bool bitangents);
	void GenerateTangents(vector<Vertex>& vertices);
	void AverageTangents(vector<Vertex>& vertices, float minTangentsPairDotProduct = 0.5f, MirrorType mirrorType = MirrorType::NoMirror); // averages if positions and normals of vertices are the same and the dot product of tangents is greater or equal to minTangentsPairDotProduct
	void OrthogonalizeTangents(vector<Vertex>& vertices);
	bool Equal(const Vertex& v1, const Vertex& v2);
}
