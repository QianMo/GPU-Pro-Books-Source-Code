#include <mesh/mesh.h>
#include <essentials/string.h>
#include <system/file.h>
#include <math/main.h>


using namespace NSystem;
using namespace NMath;


bool LoadMaterials(const string& path, vector<NMesh::Material>& materials)
{
	File file;
	string s;
	NMesh::Material material;
	bool addMaterial = false;

	if (file.Open(path, File::OpenMode::ReadText))
	{
		while (!file.EndOfFile())
		{
			file.ReadText(s);

			if (s == "newmtl")
			{
				if (addMaterial)
					materials.push_back(material);

				file.ReadLine(s);
				material.name = Trim(s);
				addMaterial = true;
			}
			else if (s == "map_Kd")
			{
				file.ReadLine(s);
				material.textureFileName = Trim(s);
			}
		}

		if (addMaterial)
			materials.push_back(material);

		file.Close();

		return true;
	}
	else
	{
		return false;
	}
}


int MaterialIndexByName(const vector<NMesh::Material>& materials, const string& name)
{
	for (uint i = 0; i < materials.size(); i++)
		if (materials[i].name == name)
			return i;

	return -1;
}


bool NMesh::ImportOBJ(const string& path, vector<Mesh>& meshes, vector<NMesh::Material>& materials)
{
	File file;
	string s;
	Mesh mesh;
	bool addMesh = false;
	vector<Vector3> vertices;
	vector<Vector3> normals;
	vector<Vector2> uvs;
	int previousLineState = -1; // 0 - vertex, 1 - normal, 2 - uv, 3 - face, 4 - smoothing group, 5 - usemtl

	mesh.nodeIndex = -1;
	mesh.materialIndex = -1;
	mesh.hasNormals = false;
	mesh.hasTangents = false;
	mesh.hasTexCoords = false;
	mesh.hasColors = false;
	mesh.vertices.clear();

	if (file.Open(path, File::OpenMode::ReadText))
	{
		while (!file.EndOfFile())
		{
			file.ReadText(s);

			if (s == "mtllib")
			{
				file.ReadLine(s);

				LoadMaterials(ExtractDir(path) + Trim(s), materials);

				previousLineState = -1;
			}
			else if (s == "v")
			{
				if (previousLineState != 0)
				{
					if (addMesh)
					{
						if (mesh.vertices.size() > 0)
							meshes.push_back(mesh);

						mesh.nodeIndex = -1;
						mesh.materialIndex = -1;
						mesh.hasNormals = false;
						mesh.hasTangents = false;
						mesh.hasTexCoords = false;
						mesh.hasColors = false;
						mesh.vertices.clear();
					}
				}

				Vector3 vertex;

				file.ReadText(vertex.x);
				file.ReadText(vertex.y);
				file.ReadText(vertex.z);

				vertices.push_back(vertex);

				addMesh = true;

				previousLineState = 0;
			}
			else if (s == "vn")
			{
				mesh.hasNormals = true;

				Vector3 normal;

				file.ReadText(normal.x);
				file.ReadText(normal.y);
				file.ReadText(normal.z);

				normals.push_back(normal);

				previousLineState = 1;
			}
			else if (s == "vt")
			{
				mesh.hasTexCoords = true;

				Vector2 uv;
				float dummy;

				file.ReadText(uv.x);
				file.ReadText(uv.y);
				file.ReadText(dummy);

				uvs.push_back(uv);

				previousLineState = 2;
			}
			else if (s == "f")
			{
				file.ReadLine(s);
				s = Trim(s);
				vector<string> verts = Split(s, ' ');

				if (verts.size() == 3)
				{
					vector<string> v1v = Split(verts[0], '/');
					vector<string> v2v = Split(verts[1], '/');
					vector<string> v3v = Split(verts[2], '/');

					int vertex1 = ToInt(v1v[0]) - 1;
					int vertex2 = ToInt(v2v[0]) - 1;
					int vertex3 = ToInt(v3v[0]) - 1;
					int normal1 = ToInt(v1v[2]) - 1;
					int normal2 = ToInt(v2v[2]) - 1;
					int normal3 = ToInt(v3v[2]) - 1;
					int uv1 = ToInt(v1v[1]) - 1;
					int uv2 = ToInt(v2v[1]) - 1;
					int uv3 = ToInt(v3v[1]) - 1;

					Vertex vertex;

					vertex.position = vertices[vertex1];
					vertex.normal = normals[normal1];
					vertex.texCoord = uvs[uv1];
					mesh.vertices.push_back(vertex);
					vertex.position = vertices[vertex2];
					vertex.normal = normals[normal2];
					vertex.texCoord = uvs[uv2];
					mesh.vertices.push_back(vertex);
					vertex.position = vertices[vertex3];
					vertex.normal = normals[normal3];
					vertex.texCoord = uvs[uv3];
					mesh.vertices.push_back(vertex);
				}
				else if (verts.size() == 4)
				{
					vector<string> v1v = Split(verts[0], '/');
					vector<string> v2v = Split(verts[1], '/');
					vector<string> v3v = Split(verts[2], '/');
					vector<string> v4v = Split(verts[3], '/');

					int vertex1 = ToInt(v1v[0]) - 1;
					int vertex2 = ToInt(v2v[0]) - 1;
					int vertex3 = ToInt(v3v[0]) - 1;
					int vertex4 = ToInt(v4v[0]) - 1;
					int normal1 = ToInt(v1v[2]) - 1;
					int normal2 = ToInt(v2v[2]) - 1;
					int normal3 = ToInt(v3v[2]) - 1;
					int normal4 = ToInt(v4v[2]) - 1;
					int uv1 = ToInt(v1v[1]) - 1;
					int uv2 = ToInt(v2v[1]) - 1;
					int uv3 = ToInt(v3v[1]) - 1;
					int uv4 = ToInt(v4v[1]) - 1;

					Vertex vertex;

					vertex.position = vertices[vertex1];
					vertex.normal = normals[normal1];
					vertex.texCoord = uvs[uv1];
					mesh.vertices.push_back(vertex);
					vertex.position = vertices[vertex2];
					vertex.normal = normals[normal2];
					vertex.texCoord = uvs[uv2];
					mesh.vertices.push_back(vertex);
					vertex.position = vertices[vertex3];
					vertex.normal = normals[normal3];
					vertex.texCoord = uvs[uv3];
					mesh.vertices.push_back(vertex);

					vertex.position = vertices[vertex1];
					vertex.normal = normals[normal1];
					vertex.texCoord = uvs[uv1];
					mesh.vertices.push_back(vertex);
					vertex.position = vertices[vertex3];
					vertex.normal = normals[normal3];
					vertex.texCoord = uvs[uv3];
					mesh.vertices.push_back(vertex);
					vertex.position = vertices[vertex4];
					vertex.normal = normals[normal4];
					vertex.texCoord = uvs[uv4];
					mesh.vertices.push_back(vertex);
				}

				previousLineState = 3;
			}
			else if (s == "s")
			{
				file.ReadLine(s);

				previousLineState = 4;
			}
			else if (s == "usemtl")
			{
				if (previousLineState == 3 || previousLineState == 4)
				{
					if (addMesh)
					{
						if (mesh.vertices.size() > 0)
							meshes.push_back(mesh);

						mesh.nodeIndex = -1;
						mesh.materialIndex = -1;
					}
				}

				file.ReadLine(s);

				mesh.materialIndex = MaterialIndexByName(materials, Trim(s));

				previousLineState = 5;
			}
			else
			{
				previousLineState = -1;
			}
		}

		if (addMesh)
			meshes.push_back(mesh);

		file.Close();

		return true;
	}
	else
	{
		return false;
	}
}
