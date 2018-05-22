#ifndef __RENDEROBJECT__H__
#define __RENDEROBJECT__H__

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "../Render/RenderObjectKey.h"
#include "../Util/Matrix4.h"
#include "../Util/Color.h"
#include "../Util/BoundingSphere.h"
#include "../Util/AxisAlignedBoundingBox.h"

#include <GL/glut.h>

#define BUFFER_OFFSET(i) ((char *)NULL + (i))

class RenderObject
{
public:
	// type of the object
	enum ObjectType
	{
		TYPE_NONE = 0,
		TYPE_SPHERE,
		TYPE_QUADER,
		TYPE_MESH
	};
	// datastructure for a vertex
	struct Vertex
	{
		Vector2 texture;
		Vector3 tangent;
		Vector3 binormal;
		float binormalHandedness;
		Vector3 normal;
		Vector3 vertex;
	};

	RenderObject(void);
	virtual ~RenderObject(void);

	// operators
	RenderObject(const RenderObject&  src);
	RenderObject& operator = (const RenderObject& rhs);
	bool operator == (const RenderObject& rhs) const;
	bool operator < (const RenderObject& rhs) const;

	// init the render object
	virtual void Init();

	// updates the render object
	virtual void Update(float deltaTime, const Matrix4& matrix);

	// renders the object
	virtual void Render(const bool& useRenderMatrix=false);

	// destroy the render object
	virtual void Exit(void);

	// returns the number of indices
	const int GetNumIndices(void) const { return numIndices; }

	// returns the number of vertices
	const int GetNumVertices(void) const { return numVertices; }

	// returns the indices
	const unsigned int* GetIndices(void) const { return indices; }

	// returns the vertices
	const Vertex* GetVertices(void) const { return vertices; }

	// returns the id
	const int GetId(void) const { return id; }

	// returns the key
	const RenderObjectKey GetKey(void) const { return key; }

	// returns the material id
	const int GetMaterialId(void) const { return materialId; }

	// sets a matrix for the render object
	void SetMatrix(const Matrix4& mat) { world = mat; }

	// returns the bounding sphere
	const BoundingSphere GetBoundingSphere(void) const { return boundingSphere; }

	// returns the world matrix of the object
	const Matrix4& GetObjectsMatrix(void) { return renderMatrix; }

	// sets the flag for frustum culling
	static void SetFrustumCulling(bool cull) { useFrustumCulling = cull; }

	// sets the planes of the view frustum
	static void SetViewFrustum(Math::FrustumPlane planes[6]) { int i; for (i=0; i<6; i++) {frustumPlanes[i] = planes[i];} }

private:
	void RenderVertexBuffer(void);

	bool RayIntersectsTriangle(const Ray& ray,
		Vector3 vert0, Vector3 vert1, Vector3 vert2,
		float& t, float& u, float& v);

	void CalculateTangents(void);

protected:
	// type of the object
	ObjectType	 objectType;

	// explicit id
	int	id;

	// key for render object
	RenderObjectKey key;

	// id of the material to use
	int materialId;

	// number of vertices and indices
	int numVertices;
	int numIndices;

	// the vertices and indices
	Vertex* vertices;
	unsigned int* indices;

	// matrix if not used in physics world
	Matrix4 world;

	// the render matrix is updated every frame and used while the rendering process (eg. alpha blending)
	Matrix4 renderMatrix;

	// buffer objects
	unsigned int vertexBufferObjectID;
	unsigned int indexBufferObjectID;

	// flags if the object should be drawn or not
	bool cullObject;

	// the bounding sphere of the render object
	BoundingSphere boundingSphere;

	// the bounding box of the render object
	AxisAlignedBoundingBox boundingBox;

	// planes of the view frustum
	static Math::FrustumPlane frustumPlanes[6];

	static bool useFrustumCulling;

	//static int testID;
};

#endif