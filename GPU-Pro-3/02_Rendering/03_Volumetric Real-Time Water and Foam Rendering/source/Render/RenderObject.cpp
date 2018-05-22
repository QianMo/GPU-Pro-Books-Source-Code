#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <windows.h>
#include "GL/glew.h"
#include <GL/gl.h>

#include <stdio.h>
#include <assert.h>

#include "../Main/DemoManager.h"

#include "../Render/RenderObject.h"
#include "../Render/MaterialManager.h"
#include "../Render/RenderManager.h"
#include "../Render/ShaderManager.h"

#include "../Util/Matrix4.h"
#include "../Util/Ray.h"
#include "../Level/Light.h"

#include "../Physic/Physic.h"

#include <GL/glut.h>

#define EPSILON 0.000001

bool RenderObject::useFrustumCulling = true;
Math::FrustumPlane RenderObject::frustumPlanes[6];

//int RenderObject::testID = -1;

RenderObject::RenderObject(void):
	objectType(TYPE_NONE),
	id(-1),
	materialId(-1),
	numVertices(0),
	numIndices(0),
	vertices(NULL),
	indices(NULL),
	cullObject(false),
	vertexBufferObjectID(0),
	indexBufferObjectID(0)
{
	world = Matrix4::IDENTITY;
}

// -----------------------------------------------------------------------------
// ----------------------- RenderObject::~RenderObject --------------------------
// -----------------------------------------------------------------------------
RenderObject::~RenderObject()
{
	Exit();
}

// -----------------------------------------------------------------------------
// ----------------------- RenderObject::RenderObject --------------------------
// -----------------------------------------------------------------------------
RenderObject::RenderObject(const RenderObject&  src)
{
}

// -----------------------------------------------------------------------------
// ----------------------- RenderObject::operator = ----------------------------
// -----------------------------------------------------------------------------
RenderObject& RenderObject::operator = (const RenderObject& rhs)
{
	return *this;
}

// -----------------------------------------------------------------------------
// ----------------------- RenderObject::operator == ---------------------------
// -----------------------------------------------------------------------------
bool RenderObject::operator == (const RenderObject& rhs) const
{
	if (this->GetKey().GetIntKey() == rhs.GetKey().GetIntKey())
		return true;
	else
		return false;
}

// -----------------------------------------------------------------------------
// ----------------------- RenderObject::operator < ----------------------------
// -----------------------------------------------------------------------------
bool RenderObject::operator < (const RenderObject& rhs) const
{
	if (this->GetKey().GetIntKey() < rhs.GetKey().GetIntKey())
		return true;
	else
		return false;
}

// -----------------------------------------------------------------------------
// ----------------------- RenderObject::Init ----------------------------------
// -----------------------------------------------------------------------------
void RenderObject::Init()
{
	assert(vertices != NULL);

	RenderObjectKey::KeyData data;
	data.useParallaxMapping = MaterialManager::Instance()->GetMaterial(materialId).useParallaxMapping;
	data.materialId = materialId;
	key.Init(data);

	boundingSphere.AddVertices((BoundingSphere::Vertex*)vertices, numVertices);
	boundingBox.AddVertices((AxisAlignedBoundingBox::Vertex*)vertices, numVertices);

	if (objectType == TYPE_MESH)
	{
		CalculateTangents();
	}

	int vertexSize = sizeof(Vertex) * numVertices;
	int fillSize = 0;

	{
		glGenBuffersARB(1, &vertexBufferObjectID);
		glBindBufferARB(GL_ARRAY_BUFFER_ARB, vertexBufferObjectID);
		glBufferDataARB(GL_ARRAY_BUFFER_ARB, vertexSize, vertices, GL_STATIC_DRAW_ARB);

		glGetBufferParameterivARB(GL_ARRAY_BUFFER_ARB, GL_BUFFER_SIZE_ARB, &fillSize);

		assert(fillSize > 0);

		glBindBufferARB(GL_ARRAY_BUFFER_ARB, NULL);
	}

	if (indices)
	{
		int indexSize = sizeof(unsigned int) * numIndices;
		fillSize = 0;

		glGenBuffersARB(1, &indexBufferObjectID);
		glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB, indexBufferObjectID);
		glBufferDataARB(GL_ELEMENT_ARRAY_BUFFER_ARB, indexSize, indices, GL_STATIC_DRAW_ARB);

		glGetBufferParameterivARB(GL_ELEMENT_ARRAY_BUFFER_ARB, GL_BUFFER_SIZE_ARB, &fillSize);

		assert(fillSize > 0);

		glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB, NULL);
	}
}

// -----------------------------------------------------------------------------
// ----------------------- RenderObject::Update --------------------------------
// -----------------------------------------------------------------------------
void RenderObject::Update(float deltaTime, const Matrix4& matrix)
{
	renderMatrix = world * matrix;
	
	// view frustum culling
	if (boundingSphere.Intersects(frustumPlanes, renderMatrix) || !useFrustumCulling)
	{
		cullObject = false;
	}
	else
	{
		cullObject = true;
	}
}

// -----------------------------------------------------------------------------
// ----------------------- RenderObject::Render --------------------------------
// -----------------------------------------------------------------------------
void RenderObject::Render(const bool& useRenderMatrix)
{
	if (cullObject && useFrustumCulling)
		return;

	Matrix4 currentMatrix;

	glPushMatrix();
	{
		if (useRenderMatrix)
			glMultMatrixf(renderMatrix.entry);
		else
			glMultMatrixf(world.entry);

		// cg shader stuff
		if (DemoManager::Instance()->GetCurrentRenderMode() == DemoManager::RENDER_SCENE)
		{
			Vector3 eyePosition = DemoManager::Instance()->GetCamera()->GetCameraPosition();
			Vector3 lightPosition = DemoManager::Instance()->GetLight()->GetLightPosition();
			Matrix4 invModelMatrix = renderMatrix.Inverse();

			eyePosition = invModelMatrix * eyePosition;
			lightPosition = invModelMatrix * lightPosition;

			ShaderManager::Instance()->SetParameter3fv(ShaderManager::SP_EYE_POS_OBJ_SPACE, eyePosition.comp);
			ShaderManager::Instance()->SetParameter3fv(ShaderManager::SP_LIGHT_POS_OBJ_SPACE, lightPosition.comp);
		}

		RenderVertexBuffer();
	}

	glPopMatrix();
}

// -----------------------------------------------------------------------------
// ----------------------- RenderObject::Exit ----------------------------------
// -----------------------------------------------------------------------------
void RenderObject::Exit(void)
{
	if (vertexBufferObjectID > 0)
		glDeleteBuffersARB(1, &vertexBufferObjectID);
	if (indexBufferObjectID > 0)
		glDeleteBuffersARB(1, &indexBufferObjectID);

	if (vertices)
	{
		delete[] vertices;
		vertices = NULL;
	}

	if (indices)
	{
		delete[] indices;
		indices = NULL;
	}
}

// -----------------------------------------------------------------------------
// ----------------------- RenderObject::RenderVertexBuffer --------------------
// -----------------------------------------------------------------------------
void RenderObject::RenderVertexBuffer(void)
{
	unsigned int stride = sizeof(Vertex);
	unsigned int offsetForTangents = sizeof(GLfloat) * 2;
	unsigned int offsetForBinormals = sizeof(GLfloat) * 5;
	unsigned int offsetForNormals = sizeof(GLfloat) * 9;
	unsigned int offsetForVertices = sizeof(GLfloat) * 12;

	glBindBufferARB(GL_ARRAY_BUFFER_ARB, vertexBufferObjectID);

	glActiveTextureARB(GL_TEXTURE0);
	glClientActiveTextureARB(GL_TEXTURE0);
	glEnableClientState(GL_TEXTURE_COORD_ARRAY);
	glTexCoordPointer(2, GL_FLOAT, stride, BUFFER_OFFSET(0));

	glActiveTextureARB(GL_TEXTURE1);
	glClientActiveTextureARB(GL_TEXTURE1);
	glEnableClientState(GL_TEXTURE_COORD_ARRAY);
	glTexCoordPointer(3, GL_FLOAT, stride, BUFFER_OFFSET(offsetForTangents));

	glActiveTextureARB(GL_TEXTURE2);
	glClientActiveTextureARB(GL_TEXTURE2);
	glEnableClientState(GL_TEXTURE_COORD_ARRAY);
	glTexCoordPointer(4, GL_FLOAT, stride, BUFFER_OFFSET(offsetForBinormals));

	glEnableClientState(GL_NORMAL_ARRAY);
	glNormalPointer(GL_FLOAT, stride, BUFFER_OFFSET(offsetForNormals));

	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(3, GL_FLOAT, stride, BUFFER_OFFSET(offsetForVertices));

	if (indices)
	{
		glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB, indexBufferObjectID);
	}

	if (objectType == TYPE_SPHERE)
	{
		glDrawArrays(GL_TRIANGLE_STRIP, 0, numVertices);
	}
	else if (objectType == TYPE_QUADER)
	{
		unsigned int i;
		for (i = 0; i < 6; i++)
		{
			glDrawArrays(GL_TRIANGLE_STRIP, i * 4, numVertices / 6);
		}
	}
	else if (objectType == TYPE_MESH)
	{
		glDrawElements(GL_TRIANGLES, numIndices, GL_UNSIGNED_INT, 0);
	}
	else
	{
		assert(false);
	}

	glDisableClientState(GL_TEXTURE_COORD_ARRAY);
	glDisableClientState(GL_NORMAL_ARRAY);
	
	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_INDEX_ARRAY);

	glBindBufferARB(GL_ARRAY_BUFFER_ARB, NULL);
	glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB, NULL);
}

// -----------------------------------------------------------------------------
// ------------------ RenderObject::RayIntersectsTriangle ----------------------
// -----------------------------------------------------------------------------
bool RenderObject::RayIntersectsTriangle(const Ray& ray,
				   Vector3 vert0, Vector3 vert1, Vector3 vert2,
				   float& t, float& u, float& v)
{
	Vector3 edge1, edge2, tvec, pvec, qvec;
	float det, inv_det;

	/* find vectors for two edges sharing vert0 */
	edge1 = vert1-vert0;
	edge2 = vert2-vert0;

	/* begin calculating determinant - also used to calculate U parameter */
	pvec = ray.GetDirection().CrossProduct(edge2);

	/* if determinant is near zero, ray lies in plane of triangle */
	det = edge1.DotProduct(pvec);

/*
	if (det < EPSILON)
		return 0;

	// calculate distance from vert0 to ray origin
	SUB(tvec, orig, vert0);

	// calculate U parameter and test bounds
	*u = DOT(tvec, pvec);
	if (*u < 0.0 || *u > det)
		return 0;

	// prepare to test V parameter
	CROSS(qvec, tvec, edge1);

	// calculate V parameter and test bounds
	*v = DOT(dir, qvec);
	if (*v < 0.0 || *u + *v > det)
		return 0;

	// calculate t, scale parameters, ray intersects triangle
	*t = DOT(edge2, qvec);
	inv_det = 1.0 / det;
	*t *= inv_det;
	*u *= inv_det;
	*v *= inv_det;
*/
	if (det > -EPSILON && det < EPSILON)
		return 0;
	inv_det = 1.0 / det;

	// calculate distance from vert0 to ray origin
	tvec = ray.GetOrigin()-vert0;

	// calculate U parameter and test bounds
	u = tvec.DotProduct(pvec) * inv_det;
	if (u < 0.0 || u > 1.0)
		return 0;

	// prepare to test V parameter
	qvec = tvec.CrossProduct(edge1);

	// calculate V parameter and test bounds
	v = ray.GetDirection().DotProduct(qvec) * inv_det;
	if (v < 0.0 || u + v > 1.0)
		return 0;

	// calculate t, ray intersects triangle
	t = edge2.DotProduct(qvec) * inv_det;

	return 1;
}

// -----------------------------------------------------------------------------
// ------------------ RenderObject::CalculateTangents --------------------------
// -----------------------------------------------------------------------------
void RenderObject::CalculateTangents(void)
{
	Vector3* tan1 = new Vector3[numVertices * 2];
	Vector3* tan2 = tan1 + numVertices;
	ZeroMemory(tan1, numVertices * sizeof(Vector3) * 2);

	int i;
	for (i=0; i<numIndices; i+=3)
	{
		int i1 = indices[i];
		int i2 = indices[i+1];
		int i3 = indices[i+2];

		const Vector3& v1 = vertices[i1].vertex;
		const Vector3& v2 = vertices[i2].vertex;
		const Vector3& v3 = vertices[i3].vertex;

		const Vector2& w1 = vertices[i1].texture;
		const Vector2& w2 = vertices[i2].texture;
		const Vector2& w3 = vertices[i3].texture;

		float x1 = v2.x - v1.x;
		float x2 = v3.x - v1.x;
		float y1 = v2.y - v1.y;
		float y2 = v3.y - v1.y;
		float z1 = v2.z - v1.z;
		float z2 = v3.z - v1.z;

		float s1 = w2.x - w1.x;
		float s2 = w3.x - w1.x;
		float t1 = w2.y - w1.y;
		float t2 = w3.y - w1.y;

		float r = 1.0f / (s1 * t2 - s2 * t1);
		Vector3 sdir((t2 * x1 - t1 * x2) * r, (t2 * y1 - t1 * y2) * r,
			(t2 * z1 - t1 * z2) * r);
		Vector3 tdir((s1 * x2 - s2 * x1) * r, (s1 * y2 - s2 * y1) * r,
		(s1 * z2 - s2 * z1) * r);

		tan1[i1] += sdir;
		tan1[i2] += sdir;
		tan1[i3] += sdir;

		tan2[i1] += tdir;
		tan2[i2] += tdir;
		tan2[i3] += tdir;
	}

	for (i=0; i<numVertices; i++)
	{
		const Vector3& n = vertices[i].normal;
		Vector3& t = tan1[i];

		// Calculate handedness
		float handedness = (n.CrossProduct(t).DotProduct(tan2[i]) < 0.0f) ? -1.0f : 1.0f;

		// Gram-Schmidt orthogonalize
		Vector3 tangent = t - n * n.DotProduct(t);
		tangent.Normalize();
		vertices[i].tangent = tangent;

		Vector3 binormal = tangent.CrossProduct(vertices[i].normal);
		vertices[i].binormal = binormal;
		vertices[i].binormalHandedness = handedness;
	}

	delete[] tan1;
}