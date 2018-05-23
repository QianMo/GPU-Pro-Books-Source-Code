
#ifndef __MESH_IMPORT_HPP__
#define __MESH_IMPORT_HPP__

class StaticBone;

#include <Common\Incopiable.hpp>
#include <Math/Vector/Vector.hpp>
#include <Math/Matrix/Matrix.hpp>

///////////////////////////////////////////////////////////////////////////
//// Mesh import takes all the data possible, and application selects from the data buffer which is necessary(pos, uv, ...)
///////////////////////////////////////////////////////////////////////////

class AnimationImport : public Incopiable
{
public:

	///< Import bone:
	struct Bone
	{
		Bone(){m_csName=NULL;}
		char* m_csName;

		~Bone(){ M::DeleteArray(&m_csName); }

		int32			m_index;
		DVector<Bone>	m_childs;
		Matrix4f		m_R, m_T;
	};

	///<
	int32 FindBoneIndex(const char* _csName)
	{
		for (uint32 i=0; i<m_boneArray.Size();++i)
		{
			if (m_boneArray[i])
			{
				if (strcmp(m_boneArray[i]->m_csName,_csName)==0)
					return i;
			}
		}

		ASSERT(false, "Bone not found!");
		return -1;
	}

	Bone m_root;

	///< Built at final step.
	DVector<Bone*> m_boneArray;

	~AnimationImport(){
	}
};


///<
class MeshImport : public Incopiable
{
public:

	void Release();

	struct VertexData
	{
		Vector3f	_x;
		Vector3f	_n;
		Vector2f	_uv;
		Vector4f	_c;

		Vector4i	_boneIndices;
		Vector4f	_weights;

		VertexData():_boneIndices(-1){}

	};

	///<
	struct SkinData
	{
		SkinData():_boneIndices(-1){}

		Vector8i	_boneIndices;
		Vector8f	_weights;

	};

	bool m_bHasNormals, m_bHasUVs, m_bHasColors, m_bGeometry, m_bHasBones, m_bHasSkinWeights;	

	AnimationImport*			m_pAnimationImport;
	SkinData*					m_pImportedSkinData;

	VertexData*					m_pVertices;

	char*						m_pTextureFileName;
	int32						m_NumVertices;

	int32						m_NumBones;
	StaticBone*					m_pBones;

	void SetTextureFileName		(const char* _csFilename);

	MeshImport() : m_pVertices(NULL),m_pTextureFileName(NULL), m_NumVertices(0), m_pAnimationImport(NULL), m_pImportedSkinData(NULL), m_pBones(NULL), m_NumBones(0),
		m_bHasNormals(false), m_bHasUVs(false), m_bHasColors(false), m_bGeometry(false), m_bHasBones(false), m_bHasSkinWeights(false)
	{
		m_pAnimationImport=new AnimationImport();
	}

	~MeshImport() { Release(); }

	typedef DVector< Vector<Vector3f, 2> >	LineVector;
	typedef DVector< Vector3f >				PointVector;

	///<
	void	CreateSurface	(const Vector2ui _dims);
	///<
	void	Import					(const char* _csFileName);


};

#endif