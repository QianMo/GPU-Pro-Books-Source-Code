
#ifndef __MESH_HPP__
#define __MESH_HPP__

#include <Common/Common.hpp>
#include <Math/Vector/Vector.hpp>
#include <Math/Matrix/Matrix.hpp>

#include <d3dx11.h>
#include <d3d11.h>

class MeshImport;

class Mesh 
{
protected:
	uint32							m_Stride;
	uint32							m_NumVertices;

	ID3D11Buffer*					m_pConstants;
	ID3D11Buffer*					m_pVertexBuffer;
	ID3D11ShaderResourceView*		m_pTextureRV;

	void			Release		();

public:

	Matrix4f						m_World;
	///<
	Mesh(){memset(this,0,sizeof(Mesh));}	

	~Mesh();	
	
	void			SetShaderRV	(int32 _index, ID3D11ShaderResourceView*);


	void			Draw		(ID3D11DeviceContext* _pImmediateContext, const int32 _iSize=0);
		
	const int32		NumVertices	() const	{return m_NumVertices;}
                 
	void			UpdateData	(ID3D11DeviceContext* _pImmediateContext, const MeshImport* _pMeshData);

	void			Create		(ID3D11Device* _pDevice, const MeshImport* _pMeshData);
};

///<
class QuadUV: public Mesh
{
public:

	///< [-1, 1]
	void	Create(ID3D11Device* _pDevice, const char* _csTextureFileName);
	
};

///<
class QuadColor: public Mesh
{
public:

	///< [-1, 1]
	void	Create(ID3D11Device* _pDevice, Vector4f _c);

};

///<
class Cube: public Mesh
{
public:
	void Create(ID3D11Device* _pDevice);
};

///<
class CubeUV: public Mesh
{
public:

	struct VolumeVertex
	{
		Vector4f _x;
		Vector3f _uv;
	};

	///<
	void					Create					(ID3D11Device* _pDevice);
	static ID3D11Buffer*	CreateCubeUV			(ID3D11Device* _pDevice, const Vector3ui _dims);
};

#endif