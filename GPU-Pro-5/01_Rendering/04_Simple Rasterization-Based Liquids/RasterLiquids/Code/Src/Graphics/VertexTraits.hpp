#ifndef __VERTEX_TRAITS_hPP__
#define __VERTEX_TRAITS_hPP__

#include <Graphics/MeshImport.hpp>

namespace M{


	template<class T>
	T* CreateVertexArray(MeshImport::VertexData* _pV, const int32 _iNumVertices)
	{
		T* pV = (T*)malloc(_iNumVertices*sizeof(T)); 

		for (int32 i=0; i<_iNumVertices; ++i)
			pV[i].Copy(_pV[i]);	

		return pV;	
	}

	///<
	struct Vertex
	{
		Vector3f _x;

		inline void Copy(MeshImport::VertexData& _v)
		{
			_x = _v._x;
		}
	};

	///<
	struct ColorVertex
	{
		Vector3f _x;
		Vector4f _c;

		inline void Copy(MeshImport::VertexData& _v)
		{
			_x = _v._x; _c = _v._c;
		}
	};

	///<
	struct UVVertex
	{
		Vector3f _x;
		Vector2f _uv;

		inline void Copy(MeshImport::VertexData& _v)
		{
			_x = _v._x; _uv = _v._uv;
		}
	};

	struct NormalVertex
	{
		Vector3f _x;
		Vector3f _n;

		inline void Copy(MeshImport::VertexData& _v)
		{
			_x = _v._x; _n = _v._n;
		}
	};

	///<
	struct NormalUVVertex
	{
		Vector3f _x;
		Vector3f _n;
		Vector2f _uv;

		inline void Copy(MeshImport::VertexData& _v)
		{
			_x = _v._x; _n=_v._n; _uv = _v._uv;
		}
	};


	///<
	struct NormalColorVertex
	{
		Vector3f _x;
		Vector3f _n;
		Vector4f _c;

		inline void Copy(MeshImport::VertexData& _v)
		{
			_x = _v._x; _n=_v._n; _c = _v._c;
		}		
	};

	struct VertexUVVolume
	{
		Vector3f _x;
		Vector3f _uv;
	};


}
#endif