#ifndef __BASIC_TERRAIN_HPP__
#define __BASIC_TERRAIN_HPP__

///< Le défis de ne pas être ringard.
#include <Graphics\Dx11\Mesh.hpp>


class Terrain : public Mesh
{

	uint32			m_uiNumIndices;
	Vector2i		m_iDims;

	ID3D11Buffer*				m_pIndexBuffer;
	ID3D11ShaderResourceView*	m_pDisplacementSRV;
	
	///<
	void CreateDisplacementTexture		(ID3D11Device* _pDevice, const char* _csHeightMap);

public:

	Terrain():m_uiNumIndices(0),m_pIndexBuffer(NULL),m_pDisplacementSRV(NULL){}

	///< Resolution: 
	void Create							(ID3D11Device* _pDevice, Vector2i _dims);
	void CreateWithHeightTexture		(ID3D11Device* _pDevice, const char* _csHeightMap);


	void Draw							(ID3D11DeviceContext* _pImmediateContext);
	
	uint32 NumIndices() const {return m_uiNumIndices;}

	~Terrain();
};

#endif