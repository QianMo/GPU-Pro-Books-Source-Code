
#include <Graphics/MeshImport.hpp>
#include <Common/Common.hpp>
#include <Math/Vector/Vector.hpp>

#include <fstream>

///< General Vertex Data
void MeshImport::Release()
{
	M::Delete(&m_pAnimationImport);
	M::DeleteArray(&m_pImportedSkinData);
	M::DeleteArray(&m_pVertices);
	if (m_pTextureFileName)
		M::DeleteArray(&m_pTextureFileName);
	
	memset(this,0,sizeof(MeshImport));
}

	

///<
void MeshImport::SetTextureFileName(const char* _csFilename)
{
	M::DeleteArray(&m_pTextureFileName);
	size_t tNameS = strlen(_csFilename);
	m_pTextureFileName = new char[tNameS+1];
	memset(m_pTextureFileName,0,tNameS+1);
	memcpy(m_pTextureFileName, _csFilename, tNameS);	
}

///<
void MeshImport::Import(const char* _csFileName)
{
	std::ifstream inFile(_csFileName, std::ios::in | std::ios::binary);

	inFile.read(reinterpret_cast<char*>(&m_NumVertices),sizeof(int32));
	
	inFile.read(reinterpret_cast<char*>(&m_bHasNormals),sizeof(bool));
	inFile.read(reinterpret_cast<char*>(&m_bHasUVs),sizeof(bool));
	inFile.read(reinterpret_cast<char*>(&m_bHasColors),sizeof(bool));	
	inFile.read(reinterpret_cast<char*>(&m_bHasBones),sizeof(bool));	

	int32 nameLength=0;
	inFile.read(reinterpret_cast<char*>(&nameLength),sizeof(int32));
	if (nameLength>0)
	{
		m_pTextureFileName = new char[nameLength];
		memset(m_pTextureFileName,0,nameLength);
		inFile.read(m_pTextureFileName, nameLength);
	}	

	if (m_NumVertices>0 )
	{
		m_pVertices = new VertexData[m_NumVertices];	
		inFile.read(reinterpret_cast<char*>(m_pVertices),m_NumVertices*sizeof(VertexData));
	}

	inFile.close();

}


