// ================================================================================ //
// Copyright (c) 2011 Arturo Garcia, Francisco Avila, Sergio Murguia and Leo Reyes	//
//																					//
// Permission is hereby granted, free of charge, to any person obtaining a copy of	//
// this software and associated documentation files (the "Software"), to deal in	//
// the Software without restriction, including without limitation the rights to		//
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies	//
// of the Software, and to permit persons to whom the Software is furnished to do	//
// so, subject to the following conditions:											//
//																					//
// The above copyright notice and this permission notice shall be included in all	//
// copies or substantial portions of the Software.									//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR		//
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,			//
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE		//
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER			//
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,	//
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE	//
// SOFTWARE.																		//
// ================================================================================ //

#include "ArgumentsParser.h"
#include "Model.h"

extern ArgumentsParser m_Parser;
//------------------------------------------------
// Constructor
//------------------------------------------------
Model::Model( string sFileName )
{
	size_t pos = sFileName.find(".");
	m_sName = sFileName.substr(0,pos);

	LoadFile(sFileName);
	Scale();

	m_pAccelStructure = new NullShader(m_ppPrimitives, m_uiNumPrimitives);
	SetCurrentStructureType(AS_BVH);
	/*if(strcmp(m_Parser.GetAccelerationStructure(),"bvh") == 0) SetCurrentStructureType(AS_BVH);	
	else if(strcmp(m_Parser.GetAccelerationStructure(),"lbvh") == 0) SetCurrentStructureType(AS_LBVH);*/
	
}

//------------------------------------------------
// Destructor
//------------------------------------------------
Model::~Model()
{
	for(unsigned int i = 0; i < m_uiNumPrimitives; ++i) SAFE_DELETE(m_ppPrimitives[i]);
	for(unsigned int i = 0; i < m_uiNumMaterials; ++i) SAFE_DELETE(m_ppMaterials[i]);
	SAFE_DELETE(m_ppPrimitives);
	SAFE_DELETE(m_ppMaterials);
	SAFE_DELETE(m_pAccelStructure);
	SAFE_DELETE(m_pVertices);
	SAFE_DELETE(m_pIndices);
}

//------------------------------------------------
// Change structure for the model and rebuild it
//------------------------------------------------
void Model::SetCurrentStructureType(ACCELERATION_STRUCTURE eStructureType) 
{	
	m_eAccelStructureType = eStructureType;
	RebuildStructure();
}

//------------------------------------------------
// Build new acceleration structure
//------------------------------------------------
void Model::RebuildStructure()
{	
	SAFE_DELETE( m_pAccelStructure );

	switch(m_eAccelStructureType)
	{
	case AS_BVH:
		m_pAccelStructure = new BVH(m_ppPrimitives, m_uiNumPrimitives, m_Parser.GetMaxPrimsInNode(), m_Parser.GetBVHSplit());
		break;
	case AS_LBVH:
		//m_pAccelStructure = new LBVH(m_ppPrimitives, m_uiNumPrimitives, 1 << m_Parser.GetLBVHDepth() );
		break;
	case AS_BIH:
		//m_pAccelStructure = new BIH(m_ppPrimitives, m_uiNumPrimitives);
		break;
	case AS_KDTREE:
		//m_pAccelStructure = new KDTree(m_ppPrimitives, m_uiNumPrimitives);
		break;
	case AS_GRID:
		//m_pAccelStructure = new CubicGrid(m_ppPrimitives, m_uiNumPrimitives);
		break;
	case AS_SIMPLE:
		//m_pAccelStructure = new Simple(m_ppPrimitives, m_uiNumPrimitives);
		break;
	default:
		m_pAccelStructure = new NullShader(m_ppPrimitives, m_uiNumPrimitives);
		break;
	}

	LARGE_INTEGER LocalTimer, Freq;
	startTimer(LocalTimer, Freq);

	m_pAccelStructure->Build();

	float totalTime;
	calculateTime(LocalTimer, Freq, totalTime);
	printf("%s Build Time: %f seconds\n",m_pAccelStructure->GetName(), totalTime);

	m_pAccelStructure->PrintOutput(totalTime);
}

//------------------------------------------------
// Load model file
//------------------------------------------------
void Model::LoadFile(string sFileName)
{
	const string sPath = "./Models/";
	const string sFilePath = sPath + sFileName;
	
	if (strstr(sFileName.c_str(), ".txt") || strstr(sFileName.c_str(), ".TXT"))
	{
		FILE *file = NULL;
		fopen_s(&file,sFilePath.c_str(),"rt");
		if(file == NULL)
		{
			printf("Could not open the file %s\n", sFilePath.c_str());
			exit(EXIT_FAILURE);
		}

		fscanf_s(file,"%d",&m_uiNumVertices);
		fscanf_s(file,"%d",&m_uiNumPrimitives);
		unsigned int numIndices = m_uiNumPrimitives*3;
		m_ppPrimitives = new Primitive*[m_uiNumPrimitives];
		m_pVertices = new Vertex[m_uiNumVertices];
		m_pIndices = new unsigned long[numIndices];
		m_uiNumMaterials = 1;
		m_ppMaterials = new Material*[m_uiNumMaterials];
		m_ppMaterials[0] = new Material(0, "default.jpg ");
		for (unsigned int i = 0; i < m_uiNumVertices; i++)
		{
			fscanf_s(file,"%f %f %f %f %f %f %f %f",
				&m_pVertices[i].Pos.x,&m_pVertices[i].Pos.y,&m_pVertices[i].Pos.z,
				&m_pVertices[i].Normal.x,&m_pVertices[i].Normal.y,&m_pVertices[i].Normal.z,
				&m_pVertices[i].U,&m_pVertices[i].V);
			Normalize(m_pVertices[i].Normal);

			float x, y, z;
			x = m_pVertices[i].Pos.x;
			y = m_pVertices[i].Pos.y;
			z = m_pVertices[i].Pos.z;
			float r = sqrtf(x*x+y*y+z*z);
			m_pVertices[i].U = x*7;
			m_pVertices[i].V = y*7;
		} // END for

		int tmpOffset = 0;
		for (unsigned int i = 0; i < m_uiNumPrimitives; i++)
		{
			int a,b,c;

			fscanf_s(file,"%d %d %d",&a,&b,&c);
			m_pIndices[tmpOffset] = a;
			m_pIndices[tmpOffset+1] = b;
			m_pIndices[tmpOffset+2] = c;
			tmpOffset += 3;
			m_ppPrimitives[i] = new Primitive(&m_pVertices[a], &m_pVertices[b], &m_pVertices[c]);
			m_ppPrimitives[i]->SetMaterial(m_ppMaterials[0]);
			m_ppPrimitives[i]->GetMaterial()->SetReflection(0.2f);
		} // END for

		fclose(file);
	}
	else if (strstr(sFileName.c_str(), ".obj") || strstr(sFileName.c_str(), ".OBJ"))
	{
		if (!m_rModelObj.import(sFilePath.c_str()))
		{
			//throw std::runtime_error("Failed to load model.");
			printf("Could not open the model %s\n", sFilePath.c_str());
			exit(EXIT_FAILURE);
		}
		m_rModelObj.normalize();

		m_uiNumVertices = m_rModelObj.getNumberOfVertices();
		m_uiNumPrimitives = m_rModelObj.getNumberOfTriangles();
		m_uiNumMaterials = m_rModelObj.getNumberOfMaterials();

		printf("Parsing model to RT format...\n");

		if(m_uiNumMaterials > 0)
		{
			m_ppMaterials = new Material*[m_uiNumMaterials];
			for(unsigned int i = 0; i < m_uiNumMaterials; ++i)
			{
				m_ppMaterials[i] = new Material( i, m_rModelObj.getMaterial(i).colorMapFilename );
			}
		}
		else
		{
			m_uiNumMaterials = 1;
			m_ppMaterials = new Material*[m_uiNumMaterials];
			m_ppMaterials[0] = new Material( 0, "default.jpg" );
		}
		

		printf("Parsing vertices...\n");
		m_pVertices = new Vertex[m_uiNumVertices];
		for(unsigned int i = 0; i < m_uiNumVertices; ++i)
		{
			m_pVertices[i].Pos.x = m_rModelObj.getVertexBuffer()[i].position[0];
			m_pVertices[i].Pos.y = m_rModelObj.getVertexBuffer()[i].position[1];
			m_pVertices[i].Pos.z = m_rModelObj.getVertexBuffer()[i].position[2];

			m_pVertices[i].Normal.x = m_rModelObj.getVertexBuffer()[i].normal[0];
			m_pVertices[i].Normal.y = m_rModelObj.getVertexBuffer()[i].normal[1];
			m_pVertices[i].Normal.z = m_rModelObj.getVertexBuffer()[i].normal[2];

			m_pVertices[i].U = m_rModelObj.getVertexBuffer()[i].texCoord[0];
			m_pVertices[i].V = -m_rModelObj.getVertexBuffer()[i].texCoord[1];
		}

		unsigned int uiNumIndices = m_uiNumPrimitives*3;
		m_pIndices = new DWORD[uiNumIndices];
		m_ppPrimitives = new Primitive*[m_uiNumPrimitives];
		int tmpOffset = 0;
		
		printf("Parsing primitives...\n");
		for(unsigned int i = 0; i < m_uiNumPrimitives; ++i)
		{
			int a,b,c;
			a = m_rModelObj.getIndexBuffer()[i*3];
			b = m_rModelObj.getIndexBuffer()[i*3+1];
			c = m_rModelObj.getIndexBuffer()[i*3+2];

			m_pIndices[tmpOffset] = a;
			m_pIndices[tmpOffset+1] = b;
			m_pIndices[tmpOffset+2] = c;
			tmpOffset += 3;

			m_ppPrimitives[i] = new Primitive(&m_pVertices[a], &m_pVertices[b], &m_pVertices[c]);
			if(m_uiNumMaterials == 1) 
				m_ppPrimitives[i]->SetMaterial(m_ppMaterials[0]);
		}

		printf("Parsing materials...\n");
		if(m_uiNumMaterials>1)
		{
			for(int i = 0; i < m_rModelObj.getNumberOfMeshes(); ++i)
			{
				int numTriangles = m_rModelObj.getMesh(i).triangleCount;
				int startIndex = m_rModelObj.getMesh(i).startIndex/3;
				int materialId = m_rModelObj.getMaterialCache()[m_rModelObj.getMesh(i).pMaterial->name];
				Material* currentMat = m_ppMaterials[materialId];
				for(int j = 0; j < numTriangles; ++j)
				{
					m_ppPrimitives[startIndex+j]->SetMaterial(currentMat);
				}
			}
		}
	}

	printf("%s imported successfully.\n",sFileName.c_str());
}

//------------------------------------------------
// Scale all models to fit a 0.8 radius sphere
//------------------------------------------------
void Model::Scale()
{
	printf("Scale scene...\n");
	// scale everything to fit the 0.8 radius sphere
	Vector3 MinVec = m_pVertices[0].Pos,MaxVec = m_pVertices[0].Pos;
	for (unsigned int i=1;i<m_uiNumVertices;i++)
	{
		if (m_pVertices[i].Pos.x < MinVec.x)
		{
			MinVec.x=m_pVertices[i].Pos.x;
		}
		if (m_pVertices[i].Pos.x>MaxVec.x)
		{
			MaxVec.x=m_pVertices[i].Pos.x;
		}

		if (m_pVertices[i].Pos.y<MinVec.y)
		{
			MinVec.y=m_pVertices[i].Pos.y;
		}
		if (m_pVertices[i].Pos.y>MaxVec.y)
		{
			MaxVec.y=m_pVertices[i].Pos.y;
		}

		if (m_pVertices[i].Pos.z<MinVec.z)
		{
			MinVec.z=m_pVertices[i].Pos.z;
		}
		if (m_pVertices[i].Pos.z>MaxVec.z)
		{
			MaxVec.z=m_pVertices[i].Pos.z;
		}
	}
	MaxVec = MaxVec - MinVec;
	float scale = 1.8f/max(MaxVec.x,max(MaxVec.y,MaxVec.z));
	MinVec= MinVec+(0.5f*MaxVec);
	for (int i=m_uiNumVertices-1;i>=0;i--)
	{
		m_pVertices[i].Pos=m_pVertices[i].Pos-MinVec;
		m_pVertices[i].Pos=scale*m_pVertices[i].Pos;
	}

	// scale again but now to fit the 0.49 radius sphere
	scale = 0.0f;
	for (unsigned int i=0;i<m_uiNumVertices;i++)
	{
		float lng;
		Dot(lng, m_pVertices[i].Pos, m_pVertices[i].Pos);
		
		if (lng>scale)
		{
			scale = lng;
		}
	}
	scale=sqrt(scale);
	scale=0.49f/scale;
	for (int i=m_uiNumVertices-1;i>=0;i--)
	{
		m_pVertices[i].Pos.x*=scale;
		m_pVertices[i].Pos.y*=scale;
		m_pVertices[i].Pos.z*=scale;
	}
}