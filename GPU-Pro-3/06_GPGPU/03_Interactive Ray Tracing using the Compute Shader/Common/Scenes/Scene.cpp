// ================================================================================ //
// Copyright (c) 2011, Intel Corporation											//
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
//																					//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR		//
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,			//
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE		//
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER			//
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,	//
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN		//
// THE SOFTWARE.																	//
// ================================================================================ //

#include "Scene.h"


//------------------------------------------------
// Constructor
//------------------------------------------------
Scene::Scene(std::map<string,vector<Point>> &sFileNames)
{	
	m_pModels = new Model*[sFileNames.size()];
	m_pObjects = new Object*[sFileNames.size()];

	m_uiNumModels = sFileNames.size();
	std::map<string,vector<Point>>::iterator it = sFileNames.begin();

	for ( it=sFileNames.begin(); it != sFileNames.end(); ++it )
	{
		for(unsigned j = 0; j < (*it).second.size(); ++j)
		{
			++m_uiNumObjects;
		}
	}

	unsigned int i = 0;
	unsigned int objectCounter = 0;

	for ( it=sFileNames.begin(); it != sFileNames.end(); ++it, ++i )
	{
		m_pModels[i] = new Model((*it).first);
		for(unsigned j = 0; j < (*it).second.size(); ++j)
		{
			m_pObjects[objectCounter] = new Object(m_pModels[i], (*it).second[j]);
			++objectCounter;
		}
	}
}

//------------------------------------------------
// Destructor
//------------------------------------------------
Scene::~Scene()
{
	for(unsigned int i = 0; i < m_uiNumObjects; ++i)
	{
		SAFE_DELETE( m_pObjects[i] );
	}
	SAFE_DELETE( m_pObjects );
	
	for(unsigned int i = 0; i < m_uiNumModels; ++i)
	{
		SAFE_DELETE( m_pModels[i] );
	}
	SAFE_DELETE( m_pModels );
}

//------------------------------------------------
// Change structure for each model on the scene
//------------------------------------------------
void Scene::ChangeStructure(ACCELERATION_STRUCTURE eStructure)
{
	for(int i = m_uiNumModels-1; i >= 0; --i)
	{
		m_pModels[i]->SetCurrentStructureType(eStructure);
	}
}

//------------------------------------------------
// Rebuidl structure for each model on the scene
//------------------------------------------------
void Scene::Rebuild()
{
	for(int i = m_uiNumModels-1; i >= 0; --i)
	{
		m_pModels[i]->RebuildStructure(); 
	}
}