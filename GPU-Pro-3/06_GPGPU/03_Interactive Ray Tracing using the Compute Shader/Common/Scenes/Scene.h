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

#ifndef __SCENE_H__
#define __SCENE_H__

#include "Object.h"
#include "Model.h"

class Scene
{
private:
	Object**					m_pObjects;
	Model**						m_pModels;
	AccelerationStructure*		m_pAccelStructure;

	unsigned int				m_uiNumObjects;
	unsigned int				m_uiNumModels;
public:
	Scene(std::map<string,vector<Point>> &a_FileNames);
	~Scene();

	void						Rebuild();
	void						ChangeStructure(ACCELERATION_STRUCTURE a_Structure);

	// Getters
	Object**					GetObjects( void )			{ return m_pObjects; }
	Model**						GetModels( void )			{ return m_pModels; }
	AccelerationStructure*		GetAccelStructure( void )	{ return m_pAccelStructure; }
	unsigned int				GetNumObjects( void )		{ return m_uiNumObjects; }
	unsigned int				GetNumModels( void )		{ return m_uiNumModels; }
};

#endif