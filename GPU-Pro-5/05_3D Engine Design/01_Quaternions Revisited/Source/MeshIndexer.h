/****************************************************************************

  GPU Pro 5 : Quaternions revisited - sample code
  All sample code written from scratch by Sergey Makeev specially for article.

  Copyright (c) 2013, Sergey Makeev

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software.

  2. If you use this software in a non-commercial product, an acknowledgment
     in the product documentation would be appreciated but is not required.

  3. If you use this software in a commercial product you requried to credit
     the author.

  4. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.

  5. This notice may not be removed or altered from any source distribution.
    

   Please let me know if you use the code in your products or have any questions or suggestions.

   e-mail: sergeymakeev@inbox.ru
   http://linkedin.com/in/sergeymakeev/

****************************************************************************/
#pragma once

#include <vector>


template<typename T>
class MeshIndexer
{
public:

	static const int INVALID_INDEX = -1;
	static const int GRID_SIZE = 16;

	struct Vertex
	{
		T v;
		int nextIndex;

		Vertex(const T & vertex)
		{
			v = vertex;
			nextIndex = -1;
		}
	};


private:

	static const int GRID_MASK = (GRID_SIZE - 1);
	static const int GRID_CELL_COUNT = GRID_SIZE * GRID_SIZE * GRID_SIZE;


	struct Cell
	{
		int firstIndex;
	};

	Cell* grid;

	int currentIndex;
	float invCellSize;

	std::vector<Vertex> vertexBuffer;
	std::vector<int> indexBuffer;


	MeshIndexer(const MeshIndexer & ) {}
	MeshIndexer & operator= (const MeshIndexer & ) { return *this; }

public:

	//////////////////////////////////////////////////////////////////////////
	MeshIndexer(float cellSize)
	{
		static_assert(Utils::CompileTimeIsPow2<GRID_SIZE>::result, "MeshIndexer GRID_SIZE must be power of 2");

		grid = new Cell[GRID_CELL_COUNT];
		clear(cellSize);
	}

	//////////////////////////////////////////////////////////////////////////
	~MeshIndexer()
	{
		delete [] grid;
		grid = NULL;
	}


	//////////////////////////////////////////////////////////////////////////
	std::vector<Vertex> & GetMutableVertexBuffer()
	{
		return vertexBuffer;
	}

	
	//////////////////////////////////////////////////////////////////////////
	const std::vector<Vertex> & GetVertexBuffer() const
	{
		return vertexBuffer;
	}

	//////////////////////////////////////////////////////////////////////////
	const std::vector<int> & GetIndexBuffer() const
	{
		return indexBuffer;
	}


	//////////////////////////////////////////////////////////////////////////
	void clear(float cellSize)
	{
		currentIndex = 0;
		invCellSize = 1.0f / cellSize;
		
		for (int i = 0; i < GRID_CELL_COUNT; i++)
		{
			grid[i].firstIndex = INVALID_INDEX;
		}

		//swap trick to free memory
		indexBuffer.clear();
		std::vector<int>().swap(indexBuffer);

		vertexBuffer.clear();
		std::vector<MeshIndexer::Vertex>().swap(vertexBuffer);
	}

	//////////////////////////////////////////////////////////////////////////
	void Reserve(int trianglesCount)
	{
		indexBuffer.reserve( trianglesCount * 3 );
		vertexBuffer.reserve( trianglesCount * 2 );
	}

	//////////////////////////////////////////////////////////////////////////
	int AddVertex(T & vertex)
	{
		unsigned int cellX = (unsigned int)(vertex.pos.x * invCellSize) & GRID_MASK;
		unsigned int cellY = (unsigned int)(vertex.pos.y * invCellSize) & GRID_MASK;
		unsigned int cellZ = (unsigned int)(vertex.pos.z * invCellSize) & GRID_MASK;

		int cellAddr = (cellZ * GRID_SIZE * GRID_SIZE) + (cellY * GRID_SIZE) + cellX;

		ASSERT(cellAddr < GRID_CELL_COUNT, "Cell address is out of bounds");

		Cell & cell = grid[cellAddr];

		int indice = cell.firstIndex;
		while(indice != INVALID_INDEX)
		{
			MeshIndexer::Vertex & cellVertex = vertexBuffer[indice];
			if (cellVertex.v.CanWeld(vertex))
			{
				indexBuffer.push_back(indice);
				return indice;
			}

			indice = cellVertex.nextIndex;
		}

		//add new vertex
		indice = (int)vertexBuffer.size();
		
		indexBuffer.push_back(indice);
		vertexBuffer.push_back(vertex);

		//link to intrusive list
		vertexBuffer.back().nextIndex = cell.firstIndex;
		cell.firstIndex = indice;

		return indice;
	}



};