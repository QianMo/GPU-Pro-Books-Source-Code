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


typedef unsigned char Bool;

static const Bool False = 0;
static const Bool True = 1;


//Create a tree of connected vertices with a root in the first vertex
template<typename INDICES_TYPE>
class AdjacencyTreeBuilder
{
public:

	typedef std::pair<INDICES_TYPE, INDICES_TYPE> Edge;

private:


	std::vector<Bool> visitedVertices;

	std::vector<Edge> adjacenciesTree;

	//////////////////////////////////////////////////////////////////////////
	int FindFirstUnvisitedNode()
	{
		for ( size_t i = 0; i < visitedVertices.size(); i++ )
		{
			if ( visitedVertices[i] == False )
			{
				return i;
			}
		}
		return -1;
	}

public:

	//////////////////////////////////////////////////////////////////////////
	void Build(const std::vector<INDICES_TYPE> &indices, int maxIndice)
	{
		adjacenciesTree.clear();

		visitedVertices.resize( maxIndice, False );

		std::vector< INDICES_TYPE > verticesToProcess;
		verticesToProcess.reserve( maxIndice );

		int firstUnvisitedNode = (int)indices[0];
		while (firstUnvisitedNode != -1)
		{
			verticesToProcess.push_back( (INDICES_TYPE)firstUnvisitedNode );
			adjacenciesTree.push_back( Edge(firstUnvisitedNode, firstUnvisitedNode) );
			size_t currentArrayIndex = 0;

			while (currentArrayIndex < verticesToProcess.size())
			{
				INDICES_TYPE currentVertexIndex = verticesToProcess[currentArrayIndex];
				visitedVertices[currentVertexIndex] = True;
				for ( size_t index = 0; index < indices.size(); index += 3 )
				{
					INDICES_TYPE index0;
					INDICES_TYPE index1;
					INDICES_TYPE index2;

					if ( indices[index + 0] == currentVertexIndex )
					{
						index0 = indices[index + 0];
						index1 = indices[index + 1];
						index2 = indices[index + 2];
					} 
					else
					{
						if ( indices[index + 1] == currentVertexIndex )
						{
							index0 = indices[index + 1];
							index1 = indices[index + 0];
							index2 = indices[index + 2];
						}
						else
						{
							if ( indices[index + 2] == currentVertexIndex )
							{
								index0 = indices[index + 2];
								index1 = indices[index + 0];
								index2 = indices[index + 1];
							}
							else
							{
								continue;
							}
						}
					}

					if ( visitedVertices[index1] == False )
					{
						visitedVertices[index1] = True;
						verticesToProcess.push_back( index1 );
						adjacenciesTree.push_back( Edge(index1, index0) );
					}

					if ( visitedVertices[index2] == False )
					{
						visitedVertices[index2] = True;
						verticesToProcess.push_back( index2 );
						adjacenciesTree.push_back( Edge(index2, index0) );
					}
				}

				currentArrayIndex++;
			} // while (currentArrayIndex < verticesToProcess.size())

			firstUnvisitedNode = FindFirstUnvisitedNode();
			verticesToProcess.clear();
		} // while ( firstUnvisitedNode != -1 )
	}


	const std::vector<Edge> & GetAdjacencies()
	{
		return adjacenciesTree;
	}
};




