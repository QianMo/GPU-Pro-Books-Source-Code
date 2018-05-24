#include <stdafx.h>
#include <TileMap.h>

static unsigned int GetLog2(float x)
{
  return (unsigned int)(ceil(log(x)/log(2.0f)));
}

void TileMap::Release()
{
  SAFE_DELETE_ARRAY(tileNodeList);
}

bool TileMap::Init(unsigned int mapSize, unsigned int maxAbsTileSize, unsigned int numLevels)
{
  if((!IS_POWEROF2(mapSize)) ||(numLevels < 1) || (maxAbsTileSize > mapSize) || (maxAbsTileSize < 16.0f))
    return false;

  this->mapSize = (float)mapSize; 
  log2MapSize = GetLog2(this->mapSize);
  this->maxAbsTileSize = (float)maxAbsTileSize;
  this->numLevels = numLevels;

  minAbsTileSize = this->mapSize;
  for(unsigned int i=0; i<(numLevels-1); i++)
    minAbsTileSize *= 0.5f;
  if((minAbsTileSize < 16.0f) || (minAbsTileSize > maxAbsTileSize))
    return false;

  numNodes = 1;
  unsigned int multiplier = 1;
  for(unsigned int i=1; i<numLevels; i++)
  {
    multiplier *= 4;
    numNodes += multiplier;
  }
  tileNodeList = new TileNode[numNodes];
  if(!tileNodeList)
    return false;

  TileNode &rootNode = tileNodeList[nodeIndex];
  rootNode.position.SetZero(); 
  rootNode.level = 0;
  rootNode.minLevel = 0;
  BuildTree(rootNode, 0);

  return true;
}

void TileMap::BuildTree(TileNode &parentNode, unsigned int level)
{
  level++;
  if(level == numLevels)
    return;

  for(unsigned int i=0; i<4; i++)
  {
    parentNode.childIndices[i] = ++nodeIndex;
    assert(nodeIndex < numNodes);
    TileNode &currentNode = tileNodeList[parentNode.childIndices[i]];
    unsigned int denominator = 1 << level;
    const float size = 1.0f/((float)denominator);
    Vector2 offsets[4] = { Vector2(-size, size), Vector2(-size, -size),  Vector2(size, -size),  Vector2(size, size) };
    currentNode.position = parentNode.position+offsets[i]; 
    currentNode.level = level;
    currentNode.minLevel = 0;
    BuildTree(currentNode, level);
  }
}

void TileMap::Clear()
{
  for(unsigned int i=0; i<numNodes; i++)
    tileNodeList[i].minLevel = 0;
}

bool TileMap::GetTile(float size, Tile &tile)
{
  CLAMP(size, minAbsTileSize, maxAbsTileSize);
  unsigned int requiredLevel = log2MapSize-GetLog2(size);

  foundNode = NULL;
  TileNode &rootNode = tileNodeList[0];
  FindNode(rootNode, requiredLevel);
  if(!foundNode)
    return false;

  tile.position = foundNode->position;
  tile.size = size/mapSize;

  return true;
}

void TileMap::FindNode(TileNode &parentNode, unsigned int level)
{
  if(foundNode)
    return;

  for(unsigned int i=0; i<4; i++)
  {
    if(foundNode)
      return;

    int childIndex = parentNode.childIndices[i];
    if(childIndex < 0)
      return;

    TileNode &currentNode = tileNodeList[childIndex];
    if(level < currentNode.minLevel)
      continue;
    
    if(level == currentNode.level)
    {
      parentNode.minLevel = level;
      currentNode.minLevel = numLevels;
      foundNode = &currentNode;
      return;
    }

    FindNode(currentNode, level);
  }
}

