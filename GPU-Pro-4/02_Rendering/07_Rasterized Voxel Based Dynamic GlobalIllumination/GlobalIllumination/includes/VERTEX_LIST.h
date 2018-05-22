#ifndef VERTEX_LIST_H
#define VERTEX_LIST_H

// VERTEX_LIST
//   Simple dynamic list that can handle dynamically vertices of different size.
class VERTEX_LIST
{
public:
	VERTEX_LIST()		
	{
		numVertices = 0;
		vertexSize = 0;
		listSize = 0; 
		entries = NULL;
	}

	~VERTEX_LIST()
	{
		if(entries)
			free(entries);
		entries = NULL;
	}

	// initializes list with the vertex size
	void Init(int vertexSize)
	{
		this->vertexSize = vertexSize;
	}

	float & operator[](int index) const
	{
		return entries[index*vertexSize];
	}

	operator void*()
	{
		return (void*)entries;
	}

	operator const void*()
	{
		return (const void*)entries;
	}

	// adds count vertices to end of list
	int AddElements(int vertexCount,const float *newEntries)
	{
		if(listSize-numVertices<vertexCount)
		{
			if(!ChangeSpace(numVertices+vertexCount))
				return -1;
		} 
		memcpy(&entries[numVertices*vertexSize],newEntries,vertexCount*vertexSize*sizeof(float));
		int firstEntryIndex = numVertices;
		numVertices += vertexCount;
		return firstEntryIndex;
	}

	// gets number of currently used vertices (not the actual list-size!!)
	const int GetSize() const
	{	
		return numVertices;	
	}

	// changes size of list
	bool Resize(int vertexCount)
	{
		if(listSize<vertexCount)
		{
			if(!ChangeSpace(vertexCount))
				return false;
		} 
		else
		{
			numVertices = vertexCount;
			if(!ChangeSpace(vertexCount))
				return false;
		}
		return true;
	}

	// resets number of currently used vertices (not the actual list-size!!)
	void Clear()
	{	
		numVertices = 0;	
	}

	// frees elements of list
	void Erase()
	{
		numVertices = 0;
		listSize = 0; 
		if(entries)
			free(entries);
		entries = NULL;
	}

	float *entries; // elements of list

private:
	// changes memory-size of list (in count vertices)
	bool ChangeSpace(int newSize)
	{
		entries = (float*)realloc(entries,sizeof(float)*newSize*vertexSize);
		listSize = newSize;
		return true;
	}

	int numVertices; 	// number of currently used vertices
	int vertexSize; // vertex size
	int listSize; // overall size of list (memory-size) in count vertices

};

#endif	