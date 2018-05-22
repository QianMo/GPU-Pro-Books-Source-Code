struct LinkedListEntryWithPrev
{
	float depth;
	float shading;	// stores final shading
	int next;
	int prev;
};

struct LinkedListEntryNeighbors
{
	int right;
	int top;
};

struct StartElementBufEntry
{
	int start;
};

unsigned int Dimension;
StructuredBuffer<LinkedListEntryWithPrev> LinkedListBufWPRO;
StructuredBuffer<LinkedListEntryNeighbors> NeighborsBufRO;
StructuredBuffer<StartElementBufEntry> StartElementBufRO;