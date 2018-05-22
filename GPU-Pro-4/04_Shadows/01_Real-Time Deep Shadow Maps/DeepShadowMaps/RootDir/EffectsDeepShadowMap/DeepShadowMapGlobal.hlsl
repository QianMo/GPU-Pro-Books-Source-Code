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

#define NUM_BUF_ELEMENTS 50
#define FILTER_SIZE 2
#define FILTER_AREA ((FILTER_SIZE * 2 + 1) * (FILTER_SIZE * 2 + 1))
#define ZFAR 1000.0f

unsigned int Dimension;
StructuredBuffer<LinkedListEntryWithPrev> LinkedListBufWPRO;
StructuredBuffer<LinkedListEntryNeighbors> NeighborsBufRO;
StructuredBuffer<StartElementBufEntry> StartElementBufRO;