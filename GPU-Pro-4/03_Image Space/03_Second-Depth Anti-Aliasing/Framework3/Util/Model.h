
/* * * * * * * * * * * * * Author's note * * * * * * * * * * * *\
*   _       _   _       _   _       _   _       _     _ _ _ _   *
*  |_|     |_| |_|     |_| |_|_   _|_| |_|     |_|  _|_|_|_|_|  *
*  |_|_ _ _|_| |_|     |_| |_|_|_|_|_| |_|     |_| |_|_ _ _     *
*  |_|_|_|_|_| |_|     |_| |_| |_| |_| |_|     |_|   |_|_|_|_   *
*  |_|     |_| |_|_ _ _|_| |_|     |_| |_|_ _ _|_|  _ _ _ _|_|  *
*  |_|     |_|   |_|_|_|   |_|     |_|   |_|_|_|   |_|_|_|_|    *
*                                                               *
*                     http://www.humus.name                     *
*                                                                *
* This file is a part of the work done by Humus. You are free to   *
* use the code in any way you like, modified, unmodified or copied   *
* into your own work. However, I expect you to respect these points:  *
*  - If you use this file and its contents unmodified, or use a major *
*    part of this file, please credit the author and leave this note. *
*  - For use in anything commercial, please request my approval.     *
*  - Share your work and ideas too as much as you can.             *
*                                                                *
\* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef _MODEL_H_
#define _MODEL_H_

#include "../Platform.h"
#include "KdTree.h"
#include "../Renderer.h"

typedef int StreamID;
typedef int BatchID;

struct Stream {
	float *vertices;
	uint *indices;
	uint nVertices;
	
	int nComponents;
	AttributeType type;

	bool optimized;
};

struct Batch {
	uint startIndex;
	uint nIndices;
	uint startVertex;
	uint nVertices;
};

class Model {
public:
	Model();
	~Model();

	// Utility functions
	void createSphere(const int subDivLevel);

	StreamID findStream(const AttributeType type, const uint index = 0) const;
	const Stream &getStream(const StreamID stream) const { return streams[stream]; }
	uint getStreamCount() const { return streams.getCount(); }

	void changeAllGeneric(const bool excludeVertex = false);
	void changeStreamType(const StreamID stream, const AttributeType type){ streams[stream].type = type; }

	BatchID addBatch(const uint startIndex, const uint nIndices);
	const Batch &getBatch(const BatchID batch) const { return batches[batch]; }
	uint getBatchCount() const { return batches.getCount(); }
	bool mergeBatches(const BatchID batch, const BatchID toMerge);
	void removeAllBatches(){ batches.clear(); }

	void getBoundingBox(const StreamID stream, float *minCoord, float *maxCoord) const;

	bool load(const char *fileName);
	bool save(const char *fileName);
	bool loadObj(const char *fileName);
	bool saveObj(const char *fileName);
	bool loadT3d(const char *fileName, const bool removePortals = true, const bool removeInvisible = true, const bool removeTwoSided = false, const float texSize = 256.0f);

	uint getVertexSize() const;
	uint getComponentCount() const;
	uint getComponentCount(const StreamID *cStreams, const uint nStreams) const;

	uint getIndexCount() const { return nIndices; }
	void setIndexCount(const uint nInds){ nIndices = nInds; }

	StreamID addStream(const AttributeType type, const int nComponents, const uint nVertices, float *vertices, uint *indices, bool optimized);
	void removeStream(const StreamID stream);

	void scale(const StreamID stream, const float *scaleFactors);
	void translate(const StreamID stream, const float *translation);
	void flipComponents(const StreamID stream, const uint c0, const uint c1);

	void reverseWinding();
	bool flipNormals();
	bool computeNormals(const bool flat = false);
	bool computeTangentSpace(const bool flat = false);
	bool addStencilVolume();

	void cleanUp();

	void copyVertex(const uint destIndex, const Model &srcModel, const uint srcIndex);
	void interpolateVertex(const uint destIndex, const Model &srcModel, const uint srcIndex0, const uint srcIndex1, const float x);
	bool split(const vec3 &normal, const float offset, Model *front, Model *back) const;
	bool merge(const Model *model);

	void clear();
	void copy(const Model *model);

	void optimize();
	void optimizeStream(const StreamID streamID);
	uint assemble(const StreamID *aStreams, const uint nStreams, float **destVertices, uint **destIndices, bool separateArrays);

	uint makeDrawable(Renderer *renderer, const bool useCache = true, const ShaderID shader = SHADER_NONE);
	void unmakeDrawable(Renderer *renderer);

	void setBuffers(Renderer *renderer);

	void draw(Renderer *renderer);
	void drawBatch(Renderer *renderer, const uint batch);
	void drawSubBatch(Renderer *renderer, const uint batch, const uint first, const uint count);

	static uint *getArrayIndices(const uint nVertices);
protected:

	uint nIndices;

	VertexFormatID vertexFormat;
	VertexBufferID vertexBuffer;
	IndexBufferID indexBuffer;
	
	Array <Stream> streams;
	Array <Batch> batches;

	// Cached
	uint lastVertexCount;
	float *lastVertices;
	uint *lastIndices;
	FormatDesc *lastFormat;
};

#endif // _MODEL_H_
