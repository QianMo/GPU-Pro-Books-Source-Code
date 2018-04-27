#include "Precompiled.h"
#include "mem_chunk.h"

namespace Mod
{

	mem_chunk::mem_chunk(void *rawPtr, size_t byteSize):
	mByteSize(byteSize),
	mRawPtr(rawPtr)
	{

	}

	//------------------------------------------------------------------------

	mem_chunk::~mem_chunk()
	{
	}

	//------------------------------------------------------------------------

	void*
	mem_chunk::raw_ptr() const
	{
		return mRawPtr;
	}

	//------------------------------------------------------------------------

	size_t
	mem_chunk::byte_size() const
	{
		return mByteSize;
	}

	//------------------------------------------------------------------------

	const_mem_chunk::const_mem_chunk( const void *rawPtr, size_t byteSize):
	mRawPtr(rawPtr),
	mByteSize(byteSize)
	{
	}

	//------------------------------------------------------------------------

	const_mem_chunk::const_mem_chunk( const mem_chunk& chunk):
	mRawPtr(chunk.raw_ptr()),
	mByteSize(chunk.byte_size())
	{
	}

	//------------------------------------------------------------------------

	const_mem_chunk::~const_mem_chunk()
	{

	}

	//------------------------------------------------------------------------

	const void*
	const_mem_chunk::raw_ptr() const
	{
		return mRawPtr;
	}

	//------------------------------------------------------------------------

	size_t const_mem_chunk::byte_size() const
	{
		return mByteSize;
	}
}
