#ifndef COMMON_MEM_CHUNK_H_INCLUDED
#define COMMON_MEM_CHUNK_H_INCLUDED

namespace Mod
{
	class mem_chunk
	{
	public:
		mem_chunk(void *rawPtr, size_t byteSize);
		void* raw_ptr() const;
		size_t byte_size() const;

	protected:
		~mem_chunk();

	private:
		size_t	mByteSize;
		void*	mRawPtr;
	};


	// this one is never used to derive from

	class const_mem_chunk
	{
	public:
		const_mem_chunk( const void *rawPtr, size_t byteSize);
		const_mem_chunk( const mem_chunk& chunk); // note implicit
		~const_mem_chunk();

		const void* raw_ptr() const;
		size_t byte_size() const;

	private:
		size_t		mByteSize;
		const void*	mRawPtr;
			
	};

}

#endif