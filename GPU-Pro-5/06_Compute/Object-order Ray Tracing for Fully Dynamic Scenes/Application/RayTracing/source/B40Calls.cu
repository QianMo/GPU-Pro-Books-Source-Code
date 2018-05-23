#include <b40c/radix_sort/enactor.cuh>
#ifdef B40_ORIGINAL
	#include <b40c/util/multiple_buffering.cuh>
#else
	#include <b40c/util/multi_buffer.cuh>
#endif

#include "B40Calls.h"

namespace cuda
{

class B40SorterI : public B40Sorter
{
	b40c::util::DoubleBuffer<lean::uint4, lean::uint4> m_buffer;
	b40c::radix_sort::Enactor m_enactor;

public:
	B40SorterI(size_t maxCount)
	{
#ifdef B40_ORIGINAL
		m_enactor.ENACTOR_DEBUG = false;
#endif
		
		if (maxCount > 0)
		{
			cudaMalloc((void**) &m_buffer.d_keys[1], sizeof(lean::uint4) * maxCount);
			cudaMalloc((void**) &m_buffer.d_values[1], sizeof(lean::uint4) * maxCount);
		}
	}
	~B40SorterI()
	{
		if (m_buffer.d_keys[1])
			cudaFree(m_buffer.d_keys[1]);
		if (m_buffer.d_values[1])
			cudaFree(m_buffer.d_values[1]);
	}

	void Sort(lean::uint4 *keys, lean::uint4 *values, size_t count, void *stream)
	{
		if (count == 0)
			return;

		b40c::util::DoubleBuffer<lean::uint4, lean::uint4> buffer;
		lean::uint4 originalSelector = buffer.selector;

		buffer.d_keys[originalSelector] = keys;
		buffer.d_values[originalSelector] = values;
		buffer.d_keys[!originalSelector] = m_buffer.d_keys[1];
		buffer.d_values[!originalSelector] = m_buffer.d_values[1];

#ifdef B40_ORIGINAL
		m_enactor.stream = (stream) ? static_cast<cudaStream_t>(stream) : 0;
		m_enactor.Sort<b40c::radix_sort::SMALL_SIZE>(buffer, count);
#else
		m_enactor.Sort(buffer, count); // <b40c::radix_sort::SMALL_PROBLEM, 0, 30>
#endif
		if (buffer.selector != originalSelector)
		{
			cudaMemcpyAsync(keys, buffer.d_keys[buffer.selector], count * sizeof(lean::uint4), cudaMemcpyDeviceToDevice);
			cudaMemcpyAsync(values, buffer.d_values[buffer.selector], count * sizeof(lean::uint4), cudaMemcpyDeviceToDevice);
		}
	}

	bool SortSwap(lean::uint4 *keys, lean::uint4 *values, lean::uint4 *keysAux, lean::uint4 *valuesAux, size_t count, void *stream)
	{
		if (count == 0)
			return false;

		b40c::util::DoubleBuffer<lean::uint4, lean::uint4> buffer;
		lean::uint4 originalSelector = buffer.selector;

		buffer.d_keys[originalSelector] = keys;
		buffer.d_values[originalSelector] = values;
		buffer.d_keys[!originalSelector] = keysAux;
		buffer.d_values[!originalSelector] = valuesAux;

#ifdef B40_ORIGINAL
		m_enactor.stream = (stream) ? static_cast<cudaStream_t>(stream) : 0;
		m_enactor.Sort<0, 30, b40c::radix_sort::SMALL_SIZE>(buffer, count);
#else
		m_enactor.Sort(buffer, count); // <b40c::radix_sort::SMALL_PROBLEM, 0, 30>
#endif
		return (buffer.selector != originalSelector);
	}
};

lean::scoped_ptr<B40Sorter> CreateB40Sorter(size_t maxCount)
{
	return lean::scoped_ptr<B40Sorter>( new B40SorterI(maxCount) );
}

} // namespace
