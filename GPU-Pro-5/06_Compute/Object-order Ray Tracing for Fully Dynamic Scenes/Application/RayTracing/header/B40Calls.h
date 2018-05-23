#pragma once

#include <lean/lean.h>
#include <lean/types.h>

#include <lean/smart/scoped_ptr.h>

namespace cuda
{

class B40Sorter
{
public:
	virtual ~B40Sorter() { };

	virtual void Sort(lean::uint4 *keys, lean::uint4 *values, size_t count, void *stream = 0) = 0;
	virtual bool SortSwap(lean::uint4 *keys, lean::uint4 *values, lean::uint4 *keysAux, lean::uint4 *valuesAux, size_t count, void *stream = 0) = 0;
};

lean::scoped_ptr<B40Sorter> CreateB40Sorter(size_t maxCount);

} // namespace