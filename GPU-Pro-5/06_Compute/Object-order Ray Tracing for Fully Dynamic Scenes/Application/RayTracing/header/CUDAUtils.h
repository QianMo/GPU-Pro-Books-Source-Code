#pragma once

#include "Tracing.h"

#include <beGraphics/Any/beAPI.h>
#include <cuda_runtime.h>

#include "CUDAErrors.h"

#include <cuda_D3D11_interop.h>
#include <lean/smart/scoped_ptr.h>

struct cuda_resource_ptr_policy
{
	static LEAN_INLINE void release(cudaGraphicsResource *object)
	{
		if (object)
			cudaGraphicsUnregisterResource(object);
	}
};

typedef lean::scoped_ptr<cudaGraphicsResource, lean::stable_ref, cuda_resource_ptr_policy> scoped_cgr_ptr;

inline scoped_cgr_ptr CreateCGR(beGraphics::API::Resource *resource, lean::uint4 flags = cudaGraphicsRegisterFlagsNone)
{
	scoped_cgr_ptr ptr;
	BE_THROW_CUDA_ERROR_MSG(
		cudaGraphicsD3D11RegisterResource(ptr.rebind(), resource, flags),
		"cudaGraphicsD3D11RegisterResource");
	return ptr;
}
