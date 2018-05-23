#ifdef LEAN_BUILD_LIB
#include "../../depconfig.h"
#endif

#include "../new_handler.h"
#include <new>
#include <stdexcept>

namespace lean
{
namespace memory
{
namespace impl
{

/// Gets a reference to the static variable storing the new handler.
inline new_handler& get_internal_new_handler()
{
	static new_handler handler = nullptr;
	return handler;
}

/// Calls the lean new handler, throws bad_alloc on failure.
LEAN_ALWAYS_LINK void lean_new_handler()
{
	if(!call_new_handler())
	{
		static const std::bad_alloc exception;
		throw exception;
	}
}

} // namespace
} // namespace
} // namespace

// Sets a new new_handler.
LEAN_MAYBE_LINK lean::memory::new_handler lean::memory::set_new_handler(new_handler newHandler)
{
	new_handler &internalHandler = impl::get_internal_new_handler();
	new_handler oldHandler = internalHandler;
	internalHandler = newHandler;
	std::set_new_handler(&impl::lean_new_handler);
	return oldHandler;
}

// Calls the current new handler.
LEAN_MAYBE_LINK bool lean::memory::call_new_handler()
{
	const new_handler &handler = impl::get_internal_new_handler();
	return (handler) ? (*handler)() : false;
}
