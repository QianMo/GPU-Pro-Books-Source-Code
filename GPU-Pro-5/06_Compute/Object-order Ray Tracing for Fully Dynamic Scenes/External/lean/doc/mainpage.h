/** @mainpage

@section Introduction

Lean C++ library is a compact library providing efficient utility classes such as accelerated STL container wrappers, liberal near-to-zero-overhead strong and weak resource pointers working with intrusive and atomic reference counting, smart COM pointers, polymorphic objects with value semantics, opaque value types allowing for the implementation of type-safe thin wrappers without necessarily imposing the heavy dependencies that might be involved on all interface users, etc.

The library makes use of some C++0x features such as move semantics, yet it tries to retain backwards-compatibility with the previous C++ standard. Currently, platform-specific components are only implemented for Microsoft Visual C++. The library is developed in the context of high-performance real-time game and graphics development to both simplify and optimize applications in this field. The library is tested using Boost.Test.

@section Components
Lean C++ library is still in an early stage of development. The following components are either currently being worked on or planned for future implementation:

	- @b Containers
		- @b simple_vector: Optimized fast vector class that partially implements the STL vector interface.
		- @b accumulation_vector: STL vector wrapper class that eliminates element construction and destruction overhead for vectors that are re-filled frequently.
		- @b unordered_vector: STL vector wrapper class that makes no guarantees about the order of its elements when it comes to erasure or insertion of elements.
		- @b accumulation_map: STL map wrapper class that eliminates heavy element insertion/removal overhead for maps that are frequently re-filled with similar elements by invalidating elements rather than erasing them.
	- @b Smart
		- @b ref_counter: Allows for atomic strong and weak (intrusive) reference counting, including a thread-safe lock mechanism.
		- @b resource: Base class that holds a ref_counter object, allowing derived classes to be managed by smart resource pointers.
		- @b resource_ptr: Holds a strong reference, preventing assigned resource objects from being destructed.
		- @b weak_resource_ptr: Holds a weak reference, checking if assigned resource objects are still alive on access.
		- @b cloneable: Abstract interface requiring derived classes to implement clone and destroy methods.
		- @b cloneable_obj: Handle class that allows polymorphic cloneable objects to be treated like plain values.
		- @b com_ptr: Smart pointer class that automates reference counting for COM-like classes such as provided by the DirectX API.
		- @b auto_restore: Automatically restores a given variable to its initial value on destruction.
		- @b scoped_lock: Performs automatic locking / unlocking on a given lockable object, following a given locking policy.
	- @b PImpl
		- @b opaque_value: Allows for forward declaration of arbitrary implementation types (including typedefs) that only get fully accessible when the user decides to include the required dependencies, while at the same time retaining type safety.
		- @b pimpl_ptr: Simple unique_ptr-like class that watches over the safe destruction of types that might only be forward-declared.
	- @b Memory
		- @b heap: Raw heap concept that allows for the specification of memory sources via template arguments, focussing on memory allocation and data alignment as opposed to STL allocators focussing on object construction.
		- @b crt_heap: Wraps the standard operators new and delete into a simple raw heap concept allowing for both aligned and unaligned memory allocation.
		- @b aligned: Base class that aligns the instances of derived classes on boundaries that are multiples of the specified alignment both on the stack and a given heap.
		- @b heap_allocator: Adapts lean's raw heap concept to the interface of the STL allocator concept, thus making it compatible with STL container classes. Optionally allows for the alignment of all memory requested via the allocator, taking a corresponding template argument.
	- @b Properties
		- @b property:
		- @b property_collection:
		- @b property_driven:
	- @b I/O
		- @b endianness: Utilities that allow for efficient single and bulk endian conversions using intrinsics.
		- @b wcharcvt: STL facet implementing proper wide-character encoding following the STL codecvt interface.
	- @b Timer
		- @b timer: CRT time() wrapper class.
		- @b clock_timer: CRT clock() wrapper class.
		- @b high_timer: High-resolution timer class that allows for precise profiling using timer hardware (via WinAPI).
	- @b Concurrent
		- @b atomic: increment, decrement, test-and-set, set
		- @b spin_lock: Synchronization primitive that implements busy waiting built on the atomics specified above.
		- @b shared_spin_lock: Enhanced spin_lock that allows for reader/writer synchronization.

@section Acknowledgements
There are quite some people who have heavily influenced the way this library has turned out, among them Chris Maiwald, who not only kindly gave broad insight into his highly optimized "Cric Framework", bearing tons of well-researched knowledge and experience, but also documented his findings in countless detailed forum posts, thus providing a valuable and evergrowing source of information on many sparsely covered topics. Also among them is Michael Kenzel, with whom I had the chance to discuss many problems and ideas, and who was never reluctant to share his expertise on manifold topics, including the C++ standard. Finally, among them is Michael Raulf, often providing me with split-second answers when it comes to C++' peculiarities, as well as bringing up important idioms whenever they could come in handy. */