#ifndef COMMON_AUTO_ARRAY_H_INCLUDED
#define COMMON_AUTO_ARRAY_H_INCLUDED

#include "mem_chunk.h"

namespace Mod
{
	template <typename T>
	class auto_array : public mem_chunk
	{
		// types
	public:
		typedef T value_type;
	public:
		explicit auto_array( size_t a_size ):
		mem_chunk( new T[a_size], a_size * sizeof(T) ),
		elSize(a_size)
		{

		}

		~auto_array()
		{
			delete [] ptr();
		}

		T& operator [] (size_t idx) const
		{
			if( !has_idx(idx) )
				MD_FERROR( L"auto_array::operator []: array index out of bounds! " );
			return ptr()[idx];
		}

		bool has_idx(size_t idx) const 
		{
			return idx >=0 && idx < elSize;
		}

		T* ptr () const
		{
			return static_cast<T*>(raw_ptr());
		}

		size_t size() const
		{
			return elSize;
		}

		// iterator, no, its not stl compliant
	public:

		class iterator
		{
			friend class auto_array;
		public:
			T& operator * ()
			{
				if(check())
					return *ptr;
				else
					MD_FERROR( L"auto_array::iterator::operator *: iterator out of bounds!" );
			}
			T* operator -> ()
			{
				if(check())
					return ptr;
				else
					MD_FERROR( L"auto_array::iterator::operator ->: iterator out of bounds!" );
			}
			iterator& operator ++ () 
			{
				++ptr;
				return *this;
			}
			iterator operator ++ (int)
			{			
				++ptr;
				return iterator(ptr-1, start, end);
			}

		private:
			bool check()
			{
				return ptr >= start && ptr < end;
			}
			iterator(T *a_ptr, T* a_start, T *a_end ): ptr(a_ptr), start(a_start), end(a_end) {}

			T* ptr;
			T* start;
			T* end;
		};

		iterator ibegin() const
		{
			T *p = ptr();
			return iterator(p, p, p+elSize);
		}

		iterator iend() const
		{
			T *p = ptr();
			return iterator(p+elSize, p, p+elSize);
		}

		T* begin() const
		{
			return ptr();
		}

		T* end() const
		{
			return ptr() + elSize;
		}

		// data
	private:
		size_t elSize; // size in elements

		// constraints
	private:
		auto_array( const auto_array& );
		void operator = (const auto_array& );
	};
}

#endif