#include <stdafx.h>

void* __cdecl operator new(size_t size)
{
  return malloc(size);
}

void* __cdecl operator new[](size_t size)
{
  return malloc(size);
}

void __cdecl operator delete(void *p)
{
  free(p);
}

void __cdecl operator delete[](void *p)
{
  free(p);
}

