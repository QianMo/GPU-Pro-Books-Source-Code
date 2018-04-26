#pragma once

#include <map>
#include <vector>
#include <string>

template<class _Kty, class _Ty, class _Pr=std::less<_Kty> >
class CompositMap : public std::map<_Kty, _Ty, _Pr>
{
public:
	void deleteAll()
	{
		iterator i = begin();
		iterator e = end();
		while(i != e)
		{
			if(i->second)
				delete i->second;
			i++;
		}
	}
};

template<class _Kty, class _Ty>
class CompositMapList : public std::map<_Kty, _Ty>
{
public:
	void deleteAll()
	{
		iterator i = begin();
		iterator e = end();
		while(i != e)
		{
			i->second.deleteAll();
			i++;
		}
	}
};

template<class _Ty>
class CompositList : public std::vector<_Ty>
{
public:
	void deleteAll()
	{
		iterator i = begin();
		iterator e = end();
		while(i != e)
		{
			if(*i)
				delete *i;
			i++;
		}
	}
};

template<class _Kty, class _Ty>
class ResourceMap : public std::map<_Kty, _Ty>
{
public:
	void releaseAll()
	{
		iterator i = begin();
		iterator e = end();
		while(i != e)
		{
			if(i->second)
				i->second->Release();
			i++;
		}
	}
};

struct ComparePointers
{
	bool operator() (const void* const& a, const void* const& b) const
	{
		return a < b;
	}
};