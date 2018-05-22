#ifndef __SINGLETON_H__
#define __SINGLETON_H__

#include <typeinfo>

template <typename T>
class Singleton
{
public:
	static T* Instance(void);

protected:
#ifndef _DEBUG
	static T* instance;
#endif

	Singleton(void)
	{
	}

	virtual ~Singleton(void)
	{
#ifndef _DEBUG
		if (instance != NULL)
		{
			delete instance;
			instance = NULL;
		}
		if (instance)
		{
			instance->~T();
			instance=NULL;
		}
#endif
	}

private:
	Singleton(const Singleton &);
	Singleton& operator= (const Singleton&);
};

template <class T>
T* Singleton<T>::Instance(void)
{
#ifndef _DEBUG
	if (instance==NULL)
	{
		instance=new T;
	}
	return instance;
#else
	static T instance;
	return &instance;
#endif
}

#ifndef _DEBUG
template <class T>
T* Singleton<T>::instance=NULL;
#endif

#endif
