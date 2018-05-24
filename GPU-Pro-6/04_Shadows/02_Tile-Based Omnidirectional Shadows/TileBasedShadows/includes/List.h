#ifndef LIST_H
#define LIST_H

// List
//
// A very simple dynamic list used instead of the stl vector. 
template <class T>
class List
{
public:
  List(): 
    entries(NULL),
    numEntries(0),
    listSize(0)   
  {
  }

  ~List()
  {
    if(entries)
      free(entries);
    entries = NULL;
  }

  T& operator[](int index) const
  {
    return entries[index];
  }

  operator void*()
  {
    return (void*)entries;
  }

  operator const void*()
  {
    return (const void*)entries;
  }

  // adds one new element to end of list
  int AddElement(const T *newEntry)
  {
    if((listSize-numEntries) < 1)
    {
      if(!ChangeSpace(numEntries+1))
        return -1;
    } 
    memcpy(&entries[numEntries], newEntry, sizeof(T));
    int firstEntryIndex = numEntries;
    numEntries++;
    return firstEntryIndex;
  }

  // adds count elements to end of list
  int AddElements(unsigned int count, const T *newEntries)
  {
    if((listSize-numEntries) < count)
    {
      if(!ChangeSpace(numEntries+count))
        return -1;
    } 
    memcpy(&entries[numEntries], newEntries, count*sizeof(T));
    int firstEntryIndex = numEntries;
    numEntries += count;
    return firstEntryIndex;
  }

  // gets number of currently used elements (not the actual list-size!!)
  unsigned int GetSize() const
  {	
    return numEntries;	
  }

  // changes size of list
  bool Resize(unsigned int count)
  {
    if(listSize < count)
    {
      if(!ChangeSpace(count))
        return false;
    } 
    else
    {
      numEntries = count;
      if(!ChangeSpace(count))
        return false;
    }
    return true;
  }

  // resets number of currently used elements (not the actual list-size!!)
  void Clear()
  {	
    numEntries = 0;	
  }

  // frees elements of list
  void Erase()
  {
    numEntries = 0;
    listSize = 0; 
    if(entries)
      free(entries);
    entries = NULL;
  }

  // performs a qsort on elements, based on passed compare-function
  void Sort(int(__cdecl *compare)(const void*, const void*))
  {
    qsort(entries, numEntries, sizeof(T), compare);
  }

protected:
  // change memory-size of list
  bool ChangeSpace(unsigned int newSize)
  {
    T *newEntries = (T*)realloc(entries, sizeof(T)*newSize); 
    if((!newEntries) && (newSize > 0))
      return false;
    entries = newEntries;
    listSize = newSize;
    return true;
  }

  T *entries; // elements of list
  unsigned int numEntries; // number of currently used elements
  unsigned int listSize; // overall size of list (memory-size)

};

#endif	