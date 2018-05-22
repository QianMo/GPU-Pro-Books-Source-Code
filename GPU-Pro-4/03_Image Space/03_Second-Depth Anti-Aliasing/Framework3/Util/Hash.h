
/* * * * * * * * * * * * * Author's note * * * * * * * * * * * *\
*   _       _   _       _   _       _   _       _     _ _ _ _   *
*  |_|     |_| |_|     |_| |_|_   _|_| |_|     |_|  _|_|_|_|_|  *
*  |_|_ _ _|_| |_|     |_| |_|_|_|_|_| |_|     |_| |_|_ _ _     *
*  |_|_|_|_|_| |_|     |_| |_| |_| |_| |_|     |_|   |_|_|_|_   *
*  |_|     |_| |_|_ _ _|_| |_|     |_| |_|_ _ _|_|  _ _ _ _|_|  *
*  |_|     |_|   |_|_|_|   |_|     |_|   |_|_|_|   |_|_|_|_|    *
*                                                               *
*                     http://www.humus.name                     *
*                                                                *
* This file is a part of the work done by Humus. You are free to   *
* use the code in any way you like, modified, unmodified or copied   *
* into your own work. However, I expect you to respect these points:  *
*  - If you use this file and its contents unmodified, or use a major *
*    part of this file, please credit the author and leave this note. *
*  - For use in anything commercial, please request my approval.     *
*  - Share your work and ideas too as much as you can.             *
*                                                                *
\* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef _HASH_H_
#define _HASH_H_

#ifndef NULL
#define NULL 0
#endif

struct HashEntry {
	unsigned int *value;
	HashEntry *next;
	unsigned int index;
};


class Hash {
public:
	Hash(const unsigned int dim, const unsigned int entryCount, const unsigned int capasity){
		curr = mem = (unsigned char *) malloc(entryCount * sizeof(HashEntry *) + capasity * (sizeof(HashEntry) + sizeof(unsigned int) * dim));


		nDim = dim;
		count = 0;
		nEntries = entryCount;
		//entries = new HashEntry *[entryCount];
		entries = (HashEntry **) newMem(entryCount * sizeof(HashEntry *));

		memset(entries, 0, entryCount * sizeof(HashEntry *));
	}

	~Hash(){
		free(mem);
/*
		for (unsigned int i = 0; i < nEntries; i++){
			HashEntry *entry = entries[i];
			while (entry){
				HashEntry *nextEntry = entry->next;
				delete entry->value;
				delete entry;
				entry = nextEntry;
			}
		}
		delete entries;
*/
	}

	bool insert(const unsigned int *value, unsigned int *index){
		unsigned int hash = 0;//0xB3F05C27;
		unsigned int i = 0;
		do {
			hash += value[i];
			hash += (hash << 11);
			//hash ^= (hash >> 6);
			i++;
		} while (i < nDim);

		hash %= nEntries;

		HashEntry *entry = entries[hash];

		while (entry){
			if (memcmp(value, entry->value, nDim * sizeof(unsigned int)) == 0){
				*index = entry->index;
				return true;
			}

			entry = entry->next;
		}
		
		//HashEntry *newEntry = new HashEntry;
		//newEntry->value = new unsigned int[nDim];

		HashEntry *newEntry = (HashEntry *) newMem(sizeof(HashEntry));
		newEntry->value = (unsigned int *) newMem(sizeof(unsigned int) * nDim);


		memcpy(newEntry->value, value, nDim * sizeof(unsigned int));
		newEntry->index = count++;

		newEntry->next = entries[hash];
		entries[hash] = newEntry;

		*index = newEntry->index;
		return false;
	}

	unsigned int getCount() const { return count; }

protected:
	unsigned int nDim;
	unsigned int count;
	unsigned int nEntries;

	HashEntry **entries;



	void *newMem(const unsigned int size){
		unsigned char *rmem = curr;
		curr += size;
		return rmem;
	}

	unsigned char *mem, *curr;
};

#endif // _HASH_H_
