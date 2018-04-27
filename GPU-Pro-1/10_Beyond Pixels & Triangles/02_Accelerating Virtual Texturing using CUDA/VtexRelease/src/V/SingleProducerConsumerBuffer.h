/**
 *
 *  This software module was originally developed for research purposes,
 *  by Multimedia Lab at Ghent University (Belgium).
 *  Its performance may not be optimized for specific applications.
 *
 *  Those intending to use this software module in hardware or software products
 *  are advized that its use may infringe existing patents. The developers of 
 *  this software module, their companies, Ghent Universtity, nor Multimedia Lab 
 *  have any liability for use of this software module or modifications thereof.
 *
 *  Ghent University and Multimedia Lab (Belgium) retain full right to modify and
 *  use the code for their own purpose, assign or donate the code to a third
 *  party, and to inhibit third parties from using the code for their products. 
 *
 *  This copyright notice must be included in all copies or derivative works.
 *
 *  For information on its use, applications and associated permission for use,
 *  please contact Prof. Rik Van de Walle (rik.vandewalle@ugent.be). 
 *
 *  Detailed information on the activities of
 *  Ghent University Multimedia Lab can be found at
 *  http://multimedialab.elis.ugent.be/.
 *
 *  Copyright (c) Ghent University 2004-2009.
 *
 **/


/**
    Thread safe queue for a single producer consumer.
    While this can be implemented without any synhro primitives we use them so our
    threads actually sleep instead of spin or whatever when no items are available.

    Fixme: This is really crude and probably needs improvement...
*/
template <class Type> class SingleProducerConsumerBuffer {
    HANDLE fillSema;
    HANDLE emptySema;
    HANDLE mutexSema;
    int size;
    int bufferSize;
    Type *buffer;

public:

    SingleProducerConsumerBuffer(int bufferSize) {
        this->bufferSize = bufferSize;
        size = 0;
        buffer = (Type *)malloc(bufferSize * sizeof(Type));
        fillSema  = CreateSemaphore(NULL, 0, bufferSize,NULL);
        emptySema = CreateSemaphore(NULL, bufferSize, bufferSize,NULL);
        mutexSema  = CreateSemaphore(NULL, 1, 1,NULL);
    }

    ~SingleProducerConsumerBuffer(void) {
        free(buffer);
        CloseHandle(fillSema);
        CloseHandle(emptySema);
        CloseHandle(mutexSema);
    }

    /*
        Push an item into the buffer, sleeps if it is full.
    */
    void push(const Type &item) {
        WaitForSingleObject(emptySema,INFINITE);
        WaitForSingleObject(mutexSema,INFINITE);
        assert(size < bufferSize);
        buffer[size] = item;
        size++;
        ReleaseSemaphore(mutexSema, 1, NULL);
        ReleaseSemaphore(fillSema, 1, NULL);
    }

    /*
        Push an item into the buffer, returns false if it is full
        and the item could not be appended.
    */
    bool pushNonBl(const Type &item) {
        unsigned int res = WaitForSingleObject(emptySema,0);
        if ( res != WAIT_TIMEOUT ) { 
            WaitForSingleObject(mutexSema,INFINITE);
            assert(size < bufferSize);
            buffer[size] = item;
            size++;
            ReleaseSemaphore(mutexSema, 1, NULL);
            ReleaseSemaphore(fillSema, 1, NULL);
            return true;
        }
        return false;
    }

    /*
        Get an item from the buffer, sleeps untill an item is available.
    */
    Type pop(void) {
        WaitForSingleObject(fillSema,INFINITE);
        WaitForSingleObject(mutexSema,INFINITE);
        assert(size > 0);
        Type item = buffer[size-1];
        size--;
        ReleaseSemaphore(mutexSema, 1, NULL);
        ReleaseSemaphore(emptySema, 1, NULL);
        return item;
    }

    /*
        Try to get an item from the buffer, returns false if no item
        was available
    */
    bool popNonBl(Type &item) {
        unsigned int res = WaitForSingleObject(fillSema,0);
        if ( res != WAIT_TIMEOUT ) { 
            WaitForSingleObject(mutexSema,INFINITE);
            item = buffer[size-1];
            size--;
            ReleaseSemaphore(mutexSema, 1, NULL);
            ReleaseSemaphore(emptySema, 1, NULL);
            return true;
        }
        return false;
    }

    void sort(int (__cdecl * ptFuncCompare)(Type*, Type*)) {
        // hmmmmmm....
        WaitForSingleObject(mutexSema,INFINITE);
        qsort(buffer,size,sizeof(Type),(int (__cdecl *)(const void *,const void *))ptFuncCompare);
        ReleaseSemaphore(mutexSema, 1, NULL);
    }
};
