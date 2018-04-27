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

#ifndef KLFILESYSTEM_H
#define KLFILESYSTEM_H

typedef int klFileTimeStamp;

class klFileName : public std::string {
public:
    klFileName(const std::string &name) : std::string(name) {}

    /* Returns extension WITHOUT the leading '.' */
    std::string getExt(void) {
        size_t dot = this->find_last_of('.');
        return this->substr(dot+1,this->size()-dot-1);
    }

    /* Returns the name of the file (without directory or extension info) */
    std::string getName(void) {
        size_t dot = this->find_last_of('.');
        size_t dirsep = this->find_last_of("\\/");
        return this->substr(dirsep+1,dot-dirsep-1);
    }

    /* Returns directory of the file (without tailing slashes) */
    std::string getFolder(void) {
        size_t dirsep = this->find_last_of("\\/");
        return this->substr(0,dirsep);
    }
};

class klStringStream {
    int lineNo;
    std::istream &stream;

    static inline bool klStringStream::isOperatorToken(char c) {
        return ( c=='/' || c==',' || c=='(' || c==')' || c==':' );
    }

public:

    klStringStream( std::istream &_stream ) : stream(_stream), lineNo(0) {}

    bool getToken(char *token);         // Returns false if EOF (and empty token)
    bool expectToken(const char *name); // Returns true if token was read, logs an error if not 
    void skipLine();                    // Skip till the end of the line
    int getLineNo() { return lineNo; }  // Get the current line number
    void skipCompound();                // skips till the next '}' (handles nested {}'s correctly)

    // Use these for manual parsing
    int isEof() { return stream.eof(); }
    int getChar() { return stream.get(); }
    void unGetChar(int c) { stream.unget(); }
    size_t tell(void) { return stream.tellg(); }
};

class klFileSystem {
    std::vector<std::string> paths;
    void *findHandle;
    std::string wildDir;
public: 

    klFileSystem(void);

    /**
        Adds an additional path to search for files.
    **/
    void addPath(const char *pathName);

    /**
        Open a file with the given name (relative to the base directory)
         * If the file is not found null is returned.
         * If canSeek is true the stream will be seekable otherwise
           seekability is not guaranteed.
         * Delete the returned stream to close the file.
         * If non-null timestamp is initialized with the file's modification time.
    **/
    std::istream *openFile(const char *fileName, klFileTimeStamp *timeStamp = NULL);

    /**
        Open a text file, returns a null terminated c-str.
         * memory should be freed with delete [].
         * If not found or empty file, NULL is returned.
    **/
    char *openText(const char *fileName);

    /**
        Find first/next file matching the given wildcard.
        Returns null when finished finding files.
    **/
    std::string findFirst(const char *wildCard);
    std::string findNext();
};

extern klFileSystem fileSystem;

#endif KLFILESYSTEM_H