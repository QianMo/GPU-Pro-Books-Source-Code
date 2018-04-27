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

#include "shared.h"
#include "FileSystem.h"

// For FindFirstFile etc...
#include <windows.h>

bool klStringStream::getToken(char *token) {
whitespace:
    int c = getChar();
    int nChars = 0;

    // First skip any leading white space
    while ( !isEof() ) {
        if( c != ' ' &&
            c != '\t' &&
            c != '\n' &&
            c != '\r' )
        {
            break;
        }
        if ( c == '\n' ) {
            lineNo++;
        }
        c = getChar();
    }

    // Handle whitespace at end of file
    if ( isEof() ) {
        return false;
    }

    // Is is a comment?
    if ( c == '/' ) {
        char fc = getChar();
        if ( fc == '/' ) {
            skipLine();
            goto whitespace;
        } else {
            unGetChar(fc);
        }
    }

    // Is is a quoted string?
    if ( c == '"' ) {
        c = getChar();
        while ( c != '"' && !isEof() ) {
            token[nChars] = c;
            nChars++;
            c = getChar();
        }
        token[nChars] = 0;
        return true;
    }

    // Is it one of the single tokens?
    if (isOperatorToken(c)) {
        token[nChars] = c;
        nChars++;
        token[nChars] = 0;
        return true;
    }

    // So it is regular now parse till whitespace or single tokens
    while ( !isEof() ) { 

        // A single char token unput it and return the found token
        if (isOperatorToken(c)) {
            unGetChar(c);
            token[nChars] = 0;
            return true;
        }

        // whitespace just return, leave the position on the first
        // whitespace after the token. (So skypline doesn't miss newline after a token)
        if( c == ' ' ||
            c == '\t' ||
            c == '\n' ||
            c == '\r' )
        {
            unGetChar(c);
            if ( c == '\n' ) {
                lineNo++;
            }
            token[nChars] = 0;
            return true;
        }

        token[nChars++] = c;

        c = getChar();
    }

    token[nChars] = 0;

    if ( isEof() ) {
        return false;
    }

    return true;
}

void klStringStream::skipLine() {
    int c = getChar();

    while ( !isEof() ) {
        if( c == '\n' || c == '\r' ) {
            if ( c == '\n' ) {
                lineNo++;
            }
            return;
        }

        c = getChar();
    }
}

bool klStringStream::expectToken(const char *name) {
    char token[128];
    getToken(token);
    if ( strcmp(token,name) != 0 ) {
        klError("line %i: Expected %s but found %s.", lineNo, name, token);
        return false;
    } else {
        return true;
    }
}

void klStringStream::skipCompound() {
    int c = getChar();
    if (c == '{') {
        c = getChar();
    } else if ( c == '}' ) {
        return;
    }

	while ( !isEof() && c != '}') {
        if (c == '{') {
			skipCompound();
        }
        c = getChar();
	}
}

klFileSystem::klFileSystem(void) {
    paths.push_back("./");
    findHandle = INVALID_HANDLE_VALUE;
}


void klFileSystem::addPath(const char *baseDir) {
    paths.push_back(baseDir);
}

std::istream *klFileSystem::openFile(const char *fileName, klFileTimeStamp *timeStamp) {
    std::ifstream *result;
    std::string fullName;

    result = new std::ifstream();
    for ( size_t i=0; i<paths.size(); i++ ) {
        std::string fullName = paths[i] + fileName;
        result->open(fullName.c_str(), std::ios_base::in | std::ios_base::binary);
        if ( result->good() ) {
            return result;
        }
    }

    delete result;
    return NULL;
}

char *klFileSystem::openText(const char *fileName) {
    std::istream *str = openFile(fileName);
    if ( str == NULL ) return NULL;

    str->seekg(0, std::ios_base::end);
    int len = str->tellg();
    str->seekg(0, std::ios_base::beg);

    if ( !len ) return NULL;

    char *result = new char [len+1];
    str->read(result,len);
    result[len] = 0;
    
    delete str;
    return result;
}

std::string klFileSystem::findFirst(const char *wildCard) {
    std::string wildPath = paths[0] + wildCard;
    klFileName fn(wildPath);
    wildDir = fn.getFolder()+"/";

    if ( findHandle != INVALID_HANDLE_VALUE ) {
        klFatalError("Recursive file find");
    }

    WIN32_FIND_DATA  findFileData;
    findHandle = FindFirstFile(wildPath.c_str(), &findFileData);
    if (findHandle == INVALID_HANDLE_VALUE) {
        return std::string();
    }

    return std::string(wildDir+findFileData.cFileName);
}

std::string  klFileSystem::findNext() {
    WIN32_FIND_DATA  findFileData;

    if ( FindNextFile(findHandle,&findFileData) ) {
        return std::string(wildDir+findFileData.cFileName);
    } else {
        FindClose(findHandle);
        findHandle = INVALID_HANDLE_VALUE;
        return std::string();
    }
}

klFileSystem fileSystem;