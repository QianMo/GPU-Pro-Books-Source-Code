/**
 ****************************************************************************
 * <P> XML.c - implementation file for basic XML parser written in ANSI C++
 * for portability. It works by using recursion and a node tree for breaking
 * down the elements of an XML document.  </P>
 *
 * @version     V2.20
 * @author      Frank Vanden Berghen
 *
 * BSD license:
 * Copyright (c) 2002, Frank Vanden Berghen
 * All rights reserved.
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other renditions provided with the distribution.
 *     * Neither the name of the Frank Vanden Berghen nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE REGENTS AND CONTRIBUTORS BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ****************************************************************************
 */
#ifndef __INCLUDE_XML_NODE__
#define __INCLUDE_XML_NODE__

#include <stdlib.h>

#ifdef _UNICODE
// If you comment the next "define" line then the library will never "switch to" _UNICODE (wchar_t*) mode (16/32 bits per characters).
// This is useful when you get error messages like:
//    'XMLNode::openFileHelper' : cannot convert parameter 2 from 'const char [5]' to 'const wchar_t *'
// The _XMLUNICODE preprocessor variable force the XMLParser library into either utf16/32-mode (the proprocessor variable
// must be defined) or utf8-mode(the pre-processor variable must be undefined).
#define _XMLUNICODE
#endif

#if defined(WIN32) || defined(UNDER_CE)
// comment the next line if you are under windows and the compiler is not Microsoft Visual Studio (6.0 or .NET)
#define _XMLWINDOWS

#ifdef _USE_XMLPARSER_DLL
#ifdef _DLL_EXPORTS_
#define DLLENTRY __declspec(dllexport)
#else
#define DLLENTRY __declspec(dllimport)
#endif
#else
#define DLLENTRY
#endif

#endif

// uncomment the next line if you want no support for wchar_t* (no need for the <wchar.h> or <tchar.h> libraries anymore to compile)
//#define XML_NO_WIDE_CHAR

#ifdef XML_NO_WIDE_CHAR
#undef _XMLWINDOWS
#undef _XMLUNICODE
#endif

#ifdef _XMLWINDOWS
//#include <tchar.h>
#else
#define DLLENTRY
#ifndef XML_NO_WIDE_CHAR
#include <wchar.h> // to have 'wcsrtombs' for ANSI version
                   // to have 'mbsrtowcs' for UNICODE version
#endif
#endif

// Some common types for char set portable code
#ifdef _XMLUNICODE
    #ifndef _T
        #define _T(c) L ## c
    #endif
    #define XMLCSTR const wchar_t *
    #define XMLSTR  wchar_t *
    #define XMLCHAR wchar_t
#else
    #ifndef _T
        #define _T(c) c
    #endif
    #define XMLCSTR const char *
    #define XMLSTR  char *
    #define XMLCHAR char
#endif
#ifndef FALSE
    #define FALSE 0
#endif /* FALSE */
#ifndef TRUE
    #define TRUE 1
#endif /* TRUE */


// Enumeration for XML parse errors.
typedef enum XMLError
{
    eXMLErrorNone = 0,
    eXMLErrorMissingEndTag,
    eXMLErrorEmpty,
    eXMLErrorFirstNotStartTag,
    eXMLErrorMissingTagName,
    eXMLErrorMissingEndTagName,
    eXMLErrorNoMatchingQuote,
    eXMLErrorUnmatchedEndTag,
    eXMLErrorUnmatchedEndClearTag,
    eXMLErrorUnexpectedToken,
    eXMLErrorInvalidTag,
    eXMLErrorNoElements,
    eXMLErrorFileNotFound,
    eXMLErrorFirstTagNotFound,
    eXMLErrorUnknownCharacterEntity,
    eXMLErrorCharConversionError,
    eXMLErrorCannotOpenWriteFile,
    eXMLErrorCannotWriteFile,

    eXMLErrorBase64DataSizeIsNotMultipleOf4,
    eXMLErrorBase64DecodeIllegalCharacter,
    eXMLErrorBase64DecodeTruncatedData,
    eXMLErrorBase64DecodeBufferTooSmall
} XMLError;

// Enumeration used to manage type of data. Use in conjunction with structure XMLNodeContents
typedef enum XMLElementType
{
    eNodeChild=0,
    eNodeAttribute=1,
    eNodeText=2,
    eNodeClear=3,
    eNodeNULL=4
} XMLElementType;

// Structure used to obtain error details if the parse fails.
typedef struct XMLResults
{
    enum XMLError error;
    int  nLine,nColumn;
} XMLResults;

// Structure for XML clear (unformatted) node (usually comments)
typedef struct {
    XMLCSTR lpszValue; XMLCSTR lpszOpenTag; XMLCSTR lpszCloseTag;
} XMLClear;

// Structure for XML attribute.
typedef struct {
    XMLCSTR lpszName; XMLCSTR lpszValue;
} XMLAttribute;

// Structure for XML clear tags.
typedef struct {
    XMLCSTR lpszOpen; int openTagLen; XMLCSTR lpszClose;
} ALLXMLClearTag;

struct XMLNodeContents;

typedef class DLLENTRY XMLNode
{
  private:

    struct XMLNodeDataTag;

    // protected constructors: use one of these four methods to get your first instance of XMLNode:
    //  - parseString
    //  - parseFile
    //  - openFileHelper
    //  - createXMLTopNode
    XMLNode(struct XMLNodeDataTag *pParent, XMLCSTR lpszName, char isDeclaration);
    XMLNode(struct XMLNodeDataTag *p);

  public:

    // You can create your first instance of XMLNode with these 4 functions:
    // (see complete explanation of parameters below)

    static XMLNode createXMLTopNode(XMLCSTR lpszName, char isDeclaration=FALSE);
    static XMLNode parseString   (XMLCSTR  lpXMLString, XMLCSTR tag=NULL, XMLResults *pResults=NULL);
    static XMLNode parseFile     (XMLCSTR     filename, XMLCSTR tag=NULL, XMLResults *pResults=NULL);
    static XMLNode openFileHelper(XMLCSTR     filename, XMLCSTR tag=NULL                           );

    // The tag parameter should be the name of the first tag inside the XML file.
    // If the tag parameter is omitted, the 3 functions return a node that represents
    // the head of the xml document including the declaration term (<? ... ?>).

    // The "openFileHelper" reports to the screen all the warnings & errors that occurred during
    // parsing of the XML file. Since each application has its own way to report and deal with errors,
    // you should rather use the "parseFile" function to parse XML files and program yourself thereafter
    // an "error reporting" tailored for your needs (instead of using the very crude "error reporting"
    // mechanism included inside the "openFileHelper" function).

    // If the XML document is corrupted:
    //   * The "openFileHelper" method will:
    //         - display an error message on the console (or inside a messageBox for windows).
    //         - stop execution (exit).
    //     I suggest that you write your own "openFileHelper" method tailored to your needs.
    //   * The 2 other methods will initialize the "pResults" variable with some information that
    //     can be used to trace the error.
    //   * If you still want to parse the file, you can use the APPROXIMATE_PARSING option as
    //     explained inside the note at the beginning of the "xmlParser.cpp" file.
    // You can have a user-friendly explanation of the parsing error with this function:
    static XMLCSTR getError(XMLError error);
    static XMLCSTR getVersion();
    static ALLXMLClearTag* getClearTagTable();

    XMLCSTR getName() const;                                         // name of the node
    XMLCSTR getText(int i=0) const;                                  // return ith text field
    int nText() const;                                               // nbr of text field
    XMLNode getParentNode() const;                                   // return the parent node
    XMLNode getChildNode(int i=0) const;                             // return ith child node
    XMLNode getChildNode(XMLCSTR name, int i)  const;                // return ith child node with specific name
                                                                     //     (return an empty node if failing)
    XMLNode getChildNode(XMLCSTR name, int *i=NULL) const;           // return next child node with specific name
                                                                     //     (return an empty node if failing)
	XMLNode operator/(XMLCSTR name);
	XMLNode getChildNodeWithAttribute(XMLCSTR tagName,               // return child node with specific name/attribute
                                      XMLCSTR attributeName,         //     (return an empty node if failing)
                                      XMLCSTR attributeValue=NULL,   //
                                      int *i=NULL)  const;           //
    int nChildNode(XMLCSTR name) const;                              // return the number of child node with specific name
    int nChildNode() const;                                          // nbr of child node
    XMLAttribute getAttribute(int i=0) const;                        // return ith attribute
    XMLCSTR      getAttributeName(int i=0) const;                    // return ith attribute name
    XMLCSTR      getAttributeValue(int i=0) const;                   // return ith attribute value
    char  isAttributeSet(XMLCSTR name) const;                        // test if an attribute with a specific name is given
    XMLCSTR getAttribute(XMLCSTR name, int i) const;                 // return ith attribute content with specific name
                                                                     //     (return a NULL if failing)
    XMLCSTR getAttribute(XMLCSTR name, int *i=NULL) const;           // return next attribute content with specific name
                                                                     //     (return a NULL if failing)
	XMLCSTR operator|(XMLCSTR name);

	std::string readString(XMLCSTR name);
	std::wstring readWString(XMLCSTR name);
	double readDouble(XMLCSTR name, double defaultValue=0.0 );
	long readLong(XMLCSTR name, long defaultValue=0 );
	bool readBool(XMLCSTR name, bool defaultValue=false );
	D3DXVECTOR3 readVector(XMLCSTR name, const D3DXVECTOR3& defaultValue=D3DXVECTOR3(0, 0, 0) );
	D3DXVECTOR4 readVector4(XMLCSTR name, const D3DXVECTOR4& defaultValue=D3DXVECTOR4(1, 1, 1, 1) );
	DXGI_FORMAT readFormat(XMLCSTR name, DXGI_FORMAT defaultValue=DXGI_FORMAT_UNKNOWN);
	D3D10_USAGE readUsage(XMLCSTR name, D3D10_USAGE defaultValue=D3D10_USAGE_DEFAULT);
	D3D10_RTV_DIMENSION readRenderTargetViewDimension(XMLCSTR name, D3D10_RTV_DIMENSION defaultValue=D3D10_RTV_DIMENSION_UNKNOWN);
	D3D10_SRV_DIMENSION readShaderResourceViewDimension(XMLCSTR name, D3D10_SRV_DIMENSION defaultValue=D3D10_SRV_DIMENSION_UNKNOWN);
	D3D10_BIND_FLAG readBindFlag(XMLCSTR name);
	D3D10_CPU_ACCESS_FLAG readCPUAccessFlag(XMLCSTR name);
	NxVec3 readNxVec3(XMLCSTR name, const NxVec3& defaultValue=NxVec3(0, 0, 0) );
	NxQuat readNxQuat(XMLCSTR name);
	void dxTraceNode(const wchar_t* message);
	const char* getFilename();

    int nAttribute() const;                                          // nbr of attribute
    XMLClear getClear(int i=0) const;                                // return ith clear field (comments)
    int nClear() const;                                              // nbr of clear field
    XMLSTR createXMLString(int nFormat=1, int *pnSize=NULL) const;   // create XML string starting from current XMLNode
                                                                     // if nFormat==0, no formatting is required
                                                                     // otherwise this returns an user friendly XML string from a
                                                                     // given element with appropriate white spaces and carriage returns.
                                                                     // if pnSize is given it returns the size in character of the string.
    XMLError writeToFile(XMLCSTR filename, const char *encoding=NULL, char nFormat=1) const;
                                                                     // save the content of an xmlNode inside a file.
                                                                     // the nFormat parameter has the same meaning as in the
                                                                     // createXMLString function. If "strictUTF8Parsing=1", the
                                                                     // the encoding parameter is ignored and always set to
                                                                     // "utf-8". If "_XMLUNICODE=1", the encoding parameter is
                                                                     // ignored and always set to "utf-16".
    XMLNodeContents enumContents(int i) const;                       // enumerate all the different contents (attribute,child,text,
                                                                     //     clear) of the current XMLNode. The order is reflecting
                                                                     //     the order of the original file/string.
                                                                     //     NOTE: 0 <= i < nElement();
    int nElement() const;                                            // nbr of different contents for current node
    char isEmpty() const;                                            // is this node Empty?
    char isDeclaration() const;                                      // is this node a declaration <? .... ?>

// to allow shallow/fast copy:
    ~XMLNode();
    XMLNode(const XMLNode &A);
    XMLNode& operator=( const XMLNode& A );

    XMLNode(): d(NULL){ };
    static XMLNode emptyXMLNode;
    static XMLClear emptyXMLClear;
    static XMLAttribute emptyXMLAttribute;

    // The following functions allows you to create from scratch (or update) a XMLNode structure
    // Start by creating your top node with the "createXMLTopNode" function and then add new nodes with the "addChild" function.
    // The parameter 'pos' gives the position where the childNode, the text or the XMLClearTag will be inserted.
    // The default value (pos=-1) inserts at the end. The value (pos=0) insert at the beginning (Insertion at the beginning is slower than at the end).
    // REMARK: 0 <= pos < nChild()+nText()+nClear()
    XMLNode       addChild(XMLCSTR lpszName, char isDeclaration=FALSE, int pos=-1);
    XMLAttribute *addAttribute(XMLCSTR lpszName, XMLCSTR lpszValuev);
    XMLCSTR       addText(XMLCSTR lpszValue, int pos=-1);
    XMLClear     *addClear(XMLCSTR lpszValue, XMLCSTR lpszOpen=NULL, XMLCSTR lpszClose=NULL, int pos=-1);
                                                                    // default values: lpszOpen=XMLNode::getClearTagTable()->lpszOpen;
                                                                    //                 lpszClose=XMLNode::getClearTagTable()->lpszClose;
    XMLNode       addChild(XMLNode nodeToAdd, int pos=-1);          // If the "nodeToAdd" has some parents, it will be detached
                                                                    // from it's parents before being attached to the current XMLNode
    // Some update functions:
    XMLCSTR       updateName(XMLCSTR lpszName);                                                    // change node's name
    XMLAttribute *updateAttribute(XMLAttribute *newAttribute, XMLAttribute *oldAttribute);         // if the attribute to update is missing, a new one will be added
    XMLAttribute *updateAttribute(XMLCSTR lpszNewValue, XMLCSTR lpszNewName=NULL,int i=0);         // if the attribute to update is missing, a new one will be added
    XMLAttribute *updateAttribute(XMLCSTR lpszNewValue, XMLCSTR lpszNewName,XMLCSTR lpszOldName);  // set lpszNewName=NULL if you don't want to change the name of the attribute
                                                                                                   // if the attribute to update is missing, a new one will be added
    XMLCSTR       updateText(XMLCSTR lpszNewValue, int i=0);                                       // if the text to update is missing, a new one will be added
    XMLCSTR       updateText(XMLCSTR lpszNewValue, XMLCSTR lpszOldValue);                          // if the text to update is missing, a new one will be added
    XMLClear     *updateClear(XMLCSTR lpszNewContent, int i=0);                                    // if the clearTag to update is missing, a new one will be added
    XMLClear     *updateClear(XMLClear *newP,XMLClear *oldP);                                      // if the clearTag to update is missing, a new one will be added
    XMLClear     *updateClear(XMLCSTR lpszNewValue, XMLCSTR lpszOldValue);                         // if the clearTag to update is missing, a new one will be added

    // Some deletion functions:
    void deleteNodeContent(char force=0);  // delete the content of this XMLNode and the subtree.
                                           // if force=0, while (references to this node still exist), no memory free occurs
                                           // if force=1, always delete the content of this XMLNode and the subtree and free associated memory
    void deleteAttribute(XMLCSTR lpszName);
    void deleteAttribute(int i=0);
    void deleteAttribute(XMLAttribute *anAttribute);
    void deleteText(int i=0);
    void deleteText(XMLCSTR lpszValue);
    void deleteClear(int i=0);
    void deleteClear(XMLClear *p);
    void deleteClear(XMLCSTR lpszValue);

    // The strings given as parameters for the following add and update methods (all these methods have
    // a name with the postfix "_WOSD" that means "WithOut String Duplication" ) will be free'd by the
    // XMLNode class. For example, it means that this is incorrect:
    //    xNode.addText_WOSD("foo");
    //    xNode.updateAttribute_WOSD("#newcolor" ,NULL,"color");
    // In opposition, this is correct:
    //    xNode.addText("foo");
    //    xNode.addText_WOSD(stringDup("foo"));
    //    xNode.updateAttribute("#newcolor" ,NULL,"color");
    //    xNode.updateAttribute_WOSD(stringDup("#newcolor"),NULL,"color");
    // Typically, you will never do:
    //    char *b=(char*)malloc(...);
    //    xNode.addText(b);
    //    free(b);
    // ... but rather:
    //    char *b=(char*)malloc(...);
    //    xNode.addText_WOSD(b);
    //    ('free(b)' is performed by the XMLNode class)

    static XMLNode createXMLTopNode_WOSD(XMLCSTR lpszName, char isDeclaration=FALSE);
    XMLNode        addChild_WOSD(XMLCSTR lpszName, char isDeclaration=FALSE, int pos=-1);
    XMLAttribute  *addAttribute_WOSD(XMLCSTR lpszName, XMLCSTR lpszValue);
    XMLCSTR        addText_WOSD(XMLCSTR lpszValue, int pos=-1);
    XMLClear      *addClear_WOSD(XMLCSTR lpszValue, XMLCSTR lpszOpen=NULL, XMLCSTR lpszClose=NULL, int pos=-1);

    XMLCSTR        updateName_WOSD(XMLCSTR lpszName);
    XMLAttribute  *updateAttribute_WOSD(XMLAttribute *newAttribute, XMLAttribute *oldAttribute);
    XMLAttribute  *updateAttribute_WOSD(XMLCSTR lpszNewValue, XMLCSTR lpszNewName=NULL,int i=0);
    XMLAttribute  *updateAttribute_WOSD(XMLCSTR lpszNewValue, XMLCSTR lpszNewName,XMLCSTR lpszOldName);
    XMLCSTR        updateText_WOSD(XMLCSTR lpszNewValue, int i=0);
    XMLCSTR        updateText_WOSD(XMLCSTR lpszNewValue, XMLCSTR lpszOldValue);
    XMLClear      *updateClear_WOSD(XMLCSTR lpszNewContent, int i=0);
    XMLClear      *updateClear_WOSD(XMLClear *newP,XMLClear *oldP);
    XMLClear      *updateClear_WOSD(XMLCSTR lpszNewValue, XMLCSTR lpszOldValue);

    // These are some useful functions when you want to insert a childNode, a text or a XMLClearTag in the
    // middle (at a specified position) of a XMLNode tree already constructed. The value returned by these
    // methods is to be used as last parameter (parameter 'pos') of addChild, addText or addClear.
    int positionOfText(int i=0) const;
    int positionOfText(XMLCSTR lpszValue) const;
    int positionOfClear(int i=0) const;
    int positionOfClear(XMLCSTR lpszValue) const;
    int positionOfClear(XMLClear *a) const;
    int positionOfChildNode(int i=0) const;
    int positionOfChildNode(XMLNode x) const;
    int positionOfChildNode(XMLCSTR name, int i=0) const; // return the position of the ith childNode with the specified name
                                                          // if (name==NULL) return the position of the ith childNode

    // The setGlobalOptions function allows you to change two global parameters that affect string&file
    // parsing. First of all, you most-probably will never have to change these 2 global parameters.
    // About the "guessUnicodeChars" parameter:
    //     If "guessUnicodeChars=1" and if this library is compiled in UNICODE mode, then the
    //     "parseFile" and "openFileHelper" functions will test if the file contains ASCII
    //     characters. If this is the case, then the file will be loaded and converted in memory to
    //     UNICODE before being parsed. If "guessUnicodeChars=0", no conversion will
    //     be performed.
    //
    //     If "guessUnicodeChars=1" and if this library is compiled in ASCII/UTF8 mode, then the
    //     "parseFile" and "openFileHelper" functions will test if the file contains UNICODE
    //     characters. If this is the case, then the file will be loaded and converted in memory to
    //     ASCII/UTF8 before being parsed. If "guessUnicodeChars=0", no conversion will
    //     be performed
    //
    //     Sometime, it's useful to set "guessUnicodeChars=0" to disable any conversion
    //     because the test to detect the file-type (ASCII/UTF8 or UNICODE) may fail (rarely).
    //
    // About the "strictUTF8Parsing" parameter:
    //     If "strictUTF8Parsing=0" then we assume that all characters have the same length of 1 byte.
    //     If "strictUTF8Parsing=1" then the characters have different lengths (from 1 byte to 4 bytes)
    //     depending on the content of the first byte of the character.
    // About the "dropWhiteSpace" parameter:
    //

    static void setGlobalOptions(char guessUnicodeChars=1, char strictUTF8Parsing=1, char dropWhiteSpace=1);

    // The next function try to guess if the character encoding is UTF-8. You most-probably will never
    // have to use this function. It then returns the appropriate value of the global parameter
    // "strictUTF8Parsing" described above. The guess is based on the content of a buffer of length
    // "bufLen" bytes that contains the first bytes (minimum 25 bytes; 200 bytes is a good value) of the
    // file to be parsed. The "openFileHelper" function is using this function to automatically compute
    // the value of the "strictUTF8Parsing" global parameter. There are several heuristics used to do the
    // guess. One of the heuristic is based on the "encoding" attribute. The original XML specifications
    // forbids to use this attribute to do the guess but you can still use it if you set
    // "useXMLEncodingAttribute" to 1 (this is the default behavior and the behavior of most parsers).

    static char guessUTF8ParsingParameterValue(void *buffer, int bufLen, char useXMLEncodingAttribute=1);

  private:

// these are functions and structures used internally by the XMLNode class (don't bother about them):

      typedef struct XMLNodeDataTag // to allow shallow copy and "intelligent/smart" pointers (automatic delete):
      {
          XMLCSTR                lpszName;        // Element name (=NULL if root)
          int                    nChild,          // Number of child nodes
                                 nText,           // Number of text fields
                                 nClear,          // Number of Clear fields (comments)
                                 nAttribute;      // Number of attributes
          char                   isDeclaration;   // Whether node is an XML declaration - '<?xml ?>'
          struct XMLNodeDataTag  *pParent;        // Pointer to parent element (=NULL if root)
          XMLNode                *pChild;         // Array of child nodes
          XMLCSTR                *pText;          // Array of text fields
          XMLClear               *pClear;         // Array of clear fields
          XMLAttribute           *pAttribute;     // Array of attributes
          int                    *pOrder;         // order of the child_nodes,text_fields,clear_fields
          int                    ref_count;       // for garbage collection (smart pointers)

		  int					fileLineNumber;
		  char*					fileName;
      } XMLNodeData;
      XMLNodeData *d;

      char parseClearTag(void *px, ALLXMLClearTag *pa);
      char maybeAddTxT(void *pa, XMLCSTR tokenPStr);
      int ParseXMLElement(void *pXML);
      void *addToOrder(int *_pos, int nc, void *p, int size, XMLElementType xtype);
      int indexText(XMLCSTR lpszValue) const;
      int indexClear(XMLCSTR lpszValue) const;
      static inline int findPosition(XMLNodeData *d, int index, XMLElementType xtype);
      static int CreateXMLStringR(XMLNodeData *pEntry, XMLSTR lpszMarker, int nFormat);
      static int removeOrderElement(XMLNodeData *d, XMLElementType t, int index);
      static void exactMemory(XMLNodeData *d);
      static int detachFromParent(XMLNodeData *d);

  public:
	  int getFileLineNumber(){return d->fileLineNumber;}
} XMLNode;

// This structure is given by the function "enumContents".
typedef struct XMLNodeContents
{
    // This dictates what's the content of the XMLNodeContent
    enum XMLElementType type;
    // should be an union to access the appropriate data.
    // compiler does not allow union of object with constructor... too bad.
    XMLNode child;
    XMLAttribute attrib;
    XMLCSTR text;
    XMLClear clear;

} XMLNodeContents;

DLLENTRY void free_XMLDLL(void *t); // {free(t);}

// Duplicate (copy in a new allocated buffer) the source string. This is
// a very handy function when used with all the "XMLNode::*_WOSD" functions.
// (If (cbData!=0) then cbData is the number of chars to duplicate)
DLLENTRY XMLSTR stringDup(XMLCSTR source, int cbData=0);

// The 3 following functions are processing strings so that all the characters
// &,",',<,> are replaced by their XML equivalent: &amp;, &quot;, &apos;, &lt;, &gt;.
// These 3 functions are useful when creating from scratch an XML file using the
// "printf", "fprintf", "cout",... functions. If you are creating from scratch an
// XML file using the provided XMLNode class you cannot use these functions (the
// XMLNode class does the processing job for you during rendering). The second
// function ("toXMLStringFast") allows you to re-use the same output buffer
// for all the conversions so that only a few memory allocations are performed.
// If the output buffer is too small to contain thee resulting string, it will
// be enlarged.
DLLENTRY XMLSTR toXMLString(XMLCSTR source);
DLLENTRY XMLSTR toXMLStringFast(XMLSTR *destBuffer,int *destSz, XMLCSTR source);

// you should not use this one (there is a possibility of "destination-buffer-overflow"):
DLLENTRY XMLSTR toXMLString(XMLSTR dest,XMLCSTR source);

// Below is a class that allows you to include any binary data (images, sounds,...)
// into an XML document using "Base64 encoding". This class is completely
// separated from the rest of the xmlParser library and can be removed without any problem.
// To include some binary data into an XML file, you must convert the binary data into
// standard text (using "encode"). To retrieve the original binary data from the
// b64-encoded text included inside the XML file use "decode". Alternatively, these
// functions can also be used to "encrypt/decrypt" some critical data contained inside
// the XML.

class DLLENTRY XMLParserBase64Tool
{
public:
    XMLParserBase64Tool(): buf(NULL),buflen(0){}
    ~XMLParserBase64Tool();

    void freeBuffer();

    // returns the length of the base64 string that encodes a data buffer of size inBufLen bytes.
    // If "formatted" parameter is true, some space will be reserved for a carriage-return every 72 chars.
    static int encodeLength(int inBufLen, char formatted=0);

    // The "base64Encode" function returns a string containing the base64 encoding of "inByteLen" bytes
    // from "inByteBuf". If "formatted" parameter is true, then there will be a carriage-return every 72 chars.
    // The string will be free'd when the XMLParserBase64Tool object is deleted.
    // All returned strings are sharing the same memory space.
    XMLSTR encode(unsigned char *inByteBuf, unsigned int inByteLen, char formatted=0);

    // returns the number of bytes which will be decoded from "inString".
    static unsigned int decodeSize(XMLCSTR inString, XMLError *xe=NULL);

    // returns a pointer to a buffer containing the binary data decoded from "inString"
    // If "inString" is malformed NULL will be returned
    // The output buffer will be free'd when the XMLParserBase64Tool object is deleted.
    // All output buffer are sharing the same memory space.
    unsigned char* decode(XMLCSTR inString, int *outByteLen=NULL, XMLError *xe=NULL);

    // The next function is deprecated.
    // decodes data from "inString" to "outByteBuf". You need to provide the size (in byte) of "outByteBuf"
    // in "inMaxByteOutBuflen". If "outByteBuf" is not large enough or if data is malformed, then "FALSE"
    // will be returned; otherwise "TRUE".
    static unsigned char decode(XMLCSTR inString, unsigned char *outByteBuf, int inMaxByteOutBuflen, XMLError *xe=NULL);

private:
    void *buf;
    int buflen;
    void alloc(int newsize);
};

#define EggXMLERR(node, x) \
{	\
		std::wostringstream msgs;	\
		msgs << x ;	\
		node.dxTraceNode(msgs.str().c_str()); \
}

#endif
