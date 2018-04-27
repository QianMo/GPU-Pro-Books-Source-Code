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

#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <string.h>
#include <assert.h>

#include "maths.h"
#include "vectors.h"
#include "matrix.h"

/* Php style explode*/
inline std::vector<std::string> explode( const std::string & in, const std::string & delim) {
    typedef std::string::size_type size_type;
    
    const size_type delim_len = delim.length();
    std::vector<std::string> result;
    size_type i = 0;
    size_type j;

    while(true)
    {
        j = in.find(delim, i);
        result.push_back(in.substr(i, j-i)) ;
        if (j == std::string::npos)
        {
            // reached end of string...
            break;
        }
        i = j + delim_len;
    }

    return result;
}

/*
    Puts text in the console buffer (manually add \n if desired)
*/
void klPrint(const char *buffer);

/*
    Shows the error + exits
*/
void klFatalError(const char *format,...);

/*
    Shows the error
*/
void klError(const char *format,...);

/*
    Print to the log (does not need \n in strings)
*/
void klLog(const char *format,...);




/******* FIXME: Put these somewhere else? */

void klCheckGlErrors(void);

struct klCachedGlState {
    int currentTextureUnit;
};

extern klCachedGlState glState;