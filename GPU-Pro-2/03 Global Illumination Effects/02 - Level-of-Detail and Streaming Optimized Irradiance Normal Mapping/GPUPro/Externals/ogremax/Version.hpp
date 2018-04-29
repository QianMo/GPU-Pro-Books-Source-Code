/*
 * OgreMax Sample Viewer and Scene Loader - Ogre3D-based viewer and code for loading and displaying .scene files
 * Copyright 2010 AND Entertainment
 *
 * This code is available under the OgreMax Free License:
 *   -You may use this code for any purpose, commercial or non-commercial.
 *   -If distributing derived works (that use this source code) in binary or source code form, 
 *    you must give the following credit in your work's end-user documentation: 
 *        "Portions of this work provided by OgreMax (www.ogremax.com)"
 *
 * AND Entertainment assumes no responsibility for any harm caused by using this code.
 * 
 * The OgreMax Sample Viewer and Scene Loader were released at www.ogremax.com 
 */


#ifndef OgreMax_Version_INCLUDED
#define OgreMax_Version_INCLUDED


//Includes---------------------------------------------------------------------
#include <OgreString.h>
#include "OgreMaxPlatform.hpp"


//Classes----------------------------------------------------------------------
namespace OgreMax
{
    /**
     * A major/minor/build/revision number collection.
     * Use the Version class to simplify the storage and comparison of version numbers
     */
    class _OgreMaxExport Version
    {
    public:
        /** Default class constructor. Constructs a Version object with all zeroes. */
        Version();

        /**
         * Constructs a Version from the major, minor, build, and revision numbers.
         * All components should be less than 0
         * @param major [in] - Major version number.
         * @param minor [in] - Minor version number.
         * @param patch [in] - Patch number.
         */
        Version(int major, int minor = 0, int patch = 0);

        /**
         * Constructs a version by parsing it from a string
         * @param version [in] - The string to parse. The string should be of the form: 'xx.xx.xx'.
         * All components are optional.
         */
        Version(const Ogre::String& version);

        /**
         * Gets the major version number.
         */
        int GetMajor() const;

        /**
         * Gets the minor version number.
         */
        int GetMinor() const;

        /**
         * Gets the patch number.
         */
        int GetPatch() const;

        /**
         * Converts the version to a string.
         * @param text [out] - The string result.
         */
        void ToString(Ogre::String& text) const;

        /**
         * Converts the version to a string.
         */
        Ogre::String ToString() const;

        /**
         * Converts the version to an integer.
         * Each component can be no larger than 99 for this to work properly
         */
        int ToInt() const;

        //Comparison operators
        bool operator == (const Version& v) const {return Compare(*this, v) == 0;}
        bool operator < (const Version& v) const {return Compare(*this, v) < 0;}
        bool operator <= (const Version& v) const {return Compare(*this, v) <= 0;}
        bool operator > (const Version& v) const {return Compare(*this, v) > 0;}
        bool operator >= (const Version& v) const {return Compare(*this, v) >= 0;}

        /**
         * Compares two versions.
         * Function for comparing two Version objects, returning an
         * integer to indicate the result of the comparison.
         * @param version1 [in] - The first Version to compare.
         * @param version2 [in] - The second Version to compare to.
         * @return An integer indicating the result of the comparison.
         * This value is less than 0 if version1 is "older" than version2,
         * greater than 0 if version1 is "newer" than version2, and equal to
         * 0 if the two versions are the same.
         */
        static int Compare(const Version& version1, const Version& version2);

    private:
        enum
        {
            MAJOR,
            MINOR,
            PATCH,
            MAX_COMPONENTS
        };

        /** Major, minor, patch */
        int components[MAX_COMPONENTS];
    };

}

#endif
