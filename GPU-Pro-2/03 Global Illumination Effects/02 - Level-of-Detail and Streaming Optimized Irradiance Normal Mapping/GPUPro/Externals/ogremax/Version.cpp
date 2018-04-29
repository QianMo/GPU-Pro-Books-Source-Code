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


//Includes---------------------------------------------------------------------
#include "Version.hpp"
#include <OgreStringConverter.h>

using namespace Ogre;
using namespace OgreMax;


//Implementation---------------------------------------------------------------
Version::Version()
{
    for (int index = 0; index < MAX_COMPONENTS; index++)
        this->components[index] = 0;
}

Version::Version(int major, int minor, int patch)
{
    this->components[MAJOR] = major;
    this->components[MINOR] = minor;
    this->components[PATCH] = patch;
}

Version::Version(const String& version)
{
    size_t length = version.length();
    size_t offset = 0;
    size_t foundAt;
    String component;

    int index = 0;
    while (index < MAX_COMPONENTS && offset < length)
    {
        //Extract the current component
        foundAt = version.find('.', offset);
        component = version.substr(offset);
        this->components[index++] = StringConverter::parseInt(component);

        //Break out if there is no next '.'
        if (foundAt == String::npos)
            break;

        //Move past the next '.'
        offset = foundAt + 1;
    }

    for (; index < MAX_COMPONENTS; index++)
        this->components[index] = 0;
}

int Version::GetMajor() const
{
    return this->components[MAJOR];
}

int Version::GetMinor() const
{
    return this->components[MINOR];
}

int Version::GetPatch() const
{
    return this->components[PATCH];
}

void Version::ToString(String& text) const
{
    StringUtil::StrStreamType versionText;

    //Find the last non-zero component
    int lastNonzeroComponent = -1;
    for (int index = MAX_COMPONENTS - 1; index >= 0; index--)
    {
        if (this->components[index] != 0)
        {
            lastNonzeroComponent = index;
            break;
        }
    }

    //Output everything up to the last non-zero component
    if (lastNonzeroComponent >= 0)
    {
        for (int index = 0; index <= lastNonzeroComponent; index++)
        {
            versionText << this->components[index];
            if (index < lastNonzeroComponent)
                versionText << ".";
        }
    }
    else
    {
        //All components are zero
        versionText << "0";
    }

    text = versionText.str();
}

String Version::ToString() const
{
    String text;
    ToString(text);
    return text;
}

int Version::ToInt() const
{
    int version = 0;
    int multiplier = 1;
    for (int index = 0; index < MAX_COMPONENTS; index++)
    {
        version += this->components[index] * multiplier;
        multiplier *= 100;
    }
    return version;
}

int Version::Compare(const Version& version1, const Version& version2)
{
    return version1.ToInt() - version2.ToInt();
}
