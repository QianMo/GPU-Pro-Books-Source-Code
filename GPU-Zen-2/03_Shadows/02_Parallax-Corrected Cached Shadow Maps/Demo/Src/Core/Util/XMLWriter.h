#pragma once

#include <stdarg.h>
#include <vector>
#include "RapidXML/rapidxml.hpp"
#include "RapidXML/rapidxml_print.hpp"

class XMLWriter
{
public:
    typedef rapidxml::xml_node<> Node;

    XMLWriter(char* pData = NULL) : m_Stack(1, &m_Document), m_OK(true)
    {
        if(pData!=NULL)
        {
            try
            {
                m_Document.parse<0>(pData);
            }
            catch(rapidxml::parse_error&)
            {
                m_OK = false;
            }
        }
    }
    bool IsOK()
    {
        return m_OK;
    }
    void OpenTag(const char* pszName, const char* pszData = NULL)
    {
        char* pData = pszData!=NULL ? m_Document.allocate_string(pszData) : NULL;
        Node* pNode = m_Document.allocate_node(rapidxml::node_element, m_Document.allocate_string(pszName), pData);
        m_Stack.back()->append_node(pNode);
        m_Stack.push_back(pNode);
    }
    void CloseTag()
    {
        m_Stack.pop_back();
    }
    void AddAttribute(const char* pszName, const char* pszValue)
    {
        rapidxml::xml_attribute<>* pAttrib = m_Document.allocate_attribute(m_Document.allocate_string(pszName), m_Document.allocate_string(pszValue));
        m_Stack.back()->append_attribute(pAttrib);
    }
    void AddAttributeF(const char* pszName, const char* fmt, ...)
    {
        char strBuf[2048] = { };
        va_list argptr;
        va_start(argptr, fmt);
        vsnprintf(strBuf, sizeof(strBuf) - 1, fmt, argptr);
        va_end(argptr);
        AddAttribute(pszName, strBuf);
    }
    void Write(std::string& s)
    {
        s.append("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        rapidxml::print(std::back_inserter(s), m_Document, 0);
    }
    Node* GetFirstNode()
    {
        return m_Document.first_node();
    }

private:
    rapidxml::xml_document<> m_Document;
    std::vector<Node*> m_Stack;
    bool m_OK;
};
