#ifndef CROWDGROUP_H
#error "Do not include CrowdGroup.inl directly!"
#endif

inline CrowdSection& CrowdGroup::GetSection( unsigned int index )
{
	return m_sections[index];
}

inline unsigned int CrowdGroup::GetSectionCount( void ) const
{
	return m_sections.size();
}
