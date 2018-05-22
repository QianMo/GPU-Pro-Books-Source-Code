#ifndef CROWDMESH_H
#error "Do not include CrowdMesh.inl directly!"
#endif

inline CrowdGroup& CrowdMesh::GetGroup( unsigned int index )
{
	return m_groups[index];
}

inline unsigned int CrowdMesh::GetGroupCount( void ) const
{
	return m_groups.size();
}
