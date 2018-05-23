/*****************************************************/
/* breeze Engine Core Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#include "beCoreInternal/stdafx.h"
#include "beCore/beFileWatch.h"

#include <unordered_map>
#include <boost/ptr_container/ptr_map_adapter.hpp>

#include <lean/tags/noncopyable.h>
#include <lean/logging/log.h>
#include <lean/logging/win_errors.h>
#include <lean/io/filesystem.h>
#include <lean/smart/handle_guard.h>

#include <lean/concurrent/thread.h>
#include <lean/functional/callable.h>

#include <lean/concurrent/critical_section.h>
#include <lean/concurrent/event.h>

#include <lean/functional/algorithm.h>

namespace beCore
{

namespace
{

/// Directory.
class Directory : public lean::tags::noncopyable
{
protected:
	FileWatch::M *m_watch;

private:
	lean::utf8_string m_directory;

	/// File observer.
	struct FileObserverInfo
	{
		lean::utf8_string file;
		lean::uint8 lastRevision;
		FileObserver *pObserver;

		FileObserverInfo(const lean::utf8_ntri &file,
			lean::uint8 revision,
			FileObserver *pObserver)
				: file(file.to<lean::utf8_string>()),
				lastRevision(revision),
				pObserver(pObserver) { }
	};
	typedef std::unordered_multimap<utf8_string, FileObserverInfo> file_observer_map;
	file_observer_map m_fileObservers;

	/// Directory observer.
	typedef std::vector<DirectoryObserver*> directory_observer_vector;
	directory_observer_vector m_directoryObservers;

	lean::critical_section m_observerLock;

protected:
	/// Gets the revision of the given file.
	virtual lean::uint8 GetRevision(const lean::utf8_ntri &file) const = 0;

	/// Called when a file observer has been added.
	virtual void FileObserverAdded() = 0;
	/// Called when a directory observer has been added.
	virtual void DirectoryObserverAdded() = 0;

public:
	/// Constructor.
	Directory(FileWatch::M *watch, const utf8_ntri &directory);
	/// Destructor.
	virtual ~Directory();

	/// Adds the given observer to be called when the given file has been modified.
	bool AddObserver(const lean::utf8_ntri &file, FileObserver *pObserver);
	/// Removes the given observer, no longer to be called when the given file is modified.
	void RemoveObserver(const lean::utf8_ntri &file, FileObserver *pObserver);

	/// Adds the given observer to be called when this directory has been modified.
	bool AddObserver(DirectoryObserver *pObserver);
	// Removes the given observer, no longer to be called when this directory is modified.
	void RemoveObserver(DirectoryObserver *pObserver);

	/// Notifies file observers about modifications to the files they are observing. 
	virtual void FilesChanged();
	/// Notifies directory observers about modifications to the directory they are observing. 
	virtual void DirectoryChanged();

	/// Gets the handle to a change notification event or NULL, if unavailable.
	virtual HANDLE GetFileChangeNotification() const = 0;
	/// Gets the handle to a change notification event or NULL, if unavailable.
	virtual HANDLE GetDirectoryChangeNotification() const = 0;

	/// Gets the directory.
	const utf8_string& GetDirectory() const { return m_directory; }
};

struct close_change_notification_handle_policy
{
	static LEAN_INLINE HANDLE invalid() { return NULL; }
	static LEAN_INLINE void release(HANDLE handle) { ::FindCloseChangeNotification(handle); }
};

// Filesystem directory.
class FileSystemDirectory : public Directory
{
private:
	lean::handle_guard<HANDLE, close_change_notification_handle_policy> m_hFileChangeNotification;
	lean::handle_guard<HANDLE, close_change_notification_handle_policy> m_hDirectoryChangeNotification;

protected:
	/// Gets the revision of the given file.
	virtual lean::uint8 GetRevision(const lean::utf8_ntri &file) const;

	/// Called when a file observer has been added.
	virtual void FileObserverAdded();
	/// Called when a directory observer has been added.
	virtual void DirectoryObserverAdded();

public:
	/// Constructor.
	FileSystemDirectory(FileWatch::M *watch, const lean::utf8_ntri &directory);
	/// Destructor.
	virtual ~FileSystemDirectory();

	// Notifies file observers about modifications to the files they are observing. 
	virtual void FilesChanged();
	// Notifies observers about modifications to the directories they are observing. 
	virtual void DirectoryChanged();

	/// Gets the handle to a change notification event or NULL, if unavailable.
	virtual HANDLE GetFileChangeNotification() const;
	/// Gets the handle to a change notification event or NULL, if unavailable.
	virtual HANDLE GetDirectoryChangeNotification() const;
};

/// Observes files & directories.
void ObservationThread(FileWatch::M &m);

} // namespace

/// Implementation of the file system class internals.
struct FileWatch::M
{
	typedef boost::ptr_map_adapter< Directory, std::unordered_map<utf8_string, void*> > directory_map;
	directory_map directories;

	lean::critical_section directoryLock;

	volatile bool bShuttingDown;
	lean::event updateFilesEvent;
	lean::event updateDirectoriesEvent;

	lean::thread observationThread;

	/// Constructor.
	M();
};

// Constructor.
LEAN_INLINE FileWatch::M::M()
	: bShuttingDown(false),
	observationThread( lean::make_callable(this, &ObservationThread) )
{
}

// Constructor.
FileWatch::FileWatch()
	: m(new M())
{
}

// Destructor.
FileWatch::~FileWatch()
{
	// ORDER: Signal update AFTER all modifications have been completed
	m->bShuttingDown = true;
	m->updateFilesEvent.set();
	m->updateDirectoriesEvent.set();

	// Wait for observation thread to exit gracefully
	m->observationThread.join();
}

namespace
{

/// Notifies file observation thread about new directories.
void FileObservationChanged(FileWatch::M &m)
{
	m.updateFilesEvent.set();
}

/// Notifies directory observation thread about new directories.
void DirectoryObservationChanged(FileWatch::M &m)
{
	m.updateDirectoriesEvent.set();
}

/// Gets the given directory.
FileWatch::M::directory_map::iterator GetDirectory(FileWatch::M &m, const utf8_string &directory)
{
	FileWatch::M::directory_map::iterator itDirectory = m.directories.find(directory);

	if (itDirectory == m.directories.end())
	{
		// Don't modify directory map while being accessed from the observer thread
		lean::scoped_cs_lock lock(m.directoryLock);

		itDirectory = m.directories.insert( const_cast<utf8_string&>(directory), new FileSystemDirectory(&m, directory) ).first;
	}

	return itDirectory;
}

} // namespace

// Adds the given observer to be called when the given file has been modified.
bool FileWatch::AddObserver(const lean::utf8_ntri &file, FileObserver *pObserver)
{
	lean::utf8_string canonicalFile = lean::absolute_path<lean::utf8_string>(file);
	lean::utf8_string directory = lean::get_directory<lean::utf8_string>(canonicalFile);

	return GetDirectory(*m, directory)->second->AddObserver(canonicalFile, pObserver);
}

// Removes the given observer no longer to be called when the given file is modified.
void FileWatch::RemoveObserver(const lean::utf8_ntri &file, FileObserver *pObserver)
{
	lean::utf8_string canonicalFile = lean::absolute_path<lean::utf8_string>(file);
	lean::utf8_string directory = lean::get_directory<lean::utf8_string>(canonicalFile);

	M::directory_map::iterator itDirectory = m->directories.find(directory);

	if (itDirectory != m->directories.end())
		itDirectory->second->RemoveObserver(canonicalFile, pObserver);
}

// Adds the given observer to be called when the given directory is modified.
bool FileWatch::AddObserver(const lean::utf8_ntri &unresolvedDirectory, DirectoryObserver *pObserver)
{
	lean::utf8_string directory = lean::absolute_path<lean::utf8_string>(unresolvedDirectory);

	return GetDirectory(*m, directory)->second->AddObserver(pObserver);
}

// Removes the given observer, no longer to be called when the given directory is modified.
void FileWatch::RemoveObserver(const lean::utf8_ntri &unresolvedDirectory, DirectoryObserver *pObserver)
{
	lean::utf8_string directory = lean::absolute_path<lean::utf8_string>(unresolvedDirectory);

	M::directory_map::iterator itDirectory = m->directories.find(directory);

	if (itDirectory != m->directories.end())
		itDirectory->second->RemoveObserver(pObserver);
}

// Gets the file watch.
FileWatch& GetFileWatch()
{
	static FileWatch instance;
	return instance;
}

namespace
{

// Observes files & directories.
void ObservationThread(FileWatch::M &m)
{
	try
	{
		std::vector<Directory*> directories;
		std::vector<HANDLE> events;

		while (!m.bShuttingDown)
		{
			events.clear();
			directories.clear();

			{
				// Don't modify directory map while accessed from this thread
				lean::scoped_cs_lock lock(m.directoryLock);

				const size_t directoryCount = m.directories.size();

				directories.reserve(2 * directoryCount + 2);
				events.reserve(2 * directoryCount + 2);

				// Interrupt waiting when there are updates outside this thread
				directories.push_back(nullptr);
				events.push_back(m.updateFilesEvent.native_handle());

				directories.push_back(nullptr);
				events.push_back(m.updateDirectoriesEvent.native_handle());

				for (FileWatch::M::directory_map::iterator itDirectory = m.directories.begin();
					itDirectory != m.directories.end(); ++itDirectory)
				{
					Directory *pDirectory = itDirectory->second;
					
					HANDLE hFileChangeNotification = pDirectory->GetFileChangeNotification();
					HANDLE hDirectoryChangeNotification = pDirectory->GetDirectoryChangeNotification();

					if (hFileChangeNotification != NULL)
					{
						// Store changed notification event handles along with their corresponding directory instances
						directories.push_back(pDirectory);
						events.push_back(hFileChangeNotification);
					}

					if (hDirectoryChangeNotification != NULL)
					{
						// Store changed notification event handles along with their corresponding directory instances
						directories.push_back(pDirectory);
						events.push_back(hDirectoryChangeNotification);
					}
				}
			}

			DWORD dwSignal = ::WaitForMultipleObjects(static_cast<DWORD>(events.size()), &events[0], false, INFINITE);

			if (dwSignal == WAIT_FAILED)
				LEAN_LOG_WIN_ERROR_CTX("WaitForMultipleObjects()", "Waiting on file changed notifications");

			// Exclude update events (index 0 & 1)
			if (WAIT_OBJECT_0 + 2 <= dwSignal && dwSignal < WAIT_OBJECT_0 + events.size())
			{
				LEAN_ASSERT(events.size() == directories.size());
				
				// Get the directory that triggered the changed notification event
				Directory *directory = directories[dwSignal - WAIT_OBJECT_0];

				LEAN_ASSERT(directory);

				// MONITOR: HACK: TODO: Ugly way to avoid write conflicts ...
				Sleep(500);

				// Update files or directories
				if (events[dwSignal - WAIT_OBJECT_0] == directory->GetFileChangeNotification())
					directory->FilesChanged();
				else
					directory->DirectoryChanged();
			}

			// ORDER: Reset BEFORE re-checking shut down flag
			m.updateFilesEvent.reset();
			m.updateDirectoriesEvent.reset();
		}
	}
	catch (const std::exception &exc)
	{
		LEAN_LOG_ERROR_CTX(exc.what(), "File change observation thread");
	}
	catch (...)
	{
		LEAN_LOG_ERROR_MSG("Unhandled exception in file change observation thread.");
	}
}

// Constructor.
Directory::Directory(FileWatch::M *watch, const utf8_ntri &directory)
	: m_watch(watch),
	m_directory( directory.to<utf8_string>() )
{
}

// Destructor.
Directory::~Directory()
{
}

// Adds the given observer to be called when the given file has been modified.
bool Directory::AddObserver(const lean::utf8_ntri &file, FileObserver *pObserver)
{
	if (!pObserver)
	{
		LEAN_LOG_ERROR_MSG("pObserver == nullptr");
		return false;
	}

	lean::utf8_string fileName = lean::get_filename<lean::utf8_string>(file);

	// Do not modify observers while processing changed notifications
	lean::scoped_cs_lock lock(m_observerLock);

	for (file_observer_map::iterator it = m_fileObservers.lower_bound(fileName), itEnd = m_fileObservers.upper_bound(fileName); it != itEnd; ++it)
		// Keep matching observer
		if (it->second.pObserver == pObserver)
			return true; 

	m_fileObservers.insert( file_observer_map::value_type(
			fileName,
			FileObserverInfo(file, GetRevision(file), pObserver)
		) );

	FileObserverAdded();
	return true;
}

// Removes the given observer no longer to be called when the given file is modified.
void Directory::RemoveObserver(const lean::utf8_ntri &file, FileObserver *pObserver)
{
	lean::utf8_string fileName = lean::get_filename<lean::utf8_string>(file);

	// Do not modify observers while processing changed notifications
	lean::scoped_cs_lock lock(m_observerLock);

	const file_observer_map::iterator itObserverInfoEnd = m_fileObservers.upper_bound(fileName);

	for (file_observer_map::iterator it = m_fileObservers.lower_bound(fileName), itEnd = m_fileObservers.upper_bound(fileName); it != itEnd; ++it)
		// Erase ALL matching observation entries
		if (it->second.pObserver == pObserver)
			m_fileObservers.erase(it);
}

// Adds the given observer to be called when this directory has been modified.
bool Directory::AddObserver(DirectoryObserver *pObserver)
{
	if (!pObserver)
	{
		LEAN_LOG_ERROR_MSG("pObserver == nullptr");
		return false;
	}

	// Do not modify observers while processing changed notifications
	lean::scoped_cs_lock lock(m_observerLock);

	lean::push_unique(m_directoryObservers, pObserver);
	
	DirectoryObserverAdded();
	return true;
}

// Removes the given observer, no longer to be called when this directory is modified.
void Directory::RemoveObserver(DirectoryObserver *pObserver)
{
	// Do not modify observers while processing changed notifications
	lean::scoped_cs_lock lock(m_observerLock);
	
	lean::remove(m_directoryObservers, pObserver);
}

// Notifies file observers about modifications to the files they are observing. 
void Directory::FilesChanged()
{
	// Do not modify observers while processing changed notifications
	lean::scoped_cs_lock lock(m_observerLock);

	const file_observer_map::iterator itObserverInfoEnd = m_fileObservers.end();

	for (file_observer_map::iterator itObserverInfo = m_fileObservers.begin();
		itObserverInfo != itObserverInfoEnd; ++itObserverInfo)
	{
		FileObserverInfo &info = itObserverInfo->second;

		// Source of revision depends on type of directory
		lean::uint8 currentRevision = GetRevision(info.file);

		// Notify about updates AND revertions
		// (custom filtering may be performed via the revision argument passed)
		if (currentRevision != info.lastRevision)
		{
			info.lastRevision = currentRevision;

			try
			{
				info.pObserver->FileChanged(info.file, currentRevision);
			}
			catch (const std::exception &exc)
			{
				LEAN_LOG_ERROR_CTX(exc.what(), info.file.c_str());
			}
			catch (...)
			{
				LEAN_LOG_ERROR_CTX("Unhandled exception on file observer notificatio.", info.file.c_str());
			}
		}
	}
}

// Notifies directory observers about modifications to the directory they are observing. 
void Directory::DirectoryChanged()
{
	// Do not modify observers while processing changed notifications
	lean::scoped_cs_lock lock(m_observerLock);

	for (directory_observer_vector::iterator itObserver = m_directoryObservers.begin();
		itObserver != m_directoryObservers.end(); ++itObserver)
	{
		try
		{
			(*itObserver)->DirectoryChanged(m_directory);
		}
		catch (const std::exception &exc)
		{
			LEAN_LOG_ERROR_CTX(exc.what(), m_directory.c_str());
		}
		catch (...)
		{
			LEAN_LOG_ERROR_CTX("Unhandled exception on file observer notificatio.", m_directory.c_str());
		}
	}
}

/// Creates a file change notification handle for the given directory.
HANDLE CreateChangeNotification(const lean::utf8_ntri &directory, bool bDirectory)
{
	HANDLE hChangeNotification = ::FindFirstChangeNotificationW(
			lean::utf_to_utf16(directory).c_str(),
			bDirectory,
			(bDirectory) 
				? FILE_NOTIFY_CHANGE_FILE_NAME | FILE_NOTIFY_CHANGE_DIR_NAME
				: FILE_NOTIFY_CHANGE_LAST_WRITE
		);

	if (hChangeNotification == INVALID_HANDLE_VALUE || hChangeNotification == (HANDLE)ERROR_INVALID_FUNCTION)
	{
		LEAN_LOG_WIN_ERROR_CTX("FindFirstChangeNotification()", directory.c_str());
		// Convert to generic error value
		hChangeNotification = NULL;
	}
	
	return hChangeNotification;
}

// Constructor.
FileSystemDirectory::FileSystemDirectory(FileWatch::M *watch, const lean::utf8_ntri &directory)
	: Directory(watch, directory)
{
}

// Destructor.
FileSystemDirectory::~FileSystemDirectory()
{
}

// Gets the revision of the given file.
lean::uint8 FileSystemDirectory::GetRevision(const lean::utf8_ntri &file) const
{
	return lean::file_revision(file);
}

// Called when a file observer has been added.
void FileSystemDirectory::FileObserverAdded()
{
	if (m_hFileChangeNotification == NULL)
	{
		m_hFileChangeNotification = CreateChangeNotification(GetDirectory(), false);
		FileObservationChanged(*m_watch);
	}
}

// Called when a directory observer has been added.
void FileSystemDirectory::DirectoryObserverAdded()
{
	if (m_hDirectoryChangeNotification == NULL)
	{
		m_hDirectoryChangeNotification = CreateChangeNotification(GetDirectory(), true);
		DirectoryObservationChanged(*m_watch);
	}
}

// Gets the handle to a change notification event or NULL, if unavailable.
HANDLE FileSystemDirectory::GetFileChangeNotification() const
{
	return m_hFileChangeNotification;
}

// Gets the handle to a change notification event or NULL, if unavailable.
HANDLE FileSystemDirectory::GetDirectoryChangeNotification() const
{
	return m_hDirectoryChangeNotification;
}

// Notifies file observers about modifications to the files they are observing. 
void FileSystemDirectory::FilesChanged()
{
	if (m_hFileChangeNotification != NULL)
	{
		if (!::FindNextChangeNotification(m_hFileChangeNotification))
			LEAN_LOG_WIN_ERROR_CTX("FindNextChangeNotification()", GetDirectory().c_str());
	}

	Directory::FilesChanged();
}

// Notifies file observers about modifications to the directory they are observing. 
void FileSystemDirectory::DirectoryChanged()
{
	if (m_hDirectoryChangeNotification != NULL)
	{
		if (!::FindNextChangeNotification(m_hDirectoryChangeNotification))
			LEAN_LOG_WIN_ERROR_CTX("FindNextChangeNotification()", GetDirectory().c_str());
	}

	Directory::DirectoryChanged();
}

} // namespace

} // namespace