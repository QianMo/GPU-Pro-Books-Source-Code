#pragma once


#include <essentials/main.h>


using namespace NEssentials;


namespace NSystem
{
	class File
	{
	public:
		enum class OpenMode
		{
			ReadText,
			ReadBinary,
			WriteText,
			WriteBinary,
		};

	public:
		bool Open(const string& path, OpenMode openMode)
		{
			ios_base::openmode openMode_stl = (ios_base::openmode)0;

			if (openMode == OpenMode::ReadText)
				openMode_stl = ios::in;
			if (openMode == OpenMode::ReadBinary)
				openMode_stl = ios::in | ios::binary;
			if (openMode == OpenMode::WriteText)
				openMode_stl = ios::out;
			if (openMode == OpenMode::WriteBinary)
				openMode_stl = ios::out | ios::binary;

			file.open(path.c_str(), openMode_stl);

			if (file)
				return true;
			else
				return false;
		}

		void Close()
		{
			file.close();
		}

		bool EndOfFile()
		{
			return file.eof();
		}

		void Seek(int position)
		{
			file.clear();
			file.seekg(position, ios::beg);
		}

		int ReadBin(char* data, int size)
		{
			file.read(data, size);
			return (int)file.gcount();
		}

		int ReadBin(string& s)
		{
			uint length;
			ReadBin((char*)&length, sizeof(uint));

			char *data = new char[length + 1];
			ReadBin(data, length);
			data[length] = '\0';

			s = string(data);
			delete[] data;

			return length;
		}

		int ReadLine(string& s)
		{
			char c;
			string line = "";
			int length = 0;

			while (ReadBin(&c, 1))
			{
				if (c == '\n')
					break;

				line += c;
				length++;
			}

			s = line;
			return length;
		}

		int ReadLines(vector<string>& lines)
		{
			int length = 0;

			while (!EndOfFile())
			{
				string s;
				length += ReadLine(s);
				lines.push_back(s);
			}

			return length;
		}

		int WriteBin(char* data, int size)
		{
			int before = (int)file.tellp();
			file.write(data, size);
			return ((int)file.tellp() - before);
		}

		int WriteBin(const string& s)
		{
			uint bytesWritten = 0;

			uint length = s.length();
			bytesWritten += WriteBin((char*)&length, sizeof(uint));
			bytesWritten += WriteBin((char*)s.c_str(), length);

			return bytesWritten;
		}

		void ReadText(bool& b) { file >> b; }
		void ReadText(int32& i) { file >> i; }
		void ReadText(uint32& i) { file >> i; }
		void ReadText(int16& i) { file >> i; }
		void ReadText(uint16& i) { file >> i; }
		void ReadText(float& f) { file >> f; }
		void ReadText(double& d) { file >> d; }
		void ReadText(string& s) { file >> s; }

		void WriteText(const bool& b) { file << b; }
		void WriteText(const int32& i) { file << i; }
		void WriteText(const uint32& i) { file << i; }
		void WriteText(const int16& i) { file << i; }
		void WriteText(const uint16& i) { file << i; }
		void WriteText(const float& f) { file << f; }
		void WriteText(const double& d) { file << d; }
		void WriteText(const string& s) { file << s; }
		void WriteText(const char* s) { file << string(s); }
		void WriteTextNewline() { file << endl; }

	private:
		fstream file;
	};


	inline int OpenAndReadFile(const string& path, string& s)
	{
		File file;

		if (file.Open(path, File::OpenMode::ReadText))
		{
			vector<string> lines;
			int length = file.ReadLines(lines);
			file.Close();
			s = Merge(lines);
			return length;
		}

		s = "";
		return 0;
	}


	inline int OpenAndReadFile(const string& path, vector<string>& lines)
	{
		File file;

		if (file.Open(path, File::OpenMode::ReadText))
		{
			int length = file.ReadLines(lines);
			file.Close();
			return length;
		}

		lines.clear();
		return 0;
	}


	inline bool FileExists(const string& path)
	{
		File file;

		if (file.Open(path, File::OpenMode::ReadText))
		{
			file.Close();
			return true;
		}
		else
		{
			return false;
		}
	}
}
