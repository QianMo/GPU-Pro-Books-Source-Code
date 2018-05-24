#pragma once


#include <essentials/main.h>

#include <SDL_net.h>


using namespace NEssentials;


namespace NSystem
{
	void InitializeSockets();
	void DeinitializeSockets();

	struct IPAddress
	{
		IPaddress ipAddress;

		void Set(const string& host, uint16 port) { ASSERT_FUNCTION(SDLNet_ResolveHost(&ipAddress, host.c_str(), port) != -1); }

		string Host() { return string(SDLNet_ResolveIP(&ipAddress)); }
		uint16 Port() { return SDLNet_Read16(&ipAddress.port); }

		bool operator == (const IPAddress& other) { return (ipAddress.host == other.ipAddress.host && ipAddress.port == other.ipAddress.port); }
		bool operator != (const IPAddress& other) { return !(*this == other); }
	};

	class TCPSocket
	{
	public:
		bool Read(uint8* data, int& dataSize, int maxDataSize);
		int Send(const uint8* data, int dataSize);

	public: // readonly
		TCPsocket socket;
	};

	class TCPServer
	{
	public:
		void Open(int maxSocketsCount, uint16 port);
		void Close();

		TCPSocket* Poll(uint32 timeout); // returns pointer to new client (allocates memory); nullptr otherwise
		void Disconnect(const TCPSocket& socket);

	public: // readonly
		SDLNet_SocketSet socketSet;
		TCPsocket serverSocket;
	};

	class UDPSocket
	{
	public:
		void Open(int maxPacketSize, uint16 port);
		void Close();

		void Poll(uint32 timeout);

		bool Read(IPAddress& ipAddress, uint8* data, int& dataSize);
		void Send(const IPAddress& ipAddress, uint8* data, int dataSize);

	public: // readonly
		SDLNet_SocketSet socketSet;
		UDPsocket socket;
		UDPpacket *readPacket;
		UDPpacket *sendPacket;
	};

	//

	inline void InitializeSockets()
	{
		ASSERT_FUNCTION(SDLNet_Init() != -1);
	}

	inline void DeinitializeSockets()
	{
		SDLNet_Quit();
	}

	//

	inline bool TCPSocket::Read(uint8* data, int& dataSize, int maxDataSize)
	{
		if (SDLNet_SocketReady(socket))
		{
			dataSize = SDLNet_TCP_Recv(socket, data, maxDataSize);
			return true;
		}
		else
		{
			return false;
		}
	}

	inline int TCPSocket::Send(const uint8* data, int dataSize)
	{
		return SDLNet_TCP_Send(socket, data, dataSize);
	}

	//

	inline void TCPServer::Open(int maxSocketsCount, uint16 port)
	{
		IPaddress ipAddress;

		ASSERT_FUNCTION(SDLNet_ResolveHost(&ipAddress, nullptr, port) != -1);
		serverSocket = SDLNet_TCP_Open(&ipAddress);
		ASSERT(serverSocket != nullptr);

		socketSet = SDLNet_AllocSocketSet(maxSocketsCount);
		ASSERT(socketSet != nullptr);
		ASSERT_FUNCTION(SDLNet_TCP_AddSocket(socketSet, serverSocket) != -1);
	}

	inline void TCPServer::Close()
	{
		ASSERT_FUNCTION(SDLNet_TCP_DelSocket(socketSet, serverSocket) != -1);
		SDLNet_FreeSocketSet(socketSet);
	}

	inline TCPSocket* TCPServer::Poll(uint32 timeout)
	{
		ASSERT_FUNCTION(SDLNet_CheckSockets(socketSet, timeout) != -1);

		if (SDLNet_SocketReady(serverSocket))
		{
			TCPSocket* clientSocket = new TCPSocket();

			clientSocket->socket = SDLNet_TCP_Accept(serverSocket);
			ASSERT(clientSocket->socket != nullptr);
			ASSERT_FUNCTION(SDLNet_TCP_AddSocket(socketSet, clientSocket->socket) != -1);

			return clientSocket;
		}
		else
		{
			return nullptr;
		}
	}

	inline void TCPServer::Disconnect(const TCPSocket& socket)
	{
		ASSERT_FUNCTION(SDLNet_TCP_DelSocket(socketSet, socket.socket) != -1);
	}

	//

	inline void UDPSocket::Open(int maxPacketSize, uint16 port)
	{
		socket = SDLNet_UDP_Open(port);
		ASSERT(socket != nullptr);

		socketSet = SDLNet_AllocSocketSet(1);
		ASSERT(socketSet != nullptr);
		ASSERT_FUNCTION(SDLNet_UDP_AddSocket(socketSet, socket) != -1);

		readPacket = SDLNet_AllocPacket(maxPacketSize);
		ASSERT(readPacket != nullptr);
		sendPacket = SDLNet_AllocPacket(maxPacketSize);
		ASSERT(sendPacket != nullptr);
	}

	inline void UDPSocket::Close()
	{
		SDLNet_UDP_Close(socket);

		ASSERT_FUNCTION(SDLNet_UDP_DelSocket(socketSet, socket) != -1);
		SDLNet_FreeSocketSet(socketSet);

		SDLNet_FreePacket(readPacket);
		SDLNet_FreePacket(sendPacket);
	}

	void UDPSocket::Poll(uint32 timeout)
	{
		ASSERT_FUNCTION(SDLNet_CheckSockets(socketSet, timeout) != -1);
	}

	bool UDPSocket::Read(IPAddress& ipAddress, uint8* data, int& dataSize)
	{
		if (SDLNet_SocketReady(socket))
		{
			int recv = SDLNet_UDP_Recv(socket, readPacket);
			ASSERT_FUNCTION(recv >= 0);

			if (recv == 0)
				return false;

			ipAddress.ipAddress = readPacket->address;
			memcpy(data, readPacket->data, readPacket->len);
			dataSize = readPacket->len;

			return true;
		}
		else
		{
			return false;
		}
	}

	void UDPSocket::Send(const IPAddress& ipAddress, uint8* data, int dataSize)
	{
		sendPacket->address = ipAddress.ipAddress;
		sendPacket->data = data;
		sendPacket->len = dataSize;

		ASSERT_FUNCTION(SDLNet_UDP_Send(socket, -1, sendPacket) != 0);
	}
}
