#pragma once


#include <essentials/main.h>
#include <system/main.h>
#include <math/main.h>


using namespace NEssentials;
using namespace NSystem;


namespace NCommon
{
	class PacketManager
	{
	private:
		const int ResendTimeout = 10000; // 10 ms
		const int ReliablePacketToSendDeletionTimeout = 10000000; // 10 secs
		const int ReceivedReliablePacketDeletionTimeout = 2 * ReliablePacketToSendDeletionTimeout; // 20 secs

	private:
		struct PacketData
		{
			uint8* data;
			int dataSize; // with header

			PacketData()
			{
				data = nullptr;
				dataSize = 0;
			}

			void Destroy()
			{
				SAFE_DELETE_ARRAY(this->data);
			}

			void SetUnreliable(uint8* data, int dataSize)
			{
				SAFE_DELETE_ARRAY(this->data);

				this->dataSize = dataSize + 1;

				this->data = new uint8[this->dataSize];
				this->data[0] = 0;
				memcpy(this->data + 1, data, dataSize);
			}
		
			void SetReliable(uint8* data, int dataSize)
			{
				SAFE_DELETE_ARRAY(this->data);

				this->dataSize = dataSize + 5;

				int32 guid = NMath::Random32(1, NMath::IntMax);

				this->data = new uint8[this->dataSize];
				this->data[0] = 1;
				memcpy(this->data + 1, &guid, 4);
				memcpy(this->data + 5, data, dataSize);
			}

			void SetACK(int32 guid)
			{
				SAFE_DELETE_ARRAY(this->data);

				this->dataSize = 5;

				this->data = new uint8[5];
				this->data[0] = 2;
				memcpy(this->data + 1, &guid, 4);
			}

			bool IsUnreliable()
			{
				return data[0] == 0;
			}

			bool IsReliable()
			{
				return data[0] == 1;
			}

			bool IsACK()
			{
				return data[0] == 2;
			}

			void Data(uint8* data, int& dataSize)
			{
				if (IsUnreliable())
				{
					dataSize = this->dataSize - 1;
					memcpy(data, this->data + 1, dataSize);
				}
				else if (IsReliable())
				{
					dataSize = this->dataSize - 5;
					memcpy(data, this->data + 5, dataSize);
				}
				else
				{
					dataSize = 0;
				}
			}

			int32 GUID()
			{
				if (IsReliable() || IsACK())
				{
					int32 value;
					memcpy(&value, data + 1, 4);
					return value;
				}
				else
				{
					return 0;
				}
			}
		};

		struct ReliablePacketToSend
		{
			UDPSocket* udpSocket;
			IPAddress ipAddress;
			PacketData packetData;
			uint64 creationTimestamp;
			uint64 resendTimestamp;
		};

		struct ReceivedReliablePacket
		{
			int32 guid;
			uint64 timestamp;
		};

	public:
		void Create(int maxPacketSize);
		void Destroy();

		void Send(bool isReliable, UDPSocket* udpSocket, IPAddress ipAddress, uint8* data, int dataSize);
		bool Read(UDPSocket* udpSocket, IPAddress& ipAddress, uint8* data, int& dataSize);

		void Update();

	private:
		uint8* tempData;
		int tempDataSize;

		vector<ReliablePacketToSend> reliablePacketsToSend;
		vector<ReceivedReliablePacket> receivedReliablePackets;
	};
}


inline void NCommon::PacketManager::Create(int maxPacketSize)
{
	tempData = new uint8[maxPacketSize];
}


inline void NCommon::PacketManager::Destroy()
{
	SAFE_DELETE_ARRAY(tempData);
}


inline void NCommon::PacketManager::Send(bool isReliable, UDPSocket* udpSocket, IPAddress ipAddress, uint8* data, int dataSize)
{
	if (isReliable)
	{
		ReliablePacketToSend reliablePacket;
		reliablePacket.udpSocket = udpSocket;
		reliablePacket.ipAddress = ipAddress;
		reliablePacket.packetData.SetReliable(data, dataSize);
		reliablePacket.creationTimestamp = TickCount();
		reliablePacket.resendTimestamp = reliablePacket.creationTimestamp;

		reliablePacketsToSend.push_back(reliablePacket);
	}
	else
	{
		PacketData packetData;
		packetData.SetUnreliable(data, dataSize);
		udpSocket->Send(ipAddress, packetData.data, packetData.dataSize);
		packetData.Destroy();
	}
}


inline bool NCommon::PacketManager::Read(UDPSocket* udpSocket, IPAddress& ipAddress, uint8* data, int& dataSize)
{
	if (udpSocket->Read(ipAddress, tempData, tempDataSize))
	{
		PacketData packetData;
		packetData.data = tempData;
		packetData.dataSize = tempDataSize;

		if (packetData.IsUnreliable())
		{
			packetData.Data(data, dataSize);
			return true;
		}
		else if (packetData.IsReliable())
		{
			bool found = false;

			for (uint i = 0; i < receivedReliablePackets.size(); i++)
			{
				if (receivedReliablePackets[i].guid == packetData.GUID())
				{
					found = true;
					break;
				}
			}

			// send ACK
			PacketData ackPacketData;
			ackPacketData.SetACK(packetData.GUID());
			udpSocket->Send(ipAddress, ackPacketData.data, ackPacketData.dataSize);

			if (!found)
			{
				ReceivedReliablePacket reliablePacket;
				reliablePacket.guid = packetData.GUID();
				reliablePacket.timestamp = TickCount();

				receivedReliablePackets.push_back(reliablePacket);

				packetData.Data(data, dataSize);
			}
			
			return !found;
		}
		else if (packetData.IsACK())
		{
			for (uint i = 0; i < reliablePacketsToSend.size(); i++)
			{
				if (reliablePacketsToSend[i].packetData.GUID() == packetData.GUID())
				{
					reliablePacketsToSend.erase(reliablePacketsToSend.begin() + i);
					break;
				}
			}

			return false;
		}
		else
		{
			return false;
		}
	}
	else
	{
		return false;
	}
}


inline void NCommon::PacketManager::Update()
{
	uint64 tickCount = TickCount();

	// send all reliable packets for which we have not received ACKs yet
	for (uint i = 0; i < reliablePacketsToSend.size(); i++)
	{
		if (tickCount - reliablePacketsToSend[i].resendTimestamp > ResendTimeout)
		{
			reliablePacketsToSend[i].udpSocket->Send(reliablePacketsToSend[i].ipAddress, reliablePacketsToSend[i].packetData.data, reliablePacketsToSend[i].packetData.dataSize);
			reliablePacketsToSend[i].resendTimestamp = tickCount;
		}
	}

	// delete one timeoutted reliable packet that was to be sent but has not received an ACK
	for (uint i = 0; i < reliablePacketsToSend.size(); i++)
	{
		if (tickCount - reliablePacketsToSend[i].creationTimestamp > ReliablePacketToSendDeletionTimeout)
		{
			reliablePacketsToSend.erase(reliablePacketsToSend.begin() + i);
			break;
		}
	}

	// delete one reliable packet that we received (we stop keeping track of it, hoping that its ACK has been delivered)
	for (uint i = 0; i < receivedReliablePackets.size(); i++)
	{
		if (tickCount - receivedReliablePackets[i].timestamp > ReceivedReliablePacketDeletionTimeout)
		{
			receivedReliablePackets.erase(receivedReliablePackets.begin() + i);
			break;
		}
	}
}
