// https://github.com/katzarsky/WebSocket
// http://stackoverflow.com/questions/8125507/how-can-i-send-and-receive-websocket-messages-on-the-server-side
// https://developer.mozilla.org/en-US/docs/Web/API/WebSockets_API/Writing_WebSocket_servers
// http://lucumr.pocoo.org/2012/9/24/websockets-101/
// https://tools.ietf.org/html/rfc6455


#pragma once


#include <essentials/types.h>
#include <essentials/string.h>


using namespace NEssentials;


namespace NCommon
{
namespace NWebSocket
{
	enum class FrameType
	{
		Error = 0xFF00,
		Incomplete = 0xFE00,
		Opening = 0x3300,
		Closing = 0x3400,
		IncompleteText = 0x01,
		IncompleteBinary = 0x02,
		Text = 0x81,
		Binary = 0x82,
		Ping = 0x19,
		Pong = 0x1A
	};

	void HandshakeExtractKeyAndProtocol(uint8* data, int dataSize, string& key, string& protocol);
	string HandshakeConstructResponse(const string& hashedKey, const string& protocol);
	vector<uint8> EncodeFrame(FrameType frameType, uint8* data, int dataSize);
	vector<uint8> DecodeFrame(uint8* data, int dataSize, FrameType* frameType = nullptr);

	//

	void HandshakeExtractKeyAndProtocol(uint8* data, int dataSize, string& key, string& protocol)
	{
		string data_string = ToString(data);
		string webSocketKey = string("Sec-WebSocket-Key: ");
		string webSocketProtocol = string("Sec-WebSocket-Protocol: ");

		int webSocketKeyIndex = data_string.find(webSocketKey) + webSocketKey.length();
		int webSocketProtocolIndex = data_string.find(webSocketProtocol) + webSocketProtocol.length();

		key = protocol = "";
		for (int i = webSocketKeyIndex; i < dataSize; i++)
		{
			if (data[i] == '\n' || data[i] == '\r')
				break;
			key += data[i];
		}
		for (int i = webSocketProtocolIndex; i < dataSize; i++)
		{
			if (data[i] == '\n' || data[i] == '\r')
				break;
			protocol += data[i];
		}
	}

	string HandshakeConstructResponse(const string& hashedKey, const string& protocol)
	{
		return
			"HTTP/1.1 101 Switching Protocols" + newline
			+ "Upgrade: websocket" + newline
			+ "Connection: Upgrade" + newline
			+ "Sec-WebSocket-Accept: " + hashedKey + newline
			+ "Sec-WebSocket-Protocol: " + protocol + newline
			+ newline;
	}

	vector<uint8> EncodeFrame(FrameType frameType, uint8* data, int dataSize)
	{
		vector<uint8> outData;
		outData.push_back((uint8)frameType);

		if (dataSize <= 125)
		{
			outData.push_back(dataSize);
		}
		else if (dataSize <= 65535)
		{
			outData.push_back(126);

			outData.push_back((dataSize >> 8) & 0xFF);
			outData.push_back(dataSize & 0xFF);
		}
		else
		{
			outData.push_back(127);

			for (int i = 3; i >= 0; i--)
				outData.push_back(0);
			for (int i = 3; i >= 0; i--)
				outData.push_back((dataSize >> 8*i) & 0xFF);
		}
		
		int headerSize = outData.size();
		outData.resize(headerSize + dataSize);
		memcpy(&outData[headerSize], data, dataSize);

		return outData;
	}

	vector<uint8> DecodeFrame(uint8* data, int dataSize, FrameType* frameType)
	{
		vector<uint8> outData;

		if (dataSize < 3)
		{
			if (frameType) *frameType = FrameType::Incomplete;
			return outData;
		}

		uint8 fin = (data[0] >> 7) & 0x01;
		uint8 opcode = data[0] & 0x0F;
		uint8 masked = (data[1] >> 7) & 0x01;
		int payloadLen = data[1] & (~0x80);

		int contentOffset = 2;
		int contentSize = 0;

		if (payloadLen <= 125)
		{
			contentSize = payloadLen;
		}
		else if (payloadLen == 126)
		{
			contentSize = data[2] + (data[3] << 8);
			contentOffset += 2;
		}
		else if (payloadLen == 127)
		{
			contentSize = data[2] + (data[3] << 8); 
			contentOffset += 8;
		}

		if (dataSize < contentOffset + contentSize)
		{
			if (frameType) *frameType = FrameType::Incomplete;
			return outData;
		}

		if (masked)
		{
			uint mask = *((uint*)(data + contentOffset));
			contentOffset += 4;

			for (int i = 0; i < contentSize; i++)
				data[contentOffset + i] = data[contentOffset + i] ^ ((uint8*)(&mask))[i % 4];
		}

		outData.resize(contentSize);
		memcpy(&outData[0], data + contentOffset, contentSize);

		if (frameType)
		{
			if (opcode == 0x0)
				*frameType = (fin) ? FrameType::Text : FrameType::IncompleteText;
			else if (opcode == 0x1)
				*frameType = (fin) ? FrameType::Text : FrameType::IncompleteText;
			else if (opcode == 0x2)
				*frameType = (fin) ? FrameType::Binary : FrameType::IncompleteBinary;
			else if (opcode == 0x9)
				*frameType = FrameType::Ping;
			else if (opcode == 0xA)
				*frameType = FrameType::Pong;
			else
				*frameType = FrameType::Error;
		}

		return outData;
	}
}
}
