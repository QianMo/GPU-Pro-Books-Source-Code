#pragma once

class DeviceContext11;
class ConstantBuffersPool11;
struct ID3D11Buffer;

class LargeConstantBuffer
{
public:
  LargeConstantBuffer(size_t, const void*, DeviceContext11&);
  ~LargeConstantBuffer();
  ID3D11Buffer* GetBuffer() const { return m_Buffer; }

protected:
  ConstantBuffersPool11* m_Pool;
  ID3D11Buffer* m_Buffer;
};
