#pragma once

template<typename T>
struct ModifiedData {
public:
	ModifiedData() : data_(new T) {};
	~ModifiedData() {
		delete data_;
	}

	void setHasChanged(bool value) { hasChanged_ = value; };
	void setData(T* data) { data_ = data; };
	T& data() { return T; };
	const T& data() const { return T; };

private:
	T* data_;
	bool hasChanged_;
};