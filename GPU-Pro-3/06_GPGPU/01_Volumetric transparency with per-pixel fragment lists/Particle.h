#pragma once

class Particle
{
	D3DXVECTOR3 position;
	D3DXVECTOR3 velocity;
	D3DXQUATERNION orientation;
	D3DXVECTOR3 spinAxis;
	float spinVelocity;
	float age;
	float lifespan;

	float radius;
	float distanceFromEye;
	D3DXVECTOR3 g;
	float tau;
	D3DXVECTOR4 gtau;
public:
	float getRadius()
	{
		
		float relAge = age / lifespan;
		return 100 * relAge * relAge;

	}

	D3DXVECTOR3 getPosition()
	{
		return position;
	}

	D3DXVECTOR4 getOrientation()
	{
		return *(D3DXVECTOR4*)&orientation;
	}

	D3DXVECTOR4 getPositionAndRadius()
	{
		return D3DXVECTOR4(
			position.x,
			position.y,
			position.z,
			getRadius());
	}

	D3DXVECTOR4 getGAndTau()
	{
//		return gtau;

		float relAge = age / lifespan;
//		float relAge2 = relAge * (1.0 - relAge);
	//	return gtau * relAge2;

		float density = 0.524 * relAge * (1.0 - relAge) * (1.0 - relAge);
		return D3DXVECTOR4(
			density * relAge * (1 - pow(relAge, 12)),
			density * pow(relAge, 4) * (1 - pow(relAge, 12)),
			density * pow(relAge, 10) * (1 - pow(relAge, 12)),
			density);
	}

	void reborn()
	{
		float density = 0.05;
		gtau = D3DXVECTOR4(
			density * (float)rand() / RAND_MAX,
			density * (float)rand() / RAND_MAX,
			density * (float)rand() / RAND_MAX,
			density);

		float x = (float)rand() / RAND_MAX;
		float y = (float)rand() / RAND_MAX;
		float z = (float)rand() / RAND_MAX;
		x = x * 2 - 1;	 y = y * 2 - 1;
		z = z * 2 - 1;
	//	x *= 5;
		//y *= 5;
		//z *= 5;
		position = D3DXVECTOR3(x, y, z);
		age = 0;
		lifespan = 6.0 + (float)rand() / RAND_MAX * 10.0;
		velocity = D3DXVECTOR3(x*7, y*7 + 20, z*7);
		orientation = D3DXQUATERNION(x, y, z, 1);
		D3DXQuaternionNormalize(&orientation, &orientation);
		D3DXVec3Cross( &spinAxis, &position, &D3DXVECTOR3(0, 1, 0));
		D3DXVec3Normalize( &spinAxis, &spinAxis );
		spinVelocity = 0.5 * (float)rand() / RAND_MAX;
	}

	void move(float dt)
	{
		position = position + velocity * dt;
		age += dt;
		D3DXQUATERNION dq;
		D3DXQuaternionRotationAxis(&dq, &spinAxis, spinVelocity * dt);
		orientation *= dq;
		if(age > lifespan)
			reborn();

	}
	void recalculateDistance(const D3DXVECTOR3 eye)
	{
		distanceFromEye = D3DXVec3Length(&(eye - position)) - getRadius();
	}


	float getDistanceFromEye()
	{
		return distanceFromEye;
	}
};
