struct material{
	float reflection;
	float refraction;
	float4 color;
};

struct sphere{
	float3 pos;
	struct material* m;
	float r;
};

struct ray{
	float3 O;
	float3 dir;
};

struct scene{
	int count;
	struct sphere spheres[10];
	struct material defaultMat;
};

int map(int x, int in_min, int in_max, int out_min, int out_max){
	return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;	
}

__kernel void rayTest(__global int* dst, uint width, uint height){
	int idx = (int)get_global_id(0);
	int x =  map(idx % width, 0, width, 0, 255);
	int y = map(idx / width, 0, height, 0, 255);
	dst[idx] = (x << 8) + y;
}