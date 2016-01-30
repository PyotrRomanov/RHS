__constant float PI = 3.14159265359f;


typedef struct
{
	float3 diffuse;
	float refl;
	float refr;
    bool emissive;
} Material;

typedef struct
{
	float3 origin;
	float3 direction;
	int objIdx;
    float t;
    float3 N;
    bool inside;
} Ray;

int Map(int x, int in_min, int in_max, int out_min, int out_max)
{
	return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

float3 SampleSkydome()
{
    return (float3)(1.0f, 1.0f, 0.0f);
}

float GetRandom(__global float* rs, int offset)
{
    return rs[(get_global_id(0) + offset) % 1000];
}

Ray GenRay(int x, int y, __global float3* camera, uint width, uint height, __global float* randoms)
{
    Ray returnRay;

    // Unpacking camera info;
    float3 pos = camera[0];
    float3 p1 = camera[1];
    float3 p2 = camera[2];
    float3 p3 = camera[3];
    float3 up = camera[4];
    float3 right = camera[5];

    float r0 = GetRandom(randoms, 0);
    float r1 = GetRandom(randoms, 1);
    float r2 = GetRandom(randoms, 2) - 0.5f;
    float r3 = GetRandom(randoms, 3) - 0.5f;

    // calculate sub-pixel ray target position on screen plane
    float u = ((float)x + r0) / (float)width;
    float v = ((float)y + r1) / (float)height;
    float3 T = p1 + u * (p2 - p1) + v * (p3 - p1);
    // calculate position on aperture
    float3 P = pos + 0.004f /*lensSize*/ * (right + up);
    //float3 P = pos + 0.004f /*lensSize*/ * (r2 * right + r3 * up);
    // calculate ray direction
    float3 D = normalize(T - P);

    // return new primary ray
    returnRay.origin = P;
    returnRay.direction = D;
    returnRay.objIdx = -1;
    return returnRay;
}

Ray IntersectSphere(int idx, Ray r, float4 s)
{
    float3 L = s.xyz - r.origin;
    float tca = dot(L, r.direction);
    if (tca < 0) return r;
    float d2 = dot(L, L) - tca * tca;
    if (d2 > s.w) return r;
    float thc = sqrt(s.w - d2);
    float t0 = tca - thc;
    float t1 = tca + thc;
    if (t0 > 0)
    {
        if (t0 > r.t) return r;
        r.N = normalize(r.origin + t0 * r.direction - s.xyz);
        r.objIdx = idx;
        r.t = t0;
        return r;
    }
    else
    {
        if ((t1 > r.t)) return r;
        r.N = normalize(s.xyz - (r.origin + t1 * r.direction));
        r.objIdx = idx;
        r.t = t1;
        return r;
    }
}

Ray Intersect(Ray r, __global float4* world, uint worldSize)
{
    for (int i = 0; i < worldSize; i++)
        r =IntersectSphere(i, r, world[i]);
    return r;
}

Material GetMaterial(int objIdx, float3 I)
{
    Material mat;
    if (objIdx == 0)
    {
        // procedural checkerboard pattern for floor plane
        mat.refl = 0;
        mat.refr = 0;
        mat.emissive = false;
        int tx = ((int)(I.x * 3.0f + 1000) + (int)(I.z * 3.0f + 1000)) & 1;
        mat.diffuse = (float3)(1.0f,1.0f,1.0f) * ((tx == 1) ? 1.0f : 0.2f);
    }
    if ((objIdx == 1) || (objIdx > 8)) { mat.refl = mat.refr = 0; mat.emissive = false; mat.diffuse = (float3)(1.0f,1.0f,1.0f); }
    if (objIdx == 2) { mat.refl = 0.8f; mat.refr = 0; mat.emissive = false; mat.diffuse = (float3)(1, 0.2f, 0.2f); }
    if (objIdx == 3) { mat.refl = 0; mat.refr = 1; mat.emissive = false; mat.diffuse = (float3)(0.9f, 1.0f, 0.9f); }
    if (objIdx == 4) { mat.refl = 0.8f; mat.refr = 0; mat.emissive = false; mat.diffuse = (float3)(0.2f, 0.2f, 1); }
    if ((objIdx > 4) && (objIdx < 8)) { mat.refl = mat.refr = 0; mat.emissive = false; mat.diffuse = (float3)(1.0f,1.0f,1.0f); }
    if (objIdx == 8) { mat.refl = mat.refr = 0; mat.emissive = true; mat.diffuse = (float3)(8.5f, 8.5f, 7.0f); }
    return mat;
}

float3 Reflect(float3 V, float3 N)
{
    return V - 2.0f * dot(V, N) * N;
}

float3 Refract(bool inside, float3 D, float3 N, float3 R, float rnd)
{
    float nc = inside ? 1 : 1.2f, nt = inside ? 1.2f : 1;
    float nnt = nt / nc, ddn = dot(D, N);
    float cos2t = 1.0f - nnt * nnt * (1 - ddn * ddn);
    R = Reflect(D, N);
    if (cos2t >= 0)
    {
        float r1 = rnd;
        float a = nt - nc;
        float b = nt + nc;
        float R0 = a * a / (b * b);
        float c = 1 + ddn;
        float Tr = 1 - (R0 + (1 - R0) * c * c * c * c * c);
        if (r1 < Tr) R = (D * nnt - N * (ddn * nnt + sqrt(cos2t)));
    }
    return R;
}

float3 DiffuseRefract(float3 N, __global float* randoms)
{
    float r1 = GetRandom(randoms, 6);
    float r2 = GetRandom(randoms, 7);
    float r = sqrt(1.0f - r1 * r1);
    float phi = 2 * PI * r2;
    float3 R;
    R.x = cos(phi) * r;
    R.y = sin(phi) * r;
    R.z = r1;
    if (dot(N, R) < 0) R *= -1.0f;
    return R;
}

float3 Sample(Ray r, int depth, __global float4* world, uint worldSize, __global float* randoms)
{ 
    r = Intersect(r, world, worldSize);

    if (r.objIdx == -1)
        return SampleSkydome();

    float3 I = r.origin + r.t * r.direction;
    Material mat = GetMaterial(r.objIdx, I);

    if (mat.emissive) return mat.diffuse;
    if (depth >= 20) return (float3)(0.0f, 0.0f, 0.0f);

    float r0 = GetRandom(randoms, 0);
    float3 R = (float3)(0.0f, 0.0f, 0.0f);
    if (r0 < mat.refr)
    {
        R = Refract(r.inside, r.direction, r.N, R, GetRandom(randoms, 5));
        Ray er;
        er.origin = I + R * 0.0001f;
        er.direction = R;
        er.inside = (dot(r.N, R) < 0);
        return mat.diffuse; // Recursive
    }
    else if ((r0 < (mat.refl + mat.refr)) && (depth < 20))
    {
        // pure specular reflection
        R = Reflect(r.direction, r.N);
        Ray er;
        er.origin = I + R * 0.0001f;
        er.direction = R;
        return mat.diffuse;// Recursiev shit * Sample(extensionRay, depth + 1);
    }
    else
    {
        // diffuse reflection
        R = DiffuseRefract(r.N, randoms);
        Ray er;
        er.origin = I + R * 0.0001f;
        er.direction = R;
        return dot(R, r.N) * mat.diffuse;// Recursive shit * Sample(extensionRay, depth + 1);
    }
}

__kernel void Main(__global int* dst, uint width, uint height, __global float4* world, uint worldSize, __global float3* camera, __global float* randoms)  //, __global float* viewTransform, __global float* worldTransforms)
{
    int idx = get_global_id(0);// * get_global_size(0);
    int idy = get_global_id(1);// * get_global_size(1);

    Ray r = GenRay(idx, idy, camera, width, height, randoms);
    float3 col = Sample(r, 0, world, worldSize, randoms);

    dst[idx + idy * width] = ((int)(col.x * 255) << 16) + ((int)(col.y * 255) << 8) + (int)(col.z * 255);
	//dst[idx + idy * width] = ((int)(GetRandom(randoms, 1) * 255) << 16) + ((int)(GetRandom(randoms, 2) * 255) << 8) + (int)(GetRandom(randoms, 3) * 255);
}