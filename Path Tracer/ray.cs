using System;
using System.Numerics;

namespace Template {

class Ray
{
	public Ray()
	{
		objIdx = -1;
		inside = false;
	}
	public Ray( Vector3 origin, Vector3 direction, float distance )
	{
		O = origin;
		D = direction;
		t = distance;
		objIdx = -1;
		inside = false;
	}
	public Vector3 O, D;
	public float t;
	public Vector3 N;
	public int objIdx;
	public bool inside;
}

} // namespace Template