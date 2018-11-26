__device__ vec3 random_in_unit_disk(curandState *local_rand_state) {
	vec3 p;
	do {
		p = 2.0f*vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), 0) - vec3(1, 1, 0);
	} while (dot(p, p) >= 1.0f);
	return p;
}

class camera {
public:
	__device__ camera(vec3 lookfrom, vec3 lookat, vec3 vup, float vfov, float aspect, float aperture, float focus_dist) { // vfov is top to bottom in degrees
		lens_radius = aperture / 2.0f;
		float theta = vfov * ((float)M_PI) / 180.0f;
		float half_height = tan(theta / 2.0f);
		float half_width = aspect * half_height;
		origin = lookfrom;
		w = unit_vector(lookfrom - lookat);
		u = unit_vector(cross(vup, w));
		v = cross(w, u);
		lower_left_corner = origin - half_width * focus_dist*u - half_height * focus_dist*v - focus_dist * w;
		horizontal = 2.0f*half_width*focus_dist*u;
		vertical = 2.0f*half_height*focus_dist*v;
	}

	vec3 origin;
	vec3 lower_left_corner;
	vec3 horizontal;
	vec3 vertical;
	vec3 u, v, w;
	float lens_radius;
};
