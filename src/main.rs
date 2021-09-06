#![feature(stdsimd)]

use core::arch::x86_64::*;
use std::{
    alloc::{alloc, dealloc, Layout},
    arch::x86_64::_mm_set_ps,
    intrinsics::transmute,
    mem::{align_of, size_of},
    ptr,
    ptr::write_bytes,
    time::Instant,
};

use minifb::{Window, WindowOptions};
use rand::{
    distributions::{Distribution, Open01, Uniform},
    prelude::ThreadRng,
    thread_rng,
    Rng,
};

const WIDTH: usize = 1024;
const HEIGHT: usize = 640;

static RATIO: f32 = 1.0 / 0xFFFFFFFFu32 as f32;
static mut T: f32 = 0.0;

fn to_radians(value: f32) -> f32 {
    value * (std::f32::consts::TAU / 360.0)
}

struct Bitmap {
    width:  usize,
    height: usize,
    pixels: Vec<u32>,
}

impl Bitmap {
    fn new(width: usize, height: usize) -> Self {
        let size = width * height;

        Self {
            width,
            height,
            pixels: Vec::with_capacity(size),
        }
    }

    fn pixels(&self) -> &[u32] {
        &self.pixels
    }

    fn clear(&mut self) {
        let size = self.width * self.height;
        unsafe {
            write_bytes(self.pixels.as_mut_ptr(), 0, size);
            self.pixels.set_len(size);
        }
    }
}

#[repr(C, align(16))]
struct FourRandomColours([i32; 4]);

impl FourRandomColours {
    pub fn from(rng: &mut ThreadRng, range: &Uniform<u128>) -> Self {
        let value: u128 = range.sample(rng);

        unsafe { Self(transmute(value)) }
    }

    pub fn into_simd(self, mask: __m128i) -> __m128i {
        unsafe { _mm_or_si128(_mm_load_epi32(self.0.as_ptr()), mask) }
    }
}

struct StarField {
    perspective_on_x: bool,
    perspective_on_y: bool,
    tan_half_fov:     f32,
    star_count:       usize,
    star_speed:       f32,
    star_spread:      f32,
    phase_speed:      f32,
    star_x:           *mut f32,
    star_y:           *mut f32,
    star_z:           *mut f32,
    pixel_addrs:      *mut i32,
    escaped:          *mut u32,
    star_colour:      *mut u32,
}

impl StarField {
    fn new() -> Self {
        let mut field = Self {
            perspective_on_x: true,
            perspective_on_y: true,
            tan_half_fov:     f32::tan(to_radians(30.0 * 0.5)),
            star_count:       16384,
            star_speed:       3.0,
            star_spread:      16.0,
            phase_speed:      64.0,
            star_x:           ptr::null_mut(),
            star_y:           ptr::null_mut(),
            star_z:           ptr::null_mut(),
            pixel_addrs:      ptr::null_mut(),
            escaped:          ptr::null_mut(),
            star_colour:      ptr::null_mut(),
        };

        assert!(field.star_count % 4 == 0);

        unsafe {
            let star_pool = alloc(field.layout()) as *mut f32;

            field.star_x = star_pool;
            field.star_y = star_pool.add(field.star_count);
            field.star_z = star_pool.add(field.star_count * 2);

            field.pixel_addrs = star_pool.add(field.star_count * 3) as *mut i32;
            field.escaped = star_pool.add(field.star_count * 4) as *mut u32;
            field.star_colour = star_pool.add(field.star_count * 5) as *mut u32;

            let mut rng = thread_rng();
            let colour_range = Uniform::from(0..!0);
            let star_range = Open01;
            let mask = _mm_set1_epi32(0x000000FF);
            let zero_ps = _mm_set1_ps(0.0);
            let one_ps = _mm_set1_ps(1.0);
            let two_ps = _mm_set1_ps(2.0);
            let ratio_ps = _mm_set1_ps(RATIO);
            let star_spread_ps = _mm_set1_ps(field.star_spread);

            for i in (0..field.star_count).step_by(4) {
                let value = FourRandomColours::from(&mut rng, &colour_range);
                let colour_epi32 = value.into_simd(mask);

                _mm_storeu_si128(field.star_colour.add(i) as *mut __m128i, colour_epi32);

                let (sin, cos) = {
                    let t1 = _mm_set_ps(
                        star_range.sample(&mut rng),
                        star_range.sample(&mut rng),
                        star_range.sample(&mut rng),
                        star_range.sample(&mut rng),
                    );

                    let t2 = _mm_set_ps(
                        star_range.sample(&mut rng),
                        star_range.sample(&mut rng),
                        star_range.sample(&mut rng),
                        star_range.sample(&mut rng),
                    );

                    (
                        _mm_sub_ps(_mm_mul_ps(two_ps, t1), one_ps),
                        _mm_sub_ps(_mm_mul_ps(two_ps, t2), one_ps),
                    )
                };

                let colour_ps = _mm_cvtepi32_ps(colour_epi32);
                let zero_to_one = _mm_mul_ps(ratio_ps, colour_ps);

                let negative_one_to_one = _mm_sub_ps(_mm_mul_ps(two_ps, zero_to_one), one_ps);

                _mm_storeu_ps(field.star_x.add(i), cos);

                _mm_storeu_ps(
                    field.star_y.add(i),
                    _mm_add_ps(
                        negative_one_to_one,
                        if field.perspective_on_x && field.perspective_on_y {
                            sin
                        } else {
                            zero_ps
                        },
                    ),
                );

                let random_zero_to_one = _mm_set_ps(
                    star_range.sample(&mut rng),
                    star_range.sample(&mut rng),
                    star_range.sample(&mut rng),
                    star_range.sample(&mut rng),
                );

                _mm_storeu_ps(
                    field.star_z.add(i),
                    _mm_mul_ps(random_zero_to_one, star_spread_ps),
                );
            }
        }

        field
    }

    fn update_and_render(&mut self, target: &mut Bitmap, delta: f32) {
        unsafe {
            let movement_ps = _mm_mul_ps(_mm_set1_ps(delta), _mm_set1_ps(self.star_speed));

            let zero_epi32 = _mm_set1_epi32(0);
            let half_ps = _mm_set1_ps(0.5);
            let half_fov_ps = _mm_set1_ps(self.tan_half_fov);

            let width_epi32 = _mm_set1_epi32(target.width as i32);
            let height_epi32 = _mm_set1_epi32(target.height as i32);

            let width_ps = _mm_set1_ps(target.width as f32);
            let height_ps = _mm_set1_ps(target.height as f32);

            let half_width_ps = _mm_mul_ps(width_ps, half_ps);
            let half_height_ps = _mm_mul_ps(height_ps, half_ps);

            for i in (0..self.star_count).step_by(4) {
                let z_addr = self.star_z.add(i);

                let x = _mm_loadu_ps(self.star_x.add(i));
                let y = _mm_loadu_ps(self.star_y.add(i));
                let z = _mm_sub_ps(_mm_loadu_ps(z_addr), movement_ps);

                _mm_storeu_ps(z_addr, z);

                let perspective = _mm_mul_ps(z, half_fov_ps);

                let screen_x = _mm_cvtps_epi32(_mm_add_ps(
                    _mm_mul_ps(
                        if self.perspective_on_x {
                            _mm_div_ps(x, perspective)
                        } else {
                            x
                        },
                        half_width_ps,
                    ),
                    half_width_ps,
                ));

                let screen_y = _mm_cvtps_epi32(_mm_add_ps(
                    _mm_mul_ps(
                        if self.perspective_on_y {
                            _mm_div_ps(y, perspective)
                        } else {
                            y
                        },
                        half_height_ps,
                    ),
                    half_height_ps,
                ));

                let mut escaped = zero_epi32;

                // screen_x < 0 || screen_x >= width
                escaped = _mm_or_si128(escaped, _mm_cmplt_epi32(screen_x, zero_epi32));
                escaped = _mm_or_si128(escaped, _mm_cmpgt_epi32(screen_x, width_epi32));
                escaped = _mm_or_si128(escaped, _mm_cmpeq_epi32(screen_x, width_epi32));

                // screen_y < 0 || screen_y >= width
                escaped = _mm_or_si128(escaped, _mm_cmplt_epi32(screen_y, zero_epi32));
                escaped = _mm_or_si128(escaped, _mm_cmpgt_epi32(screen_y, height_epi32));
                escaped = _mm_or_si128(escaped, _mm_cmpeq_epi32(screen_y, height_epi32));

                // z <= 0
                let z_epi32 = _mm_cvtps_epi32(z);
                escaped = _mm_or_si128(escaped, _mm_cmplt_epi32(z_epi32, zero_epi32));
                escaped = _mm_or_si128(escaped, _mm_cmpeq_epi32(z_epi32, zero_epi32));

                let pixel_addr = _mm_cvtps_epi32(_mm_add_ps(
                    _mm_cvtepi32_ps(screen_x),
                    _mm_mul_ps(_mm_cvtepi32_ps(screen_y), width_ps),
                ));

                _mm_storeu_si128(self.escaped.add(i) as *mut __m128i, escaped);
                _mm_storeu_si128(self.pixel_addrs.add(i) as *mut __m128i, pixel_addr);
            }

            for i in 0..self.star_count {
                if *self.escaped.add(i) == 0 {
                    target.pixels[*self.pixel_addrs.add(i) as usize] = *self.star_colour.add(i);
                } else {
                    T += delta / self.phase_speed;

                    let zero_to_one = RATIO * (*self.star_colour.add(i)) as f32;

                    *self.star_x.add(i) = f32::cos(T);
                    *self.star_y.add(i) = ((2.0 * zero_to_one) - 1.0)
                        + if self.perspective_on_x && self.perspective_on_y {
                            f32::sin(T)
                        } else {
                            0.0
                        };

                    let random_zero_to_one: f32 = thread_rng().sample(Open01);
                    *self.star_z.add(i) = random_zero_to_one * self.star_spread;
                }
            }
        }
    }

    fn layout(&self) -> Layout {
        let size = (self.star_count * size_of::<f32>() * 3)
            + (self.star_count * size_of::<i32>())
            + (self.star_count * size_of::<u32>() * 2);

        let align = align_of::<__m128i>();

        Layout::from_size_align(size, align).expect("Unable to create layout for star field")
    }
}

impl Drop for StarField {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.star_x as *mut u8, self.layout());
        }
    }
}

fn main() {
    let mut window = Window::new("Screen Saver", WIDTH, HEIGHT, WindowOptions::default())
        .expect("Unable to open window");

    // Limit to max ~60 fps update rate
    window.limit_update_rate(Some(std::time::Duration::from_micros(16600)));

    let mut bitmap = Bitmap::new(WIDTH, HEIGHT);
    let mut field = StarField::new();
    let mut delta = 0.0;

    while window.is_open() {
        let start = Instant::now();

        bitmap.clear();
        field.update_and_render(&mut bitmap, delta);

        window
            .update_with_buffer(bitmap.pixels(), bitmap.width, bitmap.height)
            .expect("Unable to update window");

        delta = start.elapsed().as_millis() as f32 / 1000.0;
    }
}
