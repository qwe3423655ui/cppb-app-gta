// ==================== EMBEDDED HEADER: core.h ====================
// NEON CITY - core math / utility types (no external dependencies)
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

constexpr float PI = 3.14159265358979f;

struct V2 { float x = 0, z = 0; };           // ground-plane vector (x east, z south)
inline V2 operator+(V2 a, V2 b) { return {a.x + b.x, a.z + b.z}; }
inline V2 operator-(V2 a, V2 b) { return {a.x - b.x, a.z - b.z}; }
inline V2 operator*(V2 a, float s) { return {a.x * s, a.z * s}; }
inline float dot(V2 a, V2 b) { return a.x * b.x + a.z * b.z; }
inline float len(V2 a) { return std::sqrt(dot(a, a)); }
inline V2 norm(V2 a) { float l = len(a); return l > 1e-6f ? a * (1.f / l) : V2{0, 1}; }

struct V3 { float x = 0, y = 0, z = 0; };
inline V3 operator+(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline V3 operator-(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline V3 operator*(V3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
inline float dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline V3 cross(V3 a, V3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
inline float len(V3 a) { return std::sqrt(dot(a, a)); }
inline V3 norm(V3 a) { float l = len(a); return l > 1e-8f ? a * (1.f / l) : V3{0, 1, 0}; }
inline V3 v3(V2 a, float y = 0) { return {a.x, y, a.z}; }

struct Col { float r = 0, g = 0, b = 0; };   // linear-ish colour (gamma 2.0 convention)
inline Col operator+(Col a, Col b) { return {a.r + b.r, a.g + b.g, a.b + b.b}; }
inline Col operator-(Col a, Col b) { return {a.r - b.r, a.g - b.g, a.b - b.b}; }
inline Col operator*(Col a, float s) { return {a.r * s, a.g * s, a.b * s}; }
inline Col operator*(Col a, Col b) { return {a.r * b.r, a.g * b.g, a.b * b.b}; }
inline Col rgb(int r, int g, int b) { float x = r / 255.f, y = g / 255.f, z = b / 255.f; return {x * x, y * y, z * z}; }
inline Col mix(Col a, Col b, float t) { return a + (b - a) * t; }

inline float sat(float v) { return v < 0 ? 0 : (v > 1 ? 1 : v); }
inline float clampf(float v, float a, float b) { return v < a ? a : (v > b ? b : v); }
inline float lerpf(float a, float b, float t) { return a + (b - a) * t; }
inline float smooth(float a, float b, float x) { float t = sat((x - a) / (b - a)); return t * t * (3 - 2 * t); }
inline float wrapPi(float a) { a = std::fmod(a + PI, 2 * PI); if (a < 0) a += 2 * PI; return a - PI; }
inline uint32_t hash32(uint32_t x) { x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16; return x; }

struct Rng {
  uint32_t s = 2463534242u;
  uint32_t u() { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; }
  float f() { return (u() >> 8) * (1.f / 16777216.f); }
  float r(float a, float b) { return a + (b - a) * f(); }
  int i(int n) { return int(u() % uint32_t(n)); }
};

// ==================== EMBEDDED HEADER: render.h ====================
// NEON CITY - tiny multithreaded software 3D renderer (pure C++, no GPU / no libs)

enum Mat : uint8_t { M_FLAT, M_EMIT, M_ROADX, M_ROADZ, M_ISECT, M_PAVE, M_GRASS, M_FACADE };
enum DecalKind : uint8_t { D_SHADOW_BLOB, D_SHADOW_HARD, D_GLOW, D_RING, D_BEAM, D_FLAT };

struct Sky {
  Col zenith, horizon, sunCol, ambSky, ambGround, lightCol, cloud;
  V3 lightDir{0, 1, 0}, sunDir{0, 1, 0};
  float night = 0, sunVis = 1, time = 0, shadowK = 0.5f;
};

struct Camera {
  V3 pos, fwd{0, 0, 1}, right{-1, 0, 0}, up{0, 1, 0};
  float fovy = 1.05f;
};
Camera makeCamera(V3 pos, float yaw, float pitch, float fovy);   // forward = (sin y cos p, sin p, cos y cos p)

struct STri {
  float x[3], y[3];
  float x0, y0, iz0, uz0, vz0, dizx, dizy, duzx, duzy, dvzx, dvzy;
  float cr, cg, cb, lr, lg, lb;
  uint32_t seed;
  uint8_t mat;
  float ymin, ymax;
};
struct SSpr { float x, y, r, iz, cr, cg, cb, a; bool add; };
struct CV { float x, y, z, u, v; };

class Renderer {
 public:
  int W = 0, H = 0, threads = 1;
  std::vector<uint32_t> px;                 // RGBA8888 (byte order R,G,B,A)
  Camera cam;
  Sky sky;
  float tanH = 1, tanV = 1, focal = 1, cx = 0, cy = 0, fogS = 70, fogInv = 1.f / 480.f, shadowStr = 0.5f, winOn = 0.2f;
  Col fogCol, glassCol, winLight;

  void init(int w, int h);
  void begin(const Camera& c, const Sky& s);
  void end();                               // rasterise everything submitted since begin()

  // --- 3D submission (world space). Normals are outward; back faces are culled.
  void quad(V3 a, V3 b, V3 c, V3 d, V3 n, Mat m, Col base, uint32_t seed = 0, bool twoSided = false);
  void solid(const V3 q[8], Col base, Mat m = M_FLAT, uint32_t seed = 0);   // convex hexahedron (bottom 0-3, top 4-7)
  void box(V3 c, V3 r, V3 u, V3 f, V3 half, Col col, Mat m = M_FLAT);
  void decal(V3 a, V3 b, V3 c, V3 d, DecalKind k, Col col, float strength);
  void sprite(V3 pos, float radius, Col col, float intensity, bool additive = true);
  bool project(V3 p, float& sx, float& sy, float& z) const;
  bool inView(V3 c, float rad, float maxDist) const;   // bounding-sphere frustum test

  // --- 2D overlay (call after end()); colours are 0xRRGGBB display values
  void rect(int x, int y, int w, int h, uint32_t rgb, float a = 1);
  void disc(float x, float y, float r, uint32_t rgb, float a = 1);
  void ring(float x, float y, float r, float th, uint32_t rgb, float a = 1);
  void line(float x0, float y0, float x1, float y1, uint32_t rgb, float a = 1);
  void tri2(float x0, float y0, float x1, float y1, float x2, float y2, uint32_t rgb, float a = 1);
  void text(int x, int y, const char* s, int scale, uint32_t rgb, int align = 0, bool shadow = true);
  int textW(const char* s, int scale) const;

 private:
  std::vector<STri> opq, shd, add;
  std::vector<SSpr> spr;
  std::vector<float> zb;
  std::vector<uint8_t> mask, vig;
  void poly(std::vector<STri>& L, const CV* in, int n, uint8_t mat, Col base, Col lit, uint32_t seed);
  void pushTri(std::vector<STri>& L, const CV& a, const CV& b, const CV& c, uint8_t mat, Col base, Col lit, uint32_t seed);
  void band(int y0, int y1);
  void skyFill(int y0, int y1);
  template <int M> void drawOne(const STri& t, int y0, int y1);
  void put(int x, int y, uint32_t rgb, float a);
};

// ==================== EMBEDDED HEADER: world.h ====================
// NEON CITY - procedural downtown: 8x8 blocks, 9x9 signalised intersections

constexpr int NB = 8, NR = NB + 1;                       // blocks / roads per axis
constexpr float PITCH = 75.f, HALFR = 9.5f, BOUND = 322.f, SIDEW = 3.f, CURB = 0.18f, STOPD = 14.6f;
static const int DX[4] = {1, 0, -1, 0}, DZ[4] = {0, 1, 0, -1};   // 0=E(+x) 1=S(+z) 2=W 3=N
static const float LANE[2] = {2.0f, 5.4f};                         // lane centre offsets from the road axis (right-hand traffic)

inline float roadC(int k) { return float(k - NB / 2) * PITCH; }
inline V2 nodePos(int i, int j) { return {roadC(i), roadC(j)}; }
inline V2 dirOf(int d) { return {(float)DX[d], (float)DZ[d]}; }
inline V2 rightOf(int d) { return {-(float)DZ[d], (float)DX[d]}; }
inline float yawOf(int d) { return std::atan2((float)DX[d], (float)DZ[d]); }
inline bool nodeOk(int i, int j) { return i >= 0 && j >= 0 && i < NR && j < NR; }

struct Building { float x0, z0, x1, z1, h; Col wall, roof; uint32_t seed; int tiers; float setback; bool antenna; };
struct Obst { float x, z, r; uint8_t kind; };            // 0 = lamp post, 1 = tree
struct Lamp { V2 p, arm; };
struct Block {
  float x0 = 0, z0 = 0, x1 = 0, z1 = 0;
  uint8_t kind = 0;                                       // 0 buildings, 1 park, 2 garage plaza, 3 gun-shop plaza
  std::vector<Building> bld;
  std::vector<Obst> obs;
  std::vector<Lamp> lamps;
};

int sigState(int ni, int nj, int d, bool left, float time);   // 0 red, 1 yellow, 2 green
Sky makeSky(float hour, float time);                           // sun/moon, colours, fog for a time of day (0..24)

class World {
 public:
  Block blocks[NB][NB];
  V2 garage, gunShop, spawn;
  void gen(uint32_t seed);
  void draw(Renderer& R, float time) const;
  bool resolve(V2& p, float r, V2& n) const;              // push a circle out of buildings/obstacles; true if touched
  float rayWorld(V2 o, V2 dir, float maxT, bool withObs = true) const;   // distance to first building/obstacle (or maxT)
};

// ==================== EMBEDDED HEADER: game.h ====================
// NEON CITY - game logic: cars, traffic, peds, weapons, money, HUD, touch controls

struct CarDef {
  const char* name;
  float len, wid, y0, bodyTop, cabTop, nose, tail, cb0, cb1, ct0, ct1, wb, wr;   // model
  float vmax, acc, grip, hp, mass;                                               // handling
  int value;                                                                     // chop-shop price
};
extern const CarDef CARS[4];                // 0 Pixie (compact) .. 3 Specter (hyper-fast)

enum VState : uint8_t { VS_FREE, VS_PARKED, VS_AI, VS_PLAYER, VS_WRECK };
struct Vehicle {
  int type = 0;
  VState st = VS_FREE;
  V2 pos, vel;
  float yaw = 0, steer = 0, steerIn = 0, throttle = 0, brake = 0, hp = 100, pitch = 0, roll = 0, lastVf = 0, burn = 0, smokeT = 0;
  bool hand = false;
  Col color;
  // traffic AI
  int ni = 0, nj = 0, d = 0, lane = 0, move = 0, nextMove = 0, dOut = 0, exitLane = 0;
  bool inside = false, planned = false;
  V2 path[6];
  int pn = 0, pi = 0;
  float cruise = 12, stuck = 0, ghost = 0;
};

struct Ped {
  bool alive = false;
  V2 pos, fleeDir, vel;
  float yaw = 0, hp = 40, t = 0, speed = 1.4f, s = 0, off = 2.5f, stT = 0;
  int st = 0;                               // 0 walk, 1 flee, 2 down
  int bi = 0, bj = 0, dirSign = 1;
  Col shirt, pants, skin;
};
struct Pickup { bool alive = false; V2 pos; int value = 100; float t = 0; };
struct Part { V3 p, v; float life = 0, max = 1, size = 1; Col c; uint8_t kind = 0; };   // 0 spark(add) 1 smoke 2 fire(add)
struct Tracer { V3 a, b; float life = 0; };
struct Note { std::string text; float t = 0; uint32_t col = 0xFFFFFF; };

struct Weapon { const char* name; int price, ammoPack, maxAmmo, dmg, pellets; float rate, spread, range; bool melee; };
extern const Weapon WEAP[4];
std::string fmtMoney(int v);

struct Player {
  V2 pos, vel;
  float yaw = 0, hp = 100, fireCd = 0, walk = 0, deadT = 0, muzzle = 0, hurtCd = 0;
  int money = 1000, weapon = 1, car = -1;
  bool owned[4] = {true, true, false, false};
  int ammo[4] = {0, 48, 0, 0};
};

struct Input {                               // one frame of control state
  float mx = 0, my = 0;                      // stick: my > 0 = forward / throttle
  bool fire = false, brake = false, use = false, swap = false;
  float camDX = 0, camDY = 0;                // camera drag in internal pixels
};
struct Touch { int id = -1, role = 0; float x = 0, y = 0, sx = 0, sy = 0, lx = 0, ly = 0; };

// Internal render size: fixed height (quality knob), width follows the screen aspect ratio.
inline void internalSize(int winW, int winH, int& w, int& h, int baseH = 360) {
  float aspect = winH > 0 ? (float)winW / (float)winH : 16.f / 9.f;
  h = baseH;
  w = (int)std::lround(h * aspect);
  w = std::max(baseH * 4 / 3, std::min(baseH * 5 / 2, w)) & ~3;
}

class Game {
 public:
  World world;
  Rng rng;
  int W = 640, H = 360;
  float time = 0, hour = 17.4f;
  std::vector<Vehicle> cars;
  std::vector<Ped> peds;
  std::vector<Pickup> picks;
  std::vector<Part> parts;
  std::vector<Tracer> tracers;
  std::vector<Note> notes;
  Player P;
  Input kb;                                  // keyboard / debug input (platform fills it)
  float camYaw = 0.3f, camPitch = 0.28f, fov = 1.05f;
  V3 camPos;
  bool timeFrozen = false, showTouch = true;

  void init(int w, int h, uint32_t seed = 1);
  void resize(int w, int h);
  void touch(int id, int phase, float x, float y);   // phase 0 down, 1 move, 2 up (internal pixel coords)
  void update(float dt);
  void draw(Renderer& R);

  // exposed for the headless tool
  int spawnVehicle(int type, V2 pos, float yaw, VState st);
  bool spawnAI(float minD, float maxD, bool hidden);
  int spawnParked(V2 near, float radius);
  bool spawnPed(bool hidden);
  void spawnPickup();
  void notify(const std::string& s, uint32_t col = 0xFFFFFF);

 private:
  Input in;
  Touch touches[6];
  bool useEdge = false, swapEdge = false;
  float spawnT = 0, trauma = 0, stickX = 0, stickY = 0, stickDX = 0, stickDY = 0, camAccX = 0, camAccY = 0;
  std::string useLabel = "USE";
  V3 camFocus;
  std::string prompt;
  int promptKind = 0;                        // 0 none, 1 enter, 2 sell, 3 shop

  void gatherInput();
  void updatePlayer(float dt);
  void updateVehicles(float dt);
  void stepVehicle(Vehicle& v, float dt);
  void aiDrive(Vehicle& v, int idx, float dt);
  void planPath(Vehicle& v);
  float obstacleLimit(const Vehicle& v, V2 fwd, V2 rt, float vf, int self);
  void collide(float dt);
  void updatePeds(float dt);
  void updateFX(float dt);
  void updateCamera(float dt);
  void manageSpawns(float dt);
  void firePlayer(float dt);
  bool hitScan(V2 o, V2 dir, float range, float dmg, bool fromPlayer, V3* endOut);
  void hurtPed(Ped& p, float dmg, V2 from);
  void damageCar(int idx, float dmg);
  void explode(int idx);
  void enterOrUse();
  void exitCar();
  int nearestCar(float reach);
  void emit(uint8_t kind, V3 p, V3 v, float life, float size, Col c);
  void killPlayer();

  void drawCar(Renderer& R, const Vehicle& v);
  void drawPerson(Renderer& R, V2 pos, float yaw, float phase, Col shirt, Col pants, Col skin, bool down, bool gun);
  void drawHUD(Renderer& R);
  void drawTouchUI(Renderer& R);
};


// ==================== EMBEDDED SOURCE: render.cpp ====================
#include <cctype>
#include <thread>

// ---------------------------------------------------------------- tables
static float g_noise[64 * 64];
static bool g_ready = false;
static void initTables() {
  if (g_ready) return;
  g_ready = true;
  Rng r; r.s = 12345u;
  for (float& f : g_noise) f = r.f();
}
static float vnoise(float x, float y) {
  int xi = (int)std::floor(x), yi = (int)std::floor(y);
  float fx = x - xi, fy = y - yi;
  fx = fx * fx * (3 - 2 * fx); fy = fy * fy * (3 - 2 * fy);
  auto n = [&](int a, int b) { return g_noise[((b & 63) << 6) | (a & 63)]; };
  return lerpf(lerpf(n(xi, yi), n(xi + 1, yi), fx), lerpf(n(xi, yi + 1), n(xi + 1, yi + 1), fx), fy);
}
static inline float fract(float x) { return x - std::floor(x); }
static inline uint32_t pack(Col c) {
  uint32_t r = (uint32_t)(std::sqrt(sat(c.r)) * 255.f + .5f), g = (uint32_t)(std::sqrt(sat(c.g)) * 255.f + .5f),
           b = (uint32_t)(std::sqrt(sat(c.b)) * 255.f + .5f);
  return 0xFF000000u | (b << 16) | (g << 8) | r;
}

Camera makeCamera(V3 pos, float yaw, float pitch, float fovy) {
  Camera c;
  c.pos = pos; c.fovy = fovy;
  c.fwd = {std::sin(yaw) * std::cos(pitch), std::sin(pitch), std::cos(yaw) * std::cos(pitch)};
  c.right = {-std::cos(yaw), 0, std::sin(yaw)};
  c.up = cross(c.right, c.fwd);
  return c;
}

// ---------------------------------------------------------------- setup
void Renderer::init(int w, int h) {
  initTables();
  W = w; H = h;
  px.assign(size_t(w) * h, 0xFF000000u);
  zb.assign(size_t(w) * h, 0.f);
  mask.assign(size_t(w) * h, 0);
  vig.resize(size_t(w) * h);
  for (int y = 0; y < h; y++)
    for (int x = 0; x < w; x++) {
      float nx = (x + .5f) / w * 2 - 1, ny = (y + .5f) / h * 2 - 1;
      vig[size_t(y) * w + x] = (uint8_t)(255.f * clampf(1.f - 0.26f * (nx * nx * 0.7f + ny * ny * 0.9f), 0.f, 1.f));
    }
  unsigned hc = std::thread::hardware_concurrency();
  threads = hc >= 4 ? 4 : (hc >= 2 ? 2 : 1);
}

void Renderer::begin(const Camera& c, const Sky& s) {
  cam = c; sky = s;
  focal = H * 0.5f / std::tan(cam.fovy * 0.5f);
  cx = W * 0.5f; cy = H * 0.5f;
  tanV = std::tan(cam.fovy * 0.5f); tanH = tanV * W / H;
  opq.clear(); shd.clear(); add.clear(); spr.clear();
  fogCol = sky.horizon;
  glassCol = mix(sky.horizon, sky.zenith, 0.35f) * 0.6f;
  winOn = 0.07f + 0.55f * sky.night;
  winLight = {1.35f, 0.9f, 0.5f};
  shadowStr = sky.shadowK;
}

// ---------------------------------------------------------------- submission
static int clipNear(const CV* in, int n, CV* out, float zn) {
  int m = 0;
  for (int i = 0; i < n; i++) {
    const CV& a = in[i];
    const CV& b = in[i + 1 == n ? 0 : i + 1];
    bool ai = a.z >= zn, bi = b.z >= zn;
    if (ai) out[m++] = a;
    if (ai != bi) {
      float t = (zn - a.z) / (b.z - a.z);
      out[m++] = {a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, zn, a.u + (b.u - a.u) * t, a.v + (b.v - a.v) * t};
    }
  }
  return m;
}

void Renderer::pushTri(std::vector<STri>& L, const CV& a, const CV& b, const CV& c, uint8_t mat, Col base, Col lit, uint32_t seed) {
  STri t;
  const CV* v[3] = {&a, &b, &c};
  float iz[3], uz[3], vz[3];
  for (int i = 0; i < 3; i++) {
    iz[i] = 1.f / v[i]->z;
    t.x[i] = cx + focal * v[i]->x * iz[i];
    t.y[i] = cy - focal * v[i]->y * iz[i];
    uz[i] = v[i]->u * iz[i];
    vz[i] = v[i]->v * iz[i];
  }
  float xmin = std::min({t.x[0], t.x[1], t.x[2]}), xmax = std::max({t.x[0], t.x[1], t.x[2]});
  t.ymin = std::min({t.y[0], t.y[1], t.y[2]}); t.ymax = std::max({t.y[0], t.y[1], t.y[2]});
  if (!(xmax > 0 && xmin < W && t.ymax > 0 && t.ymin < H)) return;   // also rejects NaN
  float area = (t.x[1] - t.x[0]) * (t.y[2] - t.y[0]) - (t.x[2] - t.x[0]) * (t.y[1] - t.y[0]);
  if (std::fabs(area) < 1e-3f) return;
  float inv = 1.f / area;
  auto grad = [&](const float* f, float& gx, float& gy) {
    float d1 = f[1] - f[0], d2 = f[2] - f[0];
    gx = (d1 * (t.y[2] - t.y[0]) - d2 * (t.y[1] - t.y[0])) * inv;
    gy = (d2 * (t.x[1] - t.x[0]) - d1 * (t.x[2] - t.x[0])) * inv;
  };
  t.x0 = t.x[0]; t.y0 = t.y[0]; t.iz0 = iz[0]; t.uz0 = uz[0]; t.vz0 = vz[0];
  grad(iz, t.dizx, t.dizy); grad(uz, t.duzx, t.duzy); grad(vz, t.dvzx, t.dvzy);
  t.cr = base.r; t.cg = base.g; t.cb = base.b; t.lr = lit.r; t.lg = lit.g; t.lb = lit.b;
  t.seed = seed; t.mat = mat;
  L.push_back(t);
}

void Renderer::poly(std::vector<STri>& L, const CV* in, int n, uint8_t mat, Col base, Col lit, uint32_t seed) {
  CV out[8];
  int m = clipNear(in, n, out, 0.15f);
  for (int i = 1; i + 1 < m; i++) pushTri(L, out[0], out[i], out[i + 1], mat, base, lit, seed);
}

void Renderer::quad(V3 a, V3 b, V3 c, V3 d, V3 n, Mat m, Col base, uint32_t seed, bool two) {
  if (!two && dot(n, cam.pos - a) <= 0.f) return;
  Col lit{1, 1, 1};
  if (m != M_EMIT) {
    float nl = std::max(0.f, dot(n, sky.lightDir));
    lit = mix(sky.ambGround, sky.ambSky, 0.5f + 0.5f * n.y) + sky.lightCol * nl;
  }
  V3 P[4] = {a, b, c, d};
  CV v[4];
  for (int i = 0; i < 4; i++) {
    V3 dd = P[i] - cam.pos;
    v[i] = {dot(dd, cam.right), dot(dd, cam.up), dot(dd, cam.fwd), 0, 0};
    if (m == M_ROADX || m == M_ROADZ || m == M_ISECT || m == M_PAVE || m == M_GRASS) { v[i].u = P[i].x; v[i].v = P[i].z; }
    else if (m == M_FACADE) { v[i].u = std::fabs(n.x) > std::fabs(n.z) ? P[i].z : P[i].x; v[i].v = P[i].y; }
  }
  if (m == M_FLAT || m == M_EMIT) { base = base * lit; lit = {1, 1, 1}; }
  poly(opq, v, 4, m, base, lit, seed);
}

void Renderer::solid(const V3 q[8], Col base, Mat m, uint32_t seed) {
  static const int F[6][4] = {{0, 1, 2, 3}, {4, 5, 6, 7}, {0, 1, 5, 4}, {1, 2, 6, 5}, {2, 3, 7, 6}, {3, 0, 4, 7}};
  V3 cen;
  for (int i = 0; i < 8; i++) cen = cen + q[i];
  cen = cen * 0.125f;
  for (const auto& f : F) {
    V3 a = q[f[0]], b = q[f[1]], c = q[f[2]], d = q[f[3]];
    V3 n = cross(b - a, d - a);
    float l = len(n);
    if (l < 1e-7f) continue;
    n = n * (1.f / l);
    if (dot(n, (a + b + c + d) * 0.25f - cen) < 0) n = n * -1.f;
    quad(a, b, c, d, n, m, base, seed);
  }
}

void Renderer::box(V3 c, V3 r, V3 u, V3 f, V3 h, Col col, Mat m) {
  static const float sx[4] = {-1, 1, 1, -1}, sz[4] = {-1, -1, 1, 1};
  V3 q[8];
  int k = 0;
  for (int yy = 0; yy < 2; yy++)
    for (int i = 0; i < 4; i++) q[k++] = c + r * (sx[i] * h.x) + u * ((yy ? 1.f : -1.f) * h.y) + f * (sz[i] * h.z);
  solid(q, col, m);
}

void Renderer::decal(V3 a, V3 b, V3 c, V3 d, DecalKind k, Col col, float strength) {
  V3 P[4] = {a, b, c, d};
  float U[4] = {-1, 1, 1, -1}, Vv[4] = {k == D_BEAM ? 0.f : -1.f, k == D_BEAM ? 0.f : -1.f, 1, 1};
  CV v[4];
  for (int i = 0; i < 4; i++) {
    V3 dd = P[i] - cam.pos;
    v[i] = {dot(dd, cam.right), dot(dd, cam.up), dot(dd, cam.fwd), U[i], Vv[i]};
  }
  bool isShadow = (k == D_SHADOW_BLOB || k == D_SHADOW_HARD);
  poly(isShadow ? shd : add, v, 4, k, col, Col{strength, strength, strength}, 0);
}

void Renderer::sprite(V3 pos, float radius, Col col, float intensity, bool additive) {
  V3 dd = pos - cam.pos;
  float z = dot(dd, cam.fwd);
  if (z < 0.6f) return;
  float iz = 1.f / (z - 0.3f);
  float x = cx + focal * dot(dd, cam.right) / z, y = cy - focal * dot(dd, cam.up) / z;
  float r = std::min(260.f, radius * focal / z);
  if (r < 0.7f || x + r < 0 || x - r > W || y + r < 0 || y - r > H) return;
  spr.push_back({x, y, r, iz, col.r, col.g, col.b, intensity, additive});
}

bool Renderer::project(V3 p, float& sx, float& sy, float& z) const {
  V3 dd = p - cam.pos;
  z = dot(dd, cam.fwd);
  if (z < 0.3f) return false;
  sx = cx + focal * dot(dd, cam.right) / z;
  sy = cy - focal * dot(dd, cam.up) / z;
  return true;
}

bool Renderer::inView(V3 c, float rad, float maxDist) const {
  V3 d = c - cam.pos;
  float zc = dot(d, cam.fwd);
  if (zc < -rad || dot(d, d) > (maxDist + rad) * (maxDist + rad)) return false;
  float zz = std::max(zc, 0.f);
  if (std::fabs(dot(d, cam.right)) > zz * tanH + rad * 1.4f) return false;
  if (std::fabs(dot(d, cam.up)) > zz * tanV + rad * 1.4f) return false;
  return true;
}

// ---------------------------------------------------------------- scanline walker
template <class F>
static inline void walk(const STri& t, int y0, int y1, int W, F&& f) {
  int ya = std::max(y0, (int)std::ceil(t.ymin - 0.5f));
  int yb = std::min(y1 - 1, (int)std::ceil(t.ymax - 0.5f) - 1);
  for (int y = ya; y <= yb; y++) {
    float yc = y + 0.5f, xl = 1e30f, xr = -1e30f;
    for (int e = 0; e < 3; e++) {
      int j = e == 2 ? 0 : e + 1;
      float ay = t.y[e], by = t.y[j];
      if ((ay <= yc && yc < by) || (by <= yc && yc < ay)) {
        float x = t.x[e] + (yc - ay) * (t.x[j] - t.x[e]) / (by - ay);
        xl = std::min(xl, x); xr = std::max(xr, x);
      }
    }
    if (xl > xr) continue;
    float fxs = std::ceil(xl - 0.5f), fxe = std::ceil(xr - 0.5f) - 1.f;
    if (fxs < 0) fxs = 0;
    if (fxe > W - 1) fxe = (float)(W - 1);
    int xs = (int)fxs, xe = (int)fxe;
    if (xs > xe) continue;
    float iz = t.iz0 + t.dizy * (yc - t.y0) + t.dizx * (xs + 0.5f - t.x0);
    for (int x = xs; x <= xe; x++, iz += t.dizx) f(x, y, iz);
  }
}

// ---------------------------------------------------------------- procedural materials
static const Col C_YELLOW = rgb(235, 185, 40), C_WHITE = rgb(228, 228, 222);

template <int M>
static inline Col shade(const Renderer& R, const STri& t, float u, float v, float d) {
  Col base{t.cr, t.cg, t.cb}, lit{t.lr, t.lg, t.lb};
  float k = sat(1.45f - d * (1.f / 130.f));          // detail fades with distance (anti-shimmer)
  if constexpr (M == M_ROADX || M == M_ROADZ || M == M_ISECT) {
    float a = (M == M_ROADZ) ? v : u, b = (M == M_ROADZ) ? u : v;
    uint32_t h = hash32((uint32_t)(int)std::floor(a * 0.8f) * 73856093u ^ (uint32_t)(int)std::floor(b * 0.8f) * 19349663u);
    Col c = base * (0.92f + 0.08f * float(h & 255) * (1.f / 255.f));
    if constexpr (M != M_ISECT) {
      float db = std::fabs(b - std::floor(b * (1.f / 75.f) + .5f) * 75.f);
      float da = std::fabs(a - std::floor(a * (1.f / 75.f) + .5f) * 75.f);
      bool yel = db > 0.14f && db < 0.46f, wht = false;
      if (!yel) {
        if (db > 3.62f && db < 3.84f && da > 16.f && fract(a * (1.f / 9.f)) < 0.55f) wht = true;
        else if (db > 7.0f && db < 7.2f) wht = true;
        else if (da > 10.5f && da < 13.5f && db < 8.8f && fract(db * (1.f / 1.5f)) < 0.55f) wht = true;
        else if (da > 14.0f && da < 14.55f && db > 0.6f && db < 7.0f) wht = true;
      }
      if (yel || wht) c = mix(c, yel ? C_YELLOW : C_WHITE, k);
    }
    return c * lit;
  } else if constexpr (M == M_PAVE) {
    uint32_t h = hash32((uint32_t)(int)std::floor(u * 0.625f) * 73856093u ^ (uint32_t)(int)std::floor(v * 0.625f) * 19349663u);
    Col c = base * (0.93f + 0.07f * float(h & 255) * (1.f / 255.f));
    if (fract(u * 0.625f) < 0.04f || fract(v * 0.625f) < 0.04f) c = mix(c, c * 0.78f, k);
    return c * lit;
  } else if constexpr (M == M_GRASS) {
    uint32_t h = hash32((uint32_t)(int)std::floor(u * 0.45f) * 73856093u ^ (uint32_t)(int)std::floor(v * 0.45f) * 19349663u);
    uint32_t h2 = hash32((uint32_t)(int)std::floor(u * 0.09f) * 83492791u ^ (uint32_t)(int)std::floor(v * 0.09f) * 2654435761u);
    float n = (0.82f + 0.36f * float(h & 255) * (1.f / 255.f)) * (0.88f + 0.24f * float(h2 & 255) * (1.f / 255.f));
    return base * (1.f + (n - 1.f) * k) * lit;
  } else {  // M_FACADE
    const float PU = 3.2f, PV = 3.6f;
    float cu = u * (1.f / PU), cv = v * (1.f / PV);
    int iu = (int)std::floor(cu), iv = (int)std::floor(cv);
    float fu = cu - iu, fv = cv - iv;
    Col wall = base * lit;
    Col avg = mix(wall, R.glassCol, 0.30f) + R.winLight * (0.12f * R.winOn);
    if (k <= 0.f) return avg;
    bool shop = v < 3.9f;
    bool win = shop ? (fu > 0.08f && fu < 0.92f && fv > 0.10f && fv < 0.72f) : (fu > 0.2f && fu < 0.8f && fv > 0.2f && fv < 0.8f);
    uint32_t h = hash32((uint32_t)(iu * 374761393 + iv * 668265263) ^ t.seed);
    Col c;
    if (win) {
      bool on = (h & 1023) < (uint32_t)(R.winOn * (shop ? 1.3f : 1.f) * 1023.f);
      if (on) c = R.winLight * ((0.35f + 0.65f * R.sky.night) * (0.65f + 0.6f * float((h >> 10) & 255) * (1.f / 255.f)));
      else c = R.glassCol * (0.55f + 0.55f * (1.f - fv)) + lit * 0.02f;
    } else {
      c = wall * (fv < 0.10f ? 0.8f : 1.f);
    }
    return avg + (c - avg) * k;
  }
}

template <int M>
void Renderer::drawOne(const STri& t, int y0, int y1) {
  float* zbuf = zb.data();
  uint32_t* pix = px.data();
  const int Wd = W;
  const float fS = fogS, fI = fogInv;
  const Col fc = fogCol;
  walk(t, y0, y1, Wd, [&](int x, int y, float iz) {
    int idx = y * Wd + x;
    if (iz <= zbuf[idx]) return;
    zbuf[idx] = iz;
    float d = 1.f / iz;
    Col c;
    if constexpr (M == M_FLAT || M == M_EMIT) c = {t.cr, t.cg, t.cb};
    else {
      float xc = x + 0.5f - t.x0, yc = y + 0.5f - t.y0;
      float uz = t.uz0 + t.duzx * xc + t.duzy * yc, vz = t.vz0 + t.dvzx * xc + t.dvzy * yc;
      c = shade<M>(*this, t, uz * d, vz * d, d);
    }
    float f = sat((d - fS) * fI);
    f *= f;
    c = {c.r + (fc.r - c.r) * f, c.g + (fc.g - c.g) * f, c.b + (fc.b - c.b) * f};
    pix[idx] = pack(c);
  });
}

// ---------------------------------------------------------------- sky
void Renderer::skyFill(int y0, int y1) {
  const float invF = 1.f / focal;
  for (int y = y0; y < y1; y++) {
    float ry = (cy - (y + 0.5f)) * invF;
    for (int x = 0; x < W; x++) {
      size_t idx = size_t(y) * W + x;
      if (zb[idx] > 0.f) continue;
      float rx = ((x + 0.5f) - cx) * invF;
      float il = 1.f / std::sqrt(1.f + rx * rx + ry * ry);
      V3 dir = (cam.fwd + cam.right * rx + cam.up * ry) * il;
      float e = dir.y;
      Col c = mix(sky.horizon, sky.zenith, std::sqrt(sat(e)));
      float sd = dot(dir, sky.sunDir);
      if (sd > 0.6f) {
        float s2 = sd * sd, s4 = s2 * s2, s8 = s4 * s4, s16 = s8 * s8, s32 = s16 * s16;
        float glow = s8 * 0.16f + s32 * 0.35f + (sd > 0.99985f ? 5.f : 0.f);
        c = c + sky.sunCol * (glow * sky.sunVis);
      }
      if (e > 0.03f) {
        float inv = 1.f / e;
        float cu = dir.x * inv * 0.5f + sky.time * 0.004f, cv = dir.z * inv * 0.5f;
        float n = vnoise(cu * 2.f, cv * 2.f) * 0.55f + vnoise(cu * 4.3f, cv * 4.3f) * 0.3f + vnoise(cu * 9.1f + 7.f, cv * 9.1f) * 0.15f;
        float cl = smooth(0.50f, 0.76f, n) * smooth(0.03f, 0.25f, e);
        c = mix(c, sky.cloud * (0.8f + 0.4f * sat(sd)), cl * 0.85f);
      }
      if (sky.night > 0.35f && e > 0.04f) {
        uint32_t h = hash32((uint32_t)(int)std::floor(dir.x * 170.f) * 73856093u ^ (uint32_t)(int)std::floor(dir.z * 170.f) * 19349663u ^ (uint32_t)(int)std::floor(dir.y * 170.f) * 83492791u);
        if ((h & 511) == 0) { float b = 0.5f + 0.5f * float((h >> 9) & 255) * (1.f / 255.f); c = c + Col{b, b, b * 1.1f} * ((sky.night - 0.35f) * 1.2f); }
      }
      px[idx] = pack(c);
    }
  }
}

// ---------------------------------------------------------------- one horizontal band (thread-safe, rows [y0,y1))
void Renderer::band(int y0, int y1) {
  std::fill(zb.begin() + size_t(y0) * W, zb.begin() + size_t(y1) * W, 0.f);
  std::fill(mask.begin() + size_t(y0) * W, mask.begin() + size_t(y1) * W, (uint8_t)0);
  for (const STri& t : opq) {
    if (t.ymax < y0 || t.ymin > y1) continue;
    switch (t.mat) {
      case M_FLAT: drawOne<M_FLAT>(t, y0, y1); break;
      case M_EMIT: drawOne<M_EMIT>(t, y0, y1); break;
      case M_ROADX: drawOne<M_ROADX>(t, y0, y1); break;
      case M_ROADZ: drawOne<M_ROADZ>(t, y0, y1); break;
      case M_ISECT: drawOne<M_ISECT>(t, y0, y1); break;
      case M_PAVE: drawOne<M_PAVE>(t, y0, y1); break;
      case M_GRASS: drawOne<M_GRASS>(t, y0, y1); break;
      default: drawOne<M_FACADE>(t, y0, y1); break;
    }
  }
  skyFill(y0, y1);

  // soft shadows -> max-mask, applied once (no double darkening)
  for (const STri& t : shd) {
    if (t.ymax < y0 || t.ymin > y1) continue;
    const bool blob = t.mat == D_SHADOW_BLOB;
    const float strength = t.lr * 255.f;
    walk(t, y0, y1, W, [&](int x, int y, float iz) {
      size_t idx = size_t(y) * W + x;
      if (iz < zb[idx] * 0.998f) return;
      float a = 1.f;
      if (blob) {
        float d = 1.f / iz, xc = x + 0.5f - t.x0, yc = y + 0.5f - t.y0;
        float u = (t.uz0 + t.duzx * xc + t.duzy * yc) * d, v = (t.vz0 + t.dvzx * xc + t.dvzy * yc) * d;
        float q = u * u + v * v;
        if (q >= 1.f) return;
        a = 1.f - q; a = std::min(1.f, a * 1.6f);
      }
      uint8_t m8 = (uint8_t)std::min(255.f, a * strength);
      if (m8 > mask[idx]) mask[idx] = m8;
    });
  }
  const float sk = shadowStr / 255.f;
  for (int y = y0; y < y1; y++)
    for (int x = 0; x < W; x++) {
      size_t idx = size_t(y) * W + x;
      uint8_t m = mask[idx];
      if (!m) continue;
      float s = 1.f - m * sk;
      uint32_t p = px[idx];
      uint32_t r = (uint32_t)((p & 255) * s), g = (uint32_t)(((p >> 8) & 255) * s), b = (uint32_t)(((p >> 16) & 255) * s);
      px[idx] = 0xFF000000u | (b << 16) | (g << 8) | r;
    }

  // additive decals (glow pools, rings, beams, tracers)
  for (const STri& t : add) {
    if (t.ymax < y0 || t.ymin > y1) continue;
    walk(t, y0, y1, W, [&](int x, int y, float iz) {
      size_t idx = size_t(y) * W + x;
      if (iz < zb[idx] * 0.998f) return;
      float d = 1.f / iz, xc = x + 0.5f - t.x0, yc = y + 0.5f - t.y0;
      float u = (t.uz0 + t.duzx * xc + t.duzy * yc) * d, v = (t.vz0 + t.dvzx * xc + t.dvzy * yc) * d;
      float a;
      switch (t.mat) {
        case D_GLOW: { float q = u * u + v * v; if (q >= 1.f) return; a = (1.f - q) * (1.f - q); break; }
        case D_RING: { float r = std::sqrt(u * u + v * v); if (r > 1.f) return; a = sat(1.f - std::fabs(r - 0.9f) * 16.f) + (r < 0.9f ? 0.10f * (1.f - r) : 0.f); break; }
        case D_BEAM: { float q = 1.f - std::fabs(u); a = q * q * (1.f - v) * 0.8f; if (a <= 0) return; break; }
        default: a = 1.f;
      }
      a *= t.lr;
      uint32_t p = px[idx];
      float r = (p & 255) + t.cr * 255.f * a, g = ((p >> 8) & 255) + t.cg * 255.f * a, b = ((p >> 16) & 255) + t.cb * 255.f * a;
      px[idx] = 0xFF000000u | ((uint32_t)std::min(255.f, b) << 16) | ((uint32_t)std::min(255.f, g) << 8) | (uint32_t)std::min(255.f, r);
    });
  }

  // sprites (glows, smoke, muzzle flashes)
  for (const SSpr& s : spr) {
    int ya = std::max(y0, (int)std::floor(s.y - s.r)), yb = std::min(y1 - 1, (int)std::ceil(s.y + s.r));
    int xa = std::max(0, (int)std::floor(s.x - s.r)), xb = std::min(W - 1, (int)std::ceil(s.x + s.r));
    float ir2 = 1.f / (s.r * s.r);
    for (int y = ya; y <= yb; y++)
      for (int x = xa; x <= xb; x++) {
        float dx = x + 0.5f - s.x, dy = y + 0.5f - s.y, q = (dx * dx + dy * dy) * ir2;
        if (q >= 1.f) continue;
        size_t idx = size_t(y) * W + x;
        if (s.iz <= zb[idx]) continue;
        float a = (1.f - q) * (1.f - q) * s.a;
        uint32_t p = px[idx];
        float r = p & 255, g = (p >> 8) & 255, b = (p >> 16) & 255;
        if (s.add) { r += s.cr * 255.f * a; g += s.cg * 255.f * a; b += s.cb * 255.f * a; }
        else { a = sat(a); r += (s.cr * 255.f - r) * a; g += (s.cg * 255.f - g) * a; b += (s.cb * 255.f - b) * a; }
        px[idx] = 0xFF000000u | ((uint32_t)std::min(255.f, b) << 16) | ((uint32_t)std::min(255.f, g) << 8) | (uint32_t)std::min(255.f, r);
      }
  }

  // vignette
  for (int y = y0; y < y1; y++)
    for (int x = 0; x < W; x++) {
      size_t idx = size_t(y) * W + x;
      uint32_t p = px[idx], v = vig[idx];
      uint32_t r = ((p & 255) * v) >> 8, g = (((p >> 8) & 255) * v) >> 8, b = (((p >> 16) & 255) * v) >> 8;
      px[idx] = 0xFF000000u | (b << 16) | (g << 8) | r;
    }
}

void Renderer::end() {
  int n = std::max(1, std::min(threads, H / 32));
  if (n == 1) { band(0, H); return; }
  std::vector<std::thread> th;
  for (int i = 1; i < n; i++) th.emplace_back([this, i, n] { band(H * i / n, H * (i + 1) / n); });
  band(0, H / n);
  for (auto& t : th) t.join();
}

// ---------------------------------------------------------------- 2D overlay
void Renderer::put(int x, int y, uint32_t c, float a) {
  if (x < 0 || y < 0 || x >= W || y >= H || a <= 0.f) return;
  uint32_t& p = px[size_t(y) * W + x];
  uint32_t sr = (c >> 16) & 255, sg = (c >> 8) & 255, sb = c & 255;
  if (a >= 1.f) { p = 0xFF000000u | (sb << 16) | (sg << 8) | sr; return; }
  float dr = p & 255, dg = (p >> 8) & 255, db = (p >> 16) & 255;
  p = 0xFF000000u | ((uint32_t)(db + (sb - db) * a) << 16) | ((uint32_t)(dg + (sg - dg) * a) << 8) | (uint32_t)(dr + (sr - dr) * a);
}
void Renderer::rect(int x, int y, int w, int h, uint32_t c, float a) {
  for (int j = y; j < y + h; j++) for (int i = x; i < x + w; i++) put(i, j, c, a);
}
void Renderer::disc(float x, float y, float r, uint32_t c, float a) {
  for (int j = (int)std::floor(y - r - 1); j <= (int)std::ceil(y + r + 1); j++)
    for (int i = (int)std::floor(x - r - 1); i <= (int)std::ceil(x + r + 1); i++) {
      float d = std::sqrt((i + .5f - x) * (i + .5f - x) + (j + .5f - y) * (j + .5f - y));
      put(i, j, c, a * sat(r + 0.5f - d));
    }
}
void Renderer::ring(float x, float y, float r, float th, uint32_t c, float a) {
  float o = r + th * 0.5f;
  for (int j = (int)std::floor(y - o - 1); j <= (int)std::ceil(y + o + 1); j++)
    for (int i = (int)std::floor(x - o - 1); i <= (int)std::ceil(x + o + 1); i++) {
      float d = std::sqrt((i + .5f - x) * (i + .5f - x) + (j + .5f - y) * (j + .5f - y));
      put(i, j, c, a * sat(th * 0.5f + 0.5f - std::fabs(d - r)));
    }
}
void Renderer::line(float x0, float y0, float x1, float y1, uint32_t c, float a) {
  int n = (int)std::max(std::fabs(x1 - x0), std::fabs(y1 - y0)) + 1;
  for (int i = 0; i <= n; i++) {
    float t = float(i) / n;
    put((int)std::floor(x0 + (x1 - x0) * t), (int)std::floor(y0 + (y1 - y0) * t), c, a);
  }
}
void Renderer::tri2(float x0, float y0, float x1, float y1, float x2, float y2, uint32_t c, float a) {
  int xa = (int)std::floor(std::min({x0, x1, x2})), xb = (int)std::ceil(std::max({x0, x1, x2}));
  int ya = (int)std::floor(std::min({y0, y1, y2})), yb = (int)std::ceil(std::max({y0, y1, y2}));
  float area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
  if (std::fabs(area) < 1e-4f) return;
  for (int j = ya; j <= yb; j++)
    for (int i = xa; i <= xb; i++) {
      float px_ = i + .5f, py_ = j + .5f;
      float w0 = ((x1 - px_) * (y2 - py_) - (x2 - px_) * (y1 - py_)) / area;
      float w1 = ((x2 - px_) * (y0 - py_) - (x0 - px_) * (y2 - py_)) / area;
      float w2 = 1.f - w0 - w1;
      if (w0 >= 0 && w1 >= 0 && w2 >= 0) put(i, j, c, a);
    }
}

// 5x7 bitmap font (uppercase, digits, punctuation)
static const char FCH[] = " ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,:-+$!?/%()<>=*'";
static const uint8_t FONT[][7] = {
  {0, 0, 0, 0, 0, 0, 0},
  {0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11}, {0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E}, {0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E},
  {0x1E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1E}, {0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F}, {0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10},
  {0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F}, {0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11}, {0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E},
  {0x07, 0x02, 0x02, 0x02, 0x02, 0x12, 0x0C}, {0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11}, {0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F},
  {0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11}, {0x11, 0x11, 0x19, 0x15, 0x13, 0x11, 0x11}, {0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E},
  {0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10}, {0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D}, {0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11},
  {0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E}, {0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04}, {0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E},
  {0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04}, {0x11, 0x11, 0x11, 0x15, 0x15, 0x15, 0x0A}, {0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11},
  {0x11, 0x11, 0x11, 0x0A, 0x04, 0x04, 0x04}, {0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F},
  {0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E}, {0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E}, {0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F},
  {0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E}, {0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02}, {0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E},
  {0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E}, {0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08}, {0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E},
  {0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C},
  {0, 0, 0, 0, 0, 0x0C, 0x0C}, {0, 0, 0, 0, 0x0C, 0x04, 0x08}, {0, 0x0C, 0x0C, 0, 0x0C, 0x0C, 0}, {0, 0, 0, 0x1F, 0, 0, 0},
  {0, 0x04, 0x04, 0x1F, 0x04, 0x04, 0}, {0x04, 0x0F, 0x14, 0x0E, 0x05, 0x1E, 0x04}, {0x04, 0x04, 0x04, 0x04, 0x04, 0, 0x04},
  {0x0E, 0x11, 0x01, 0x02, 0x04, 0, 0x04}, {0x01, 0x01, 0x02, 0x04, 0x08, 0x10, 0x10}, {0x18, 0x19, 0x02, 0x04, 0x08, 0x13, 0x03},
  {0x02, 0x04, 0x08, 0x08, 0x08, 0x04, 0x02}, {0x08, 0x04, 0x02, 0x02, 0x02, 0x04, 0x08}, {0x02, 0x04, 0x08, 0x10, 0x08, 0x04, 0x02},
  {0x08, 0x04, 0x02, 0x01, 0x02, 0x04, 0x08}, {0, 0, 0x1F, 0, 0x1F, 0, 0}, {0, 0x04, 0x15, 0x0E, 0x15, 0x04, 0}, {0x04, 0x04, 0x08, 0, 0, 0, 0}};
static_assert(sizeof(FONT) / 7 == sizeof(FCH) - 1, "font table must match character list");

int Renderer::textW(const char* s, int scale) const {
  int n = (int)std::strlen(s);
  return n ? n * 6 * scale - scale : 0;
}

void Renderer::text(int x, int y, const char* s, int scale, uint32_t col, int align, bool shadow) {
  int w = textW(s, scale);
  if (align == 1) x -= w / 2; else if (align == 2) x -= w;
  for (int pass = shadow ? 0 : 1; pass < 2; pass++) {
    int ox = pass == 0 ? scale : 0, oy = pass == 0 ? scale : 0;
    uint32_t c = pass == 0 ? 0x000000u : col;
    float a = pass == 0 ? 0.65f : 1.f;
    int cx_ = x;
    for (const char* p = s; *p; p++, cx_ += 6 * scale) {
      char ch = (char)std::toupper((unsigned char)*p);
      const char* f = std::strchr(FCH, ch);
      if (!f || !ch) continue;
      const uint8_t* g = FONT[f - FCH];
      for (int r = 0; r < 7; r++)
        for (int b = 0; b < 5; b++)
          if (g[r] & (0x10 >> b)) rect(cx_ + b * scale + ox, y + r * scale + oy, scale, scale, c, a);
    }
  }
}


// ==================== EMBEDDED SOURCE: world.cpp ====================

int sigState(int ni, int nj, int d, bool left, float time) {
  (void)left;                                   // two-phase signal: left turns are permissive (they yield to oncoming traffic)
  float t = std::fmod(time + float((ni * 7 + nj * 13) % 26), 26.f);
  float u = t - ((d & 1) ? 0.f : 13.f);         // N/S green first, then E/W
  if (u < 0.f) u += 26.f;
  if (u < 10.f) return 2;
  if (u < 12.f) return 1;
  return 0;
}

Sky makeSky(float hour, float time) {
  Sky s;
  float phi = (hour - 6.f) / 24.f * 2.f * PI;
  V3 sun = norm(V3{std::cos(phi), std::sin(phi), -0.35f});
  float el = sun.y;
  float dayK = smooth(-0.06f, 0.22f, el);
  float warm = sat(1.f - std::fabs(el) * 3.2f) * smooth(-0.14f, 0.02f, el);
  s.sunDir = sun; s.time = time;
  s.night = 1.f - smooth(-0.12f, 0.12f, el);
  s.zenith = mix(mix(rgb(6, 9, 24), rgb(58, 108, 200), dayK), rgb(70, 90, 150), warm * 0.6f);
  s.horizon = mix(mix(rgb(16, 22, 44), rgb(176, 204, 232), dayK), rgb(255, 150, 96), warm * 0.85f);
  Col sc = mix(Col{1.15f, 1.05f, 0.88f}, Col{1.25f, 0.62f, 0.30f}, warm);
  float sunI = smooth(-0.02f, 0.2f, el);
  V3 moon = norm(V3{-sun.x, std::max(0.6f, -sun.y), -sun.z});
  s.lightDir = norm(moon * (1.f - sunI) + sun * sunI);
  s.lightCol = sc * sunI + Col{0.10f, 0.15f, 0.30f} * (1.f - sunI);
  s.sunCol = sc;
  s.sunVis = smooth(-0.10f, 0.0f, el);
  s.ambSky = mix(mix(rgb(40, 55, 105) * 0.8f, rgb(130, 160, 215) * 0.62f, dayK), rgb(190, 130, 120) * 0.45f, warm * 0.5f);
  s.ambGround = mix(rgb(22, 24, 36) * 0.9f, rgb(105, 95, 80) * 0.9f, dayK);
  s.cloud = mix(mix(rgb(30, 36, 60), rgb(238, 238, 244), dayK), rgb(255, 170, 130), warm * 0.8f);
  s.shadowK = 0.5f * smooth(0.03f, 0.25f, el);
  return s;
}

static const Col PAL_DOWN[] = {rgb(150, 160, 172), rgb(118, 134, 152), rgb(176, 172, 162), rgb(96, 112, 128), rgb(140, 152, 146), rgb(186, 186, 192)};
static const Col PAL_MID[] = {rgb(192, 176, 150), rgb(172, 122, 102), rgb(160, 166, 172), rgb(202, 192, 172), rgb(150, 132, 116)};
static const Col PAL_RES[] = {rgb(212, 192, 160), rgb(188, 142, 122), rgb(202, 202, 192), rgb(162, 186, 170), rgb(222, 206, 182), rgb(172, 152, 142)};

void World::gen(uint32_t seed) {
  Rng rng; rng.s = seed * 2654435761u + 12345u; if (!rng.s) rng.s = 1;
  for (int i = 0; i < NB; i++)
    for (int j = 0; j < NB; j++) {
      Block& B = blocks[i][j];
      B = Block();
      B.x0 = roadC(i) + HALFR; B.x1 = roadC(i + 1) - HALFR; B.z0 = roadC(j) + HALFR; B.z1 = roadC(j + 1) - HALFR;
      float cd = std::hypot(i - 3.5f, j - 3.5f);
      uint8_t kind = 0;
      if (i == 4 && j == 3) kind = 2;
      else if (i == 2 && j == 4) kind = 3;
      else if (cd > 1.6f && rng.f() < 0.12f) kind = 1;
      B.kind = kind;
      float L = B.x1 - B.x0, cxm = (B.x0 + B.x1) * 0.5f, czm = (B.z0 + B.z1) * 0.5f;

      // street furniture along the four kerbs
      for (int side = 0; side < 4; side++) {
        V2 o, al, in;
        switch (side) {
          case 0: o = {B.x0, B.z0}; al = {1, 0}; in = {0, 1}; break;
          case 1: o = {B.x0, B.z1}; al = {1, 0}; in = {0, -1}; break;
          case 2: o = {B.x0, B.z0}; al = {0, 1}; in = {1, 0}; break;
          default: o = {B.x1, B.z0}; al = {0, 1}; in = {-1, 0}; break;
        }
        size_t lampStart = B.lamps.size();
        for (float t = 9 + rng.f() * 6; t < L - 6; t += 26) {
          V2 p = o + al * t + in * 1.0f;
          B.lamps.push_back({p, in * -1.f});
          B.obs.push_back({p.x, p.z, 0.3f, 0});
        }
        for (float t = 4 + rng.f() * 3; t < L - 3; t += 11.5f) {
          V2 p = o + al * t + in * 1.9f;
          bool near = false;
          for (size_t k = lampStart; k < B.lamps.size(); k++) { V2 d = B.lamps[k].p - p; if (len(d) < 3.4f) near = true; }
          if (!near && rng.f() < 0.8f) B.obs.push_back({p.x, p.z, 0.5f, 1});
        }
      }

      if (kind == 1) {                              // park: scattered trees
        for (int k = 0; k < 22; k++) B.obs.push_back({rng.r(B.x0 + 6, B.x1 - 6), rng.r(B.z0 + 6, B.z1 - 6), 0.6f, 1});
      } else if (kind >= 2) {                       // plaza with a single shop at the north kerb
        Building b;
        b.x0 = cxm - 16; b.x1 = cxm + 16; b.z0 = B.z0 + SIDEW + 1; b.z1 = B.z0 + SIDEW + 15;
        b.h = 7.5f; b.wall = kind == 2 ? rgb(214, 140, 50) : rgb(150, 50, 46); b.roof = rgb(70, 70, 74);
        b.seed = rng.u(); b.tiers = 1; b.setback = 0; b.antenna = false;
        B.bld.push_back(b);
        (kind == 2 ? garage : gunShop) = {cxm, czm + 8};
      } else {
        float ix0 = B.x0 + SIDEW, ix1 = B.x1 - SIDEW, iz0 = B.z0 + SIDEW, iz1 = B.z1 - SIDEW;
        int cols = 1 + rng.i(3), rows = 1 + rng.i(3);
        if (cd < 1.9f) { cols = 2; rows = 2 + rng.i(2); }
        float cw = (ix1 - ix0) / cols, ch = (iz1 - iz0) / rows;
        for (int c = 0; c < cols; c++)
          for (int r = 0; r < rows; r++) {
            if (cols * rows > 2 && rng.f() < 0.06f) continue;
            Building b;
            b.x0 = ix0 + c * cw + 1.0f; b.x1 = ix0 + (c + 1) * cw - 1.0f;
            b.z0 = iz0 + r * ch + 1.0f; b.z1 = iz0 + (r + 1) * ch - 1.0f;
            b.h = cd < 1.9f ? rng.r(45, 115) : (cd < 3.3f ? rng.r(20, 55) : rng.r(8, 26));
            const Col* pal = cd < 1.9f ? PAL_DOWN : (cd < 3.3f ? PAL_MID : PAL_RES);
            int pn = cd < 1.9f ? 6 : (cd < 3.3f ? 5 : 6);
            b.wall = pal[rng.i(pn)] * rng.r(0.85f, 1.1f);
            b.roof = rgb(84, 84, 90) * rng.r(0.8f, 1.2f);
            b.seed = rng.u();
            b.tiers = b.h > 55 ? 3 : (b.h > 30 ? 2 : 1);
            float mind = std::min(b.x1 - b.x0, b.z1 - b.z0);
            b.setback = b.tiers > 1 ? std::max(0.f, std::min(rng.r(2.0f, 3.6f), (mind - 7.f) / (2.f * (b.tiers - 1)))) : 0.f;
            b.antenna = b.h > 45 && rng.f() < 0.7f;
            B.bld.push_back(b);
          }
      }
    }
  spawn = {-31.f, -12.f};
  // keep the spawn point and the camera run-up free of trees
  for (int i = 0; i < NB; i++)
    for (int j = 0; j < NB; j++) {
      auto& o = blocks[i][j].obs;
      o.erase(std::remove_if(o.begin(), o.end(), [&](const Obst& t) { return t.kind == 1 && t.x > spawn.x - 12.f && t.x < spawn.x + 6.f && std::fabs(t.z - spawn.z) < 5.f; }), o.end());
    }
}

// ---------------------------------------------------------------- collisions
bool World::resolve(V2& p, float r, V2& nOut) const {
  bool hit = false;
  auto push = [&](float nx, float nz, float pen) { p.x += nx * pen; p.z += nz * pen; nOut = {nx, nz}; hit = true; };
  if (p.x < -BOUND + r) push(1, 0, -BOUND + r - p.x);
  if (p.x > BOUND - r) push(-1, 0, p.x - (BOUND - r));
  if (p.z < -BOUND + r) push(0, 1, -BOUND + r - p.z);
  if (p.z > BOUND - r) push(0, -1, p.z - (BOUND - r));
  int bi = (int)std::floor((p.x - roadC(0)) / PITCH), bj = (int)std::floor((p.z - roadC(0)) / PITCH);
  for (int i = std::max(0, bi - 1); i <= std::min(NB - 1, bi + 1); i++)
    for (int j = std::max(0, bj - 1); j <= std::min(NB - 1, bj + 1); j++) {
      const Block& B = blocks[i][j];
      if (p.x < B.x0 - r - 1 || p.x > B.x1 + r + 1 || p.z < B.z0 - r - 1 || p.z > B.z1 + r + 1) continue;
      for (const Building& b : B.bld) {
        float cx = clampf(p.x, b.x0, b.x1), cz = clampf(p.z, b.z0, b.z1);
        float dx = p.x - cx, dz = p.z - cz, d2 = dx * dx + dz * dz;
        if (d2 >= r * r) continue;
        if (d2 > 1e-8f) { float d = std::sqrt(d2); push(dx / d, dz / d, r - d); }
        else {
          float l = p.x - b.x0, rr = b.x1 - p.x, t = p.z - b.z0, bt = b.z1 - p.z, m = std::min({l, rr, t, bt});
          if (m == l) push(-1, 0, l + r); else if (m == rr) push(1, 0, rr + r); else if (m == t) push(0, -1, t + r); else push(0, 1, bt + r);
        }
      }
      for (const Obst& o : B.obs) {
        float dx = p.x - o.x, dz = p.z - o.z, rr = r + o.r, d2 = dx * dx + dz * dz;
        if (d2 >= rr * rr) continue;
        float d = std::sqrt(d2);
        if (d < 1e-4f) push(1, 0, rr); else push(dx / d, dz / d, rr - d);
      }
    }
  return hit;
}

static bool rayBox(V2 o, V2 d, float x0, float z0, float x1, float z1, float& tOut) {
  float tmin = 0, tmax = 1e9f;
  for (int a = 0; a < 2; a++) {
    float oo = a ? o.z : o.x, dd = a ? d.z : d.x, lo = a ? z0 : x0, hi = a ? z1 : x1;
    if (std::fabs(dd) < 1e-8f) { if (oo < lo || oo > hi) return false; }
    else {
      float t1 = (lo - oo) / dd, t2 = (hi - oo) / dd;
      if (t1 > t2) std::swap(t1, t2);
      tmin = std::max(tmin, t1); tmax = std::min(tmax, t2);
      if (tmin > tmax) return false;
    }
  }
  tOut = tmin;
  return true;
}

float World::rayWorld(V2 o, V2 dir, float maxT, bool withObs) const {
  float best = maxT;
  V2 e = o + dir * maxT;
  float bx0 = std::min(o.x, e.x) - 1, bx1 = std::max(o.x, e.x) + 1, bz0 = std::min(o.z, e.z) - 1, bz1 = std::max(o.z, e.z) + 1;
  for (int i = 0; i < NB; i++)
    for (int j = 0; j < NB; j++) {
      const Block& B = blocks[i][j];
      if (B.x1 < bx0 || B.x0 > bx1 || B.z1 < bz0 || B.z0 > bz1) continue;
      for (const Building& b : B.bld) { float t; if (rayBox(o, dir, b.x0, b.z0, b.x1, b.z1, t) && t < best) best = t; }
      if (withObs) for (const Obst& ob : B.obs) {
        V2 oc = o - V2{ob.x, ob.z};
        float bq = dot(oc, dir), cq = dot(oc, oc) - ob.r * ob.r, disc = bq * bq - cq;
        if (disc >= 0) { float t = -bq - std::sqrt(disc); if (t >= 0 && t < best) best = t; }
      }
    }
  return best;
}

// ---------------------------------------------------------------- drawing
static void drawBuilding(Renderer& R, const Building& b, float time) {
  const V3 up{0, 1, 0};
  float hs[3] = {b.h, 0, 0};
  if (b.tiers == 2) { hs[0] = b.h * 0.62f; hs[1] = b.h * 0.38f; }
  if (b.tiers == 3) { hs[0] = b.h * 0.5f; hs[1] = b.h * 0.3f; hs[2] = b.h * 0.2f; }
  float y = CURB, x0 = b.x0, x1 = b.x1, z0 = b.z0, z1 = b.z1;
  for (int t = 0; t < b.tiers; t++) {
    float ins = b.setback * t;
    x0 = b.x0 + ins; x1 = b.x1 - ins; z0 = b.z0 + ins; z1 = b.z1 - ins;
    float y1 = y + hs[t];
    R.quad({x0, y, z0}, {x1, y, z0}, {x1, y1, z0}, {x0, y1, z0}, {0, 0, -1}, M_FACADE, b.wall, b.seed);
    R.quad({x0, y, z1}, {x0, y1, z1}, {x1, y1, z1}, {x1, y, z1}, {0, 0, 1}, M_FACADE, b.wall, b.seed);
    R.quad({x0, y, z0}, {x0, y1, z0}, {x0, y1, z1}, {x0, y, z1}, {-1, 0, 0}, M_FACADE, b.wall, b.seed);
    R.quad({x1, y, z0}, {x1, y, z1}, {x1, y1, z1}, {x1, y1, z0}, {1, 0, 0}, M_FACADE, b.wall, b.seed);
    R.quad({x0, y1, z0}, {x1, y1, z0}, {x1, y1, z1}, {x0, y1, z1}, up, M_FLAT, b.roof);
    y = y1;
  }
  // rooftop clutter (only when the roof is reasonably close)
  V3 dc = V3{(x0 + x1) * 0.5f, y, (z0 + z1) * 0.5f} - R.cam.pos;
  if (dot(dc, dc) < 160.f * 160.f) {
    Rng rr; rr.s = b.seed | 1u;
    int n = 1 + rr.i(3);
    for (int k = 0; k < n; k++) {
      float hx = rr.r(0.8f, 1.8f), hz = rr.r(0.8f, 1.8f);
      float px = rr.r(x0 + hx + 0.8f, std::max(x0 + hx + 0.9f, x1 - hx - 0.8f)), pz = rr.r(z0 + hz + 0.8f, std::max(z0 + hz + 0.9f, z1 - hz - 0.8f));
      R.box({px, y + 0.7f, pz}, {1, 0, 0}, up, {0, 0, 1}, {hx, 0.7f, hz}, rgb(128, 130, 134));
    }
  }
  if (b.antenna) {
    V3 top{(x0 + x1) * 0.5f, y, (z0 + z1) * 0.5f};
    R.box(top + V3{0, 6, 0}, {1, 0, 0}, up, {0, 0, 1}, {0.18f, 6.f, 0.18f}, rgb(70, 70, 76));
    if (R.sky.night > 0.3f && std::fmod(time + b.seed % 7, 1.6f) < 0.8f) R.sprite(top + V3{0, 12.3f, 0}, 2.2f, {1.f, 0.12f, 0.06f}, 1.4f);
  }
}

static void shadowOf(Renderer& R, const Building& b, V2 sunOff) {
  float hs[3] = {b.h, 0, 0};
  if (b.tiers == 2) { hs[0] = b.h * 0.62f; hs[1] = b.h * 0.38f; }
  if (b.tiers == 3) { hs[0] = b.h * 0.5f; hs[1] = b.h * 0.3f; hs[2] = b.h * 0.2f; }
  float hTop = 0;
  const float y = CURB + 0.03f;
  for (int t = 0; t < b.tiers; t++) {
    hTop += hs[t];
    float ins = b.setback * t;
    V2 c[4] = {{b.x0 + ins, b.z0 + ins}, {b.x1 - ins, b.z0 + ins}, {b.x1 - ins, b.z1 - ins}, {b.x0 + ins, b.z1 - ins}};
    V2 sh = sunOff * hTop;
    float l = len(sh);
    if (l > 70.f) sh = sh * (70.f / l);
    for (int k = 0; k < 4; k++) {
      V2 p = c[k], q = c[(k + 1) & 3];
      R.decal({p.x, y, p.z}, {q.x, y, q.z}, {q.x + sh.x, y, q.z + sh.z}, {p.x + sh.x, y, p.z + sh.z}, D_SHADOW_HARD, Col{}, 1.f);
    }
    R.decal({c[0].x + sh.x, y, c[0].z + sh.z}, {c[1].x + sh.x, y, c[1].z + sh.z}, {c[2].x + sh.x, y, c[2].z + sh.z}, {c[3].x + sh.x, y, c[3].z + sh.z}, D_SHADOW_HARD, Col{}, 1.f);
  }
}

void World::draw(Renderer& R, float time) const {
  const Camera& cam = R.cam;
  const Sky& sky = R.sky;
  const V3 up{0, 1, 0}, X{1, 0, 0}, Z{0, 0, 1};
  const float night = sky.night;
  const float lampOn = sat(night * 1.6f);
  static const Col asphalt = rgb(62, 64, 70), pave = rgb(170, 168, 162), grass = rgb(78, 128, 64), curbc = rgb(150, 148, 142);

  const float T = 4000.f;
  R.quad({-T, -0.3f, -T}, {T, -0.3f, -T}, {T, -0.3f, T}, {-T, -0.3f, T}, up, M_GRASS, rgb(88, 118, 66));

  // perimeter barrier
  static const Col barrier = rgb(172, 172, 170);
  R.box({0, 0.6f, -BOUND - 0.5f}, X, up, Z, {BOUND + 1, 0.6f, 0.45f}, barrier);
  R.box({0, 0.6f, BOUND + 0.5f}, X, up, Z, {BOUND + 1, 0.6f, 0.45f}, barrier);
  R.box({-BOUND - 0.5f, 0.6f, 0}, X, up, Z, {0.45f, 0.6f, BOUND + 1}, barrier);
  R.box({BOUND + 0.5f, 0.6f, 0}, X, up, Z, {0.45f, 0.6f, BOUND + 1}, barrier);

  // roads + intersections
  for (int k = 0; k < NR; k++) {
    float c = roadC(k);
    for (int s = 0; s < NB; s++) {
      float a = roadC(s) + HALFR, b = roadC(s + 1) - HALFR, m = (a + b) * 0.5f;
      if (R.inView({m, 0, c}, 32, 520)) R.quad({a, 0, c - HALFR}, {b, 0, c - HALFR}, {b, 0, c + HALFR}, {a, 0, c + HALFR}, up, M_ROADX, asphalt);
      if (R.inView({c, 0, m}, 32, 520)) R.quad({c - HALFR, 0, a}, {c + HALFR, 0, a}, {c + HALFR, 0, b}, {c - HALFR, 0, b}, up, M_ROADZ, asphalt);
    }
    for (int i = 0; i < NR; i++) {
      float x = roadC(i);
      if (R.inView({x, 0, c}, 14, 520)) R.quad({x - HALFR, 0, c - HALFR}, {x + HALFR, 0, c - HALFR}, {x + HALFR, 0, c + HALFR}, {x - HALFR, 0, c + HALFR}, up, M_ISECT, asphalt);
    }
  }

  // sun shadow offset per metre of height
  const bool sunShadows = sky.sunDir.y > 0.07f && sky.shadowK > 0.05f;
  V2 sunOff = sunShadows ? V2{-sky.sunDir.x / sky.sunDir.y, -sky.sunDir.z / sky.sunDir.y} : V2{};

  // blocks
  for (int i = 0; i < NB; i++)
    for (int j = 0; j < NB; j++) {
      const Block& B = blocks[i][j];
      float cx = (B.x0 + B.x1) * 0.5f, cz = (B.z0 + B.z1) * 0.5f;
      if (!R.inView({cx, 35, cz}, 90, 540)) continue;
      float dist = std::hypot(cx - cam.pos.x, cz - cam.pos.z);
      R.quad({B.x0, CURB, B.z0}, {B.x1, CURB, B.z0}, {B.x1, CURB, B.z1}, {B.x0, CURB, B.z1}, up, B.kind == 1 ? M_GRASS : M_PAVE, B.kind == 1 ? grass : pave);
      R.quad({B.x0, 0, B.z0}, {B.x1, 0, B.z0}, {B.x1, CURB, B.z0}, {B.x0, CURB, B.z0}, {0, 0, -1}, M_FLAT, curbc);
      R.quad({B.x0, 0, B.z1}, {B.x0, CURB, B.z1}, {B.x1, CURB, B.z1}, {B.x1, 0, B.z1}, {0, 0, 1}, M_FLAT, curbc);
      R.quad({B.x0, 0, B.z0}, {B.x0, CURB, B.z0}, {B.x0, CURB, B.z1}, {B.x0, 0, B.z1}, {-1, 0, 0}, M_FLAT, curbc);
      R.quad({B.x1, 0, B.z0}, {B.x1, 0, B.z1}, {B.x1, CURB, B.z1}, {B.x1, CURB, B.z0}, {1, 0, 0}, M_FLAT, curbc);
      if (B.kind == 1 && ((i + j) & 1) == 0) {      // pond
        Col wc = mix(sky.horizon, sky.zenith, 0.4f) * 0.55f;
        R.quad({cx - 9, CURB + 0.03f, cz - 6}, {cx + 9, CURB + 0.03f, cz - 6}, {cx + 9, CURB + 0.03f, cz + 6}, {cx - 9, CURB + 0.03f, cz + 6}, up, M_EMIT, wc);
      }
      for (const Building& b : B.bld) {
        V3 bc{(b.x0 + b.x1) * 0.5f, b.h * 0.5f, (b.z0 + b.z1) * 0.5f};
        if (!R.inView(bc, std::max(len(V2{b.x1 - b.x0, b.z1 - b.z0}) * 0.5f, b.h * 0.5f), 540)) continue;
        drawBuilding(R, b, time);
        if (sunShadows && dist < 170) shadowOf(R, b, sunOff);
      }
      // plaza signs
      if (B.kind >= 2) {
        const Building& b = B.bld[0];
        Col sc = B.kind == 2 ? Col{2.2f, 1.7f, 0.3f} : Col{2.4f, 0.4f, 0.25f};
        R.box({(b.x0 + b.x1) * 0.5f, b.h + 1.2f, (b.z0 + b.z1) * 0.5f}, X, up, Z, {10, 1.2f, 0.4f}, sc, M_EMIT);
      }
      // lamps and trees (near the camera only)
      if (dist < 175) {
        for (const Lamp& l : B.lamps) {
          if (!R.inView({l.p.x, 4, l.p.z}, 4, 175)) continue;
          R.box({l.p.x, 3.6f, l.p.z}, X, up, Z, {0.09f, 3.6f, 0.09f}, rgb(60, 62, 66));
          float ax = std::fabs(l.arm.x) > 0.5f ? 1.0f : 0.06f, az = std::fabs(l.arm.z) > 0.5f ? 1.0f : 0.06f;
          R.box({l.p.x + l.arm.x * 1.0f, 7.2f, l.p.z + l.arm.z * 1.0f}, X, up, Z, {ax, 0.06f, az}, rgb(60, 62, 66));
          V3 head{l.p.x + l.arm.x * 2.0f, 7.0f, l.p.z + l.arm.z * 2.0f};
          R.box(head, X, up, Z, {0.4f, 0.1f, 0.4f}, Col{1.5f, 1.15f, 0.7f} * (0.25f + 0.75f * lampOn), M_EMIT);
          if (lampOn > 0.05f) {
            R.sprite(head - V3{0, 0.2f, 0}, 2.6f, {1.0f, 0.78f, 0.45f}, 0.9f * lampOn);
            R.decal({head.x - 9, 0.25f, head.z - 9}, {head.x + 9, 0.25f, head.z - 9}, {head.x + 9, 0.25f, head.z + 9}, {head.x - 9, 0.25f, head.z + 9}, D_GLOW, {1.0f, 0.72f, 0.38f}, 0.55f * lampOn);
          }
        }
        for (const Obst& o : B.obs) {
          if (o.kind != 1 || !R.inView({o.x, 3, o.z}, 4, 160)) continue;
          uint32_t h = hash32((uint32_t)(int)(o.x * 10) * 73856093u ^ (uint32_t)(int)(o.z * 10) * 19349663u);
          Col leaf = (h & 7) == 0 ? rgb(196, 120, 40) : rgb(52 + h % 30, 108 + (h >> 8) % 50, 50 + (h >> 16) % 25);
          float s = o.r > 0.55f ? 1.25f : 1.f;
          R.box({o.x, 1.0f, o.z}, X, up, Z, {0.18f, 1.0f, 0.18f}, rgb(92, 66, 46));
          V3 q1[8] = {{o.x - 1.5f * s, 1.6f, o.z - 1.5f * s}, {o.x + 1.5f * s, 1.6f, o.z - 1.5f * s}, {o.x + 1.5f * s, 1.6f, o.z + 1.5f * s}, {o.x - 1.5f * s, 1.6f, o.z + 1.5f * s},
                      {o.x - 1.9f * s, 2.9f, o.z - 1.9f * s}, {o.x + 1.9f * s, 2.9f, o.z - 1.9f * s}, {o.x + 1.9f * s, 2.9f, o.z + 1.9f * s}, {o.x - 1.9f * s, 2.9f, o.z + 1.9f * s}};
          V3 q2[8] = {q1[4], q1[5], q1[6], q1[7],
                      {o.x - 0.7f * s, 4.7f, o.z - 0.7f * s}, {o.x + 0.7f * s, 4.7f, o.z - 0.7f * s}, {o.x + 0.7f * s, 4.7f, o.z + 0.7f * s}, {o.x - 0.7f * s, 4.7f, o.z + 0.7f * s}};
          R.solid(q1, leaf);
          R.solid(q2, leaf * 1.12f);
        }
      }
    }

  // traffic signals
  for (int i = 0; i < NR; i++)
    for (int j = 0; j < NR; j++) {
      V2 n = nodePos(i, j);
      if (!R.inView({n.x, 4, n.z}, 18, 150)) continue;
      for (int d = 0; d < 4; d++) {
        V2 c = n + dirOf(d) * 10.9f + rightOf(d) * 10.9f;
        V2 f = dirOf(d) * -1.f;                      // head faces the approaching traffic
        R.box({c.x, 2.4f, c.z}, X, up, Z, {0.1f, 2.4f, 0.1f}, rgb(55, 57, 60));
        bool alongX = std::fabs(f.x) > 0.5f;
        V3 hh = alongX ? V3{0.15f, 0.55f, 0.2f} : V3{0.2f, 0.55f, 0.15f};
        V3 hc{c.x, 5.0f, c.z};
        R.box(hc, X, up, Z, hh, rgb(30, 31, 34));
        int st = sigState(i, j, d, false, time);
        const Col lc[3] = {{2.0f, 0.12f, 0.08f}, {2.0f, 1.4f, 0.1f}, {0.1f, 1.8f, 0.5f}};
        for (int k = 0; k < 3; k++) {
          bool on = (k == 0 && st == 0) || (k == 1 && st == 1) || (k == 2 && st == 2);
          V3 lp{hc.x + f.x * (hh.x + 0.03f), hc.y + (k == 0 ? 0.35f : (k == 1 ? 0.0f : -0.35f)), hc.z + f.z * (hh.z + 0.03f)};
          R.box(lp, X, up, Z, alongX ? V3{0.03f, 0.14f, 0.14f} : V3{0.14f, 0.14f, 0.03f}, on ? lc[k] : lc[k] * 0.07f, M_EMIT);
          if (on) R.sprite(lp + V3{f.x * 0.2f, 0, f.z * 0.2f}, 0.9f, {lc[k].r * 0.5f, lc[k].g * 0.5f, lc[k].b * 0.5f}, 0.9f);
        }
      }
    }

  // mission markers (garage / gun shop)
  struct Mk { V2 p; Col c; };
  const Mk mk[2] = {{garage, {1.0f, 0.82f, 0.15f}}, {gunShop, {1.0f, 0.3f, 0.15f}}};
  for (const Mk& m : mk) {
    if (!R.inView({m.p.x, 10, m.p.z}, 40, 420)) continue;
    float pulse = 0.75f + 0.25f * std::sin(time * 3.f);
    float y = CURB + 0.04f, r = 8.f;
    R.decal({m.p.x - r, y, m.p.z - r}, {m.p.x + r, y, m.p.z - r}, {m.p.x + r, y, m.p.z + r}, {m.p.x - r, y, m.p.z + r}, D_RING, m.c, pulse);
    R.decal({m.p.x - r, y, m.p.z - r}, {m.p.x + r, y, m.p.z - r}, {m.p.x + r, y, m.p.z + r}, {m.p.x - r, y, m.p.z + r}, D_GLOW, m.c, 0.22f * pulse);
    R.decal({m.p.x - 2.4f, y, m.p.z}, {m.p.x + 2.4f, y, m.p.z}, {m.p.x + 2.4f, 70, m.p.z}, {m.p.x - 2.4f, 70, m.p.z}, D_BEAM, m.c, 0.9f * pulse);
    R.decal({m.p.x, y, m.p.z - 2.4f}, {m.p.x, y, m.p.z + 2.4f}, {m.p.x, 70, m.p.z + 2.4f}, {m.p.x, 70, m.p.z - 2.4f}, D_BEAM, m.c, 0.9f * pulse);
  }
}


// ==================== EMBEDDED SOURCE: game.cpp ====================

// ================================================================ data tables
const CarDef CARS[4] = {
  // name       len    wid    y0     body   cab    nose   tail   cb0     cb1    ct0     ct1     wb     wr     vmax  acc   grip  hp     mass    value
  {"PIXIE",   3.70f, 1.68f, 0.26f, 0.86f, 1.46f, 0.30f, 0.12f, -1.55f, 0.65f, -1.35f,  0.05f, 2.30f, 0.30f, 32.f, 8.5f, 7.5f, 90.f,  900.f,  450},
  {"ORION",   4.60f, 1.82f, 0.27f, 0.90f, 1.42f, 0.35f, 0.20f, -1.60f, 0.90f, -1.05f,  0.30f, 2.75f, 0.33f, 42.f, 11.f, 7.5f, 120.f, 1400.f, 900},
  {"BRUTE",   4.90f, 1.96f, 0.28f, 0.98f, 1.42f, 0.45f, 0.15f, -1.65f, 0.55f, -1.05f, -0.05f, 2.90f, 0.36f, 52.f, 15.f, 6.5f, 160.f, 1800.f, 2000},
  {"SPECTER", 4.50f, 2.02f, 0.22f, 0.72f, 1.10f, 0.75f, 0.20f, -1.25f, 0.95f, -0.75f,  0.15f, 2.70f, 0.34f, 85.f, 20.f, 8.5f, 110.f, 1300.f, 6500},
};
const Weapon WEAP[4] = {
  {"FISTS",   0,    0,  0,   14, 1, 2.2f,  0.f,    1.8f,  true},
  {"PISTOL",  300,  24, 120, 30, 1, 3.5f,  0.015f, 80.f,  false},
  {"SMG",     1200, 90, 360, 12, 1, 13.f,  0.06f,  65.f,  false},
  {"SHOTGUN", 1800, 16, 64,  15, 8, 1.1f,  0.14f,  32.f,  false},
};
static const Col CAR_COL[9] = {rgb(196, 40, 44), rgb(34, 92, 190), rgb(232, 232, 236), rgb(32, 32, 36), rgb(226, 170, 34), rgb(44, 140, 84), rgb(150, 152, 160), rgb(230, 112, 32), rgb(122, 56, 160)};
static const Col SPC_COL[5] = {rgb(24, 24, 28), rgb(236, 236, 242), rgb(210, 20, 44), rgb(30, 170, 210), rgb(250, 200, 0)};
static const Col SHIRTS[8] = {rgb(200, 60, 60), rgb(60, 110, 200), rgb(230, 200, 60), rgb(70, 160, 90), rgb(220, 220, 225), rgb(150, 80, 170), rgb(240, 130, 50), rgb(50, 50, 56)};
static const Col SKINS[5] = {rgb(240, 200, 170), rgb(214, 160, 120), rgb(170, 118, 84), rgb(120, 80, 56), rgb(90, 58, 40)};

std::string fmtMoney(int v) {
  std::string s = std::to_string(v < 0 ? -v : v);
  for (int i = (int)s.size() - 3; i > 0; i -= 3) s.insert(i, ",");
  return (v < 0 ? "-$" : "$") + s;
}

static const float START_YAW = 1.22f;   // camera looks east along the pavement at the start

// ================================================================ helpers
static inline V2 fwdOf(float yaw) { return {std::sin(yaw), std::cos(yaw)}; }
static inline V2 rtOf(float yaw) { return {-std::cos(yaw), std::sin(yaw)}; }
static V2 stopPoint(int ni, int nj, int d, int lane) { return nodePos(ni, nj) - dirOf(d) * STOPD + rightOf(d) * LANE[lane]; }
static V2 exitPoint(int ni, int nj, int d, int lane) { return nodePos(ni, nj) + dirOf(d) * STOPD + rightOf(d) * LANE[lane]; }
static float carRadius(const CarDef& d) { return d.wid * 0.5f * 0.96f; }
static void carCircles(const Vehicle& v, V2 out[3], float& r) {
  const CarDef& d = CARS[v.type];
  r = carRadius(d);
  V2 f = fwdOf(v.yaw);
  float off = d.len * 0.5f - r;
  out[0] = v.pos - f * off; out[1] = v.pos; out[2] = v.pos + f * off;
}
static float maxSteer(const CarDef& d, float vf) { float k = std::fabs(vf) / (0.42f * d.vmax); return 0.62f / (1.f + k * k); }
static int chooseMove(Rng& rng, int ni, int nj, int d) {          // 0 straight, 1 right, 2 left
  float w[3] = {nodeOk(ni + DX[d], nj + DZ[d]) ? 0.6f : 0.f, nodeOk(ni + DX[(d + 1) & 3], nj + DZ[(d + 1) & 3]) ? 0.26f : 0.f,
                nodeOk(ni + DX[(d + 3) & 3], nj + DZ[(d + 3) & 3]) ? 0.14f : 0.f};
  float sum = w[0] + w[1] + w[2];
  if (sum <= 0) return 0;
  float r = rng.f() * sum;
  for (int i = 0; i < 3; i++) { if (r < w[i]) return i; r -= w[i]; }
  return 0;
}
static V2 perimPoint(const Block& B, float off, float s, V2* dirOut) {
  float x0 = B.x0 + off, x1 = B.x1 - off, z0 = B.z0 + off, z1 = B.z1 - off, w = x1 - x0, h = z1 - z0, per = 2 * (w + h);
  s = std::fmod(s, per); if (s < 0) s += per;
  V2 d;
  if (s < w) { d = {1, 0}; if (dirOut) *dirOut = d; return {x0 + s, z0}; }
  s -= w;
  if (s < h) { d = {0, 1}; if (dirOut) *dirOut = d; return {x1, z0 + s}; }
  s -= h;
  if (s < w) { d = {-1, 0}; if (dirOut) *dirOut = d; return {x1 - s, z1}; }
  s -= w;
  d = {0, -1}; if (dirOut) *dirOut = d;
  return {x0, z1 - s};
}

void Game::notify(const std::string& s, uint32_t col) {
  notes.push_back({s, 3.6f, col});
  if (notes.size() > 4) notes.erase(notes.begin());
}
void Game::emit(uint8_t kind, V3 p, V3 v, float life, float size, Col c) {
  if (parts.size() > 900) return;
  Part q; q.kind = kind; q.p = p; q.v = v; q.life = q.max = life; q.size = size; q.c = c;
  parts.push_back(q);
}

// ================================================================ setup / spawning
void Game::init(int w, int h, uint32_t seed) {
  rng.s = seed * 747796405u + 2891336453u; if (!rng.s) rng.s = 1;
  world.gen(seed);
  W = w; H = h;
  cars.clear(); cars.reserve(160); peds.clear(); peds.reserve(100); picks.clear(); parts.clear(); tracers.clear(); notes.clear();
  P = Player();
  P.pos = {-31.f, -12.0f};
  V2 n; world.resolve(P.pos, 0.45f, n);
  camYaw = START_YAW; camPitch = 0.3f; time = 0; hour = 17.4f;
  P.yaw = camYaw;
  spawnVehicle(3, {-32.f, -8.3f}, -PI * 0.5f, VS_PARKED);     // the hyper-fast one, right next to the spawn
  spawnVehicle(1, {-20.f, -8.3f}, -PI * 0.5f, VS_PARKED);
  spawnVehicle(2, {-44.f, -8.3f}, -PI * 0.5f, VS_PARKED);
  spawnVehicle(0, {-56.f, -8.3f}, -PI * 0.5f, VS_PARKED);
  for (int i = 0; i < 22; i++) spawnParked(P.pos, 280);
  for (int i = 0; i < 26; i++) spawnAI(35, 240, false);
  for (int i = 0; i < 28; i++) spawnPed(false);
  for (int i = 0; i < 26; i++) spawnPickup();
  camFocus = {P.pos.x, 1.55f, P.pos.z};
  camPos = {P.pos.x, 3.f, P.pos.z - 5.f};
  notify("STEAL A CAR  -  SELL IT AT THE GARAGE", 0xFFD84A);
}
void Game::resize(int w, int h) { W = w; H = h; }

int Game::spawnVehicle(int type, V2 pos, float yaw, VState st) {
  int idx = -1;
  for (size_t i = 0; i < cars.size(); i++) if (cars[i].st == VS_FREE) { idx = (int)i; break; }
  if (idx < 0) { if (cars.size() >= cars.capacity()) return -1; cars.emplace_back(); idx = (int)cars.size() - 1; }
  Vehicle& v = cars[idx];
  v = Vehicle();
  v.type = type; v.st = st; v.pos = pos; v.yaw = yaw; v.hp = CARS[type].hp;
  v.color = type == 3 ? SPC_COL[rng.i(5)] : CAR_COL[rng.i(9)];
  v.cruise = rng.r(9.5f, 14.5f);
  return idx;
}

bool Game::spawnAI(float minD, float maxD, bool hidden) {
  for (int tries = 0; tries < 30; tries++) {
    int axis = rng.i(2), k = rng.i(NR), s = rng.i(NB), sg = rng.i(2);
    int d = axis == 0 ? (sg ? 0 : 2) : (sg ? 1 : 3);
    int ni = axis == 0 ? (d == 0 ? s + 1 : s) : k, nj = axis == 0 ? k : (d == 1 ? s + 1 : s);
    int move = chooseMove(rng, ni, nj, d);
    int lane = move == 2 ? 0 : (move == 1 ? 1 : rng.i(2));
    V2 a = axis == 0 ? nodePos(s, k) : nodePos(k, s), b = axis == 0 ? nodePos(s + 1, k) : nodePos(k, s + 1);
    V2 al = axis == 0 ? V2{1, 0} : V2{0, 1};
    V2 p0 = a + al * (STOPD + 4.f), p1 = b - al * (STOPD + 4.f);
    V2 pos = p0 + (p1 - p0) * rng.r(0.1f, 0.9f) + rightOf(d) * LANE[lane];
    float dist = len(pos - P.pos);
    if (dist < minD || dist > maxD) continue;
    if (hidden) {
      V2 to = norm(pos - V2{camPos.x, camPos.z}), cf = norm(V2{std::sin(camYaw), std::cos(camYaw)});
      if (dot(to, cf) > 0.35f && dist < 170.f) continue;
    }
    bool clear = true;
    for (const Vehicle& o : cars) if (o.st != VS_FREE && len(o.pos - pos) < 10.f) { clear = false; break; }
    if (!clear) continue;
    float r = rng.f();
    int type = r < 0.38f ? 0 : (r < 0.74f ? 1 : (r < 0.93f ? 2 : 3));
    int idx = spawnVehicle(type, pos, yawOf(d), VS_AI);
    if (idx < 0) return false;
    Vehicle& v = cars[idx];
    v.d = d; v.lane = lane; v.ni = ni; v.nj = nj; v.move = move;
    v.vel = fwdOf(v.yaw) * (v.cruise * 0.85f);
    return true;
  }
  return false;
}

int Game::spawnParked(V2 nearP, float radius) {
  for (int tries = 0; tries < 30; tries++) {
    int axis = rng.i(2), k = rng.i(NR), s = rng.i(NB), sg = rng.i(2);
    int d = axis == 0 ? (sg ? 0 : 2) : (sg ? 1 : 3);
    V2 a = axis == 0 ? nodePos(s, k) : nodePos(k, s), b = axis == 0 ? nodePos(s + 1, k) : nodePos(k, s + 1);
    V2 al = axis == 0 ? V2{1, 0} : V2{0, 1};
    V2 p0 = a + al * (STOPD + 5.f), p1 = b - al * (STOPD + 5.f);
    V2 pos = p0 + (p1 - p0) * rng.f() + rightOf(d) * 8.3f;
    if (len(pos - nearP) > radius || len(pos - P.pos) < 12.f) continue;
    bool clear = true;
    for (const Vehicle& o : cars) if (o.st != VS_FREE && len(o.pos - pos) < 6.5f) { clear = false; break; }
    if (!clear) continue;
    float r = rng.f();
    return spawnVehicle(r < 0.34f ? 0 : (r < 0.68f ? 1 : (r < 0.92f ? 2 : 3)), pos, yawOf(d) + rng.r(-0.03f, 0.03f), VS_PARKED);
  }
  return -1;
}

bool Game::spawnPed(bool hidden) {
  for (int tries = 0; tries < 30; tries++) {
    int i = rng.i(NB), j = rng.i(NB);
    const Block& B = world.blocks[i][j];
    float off = rng.r(2.3f, 2.8f), per = 2.f * ((B.x1 - B.x0 - 2 * off) + (B.z1 - B.z0 - 2 * off)), s = rng.f() * per;
    V2 p = perimPoint(B, off, s, nullptr);
    float dist = len(p - P.pos);
    if (dist < 18.f || dist > 230.f) continue;
    if (hidden) {
      V2 to = norm(p - V2{camPos.x, camPos.z}), cf = norm(V2{std::sin(camYaw), std::cos(camYaw)});
      if (dot(to, cf) > 0.35f && dist < 150.f) continue;
    }
    int idx = -1;
    for (size_t q = 0; q < peds.size(); q++) if (!peds[q].alive) { idx = (int)q; break; }
    if (idx < 0) { if (peds.size() >= peds.capacity()) return false; peds.emplace_back(); idx = (int)peds.size() - 1; }
    Ped& pd = peds[idx];
    pd = Ped();
    pd.alive = true; pd.bi = i; pd.bj = j; pd.off = off; pd.s = s; pd.dirSign = rng.i(2) ? 1 : -1; pd.pos = p;
    pd.speed = rng.f() < 0.12f ? rng.r(2.4f, 3.2f) : rng.r(1.1f, 1.7f);
    pd.shirt = SHIRTS[rng.i(8)]; pd.pants = rng.f() < 0.5f ? rgb(48, 56, 90) : rgb(60, 60, 64); pd.skin = SKINS[rng.i(5)];
    pd.t = rng.r(0, 6.f);
    return true;
  }
  return false;
}

static int addPickup(std::vector<Pickup>& picks, V2 pos, int value) {
  for (size_t i = 0; i < picks.size(); i++) if (!picks[i].alive) { picks[i] = {true, pos, value, 0}; return (int)i; }
  picks.push_back({true, pos, value, 0});
  return (int)picks.size() - 1;
}
void Game::spawnPickup() {
  for (int tries = 0; tries < 30; tries++) {
    int i = rng.i(NB), j = rng.i(NB);
    const Block& B = world.blocks[i][j];
    float per = 2.f * ((B.x1 - B.x0 - 5.f) + (B.z1 - B.z0 - 5.f));
    V2 p = perimPoint(B, 2.5f, rng.f() * per, nullptr);
    if (len(p - P.pos) < 25.f || len(p - P.pos) > 300.f) continue;
    addPickup(picks, p, 50 + 25 * rng.i(7));
    return;
  }
}

// ================================================================ main update
void Game::update(float dt) {
  dt = std::min(dt, 0.05f);
  time += dt;
  if (!timeFrozen) hour = std::fmod(hour + dt / 40.f, 24.f);
  gatherInput();
  updatePlayer(dt);
  for (size_t i = 0; i < cars.size(); i++) if (cars[i].st == VS_AI) aiDrive(cars[i], (int)i, dt);
  int n = std::max(1, (int)std::ceil(dt / 0.0167f));
  float h = dt / n;
  for (int k = 0; k < n; k++) { updateVehicles(h); collide(h); }
  if (P.car >= 0) { P.pos = cars[P.car].pos; P.yaw = cars[P.car].yaw; }
  updatePeds(dt);
  updateFX(dt);
  manageSpawns(dt);
  updateCamera(dt);
}

void Game::manageSpawns(float dt) {
  spawnT -= dt;
  if (spawnT > 0) return;
  spawnT = 0.35f;
  int ai = 0, live = 0;
  for (Vehicle& v : cars) {
    if (v.st == VS_FREE) continue;
    live++;
    float d = len(v.pos - P.pos);
    if (v.st == VS_AI) { ai++; if (d > 290.f) { v.st = VS_FREE; ai--; } }
    else if (v.st == VS_WRECK && v.burn <= 0 && d > 140.f) v.st = VS_FREE;
  }
  if (ai < 26 && live < 120) spawnAI(70.f, 230.f, true);
  int np = 0;
  for (const Ped& p : peds) if (p.alive) np++;
  if (np < 30) spawnPed(true);
}

// ================================================================ player
int Game::nearestCar(float reach) {
  int best = -1;
  float bd = reach;
  for (size_t i = 0; i < cars.size(); i++) {
    const Vehicle& v = cars[i];
    if (v.st != VS_PARKED && v.st != VS_AI) continue;
    if (len(v.vel) > 4.f) continue;
    const CarDef& D = CARS[v.type];
    V2 rel = P.pos - v.pos, f = fwdOf(v.yaw), r = rtOf(v.yaw);
    float dx = std::max(std::fabs(dot(rel, r)) - D.wid * 0.5f, 0.f), dz = std::max(std::fabs(dot(rel, f)) - D.len * 0.5f, 0.f);
    float d = std::sqrt(dx * dx + dz * dz);
    if (d < bd) { bd = d; best = (int)i; }
  }
  return best;
}

void Game::enterOrUse() {
  if (P.car >= 0) {
    Vehicle& v = cars[P.car];
    const CarDef& D = CARS[v.type];
    if (len(v.pos - world.garage) < 9.5f && len(v.vel) < 5.f) {
      int value = (int)(D.value * (0.35f + 0.65f * clampf(v.hp / D.hp, 0.f, 1.f)));
      P.money += value;
      notify(std::string("SOLD ") + D.name + "  +" + fmtMoney(value), 0x6CFF7A);
      P.pos = v.pos + rtOf(v.yaw) * 3.f;
      v.st = VS_FREE; P.car = -1;
      return;
    }
    exitCar();
    return;
  }
  if (len(P.pos - world.gunShop) < 9.5f) {
    const Weapon& cw = WEAP[P.weapon];
    if (!cw.melee && P.ammo[P.weapon] < cw.maxAmmo) {               // refill current weapon
      int cost = cw.price / 5;
      if (P.money < cost) { notify("NOT ENOUGH CASH", 0xFF7070); return; }
      P.money -= cost; P.ammo[P.weapon] = std::min(cw.maxAmmo, P.ammo[P.weapon] + cw.ammoPack);
      notify(std::string("AMMO ") + cw.name + "  -" + fmtMoney(cost), 0xFFD84A);
      return;
    }
    for (int w = 1; w < 4; w++) if (!P.owned[w]) {                   // buy next weapon
      if (P.money < WEAP[w].price) { notify("NOT ENOUGH CASH", 0xFF7070); return; }
      P.money -= WEAP[w].price; P.owned[w] = true; P.ammo[w] = WEAP[w].ammoPack; P.weapon = w;
      notify(std::string("BOUGHT ") + WEAP[w].name, 0xFFD84A);
      return;
    }
    notify("ALL WEAPONS OWNED  -  AMMO FULL", 0xBBBBBB);
    return;
  }
  int k = nearestCar(3.0f);
  if (k < 0) return;
  Vehicle& v = cars[k];
  const CarDef& D = CARS[v.type];
  if (v.st == VS_AI) {                                               // carjack: driver jumps out and runs
    V2 door = v.pos + rtOf(v.yaw) * (-(D.wid * 0.5f + 0.7f));
    for (size_t q = 0; q < peds.size() + 1; q++) {
      if (q == peds.size()) { if (peds.size() >= peds.capacity()) break; peds.emplace_back(); }
      if (!peds[q].alive) {
        Ped& pd = peds[q];
        pd = Ped(); pd.alive = true; pd.pos = door; pd.st = 1; pd.hp = 40;
        pd.fleeDir = norm(door - P.pos); pd.shirt = SHIRTS[rng.i(8)]; pd.pants = rgb(60, 60, 64); pd.skin = SKINS[rng.i(5)]; pd.speed = 5.f;
        break;
      }
    }
    notify(std::string("CARJACKED ") + D.name + "!", 0xFF9A3C);
  } else notify(std::string("STOLEN ") + D.name + (v.type == 3 ? "  -  HOLD ON!" : ""), 0xFF9A3C);
  v.st = VS_PLAYER; v.throttle = v.brake = v.steerIn = 0; v.hand = false; v.planned = false; v.inside = false;
  P.car = k; P.vel = {};
}

void Game::exitCar() {
  Vehicle& v = cars[P.car];
  if (len(v.vel) > 9.f) { notify("SLOW DOWN FIRST", 0xFF7070); return; }
  const CarDef& D = CARS[v.type];
  P.pos = v.pos + rtOf(v.yaw) * (-(D.wid * 0.5f + 0.8f));
  V2 n; world.resolve(P.pos, 0.45f, n);
  P.yaw = v.yaw; P.vel = {};
  v.st = VS_PARKED; v.throttle = 0; v.brake = 1; v.steerIn = 0;
  P.car = -1;
}

void Game::killPlayer() {
  if (P.deadT > 0) return;
  P.deadT = 3.6f; P.hp = 0;
  notify("KNOCKED OUT", 0xFF5050);
  if (P.car >= 0) { cars[P.car].st = VS_PARKED; cars[P.car].throttle = 0; cars[P.car].brake = 1; P.car = -1; }
  P.vel = {};
  trauma = 1.f;
}

void Game::updatePlayer(float dt) {
  P.muzzle = std::max(0.f, P.muzzle - dt);
  P.hurtCd = std::max(0.f, P.hurtCd - dt);
  if (P.deadT > 0) {
    P.deadT -= dt;
    if (P.deadT <= 0) {
      int fee = std::min(P.money, 200);
      P.money -= fee; P.hp = 100; P.pos = world.spawn; P.yaw = START_YAW; camYaw = START_YAW;
      V2 n; world.resolve(P.pos, 0.45f, n);
      camFocus = {P.pos.x, 1.55f, P.pos.z};
      notify("HOSPITAL FEE  -" + fmtMoney(fee), 0xFF7070);
    }
    return;
  }
  if (P.hp < 100.f && P.hurtCd <= 0.f) P.hp = std::min(100.f, P.hp + 1.5f * dt);
  if (P.hp <= 0.f) { killPlayer(); return; }
  if (in.use) enterOrUse();
  if (in.swap && P.car < 0)
    for (int k = 1; k <= 4; k++) { int w = (P.weapon + k) % 4; if (P.owned[w]) { P.weapon = w; break; } }

  prompt.clear(); promptKind = 0;
  if (P.car >= 0) {
    Vehicle& v = cars[P.car];
    float sy = std::fabs(in.my) < 0.08f ? 0.f : in.my, sx = std::fabs(in.mx) < 0.06f ? 0.f : in.mx;
    v.throttle = clampf(sy, -1.f, 1.f); v.steerIn = clampf(sx * 1.25f, -1.f, 1.f); v.hand = in.brake; v.brake = 0;
    if (len(v.pos - world.garage) < 9.5f && len(v.vel) < 5.f) {
      const CarDef& D = CARS[v.type];
      promptKind = 2;
      prompt = std::string("SELL ") + D.name + "  +" + fmtMoney((int)(D.value * (0.35f + 0.65f * clampf(v.hp / D.hp, 0.f, 1.f))));
    }
    return;
  }

  // ---- on foot
  V2 fc = fwdOf(camYaw), rc = rtOf(camYaw);
  V2 wish = fc * in.my + rc * in.mx;
  float mag = std::min(1.f, len(wish));
  V2 wd = mag > 0.01f ? wish * (1.f / len(wish)) : V2{};
  P.vel = P.vel + (wd * (mag * 6.0f) - P.vel) * std::min(1.f, dt * 12.f);
  P.pos = P.pos + P.vel * dt;
  V2 n;
  if (world.resolve(P.pos, 0.42f, n)) { float vn = dot(P.vel, n); if (vn < 0) P.vel = P.vel - n * vn; }
  for (Vehicle& v : cars) {                                          // cars are solid; fast ones hurt
    if (v.st == VS_FREE) continue;
    V2 c[3]; float r; carCircles(v, c, r);
    float sp = len(v.vel);
    for (int k = 0; k < 3; k++) {
      V2 d = P.pos - c[k];
      float dd = len(d);
      if (dd < r + 0.42f && dd > 1e-4f) {
        P.pos = P.pos + d * ((r + 0.42f - dd) / dd);
        if (sp > 3.5f && P.hurtCd <= 0.f) { P.hp -= sp * 2.4f; P.hurtCd = 0.5f; P.vel = v.vel * 0.5f; trauma = std::min(1.f, trauma + 0.4f); }
      }
    }
  }
  float sp = len(P.vel);
  if (sp > 0.4f && !in.fire) P.yaw += wrapPi(std::atan2(P.vel.x, P.vel.z) - P.yaw) * std::min(1.f, dt * 14.f);
  P.walk += sp * dt * 1.7f;

  if (len(P.pos - world.gunShop) < 9.5f) {
    const Weapon& cw = WEAP[P.weapon];
    promptKind = 3;
    if (!cw.melee && P.ammo[P.weapon] < cw.maxAmmo) prompt = std::string("AMMO ") + cw.name + "  " + fmtMoney(cw.price / 5);
    else {
      int nw = -1;
      for (int w = 1; w < 4; w++) if (!P.owned[w]) { nw = w; break; }
      prompt = nw < 0 ? "ALL WEAPONS OWNED" : std::string("BUY ") + WEAP[nw].name + "  " + fmtMoney(WEAP[nw].price);
    }
  } else {
    int k = nearestCar(3.0f);
    if (k >= 0) { promptKind = 1; prompt = std::string(cars[k].st == VS_AI ? "CARJACK " : "ENTER ") + CARS[cars[k].type].name; }
  }
  firePlayer(dt);
  if (P.hp <= 0.f) killPlayer();
}

// ================================================================ combat
void Game::hurtPed(Ped& p, float dmg, V2 from) {
  if (p.st == 2) return;
  p.hp -= dmg;
  if (p.hp <= 0.f) {
    p.st = 2; p.stT = 0; p.vel = from * 3.5f;
    if (rng.f() < 0.85f) addPickup(picks, p.pos + from * 0.6f, 15 + 10 * rng.i(10));
    for (int i = 0; i < 6; i++) emit(1, {p.pos.x, 0.9f, p.pos.z}, {rng.r(-1, 1), rng.r(0.5f, 1.5f), rng.r(-1, 1)}, 0.7f, 0.35f, {0.6f, 0.6f, 0.6f});
  } else {
    p.st = 1; p.stT = 0;
    V2 away = norm(p.pos - P.pos);
    p.fleeDir = norm(away + V2{rng.r(-0.5f, 0.5f), rng.r(-0.5f, 0.5f)});
    p.speed = 5.2f;
  }
}

void Game::damageCar(int idx, float dmg) {
  Vehicle& v = cars[idx];
  if (v.st == VS_WRECK || v.st == VS_FREE) return;
  v.hp -= dmg;
  if (v.hp <= 0.f) explode(idx);
}

void Game::explode(int idx) {
  Vehicle& v = cars[idx];
  bool wasPlayer = v.st == VS_PLAYER;
  V2 pos = v.pos;
  v.st = VS_WRECK; v.hp = 0; v.burn = 14.f; v.throttle = 0; v.brake = 1; v.steerIn = 0;
  v.color = rgb(34, 30, 28);
  V3 c{pos.x, 0.8f, pos.z};
  for (int i = 0; i < 16; i++) emit(2, c, {rng.r(-5, 5), rng.r(2, 8), rng.r(-5, 5)}, rng.r(0.5f, 1.1f), rng.r(1.2f, 2.6f), {1.0f, rng.r(0.4f, 0.7f), 0.15f});
  for (int i = 0; i < 14; i++) emit(0, c, {rng.r(-12, 12), rng.r(2, 12), rng.r(-12, 12)}, rng.r(0.4f, 0.9f), 0.35f, {1.0f, 0.8f, 0.4f});
  for (int i = 0; i < 6; i++) emit(1, c, {rng.r(-2, 2), rng.r(2, 5), rng.r(-2, 2)}, rng.r(2.f, 3.5f), rng.r(1.5f, 2.8f), {0.12f, 0.12f, 0.12f});
  trauma = std::min(1.f, trauma + (len(pos - P.pos) < 40.f ? 0.9f : 0.2f));
  for (Ped& p : peds) if (p.alive && p.st != 2 && len(p.pos - pos) < 8.f) hurtPed(p, 999.f, norm(p.pos - pos));
  if (wasPlayer) {
    P.pos = pos + V2{2.f, 0.f}; P.car = -1; P.vel = {}; P.hp -= 45.f; P.hurtCd = 1.f;
    V2 n; world.resolve(P.pos, 0.45f, n);
    notify("CAR DESTROYED!", 0xFF5050);
  } else if (P.car < 0 && len(P.pos - pos) < 8.f) { P.hp -= 45.f * (1.f - len(P.pos - pos) / 8.f); P.hurtCd = 1.f; }
  for (size_t i = 0; i < cars.size(); i++) {
    if ((int)i == idx || cars[i].st == VS_FREE || cars[i].st == VS_WRECK) continue;
    float d = len(cars[i].pos - pos);
    if (d < 9.f) { cars[i].vel = cars[i].vel + norm(cars[i].pos - pos) * (6.f * (1.f - d / 9.f)); damageCar((int)i, 55.f * (1.f - d / 9.f)); }
  }
}

bool Game::hitScan(V2 o, V2 dir, float range, float dmg, bool melee, V3* endOut) {
  float best = range;
  int kind = 0, idx = -1;
  if (!melee) { float tw = world.rayWorld(o, dir, range); if (tw < best) { best = tw; kind = 3; } }
  auto rayCircle = [&](V2 c, float r) -> float {
    V2 oc = o - c;
    float b = dot(oc, dir), cq = dot(oc, oc) - r * r, disc = b * b - cq;
    if (disc < 0) return 1e9f;
    float t = -b - std::sqrt(disc);
    return t >= 0 ? t : 1e9f;
  };
  for (size_t i = 0; i < peds.size(); i++) {
    if (!peds[i].alive || peds[i].st == 2) continue;
    float t = rayCircle(peds[i].pos, 0.45f);
    if (t < best) { best = t; kind = 1; idx = (int)i; }
  }
  if (!melee)
    for (size_t i = 0; i < cars.size(); i++) {
      if (cars[i].st == VS_FREE || cars[i].st == VS_PLAYER) continue;
      V2 c[3]; float r; carCircles(cars[i], c, r);
      for (int k = 0; k < 3; k++) { float t = rayCircle(c[k], r); if (t < best) { best = t; kind = 2; idx = (int)i; } }
    }
  V2 hit = o + dir * best;
  if (endOut) *endOut = {hit.x, 1.2f, hit.z};
  if (kind == 1) { hurtPed(peds[idx], dmg, dir); if (!melee) emit(0, {hit.x, 1.1f, hit.z}, {0, 1, 0}, 0.15f, 0.2f, {1.0f, 0.6f, 0.4f}); }
  else if (kind == 2) { damageCar(idx, dmg * 0.3f); for (int i = 0; i < 4; i++) emit(0, {hit.x, 1.0f, hit.z}, {rng.r(-4, 4), rng.r(1, 5), rng.r(-4, 4)}, 0.3f, 0.18f, {1.0f, 0.85f, 0.5f}); }
  else if (kind == 3) for (int i = 0; i < 4; i++) emit(0, {hit.x, 1.2f, hit.z}, {rng.r(-4, 4) - dir.x * 3.f, rng.r(1, 5), rng.r(-4, 4) - dir.z * 3.f}, 0.3f, 0.18f, {1.0f, 0.85f, 0.5f});
  return kind != 0;
}

void Game::firePlayer(float dt) {
  P.fireCd -= dt;
  if (!in.fire || P.fireCd > 0.f) return;
  const Weapon& Wp = WEAP[P.weapon];
  if (!Wp.melee && P.ammo[P.weapon] <= 0) { P.fireCd = 0.4f; notify("OUT OF AMMO - FIND THE GUN SHOP", 0xFF7070); return; }
  P.fireCd = 1.f / Wp.rate;
  V2 face = fwdOf(camYaw), aim = face;
  float bestA = 0.5f;
  const Ped* tgt = nullptr;
  for (const Ped& p : peds) {
    if (!p.alive || p.st == 2) continue;
    V2 to = p.pos - P.pos;
    float dist = len(to);
    if (dist < 0.4f || dist > Wp.range * 0.9f) continue;
    float a = std::acos(clampf(dot(norm(to), face), -1.f, 1.f));
    if (a < bestA) { bestA = a; tgt = &p; }
  }
  if (tgt) aim = norm(tgt->pos - P.pos);
  P.yaw = std::atan2(aim.x, aim.z);
  P.muzzle = 0.07f;
  V2 rr = rtOf(P.yaw);
  V3 mz{P.pos.x + aim.x * 0.8f + rr.x * 0.25f, 1.25f, P.pos.z + aim.z * 0.8f + rr.z * 0.25f};
  if (Wp.melee) {
    hitScan(P.pos, aim, Wp.range, (float)Wp.dmg, true, nullptr);
  } else {
    P.ammo[P.weapon]--;
    for (int k = 0; k < Wp.pellets; k++) {
      float a = (rng.f() * 2.f - 1.f) * Wp.spread, cs = std::cos(a), sn = std::sin(a);
      V2 d{aim.x * cs - aim.z * sn, aim.x * sn + aim.z * cs};
      V3 end;
      hitScan(P.pos + aim * 0.4f, d, Wp.range, (float)Wp.dmg, false, &end);
      tracers.push_back({mz, end, 0.07f});
    }
    trauma = std::min(1.f, trauma + (P.weapon == 3 ? 0.18f : 0.05f));
    emit(0, mz, {aim.x * 3, 0.5f, aim.z * 3}, 0.08f, 0.35f, {1.0f, 0.8f, 0.4f});
  }
  for (Ped& p : peds)                                                   // gunfire scares everyone nearby
    if (p.alive && p.st == 0 && len(p.pos - P.pos) < 32.f) { p.st = 1; p.stT = 0; p.speed = 5.f; p.fleeDir = norm(norm(p.pos - P.pos) + V2{rng.r(-0.6f, 0.6f), rng.r(-0.6f, 0.6f)}); }
}

// ================================================================ vehicles
void Game::updateVehicles(float dt) {
  for (Vehicle& v : cars) if (v.st != VS_FREE) stepVehicle(v, dt);
}

void Game::stepVehicle(Vehicle& v, float dt) {
  const CarDef& D = CARS[v.type];
  bool dead = v.st == VS_PARKED || v.st == VS_WRECK;
  float thr = dead ? 0.f : v.throttle, brk = dead ? 1.f : v.brake, sIn = dead ? 0.f : v.steerIn;
  bool hand = !dead && v.hand;
  V2 f = fwdOf(v.yaw), r = rtOf(v.yaw);
  float vf = dot(v.vel, f), vl = dot(v.vel, r);
  float a = 0;
  if (thr > 0) { if (vf >= 0) { float k = vf / D.vmax; a = D.acc * thr * std::max(0.f, 1.f - k * k); } else a = D.acc * 2.2f * thr; }
  else if (thr < 0) {
    if (vf > 0.6f) a = D.acc * 1.5f * thr;
    else { float k = -vf / (D.vmax * 0.25f); a = D.acc * 0.6f * thr * std::max(0.f, 1.f - k * k); }
  }
  vf += a * dt;
  vf *= std::exp(-0.03f * dt);
  if (thr == 0.f) { float rr = 0.6f * dt; vf = std::fabs(vf) <= rr ? 0.f : vf - (vf > 0 ? rr : -rr); }
  float b = 20.f * brk * dt;
  vf = std::fabs(vf) <= b ? 0.f : vf - (vf > 0 ? b : -b);
  float grip = dead ? 6.f : (hand ? 1.4f : D.grip);
  vl *= std::exp(-grip * dt);
  float ms = maxSteer(D, vf);
  v.steer += clampf(-sIn * ms - v.steer, -3.5f * dt, 3.5f * dt);
  float yawRate = vf / D.wb * std::tan(v.steer);
  if (hand) yawRate *= 1.5f;
  float yMax = D.grip * 4.5f * (hand ? 1.5f : 1.f) / std::max(std::fabs(vf), 4.f);
  yawRate = clampf(yawRate, -yMax, yMax);
  v.vel = f * vf + r * vl;
  v.yaw += yawRate * dt;
  v.pos = v.pos + v.vel * dt;
  float acc = (vf - v.lastVf) / std::max(dt, 1e-4f);
  v.lastVf = vf;
  float tp = clampf(-acc * 0.0035f, -0.06f, 0.06f), tr = clampf(yawRate * vf * 0.0055f, -0.08f, 0.08f);
  v.pitch += (tp - v.pitch) * std::min(1.f, dt * 8.f);
  v.roll += (tr - v.roll) * std::min(1.f, dt * 8.f);
}

float Game::obstacleLimit(const Vehicle& v, V2 fwd, V2 rt, float vf, int self) {
  if (v.ghost > 0.f) return 99.f;
  const CarDef& D = CARS[v.type];
  float best = 99.f, range = 7.f + std::max(vf, 0.f) * 1.4f;
  auto consider = [&](V2 pos, float halfLen, float halfW, float ov) {
    V2 rel = pos - v.pos;
    float fd = dot(rel, fwd), lat = std::fabs(dot(rel, rt));
    if (fd <= 0.f || fd > range + halfLen || lat > D.wid * 0.5f + halfW + 0.55f) return;
    float gap = fd - D.len * 0.5f - halfLen;
    best = std::min(best, std::sqrt(std::max(0.f, ov * ov + 2.f * 5.f * std::max(0.f, gap - 1.8f))));
  };
  for (size_t k = 0; k < cars.size(); k++) {
    if ((int)k == self || cars[k].st == VS_FREE) continue;
    const Vehicle& o = cars[k];
    consider(o.pos, CARS[o.type].len * 0.5f, CARS[o.type].wid * 0.5f, std::max(0.f, dot(o.vel, fwd)));
  }
  for (const Ped& p : peds) if (p.alive && p.st != 2) consider(p.pos, 0.35f, 0.35f, 0.f);
  if (P.car < 0 && P.deadT <= 0) consider(P.pos, 0.4f, 0.4f, 0.f);
  return best;
}

void Game::planPath(Vehicle& v) {
  int dOut = v.move == 0 ? v.d : (v.move == 1 ? (v.d + 1) & 3 : (v.d + 3) & 3);
  int mi = v.ni + DX[dOut], mj = v.nj + DZ[dOut];
  v.nextMove = nodeOk(mi, mj) ? chooseMove(rng, mi, mj, dOut) : 0;
  v.exitLane = v.nextMove == 2 ? 0 : (v.nextMove == 1 ? 1 : rng.i(2));
  v.dOut = dOut;
  V2 p0 = stopPoint(v.ni, v.nj, v.d, v.lane), p1 = exitPoint(v.ni, v.nj, dOut, v.exitLane), c;
  if (dOut == v.d) c = (p0 + p1) * 0.5f;
  else if ((v.d & 1) == 0) c = {p1.x, p0.z};
  else c = {p0.x, p1.z};
  v.pn = 6;
  for (int k = 0; k < 6; k++) { float t = (k + 1) / 6.f, u = 1.f - t; v.path[k] = p0 * (u * u) + c * (2.f * u * t) + p1 * (t * t); }
  v.pi = 0; v.planned = true;
}

void Game::aiDrive(Vehicle& v, int idx, float dt) {
  const CarDef& D = CARS[v.type];
  V2 fwd = fwdOf(v.yaw), rt = rtOf(v.yaw);
  float vf = dot(v.vel, fwd), vdes = v.cruise;
  V2 target = v.pos + fwd * 10.f;
  if (!v.inside) {
    V2 dir = dirOf(v.d), stop = stopPoint(v.ni, v.nj, v.d, v.lane);
    float s = dot(v.pos - stop, dir);                                   // < 0 before the stop line
    if (!v.planned && s > -34.f) planPath(v);
    float L = clampf(5.f + 0.3f * std::fabs(vf), 6.f, 15.f);
    target = (s + L > 0.f && v.planned) ? v.path[0] : stop + dir * (s + L);
    float ds = -s;
    if (ds < 70.f) {
      int st = sigState(v.ni, v.nj, v.d, v.move == 2, time);
      float stopAt = ds - (D.len * 0.5f + 0.7f);
      bool must = st == 0 || (st == 1 && vf * vf / (2.f * 5.f) <= stopAt);
      if (must && stopAt > -3.f) vdes = std::min(vdes, std::sqrt(std::max(0.f, 2.f * 3.2f * std::max(stopAt, 0.f))));
    }
    if (v.move == 2 && ds < 30.f) {                                     // permissive left: yield to oncoming traffic
      V2 c = nodePos(v.ni, v.nj), rgt = rightOf(v.d);
      bool oncoming = false;
      for (size_t k = 0; k < cars.size() && !oncoming; k++) {
        const Vehicle& o = cars[k];
        if ((int)k == idx || o.st == VS_FREE || o.st == VS_PARKED || o.st == VS_WRECK) continue;
        V2 rel = o.pos - c;
        float along = dot(rel, dir), lat = dot(rel, rgt);
        if (along < -6.f || along > 48.f || std::fabs(lat) > 11.f) continue;
        if (dot(fwdOf(o.yaw), dir) > -0.6f) continue;                    // not heading towards us
        if (o.st == VS_AI && o.move == 2) continue;                      // opposing left turns do not cross
        if (len(o.vel) < 1.f && along > 12.f) continue;                  // queued far away
        oncoming = true;
      }
      if (oncoming) {
        float stopAt = ds - (D.len * 0.5f + 0.7f);
        if (stopAt > -3.f) vdes = std::min(vdes, std::sqrt(std::max(0.f, 2.f * 3.2f * std::max(stopAt, 0.f))));
      }
    }
    if (v.move != 0 && ds < 24.f) vdes = std::min(vdes, 8.f);
    if (s >= -0.2f && v.planned) { v.inside = true; v.pi = 0; }
  } else if (v.pi >= v.pn) {                                            // left the junction
    v.d = v.dOut; v.ni += DX[v.dOut]; v.nj += DZ[v.dOut]; v.lane = v.exitLane; v.move = v.nextMove;
    v.inside = false; v.planned = false; v.pn = 0; v.pi = 0;
  } else {
    if (len(v.path[v.pi] - v.pos) < std::max(3.f, 0.22f * std::fabs(vf))) v.pi++;
    if (v.pi < v.pn) target = v.path[v.pi];
    if (v.move != 0) vdes = std::min(vdes, 8.5f);
  }
  float obs = obstacleLimit(v, fwd, rt, vf, idx);
  bool blocked = obs < vdes - 0.5f && obs < 2.5f;
  vdes = std::min(vdes, obs);
  if (blocked && vf < 0.6f) v.stuck += dt; else v.stuck = std::max(0.f, v.stuck - dt * 2.f);
  if (v.stuck > 9.f) { v.ghost = 3.5f; v.stuck = 0; }
  if (v.ghost > 0.f) v.ghost -= dt;

  V2 to = target - v.pos;
  float Ld = std::max(3.f, len(to)), ang = wrapPi(std::atan2(to.x, to.z) - v.yaw);
  float delta = std::atan(D.wb * 2.f * std::sin(ang) / Ld);
  v.steerIn = clampf(-delta / maxSteer(D, vf), -1.f, 1.f);
  float dv = vdes - vf;
  if (dv > 0.f) { v.throttle = clampf(dv * 0.6f, 0.f, 0.85f); v.brake = 0.f; }
  else { v.throttle = 0.f; v.brake = clampf(-dv * 0.22f, 0.f, 0.5f); }
  if (vdes < 0.3f) { v.throttle = 0.f; v.brake = vf > 0.5f ? 0.6f : 0.9f; }
}

void Game::collide(float dt) {
  (void)dt;
  // cars vs world
  for (size_t i = 0; i < cars.size(); i++) {
    Vehicle& v = cars[i];
    if (v.st == VS_FREE) continue;
    V2 c[3]; float r; carCircles(v, c, r);
    bool hit = false; V2 nrm;
    for (int k = 0; k < 3; k++) {
      V2 p = c[k], n;
      if (world.resolve(p, r, n)) { V2 push = p - c[k]; v.pos = v.pos + push; for (int q = k + 1; q < 3; q++) c[q] = c[q] + push; nrm = n; hit = true; }
    }
    if (hit) {
      float vn = dot(v.vel, nrm);
      if (vn < 0) {
        v.vel = v.vel - nrm * (vn * 1.25f);
        float dmg = std::max(0.f, -vn - 3.5f) * 2.2f;
        if (dmg > 0.f) {
          if (v.st == VS_PLAYER) trauma = std::min(1.f, trauma + dmg * 0.02f);
          if (v.st != VS_PARKED) damageCar((int)i, dmg);
        }
      }
    }
  }
  // cars vs cars
  for (size_t a = 0; a < cars.size(); a++) {
    if (cars[a].st == VS_FREE) continue;
    for (size_t b = a + 1; b < cars.size(); b++) {
      if (cars[b].st == VS_FREE) continue;
      Vehicle &A = cars[a], &B = cars[b];
      const CarDef &DA = CARS[A.type], &DB = CARS[B.type];
      if (len(A.pos - B.pos) > (DA.len + DB.len) * 0.5f + 0.3f) continue;
      V2 ca[3], cb[3]; float ra, rb;
      carCircles(A, ca, ra); carCircles(B, cb, rb);
      float best = 0; V2 n;
      for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) {
        V2 d = cb[j] - ca[i];
        float dd = len(d), pen = ra + rb - dd;
        if (pen > best && dd > 1e-4f) { best = pen; n = d * (1.f / dd); }
      }
      if (best <= 0.f) continue;
      float ma = DA.mass * (A.st == VS_PARKED ? 1.6f : 1.f), mb = DB.mass * (B.st == VS_PARKED ? 1.6f : 1.f);
      A.pos = A.pos - n * (best * mb / (ma + mb)); B.pos = B.pos + n * (best * ma / (ma + mb));
      float vn = dot(B.vel - A.vel, n);
      if (vn < 0) {
        float j = -(1.f + 0.3f) * vn / (1.f / ma + 1.f / mb);
        A.vel = A.vel - n * (j / ma); B.vel = B.vel + n * (j / mb);
        float dmg = std::max(0.f, -vn - 3.f) * 2.2f;
        if (dmg > 0.f) {
          if (A.st == VS_PLAYER || B.st == VS_PLAYER) trauma = std::min(1.f, trauma + dmg * 0.02f);
          damageCar((int)a, dmg * mb / (ma + mb) * 2.f); damageCar((int)b, dmg * ma / (ma + mb) * 2.f);
        }
      }
    }
  }
}

// ================================================================ pedestrians, pickups, effects
void Game::updatePeds(float dt) {
  for (Ped& p : peds) {
    if (!p.alive) continue;
    p.t += dt * (p.st == 1 ? 2.2f : 1.f);
    if (p.st == 2) { p.stT += dt; p.pos = p.pos + p.vel * dt; p.vel = p.vel * std::exp(-6.f * dt); if (p.stT > 6.f) p.alive = false; continue; }
    if (p.st == 0) {
      const Block& B = world.blocks[p.bi][p.bj];
      p.s += p.dirSign * p.speed * dt;
      V2 d;
      p.pos = perimPoint(B, p.off, p.s, &d);
      p.yaw += wrapPi(std::atan2(d.x * p.dirSign, d.z * p.dirSign) - p.yaw) * std::min(1.f, dt * 8.f);
    } else {
      p.pos = p.pos + p.fleeDir * (p.speed * dt);
      V2 n; world.resolve(p.pos, 0.3f, n);
      p.yaw = std::atan2(p.fleeDir.x, p.fleeDir.z);
      p.stT += dt;
      if (p.stT > 12.f && len(p.pos - P.pos) > 45.f) p.alive = false;
    }
    for (Vehicle& v : cars) {                                           // run-over check
      if (v.st == VS_FREE || len(v.pos - p.pos) > 4.f) continue;
      float sp = len(v.vel);
      if (sp < 2.5f) continue;
      V2 c[3]; float r; carCircles(v, c, r);
      for (int k = 0; k < 3; k++) if (len(c[k] - p.pos) < r + 0.35f) { hurtPed(p, sp * 9.f, norm(v.vel)); p.vel = v.vel * 0.55f; break; }
    }
    if (len(p.pos - P.pos) > 240.f) p.alive = false;
  }
}

void Game::updateFX(float dt) {
  for (size_t i = 0; i < parts.size();) {
    Part& q = parts[i];
    q.life -= dt;
    if (q.life <= 0.f) { parts[i] = parts.back(); parts.pop_back(); continue; }
    q.p = q.p + q.v * dt;
    if (q.kind == 0) q.v.y -= 9.f * dt;
    else if (q.kind == 1) q.v = q.v * (1.f - dt * 0.7f);
    else q.v = q.v * (1.f - dt * 2.f);
    i++;
  }
  for (size_t i = 0; i < tracers.size();) { tracers[i].life -= dt; if (tracers[i].life <= 0) { tracers[i] = tracers.back(); tracers.pop_back(); } else i++; }
  for (size_t i = 0; i < notes.size();) { notes[i].t -= dt; if (notes[i].t <= 0) notes.erase(notes.begin() + i); else i++; }
  trauma = std::max(0.f, trauma - dt * 1.4f);

  for (Vehicle& v : cars) {                                             // fire & smoke
    if (v.st == VS_FREE) continue;
    const CarDef& D = CARS[v.type];
    v.smokeT -= dt;
    if (v.st == VS_WRECK && v.burn > 0.f) {
      v.burn -= dt;
      if (v.smokeT <= 0.f) {
        v.smokeT = 0.09f;
        V3 c{v.pos.x + rng.r(-0.8f, 0.8f), 0.9f, v.pos.z + rng.r(-1.2f, 1.2f)};
        emit(2, c, {rng.r(-0.5f, 0.5f), rng.r(1.5f, 3.f), rng.r(-0.5f, 0.5f)}, 0.5f, rng.r(0.8f, 1.5f), {1.0f, 0.5f, 0.12f});
        emit(1, c, {rng.r(-0.4f, 0.4f), rng.r(2.f, 3.5f), rng.r(-0.4f, 0.4f)}, 2.2f, 1.4f, {0.1f, 0.1f, 0.1f});
      }
    } else if (v.st != VS_WRECK && v.hp < D.hp * 0.35f && v.smokeT <= 0.f) {
      v.smokeT = 0.12f;
      V2 f = fwdOf(v.yaw);
      emit(1, {v.pos.x + f.x * D.len * 0.35f, 1.0f, v.pos.z + f.z * D.len * 0.35f}, {rng.r(-0.3f, 0.3f), 1.8f, rng.r(-0.3f, 0.3f)}, 1.6f, 0.9f, {0.2f, 0.2f, 0.2f});
    }
  }
  for (Pickup& k : picks) {                                             // cash
    if (!k.alive) continue;
    k.t += dt;
    float reach = P.car >= 0 ? 3.2f : 1.5f;
    if (P.deadT <= 0 && len(k.pos - P.pos) < reach) {
      k.alive = false; P.money += k.value;
      notify("+" + fmtMoney(k.value), 0x6CFF7A);
      int alive = 0;
      for (const Pickup& q : picks) if (q.alive) alive++;
      if (alive < 24) spawnPickup();
    }
  }
}

// ================================================================ camera
void Game::updateCamera(float dt) {
  bool drag = std::fabs(in.camDX) + std::fabs(in.camDY) > 0.01f;
  camYaw -= in.camDX * 0.0055f;
  camPitch = clampf(camPitch + in.camDY * 0.004f, 0.03f, 1.1f);
  V3 focus;
  float dist, sp = 0;
  if (P.car >= 0) {
    const Vehicle& v = cars[P.car];
    sp = len(v.vel);
    float ty = v.yaw;
    if (sp > 5.f) { float dv = wrapPi(std::atan2(v.vel.x, v.vel.z) - v.yaw); if (std::fabs(dv) < 1.9f) ty = v.yaw + dv * 0.45f; }
    if (!drag) {
      camYaw += wrapPi(ty - camYaw) * std::min(1.f, dt * (2.2f + sp * 0.03f));
      camPitch += (0.2f + std::min(sp, 80.f) * 0.0008f - camPitch) * std::min(1.f, dt * 1.5f);
    }
    focus = {v.pos.x, 1.2f, v.pos.z};
    dist = 7.4f + std::min(sp, 80.f) * 0.05f;
  } else {
    focus = {P.pos.x, 1.55f, P.pos.z};
    dist = 5.4f;
  }
  camFocus = camFocus + (focus - camFocus) * std::min(1.f, dt * (P.car >= 0 ? 16.f : 14.f));
  float cp = std::cos(camPitch), spn = std::sin(camPitch);
  V3 f{std::sin(camYaw) * cp, -spn, std::cos(camYaw) * cp};
  V2 hd{-f.x, -f.z};
  float hl = len(hd), dd = dist;
  if (hl > 1e-3f) {
    float maxT = dd * hl, t = world.rayWorld({camFocus.x, camFocus.z}, hd * (1.f / hl), maxT, false);
    if (t < maxT) dd = std::max(1.2f, dd * ((t - 0.7f) / maxT));
  }
  camPos = camFocus - f * dd;
  camPos.y = std::max(camPos.y, 0.8f);
  fov += (1.05f + std::min(sp, 85.f) * 0.0035f - fov) * std::min(1.f, dt * 3.f);
}


// ==================== EMBEDDED SOURCE: game_gfx.cpp ====================
// NEON CITY - 3D drawing of vehicles, people, pickups, particles

namespace {
struct Basis { V3 r, u, f; };
Basis makeBasis(float yaw, float pitch, float roll) {
  V3 f0{std::sin(yaw), 0, std::cos(yaw)}, r0{-std::cos(yaw), 0, std::sin(yaw)}, u0{0, 1, 0};
  V3 f = f0 * std::cos(pitch) + u0 * std::sin(pitch);
  V3 u1 = cross(r0, f);
  Basis b;
  b.r = r0 * std::cos(roll) - u1 * std::sin(roll);
  b.u = u1 * std::cos(roll) + r0 * std::sin(roll);
  b.f = f;
  return b;
}
}  // namespace

void Game::drawCar(Renderer& R, const Vehicle& v) {
  const CarDef& D = CARS[v.type];
  const float night = R.sky.night;
  Basis B = makeBasis(v.yaw, v.pitch, v.roll);
  V3 o{v.pos.x, 0.03f, v.pos.z};
  auto P = [&](float x, float y, float z) { return o + B.r * x + B.u * y + B.f * z; };
  const bool wreck = v.st == VS_WRECK;
  const bool lit = (v.st == VS_AI || v.st == VS_PLAYER);
  Col body = wreck ? rgb(36, 32, 30) : v.color;
  Col glass = wreck ? rgb(14, 12, 12) : rgb(28, 40, 62);
  Col trim = rgb(30, 31, 35);
  const float hl = D.len * 0.5f, hw = D.wid * 0.5f;
  auto frus = [&](float y0, float y1, float bw, float bz0, float bz1, float tw, float tz0, float tz1, Col c) {
    V3 q[8] = {P(-bw, y0, bz0), P(bw, y0, bz0), P(bw, y0, bz1), P(-bw, y0, bz1), P(-tw, y1, tz0), P(tw, y1, tz0), P(tw, y1, tz1), P(-tw, y1, tz1)};
    R.solid(q, c);
  };
  auto bxs = [&](V3 c, V3 half, Col col, Mat m = M_FLAT) { R.box(c, B.r, B.u, B.f, half, col, m); };

  V2 f2 = fwdOf(v.yaw), r2 = rtOf(v.yaw);
  {  // soft ground shadow
    float sl = hl * 1.12f, sw = hw * 1.2f, y = CURB + 0.03f;
    V3 c{v.pos.x, y, v.pos.z}, F{f2.x * sl, 0, f2.z * sl}, Rr{r2.x * sw, 0, r2.z * sw};
    R.decal(c - F - Rr, c + F - Rr, c + F + Rr, c - F + Rr, D_SHADOW_BLOB, Col{}, 0.9f);
  }

  frus(D.y0, D.bodyTop, hw, -hl, hl, hw * 0.92f, -hl + D.tail, hl - D.nose, body);                       // body
  float cw = hw * 0.9f, ctw = hw * 0.78f;
  frus(D.bodyTop, D.cabTop, cw, D.cb0, D.cb1, ctw, D.ct0, D.ct1, glass);                                  // cabin glass
  frus(D.cabTop - 0.02f, D.cabTop + 0.04f, ctw + 0.02f, D.ct0 - 0.03f, D.ct1 + 0.03f, ctw + 0.02f, D.ct0 - 0.03f, D.ct1 + 0.03f, body);   // roof
  bxs(P(0, D.y0 + 0.14f, hl - 0.03f), {hw * 0.96f, 0.12f, 0.07f}, trim);                                  // bumpers
  bxs(P(0, D.y0 + 0.14f, -hl + 0.03f), {hw * 0.96f, 0.12f, 0.07f}, trim);

  for (int i = 0; i < 4; i++) {                                                                           // wheels
    float sx = (i & 1) ? 1.f : -1.f, sz = (i & 2) ? 1.f : -1.f;
    V3 wf = B.f, wr = B.r;
    if (sz > 0) { float cs = std::cos(v.steer), sn = std::sin(v.steer); wf = B.f * cs - B.r * sn; wr = B.r * cs + B.f * sn; }
    R.box(P(sx * (hw - 0.1f), D.wr, sz * D.wb * 0.5f), wr, B.u, wf, {0.13f, D.wr, D.wr}, rgb(20, 20, 22));
  }

  if (!wreck) {                                                                                           // lights
    float hy = D.y0 + (D.bodyTop - D.y0) * 0.55f;
    float zf = hl - D.nose * 0.55f + 0.015f, zr = -hl + D.tail * 0.55f - 0.015f;
    bool braking = v.brake > 0.15f || (v.throttle < -0.1f && v.lastVf > 0.5f);
    Col headC = Col{1.0f, 0.97f, 0.85f} * (1.0f + 1.4f * night);
    Col tailC = braking ? Col{2.6f, 0.15f, 0.1f} : Col{1.2f + 0.6f * night, 0.08f, 0.06f};
    for (int sg = -1; sg <= 1; sg += 2) {
      bxs(P(sg * (hw - 0.4f), hy, zf), {0.22f, 0.07f, 0.03f}, headC, M_EMIT);
      bxs(P(sg * (hw - 0.35f), hy + 0.02f, zr), {0.24f, 0.06f, 0.03f}, tailC, M_EMIT);
      if (lit && night > 0.2f) R.sprite(P(sg * (hw - 0.4f), hy, hl + 0.25f), 0.6f, {1.0f, 0.95f, 0.8f}, 0.8f * night);
      if (lit && (braking || night > 0.2f)) R.sprite(P(sg * (hw - 0.35f), hy, zr - 0.2f), 0.45f, {1.0f, 0.1f, 0.05f}, (braking ? 0.9f : 0.35f) * (0.4f + 0.6f * night));
    }
    if (lit && night > 0.2f) {                                                                            // head-light pool on the road
      V3 c{v.pos.x + f2.x * (hl + 6.f), CURB + 0.05f, v.pos.z + f2.z * (hl + 6.f)}, F{f2.x * 6.5f, 0, f2.z * 6.5f}, Rr{r2.x * 4.5f, 0, r2.z * 4.5f};
      R.decal(c - F - Rr, c + F - Rr, c + F + Rr, c - F + Rr, D_GLOW, {1.0f, 0.92f, 0.75f}, 0.4f * night);
    }
  }
  if (v.type == 3 && !wreck) {                                                                            // hypercar: wing + neon underglow
    bxs(P(0, D.bodyTop + 0.34f, -hl + 0.45f), {hw * 0.88f, 0.025f, 0.2f}, body * 0.6f);
    for (int sg = -1; sg <= 1; sg += 2) bxs(P(sg * hw * 0.45f, D.bodyTop + 0.17f, -hl + 0.45f), {0.03f, 0.17f, 0.06f}, trim);
    bxs(P(0, D.y0 + 0.06f, hl - 0.05f), {hw * 0.9f, 0.03f, 0.12f}, trim);
    V3 c{v.pos.x, CURB + 0.06f, v.pos.z}, F{f2.x * (hl + 0.7f), 0, f2.z * (hl + 0.7f)}, Rr{r2.x * (hw + 0.7f), 0, r2.z * (hw + 0.7f)};
    R.decal(c - F - Rr, c + F - Rr, c + F + Rr, c - F + Rr, D_GLOW, {0.1f, 0.75f, 1.0f}, 0.3f + 0.6f * night);
  }
  if (v.type == 2 && !wreck) {                                                                            // muscle: hood scoop + stripes
    float z1 = hl - D.nose - 0.05f;
    bxs(P(0, D.bodyTop + 0.07f, (D.cb1 + z1) * 0.5f + 0.1f), {0.28f, 0.07f, 0.3f}, trim);
    Col st = rgb(235, 235, 235);
    for (int sg = -1; sg <= 1; sg += 2) {
      float zc = (D.cb1 + 0.05f + z1) * 0.5f, zh = (z1 - D.cb1 - 0.05f) * 0.5f;
      bxs(P(sg * 0.32f, D.bodyTop + 0.012f, zc), {0.1f, 0.006f, zh}, st);
      float tz0 = -hl + D.tail + 0.05f, tz1 = D.cb0 - 0.05f;
      bxs(P(sg * 0.32f, D.bodyTop + 0.012f, (tz0 + tz1) * 0.5f), {0.1f, 0.006f, (tz1 - tz0) * 0.5f}, st);
    }
  }
}

void Game::drawPerson(Renderer& R, V2 pos, float yaw, float phase, Col shirt, Col pants, Col skin, bool down, bool gun) {
  V3 f{std::sin(yaw), 0, std::cos(yaw)}, r{-std::cos(yaw), 0, std::sin(yaw)}, u{0, 1, 0};
  V3 o{pos.x, 0.03f, pos.z};
  {
    float y = CURB + 0.03f;
    R.decal({pos.x - 0.6f, y, pos.z - 0.6f}, {pos.x + 0.6f, y, pos.z - 0.6f}, {pos.x + 0.6f, y, pos.z + 0.6f}, {pos.x - 0.6f, y, pos.z + 0.6f}, D_SHADOW_BLOB, Col{}, 0.8f);
  }
  if (down) {                                                         // lying on the ground
    R.box(o + V3{0, 0.16f, 0} + f * 0.1f, r, u, f, {0.25f, 0.15f, 0.3f}, shirt);
    R.box(o + V3{0, 0.14f, 0} - f * 0.65f, r, u, f, {0.17f, 0.12f, 0.4f}, pants);
    R.box(o + V3{0, 0.14f, 0} + f * 0.55f, r, u, f, {0.12f, 0.12f, 0.12f}, skin);
    return;
  }
  float sw = std::sin(phase) * 0.7f;
  auto limb = [&](V3 hip, float ang, float length, float hx, Col c) {         // hangs from `hip`, swings forward for ang > 0
    V3 uu = u * std::cos(ang) - f * std::sin(ang), ff = f * std::cos(ang) + u * std::sin(ang);
    R.box(hip - uu * (length * 0.5f), r, uu, ff, {hx, length * 0.5f, hx}, c);
  };
  limb(o + V3{0, 0.92f, 0} + r * 0.11f, sw, 0.92f, 0.09f, pants);
  limb(o + V3{0, 0.92f, 0} - r * 0.11f, -sw, 0.92f, 0.09f, pants);
  R.box(o + V3{0, 1.22f, 0}, r, u, f, {0.24f, 0.3f, 0.13f}, shirt);
  R.box(o + V3{0, 1.66f, 0}, r, u, f, {0.12f, 0.13f, 0.12f}, skin);
  limb(o + V3{0, 1.48f, 0} - r * 0.34f, sw, 0.62f, 0.07f, shirt);
  if (gun) {
    limb(o + V3{0, 1.48f, 0} + r * 0.34f, 1.5f, 0.62f, 0.07f, shirt);
    R.box(o + V3{0, 1.45f, 0} + r * 0.34f + f * 0.62f, r, u, f, {0.04f, 0.07f, 0.17f}, rgb(28, 28, 30));
  } else limb(o + V3{0, 1.48f, 0} + r * 0.34f, -sw, 0.62f, 0.07f, shirt);
}

void Game::draw(Renderer& R) {
  Sky sky = makeSky(hour, time);
  float sh = trauma * trauma * 0.3f;
  V3 cp = camPos + V3{rng.r(-sh, sh), rng.r(-sh, sh), rng.r(-sh, sh)};
  Camera cam = makeCamera(cp, camYaw, -camPitch, fov);
  R.begin(cam, sky);
  world.draw(R, time);

  for (const Vehicle& v : cars) {
    if (v.st == VS_FREE || !R.inView({v.pos.x, 1.0f, v.pos.z}, 4.5f, 240.f)) continue;
    drawCar(R, v);
  }
  for (const Ped& p : peds) {
    if (!p.alive || !R.inView({p.pos.x, 1.0f, p.pos.z}, 1.5f, 170.f)) continue;
    drawPerson(R, p.pos, p.yaw, p.t * 6.f, p.shirt, p.pants, p.skin, p.st == 2, false);
  }
  if (P.car < 0) {
    const Weapon& wp = WEAP[P.weapon];
    drawPerson(R, P.pos, P.yaw, P.walk * 2.2f, rgb(222, 86, 40), rgb(40, 48, 84), rgb(214, 160, 120), P.deadT > 0.f, !wp.melee && (in.fire || P.muzzle > 0.f));
    if (P.muzzle > 0.f) {
      V2 a = fwdOf(P.yaw), rr = rtOf(P.yaw);
      R.sprite({P.pos.x + a.x * 0.95f + rr.x * 0.25f, 1.28f, P.pos.z + a.z * 0.95f + rr.z * 0.25f}, 0.9f, {1.0f, 0.8f, 0.4f}, 1.5f);
    }
  }
  for (const Pickup& k : picks) {
    if (!k.alive || !R.inView({k.pos.x, 1.0f, k.pos.z}, 1.5f, 190.f)) continue;
    float bob = 0.95f + 0.12f * std::sin(k.t * 3.f), ang = k.t * 2.f;
    V3 f{std::sin(ang), 0, std::cos(ang)}, r{-std::cos(ang), 0, std::sin(ang)}, up{0, 1, 0};
    R.box({k.pos.x, bob, k.pos.z}, r, up, f, {0.34f, 0.09f, 0.18f}, rgb(60, 180, 80));
    R.box({k.pos.x, bob + 0.095f, k.pos.z}, r, up, f, {0.12f, 0.02f, 0.19f}, rgb(235, 235, 200));
    R.sprite({k.pos.x, bob, k.pos.z}, 0.9f, {0.2f, 1.0f, 0.4f}, 0.6f);
  }
  for (const Part& q : parts) {
    float t = q.life / q.max;
    if (q.kind == 0) R.sprite(q.p, q.size, q.c, t, true);
    else if (q.kind == 2) R.sprite(q.p, q.size * (0.6f + 0.6f * t), q.c, t * 1.2f, true);
    else R.sprite(q.p, q.size * (1.6f - t), q.c, 0.55f * t + 0.1f, false);
  }
  for (const Tracer& t : tracers) {
    V3 d = t.b - t.a;
    if (len(d) < 0.1f) continue;
    V3 side = norm(cross(d, cam.pos - t.a)) * 0.04f;
    R.decal(t.a - side, t.a + side, t.b + side, t.b - side, D_FLAT, {1.0f, 0.85f, 0.45f}, 0.9f * (t.life / 0.07f));
  }
  R.end();
  drawHUD(R);
  if (showTouch) drawTouchUI(R);
}


// ==================== EMBEDDED SOURCE: game_ui.cpp ====================
// NEON CITY - touch input, HUD, minimap

namespace {
struct Btn { float x, y, r; };
Btn btnFire(int W, int H) { return {W - 66.f, H - 78.f, 38.f}; }
Btn btnUse(int W, int H) { return {W - 152.f, H - 42.f, 28.f}; }
Btn btnSwap(int W, int H) { return {W - 40.f, H - 160.f, 24.f}; }
bool inBtn(const Btn& b, float x, float y) { float dx = x - b.x, dy = y - b.y, r = b.r + 6.f; return dx * dx + dy * dy <= r * r; }
constexpr float STICK_R = 44.f;
}  // namespace

void Game::touch(int id, int phase, float x, float y) {
  if (phase == 0) {
    Touch* t = nullptr;
    for (auto& q : touches) if (q.id < 0) { t = &q; break; }
    if (!t) return;
    bool stickTaken = false;
    for (auto& q : touches) if (q.id >= 0 && q.role == 1) stickTaken = true;
    *t = Touch();
    t->id = id; t->x = t->sx = t->lx = x; t->y = t->sy = t->ly = y;
    if (inBtn(btnFire(W, H), x, y)) t->role = 2;
    else if (inBtn(btnUse(W, H), x, y)) { t->role = 3; useEdge = true; }
    else if (P.car < 0 && inBtn(btnSwap(W, H), x, y)) { t->role = 4; swapEdge = true; }
    else if (x < W * 0.45f && y > H * 0.28f && !stickTaken) t->role = 1;
    else t->role = 5;
    return;
  }
  for (auto& q : touches) {
    if (q.id != id) continue;
    if (phase == 1) {
      q.x = x; q.y = y;
      if (q.role == 5) { camAccX += x - q.lx; camAccY += y - q.ly; q.lx = x; q.ly = y; }
    } else q.id = -1;
  }
}

void Game::gatherInput() {
  in = Input();
  in.mx = kb.mx; in.my = kb.my; in.fire = kb.fire; in.brake = kb.brake; in.use = kb.use; in.swap = kb.swap;
  in.camDX = kb.camDX; in.camDY = kb.camDY;
  kb.use = kb.swap = false; kb.camDX = kb.camDY = 0;
  bool held = false;
  for (const Touch& t : touches) {
    if (t.id < 0) continue;
    if (t.role == 1) {
      float dx = (t.x - t.sx) / STICK_R, dy = -(t.y - t.sy) / STICK_R, m = std::sqrt(dx * dx + dy * dy);
      if (m > 0.12f) {
        float k = std::min(1.f, (m - 0.12f) / 0.88f) / m;
        in.mx += dx * k; in.my += dy * k;
      }
    }
    if (t.role == 2) held = true;
  }
  in.mx = clampf(in.mx, -1.f, 1.f); in.my = clampf(in.my, -1.f, 1.f);
  if (held) { in.fire = true; in.brake = true; }
  in.use = in.use || useEdge; in.swap = in.swap || swapEdge;
  useEdge = swapEdge = false;
  in.camDX += camAccX; in.camDY += camAccY;
  camAccX = camAccY = 0;
}

// ---------------------------------------------------------------- HUD
void Game::drawHUD(Renderer& R) {
  char buf[96];
  if (P.hurtCd > 0.f && P.deadT <= 0.f) R.rect(0, 0, W, H, 0xC01010, 0.16f * std::min(1.f, P.hurtCd * 2.f));
  if (P.deadT > 0.f) {
    R.rect(0, 0, W, H, 0x300000, 0.5f);
    R.text(W / 2, H / 2 - 16, "KNOCKED OUT", 4, 0xFF4848, 1);
  }

  // ---- minimap (north-up, +-120 m around the player)
  const int mS = 96, mX = 8, mY = 8;
  const float range = 120.f, sc = mS / (2.f * range);
  V2 mc = P.pos;
  auto mapX = [&](float x) { return mX + mS * 0.5f + (x - mc.x) * sc; };
  auto mapY = [&](float z) { return mY + mS * 0.5f + (z - mc.z) * sc; };
  auto rectClip = [&](float x0, float y0, float x1, float y1, uint32_t col, float a) {
    int ax = std::max(mX, (int)std::floor(x0)), ay = std::max(mY, (int)std::floor(y0));
    int bx = std::min(mX + mS, (int)std::ceil(x1)), by = std::min(mY + mS, (int)std::ceil(y1));
    if (ax < bx && ay < by) R.rect(ax, ay, bx - ax, by - ay, col, a);
  };
  R.rect(mX, mY, mS, mS, 0x0d131b, 0.62f);
  for (int k = 0; k < NR; k++) {
    float c = roadC(k);
    rectClip(mapX(-BOUND), mapY(c - HALFR), mapX(BOUND), mapY(c + HALFR), 0x5b6573, 0.95f);
    rectClip(mapX(c - HALFR), mapY(-BOUND), mapX(c + HALFR), mapY(BOUND), 0x5b6573, 0.95f);
  }
  for (const Pickup& k : picks)
    if (k.alive) { float x = mapX(k.pos.x), y = mapY(k.pos.z); if (x > mX && x < mX + mS && y > mY && y < mY + mS) R.rect((int)x - 1, (int)y - 1, 2, 2, 0x56ff7a, 0.95f); }
  for (const Vehicle& v : cars)
    if (v.st == VS_AI || v.st == VS_PARKED) { float x = mapX(v.pos.x), y = mapY(v.pos.z); if (x > mX && x < mX + mS && y > mY && y < mY + mS) R.rect((int)x, (int)y, 1, 1, v.type == 3 ? 0x40d8ff : 0xc8ccd4, 0.8f); }
  auto marker = [&](V2 w, uint32_t col) {
    float x = mapX(w.x), y = mapY(w.z), cx = mX + mS * 0.5f, cy = mY + mS * 0.5f;
    float dx = x - cx, dy = y - cy, lim = mS * 0.5f - 5.f;
    float m = std::max(std::fabs(dx), std::fabs(dy));
    if (m > lim) { dx *= lim / m; dy *= lim / m; }
    R.disc(cx + dx, cy + dy, 3.2f, 0x000000, 0.7f);
    R.disc(cx + dx, cy + dy, 2.3f, col, 1.f);
  };
  marker(world.garage, 0xFFD21E);
  marker(world.gunShop, 0xFF4A26);
  {
    float cx = mX + mS * 0.5f, cy = mY + mS * 0.5f, hx = std::sin(P.yaw), hz = std::cos(P.yaw);
    R.tri2(cx + hx * 6, cy + hz * 6, cx - hx * 3 - hz * 3.5f, cy - hz * 3 + hx * 3.5f, cx - hx * 3 + hz * 3.5f, cy - hz * 3 - hx * 3.5f, 0xFFFFFF, 1.f);
  }
  R.rect(mX - 1, mY - 1, mS + 2, 1, 0xFFFFFF, 0.5f); R.rect(mX - 1, mY + mS, mS + 2, 1, 0xFFFFFF, 0.5f);
  R.rect(mX - 1, mY, 1, mS, 0xFFFFFF, 0.5f); R.rect(mX + mS, mY, 1, mS, 0xFFFFFF, 0.5f);
  int hh = (int)hour, mm = (int)((hour - hh) * 60.f);
  std::snprintf(buf, sizeof buf, "%02d:%02d", hh, mm);
  R.text(mX, mY + mS + 5, buf, 2, 0xC4D0E2);

  // ---- money / weapon
  R.text(W - 10, 10, fmtMoney(P.money).c_str(), 3, 0x7CFF8A, 2);
  if (P.car < 0) {
    const Weapon& wp = WEAP[P.weapon];
    if (wp.melee) std::snprintf(buf, sizeof buf, "%s", wp.name);
    else std::snprintf(buf, sizeof buf, "%s  %d", wp.name, P.ammo[P.weapon]);
    R.text(W - 10, 38, buf, 2, 0xFFFFFF, 2);
  } else {
    R.text(W - 10, 38, CARS[cars[P.car].type].name, 2, 0xFFD84A, 2);
  }

  // ---- health bars
  R.rect(10, H - 16, 86, 8, 0x000000, 0.55f);
  R.rect(11, H - 15, (int)(84.f * clampf(P.hp / 100.f, 0.f, 1.f)), 6, 0xE04848);
  if (P.car >= 0) {
    const Vehicle& v = cars[P.car];
    R.rect(10, H - 27, 86, 8, 0x000000, 0.55f);
    R.rect(11, H - 26, (int)(84.f * clampf(v.hp / CARS[v.type].hp, 0.f, 1.f)), 6, 0xF0A030);
    float sp = len(v.vel) * 3.6f;
    std::snprintf(buf, sizeof buf, "%d", (int)(sp + 0.5f));
    R.text(W / 2, H - 42, buf, 4, 0xFFFFFF, 1);
    R.text(W / 2, H - 14, "KM/H", 1, 0xB8C4D6, 1);
  }

  // ---- prompts, notifications, world labels
  if (!prompt.empty() && P.deadT <= 0.f) R.text(W / 2, H - 78, prompt.c_str(), 2, 0xFFE27A, 1);
  for (size_t i = 0; i < notes.size(); i++) R.text(W / 2, 12 + (int)i * 16, notes[i].text.c_str(), 2, notes[i].col, 1);
  auto label = [&](V2 p, const char* name, uint32_t col) {
    float sx, sy, z;
    if (!R.project({p.x, 9.f, p.z}, sx, sy, z)) return;
    float d = len(p - P.pos);
    if (sx < 10 || sx > W - 10 || sy < 10 || sy > H - 20 || d < 12.f) return;
    std::snprintf(buf, sizeof buf, "%dM", (int)d);
    R.text((int)sx, (int)sy, name, 2, col, 1);
    R.text((int)sx, (int)sy + 15, buf, 1, 0xFFFFFF, 1);
  };
  label(world.garage, "GARAGE", 0xFFD21E);
  label(world.gunShop, "GUN SHOP", 0xFF6A40);
  if (time < 14.f && P.deadT <= 0.f) {
    R.text(W / 2, (int)(H * 0.58f), "LEFT STICK: MOVE    DRAG: LOOK    USE: ENTER CAR", 1, 0xE6EBF2, 1);
    R.text(W / 2, (int)(H * 0.58f) + 12, "DRIVE THE CYAN SPECTER - IT IS FAST", 1, 0x7FE0FF, 1);
  }
}

void Game::drawTouchUI(Renderer& R) {
  float bx = 76.f, by = H - 84.f, kx = bx, ky = by;
  bool active = false, fireHeld = false, useHeld = false, swapHeld = false;
  for (const Touch& t : touches) {
    if (t.id < 0) continue;
    if (t.role == 1) {
      active = true; bx = t.sx; by = t.sy;
      float dx = t.x - t.sx, dy = t.y - t.sy, m = std::sqrt(dx * dx + dy * dy);
      if (m > STICK_R) { dx *= STICK_R / m; dy *= STICK_R / m; }
      kx = bx + dx; ky = by + dy;
    }
    if (t.role == 2) fireHeld = true;
    if (t.role == 3) useHeld = true;
    if (t.role == 4) swapHeld = true;
  }
  R.ring(bx, by, STICK_R - 2, 2, 0xFFFFFF, active ? 0.55f : 0.25f);
  R.disc(kx, ky, 16, 0xFFFFFF, active ? 0.5f : 0.22f);
  auto button = [&](Btn b, const char* label, int scale, bool pressed, uint32_t col, bool hot) {
    R.disc(b.x, b.y, b.r, col, pressed ? 0.55f : (hot ? 0.38f : 0.22f));
    R.ring(b.x, b.y, b.r - 1, 2, hot ? 0xFFE27A : 0xFFFFFF, pressed ? 0.9f : 0.5f);
    R.text((int)b.x, (int)b.y - 3 * scale, label, scale, 0xFFFFFF, 1);
  };
  if (P.car >= 0) button(btnFire(W, H), "DRIFT", 1, fireHeld, 0x3070D0, false);
  else button(btnFire(W, H), WEAP[P.weapon].melee ? "PUNCH" : "FIRE", 2, fireHeld, 0xD03030, false);
  const char* ul = "USE";
  if (P.car >= 0) ul = promptKind == 2 ? "SELL" : "EXIT";
  else if (promptKind == 3) ul = "BUY";
  else if (promptKind == 1) ul = prompt.compare(0, 7, "CARJACK") == 0 ? "JACK" : "ENTER";
  button(btnUse(W, H), ul, 1, useHeld, 0x30A060, promptKind != 0);
  if (P.car < 0) button(btnSwap(W, H), "GUN", 1, swapHeld, 0x808090, false);
}


#if defined(__ANDROID__) && __has_include(<android_native_app_glue.h>)
// ==================== EMBEDDED SOURCE: platform_android.cpp ====================
// NEON CITY - Android launcher (NativeActivity, no Java code, no GL: the software-rendered frame is blitted
// straight into the window buffer). Touch input is forwarded to Game::touch().
#include <android/input.h>
#include <android/log.h>
#include <android/looper.h>
#include <android/native_activity.h>
#include <android/native_window.h>
#include <android_native_app_glue.h>
#include <time.h>

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "NeonCity", __VA_ARGS__)

namespace {
struct App {
  android_app* app = nullptr;
  Game game;
  Renderer R;
  bool haveWindow = false, resumed = false, inited = false;
  int rw = 640, rh = 360, sw = 1, sh = 1;
  bool active() const { return haveWindow && resumed; }
};

double nowSec() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void setupWindow(App& a) {
  ANativeWindow* win = a.app->window;
  a.sw = std::max(1, ANativeWindow_getWidth(win));      // physical size; touch coordinates use the same space
  a.sh = std::max(1, ANativeWindow_getHeight(win));
  internalSize(a.sw, a.sh, a.rw, a.rh);
  ANativeWindow_setBuffersGeometry(win, a.rw, a.rh, WINDOW_FORMAT_RGBA_8888);   // the compositor scales it to the screen
  a.R.init(a.rw, a.rh);
  if (!a.inited) { a.game.init(a.rw, a.rh, 1); a.game.showTouch = true; a.inited = true; }
  else a.game.resize(a.rw, a.rh);
  a.haveWindow = true;
  LOGI("window %dx%d -> render %dx%d, %d render thread(s)", a.sw, a.sh, a.rw, a.rh, a.R.threads);
}

void onCmd(android_app* app, int32_t cmd) {
  App* a = static_cast<App*>(app->userData);
  switch (cmd) {
    case APP_CMD_INIT_WINDOW: if (app->window) setupWindow(*a); break;
    case APP_CMD_TERM_WINDOW: a->haveWindow = false; break;
    case APP_CMD_RESUME: a->resumed = true; break;
    case APP_CMD_PAUSE: a->resumed = false; break;
    default: break;
  }
}

int32_t onInput(android_app* app, AInputEvent* ev) {
  App* a = static_cast<App*>(app->userData);
  if (AInputEvent_getType(ev) != AINPUT_EVENT_TYPE_MOTION) return 0;
  int32_t act = AMotionEvent_getAction(ev);
  int32_t kind = act & AMOTION_EVENT_ACTION_MASK;
  size_t idx = (size_t)((act & AMOTION_EVENT_ACTION_POINTER_INDEX_MASK) >> AMOTION_EVENT_ACTION_POINTER_INDEX_SHIFT);
  auto fx = [&](size_t i) { return AMotionEvent_getX(ev, i) * (float)a->rw / (float)a->sw; };
  auto fy = [&](size_t i) { return AMotionEvent_getY(ev, i) * (float)a->rh / (float)a->sh; };
  size_t n = AMotionEvent_getPointerCount(ev);
  switch (kind) {
    case AMOTION_EVENT_ACTION_DOWN:
    case AMOTION_EVENT_ACTION_POINTER_DOWN: a->game.touch(AMotionEvent_getPointerId(ev, idx), 0, fx(idx), fy(idx)); break;
    case AMOTION_EVENT_ACTION_MOVE: for (size_t i = 0; i < n; i++) a->game.touch(AMotionEvent_getPointerId(ev, i), 1, fx(i), fy(i)); break;
    case AMOTION_EVENT_ACTION_UP:
    case AMOTION_EVENT_ACTION_POINTER_UP: a->game.touch(AMotionEvent_getPointerId(ev, idx), 2, fx(idx), fy(idx)); break;
    case AMOTION_EVENT_ACTION_CANCEL: for (size_t i = 0; i < n; i++) a->game.touch(AMotionEvent_getPointerId(ev, i), 2, 0, 0); break;
    default: return 0;
  }
  return 1;
}
}  // namespace

void android_main(android_app* app) {
  static App a;                                           // static: lives as long as the process
  a.app = app;
  app->userData = &a;
  app->onAppCmd = onCmd;
  app->onInputEvent = onInput;
  ANativeActivity_setWindowFlags(app->activity, AWINDOW_FLAG_KEEP_SCREEN_ON | AWINDOW_FLAG_FULLSCREEN, 0);
  double last = nowSec();
  for (;;) {
    int events = 0;
    android_poll_source* src = nullptr;
    int ident;
    while ((ident = ALooper_pollOnce(a.active() ? 0 : -1, nullptr, &events, reinterpret_cast<void**>(&src))) >= 0) {
      if (src) src->process(app, src);
      if (app->destroyRequested) return;
    }
    (void)ident;
    if (!a.active()) { last = nowSec(); continue; }

    double t = nowSec();
    float dt = (float)std::min(0.05, t - last);
    last = t;
    a.game.update(dt);
    a.game.draw(a.R);

    ANativeWindow_Buffer buf;
    if (ANativeWindow_lock(app->window, &buf, nullptr) == 0) {
      uint32_t* dst = static_cast<uint32_t*>(buf.bits);
      int rows = std::min(buf.height, a.rh), cols = std::min(buf.width, a.rw);
      for (int y = 0; y < rows; y++) std::memcpy(dst + (size_t)y * buf.stride, a.R.px.data() + (size_t)y * a.rw, (size_t)cols * 4);
      ANativeWindow_unlockAndPost(app->window);
    }
  }
}

#endif // Android NativeActivity launcher (requires NDK android_native_app_glue)
