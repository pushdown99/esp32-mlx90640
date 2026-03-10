#include <Wire.h>
#include <SPI.h>
#include <Adafruit_MLX90640.h>
#include <TFT_22_ILI9225.h>
#include <math.h>

// =====================================================
// 핀 설정
// =====================================================
#define I2C_SDA         21
#define I2C_SCL         22
#define I2C_FREQ        800000

#define TFT_RST         26
#define TFT_RS          25
#define TFT_CS          15
#define TFT_LED         0
#define TFT_SCK         14
#define TFT_MOSI        13
#define TFT_BRIGHTNESS  255

// =====================================================
// 해상도
// =====================================================
#define SRC_W           32
#define SRC_H           24

#define TFT_W           220
#define TFT_H           176

// =====================================================
// 근접 상체 촬영용 파라미터
// =====================================================
#define EMA_ALPHA               0.35f

// heatmap은 고정 온도 범위
#define USE_ADAPTIVE_RANGE      0
#define TEMP_MIN_FIXED          25.0f
#define TEMP_MAX_FIXED          35.0f

#define MLX_REFRESH_RATE        MLX90640_8_HZ
#define KEEP_ASPECT_RATIO       1

// 검출은 절대온도 threshold
#define USE_FIXED_DETECT_TEMP   1
#define DETECT_TEMP_FIXED       28.5f

// fallback용
#define THRESH_RATIO            0.55f

// blob 조건
#define MIN_BLOB_PIXELS         4
#define USE_8_CONNECTED         1

// morphology
#define ENABLE_MORPHOLOGY       1
#define MORPH_KERNEL_RADIUS     1
#define DO_OPENING              0
#define DO_CLOSING              1

// 열원 기준 torso 확장
#define ENABLE_HOT_TORSO_EXPAND     1
#define HOT_EXPAND_TEMP_MARGIN      1.8f   // hottest 보다 이만큼 낮은 온도까지 허용
#define HOT_EXPAND_MIN_TEMP         27.0f  // torso 후보 최소 온도
#define HOT_EXPAND_DOWN_CELLS       10     // 얼굴 아래로 최대 확장 셀 수
#define HOT_EXPAND_SIDE_CELLS       5      // 좌우 최대 확장 셀 수

#define LOOP_DELAY_MS           5

// =====================================================
// 객체
// =====================================================
Adafruit_MLX90640 mlx;
SPIClass hspi(HSPI);
TFT_22_ILI9225 tft(TFT_RST, TFT_RS, TFT_CS, TFT_LED, TFT_BRIGHTNESS);

// =====================================================
// 버퍼
// =====================================================
static float mlxRaw[SRC_W * SRC_H];
static float smoothFrame[SRC_H][SRC_W];
static uint16_t lineBuf[TFT_W];

static uint8_t maskFrame[SRC_H][SRC_W];
static uint8_t morphTemp[SRC_H][SRC_W];
static uint8_t visited[SRC_H][SRC_W];

static int queueX[SRC_W * SRC_H];
static int queueY[SRC_W * SRC_H];

// =====================================================
// 구조체
// =====================================================
typedef struct {
  bool valid;
  int count;

  int minX;
  int minY;
  int maxX;
  int maxY;

  float cx;
  float cy;

  int hotX;
  int hotY;
  float hotTemp;

  float thresholdTemp;
  float tMin;
  float tMax;
} DetectionResult;

typedef struct {
  bool valid;
  int boxX0;
  int boxY0;
  int boxX1;
  int boxY1;
  int cx;
  int cy;
  int hx;
  int hy;
  uint16_t boxColor;
} OverlayScreen;

typedef enum {
  BODY_UNKNOWN = 0,
  BODY_UPPER,
  BODY_FULL
} BodyType;

typedef struct {
  BodyType type;
  float aspect;
  float bottomRatio;
  float hotTopRatio;
  int bw;
  int bh;
} BodyClassResult;

// =====================================================
// 유틸
// =====================================================
static inline float clampf(float v, float lo, float hi) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}

static inline int clampi(int v, int lo, int hi) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}

static inline uint16_t rgb565(uint8_t r, uint8_t g, uint8_t b) {
  return ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3);
}

uint16_t colorMap565(uint8_t v) {
  uint8_t r = 0, g = 0, b = 0;

  if (v < 64) {
    r = 0; g = v * 4; b = 255;
  } else if (v < 128) {
    r = 0; g = 255; b = 255 - (v - 64) * 4;
  } else if (v < 192) {
    r = (v - 128) * 4; g = 255; b = 0;
  } else {
    r = 255; g = 255 - (v - 192) * 4; b = 0;
  }

  return rgb565(r, g, b);
}

const char* bodyTypeToString(BodyType t) {
  switch (t) {
    case BODY_UPPER: return "UPPER";
    case BODY_FULL:  return "FULL";
    default:         return "UNKNOWN";
  }
}

// =====================================================
// MLX90640 읽기
// =====================================================
bool readMLXFrame(float out[SRC_H][SRC_W]) {
  if (mlx.getFrame(mlxRaw) != 0) return false;

  for (int y = 0; y < SRC_H; y++) {
    for (int x = 0; x < SRC_W; x++) {
      out[y][x] = mlxRaw[y * SRC_W + x];
    }
  }
  return true;
}

// =====================================================
// EMA smoothing
// =====================================================
void applyEMA(float frame[SRC_H][SRC_W], bool firstFrame) {
  for (int y = 0; y < SRC_H; y++) {
    for (int x = 0; x < SRC_W; x++) {
      if (firstFrame) {
        smoothFrame[y][x] = frame[y][x];
      } else {
        smoothFrame[y][x] =
          EMA_ALPHA * frame[y][x] +
          (1.0f - EMA_ALPHA) * smoothFrame[y][x];
      }
    }
  }
}

// =====================================================
// adaptive range
// =====================================================
void estimateAdaptiveRange(float &tMin, float &tMax) {
  float sum = 0.0f, sum2 = 0.0f;
  const int n = SRC_W * SRC_H;

  for (int y = 0; y < SRC_H; y++) {
    for (int x = 0; x < SRC_W; x++) {
      float v = smoothFrame[y][x];
      sum += v;
      sum2 += v * v;
    }
  }

  float mean = sum / n;
  float var = (sum2 / n) - (mean * mean);
  if (var < 0.0f) var = 0.0f;
  float stdv = sqrtf(var);

  tMin = mean - 1.2f * stdv;
  tMax = mean + 1.8f * stdv;

  if ((tMax - tMin) < 3.0f) {
    tMin = mean - 1.5f;
    tMax = mean + 1.5f;
  }
}

// =====================================================
// 표시 창 계산
// =====================================================
void calcDisplayWindow(int &drawX, int &drawY, int &drawW, int &drawH) {
#if KEEP_ASPECT_RATIO
  float srcAspect = (float)SRC_W / (float)SRC_H;
  float tftAspect = (float)TFT_W / (float)TFT_H;

  if (tftAspect > srcAspect) {
    drawH = TFT_H;
    drawW = (int)(drawH * srcAspect + 0.5f);
  } else {
    drawW = TFT_W;
    drawH = (int)(drawW / srcAspect + 0.5f);
  }

  drawX = (TFT_W - drawW) / 2;
  drawY = (TFT_H - drawH) / 2;
#else
  drawX = 0;
  drawY = 0;
  drawW = TFT_W;
  drawH = TFT_H;
#endif
}

void clearMargins(int drawX, int drawY, int drawW, int drawH) {
  if (drawY > 0) tft.fillRectangle(0, 0, TFT_W - 1, drawY - 1, COLOR_BLACK);
  if (drawY + drawH < TFT_H) tft.fillRectangle(0, drawY + drawH, TFT_W - 1, TFT_H - 1, COLOR_BLACK);
  if (drawX > 0) tft.fillRectangle(0, drawY, drawX - 1, drawY + drawH - 1, COLOR_BLACK);
  if (drawX + drawW < TFT_W) tft.fillRectangle(drawX + drawW, drawY, TFT_W - 1, drawY + drawH - 1, COLOR_BLACK);
}

// =====================================================
// threshold mask 생성
// =====================================================
float buildThresholdMask(float tMin, float tMax) {
  float thresholdTemp;
#if USE_FIXED_DETECT_TEMP
  thresholdTemp = DETECT_TEMP_FIXED;
#else
  thresholdTemp = tMin + THRESH_RATIO * (tMax - tMin);
#endif

  for (int y = 0; y < SRC_H; y++) {
    for (int x = 0; x < SRC_W; x++) {
      maskFrame[y][x] = (smoothFrame[y][x] >= thresholdTemp) ? 1 : 0;
      visited[y][x] = 0;
    }
  }

  return thresholdTemp;
}

// =====================================================
// morphology
// =====================================================
void erodeMask(const uint8_t src[SRC_H][SRC_W], uint8_t dst[SRC_H][SRC_W], int radius) {
  for (int y = 0; y < SRC_H; y++) {
    for (int x = 0; x < SRC_W; x++) {
      uint8_t keep = 1;
      for (int ky = -radius; ky <= radius && keep; ky++) {
        for (int kx = -radius; kx <= radius; kx++) {
          int nx = x + kx, ny = y + ky;
          if (nx < 0 || nx >= SRC_W || ny < 0 || ny >= SRC_H || src[ny][nx] == 0) {
            keep = 0;
            break;
          }
        }
      }
      dst[y][x] = keep;
    }
  }
}

void dilateMask(const uint8_t src[SRC_H][SRC_W], uint8_t dst[SRC_H][SRC_W], int radius) {
  for (int y = 0; y < SRC_H; y++) {
    for (int x = 0; x < SRC_W; x++) {
      uint8_t on = 0;
      for (int ky = -radius; ky <= radius && !on; ky++) {
        for (int kx = -radius; kx <= radius; kx++) {
          int nx = x + kx, ny = y + ky;
          if (nx < 0 || nx >= SRC_W || ny < 0 || ny >= SRC_H) continue;
          if (src[ny][nx]) {
            on = 1;
            break;
          }
        }
      }
      dst[y][x] = on;
    }
  }
}

void applyMorphology() {
#if ENABLE_MORPHOLOGY
  #if DO_OPENING
    erodeMask(maskFrame, morphTemp, MORPH_KERNEL_RADIUS);
    dilateMask(morphTemp, maskFrame, MORPH_KERNEL_RADIUS);
  #endif

  #if DO_CLOSING
    dilateMask(maskFrame, morphTemp, MORPH_KERNEL_RADIUS);
    erodeMask(morphTemp, maskFrame, MORPH_KERNEL_RADIUS);
    dilateMask(maskFrame, morphTemp, MORPH_KERNEL_RADIUS);
    erodeMask(morphTemp, maskFrame, MORPH_KERNEL_RADIUS);
  #endif
#endif
}

// =====================================================
// flood fill
// =====================================================
DetectionResult floodFillBlob(int startX, int startY, float thresholdTemp, float tMin, float tMax) {
  DetectionResult r;
  r.valid = false;
  r.count = 0;
  r.minX = SRC_W - 1;
  r.minY = SRC_H - 1;
  r.maxX = 0;
  r.maxY = 0;
  r.cx = 0.0f;
  r.cy = 0.0f;
  r.hotX = startX;
  r.hotY = startY;
  r.hotTemp = -1000.0f;
  r.thresholdTemp = thresholdTemp;
  r.tMin = tMin;
  r.tMax = tMax;

  long sumX = 0, sumY = 0;

  int head = 0, tail = 0;
  queueX[tail] = startX;
  queueY[tail] = startY;
  tail++;
  visited[startY][startX] = 1;

  while (head < tail) {
    int x = queueX[head];
    int y = queueY[head];
    head++;

    float temp = smoothFrame[y][x];
    r.count++;
    sumX += x;
    sumY += y;

    if (x < r.minX) r.minX = x;
    if (y < r.minY) r.minY = y;
    if (x > r.maxX) r.maxX = x;
    if (y > r.maxY) r.maxY = y;

    if (temp > r.hotTemp) {
      r.hotTemp = temp;
      r.hotX = x;
      r.hotY = y;
    }

#if USE_8_CONNECTED
    const int dxs[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    const int dys[8] = {-1,-1,-1,  0, 0,  1, 1, 1};
    const int nNbr = 8;
#else
    const int dxs[4] = {-1, 1, 0, 0};
    const int dys[4] = { 0, 0,-1, 1};
    const int nNbr = 4;
#endif

    for (int i = 0; i < nNbr; i++) {
      int nx = x + dxs[i];
      int ny = y + dys[i];
      if (nx < 0 || nx >= SRC_W || ny < 0 || ny >= SRC_H) continue;
      if (visited[ny][nx]) continue;
      if (!maskFrame[ny][nx]) continue;

      visited[ny][nx] = 1;
      queueX[tail] = nx;
      queueY[tail] = ny;
      tail++;
    }
  }

  if (r.count >= MIN_BLOB_PIXELS) {
    r.valid = true;
    r.cx = (float)sumX / (float)r.count;
    r.cy = (float)sumY / (float)r.count;
  }

  return r;
}

// =====================================================
// 가장 큰 blob 선택
// =====================================================
DetectionResult detectLargestBlob(float tMin, float tMax) {
  float thresholdTemp = buildThresholdMask(tMin, tMax);
  applyMorphology();

  for (int y = 0; y < SRC_H; y++) {
    for (int x = 0; x < SRC_W; x++) {
      visited[y][x] = 0;
    }
  }

  DetectionResult best;
  best.valid = false;
  best.count = 0;
  best.minX = 0;
  best.minY = 0;
  best.maxX = 0;
  best.maxY = 0;
  best.cx = 0;
  best.cy = 0;
  best.hotX = 0;
  best.hotY = 0;
  best.hotTemp = -1000.0f;
  best.thresholdTemp = thresholdTemp;
  best.tMin = tMin;
  best.tMax = tMax;

  for (int y = 0; y < SRC_H; y++) {
    for (int x = 0; x < SRC_W; x++) {
      if (!maskFrame[y][x]) continue;
      if (visited[y][x]) continue;

      DetectionResult blob = floodFillBlob(x, y, thresholdTemp, tMin, tMax);
      if (blob.valid && blob.count > best.count) best = blob;
    }
  }

  return best;
}

// =====================================================
// 얼굴만 잡힐 때 상체 박스로 1차 확장
// =====================================================
void expandBoxForUpperBody(DetectionResult &r) {
  if (!r.valid) return;

  int bw = r.maxX - r.minX + 1;
  int bh = r.maxY - r.minY + 1;
  if (bw <= 0) bw = 1;

  if (bh <= 6 || ((float)bh / (float)bw < 0.9f)) {
    int extraDown = bh * 2 + 4;
    int extraSide = bw / 2 + 1;
    int extraUp   = 1;

    r.minX = clampi(r.minX - extraSide, 0, SRC_W - 1);
    r.maxX = clampi(r.maxX + extraSide, 0, SRC_W - 1);
    r.minY = clampi(r.minY - extraUp,   0, SRC_H - 1);
    r.maxY = clampi(r.maxY + extraDown, 0, SRC_H - 1);

    r.cx = 0.5f * (r.minX + r.maxX);
    r.cy = 0.5f * (r.minY + r.maxY);
  }
}

// =====================================================
// 열원(얼굴) 기준 torso 추가 확장
// hottest point 아래쪽 warm region을 찾아 박스 확장
// =====================================================
void expandTorsoFromHotRegion(DetectionResult &r) {
#if ENABLE_HOT_TORSO_EXPAND
  if (!r.valid) return;

  float torsoTempThr = r.hotTemp - HOT_EXPAND_TEMP_MARGIN;
  if (torsoTempThr < HOT_EXPAND_MIN_TEMP) {
    torsoTempThr = HOT_EXPAND_MIN_TEMP;
  }

  int xStart = clampi(r.hotX - HOT_EXPAND_SIDE_CELLS, 0, SRC_W - 1);
  int xEnd   = clampi(r.hotX + HOT_EXPAND_SIDE_CELLS, 0, SRC_W - 1);
  int yStart = clampi(r.hotY, 0, SRC_H - 1);
  int yEnd   = clampi(r.hotY + HOT_EXPAND_DOWN_CELLS, 0, SRC_H - 1);

  int addMinX = SRC_W - 1;
  int addMinY = SRC_H - 1;
  int addMaxX = 0;
  int addMaxY = 0;
  int addCount = 0;
  long sumX = 0;
  long sumY = 0;

  for (int y = yStart; y <= yEnd; y++) {
    for (int x = xStart; x <= xEnd; x++) {
      float temp = smoothFrame[y][x];
      if (temp >= torsoTempThr) {
        addCount++;
        sumX += x;
        sumY += y;
        if (x < addMinX) addMinX = x;
        if (y < addMinY) addMinY = y;
        if (x > addMaxX) addMaxX = x;
        if (y > addMaxY) addMaxY = y;
      }
    }
  }

  if (addCount >= 4) {
    if (addMinX < r.minX) r.minX = addMinX;
    if (addMinY < r.minY) r.minY = addMinY;
    if (addMaxX > r.maxX) r.maxX = addMaxX;
    if (addMaxY > r.maxY) r.maxY = addMaxY;

    // 중심은 기존 box 중심보다 torso 쪽을 조금 반영
    float torsoCx = (float)sumX / (float)addCount;
    float torsoCy = (float)sumY / (float)addCount;
    r.cx = 0.35f * r.cx + 0.65f * torsoCx;
    r.cy = 0.30f * r.cy + 0.70f * torsoCy;
  }
#endif
}

// =====================================================
// 상체 / 전신 분류
// =====================================================
BodyClassResult classifyBodyType(const DetectionResult &r) {
  BodyClassResult c;
  c.type = BODY_UNKNOWN;
  c.aspect = 0.0f;
  c.bottomRatio = 0.0f;
  c.hotTopRatio = 0.0f;
  c.bw = 0;
  c.bh = 0;

  if (!r.valid) return c;

  c.bw = r.maxX - r.minX + 1;
  c.bh = r.maxY - r.minY + 1;
  if (c.bw <= 0) c.bw = 1;
  if (c.bh <= 0) c.bh = 1;

  c.aspect = (float)c.bh / (float)c.bw;
  c.bottomRatio = (float)r.maxY / (float)(SRC_H - 1);
  c.hotTopRatio = (float)r.hotY / (float)(SRC_H - 1);

  // 전신: 세로로 길고, 하단이 충분히 내려옴
  if (c.aspect >= 1.20f && c.bottomRatio >= 0.88f) {
    c.type = BODY_FULL;
    return c;
  }

  // 상체: 하단이 충분히 안 내려오거나 세로/가로 비가 낮음
  if (c.aspect < 1.20f || c.bottomRatio < 0.84f) {
    c.type = BODY_UPPER;
    return c;
  }

  // 열원이 상단에 몰리고 하단이 끝까지 안 가면 상체
  if (c.hotTopRatio < 0.45f && c.bottomRatio < 0.90f) {
    c.type = BODY_UPPER;
    return c;
  }

  c.type = BODY_UNKNOWN;
  return c;
}

// =====================================================
// 좌표 변환
// =====================================================
int mapSrcXToScreen(float srcX) {
  int drawX, drawY, drawW, drawH;
  calcDisplayWindow(drawX, drawY, drawW, drawH);
  return drawX + (int)((srcX / (SRC_W - 1)) * (drawW - 1) + 0.5f);
}

int mapSrcYToScreen(float srcY) {
  int drawX, drawY, drawW, drawH;
  calcDisplayWindow(drawX, drawY, drawW, drawH);
  return drawY + (int)((srcY / (SRC_H - 1)) * (drawH - 1) + 0.5f);
}

OverlayScreen makeOverlayScreen(const DetectionResult &r, const BodyClassResult &cls) {
  OverlayScreen o;
  o.valid = false;
  o.boxX0 = o.boxY0 = o.boxX1 = o.boxY1 = 0;
  o.cx = o.cy = o.hx = o.hy = 0;
  o.boxColor = COLOR_WHITE;

  if (!r.valid) return o;

  o.valid = true;
  o.boxX0 = mapSrcXToScreen(r.minX);
  o.boxY0 = mapSrcYToScreen(r.minY);
  o.boxX1 = mapSrcXToScreen(r.maxX);
  o.boxY1 = mapSrcYToScreen(r.maxY);
  o.cx    = mapSrcXToScreen(r.cx);
  o.cy    = mapSrcYToScreen(r.cy);
  o.hx    = mapSrcXToScreen(r.hotX);
  o.hy    = mapSrcYToScreen(r.hotY);

  if (cls.type == BODY_UPPER) o.boxColor = COLOR_YELLOW;
  else if (cls.type == BODY_FULL) o.boxColor = COLOR_GREEN;
  else o.boxColor = COLOR_WHITE;

  return o;
}

// =====================================================
// lineBuf 위에 overlay 직접 그리기
// =====================================================
void applyOverlayToLine(uint16_t *buf, int screenY, int width, const OverlayScreen &o) {
  if (!o.valid) return;

  // 1) bounding box
  if (screenY == o.boxY0 || screenY == o.boxY1) {
    int x0 = clampi(o.boxX0, 0, width - 1);
    int x1 = clampi(o.boxX1, 0, width - 1);
    if (x1 < x0) {
      int t = x0; x0 = x1; x1 = t;
    }
    for (int x = x0; x <= x1; x++) {
      buf[x] = o.boxColor;
    }
  }

  if (screenY >= o.boxY0 && screenY <= o.boxY1) {
    if (o.boxX0 >= 0 && o.boxX0 < width) buf[o.boxX0] = o.boxColor;
    if (o.boxX1 >= 0 && o.boxX1 < width) buf[o.boxX1] = o.boxColor;
    if (o.boxX0 + 1 >= 0 && o.boxX0 + 1 < width) buf[o.boxX0 + 1] = o.boxColor;
    if (o.boxX1 - 1 >= 0 && o.boxX1 - 1 < width) buf[o.boxX1 - 1] = o.boxColor;
  }

  // 2) center
  if (screenY == o.cy || screenY == o.cy - 1 || screenY == o.cy + 1) {
    for (int dx = -6; dx <= 6; dx++) {
      int x = o.cx + dx;
      if (x >= 0 && x < width) buf[x] = COLOR_GREEN;
    }
  }

  if (screenY >= o.cy - 6 && screenY <= o.cy + 6) {
    if (o.cx >= 0 && o.cx < width) buf[o.cx] = COLOR_GREEN;
    if (o.cx - 1 >= 0 && o.cx - 1 < width) buf[o.cx - 1] = COLOR_GREEN;
    if (o.cx + 1 >= 0 && o.cx + 1 < width) buf[o.cx + 1] = COLOR_GREEN;
  }

  // 3) hot point
  for (int dx = -5; dx <= 5; dx++) {
    int x1 = o.hx + dx;
    int y1 = o.hy + dx;
    int x2 = o.hx + dx;
    int y2 = o.hy - dx;

    if (screenY == y1 && x1 >= 0 && x1 < width) buf[x1] = COLOR_RED;
    if (screenY == y2 && x2 >= 0 && x2 < width) buf[x2] = COLOR_RED;
  }

  if (screenY == o.hy || screenY == o.hy - 1 || screenY == o.hy + 1) {
    for (int dx = -4; dx <= 4; dx++) {
      int x = o.hx + dx;
      if (x >= 0 && x < width) buf[x] = COLOR_RED;
    }
  }

  if (screenY >= o.hy - 4 && screenY <= o.hy + 4) {
    if (o.hx >= 0 && o.hx < width) buf[o.hx] = COLOR_RED;
  }
}

// =====================================================
// heatmap + overlay 스트리밍 출력
// =====================================================
void drawHeatmapStreamingWithOverlay(float tMin, float tMax, const DetectionResult &det, const BodyClassResult &cls) {
  int drawX, drawY, drawW, drawH;
  calcDisplayWindow(drawX, drawY, drawW, drawH);
  clearMargins(drawX, drawY, drawW, drawH);

  OverlayScreen ov = makeOverlayScreen(det, cls);

  for (int dy = 0; dy < drawH; dy++) {
    int screenY = drawY + dy;

    float sy = ((float)dy * (SRC_H - 1)) / (drawH - 1);
    int y0 = (int)sy;
    int y1 = clampi(y0 + 1, 0, SRC_H - 1);
    float fy = sy - y0;

    for (int dx = 0; dx < drawW; dx++) {
      float sx = ((float)dx * (SRC_W - 1)) / (drawW - 1);
      int x0 = (int)sx;
      int x1 = clampi(x0 + 1, 0, SRC_W - 1);
      float fx = sx - x0;

      float p00 = smoothFrame[y0][x0];
      float p10 = smoothFrame[y0][x1];
      float p01 = smoothFrame[y1][x0];
      float p11 = smoothFrame[y1][x1];

      float top = p00 + (p10 - p00) * fx;
      float bot = p01 + (p11 - p01) * fx;
      float temp = top + (bot - top) * fy;

      float n = (temp - tMin) / (tMax - tMin);
      n = clampf(n, 0.0f, 1.0f);
      lineBuf[dx] = colorMap565((uint8_t)(n * 255.0f));
    }

    if (ov.valid) {
      OverlayScreen local = ov;
      local.boxX0 -= drawX;
      local.boxX1 -= drawX;
      local.cx    -= drawX;
      local.hx    -= drawX;
      applyOverlayToLine(lineBuf, screenY, drawW, local);
    }

    tft.drawBitmap(drawX, screenY, lineBuf, drawW, 1);
    if ((dy & 7) == 0) yield();
  }
}

// =====================================================
// 디버그
// =====================================================
void printDetection(const DetectionResult &r) {
  Serial.print("DetectThr=");
  Serial.print(r.thresholdTemp, 2);

  if (r.valid) {
    Serial.print(" | Count=");
    Serial.print(r.count);

    Serial.print(" | Box=(");
    Serial.print(r.minX); Serial.print(",");
    Serial.print(r.minY); Serial.print(")-(");
    Serial.print(r.maxX); Serial.print(",");
    Serial.print(r.maxY); Serial.print(")");

    Serial.print(" | Center=(");
    Serial.print(r.cx, 1); Serial.print(",");
    Serial.print(r.cy, 1); Serial.print(")");

    Serial.print(" | Hot=(");
    Serial.print(r.hotX); Serial.print(",");
    Serial.print(r.hotY); Serial.print(")=");
    Serial.print(r.hotTemp, 1);
  } else {
    Serial.print(" | No valid blob");
  }
  Serial.println();
}

void printBodyClass(const BodyClassResult &c) {
  Serial.print("BodyType=");
  Serial.print(bodyTypeToString(c.type));
  Serial.print(" | bw=");
  Serial.print(c.bw);
  Serial.print(" bh=");
  Serial.print(c.bh);
  Serial.print(" aspect=");
  Serial.print(c.aspect, 2);
  Serial.print(" bottom=");
  Serial.print(c.bottomRatio, 2);
  Serial.print(" hotTop=");
  Serial.println(c.hotTopRatio, 2);
}

// =====================================================
// setup
// =====================================================
void setup() {
  Serial.begin(115200);
  delay(300);

  pinMode(TFT_LED, OUTPUT);
  digitalWrite(TFT_LED, HIGH);

  Wire.begin(I2C_SDA, I2C_SCL);
  Wire.setClock(I2C_FREQ);

  if (!mlx.begin(0x33, &Wire)) {
    Serial.println("MLX90640 init failed");
    while (1) delay(1000);
  }

  mlx.setMode(MLX90640_CHESS);
  mlx.setResolution(MLX90640_ADC_18BIT);
  mlx.setRefreshRate(MLX_REFRESH_RATE);

  hspi.begin(TFT_SCK, -1, TFT_MOSI, TFT_CS);

  tft.begin(hspi);
  tft.setDisplay(true);
  tft.setOrientation(3);
  tft.clear();

  Serial.println("MLX90640 upper/full body classified demo start");
}

// =====================================================
// loop
// =====================================================
void loop() {
  static bool firstFrame = true;
  float tempFrame[SRC_H][SRC_W];

  if (readMLXFrame(tempFrame)) {
    applyEMA(tempFrame, firstFrame);
    firstFrame = false;

    float tMin, tMax;
#if USE_ADAPTIVE_RANGE
    estimateAdaptiveRange(tMin, tMax);
#else
    tMin = TEMP_MIN_FIXED;
    tMax = TEMP_MAX_FIXED;
#endif

    DetectionResult det = detectLargestBlob(tMin, tMax);

    // 1차: 얼굴만 잡히면 상체 box 확장
    expandBoxForUpperBody(det);

    // 2차: hottest point 아래 torso warm region 반영
    expandTorsoFromHotRegion(det);

    // 분류
    BodyClassResult cls = classifyBodyType(det);

    // 출력
    drawHeatmapStreamingWithOverlay(tMin, tMax, det, cls);
    printDetection(det);
    printBodyClass(cls);
  } else {
    Serial.println("MLX90640 frame read error");
  }

  delay(LOOP_DELAY_MS);
}