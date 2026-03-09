#include <Wire.h>
#include <SPI.h>
#include <Adafruit_MLX90640.h>
#include <TFT_22_ILI9225.h>
#include <math.h>

// =====================================================
// 핀 설정
// =====================================================
// MLX90640 I2C
#define I2C_SDA         21
#define I2C_SCL         22
#define I2C_FREQ        800000

// ILI9225 SPI
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
#define DETECT_TEMP_FIXED       29.5f

// fallback용
#define THRESH_RATIO            0.55f

// blob 조건
#define MIN_BLOB_PIXELS         4
#define USE_8_CONNECTED         1

// morphology
#define ENABLE_MORPHOLOGY       1
#define MORPH_KERNEL_RADIUS     1   // 3x3
#define DO_OPENING              0   // 상체가 깎이지 않게 끔
#define DO_CLOSING              1   // 몸통 연결 강화

// 선택 blob 격자 표시
#define SHOW_BLOB_GRID          0

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
static uint8_t selectedBlob[SRC_H][SRC_W];

static int queueX[SRC_W * SRC_H];
static int queueY[SRC_W * SRC_H];

// =====================================================
// 검출 결과 구조체
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

// blue -> cyan -> green -> yellow -> red
uint16_t colorMap565(uint8_t v) {
  uint8_t r = 0, g = 0, b = 0;

  if (v < 64) {
    r = 0;
    g = v * 4;
    b = 255;
  } else if (v < 128) {
    r = 0;
    g = 255;
    b = 255 - (v - 64) * 4;
  } else if (v < 192) {
    r = (v - 128) * 4;
    g = 255;
    b = 0;
  } else {
    r = 255;
    g = 255 - (v - 192) * 4;
    b = 0;
  }

  return rgb565(r, g, b);
}

// =====================================================
// MLX90640 읽기
// =====================================================
bool readMLXFrame(float out[SRC_H][SRC_W]) {
  if (mlx.getFrame(mlxRaw) != 0) {
    return false;
  }

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
  float sum = 0.0f;
  float sum2 = 0.0f;
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
  if (drawY > 0) {
    tft.fillRectangle(0, 0, TFT_W - 1, drawY - 1, COLOR_BLACK);
  }
  if (drawY + drawH < TFT_H) {
    tft.fillRectangle(0, drawY + drawH, TFT_W - 1, TFT_H - 1, COLOR_BLACK);
  }
  if (drawX > 0) {
    tft.fillRectangle(0, drawY, drawX - 1, drawY + drawH - 1, COLOR_BLACK);
  }
  if (drawX + drawW < TFT_W) {
    tft.fillRectangle(drawX + drawW, drawY, TFT_W - 1, drawY + drawH - 1, COLOR_BLACK);
  }
}

// =====================================================
// heatmap 스트리밍 출력
// =====================================================
void drawHeatmapStreaming(float tMin, float tMax) {
  int drawX, drawY, drawW, drawH;
  calcDisplayWindow(drawX, drawY, drawW, drawH);
  clearMargins(drawX, drawY, drawW, drawH);

  for (int dy = 0; dy < drawH; dy++) {
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

    tft.drawBitmap(drawX, drawY + dy, lineBuf, drawW, 1);

    if ((dy & 7) == 0) yield();
  }
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
      selectedBlob[y][x] = 0;
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
          int nx = x + kx;
          int ny = y + ky;

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
          int nx = x + kx;
          int ny = y + ky;

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
  #endif
#endif
}

// =====================================================
// 한 개 blob flood fill
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

  long sumX = 0;
  long sumY = 0;

  int head = 0;
  int tail = 0;

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
      selectedBlob[y][x] = 0;
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

  int bestSeedX = -1;
  int bestSeedY = -1;

  for (int y = 0; y < SRC_H; y++) {
    for (int x = 0; x < SRC_W; x++) {
      if (!maskFrame[y][x]) continue;
      if (visited[y][x]) continue;

      DetectionResult blob = floodFillBlob(x, y, thresholdTemp, tMin, tMax);

      if (blob.valid && blob.count > best.count) {
        best = blob;
        bestSeedX = x;
        bestSeedY = y;
      }
    }
  }

  if (best.valid && bestSeedX >= 0 && bestSeedY >= 0) {
    for (int y = 0; y < SRC_H; y++) {
      for (int x = 0; x < SRC_W; x++) {
        visited[y][x] = 0;
      }
    }

    int head = 0;
    int tail = 0;
    queueX[tail] = bestSeedX;
    queueY[tail] = bestSeedY;
    tail++;

    visited[bestSeedY][bestSeedX] = 1;
    selectedBlob[bestSeedY][bestSeedX] = 1;

    while (head < tail) {
      int x = queueX[head];
      int y = queueY[head];
      head++;

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
        selectedBlob[ny][nx] = 1;
        queueX[tail] = nx;
        queueY[tail] = ny;
        tail++;
      }
    }
  }

  return best;
}

// =====================================================
// 소스 좌표 -> 화면 좌표
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

// =====================================================
// 안전한 박스 그리기
// =====================================================
void drawBoxSafe(int x0, int y0, int x1, int y1, uint16_t color) {
  x0 = clampi(x0, 0, TFT_W - 1);
  x1 = clampi(x1, 0, TFT_W - 1);
  y0 = clampi(y0, 0, TFT_H - 1);
  y1 = clampi(y1, 0, TFT_H - 1);

  if (x1 < x0) {
    int t = x0; x0 = x1; x1 = t;
  }
  if (y1 < y0) {
    int t = y0; y0 = y1; y1 = t;
  }

  tft.drawLine(x0, y0, x1, y0, color);
  tft.drawLine(x1, y0, x1, y1, color);
  tft.drawLine(x1, y1, x0, y1, color);
  tft.drawLine(x0, y1, x0, y0, color);
}

// =====================================================
// overlay
// =====================================================
void drawSelectedBlobOverlay() {
#if SHOW_BLOB_GRID
  int drawX, drawY, drawW, drawH;
  calcDisplayWindow(drawX, drawY, drawW, drawH);

  for (int y = 0; y < SRC_H; y++) {
    for (int x = 0; x < SRC_W; x++) {
      if (selectedBlob[y][x]) {
        int x0 = drawX + (x * drawW) / SRC_W;
        int y0 = drawY + (y * drawH) / SRC_H;
        int x1 = drawX + ((x + 1) * drawW) / SRC_W - 1;
        int y1 = drawY + ((y + 1) * drawH) / SRC_H - 1;
        drawBoxSafe(x0, y0, x1, y1, COLOR_WHITE);
      }
    }

    if ((y & 3) == 0) yield();
  }
#endif
}

void drawDetectionOverlay(const DetectionResult &r) {
  if (!r.valid) return;

  int boxX0 = mapSrcXToScreen(r.minX);
  int boxY0 = mapSrcYToScreen(r.minY);
  int boxX1 = mapSrcXToScreen(r.maxX);
  int boxY1 = mapSrcYToScreen(r.maxY);

  int cx = mapSrcXToScreen(r.cx);
  int cy = mapSrcYToScreen(r.cy);

  int hx = mapSrcXToScreen(r.hotX);
  int hy = mapSrcYToScreen(r.hotY);

  // bounding box 강조
  drawBoxSafe(boxX0, boxY0, boxX1, boxY1, COLOR_YELLOW);
  drawBoxSafe(boxX0 + 1, boxY0 + 1, boxX1 - 1, boxY1 - 1, COLOR_YELLOW);

  // centroid
  tft.drawLine(cx - 6, cy, cx + 6, cy, COLOR_GREEN);
  tft.drawLine(cx, cy - 6, cx, cy + 6, COLOR_GREEN);
  tft.drawCircle(cx, cy, 3, COLOR_GREEN);
  tft.drawCircle(cx, cy, 5, COLOR_GREEN);

  // hottest area
  tft.drawCircle(hx, hy, 3, COLOR_RED);
  tft.drawCircle(hx, hy, 6, COLOR_RED);
  tft.drawCircle(hx, hy, 9, COLOR_RED);
  tft.drawLine(hx - 4, hy - 4, hx + 4, hy + 4, COLOR_RED);
  tft.drawLine(hx - 4, hy + 4, hx + 4, hy - 4, COLOR_RED);
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

// =====================================================
// setup
// =====================================================
void setup() {
  Serial.begin(115200);
  delay(300);

  // 백라이트 수동 ON
  pinMode(TFT_LED, OUTPUT);
  digitalWrite(TFT_LED, HIGH);

  // I2C
  Wire.begin(I2C_SDA, I2C_SCL);
  Wire.setClock(I2C_FREQ);

  // MLX90640
  if (!mlx.begin(0x33, &Wire)) {
    Serial.println("MLX90640 init failed");
    while (1) delay(1000);
  }

  mlx.setMode(MLX90640_CHESS);
  mlx.setResolution(MLX90640_ADC_18BIT);
  mlx.setRefreshRate(MLX_REFRESH_RATE);

  // SPI
  hspi.begin(TFT_SCK, -1, TFT_MOSI, TFT_CS);

  // TFT
  tft.begin(hspi);
  tft.setDisplay(true);
  tft.setOrientation(3);
  tft.clear();

  Serial.println("MLX90640 close upper-body demo start");
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

    // 1) heatmap
    drawHeatmapStreaming(tMin, tMax);

    // 2) detect
    DetectionResult det = detectLargestBlob(tMin, tMax);

    // 3) overlay
    drawSelectedBlobOverlay();
    drawDetectionOverlay(det);

    // 4) serial debug
    printDetection(det);
  } else {
    Serial.println("MLX90640 frame read error");
  }
  drawBoxSafe(20, 20, 120, 120, COLOR_WHITE);

  delay(LOOP_DELAY_MS);
}