let video;
let faceMesh;
let faces = [];
let bodySegmentation;
let segmentation;
let personMask;
let asciiLayer;
let glowLayer;
let classifier;
let classificationLabel = 'loading classifier...';
let candidateLabel = '';
let candidateCount = 0;
let transitionFrom = '';
let transitionTo = '';
let transitionStartedAt = 0;

// Keep computer-vision processing at a stable camera resolution. The final
// canvas is independently created at windowWidth × windowHeight in setup().
const W = 640;
const H = 480;
let fallbackTextAnchor = { x: W / 2, y: H * 0.36, width: W * 0.3 };
const TM_MODEL_URL = 'https://teachablemachine.withgoogle.com/models/uWbtrf3Xp/model.json';
const ASCII_SIZE = 11;
const EDGE_RADIUS = ASCII_SIZE * 0.72;
const GAP_AMOUNT = 0.22;
const NOISE_AMOUNT = 34;
const TEXT_TRANSITION_MS = 800;
const FACE_TEXT_SIZE = 24;
const FACE_TEXT_BOX_WIDTH = 170;
const FACE_TEXT_LINE_HEIGHT = 30;
// A brief, irregular offset makes the centered type feel like a loose film frame.
const FILM_JITTER_INTERVAL_MIN = 380;
const FILM_JITTER_INTERVAL_MAX = 1300;
const FILM_JITTER_DISTANCE = 4;
let filmJitter = { x: 0, y: 0, nextAt: 0 };

// Change only the text on the right to customize what each detected class says.
const CLASS_TEXT = {
  'headless horse': 'One day I woke up, I lost my head',
  'elephant head': 'ELEPHANT',
  'snoopy head': 'SNOOPY',
  'me': 'hold some object up',
  'empty': 'hold some object up'
};

function preload() {
  // FaceMesh runs in the browser: no Roboflow key or cloud inference is used.
  faceMesh = ml5.faceMesh({
    maxFaces: 1,
    refineLandmarks: false,
    flipHorizontal: false
  });

  // This is the person/background mask, following the ml5 BodySegmentation example.
  bodySegmentation = ml5.bodySegmentation('SelfieSegmentation', {
    maskType: 'person'
  });

  classifier = ml5.imageClassifier(TM_MODEL_URL);
}

function setup() {
  pixelDensity(1);
  const canvas = createCanvas(windowWidth, windowHeight);
  canvas.parent('canvas-holder');

  asciiLayer = createGraphics(W, H);
  glowLayer = createGraphics(W, H);
  asciiLayer.pixelDensity(1);
  glowLayer.pixelDensity(1);
  asciiLayer.textFont('monospace');
  asciiLayer.textAlign(CENTER, CENTER);

  video = createCapture({ video: { width: W, height: H }, audio: false });
  video.size(W, H);
  video.hide();

  faceMesh.detectStart(video, gotFaces);
  bodySegmentation.detectStart(video, gotSegmentation);
  classifyVideo();
}

function draw() {
  background(0);

  if (personMask) {
    drawAsciiPerson();
  } else {
    asciiLayer.clear();
  }

  drawFaceText(getCenteredTextAnchor());

  renderFilteredAscii();
}

function getCenteredTextAnchor() {
  // Work in the 640 × 480 processing space: renderFilteredAscii() centers this
  // layer in the browser while preserving the camera ratio.
  if (millis() >= filmJitter.nextAt) {
    filmJitter.x = random(-FILM_JITTER_DISTANCE, FILM_JITTER_DISTANCE);
    filmJitter.y = random(-FILM_JITTER_DISTANCE, FILM_JITTER_DISTANCE);
    filmJitter.nextAt = millis() + random(FILM_JITTER_INTERVAL_MIN, FILM_JITTER_INTERVAL_MAX);
  }

  return {
    x: W * 0.5 + filmJitter.x,
    y: H * 0.5 + filmJitter.y
  };
}

function drawAsciiPerson() {
  video.loadPixels();
  personMask.loadPixels();
  if (video.pixels.length === 0 || personMask.pixels.length === 0) return;

  asciiLayer.clear();
  asciiLayer.noStroke();
  asciiLayer.textSize(ASCII_SIZE + 1);

  for (let y = ASCII_SIZE / 2; y < H; y += ASCII_SIZE) {
    for (let x = ASCII_SIZE / 2; x < W; x += ASCII_SIZE) {
      const videoX = floor(x * video.width / W);
      const videoY = floor(y * video.height / H);
      const videoIndex = 4 * (videoY * video.width + videoX);

      const personAlpha = maskAlphaAt(x, y);

      if (personAlpha < 80) continue;

      // Keep only cells close to the silhouette boundary.
      const isEdge =
        maskAlphaAt(x - EDGE_RADIUS, y) < 80 ||
        maskAlphaAt(x + EDGE_RADIUS, y) < 80 ||
        maskAlphaAt(x, y - EDGE_RADIUS) < 80 ||
        maskAlphaAt(x, y + EDGE_RADIUS) < 80 ||
        maskAlphaAt(x - EDGE_RADIUS, y - EDGE_RADIUS) < 80 ||
        maskAlphaAt(x + EDGE_RADIUS, y - EDGE_RADIUS) < 80 ||
        maskAlphaAt(x - EDGE_RADIUS, y + EDGE_RADIUS) < 80 ||
        maskAlphaAt(x + EDGE_RADIUS, y + EDGE_RADIUS) < 80;

      if (!isEdge) continue;

      // Leave stable gaps in the contour instead of filling every grid cell.
      if (positionHash(x, y, 17) < GAP_AMOUNT) continue;

      const r = video.pixels[videoIndex];
      const g = video.pixels[videoIndex + 1];
      const b = video.pixels[videoIndex + 2];
      const brightnessValue = 0.2126 * r + 0.7152 * g + 0.0722 * b;
      const symbolChoice = positionHash(x, y, 91);
      const char = symbolChoice < 0.48 ? '*' : symbolChoice < 0.95 ? '+' : '&';

      // Mirror x so the installation behaves like a mirror.
      asciiLayer.fill(180 + brightnessValue * 0.29, personAlpha);
      asciiLayer.text(char, W - x, y);
    }
  }

}

function drawFaceText(faceCenter) {
  const chars = getTransitionCharacters();
  if (chars.length === 0 || chars.every(char => char === ' ')) return;

  asciiLayer.push();
  asciiLayer.noStroke();
  asciiLayer.fill(255);
  asciiLayer.textFont('monospace');
  asciiLayer.textAlign(LEFT, CENTER);
  asciiLayer.textSize(FACE_TEXT_SIZE);

  const charWidth = asciiLayer.textWidth('M');
  const maxColumns = max(1, floor(FACE_TEXT_BOX_WIDTH / charWidth));
  const lines = wrapCharacters(chars, maxColumns);
  const blockHeight = lines.length * FACE_TEXT_LINE_HEIGHT;
  const startY = faceCenter.y - blockHeight * 0.5 + FACE_TEXT_LINE_HEIGHT * 0.5;
  const startX = faceCenter.x - FACE_TEXT_BOX_WIDTH * 0.5;

  lines.forEach((line, lineIndex) => {
    line.forEach((char, charIndex) => {
      asciiLayer.text(
        char,
        startX + charIndex * charWidth,
        startY + lineIndex * FACE_TEXT_LINE_HEIGHT
      );
    });
  });

  asciiLayer.pop();
}

function wrapCharacters(chars, maxColumns) {
  const lines = [];
  let line = [];
  let word = [];

  function placeWord() {
    if (word.length === 0) return;
    const requiredSpace = line.length > 0 ? 1 : 0;

    if (line.length > 0 && line.length + requiredSpace + word.length > maxColumns) {
      lines.push(line);
      line = [];
    }

    if (line.length > 0) line.push(' ');
    line.push(...word);
    word = [];
  }

  chars.forEach(char => {
    if (char === ' ') {
      placeWord();
    } else {
      word.push(char);
      // A long word still wraps rather than overflowing the chosen text-box width.
      if (word.length >= maxColumns) placeWord();
    }
  });

  placeWord();
  if (line.length > 0) lines.push(line);
  return lines;
}

function getTransitionCharacters() {
  const progress = constrain((millis() - transitionStartedAt) / TEXT_TRANSITION_MS, 0, 1);
  const length = max(transitionFrom.length, transitionTo.length);
  const chars = [];

  for (let index = 0; index < length; index += 1) {
    const oldChar = transitionFrom[index] || ' ';
    const newChar = transitionTo[index] || ' ';

    // The handoff order is stable but irregular, so it feels less mechanical.
    const handoff = positionHash(index, 0, 331);
    chars.push(progress >= handoff ? newChar : oldChar);
  }

  return chars;
}

function setTextTarget(nextText) {
  if (nextText === transitionTo) return;

  // If the classifier changes again mid-transition, continue from the visible state.
  transitionFrom = getTransitionCharacters().join('');
  transitionTo = nextText;
  transitionStartedAt = millis();
}

function renderFilteredAscii() {
  // The face text and ASCII contour share this blur-and-grain filter.
  glowLayer.clear();
  glowLayer.image(asciiLayer, 0, 0);
  glowLayer.filter(BLUR, 1);
  addGrain(glowLayer, NOISE_AMOUNT);

  // Fill the browser canvas without stretching the 4:3 camera image.
  const display = getDisplayRect();
  image(glowLayer, display.x, display.y, display.w, display.h);
}

function windowResized() {
  resizeCanvas(windowWidth, windowHeight);
}

function getDisplayRect() {
  const scale = min(width / W, height / H);
  const displayWidth = W * scale;
  const displayHeight = H * scale;
  return {
    x: (width - displayWidth) * 0.5,
    y: (height - displayHeight) * 0.5,
    w: displayWidth,
    h: displayHeight
  };
}

function addGrain(layer, amount) {
  layer.loadPixels();

  for (let i = 0; i < layer.pixels.length; i += 4) {
    const alpha = layer.pixels[i + 3];
    if (alpha === 0) continue;

    const grain = random(-amount, amount);
    layer.pixels[i] = constrain(layer.pixels[i] + grain, 0, 255);
    layer.pixels[i + 1] = constrain(layer.pixels[i + 1] + grain, 0, 255);
    layer.pixels[i + 2] = constrain(layer.pixels[i + 2] + grain, 0, 255);
    layer.pixels[i + 3] = constrain(
      alpha + random(-amount * 0.65, amount * 0.35),
      0,
      255
    );
  }

  layer.updatePixels();
}

function maskAlphaAt(canvasX, canvasY) {
  if (canvasX < 0 || canvasX >= W || canvasY < 0 || canvasY >= H) return 0;

  const maskX = constrain(floor(canvasX * personMask.width / W), 0, personMask.width - 1);
  const maskY = constrain(floor(canvasY * personMask.height / H), 0, personMask.height - 1);
  return personMask.pixels[4 * (maskY * personMask.width + maskX) + 3];
}

function positionHash(x, y, seed) {
  const value = sin(x * 12.9898 + y * 78.233 + seed) * 43758.5453;
  return value - floor(value);
}

function mirrored(point) {
  // FaceMesh coordinates are in the 640×480 camera coordinate system.
  return { x: W - point.x, y: point.y };
}

function gotFaces(results) {
  faces = results || [];
}

function classifyVideo() {
  classifier.classify(video, gotClassification);
}

// Supports both the current results-only callback and the older error-first form.
function gotClassification(firstArgument, secondArgument) {
  const results = Array.isArray(firstArgument) ? firstArgument : secondArgument;
  const error = Array.isArray(firstArgument) ? null : firstArgument;

  if (error) {
    console.error('Classification error:', error);
  } else if (results && results.length > 0) {
    const nextLabel = results[0].label;
    candidateCount = nextLabel === candidateLabel ? candidateCount + 1 : 1;
    candidateLabel = nextLabel;

    // Show the first valid classification immediately. After that, ignore
    // one-off flickers before starting another text transition.
    const isFirstClassification = classificationLabel === 'loading classifier...';
    if ((isFirstClassification || candidateCount >= 2) && nextLabel !== classificationLabel) {
      classificationLabel = nextLabel;
      setTextTarget(CLASS_TEXT[classificationLabel] || '');
    }
  }

  // The classifier is only for debugging, so it does not need to run every frame.
  setTimeout(classifyVideo, 150);
}

function gotSegmentation(result) {
  segmentation = result;
  personMask = invertMask(segmentation.mask);
  fallbackTextAnchor = findMaskHeadAnchor(personMask) || fallbackTextAnchor;
}

function findMaskHeadAnchor(mask) {
  mask.loadPixels();
  let minX = mask.width;
  let minY = mask.height;
  let maxX = -1;
  let maxY = -1;

  // Sampling every four pixels is enough for a quiet, stable fallback anchor.
  for (let y = 0; y < mask.height; y += 4) {
    for (let x = 0; x < mask.width; x += 4) {
      const alpha = mask.pixels[4 * (y * mask.width + x) + 3];
      if (alpha < 80) continue;
      minX = min(minX, x);
      minY = min(minY, y);
      maxX = max(maxX, x);
      maxY = max(maxY, y);
    }
  }

  if (maxX < minX || maxY < minY) return null;

  const scaleX = W / mask.width;
  const scaleY = H / mask.height;
  const bodyWidth = (maxX - minX) * scaleX;
  const bodyHeight = (maxY - minY) * scaleY;

  return {
    x: W - ((minX + maxX) * 0.5 * scaleX),
    y: (minY * scaleY) + bodyHeight * 0.19,
    width: bodyWidth * 0.45
  };
}

// The SelfieSegmentation mask supplied here has transparent people and an
// opaque background. Reverse only alpha so the person stays visible instead.
function invertMask(mask) {
  const inverted = mask.get();
  inverted.loadPixels();

  for (let i = 3; i < inverted.pixels.length; i += 4) {
    inverted.pixels[i] = 255 - inverted.pixels[i];
  }

  inverted.updatePixels();
  return inverted;
}
