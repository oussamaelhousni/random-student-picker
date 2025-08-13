import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl";
import * as cocoSsd from "@tensorflow-models/coco-ssd";

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const statusEl = document.getElementById("status");
const toast = document.getElementById("toast");

const fsBtn = document.getElementById("fs");
const flipBtn = document.getElementById("flip");
const pickBtn = document.getElementById("pick");
const confSlider = document.getElementById("conf");
const confVal = document.getElementById("confv");

// Spotlight elements
const spotlight = document.getElementById("spotlight");
const spotImg = document.getElementById("spotImg");
const spotCaption = document.getElementById("spotCaption");
const spotClose = document.getElementById("spotClose");

let model;
let running = true;
let minScore = parseFloat(confSlider.value);
let facingMode = "user"; // "user" | "environment"
let currentPeople = []; // filtered detections of class "person"
let selectedIdx = null; // index into currentPeople for highlight
let selectUntil = 0; // timestamp until highlight remains

// Resize canvas to viewport
function fitCanvas() {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
}
window.addEventListener("resize", fitCanvas);

function showToast(msg, ms = 1500) {
  toast.textContent = msg;
  toast.classList.add("show");
  setTimeout(() => toast.classList.remove("show"), ms);
}

async function setupCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: {
      facingMode,
      width: { ideal: 1920 },
      height: { ideal: 1080 },
    },
  });
  if (video.srcObject) {
    for (const t of video.srcObject.getTracks()) t.stop();
  }
  video.srcObject = stream;

  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      video.play();
      requestAnimationFrame(() => {
        fitCanvas();
        resolve();
      });
    };
  });
}

/** Draw live detections on overlay; maintain currentPeople list */
function draw(preds) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const vw = video.videoWidth || 1280;
  const vh = video.videoHeight || 720;
  const cw = canvas.width;
  const ch = canvas.height;
  const scaleX = cw / vw;
  const scaleY = ch / vh;

  currentPeople = preds.filter(
    (p) => p.class === "person" && p.score >= minScore
  );

  currentPeople.forEach((p, i) => {
    const [x, y, w, h] = p.bbox;
    const rx = x * scaleX,
      ry = y * scaleY,
      rw = w * scaleX,
      rh = h * scaleY;

    const isChosen = selectedIdx === i && performance.now() < selectUntil;

    ctx.lineWidth = isChosen ? 4 : 2;
    ctx.strokeStyle = isChosen ? "#22c55e" : "#2563eb";
    ctx.fillStyle = isChosen ? "rgba(34,197,94,0.15)" : "rgba(37,99,235,0.15)";
    ctx.strokeRect(rx, ry, rw, rh);
    ctx.fillRect(rx, ry, rw, rh);

    const label = isChosen
      ? "🎉 Selected!"
      : `person ${(p.score * 100).toFixed(0)}%`;
    ctx.font = "16px system-ui, -apple-system, Segoe UI, Roboto, Inter, Arial";
    ctx.textBaseline = "top";
    const padX = 6,
      padY = 4;
    const textW = ctx.measureText(label).width;
    const boxW = textW + padX * 2;
    const boxH = 22 + padY * 2;
    const ly = Math.max(ry - boxH, 0);

    ctx.fillStyle = "rgba(0,0,0,0.65)";
    ctx.fillRect(rx, ly, boxW, boxH);
    ctx.fillStyle = "#fff";
    ctx.fillText(label, rx + padX, ly + padY);

    if (isChosen) {
      ctx.beginPath();
      ctx.lineWidth = 6;
      ctx.strokeStyle = "rgba(34,197,94,0.7)";
      ctx.strokeRect(rx - 4, ry - 4, rw + 8, rh + 8);
    }
  });
}

/** Capture a cropped snapshot of the selected person from the raw video.
 *  Returns a data URL (PNG) suitable for an <img>.
 */
function captureSelectedCrop(det) {
  const [x, y, w, h] = det.bbox; // coordinates in video space (raw pixels)
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  if (!vw || !vh) return null;

  // Clamp bbox to video bounds
  const sx = Math.max(0, Math.floor(x));
  const sy = Math.max(0, Math.floor(y));
  const sw = Math.min(vw - sx, Math.floor(w));
  const sh = Math.min(vh - sy, Math.floor(h));
  if (sw <= 1 || sh <= 1) return null;

  // Draw crop to offscreen canvas at source resolution for best quality
  const off = document.createElement("canvas");
  off.width = sw;
  off.height = sh;
  const octx = off.getContext("2d", { willReadFrequently: true });

  // Mirror if front camera (so snapshot matches what user sees)
  const mirrored = facingMode === "user";
  if (mirrored) {
    octx.translate(sw, 0);
    octx.scale(-1, 1);
    octx.drawImage(video, sx, sy, sw, sh, 0, 0, sw, sh);
  } else {
    octx.drawImage(video, sx, sy, sw, sh, 0, 0, sw, sh);
  }

  return off.toDataURL("image/png");
}

/** Show spotlight modal with image + caption */
function showSpotlight(
  dataUrl,
  caption = "Ready to answer the next question!"
) {
  if (!dataUrl) return;
  spotImg.src = dataUrl;
  spotCaption.textContent = caption;
  spotlight.classList.add("show");
  spotlight.setAttribute("aria-hidden", "false");
}

/** Hide spotlight modal */
function hideSpotlight() {
  spotlight.classList.remove("show");
  spotlight.setAttribute("aria-hidden", "true");
  // Optional: clear image to free memory
  spotImg.src = "";
}

async function loop() {
  if (!running) return;
  const preds = await model.detect(video, 50);
  draw(preds);
  requestAnimationFrame(loop);
}

async function main() {
  try {
    statusEl.textContent = "Starting camera…";
    await setupCamera();

    statusEl.textContent = "Initializing TensorFlow.js…";
    await tf.setBackend("webgl").catch(() => tf.setBackend("cpu"));
    await tf.ready();

    statusEl.textContent = "Loading detector…";
    model = await cocoSsd.load({ base: "lite_mobilenet_v2" });

    statusEl.textContent = "Warming up…";
    tf.tidy(() => tf.zeros([1, 224, 224, 3]).dataSync());

    statusEl.textContent = "Detecting people…";
    loop();
  } catch (e) {
    console.error(e);
    statusEl.textContent = "Error: " + e.message;
  }
}
main();

// UI: Fullscreen
fsBtn.addEventListener("click", async () => {
  try {
    if (!document.fullscreenElement) {
      await document.documentElement.requestFullscreen();
    } else {
      await document.exitFullscreen();
    }
  } catch {}
});

// UI: Flip camera
flipBtn.addEventListener("click", async () => {
  facingMode = facingMode === "user" ? "environment" : "user";
  showToast(`Switching camera: ${facingMode === "user" ? "Front" : "Rear"}`);
  await setupCamera();
});

// UI: Confidence threshold
confSlider.addEventListener("input", (e) => {
  minScore = parseFloat(e.target.value);
  confVal.textContent = e.target.value;
});

// UI: Pick random student -> capture crop -> spotlight
pickBtn.addEventListener("click", async () => {
  if (currentPeople.length === 0) {
    showToast("No students detected. Adjust camera/angle.", 2000);
    selectedIdx = null;
    return;
  }

  // Fun countdown
  showToast("Picking in 3…");
  await new Promise((r) => setTimeout(r, 400));
  showToast("2…");
  await new Promise((r) => setTimeout(r, 400));
  showToast("1…");
  await new Promise((r) => setTimeout(r, 400));

  selectedIdx = Math.floor(Math.random() * currentPeople.length);
  selectUntil = performance.now() + 3500; // temporary highlight
  showToast("🎉 Selected!", 1000);

  // Capture high-quality crop and show spotlight
  const chosen = currentPeople[selectedIdx];
  const dataUrl = captureSelectedCrop(chosen);
  showSpotlight(dataUrl, "Ready to answer the next question!");
});

// Spotlight close
spotClose.addEventListener("click", hideSpotlight);
// Also close on backdrop click (but ignore clicks on the card itself)
spotlight.addEventListener("click", (e) => {
  if (e.target === spotlight) hideSpotlight();
});
