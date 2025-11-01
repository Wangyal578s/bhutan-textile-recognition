const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const labelText = document.getElementById("label-text");

async function sendFrame() {
  const response = await fetch("/predict", {
    method: "POST",
    body: await captureFrame()
  });

  const data = await response.json();
  drawResult(data);
}

async function captureFrame() {
  const canvasTemp = document.createElement("canvas");
  canvasTemp.width = video.videoWidth;
  canvasTemp.height = video.videoHeight;
  const ctxTemp = canvasTemp.getContext("2d");
  ctxTemp.drawImage(video, 0, 0);
  const blob = await new Promise((resolve) => canvasTemp.toBlob(resolve, "image/jpeg"));
  const formData = new FormData();
  formData.append("frame", blob, "frame.jpg");
  return formData;
}

function drawResult(data) {
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  if (data.success && data.bbox) {
    const [x1, y1, x2, y2] = data.bbox;

    const color =
      data.confidence >= 85 ? "lime" : data.confidence >= 70 ? "yellow" : "red";

    // Draw bounding box
    ctx.strokeStyle = color;
    ctx.lineWidth = 4;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    // Draw label + confidence
    ctx.fillStyle = color;
    ctx.font = "20px Arial";
    ctx.fillText(`${data.label} (${data.confidence}%)`, x1 + 5, y1 - 10);

    labelText.innerText = `${data.label} (${data.confidence}%)`;
  } else {
    labelText.innerText = "Not Registered Textile";
  }
}

async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  video.srcObject = stream;
  video.play();
  setInterval(sendFrame, 2000); // every 2 sec
}

startCamera();
