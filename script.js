const imageUpload = document.getElementById("imageUpload");

Promise.all([
  faceapi.nets.faceRecognitionNet.loadFromUri("/models"),
  faceapi.nets.faceLandmark68Net.loadFromUri("/models"),
  faceapi.nets.ssdMobilenetv1.loadFromUri("/models"),
]).then(start);

let image;
let canvas;

//function for adding names to faces
async function start() {
  const container = document.createElement("div");
  container.style.position = "relative";
  document.body.append(container);
  const labeledFaceDescriptors = await loadLabeledImages();
  //faceMatcher is used to match the face with the labeled face
  const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.6);
  document.body.append("Loaded");
  imageUpload.addEventListener("change", async () => {
    if (image) image.remove();
    if (canvas) canvas.remove();
    image = await faceapi.bufferToImage(imageUpload.files[0]);
    container.append(image);
    canvas = faceapi.createCanvasFromMedia(image);
    container.append(canvas);
//displaySize is used to display the image in the canvas
    const displaySize = { width: image.width, height: image.height };
    faceapi.matchDimensions(canvas, displaySize);
    const detections = await faceapi
      .detectAllFaces(image)
      .withFaceLandmarks()
      .withFaceDescriptors();
    const resizedDetections = faceapi.resizeResults(detections, displaySize);
    const results = resizedDetections.map((d) =>
      faceMatcher.findBestMatch(d.descriptor)
    );
    //drawBox is used to draw the box around the face
    results.forEach((result, i) => {
      const box = resizedDetections[i].detection.box;
      const drawBox = new faceapi.draw.DrawBox(box, {
        label: result.toString(),
      });
      //drawing face on the canvas
      drawBox.draw(canvas);
    });
  });
}

//function for loading the labeled images with the names
function loadLabeledImages() {
  const labels = [
    "Black Widow",
    "Captain America",
    "Captain Marvel",
    "Hawkeye",
    "Jim Rhodes",
    "Thor",
    "Tony Stark",
  ];
  //getting images from external source
  return Promise.all(
    labels.map(async (label) => {
      const descriptions = [];
      for (let i = 1; i <= 2; i++) {
        const img = await faceapi.fetchImage(
          `https://raw.githubusercontent.com/WebDevSimplified/Face-Recognition-JavaScript/master/labeled_images/${label}/${i}.jpg`
        );
        //declaring the face landmarks and face descriptor
        const detections = await faceapi
          .detectSingleFace(img)
          .withFaceLandmarks()
          .withFaceDescriptor();
        descriptions.push(detections.descriptor);
      }
      //retreiving the labeled face descriptors on each character
      return new faceapi.LabeledFaceDescriptors(label, descriptions);
    })
  );
}

//inspired by https://www.youtube.com/watch?v=AZ4PdALMqx0&t=339s 
