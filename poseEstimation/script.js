const status = document.getElementById('status');
if (status) {
  status.innerText = 'Loaded TensorFlow.js - version: ' + tf.version.tfjs;
}

const MODEl_PATH = 'https://www.kaggle.com/api/v1/models/google/movenet/tfJs/singlepose-lightning/4/download';
const EXAMPLE_IMG = document.getElementById('exampleImg');

let movenet = undefined;

const loadAndRunModel = async () => {
  movenet = await tf.loadGraphModel(MODEl_PATH, { fromTFHub: true });

  let imageTensor = tf.browser.fromPixels(EXAMPLE_IMG);

  let cropStartPoint = [15, 170, 0];
  let cropSize = [345, 345, 3];
  let croppedTensor = tf.slice(imageTensor, cropStartPoint, cropSize);

  let resizedTensor = tf.image.resizeBilinear(croppedTensor, [192, 192], true).toInt();
  console.log(resizedTensor.shape);

  let tensorOutput = movenet.predict(tf.expandDims(resizedTensor));
  let arrayOutput = await tensorOutput.array();

  imageTensor.dispose();
  croppedTensor.dispose();
  resizedTensor.dispose();
  tensorOutput.dispose();

  console.log(arrayOutput);
}

loadAndRunModel();

