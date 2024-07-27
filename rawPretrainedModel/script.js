const status = document.getElementById('status');
if (status) {
  status.innerText = 'Loaded TensorFlow.js - version: ' + tf.version.tfjs;
}

const modelLink = 'https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/SavedModels/sqftToPropertyPrice/model.json';
let model = undefined;

const loadModel = async () => {
  
  if (JSON.stringify(await tf.io.listModels())){
    model = await tf.loadLayersModel('localstorage://demo/sqftToPropertyPrice');
    model.summary();
  }
  else {
    model = await tf.loadLayersModel(modelLink);
    await model.save('localstorage://demo/sqftToPropertyPrice')
    model.summary();
  }
  
  // a batch of 1
  const input = tf.tensor2d([[870]]);

  // a batch of 3
  const inputBatch = tf.tensor2d([[500], [1100], [970]])

  const result = model.predict(input);
  const resultBatch = model.predict(inputBatch);

  result.print();
  resultBatch.print();

  input.dispose();
  inputBatch.dispose();
  result.dispose();
  resultBatch.dispose();
  model.dispose();
}

loadModel();
