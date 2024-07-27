const normalize = (tensor, min, max) => {
  const result = tf.tidy(() => {
    const MIN_VALUES = min || tf.min(tensor, 0);
    const MAX_VALUES = max || tf.max(tensor, 0);

    const TENSOR_SUBTRACT_MIN_VALUE = tf.sub(tensor, MIN_VALUES);
    const RANGE_SIZE = tf.sub(MAX_VALUES, MIN_VALUES);
    const NORMALIZED_VALUES = tf.div(TENSOR_SUBTRACT_MIN_VALUE, RANGE_SIZE);

    return { NORMALIZED_VALUES, MIN_VALUES, MAX_VALUES };
  })

  return result;
}

const logProgress = (epoch, logs) => {
  console.log('Data for epoch ' + epoch, Math.sqrt(logs.loss));
}

const evaluate = async () => {
  tf.tidy(() => {
    let newInput = normalize(tf.tensor1d([7]), FEATURE_RESULTS.MIN_VALUES, FEATURE_RESULTS.MAX_VALUES);

    let output = model.predict(newInput.NORMALIZED_VALUES);
    output.print();
  });

  await model.save('downloads://firstTrainedModel');

  FEATURE_RESULTS.MIN_VALUES.dispose();
  FEATURE_RESULTS.MAX_VALUES.dispose();
  model.dispose();

  console.log(tf.memory().numTensors);
};

const train = async () => {
  const LEARNING_RATE = 0.0001;

  model.compile({
    optimizer: tf.train.sgd(LEARNING_RATE),
    loss: 'meanSquaredError'
  });

  let results = await model.fit(FEATURE_RESULTS.NORMALIZED_VALUES, OUTPUTS_TENSOR, {
    callbacks: {onEpochEnd: logProgress},
    shuffle: true,
    batchSize: 2,
    epochs: 200
  });

  OUTPUTS_TENSOR.dispose();
  FEATURE_RESULTS.NORMALIZED_VALUES.dispose();

  console.log('Average error loss: ' + Math.sqrt(results.history.loss[results.history.loss.length - 1]));

  evaluate();
}

const status = document.getElementById('status');
if (status) {
  status.innerText = 'Loaded TensorFlow.js - version: ' + tf.version.tfjs;
}

const INPUTS = [];
for (let n = 1; n <= 20; n++) {
  INPUTS.push(n);
}

const OUTPUTS = [];
for (let n = 1; n <= 20; n++) {
  OUTPUTS.push(n * n);
}

tf.util.shuffleCombo(INPUTS, OUTPUTS);

const INPUTS_TENSOR = tf.tensor1d(INPUTS);
const OUTPUTS_TENSOR = tf.tensor1d(OUTPUTS);

const FEATURE_RESULTS = normalize(INPUTS_TENSOR);

INPUTS_TENSOR.dispose();

const model = tf.sequential();

model.add(tf.layers.dense({ inputShape: [1], units: 25, activation: 'relu' }));
model.add(tf.layers.dense({ units: 5, activation: 'relu' }));
model.add(tf.layers.dense({ units: 1 }));
model.summary();

train();




