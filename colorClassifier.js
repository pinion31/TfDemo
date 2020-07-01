require('@tensorflow/tfjs-node');
const data = require('./colorData.json');
const tf = require('@tensorflow/tfjs');

let model;

const labelList = [
  'red-ish',
  'green-ish',
  'blue-ish',
  'orange-ish',
  'yellow-ish',
  'pink-ish',
  'purple-ish',
  'brown-ish',
  'grey-ish'
]

const setup = ()  => {
  let colors = [];
  let labels = [];

  for (let record of data.entries) {
    let col = [record.r / 255, record.g / 255, record.b / 255];
    colors.push(col);
    // [ 0.3176470588235294, 0.7176470588235294, 0.6078431372549019 ],
    labels.push(labelList.indexOf(record.label));
    // [1, 5, 3, 2, 2, 1, 1, 1, 2, 3, 7, 0, ..... one for reach rgb value
    // number represents index from labelList
  }

  const xs = tf.tensor2d(colors);
  let labelsTensor = tf.tensor1d(labels, 'int32');

  const ys = tf.oneHot(labelsTensor, 9).cast('float32');
  // converts. ex: 1 => [0,1,0,0,0,0,0,0,0]
  labelsTensor.dispose();
  return { xs, ys };
}

async function train(model, featureSet, labelSet) {
  await model.fit(featureSet, labelSet, {
    epochs: 10, // how many iterations where it tries to make predictions
    shuffle: true,
    validationSplit: 0.1, // represents the fraction of the data to be reserved for validation
    callbacks: {
      onEpochEnd: (epoch, logs) => { // similar to lifecycle methods
        console.log(epoch);
      },
      onBatchEnd: async (batch, logs) => {
        await tf.nextFrame();
      },
      onTrainEnd: () => {
        istraining = false;
        console.log('finished training');
      },
    },
  });
}

function buildModel() {
  let md = tf.sequential();
  // input layer => dense layers => output layer


  const nodes = tf.layers.dense({
    units: 15,
    inputShape: [3], // three inputs: one for red value, one for green value, one for blue value
    activation: 'sigmoid'
  });

  const output = tf.layers.dense({
    units: 9, 
    //[0,0,0,1,0,0,0,0,0] => 
    //corresponds to our labellist
    //[
    //   'red-ish',
    //   'green-ish',
    //   'blue-ish',
    //   'orange-ish',
    //   'yellow-ish',
    //   'pink-ish',
    //   'purple-ish',
    //   'brown-ish',
    //   'grey-ish'
    // ]
    activation: 'softmax'
  });
  md.add(nodes);
  md.add(output);

  const LEARNING_RATE = 0.25; // rate of adjustment
  const optimizer = tf.train.sgd(LEARNING_RATE); // stochastic gradient descent

  md.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  return md
}

const buildAndTrainModel = async () => {
  // split into feature and label
  const { xs, ys } = setup(); // setup data
  const model = buildModel(); // build Model first
  await train(model, xs, ys); // setup training for model
  return model;

}


buildAndTrainModel().then((model) => {
  const input = tf.tensor2d([
    [0, 255, 0] //(r,g,b)
  ]);
  const predictedResult = model.predict(input);
  const argMax = predictedResult.argMax(1);
  const index = argMax.dataSync()[0];
  const label = labelList[index];
  console.log('Color:', label)
});