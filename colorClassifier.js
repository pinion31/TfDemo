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

async function train(model, xs, ys) {
  await model.fit(xs, ys, {
    shuffle: true,
    validationSplit: 0.1,
    epochs: 10,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log(epoch);
      },
      onBatchEnd: async (batch, logs) => {
        await tf.nextFrame();
      },
      onTrainEnd: () => {
        istraining = false;
        console.log('finished');
      },
    },
  });
}

function buildModel() {
  let md = tf.sequential();
  // input => nodes layers => output


  const nodes = tf.layers.dense({
    units: 15,
    inputShape: [3], // three inputs: one for red value, one for green value, one for blue value
    activation: 'sigmoid'
  });

  const output = tf.layers.dense({
    units: 9, 
    //[0,0,0,1,0,0,0,0,0] => [
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

  const LEARNING_RATE = 0.25;
  const optimizer = tf.train.sgd(LEARNING_RATE);

  md.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  return md
}

const trainAndRunModel = async () => {
  const { xs, ys } = setup(); // setup data
  const model = buildModel(); // build Model first
  await train(model, xs, ys); // setup training for model
  return model;

}


trainAndRunModel().then((model) => {
  console.log('finished training');
  const input = tf.tensor2d([
    [0, 255, 0] //(r,g,b)
  ]);
  let predictedResult = model.predict(input);
  let argMax = predictedResult.argMax(1);
  let index = argMax.dataSync()[0];
  console.log('Color:')
  let label = labelList[index];
  console.log('Color:', label)
});