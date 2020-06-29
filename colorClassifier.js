require('@tensorflow/tfjs-node');
const data = require('./colorData.json');
const tf = require('@tensorflow/tfjs');

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

const setup = async ()  => {
  let colors = [];
  let labels = [];

  for (let record of data.entries) {
    let col = [record.r / 255, record.g / 255, record.b / 255];
    colors.push(col);
    labels.push(labelList.indexOf(record.label));
  }

  const xs = tf.tensor2d(colors);
  let labelsTensor = tf.tensor1d(labels, 'int32');

  const ys = tf.oneHot(labelsTensor, 9).cast('float32');
  labelsTensor.dispose();

  model = buildModel();
  await train(model, xs, ys);
  return model;
}

async function train(model, xs, ys) {
  await model.fit(xs, ys, {
    shuffle: true,
    validationSplit: 0.1,
    epochs: 10,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log(epoch);
        // lossY.push(logs.val_loss.toFixed(2));
        // accY.push(logs.val_acc.toFixed(2));
        // lossX.push(lossX.length + 1);
        // lossP.html('Loss: ' + logs.loss.toFixed(5));
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
  const hidden = tf.layers.dense({
    units: 15,
    inputShape: [3],
    activation: 'sigmoid'
  });

  const output = tf.layers.dense({
    units: 9,
    activation: 'softmax'
  });
  md.add(hidden);
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

setup().then(() => {
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