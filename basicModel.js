require('@tensorflow/tfjs-node');

const tf = require('@tensorflow/tfjs');


const model = tf.sequential(); // model is linear

const configDense = {
    units: '4',
    inputShape: [3],
    activation: 'sigmoid'
}

// input layer is really part of dense layer
const dense = tf.layers.dense(configDense);

const configOutput = {
    units: '4',
    activation: 'sigmoid'
}
const output = tf.layers.dense(configOutput);

model.add(dense);
model.add(output);

const LEARNING_RATE = 0.25;
// optimizer minimizes a loss function
const optimizer = tf.train.sgd(LEARNING_RATE); //(sigmoid)

md.compile({
  optimizer: optimizer,
  loss: 'categoricalCrossentropy',
  metrics: ['accuracy'],
});

