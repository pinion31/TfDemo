require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');

//const w1 = tf.variable(tf.randomNormal([1, 6]));
const w1 = tf.variable(tf.tensor([
  [3,2,4,4,5,2] 
]));

function model(x) {
  console.log('x is', x);
  console.log('w1','[3,2,4,4,5,2] ');
  const t = tf.tensor(x);
  const result = 
  t.sub(w1) 
   .pow(2)
   .sum(1) // 1 for horizontal axis, 0 or blank for vertical axis
   .pow(.5)
  // .unstack()
  // .sort((a,b) => a.arraySync()[0] > b.arraySync()[0] ? 1 : -1)[0];
  console.log('resulty', result.print());
  return result;
};

// const data = tf.tensor([
//   [1,3,4,2,3,4],
//   [3,5,2,3,2,1],
//   [5,2,4,4,5,3],
//   [4,3,5,4,4,3],
//   [2,2,2,5,3,2],
// ]);

// const labels = tf.tensor([
//   [0,1,2,3,4]
// ]);

const data = tf.data.array([
  [1,3,4,2,3,4],
  [3,5,2,3,2,1],
  [5,2,4,4,5,3],
  [4,3,5,4,4,3],
  [2,2,2,5,3,2],
]);

const labels = tf.data.array([
  [3,4,4,1,5,2,3,2,1],
  [5,3,3,4,2,2,1,2,1],
  [5,3,5,5,4,4,3,3,2],
  [4,3,3,3,4,5,1,3,3],
  [2,2,3,3,1,1,3,4,2],
]);

const xs = data;
const ys = labels;
console.log('shappe', xs);
console.log('shappeY', ys);
// Zip the data and labels together, shuffle and batch 32 samples at a time.
//const ds = tf.data.zip({xs, ys}).shuffle(100 /* bufferSize */).batch(32);
const ds = tf.data.zip({xs, ys});


const trainModel = async () => {
  const optimizer = tf.train.sgd(0.1 /* learningRate */);
  // Train for 5 epochs.
  for (let epoch = 0; epoch < 5; epoch++) {
    await ds.forEachAsync(({xs, ys}) => {
      optimizer.minimize(() => {
        const predYs = model(xs);
        const loss = tf.losses.softmaxCrossEntropy(ys, predYs);
        loss.data().then(l => console.log('Loss', l));
        return loss;
      });
    });
    console.log('Epoch', epoch);
  }
}

trainModel();


const testData = tf.tensor([
  [3,2,4,4,5,2]
]);

//const prediction = model.predict(testData);
//prediction.print();

