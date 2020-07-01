require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');

// CORE API

// Tensor is the base structure in
// Tensorflow that wraps arrays
// Base unit for tensorflow

// Shape [5, 6]
// Shape describes dimensionality of tensor
// 2d tensor
const commonCollectionsRated = tf.tensor([
    [1,3,4,2,3,4],
    [3,5,2,3,2,1],
    [5,2,4,4,5,3],
    [4,3,5,4,4,3],
    [2,2,2,5,3,2],
]);

// Or you can create a tensor from a flat array and specify a shape and data type.
const shape = [2, 2]; 
const b = tf.tensor([1, 2, 3, 4], shape, 'int32'); // last 2 are optional
// Tensor can contain numbers, strings or booleans
// but only one data type
b.print();
// Tensor
//     [[1, 2],
//      [3, 4]]
// 
console.log(b);
// Tensor {
//   kept: false,
//   isDisposedInternal: false,
//   shape: [ 2, 2 ],
//   dtype: 'int32',
//   size: 4,
//   strides: [ 2 ],
//   dataId: {},
//   id: 1,
//   rankType: '2'
// }
//



// More descriptive with tensor creation
tf.scalar(3).print();  // 3
tf.tensor1d([1,2,3]).print() // [1,2,3]
tf.tensor2d([[1, 2], [3, 4]]).print();
/* Tensor
    [[1, 2],
     [3, 4]]*/
tf.tensor6d([[[[[[1],[2]],[[3],[4]]],[[[5],[6]],[[7],[8]]]]]]).print();
/*
Tensor
    [[[[[[1],
         [2]],

        [[3],
         [4]]],


       [[[5],
         [6]],

        [[7],
         [8]]]]]] */


// *********OPERATIONS****************
// tf.add
// tf.sub
// tf.mul
// tf.div
// tf.addN
// tf.divNoNan
// tf.floorDiv
// tf.maximum
// tf.minimum
// tf.mod
// tf.pow
// tf.squaredDifference

// Make shapes match when performing operations
// [2,3]
// Addition example
const tensor1 = tf.tensor([
  [1,1,1],
  [1,1,1],
]);

// [1,4]
const tensorWithWrongShape = tf.tensor([
  [1,2,3,4],
]);

//Message: Incompatible shapes: [2,3] vs. [1,4]
tensor1.add(tensorWithWrongShape).print();

const tensorWithCorrectShape1 = tf.tensor([
  [1,1,1],
  [1,1,1],
]);

const tensorWithCorrectShape2 = tf.tensor([
  [1,2,3],
]);

tensorWithCorrectShape1.add(tensorWithCorrectShape2).print();

// result:
// Tensor
//     [[2, 3, 4],
//      [2, 3, 4]]

//Tensors are immutable, so all operations always return new Tensors.

