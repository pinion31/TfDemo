require('@tensorflow/tfjs-node');
// require('@tensorflow/tfjs-node-gpu');
// maybe another lib required

const tf = require('@tensorflow/tfjs')

const data = {
  chris: {
    I: 5,
    II: 3,
    III: 2,
    IV: 4,
    V: 4,
    VI: 3,
    VII: 5,
  },
  nicole: {
    I: 1,
    II: 3,
    III: 3,
    IV: 2,
    V: 2,
    VI: 2,
    VII: 1,
  },
  lucy: {
    I: 5,
    II: 3,
    III: 4,
    IV: 4,
    V: 5,
    VI:5,
    VII: 5,
  },
  mini: {
    I: 3,
    II: 3,
    III: 4,
    IV: null,
    V: 3,
    VI: 2,
    VII: 2,
  },
};


const getDeviation = (tensor => {
  const avgs = tensor.sum(1).div(tensor.shape[1]).expandDims(1);
  return tensor.sub(avgs);
});

// ratings that the test user has in common
// 5, 6 shape (can run commonRatings.shape )
const commonCollectionsRated = tf.tensor([
  [1,3,4,2,3,4],
  [3,5,2,3,2,1],
  [5,2,4,4,5,3],
  [4,3,5,4,4,3],
  [2,2,2,5,3,2],
]);

// getDeviation(commonCollectionsRated).print();
// rating that the test user does not have in common
// 5,2 shape
const uncommonCollectionsRated = tf.tensor([
  [3,4,4,1,5,2,3,2,1],
  [5,3,3,4,2,2,1,2,1],
  [5,3,5,5,4,4,3,3,2],
  [4,3,3,3,4,5,1,3,3],
  [2,2,3,3,1,1,3,4,2],
]);


//1, 6 shape -- assuming user has rated 2 or more collections
const userInputData = tf.tensor([
  [3,2,4,4,5,2] 
]);

const unabiasedCommonCollectionsRated = getDeviation(commonCollectionsRated);
const unabiasedUserInputData = getDeviation(userInputData);
const unabiasedUncommonCollectionsRated = getDeviation(uncommonCollectionsRated);

const result = 
    unabiasedCommonCollectionsRated
    .sub(unabiasedUserInputData) 
    .pow(2)
    .sum(1) // 1 for horizontal axis, 0 or blank for vertical axis
    .pow(.5)
    // now we want to sort but we have to match shape
    .expandDims(1)
    .concat(unabiasedUncommonCollectionsRated, 1) // 1 along horizontal axis
    .unstack() // converts to array of tensors
    .sort((a,b) => a.arraySync()[0] > b.arraySync()[0] ? 1 : -1)[0]
    .slice([1],[9])


    
  const highestRating = result.max().arraySync();
  console.log('highestRating', highestRating);
  const ratingsInArray = result.arraySync();
  console.log('ratingsInArray', ratingsInArray);
  console.log("index", ratingsInArray.indexOf(highestRating));

    

  


result.print();
// console.log('result', result.print());

// //[    [4.1231055],[4.8989797],[2.236068 ],[2.236068 ],[3.1622777] ]


// console.log(uncommonCollectionsRated.shape);
// console.log(userInputData.shape);
// console.log(result.shape);

