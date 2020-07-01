require('@tensorflow/tfjs-node');
// require('@tensorflow/tfjs-node-gpu');
// maybe another lib required

// Simple recommendation system using KNN algorithm

const tf = require('@tensorflow/tfjs')

const potentialCollectionsToRecommend = [
  'collection1_id',
  'collection2_id',
  'collection3_id',
  'collection4_id',
  'collection5_id',
  'collection6_id']
];

// ratings that the test user has in common
// 5, 6 shape (can run commonRatings.shape )
const commonCollectionsRated = tf.tensor([
  // each index/column represents a collection
  [1,3,4,2,3,4], // feature user 1 - rating for 6 collections
  [3,5,2,3,2,1], // feature user 2
  [5,2,4,4,5,3], // feature user 3
  [4,3,5,4,4,3], // feature user 4
  [2,2,2,5,3,2], // feature user 5
]);

//1, 6 shape -- assuming user has rated 2 or more collections
// here is our user that we will recommend to
const userInputData = tf.tensor([
  [3,2,4,4,5,2] // user input
]);

// rating that the test user does not have in common
// 5,2 shape
// Another tensor of possible recommendations
const possibleRecommendations = tf.tensor([
  [3,4,4,1,5,2,3,2,1], // feature user 1
  [5,3,3,4,2,2,1,2,1], // feature user 2
  [5,3,5,5,4,4,3,3,2], // feature user 3
  [4,3,3,3,4,5,1,3,3], // feature user 4
  [2,2,3,3,1,1,3,4,2], // feature user 5
]);





// chaining operations 
const result = 
    commonCollectionsRated
    .sub(userInputData) // x2 - x1 or y1- y2
    .pow(2) //(x2 - x1)**2
    .sum(1) // 1 for horizontal axis, 0 or blank for vertical axis
    .pow(.5) // squaring the result 
    // now we want to sort but we have to match shape
    .expandDims(1)
    // [3.9, 4.6, 1.9, 1.8, 2.7]  ==>  [                  [                                 [
    //                                  [3.9],               [3,4,4,1,5,2,3,2,1],              [3.9,3,4,4,1,5,2,3,2,1],
    //                                  [4.6],               [5,3,3,4,2,2,1,2,1],              [4.6,5,3,3,4,2,2,1,2,1],
    //                                  [1.9],       =====>  [5,3,5,5,4,4,3,3,2],       ===>   [1.9,5,3,5,5,4,4,3,3,2], 
    //                                  [1.8],               [4,3,3,3,4,5,1,3,3],              [1.8,4,3,3,3,4,5,1,3,3], 
    //                                  [2.7],               [2,2,3,3,1,1,3,4,2],              [2.7,2,2,3,3,1,1,3,4,2],
    //                                  ]                  ]                                 ]
    .concat(possibleRecommendations, 1) // 1 along horizontal axis
    .unstack() // converts to array of tensors
    // arraySync is synchronous; fetch from values from CPU or GPU is an asynchronous
    // pull array and fetching first value with a.arraySync()[0]
    .sort((a,b) => a.arraySync()[0] > b.arraySync()[0] ? 1 : -1)[0]
    .slice(1,9);


   // result is still a tensor here 
  const highestRating = result.max().arraySync(); // highest Rating for uncommon collection for closest neighbor
  console.log('highestRating', highestRating);
  const ratingsInArray = result.arraySync();
  // get index of highest rating to determine which collection to recommend
  
  const index = ratingsInArray.indexOf(highestRating);
  console.log('index', index);
  const collectionToRecommend = potentialCollectionsToRecommend[index];
  console.log('collectionToRecommend', `\x1b[33m${collectionToRecommend}`);
  console.log('\x1b[37m');