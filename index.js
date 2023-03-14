require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs')
const loadCSV = require('./load-csv')
const LinearRegression = require('./linear-regression')


let { features, labels, testFeatures, testLabels } = loadCSV('./cars.csv', {
    shuffle: true,
    splitTest: 50,
    dataColumns: ['horsepower'],
    labelColumns: ['mpg']
})

console.log(features, labels)

const regression =  new LinearRegression(features, labels, {
     iterations: 1,
     learningRate: 0.001
 })

 regression.train();

 console.log('Updated M is: ', regression.m, "Updated B is: ", regression.b);