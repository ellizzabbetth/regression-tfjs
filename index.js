require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs')
const loadCSV = require('./load-csv')
const LinearRegression = require('./linear-regression')
const plot = require('node-remote-plot');


let { features, labels, testFeatures, testLabels } = loadCSV('./cars.csv', {
    shuffle: true,
    splitTest: 50,
    dataColumns: ['horsepower', 'weight', 'displacement'],
    labelColumns: ['mpg']
})

console.log(features, labels)

const regression = new LinearRegression(features, labels, {
    iterations: 100,
    learningRate: 0.1
})

regression.features.print(); 

regression.train();
const r2 = regression.test(testFeatures, testLabels)

plot({
    x: regression.bHistory,
    y: regression.mseHistory.reverse(),
    xLabel: 'Value of B',
    yLabel: 'Mean Square Error'
})

console.log('r2 is: ', r2)
//  console.log('Updated M is: ', regression.weights.get(1,0), "Updated B is: ", regression.weights.get(0,0));