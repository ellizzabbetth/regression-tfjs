const tf = require('@tensorflow/tfjs')

class LinearRegression {
    constructor(features, labels, options) {
        this.features = features;
        this.labels = labels;

        this.options = Objects.assign({ learningRate: 0.1 }, options);
    }


}

// new LinearRegression(features, labels, {
//     iterations: 99,
//     learningRate: 0.01
// })

module.exports = LinearRegression;