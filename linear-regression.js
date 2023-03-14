const tf = require('@tensorflow/tfjs')
const _ = require('lodash')

class LinearRegression {
    constructor(features, labels, options) {
        this.features = tf.tensor(features);
        this.labels = tf.tensor(labels);
        this.features = tf.ones([this.features.shape[0], 1]).concat(this.features, 1)
        this.options = Object.assign({ learningRate: 0.1, iterations: 1000 }, options);

        this.weights =tf.zeros([2,1])
    }

    // gradientDescent() {
    //     const currentGuessesForMPG = this.features.map(row => {
    //         return this.m * row[0] + this.b;
    //     });

    //     const bSlope = _.sum(currentGuessesForMPG.map((guess, i) => {
    //         // inner logic
    //         return guess - this.labels[i][0];
    //     })) * 2 / this.features.length;

    //     const mSlope = _.sum(currentGuessesForMPG.map((guess, i) => {
    //         return -1 * this.features[i][0] * (this.labels[i][0] - guess);
    //     })) * 2 / this.features.length;

    //     this.m = this.m - mSlope * this.options.learningRate;
    //     this.b = this.b - bSlope * this.options.learningRate;
    // }

    gradientDescent() {
        // features * weights
        const currentGuesses = this.features.matMul(this.weights);
        // (features * weights) -labels
        const differences = currentGuesses.sub(this.labels);
        
        const slopes = this.features.transpose()
        .matMul(differences)
        .div(this.features.shape[0])

        // update weights (tensors are immutable)
        this.weights = this.weights.sub(slopes.mul(this.options.learningRate));
    }

    train() {
        for (let i = 0; i < this.options.iterations; i++) {
            this.gradientDescent();
        }
    }
}

// Example of how to instantiate this class
// new LinearRegression(features, labels, {
//     iterations: 99,
//     learningRate: 0.01
// })

module.exports = LinearRegression;