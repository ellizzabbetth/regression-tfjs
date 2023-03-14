const tf = require('@tensorflow/tfjs')
const _ = require('lodash')

class LinearRegression {
    constructor(features, labels, options) {
        this.features = features;
        this.labels = labels;

        this.options = Objects.assign({ learningRate: 0.1, iterations: 1000 }, options);

        this.m = 0;
        this.b = 0;
    }

    gradientDescent() {
        const currentGuessesForMPG = this.features.map(row => {
            return this.m * row[0] + this.b;
        });

        const bSlope = _.sum(currentGuessesForMPG.map((guess, i) => {
            // inner logic
            return guess - this.labels[i][0];
        })) * 2 / this.features.length;

        const mSlope = _sum(currentGuessesForMPG.map((guess, i) => {
            return -1 * this.features[i][0] * (this.labels[i][0] - guess);
        })) * 2 / this.features.length;
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