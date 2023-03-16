const tf = require('@tensorflow/tfjs')
const _ = require('lodash')

class LogisticRegression {
    constructor(features, labels, options) {
        this.features = this.processFeatures(features);// tf.tensor(features);
        this.labels = tf.tensor(labels);
        this.mseHistory = [];
       // this.features = tf.ones([this.features.shape[0], 1]).concat(this.features, 1)
        this.options = Object.assign({ learningRate: 0.1, iterations: 1000 }, options);

        this.weights = tf.zeros([this.features.shape[1], 1])
    }

    gradientDescent(features, labels) {
        // sigmoid(features * weights)
        const currentGuesses = features.matMul(this.weights).sigmoid();
        // (features * weights) -labels
        const differences = currentGuesses.sub(labels);

        const slopes = features.transpose()
            .matMul(differences)
            .div(this.features.shape[0])

        // update weights (tensors are immutable)
        this.weights = this.weights.sub(slopes.mul(this.options.learningRate));
    }

    train() {
        const batchQuantity = Math.floor(this.features.shape[0] / this.options.batchSize);

        for (let i = 0; i < this.options.iterations; i++) {
            for (let j = 0; j < batchQuantity; j++) {
                const startIndex = j * this.options.batchSize;
                const { batchSize } = this.options;
                const featureSlice = this.features.slice([startIndex, 0], [batchSize, -1]);
                const labelSlice =  this.labels.slice([startIndex, 0], [batchSize, -1])
                this.gradientDescent(featureSlice, labelSlice);

            }
            this.recordMSE();
            this.updateLearningRate();
        }
    }

    predict(observations) {
        // [
        //     [horsepower, weights, displacement]            
        // ]
        return this.processFeatures(observations).matMul(this.weights).sigmoid();
    }

    test(testFeatures, testLabels) {
        testFeatures =  this.processFeatures(testFeatures);//tf.tensor(testFeatures);
        testLabels = tf.tensor(testLabels);

       // testFeatures = tf.ones([testFeatures.shape[0], 1]).concat(testFeatures, 1);

        const predictions = testFeatures.matMul(this.weights)
        const res = testLabels.sub(predictions)
            .pow(2)
            .sum()
            .get();

        const tot = testLabels.sub(testLabels.mean())
            .pow(2)
            .sum()
            .get();

        return 1 - res / tot;
    }

    processFeatures(features) {
        features = tf.tensor(features);
        
        if (this.mean && this.variance) {
            features = features.sub(this.mean).div(this.variance.pow(0.5));
        } else {
            features = this.standardize(features);
        }
        features = tf.ones([features.shape[0], 1]).concat(features, 1);
        return features;
    }

    standardize(features) {
        const { mean, variance } = tf.moments(features, 0);

        this.mean = mean;
        this.variance = variance;

        return features.sub(mean).div(variance.pow(0.5));
    }

    recordMSE() {
        const mse = this.features
        .matMul(this.weights)
        .sub(this.labels)
        .pow(2)
        .sum()
        .div(this.features.shape[0])
        .get();
        this.mseHistory.unshift(mse);
    }

    updateLearningRate() {
        if(this.mseHistory.length < 2) {
            return;
        }

        if (this.mseHistory[0] > this.mseHistory[1]) {
            this.options.learningRate /= 2;
        } else {
            this.options.learningRate *= 1.05;
        }
        // const lastValue = this.mseHistory[this.mseHistory.length - 1];
        // const secondLast  = this.mseHistory[this.mseHistory.length - 2];
    }
}



// Example of how to instantiate this class
// new LinearRegression(features, labels, {
//     iterations: 99,
//     learningRate: 0.01
// })

module.exports = LogisticRegression;