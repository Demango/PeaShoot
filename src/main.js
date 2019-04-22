const fs = require('fs');

const sigmoid = value =>
  1 / (1 + Math.E ** -value)

// sigmoid function has a larger slope as it approaches 1 or 0
// this can be interpreted as a "confidence"
const sigmoidDerivative = value => {
  const sig = sigmoid(value)

  return sig * (1 - sig)
}

const propagate = (input, layers, weights, biases) => {
  const activations = []
  layers.forEach((neuronCount, layer) => {
    if (layer === 0) {
      activations[0] = input
    } else {
      activations[layer] = []
      for (let neuron = 0; neuron < neuronCount; neuron++) {
        let neuronAccumulation =
          activations[layer-1]
            .map((prevAct, prevNeuron) => {
              return prevAct * weights[layer-1][neuron][prevNeuron]
            })
            .reduce((val, acc) => acc + val, 0)

        neuronAccumulation += biases[layer][neuron]
        activations[layer][neuron] = sigmoid(neuronAccumulation)
      }
    }
  })

  return activations
}

const fillWeights = layers => {
  const weights = []

  for (var i = 0; i < layers.length - 1; i++) {
    weights[i] = Array(layers[i+1]).fill()
    weights[i].forEach((v, j) => {
      weights[i][j] = Array(layers[i])
        .fill()
        .map(() => Math.random()*2-1)
    })
  }

  return weights
}

const fillBiases = layers => {
  const biases = []

  for (var i = 0; i < layers.length; i++) {
    biases[i] = Array(layers[i]).fill(0)
  }

  return biases
}

export default class Network {
  constructor() {
    this.layers = [3, 2, 1]
    this.newWeights = fillWeights(this.layers)
    this.biases = fillBiases(this.layers)

    this.trainingSets = [
      {
        input: [0, 0, 1],
        output: [0]
      },
      {
        input: [1, 1, 1],
        output: [1]
      },
      {
        input: [1, 0, 1],
        output: [1]
      },
      {
        input: [0, 1, 1],
        output: [0]
      }
    ]
  }

  think(input) {
    return propagate(input, this.layers, this.newWeights, this.biases)
  }

  train(iterations) {
    // Placeholder before implementing dynamic backpropagation

    console.log(this.think(this.trainingSets[0].input))
  }
}
