const fs = require('fs');

const sigmoid = value =>
  1 / (1 + Math.E ** -value)

// chain rule component derivatives
const costActDerivative = cost => 2 * cost
const actAccDerivative = accumulation => sigmoidDerivative(accumulation)
const accWeightDerivative = activation => activation
const accPrevActDerivative = weight => weight
const sigmoidDerivative = value => {
  const sig = sigmoid(value)

  return sig * (1 - sig)
}

// cost derivatives
const costWeightDerivative = (cost, accumulation, prevActivation) =>
  accWeightDerivative(prevActivation) * actAccDerivative(accumulation) * costActDerivative(cost)
const actWeightDerivative = (accumulation, prevActivation) =>
  accWeightDerivative(prevActivation) * actAccDerivative(accumulation)
const costBiasDerivative = (cost, accumulation) =>
  acccumulationDerivative(accumulation) * costActDerivative(cost)
const actBiasDerivative = (accumulation) =>
  acccumulationDerivative(accumulation)
const costPrevActivationDerivative = products => {
  // requires cost, accumulation, weight of all nodes in L+1 of the computed
  let total = 0

  products.forEach(product => {
    const {cost, accumulation, weight} = product
    total += accPrevActDerivative(weight) * acccumulationDerivative(accumulation) * costActDerivative(cost)
  })

  return total
}
const actPrevActivationDerivative = products => {
  // requires accumulation, weight of all nodes in L+1 of the computed
  let total = 0

  products.forEach(product => {
    const {accumulation, weight} = product
    total += accPrevActDerivative(weight) * acccumulationDerivative(accumulation)
  })

  return total
}

const calculateCost = (actual, expected) => {
  let error = 0
  actual.forEach(result => {
    error += result - expected
  })

  return error ** 2
}

const propagate = (input, layers, weights, biases) => {
  const activations = []
  const accumulations = []
  layers.forEach((neuronCount, layer) => {
    if (layer === 0) {
      activations[0] = input
      accumulations[0] = input
    } else {
      activations[layer] = []
      accumulations[layer] = []
      for (let neuron = 0; neuron < neuronCount; neuron++) {
        let neuronAccumulation =
          activations[layer-1]
            .map((prevAct, prevNeuron) => {
              return prevAct * weights[layer-1][neuron][prevNeuron]
            })
            .reduce((val, acc) => acc + val, 0)

        neuronAccumulation += biases[layer][neuron]
        accumulations[layer][neuron] = neuronAccumulation
        activations[layer][neuron] = sigmoid(neuronAccumulation)
      }
    }
  })

  return {activations, accumulations}
}

const computeDerivatives = (accumulations, activations, weights, cost) => {
  accumulations.reverse()
  activations.reverse()
  weights.reverse()

  for (var i = 0; i < activations.length-1; i++) {
    console.log(activations[i])
    console.log(weights[i])
  }
  weights.forEach((layerWeights, layer) => {
    layerWeights.forEach((weight, neuron) => {
      costWeightDerivative(cost, accumulations[layer][neuron], activations[layer+1][neuron])
    })
  })
}

const backpropagate = () => null

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
    this.layers = [3, 2, 2]
    this.weights = fillWeights(this.layers)
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
    return propagate(input, this.layers, this.weights, this.biases)
  }

  train(iterations) {
    // Placeholder before implementing dynamic backpropagation
    const set = this.trainingSets[0]
    const {activations, accumulations} = this.think(set.input)
    const cost = calculateCost(activations[activations.length-1], set.output)

    computeDerivatives(accumulations, activations, this.weights, cost)

    console.log(this.think(this.trainingSets[0].input))
  }
}
