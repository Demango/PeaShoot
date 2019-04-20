const sigmoid = value =>
  1 / (1 + Math.E ** -value)

const activation = (values, weights) =>
  sigmoid(
    values
      .map((value, i) => value * weights[i])
      .reduce((value, acc) => value + acc, 0)
  )


// Sigmoid function has a larger slope as it approaches 1 or 0
const sigmoidDerivative = value => {
  const sig = sigmoid(value)

  return sig * (1 - sig)
}


class Network {
  constructor() {
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

    // random values between -1 and 1
    this.weights = Array(3).fill().map(() => Math.random()*2-1)
  }

  think(input) {
    return activation(input, this.weights)
  }

  train(iterations) {
    for (let iter = 0; iter < iterations; iter++) {
      this.trainingSets.forEach(set => {
        const result = this.think(set.input)
        const error = set.output[0] - result
        const adjustments = Array(3).fill(0)

        // Interpret errors as adjustment vectors to be summed
        // Makes us able to apply them simultaneously

        this.weights.forEach((weight, i) => {
          adjustments[i] += sigmoidDerivative(result) * error * set.input[i]
        })

        adjustments.forEach((adjustment, i) => {this.weights[i] += adjustment})
      })
    }


    // Display results of training
    this.trainingSets.forEach(set => {
      const result = this.think(set.input)
      const error = set.output[0] - result

      console.log(result.toFixed(3), set.output[0], error.toFixed(2))

      this.weights.forEach((weight, i) => {
        this.weights[i] += sigmoidDerivative(result) * error * set.input[i]
      })
    })
  }

}

const net = new Network()

net.train(1000)
