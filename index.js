import 'babel-polyfill'

import { ask } from './userInput'
import Network from './main.js'

const net = new Network()

async function collectInput() {
  const ans = await ask('Start?')
  // Placeholder before implementing dynamic backpropagation
  net.train(0)

  process.exit()
}

collectInput()

