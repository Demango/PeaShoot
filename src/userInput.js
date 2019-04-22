const readline = require('readline')

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
})

export const ask = question => {
  return new Promise(resolve => {
    rl.question(`${question} \n`, answer => {
      resolve(answer)

      rl.close()
    })
  })
}
