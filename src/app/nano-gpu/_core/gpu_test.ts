import { strict as assert } from 'node:assert'
import { Gpu, helloProgram } from './gpu'

const gpu = new Gpu({
  smCount: 2,
  warpSize: 4,
  registersPerLane: 4,
  sharedMemoryWords: 4,
  hbmWords: 16,
  l2Words: 16,
})

gpu.launch(helloProgram, 4)
const result = gpu.run()

assert.deepEqual(
  gpu.hbm.snapshot(),
  Array.from({ length: 16 }, (_, i) => i)
)
assert.deepEqual(gpu.sms[0].sharedMemory.snapshot(), [8, 9, 10, 11])
assert.deepEqual(gpu.sms[1].sharedMemory.snapshot(), [12, 13, 14, 15])
assert.equal(gpu.l2.hits, 16)
assert.equal(gpu.l2.misses, 0)
assert.equal(result.cycles, 16)

console.log('nano_gpu self-check passed:', {
  cycles: result.cycles,
  hbm: gpu.hbm.snapshot(),
  l2Hits: gpu.l2.hits,
})
