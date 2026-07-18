import assert from 'node:assert/strict'
import test from 'node:test'

import {
  clip,
  compile,
  createEngine,
  parallel,
  sample,
  sequence,
} from './animation.ts'

test('composition is immutable, deterministic, and strictly validated', () => {
  const timeline = compile(
    sequence(parallel(clip('a', 100), clip('b', 200)), clip('c', 100))
  )

  assert.equal(timeline.duration, 300)
  assert.deepEqual(sample(timeline, 50), { a: 0.5, b: 0.25, c: 0 })
  assert.deepEqual(sample(timeline, 250), { a: 1, b: 1, c: 0.5 })
  assert(Object.isFrozen(timeline))
  assert(Object.isFrozen(timeline.clips))
  assert.throws(() => compile(sequence(clip('a'), clip('a'))))
  assert.throws(() => clip('bad', { duration: 1, typo: true } as never))
})

test('runtime samples one absolute clock and honors reduced motion', () => {
  const scene = clip('value', 100)
  const frames: number[] = []
  let now = 0
  let nextFrame: FrameRequestCallback | undefined
  const engine = createEngine(
    scene,
    (_root, frame) => frames.push(frame.value),
    {
      clock: () => now,
      requestFrame: (callback) => {
        nextFrame = callback
        return 1
      },
      cancelFrame: () => {},
      prefersReducedMotion: () => false,
    }
  )

  engine.mount({} as Element)
  now = 50
  nextFrame!(50)
  engine.pause()
  assert.equal(frames.at(-1), 0.5)
  assert.deepEqual(engine.getSnapshot(), {
    status: 'paused',
    time: 50,
    duration: 100,
  })

  const reducedFrames: number[] = []
  const reduced = createEngine(
    scene,
    (_root, frame) => reducedFrames.push(frame.value),
    { prefersReducedMotion: () => true }
  )
  reduced.mount({} as Element)
  assert.equal(reducedFrames.at(-1), 1)
  assert.equal(reduced.getSnapshot().status, 'finished')
  assert.throws(() => createEngine(scene, () => {}, { typo: true } as never))
})
