import { z } from 'zod'

export type Ease = (progress: number) => number

const linear: Ease = (progress) => progress
const easeSchema = z.custom<Ease>(
  (value): value is Ease => typeof value === 'function'
)
const durationSchema = z.number().finite().nonnegative()
const clipOptionsSchema = z
  .strictObject({
    duration: durationSchema.default(1000),
    ease: easeSchema.default(() => linear),
  })
  .readonly()

export type Clip<Name extends string = string> = Readonly<{
  kind: 'clip'
  name: Name
  duration: number
  ease: Ease
}>

export type Scene =
  | Clip
  | Readonly<{
      kind: 'sequence' | 'parallel'
      children: readonly Scene[]
    }>

export type Group<Children extends readonly Scene[] = readonly Scene[]> =
  Readonly<{
    kind: 'sequence' | 'parallel'
    children: Children
  }>
type Children = readonly [Scene, ...Scene[]]
type Names<S> = Scene extends S
  ? string
  : S extends Clip<infer Name>
    ? Name
    : S extends { children: infer Items extends readonly Scene[] }
      ? Names<Items[number]>
      : never

const clipSchema = z
  .strictObject({
    kind: z.literal('clip'),
    name: z.string().min(1),
    duration: durationSchema,
    ease: easeSchema,
  })
  .readonly()

export const sceneSchema: z.ZodType<Scene> = z.lazy(() =>
  z.union([
    clipSchema,
    z
      .strictObject({
        kind: z.enum(['sequence', 'parallel']),
        children: z.array(sceneSchema).min(1).readonly(),
      })
      .readonly(),
  ])
)

export function clip<const Name extends string>(
  name: Name,
  options: number | z.input<typeof clipOptionsSchema> = {}
): Clip<Name> {
  const parsed = clipOptionsSchema.parse(
    typeof options === 'number' ? { duration: options } : options
  )
  return clipSchema.parse({ kind: 'clip', name, ...parsed }) as Clip<Name>
}

export function sequence<const Items extends Children>(
  ...children: Items
): Group<Items> {
  return sceneSchema.parse({ kind: 'sequence', children }) as Group<Items>
}

export function parallel<const Items extends Children>(
  ...children: Items
): Group<Items> {
  return sceneSchema.parse({ kind: 'parallel', children }) as Group<Items>
}

type CompiledClip = Readonly<Clip & { start: number }>

export type Timeline<S extends Scene = Scene> = Readonly<{
  duration: number
  clips: readonly CompiledClip[]
  scene: S
}>

export type Frame<S extends Scene> = Readonly<Record<Names<S>, number>>

function compileNode(node: Scene, start: number) {
  if (node.kind === 'clip') {
    return {
      clips: [Object.freeze({ ...node, start })],
      duration: node.duration,
    }
  }

  let duration = 0
  const clips: CompiledClip[] = []
  for (const child of node.children) {
    const offset = node.kind === 'sequence' ? duration : 0
    const compiled = compileNode(child, start + offset)
    clips.push(...compiled.clips)
    duration =
      node.kind === 'sequence'
        ? duration + compiled.duration
        : Math.max(duration, compiled.duration)
  }
  return { clips, duration }
}

export function compile<const S extends Scene>(scene: S): Timeline<S> {
  const parsed = sceneSchema.parse(scene) as S
  const compiled = compileNode(parsed, 0)
  const names = new Set<string>()
  for (const item of compiled.clips) {
    if (names.has(item.name)) throw new Error(`Duplicate clip: ${item.name}`)
    names.add(item.name)
  }
  return Object.freeze({
    ...compiled,
    clips: Object.freeze(compiled.clips),
    scene: parsed,
  })
}

export function sample<const S extends Scene>(
  timeline: Timeline<S>,
  time: number
): Frame<S> {
  const values: Record<string, number> = {}
  for (const item of timeline.clips) {
    const progress =
      time <= item.start
        ? 0
        : time >= item.start + item.duration || item.duration === 0
          ? 1
          : item.ease((time - item.start) / item.duration)
    if (!Number.isFinite(progress)) {
      throw new Error(`Clip ${item.name} produced non-finite progress`)
    }
    values[item.name] = progress
  }
  return Object.freeze(values) as Frame<S>
}

export const engineOptionsSchema = z
  .strictObject({
    autoplay: z.boolean().default(true),
    playbackRate: z.number().finite().positive().default(1),
    reducedMotion: z.enum(['finish', 'pause', 'ignore']).default('finish'),
    notifyOnFrame: z.boolean().default(false),
    clock: z
      .custom<() => number>((value) => typeof value === 'function')
      .default(() => () => performance.now()),
    requestFrame: z
      .custom<
        (callback: FrameRequestCallback) => number
      >((value) => typeof value === 'function')
      .default(
        () => (callback: FrameRequestCallback) =>
          requestAnimationFrame(callback)
      ),
    cancelFrame: z
      .custom<(id: number) => void>((value) => typeof value === 'function')
      .default(() => (id: number) => cancelAnimationFrame(id)),
    prefersReducedMotion: z
      .custom<
        (root: Element) => boolean
      >((value) => typeof value === 'function')
      .default(
        () => (root: Element) =>
          root.ownerDocument.defaultView?.matchMedia(
            '(prefers-reduced-motion: reduce)'
          ).matches ?? false
      ),
  })
  .readonly()

export type EngineOptions = z.input<typeof engineOptionsSchema>
export type PlaybackStatus = 'idle' | 'playing' | 'paused' | 'finished'
export type Snapshot = Readonly<{
  status: PlaybackStatus
  time: number
  duration: number
}>
export type Renderer<S extends Scene> = (root: Element, frame: Frame<S>) => void

const snapshotSchema = z
  .strictObject({
    status: z.enum(['idle', 'playing', 'paused', 'finished']),
    time: durationSchema,
    duration: durationSchema,
  })
  .readonly()

type Playback = Readonly<{
  status: PlaybackStatus
  time: number
  startedAt: number
  rate: number
}>

export function createEngine<const S extends Scene>(
  scene: S,
  render: Renderer<S>,
  input: EngineOptions = {}
) {
  const timeline = compile(scene)
  const options = engineOptionsSchema.parse(input)
  z.custom<Renderer<S>>((value) => typeof value === 'function').parse(render)

  const listeners = new Set<() => void>()
  const serverSnapshot = snapshotSchema.parse({
    status: 'idle',
    time: 0,
    duration: timeline.duration,
  })
  let snapshot: Snapshot = serverSnapshot
  let playback: Playback = Object.freeze({
    status: 'idle',
    time: 0,
    startedAt: 0,
    rate: options.playbackRate,
  })
  let root: Element | null = null
  let frameId: number | null = null

  const timeAt = (now: number) =>
    Math.min(
      timeline.duration,
      playback.status === 'playing' && root
        ? playback.time + (now - playback.startedAt) * playback.rate
        : playback.time
    )

  const emit = (status = playback.status, time = playback.time) => {
    if (snapshot.status === status && snapshot.time === time) return
    snapshot = snapshotSchema.parse({
      status,
      time,
      duration: timeline.duration,
    })
    listeners.forEach((listener) => listener())
  }

  const renderAt = (time: number) => {
    if (root) render(root, sample(timeline, time))
  }

  const cancel = () => {
    if (frameId === null) return
    options.cancelFrame(frameId)
    frameId = null
  }

  const finish = () => {
    cancel()
    playback = Object.freeze({
      ...playback,
      status: 'finished',
      time: timeline.duration,
    })
    renderAt(timeline.duration)
    emit()
  }

  const tick = (now: number) => {
    frameId = null
    if (playback.status !== 'playing' || !root) return
    const time = timeAt(now)
    if (time >= timeline.duration) return finish()
    renderAt(time)
    if (options.notifyOnFrame) emit('playing', time)
    frameId = options.requestFrame(tick)
  }

  const schedule = () => {
    if (frameId === null && root && playback.status === 'playing') {
      frameId = options.requestFrame(tick)
    }
  }

  const play = () => {
    if (playback.status === 'playing') return
    const now = options.clock()
    const time = playback.status === 'finished' ? 0 : playback.time
    playback = Object.freeze({
      ...playback,
      status: 'playing',
      time,
      startedAt: now,
    })
    renderAt(time)
    emit()
    schedule()
  }

  const pause = () => {
    if (playback.status !== 'playing') return
    const time = timeAt(options.clock())
    cancel()
    playback = Object.freeze({ ...playback, status: 'paused', time })
    renderAt(time)
    emit()
  }

  const replay = () => {
    cancel()
    if (
      root &&
      options.reducedMotion !== 'ignore' &&
      options.prefersReducedMotion(root)
    ) {
      playback = Object.freeze({
        ...playback,
        status: options.reducedMotion === 'finish' ? 'finished' : 'paused',
        time: options.reducedMotion === 'finish' ? timeline.duration : 0,
      })
      renderAt(playback.time)
      emit()
      return
    }
    playback = Object.freeze({
      ...playback,
      status: 'playing',
      time: 0,
      startedAt: options.clock(),
    })
    renderAt(0)
    emit()
    schedule()
  }

  const seek = (time: number) => {
    const next = durationSchema.parse(time)
    const clamped = Math.min(next, timeline.duration)
    const playing = playback.status === 'playing' && clamped < timeline.duration
    cancel()
    playback = Object.freeze({
      ...playback,
      status:
        clamped === timeline.duration
          ? 'finished'
          : playing
            ? 'playing'
            : 'paused',
      time: clamped,
      startedAt: options.clock(),
    })
    renderAt(clamped)
    emit()
    schedule()
  }

  const setPlaybackRate = (rate: number) => {
    const next = z.number().finite().positive().parse(rate)
    const now = options.clock()
    playback = Object.freeze({
      ...playback,
      time: timeAt(now),
      startedAt: now,
      rate: next,
    })
    emit()
  }

  const toggle = () =>
    playback.status === 'playing'
      ? pause()
      : playback.status === 'finished'
        ? replay()
        : play()

  const detach = () => {
    const now = options.clock()
    playback = Object.freeze({ ...playback, time: timeAt(now), startedAt: now })
    cancel()
    root = null
  }

  const mount = (nextRoot: Element | null) => {
    if (!nextRoot) return detach()
    if (root && root !== nextRoot) detach()
    root = nextRoot

    if (playback.status === 'idle') {
      renderAt(0)
      if (options.autoplay) replay()
      else {
        playback = Object.freeze({ ...playback, status: 'paused' })
        emit()
      }
    } else if (playback.status === 'playing') {
      playback = Object.freeze({ ...playback, startedAt: options.clock() })
      renderAt(playback.time)
      schedule()
    } else renderAt(playback.time)

    return () => {
      if (root === nextRoot) detach()
    }
  }

  return Object.freeze({
    mount,
    play,
    pause,
    replay,
    seek,
    setPlaybackRate,
    toggle,
    getSnapshot: () => snapshot,
    getServerSnapshot: () => serverSnapshot,
    subscribe: (listener: () => void) => {
      listeners.add(listener)
      return () => {
        listeners.delete(listener)
      }
    },
  })
}
