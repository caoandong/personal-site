'use client'

import { type ReactNode, useRef, useSyncExternalStore } from 'react'
import { Pause, Play, RotateCcw } from 'lucide-react'

import {
  clip,
  createEngine,
  parallel,
  sequence,
  type Frame,
} from '@/lib/animation'

import './gpu_visualization.css'

const scene = sequence(
  clip('inputs', 900),
  clip('grid', 800),
  parallel(clip('thread0', 1400), clip('thread1', 1400)),
  clip('output', 600)
)

type AnimationName = keyof Frame<typeof scene>

function renderFrame(root: Element, frame: Frame<typeof scene>) {
  const highlight = (name: AnimationName) => {
    const progress = frame[name]
    root
      .querySelectorAll<HTMLElement>(`[data-animation="${name}"]`)
      .forEach((element) => {
        element.style.setProperty('--gpu-viz-offset', `${progress * 100}%`)
        element.style.setProperty(
          '--gpu-viz-intensity',
          `${Math.sin(Math.PI * progress) * 0.6}`
        )
      })
  }

  highlight('inputs')
  highlight('grid')
  highlight('thread0')
  highlight('thread1')
  highlight('output')

  root.querySelector<HTMLElement>('[data-thread-result="0"]')!.textContent =
    frame.thread0 > 0.6 ? '3' : '—'
  root.querySelector<HTMLElement>('[data-thread-result="1"]')!.textContent =
    frame.thread1 > 0.6 ? '7' : '—'
  root.querySelector<HTMLElement>('[data-grid-output="0"]')!.textContent =
    frame.output > 0 ? '3' : '—'
  root.querySelector<HTMLElement>('[data-grid-output="1"]')!.textContent =
    frame.output > 0 ? '7' : '—'
}

const createGridEngine = () => createEngine(scene, renderFrame)
type GridEngine = ReturnType<typeof createGridEngine>

const threads = [
  { index: 0, a: 2, b: 1 },
  { index: 1, a: 4, b: 3 },
] as const

export function GridVisualization() {
  const engineRef = useRef<GridEngine | null>(null)
  engineRef.current ??= createGridEngine()
  const engine = engineRef.current
  const { status } = useSyncExternalStore(
    engine.subscribe,
    engine.getSnapshot,
    engine.getServerSnapshot
  )
  const isPlaying = status === 'playing'
  const ControlIcon =
    status === 'finished' ? RotateCcw : isPlaying ? Pause : Play

  return (
    <figure ref={engine.mount} className="not-prose my-12 text-sm">
      <figcaption className="sr-only">
        One CUDA grid containing one block with two threads. Each thread adds
        one pair of input values.
      </figcaption>

      <div className="mb-6 grid grid-cols-2 gap-3">
        <ValueBox label="Input A" value="A = [2, 4]" />
        <ValueBox label="Input B" value="B = [1, 3]" />
      </div>

      <div className="border-border relative rounded-2xl border p-4 pt-7 md:p-6 md:pt-8">
        <Shimmer name="grid" />
        <BoundaryLabel>Grid · 1 block</BoundaryLabel>

        <div className="border-border relative rounded-xl border p-4 pt-7 md:p-5 md:pt-8">
          <Shimmer name="grid" />
          <BoundaryLabel>Block 0 · 2 threads</BoundaryLabel>

          <div className="grid gap-3 md:grid-cols-2">
            {threads.map(({ index, a, b }) => (
              <div
                key={index}
                className="border-border relative rounded-lg border p-4"
              >
                <Shimmer name={`thread${index}` as 'thread0' | 'thread1'} />
                <div className="mb-4 flex items-baseline justify-between gap-3">
                  <p className="font-medium">Thread {index}</p>
                  <code className="text-muted-foreground">i = {index}</code>
                </div>
                <div className="grid gap-1 font-mono">
                  <span>
                    C[{index}] = A[{index}] + B[{index}]
                  </span>
                  <span className="text-muted-foreground">
                    &nbsp;&nbsp;&nbsp;&nbsp; = {a} + {b}
                  </span>
                  <span>
                    &nbsp;&nbsp;&nbsp;&nbsp; ={' '}
                    <span data-thread-result={index}>—</span>
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="mt-6 flex items-center justify-between gap-3">
        <div
          className="border-border relative rounded-xl border px-4 py-3"
          aria-live="polite"
        >
          <Shimmer name="output" />
          <span className="text-muted-foreground mr-2">Output</span>
          <code>
            C = [<span data-grid-output="0">—</span>,{' '}
            <span data-grid-output="1">—</span>]
          </code>
        </div>

        <button
          onClick={engine.toggle}
          onPointerUp={(event) => event.currentTarget.blur()}
          className="border-border bg-background text-muted-foreground hover:text-foreground shrink-0 cursor-pointer rounded-full border p-2"
          aria-label={
            status === 'finished'
              ? 'Replay animation'
              : isPlaying
                ? 'Pause animation'
                : 'Play animation'
          }
        >
          <ControlIcon
            className="size-3.5"
            strokeWidth={1.5}
            aria-hidden="true"
          />
        </button>
      </div>
    </figure>
  )
}

function ValueBox({ label, value }: { label: string; value: string }) {
  return (
    <div className="border-border relative rounded-xl border p-4">
      <Shimmer name="inputs" />
      <p className="text-muted-foreground mb-1">{label}</p>
      <code>{value}</code>
    </div>
  )
}

function BoundaryLabel({ children }: { children: ReactNode }) {
  return (
    <p className="bg-background text-muted-foreground absolute -top-2.5 left-4 px-2">
      {children}
    </p>
  )
}

function Shimmer({ name }: { name: AnimationName }) {
  return (
    <span
      className="gpu-viz-border text-muted-foreground pointer-events-none absolute"
      data-animation={name}
      aria-hidden="true"
    />
  )
}
