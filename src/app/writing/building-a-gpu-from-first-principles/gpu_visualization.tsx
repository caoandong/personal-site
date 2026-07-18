'use client'

import { useId, useRef, useSyncExternalStore } from 'react'
import { Pause, Play, RotateCcw } from 'lucide-react'

import { clip, createEngine, sequence, type Frame } from '@/lib/animation'

import './gpu_visualization.css'

const easeInOut = (progress: number) =>
  progress < 0.5 ? 4 * progress ** 3 : 1 - Math.pow(-2 * progress + 2, 3) / 2

const scene = sequence(
  clip('inputs', { duration: 1800, ease: easeInOut }),
  clip('inputFlow', 1400),
  clip('gpu', { duration: 1800, ease: easeInOut }),
  clip('outputFlow', 1400),
  clip('output', 500)
)

const inputPaths = ['M0 25 C28 25 28 50 56 50', 'M0 75 C28 75 28 50 56 50']
type AnimationName = keyof Frame<typeof scene>

function renderFrame(root: Element, frame: Frame<typeof scene>) {
  const border = (name: 'inputs' | 'gpu') => {
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
  const flow = (name: 'inputFlow' | 'outputFlow') => {
    const progress = frame[name]
    root
      .querySelectorAll<SVGRectElement>(`[data-animation="${name}"]`)
      .forEach((element) => {
        element.style.setProperty('--gpu-viz-x', `${progress * 80}px`)
        element.style.setProperty(
          '--gpu-viz-flow-opacity',
          `${Math.sin(Math.PI * progress)}`
        )
      })
  }

  border('inputs')
  flow('inputFlow')
  border('gpu')
  flow('outputFlow')
  root.querySelector<HTMLElement>('[data-output="0"]')!.textContent =
    frame.output > 0 ? '3' : '—'
  root.querySelector<HTMLElement>('[data-output="1"]')!.textContent =
    frame.output >= 1 ? '7' : '—'
}

const createGpuEngine = () => createEngine(scene, renderFrame)
type GpuEngine = ReturnType<typeof createGpuEngine>

export function GpuHighLevelVisualization() {
  const engineRef = useRef<GpuEngine | null>(null)
  engineRef.current ??= createGpuEngine()
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
    <figure
      ref={engine.mount}
      className="gpu-viz group not-prose relative my-12"
    >
      <div className="grid items-center text-sm md:grid-cols-[1fr_auto_1fr_auto_1fr]">
        <div className="grid grid-rows-2 gap-4 self-stretch">
          <div className="border-border relative flex flex-col justify-center rounded-xl border p-4">
            <Shimmer name="inputs" />
            <p className="text-muted-foreground mb-1">Instructions</p>
            <code>C[i] = A[i] + B[i]</code>
          </div>
          <div className="border-border relative flex flex-col justify-center rounded-xl border p-4">
            <Shimmer name="inputs" />
            <p className="text-muted-foreground mb-1">Data</p>
            <code>A = [2, 4]</code>
            <br />
            <code>B = [1, 3]</code>
          </div>
        </div>
        <InputArrows name="inputFlow" />
        <button
          onClick={engine.replay}
          onPointerUp={(event) => event.currentTarget.blur()}
          className="border-border bg-muted text-foreground relative flex min-h-32 cursor-pointer items-center justify-center rounded-xl border p-6"
          aria-label="Run the program on the GPU"
        >
          <Shimmer name="gpu" />
          <span className="text-lg font-medium">GPU</span>
        </button>
        <Arrow name="outputFlow" />
        <div
          className="border-border relative flex flex-col justify-center rounded-xl border p-4"
          aria-live="polite"
        >
          <p className="text-muted-foreground mb-1">Output</p>
          <code>
            C = [<span data-output="0">—</span>, <span data-output="1">—</span>]
          </code>
        </div>
      </div>
      <button
        onClick={engine.toggle}
        onPointerUp={(event) => event.currentTarget.blur()}
        className={`border-border bg-background/80 text-muted-foreground hover:text-foreground absolute right-2 bottom-2 rounded-full border p-2 backdrop-blur-sm transition-opacity ${isPlaying ? 'opacity-0 group-focus-within:opacity-100 group-hover:opacity-100' : 'opacity-100'}`}
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
    </figure>
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

function Arrow({ name }: { name: AnimationName }) {
  return (
    <div className="relative mx-auto h-12 w-14 rotate-90 md:h-10 md:rotate-0">
      <svg
        className="text-border size-full"
        viewBox="0 0 56 40"
        aria-hidden="true"
      >
        <Lines paths={['M0 20 C18 20 38 20 56 20']} name={name} />
      </svg>
      <Arrowhead />
    </div>
  )
}

function InputArrows({ name }: { name: AnimationName }) {
  return (
    <>
      <div className="relative hidden h-full w-14 md:block">
        <svg
          className="text-border size-full"
          viewBox="0 0 56 100"
          preserveAspectRatio="none"
          aria-hidden="true"
        >
          <Lines paths={inputPaths} name={name} />
        </svg>
        <Arrowhead />
      </div>
      <div className="md:hidden">
        <Arrow name={name} />
      </div>
    </>
  )
}

function Lines({ paths, name }: { paths: string[]; name: AnimationName }) {
  const id = useId().replaceAll(':', '')

  return (
    <>
      <defs>
        <linearGradient id={`${id}-gradient`}>
          <stop offset="0" stopColor="currentColor" stopOpacity="0" />
          <stop offset=".35" stopColor="currentColor" stopOpacity=".08" />
          <stop offset=".65" stopColor="currentColor" stopOpacity=".35" />
          <stop offset=".82" stopColor="currentColor" />
          <stop offset="1" stopColor="currentColor" stopOpacity="0" />
        </linearGradient>
        <mask
          id={`${id}-mask`}
          x="-24"
          y="0"
          width="104"
          height="100"
          maskUnits="userSpaceOnUse"
          className="gpu-viz-mask"
        >
          <rect
            x="-24"
            width="24"
            height="100"
            fill={`url(#${id}-gradient)`}
            className="gpu-viz-flow-mask"
            data-animation={name}
          />
        </mask>
      </defs>
      {paths.map((path) => (
        <g key={path}>
          <path
            d={path}
            className="fill-none stroke-current [stroke-width:1.25] [vector-effect:non-scaling-stroke]"
          />
          <path d={path} mask={`url(#${id}-mask)`} className="gpu-viz-flow" />
        </g>
      ))}
    </>
  )
}

function Arrowhead() {
  return (
    <svg
      className="text-border absolute top-1/2 right-0 h-2 w-[5px] -translate-y-1/2 overflow-visible"
      viewBox="-5 -4 5 8"
      aria-hidden="true"
    >
      <polyline
        points="-5,-4 0,0 -5,4"
        className="fill-none stroke-current [stroke-width:1.25] [stroke-linecap:round] [stroke-linejoin:round] [vector-effect:non-scaling-stroke]"
      />
    </svg>
  )
}
