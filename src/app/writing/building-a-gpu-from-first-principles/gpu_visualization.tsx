'use client'

import { useEffect, useId, useRef, useState } from 'react'
import { Pause, Play, RotateCcw } from 'lucide-react'

import './gpu_visualization.css'

const delays = [0, 1800, 1400, 1800, 1400, 500]
const input_paths = ['M0 25 C28 25 28 50 56 50', 'M0 75 C28 75 28 50 56 50']

export function GpuHighLevelVisualization() {
  const [phase, set_phase] = useState(1)
  const [playing, set_playing] = useState(true)
  const remaining = useRef(delays[1])

  useEffect(() => {
    if (!playing || phase === 6) return
    const started = performance.now()
    let completed = false
    const timer = setTimeout(() => {
      completed = true
      remaining.current = delays[phase + 1] ?? 0
      set_phase(phase + 1)
    }, remaining.current)
    return () => {
      clearTimeout(timer)
      if (!completed) remaining.current -= performance.now() - started
    }
  }, [phase, playing])

  const is_playing = playing && phase < 6
  const ControlIcon = phase === 6 ? RotateCcw : is_playing ? Pause : Play
  const replay = () => {
    const reduced_motion = matchMedia(
      '(prefers-reduced-motion: reduce)'
    ).matches
    remaining.current = delays[1]
    set_phase(reduced_motion ? 6 : 1)
    set_playing(!reduced_motion)
  }
  const toggle = () => (phase === 6 ? replay() : set_playing(!playing))

  return (
    <figure
      className="gpu-viz group not-prose relative my-12"
      data-paused={!is_playing}
    >
      <div className="grid items-center text-sm md:grid-cols-[1fr_auto_1fr_auto_1fr]">
        <div className="grid grid-rows-2 gap-4 self-stretch">
          <div className="border-border relative flex flex-col justify-center rounded-xl border p-4">
            <Shimmer active={phase === 1} />
            <p className="text-muted-foreground mb-1">Instructions</p>
            <code>C[i] = A[i] + B[i]</code>
          </div>
          <div className="border-border relative flex flex-col justify-center rounded-xl border p-4">
            <Shimmer active={phase === 1} />
            <p className="text-muted-foreground mb-1">Data</p>
            <code>A = [2, 4]</code>
            <br />
            <code>B = [1, 3]</code>
          </div>
        </div>
        <InputArrows active={phase === 2} />
        <button
          onClick={replay}
          onPointerUp={(event) => event.currentTarget.blur()}
          className="border-border bg-muted text-foreground relative flex min-h-32 cursor-pointer items-center justify-center rounded-xl border p-6"
          aria-label="Run the program on the GPU"
        >
          <Shimmer active={phase === 3} />
          <span className="text-lg font-medium">GPU</span>
        </button>
        <Arrow active={phase === 4} />
        <div
          className="border-border relative flex flex-col justify-center rounded-xl border p-4"
          aria-live="polite"
        >
          <p className="text-muted-foreground mb-1">Output</p>
          <code>
            C = [{phase >= 5 ? '3' : '—'}, {phase >= 6 ? '7' : '—'}]
          </code>
        </div>
      </div>
      <button
        onClick={toggle}
        onPointerUp={(event) => event.currentTarget.blur()}
        className={`border-border bg-background/80 text-muted-foreground hover:text-foreground absolute right-2 bottom-2 rounded-full border p-2 backdrop-blur-sm transition-opacity ${is_playing ? 'opacity-0 group-focus-within:opacity-100 group-hover:opacity-100' : 'opacity-100'}`}
        aria-label={
          phase === 6
            ? 'Replay animation'
            : is_playing
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

function Shimmer({ active }: { active: boolean }) {
  return (
    <span
      className="gpu-viz-border text-muted-foreground pointer-events-none absolute"
      data-active={active}
      aria-hidden="true"
    />
  )
}

function Arrow({ active }: { active: boolean }) {
  return (
    <div className="relative mx-auto h-12 w-14 rotate-90 md:h-10 md:rotate-0">
      <svg
        className="text-border size-full"
        viewBox="0 0 56 40"
        aria-hidden="true"
      >
        <Lines paths={['M0 20 C18 20 38 20 56 20']} active={active} />
      </svg>
      <Arrowhead />
    </div>
  )
}

function InputArrows({ active }: { active: boolean }) {
  return (
    <>
      <div className="relative hidden h-full w-14 md:block">
        <svg
          className="text-border size-full"
          viewBox="0 0 56 100"
          preserveAspectRatio="none"
          aria-hidden="true"
        >
          <Lines paths={input_paths} active={active} />
        </svg>
        <Arrowhead />
      </div>
      <div className="md:hidden">
        <Arrow active={active} />
      </div>
    </>
  )
}

function Lines({ paths, active }: { paths: string[]; active: boolean }) {
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
            data-active={active}
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
