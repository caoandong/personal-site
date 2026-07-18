'use client'

import { useEffect, useState } from 'react'

const delays = [0, 1000, 800, 1000, 800, 1000, 350]
const input_paths = ['M0 25 C28 25 28 50 56 50', 'M0 75 C28 75 28 50 56 50']

export function GpuHighLevelVisualization() {
  const [phase, set_phase] = useState(0)

  useEffect(() => {
    if (!phase || phase === 7) return
    const timer = setTimeout(() => set_phase(phase + 1), delays[phase])
    return () => clearTimeout(timer)
  }, [phase])

  const run = () =>
    set_phase(matchMedia('(prefers-reduced-motion: reduce)').matches ? 7 : 1)

  return (
    <figure className="gpu-viz not-prose my-12">
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
          onClick={run}
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
          <Shimmer active={phase === 5} />
          <p className="text-muted-foreground mb-1">Output</p>
          <code>
            C = [<Value active={phase === 6}>{phase >= 6 ? '3' : '—'}</Value>,{' '}
            <Value active={phase === 7}>{phase >= 7 ? '7' : '—'}</Value>]
          </code>
        </div>
      </div>
    </figure>
  )
}

function Shimmer({ active }: { active: boolean }) {
  return (
    <svg
      className="text-foreground pointer-events-none absolute inset-0 size-full overflow-visible"
      aria-hidden="true"
    >
      <rect
        x="1"
        y="1"
        width="calc(100% - 2px)"
        height="calc(100% - 2px)"
        rx="11"
        pathLength="1"
        className="gpu-viz-border"
        data-active={active}
      />
    </svg>
  )
}

function Value({
  active,
  children,
}: {
  active: boolean
  children: React.ReactNode
}) {
  return <span className={active ? 'gpu-viz-value' : ''}>{children}</span>
}

function Arrow({ active }: { active: boolean }) {
  return (
    <div className="relative mx-auto h-12 w-14 rotate-90 md:h-10 md:rotate-0">
      <svg
        className="text-muted-foreground/70 size-full"
        viewBox="0 0 56 40"
        aria-hidden="true"
      >
        <Lines paths={['M0 20 C18 20 38 20 56 20']} active={active} />
      </svg>
      <Arrowhead active={active} />
    </div>
  )
}

function InputArrows({ active }: { active: boolean }) {
  return (
    <>
      <div className="relative hidden h-full w-14 md:block">
        <svg
          className="text-muted-foreground/70 size-full"
          viewBox="0 0 56 100"
          preserveAspectRatio="none"
          aria-hidden="true"
        >
          <Lines paths={input_paths} active={active} />
        </svg>
        <Arrowhead active={active} />
      </div>
      <div className="md:hidden">
        <Arrow active={active} />
      </div>
    </>
  )
}

function Lines({ paths, active }: { paths: string[]; active: boolean }) {
  return paths.map((path) => (
    <g key={path}>
      <path
        d={path}
        className="fill-none stroke-current [stroke-width:1.25] [vector-effect:non-scaling-stroke]"
      />
      <path
        d={path}
        pathLength="1"
        className="gpu-viz-flow"
        data-active={active}
      />
    </g>
  ))
}

function Arrowhead({ active }: { active: boolean }) {
  return (
    <svg
      className="text-muted-foreground/70 absolute top-1/2 right-0 h-2 w-[5px] -translate-y-1/2 overflow-visible"
      viewBox="-5 -4 5 8"
      aria-hidden="true"
    >
      <polyline
        points="-5,-4 0,0 -5,4"
        className="gpu-viz-head fill-none stroke-current [stroke-width:1.25] [stroke-linecap:round] [stroke-linejoin:round] [vector-effect:non-scaling-stroke]"
        data-active={active}
      />
    </svg>
  )
}
