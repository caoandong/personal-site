'use client'

import { useState } from 'react'
import { Pause, Play } from 'lucide-react'

import './sm_visualization.css'

const sms = [0, 1] as const
const warps = [0, 1] as const

export function SmVisualization() {
  const [playing, setPlaying] = useState(true)

  return (
    <figure
      className={`sm-viz not-prose relative my-12 text-sm ${playing ? '' : 'sm-viz-paused'}`}
    >
      <figcaption className="sr-only">
        Two blocks are assigned to two streaming multiprocessors. Each SM
        repeatedly selects a warp, passes its instruction through a warp
        scheduler, and issues it to an execution unit.
      </figcaption>

      <div className="grid gap-5 md:grid-cols-2" aria-hidden="true">
        {sms.map((sm) => (
          <div key={sm}>
            <div className="border-border mx-auto w-fit rounded-lg border px-4 py-2">
              Block {sm}
            </div>
            <Connector />

            <div className="border-border rounded-xl border p-4">
              <p className="text-muted-foreground mb-4 text-center">SM {sm}</p>

              <div className="flex items-center justify-center gap-2">
                {warps.map((warp) => (
                  <div
                    key={warp}
                    data-animated
                    className={`sm-viz-node border-border min-w-0 flex-1 rounded-lg border px-2 py-3 text-center sm-viz-warp-${(warp + sm) % 2}`}
                  >
                    Warp {warp}
                  </div>
                ))}
              </div>

              <Connector phase="warp" />
              <div
                data-animated
                className="sm-viz-node sm-viz-scheduler border-border mx-auto max-w-52 rounded-lg border p-3 text-center"
              >
                Warp scheduler
              </div>
              <Connector phase="execution" />
              <div
                data-animated
                className="sm-viz-node sm-viz-execution border-border mx-auto max-w-52 rounded-lg border p-3 text-center"
              >
                Execution unit
              </div>
            </div>
          </div>
        ))}
      </div>

      <button
        type="button"
        onClick={() => setPlaying((value) => !value)}
        onPointerUp={(event) => event.currentTarget.blur()}
        className="sm-viz-control border-border bg-background/80 text-muted-foreground hover:text-foreground absolute top-1 right-1 cursor-pointer rounded-full border p-2 backdrop-blur-sm"
        aria-label={playing ? 'Pause animation' : 'Play animation'}
      >
        {playing ? (
          <Pause className="size-3.5" strokeWidth={1.5} aria-hidden="true" />
        ) : (
          <Play className="size-3.5" strokeWidth={1.5} aria-hidden="true" />
        )}
      </button>
    </figure>
  )
}

function Connector({ phase }: { phase?: 'warp' | 'execution' }) {
  return (
    <div className="bg-border relative mx-auto h-7 w-px" aria-hidden="true">
      <span className="border-border absolute -bottom-0.5 left-1/2 size-2 -translate-x-1/2 rotate-45 border-r border-b" />
      {phase && (
        <span data-animated className={`sm-viz-pulse sm-viz-pulse-${phase}`} />
      )}
    </div>
  )
}
