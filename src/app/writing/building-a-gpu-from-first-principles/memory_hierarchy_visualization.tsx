'use client'

import { useEffect, useRef, useState } from 'react'

import './memory_hierarchy_visualization.css'

type MemoryLayer = 'global' | 'l2' | 'shared' | 'l1' | 'registers'
type Detail = 'die' | 'sm' | 'quadrant'
type Camera = readonly [number, number, number, number]

const layers: { id: MemoryLayer; label: string; description: string }[] = [
  {
    id: 'global',
    label: 'Global memory',
    description: 'HBM sits beside the GPU die. Every SM can access it.',
  },
  {
    id: 'l2',
    label: 'L2 cache',
    description:
      'One shared cache, physically split into slices across the die.',
  },
  {
    id: 'shared',
    label: 'Shared memory',
    description: 'Program-managed SRAM, allocated separately to each block.',
  },
  {
    id: 'l1',
    label: 'L1 cache',
    description: 'Hardware-managed cache local to one SM.',
  },
  {
    id: 'registers',
    label: 'Register file',
    description: 'The fastest storage. Registers belong to individual threads.',
  },
]

const cameras: Record<Detail, Camera> = {
  die: [0, 0, 1200, 720],
  sm: [600, 120, 210, 140],
  quadrant: [655, 150, 80, 54],
}

const gpcs = Array.from({ length: 8 }, (_, index) => index)
const smTiles = Array.from({ length: 12 }, (_, index) => index)

const detailFor = (layer: MemoryLayer): Detail =>
  layer === 'registers'
    ? 'quadrant'
    : layer === 'shared' || layer === 'l1'
      ? 'sm'
      : 'die'

const easeOut = (value: number) => 1 - (1 - value) ** 4

export function MemoryHierarchyVisualization() {
  const [active, setActive] = useState<MemoryLayer>('global')
  const detail = detailFor(active)
  const [camera, setCamera] = useState<Camera>(cameras.die)
  const currentCamera = useRef<Camera>(cameras.die)
  const layer = layers.find(({ id }) => id === active)!

  useEffect(() => {
    const start = currentCamera.current
    const target = cameras[detail]
    const startedAt = performance.now()
    let frame = 0

    const animate = (now: number) => {
      const progress = Math.min((now - startedAt) / 560, 1)
      const eased = easeOut(progress)
      const next: Camera = [
        start[0] + (target[0] - start[0]) * eased,
        start[1] + (target[1] - start[1]) * eased,
        start[2] + (target[2] - start[2]) * eased,
        start[3] + (target[3] - start[3]) * eased,
      ]
      currentCamera.current = next
      setCamera(next)
      if (progress < 1) frame = requestAnimationFrame(animate)
    }

    frame = requestAnimationFrame(animate)
    return () => cancelAnimationFrame(frame)
  }, [detail])

  return (
    <figure
      className="memory-viz not-prose my-12"
      data-active={active}
      data-detail={detail}
    >
      <figcaption className="sr-only">
        Interactive diagram of the H100 memory hierarchy. Select a memory layer
        to zoom to its physical location.
      </figcaption>
      <div className="memory-viz-layout">
        <div className="memory-viz-stage" aria-hidden="true">
          <svg viewBox={camera.join(' ')} role="presentation">
            <Die />
          </svg>
        </div>
        <aside className="memory-viz-sidebar">
          <p className="text-muted-foreground text-xs font-medium tracking-wide uppercase">
            Memory hierarchy
          </p>
          <div className="memory-viz-layers">
            {layers.map((item) => (
              <button
                key={item.id}
                type="button"
                className="memory-viz-layer"
                data-active={item.id === active}
                aria-pressed={item.id === active}
                onClick={() => setActive(item.id)}
              >
                {item.label}
              </button>
            ))}
          </div>
          <div className="memory-viz-description" aria-live="polite">
            <p className="font-medium">{layer.label}</p>
            <p className="text-muted-foreground">{layer.description}</p>
          </div>
        </aside>
      </div>
    </figure>
  )
}

function Die() {
  return (
    <g className="memory-viz-scene">
      <rect
        className="memory-viz-package"
        x="18"
        y="18"
        width="1164"
        height="684"
        rx="16"
      />
      <rect
        className="memory-viz-die"
        x="110"
        y="52"
        width="980"
        height="616"
        rx="10"
      />
      <Hbm x={48} />
      <Hbm x={1118} />
      <g className="memory-viz-gpcs">
        {gpcs.map((index) => (
          <Gpc key={index} index={index} />
        ))}
      </g>
      <g data-layer="l2">
        {Array.from({ length: 8 }, (_, index) => (
          <rect
            key={index}
            className="memory-viz-l2"
            x={185 + index * 105}
            y="321"
            width="96"
            height="78"
            rx="3"
          />
        ))}
        <text
          className="memory-viz-die-label"
          x="600"
          y="366"
          textAnchor="middle"
        >
          L2 cache slices
        </text>
      </g>
    </g>
  )
}

function Hbm({ x }: { x: number }) {
  return (
    <g data-layer="global">
      {Array.from({ length: 4 }, (_, index) => (
        <rect
          key={index}
          className="memory-viz-hbm"
          x={x}
          y={112 + index * 132}
          width="40"
          height="104"
          rx="4"
        />
      ))}
    </g>
  )
}

function Gpc({ index }: { index: number }) {
  const column = index % 4
  const row = Math.floor(index / 4)
  const x = 135 + column * 235
  const y = row === 0 ? 82 : 438
  const height = 200
  const focused = index === 2

  return (
    <g className="memory-viz-gpc" transform={`translate(${x} ${y})`}>
      <rect width="214" height={height} rx="5" />
      {smTiles.map((tile) => {
        const tileX = 18 + (tile % 4) * 47
        const tileY = 21 + Math.floor(tile / 4) * 57
        return focused && tile === 5 ? (
          <FocusedSm key={tile} x={tileX} y={tileY} />
        ) : (
          <rect
            key={tile}
            className="memory-viz-sm"
            x={tileX}
            y={tileY}
            width="37"
            height="45"
            rx="2"
          />
        )
      })}
    </g>
  )
}

function FocusedSm({ x, y }: { x: number; y: number }) {
  return (
    <g className="memory-viz-sm-focus" transform={`translate(${x} ${y})`}>
      <rect className="memory-viz-sm" width="37" height="45" rx="2" />
      <rect
        data-layer="l1"
        className="memory-viz-l1"
        x="2"
        y="2"
        width="16"
        height="12"
        rx="1"
      />
      <rect
        data-layer="shared"
        className="memory-viz-shared"
        x="19"
        y="2"
        width="16"
        height="12"
        rx="1"
      />
      <text
        className="memory-viz-sm-label memory-viz-l1-label"
        x="10"
        y="8"
        textAnchor="middle"
        dominantBaseline="middle"
      >
        L1
      </text>
      <text
        className="memory-viz-sm-label memory-viz-shared-label"
        x="27"
        y="8"
        textAnchor="middle"
        dominantBaseline="middle"
      >
        Shared
      </text>
      {[0, 1, 2, 3].map((index) => {
        const quadrantX = 2 + (index % 2) * 17
        const quadrantY = 16 + Math.floor(index / 2) * 14
        return (
          <g key={index} transform={`translate(${quadrantX} ${quadrantY})`}>
            <rect
              className="memory-viz-quadrant"
              width="15"
              height="12"
              rx="1"
            />
            <rect
              data-layer="registers"
              className="memory-viz-registers"
              x="1.5"
              y="1.5"
              width="12"
              height="4"
              rx="0.5"
            />
            <text
              className="memory-viz-register-label"
              x="7.5"
              y="3.5"
              textAnchor="middle"
              dominantBaseline="middle"
            >
              Registers
            </text>
          </g>
        )
      })}
    </g>
  )
}
