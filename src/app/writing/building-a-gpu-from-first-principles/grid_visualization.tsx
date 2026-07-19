import type { ReactNode } from 'react'

const threads = [
  { index: 0, a: 2, b: 1, result: 3 },
  { index: 1, a: 4, b: 3, result: 7 },
] as const

export function GridVisualization() {
  return (
    <figure className="not-prose my-12 text-sm">
      <figcaption className="sr-only">
        One CUDA grid containing one block with two threads. Each thread adds
        one pair of input values.
      </figcaption>

      <div className="border-border relative rounded-2xl border p-4 pt-7 md:p-6 md:pt-8">
        <BoundaryLabel>Grid</BoundaryLabel>

        <div className="border-border relative rounded-xl border p-4 pt-7 md:p-5 md:pt-8">
          <BoundaryLabel>Block 0</BoundaryLabel>

          <div className="grid gap-3 md:grid-cols-2">
            {threads.map(({ index, a, b, result }) => (
              <div key={index} className="border-border rounded-lg border p-4">
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
                  <span>&nbsp;&nbsp;&nbsp;&nbsp; = {result}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </figure>
  )
}

function BoundaryLabel({ children }: { children: ReactNode }) {
  return (
    <p className="bg-background text-muted-foreground absolute -top-2.5 left-4 px-2">
      {children}
    </p>
  )
}
