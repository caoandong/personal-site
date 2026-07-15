import type { Metadata } from 'next'
import Link from 'next/link'
import { ThemeToggle } from '@/components/ThemeToggle'

export const metadata: Metadata = {
  title: 'Nano GPU — Antonio Cao',
  description:
    'A browser laboratory for building a GPU, compiler, and transformer from first principles.',
}

const stages = [
  {
    name: 'Hardware',
    description: 'Describe a tiny GPU with a learner-friendly HDL.',
  },
  {
    name: 'Compiler',
    description: 'Lower kernels through PTX-like instructions into machine code.',
  },
  {
    name: 'Simulator',
    description: 'Execute every lane, register, memory access, and clock cycle.',
  },
  {
    name: 'Visualizer',
    description: 'Pause, step, inspect, and rewind the complete machine state.',
  },
]

export default function NanoGpuPage() {
  return (
    <div className="mx-auto min-h-screen w-full max-w-7xl px-6 py-6 md:px-10">
      <header className="flex items-center justify-between">
        <Link
          href="/"
          className="text-nav text-muted-foreground hover:text-heading"
        >
          ← Antonio Cao
        </Link>
        <ThemeToggle />
      </header>

      <main className="py-20 md:py-28">
        <p className="font-mono text-sm tracking-widest text-muted-foreground uppercase">
          Under construction
        </p>
        <h1 className="mt-4 max-w-4xl text-5xl font-light tracking-tight text-heading md:text-7xl">
          Nano GPU
        </h1>
        <p className="mt-8 max-w-3xl text-xl leading-relaxed text-body md:text-2xl">
          A fast, visual browser laboratory for building a GPU and its compiler
          from first principles—then running a tiny transformer on the machine
          you created.
        </p>

        <section className="mt-16 grid gap-px overflow-hidden rounded-xl border border-border bg-border md:grid-cols-2 lg:grid-cols-4">
          {stages.map((stage, index) => (
            <article key={stage.name} className="bg-background p-6">
              <span className="font-mono text-sm text-muted-foreground">
                {String(index + 1).padStart(2, '0')}
              </span>
              <h2 className="mt-8 text-xl font-normal text-heading">
                {stage.name}
              </h2>
              <p className="mt-3 leading-relaxed text-muted-foreground">
                {stage.description}
              </p>
            </article>
          ))}
        </section>
      </main>
    </div>
  )
}
