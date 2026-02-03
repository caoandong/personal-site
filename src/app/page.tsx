import Link from 'next/link'
import { ThemeToggle } from '@/components/ThemeToggle'

export default function Home() {
  return (
    <div className="mx-auto max-w-3xl px-4 py-8">
      <header className="flex items-center justify-end mb-24">
        <ThemeToggle />
      </header>

      <main className="space-y-8">
        <h1 className="text-4xl font-bold tracking-tight">
          Personal Blog
        </h1>

        <p className="text-lg text-muted-foreground leading-relaxed max-w-xl">
          A minimalistic blog exploring mathematics, design engineering, and
          interactive visualizations. Built with Next.js, MDX, and full LaTeX support.
        </p>

        <nav className="flex gap-6 pt-4">
          <Link
            href="/blog"
            className="text-lg font-medium underline underline-offset-4 hover:no-underline"
          >
            Read the blog &rarr;
          </Link>
        </nav>
      </main>

      <footer className="mt-24 pt-8 border-t text-sm text-muted-foreground">
        <p>
          Built with{' '}
          <a
            href="https://nextjs.org"
            className="underline underline-offset-4 hover:no-underline"
            target="_blank"
            rel="noopener noreferrer"
          >
            Next.js
          </a>
          ,{' '}
          <a
            href="https://mdxjs.com"
            className="underline underline-offset-4 hover:no-underline"
            target="_blank"
            rel="noopener noreferrer"
          >
            MDX
          </a>
          , and{' '}
          <a
            href="https://katex.org"
            className="underline underline-offset-4 hover:no-underline"
            target="_blank"
            rel="noopener noreferrer"
          >
            KaTeX
          </a>
          .
        </p>
      </footer>
    </div>
  )
}
