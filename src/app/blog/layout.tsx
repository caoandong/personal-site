import { ThemeToggle } from '@/components/ThemeToggle'
import Link from 'next/link'

export default function BlogLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <div className="mx-auto max-w-3xl px-4 py-8">
      <header className="mb-12 flex items-center justify-between">
        <Link
          href="/"
          className="text-sm font-medium text-muted-foreground hover:text-foreground transition-colors"
        >
          &larr; Home
        </Link>
        <ThemeToggle />
      </header>
      <article className="prose prose-neutral dark:prose-invert max-w-none">
        {children}
      </article>
    </div>
  )
}
