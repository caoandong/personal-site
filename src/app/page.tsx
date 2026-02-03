import Link from 'next/link'
import { ThemeToggle } from '@/components/ThemeToggle'
import { cn } from '@/lib/utils'
import { typography, layout } from '@/lib/typography'

export default function Home() {
  return (
    <div className={layout.container}>
      <header className={cn(layout.headerEnd, layout.headerSpacing)}>
        <ThemeToggle />
      </header>

      <main className={layout.sectionSpacing}>
        <h1 className={typography({ variant: 'h1' })}>Personal Blog</h1>

        <p
          className={cn(
            typography({ variant: 'body', color: 'muted' }),
            'max-w-2xl'
          )}
        >
          A minimalistic blog exploring mathematics, design engineering, and
          interactive visualizations.
        </p>

        <nav className="flex gap-8 pt-4">
          <Link
            href="/blog"
            className={cn(
              typography({ variant: 'nav' }),
              'hover:text-foreground transition-colors duration-200'
            )}
          >
            Read the blog &rarr;
          </Link>
        </nav>
      </main>
    </div>
  )
}
