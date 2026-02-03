import { ThemeToggle } from '@/components/ThemeToggle'
import Link from 'next/link'
import { cn } from '@/lib/utils'
import { typography, layout, prose, proseCustom } from '@/lib/typography'

export default function BlogLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <div className={layout.container}>
      <header className={cn(layout.header, layout.headerSpacing)}>
        <Link
          href="/"
          className={cn(typography({ variant: 'nav', color: 'muted' }), 'hover:text-foreground transition-colors duration-200')}
        >
          &larr; Home
        </Link>
        <ThemeToggle />
      </header>
      <article className={cn(prose(), proseCustom)}>
        {children}
      </article>
    </div>
  )
}
